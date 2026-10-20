// Standalone HIP probe: CPU -> GPU -> CPU handoffs on the same allocation.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1151 zero_copy.hip -o zero_copy
// Run:   ./zero_copy host_noncoherent 1024 15
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static void check(hipError_t status, const char* expression) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(expression) + ": " + hipGetErrorString(status));
}
#define HIP_CHECK(expr) check((expr), #expr)

static std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    return out.str() + '"';
}

static int64_t read_number(const std::string& path) {
    std::ifstream in(path);
    int64_t value = -1;
    in >> value;
    return value;
}

static std::string gpu_path() {
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm")) {
        const auto path = entry.path() / "device";
        if (std::filesystem::exists(path / "mem_info_gtt_total")) return path.string();
    }
    return "";
}

static std::string snapshot(const std::string& path) {
    std::ostringstream out;
    out << "{\"gtt_used_bytes\":" << read_number(path + "/mem_info_gtt_used")
        << ",\"vram_used_bytes\":" << read_number(path + "/mem_info_vram_used");
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmLck:", 0) == 0) {
            std::istringstream fields(line);
            std::string name;
            int64_t kib;
            fields >> name >> kib;
            out << ",\"" << (name == "VmRSS:" ? "rss_bytes" : "mlock_bytes")
                << "\":" << kib * 1024;
        }
    }
    return out.str() + "}";
}

__global__ void affine_kernel(uint4* data, size_t vectors) {
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors; i += stride) {
        uint4 x = data[i];
        x.x = x.x * 3u + 7u;
        x.y = x.y * 3u + 7u;
        x.z = x.z * 3u + 7u;
        x.w = x.w * 3u + 7u;
        data[i] = x;
    }
}

static uint32_t pattern(size_t i, uint32_t seed) {
    return uint32_t(i) * 2654435761u + seed;
}

static void initialize(uint32_t* data, size_t count, uint32_t seed) {
    for (size_t i = 0; i < count; ++i) data[i] = pattern(i, seed);
}

static size_t verify(const uint32_t* data, size_t count, uint32_t seed, int passes) {
    // Fold repeated affine transforms modulo 2^32, independently on the CPU.
    uint32_t a = 1, b = 0;
    for (int i = 0; i < passes; ++i) { a *= 3u; b = b * 3u + 7u; }
    size_t errors = 0;
    for (size_t i = 0; i < count; ++i)
        errors += data[i] != a * pattern(i, seed) + b;
    return errors;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: zero_copy MODE MiB SAMPLES");
        const std::string mode = argv[1];
        const size_t mib = std::stoull(argv[2]);
        const int samples = std::stoi(argv[3]);
        // Keep this diagnostic comfortably below system memory capacity.
        if (!mib || mib > 8192 || samples < 1 || samples > 1000)
            throw std::runtime_error("require 1..8192 MiB and 1..1000 samples");
        const bool device_mode = mode == "device";
        const bool registered = mode == "registered";
        const bool managed = mode == "managed";
        unsigned flags = hipHostMallocMapped;
        if (mode == "mapped_coherent") flags |= hipHostMallocCoherent;
        // Deliberately avoid combining Mapped and NonCoherent: the documented
        // precedence differs across HIP versions. HIP still exposes a GPU alias.
        else if (mode == "host_noncoherent") flags = hipHostMallocNonCoherent;
        else if (mode != "mapped" && !device_mode && !registered && !managed)
            throw std::runtime_error("unknown mode");

        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        int pageable = 0, concurrent = 0, managed_support = 0;
        HIP_CHECK(hipDeviceGetAttribute(&pageable, hipDeviceAttributePageableMemoryAccess, 0));
        HIP_CHECK(hipDeviceGetAttribute(&concurrent, hipDeviceAttributeConcurrentManagedAccess, 0));
        HIP_CHECK(hipDeviceGetAttribute(&managed_support, hipDeviceAttributeManagedMemory, 0));
        const std::string sysfs = gpu_path();
        const std::string before = snapshot(sysfs);
        size_t hip_free = 0, hip_total = 0;
        HIP_CHECK(hipMemGetInfo(&hip_free, &hip_total));
        const size_t bytes = mib * 1024 * 1024, count = bytes / sizeof(uint32_t);
        uint32_t *host = nullptr, *gpu = nullptr;
        const auto allocation_start = Clock::now();
        if (device_mode || registered) {
            void* allocation = nullptr;
            if (posix_memalign(&allocation, 4096, bytes)) throw std::runtime_error("posix_memalign failed");
            host = static_cast<uint32_t*>(allocation);
            // Fault in original CPU pages before registering them with HIP.
            initialize(host, count, 11u);
            if (device_mode) HIP_CHECK(hipMalloc(&gpu, bytes));
            else {
                HIP_CHECK(hipHostRegister(host, bytes, hipHostRegisterMapped));
                HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&gpu), host, 0));
            }
        } else if (managed) {
            HIP_CHECK(hipMallocManaged(&host, bytes, hipMemAttachGlobal));
            gpu = host;
        } else {
            HIP_CHECK(hipHostMalloc(&host, bytes, flags));
            HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&gpu), host, 0));
        }
        const double allocation_ms = std::chrono::duration<double, std::milli>(Clock::now() - allocation_start).count();
        hipPointerAttribute_t attr{};
        HIP_CHECK(hipPointerGetAttributes(&attr, gpu));
        initialize(host, count, 11u);
        uint64_t explicit_copy_bytes = 0;
        if (device_mode) {
            HIP_CHECK(hipMemcpy(gpu, host, bytes, hipMemcpyHostToDevice));
            explicit_copy_bytes += bytes;
        }
        const std::string allocated = snapshot(sysfs);
        const int blocks = int(std::min<size_t>((bytes / sizeof(uint4) + 255) / 256,
                                                 prop.multiProcessorCount * 32));
        auto launch = [&] {
            hipLaunchKernelGGL(affine_kernel, dim3(blocks), dim3(256), 0, 0,
                               reinterpret_cast<uint4*>(gpu), bytes / sizeof(uint4));
            HIP_CHECK(hipGetLastError());
        };
        constexpr int warmups = 3;
        for (int i = 0; i < warmups; ++i) launch();
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreateWithFlags(&start, hipEventReleaseToSystem));
        HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventReleaseToSystem));
        std::vector<float> times;
        for (int i = 0; i < samples; ++i) {
            HIP_CHECK(hipEventRecord(start, nullptr));
            launch();
            HIP_CHECK(hipEventRecord(stop, nullptr));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            times.push_back(ms);
        }
        // Device synchronization includes a system-scope release for CPU access
        // to coarse-grained / non-coherent host allocations.
        HIP_CHECK(hipDeviceSynchronize());
        const std::string after_gpu = snapshot(sysfs);
        if (device_mode) {
            HIP_CHECK(hipMemcpy(host, gpu, bytes, hipMemcpyDeviceToHost));
            explicit_copy_bytes += bytes;
        }
        const size_t errors_first = verify(host, count, 11u, warmups + samples);
        // Modify the same CPU pages again after GPU completion. The next GPU
        // launch must observe CPU changes, and CPU must observe GPU writes.
        initialize(host, count, 29u);
        if (device_mode) {
            HIP_CHECK(hipMemcpy(gpu, host, bytes, hipMemcpyHostToDevice));
            explicit_copy_bytes += bytes;
        }
        launch();
        HIP_CHECK(hipDeviceSynchronize());
        if (device_mode) {
            HIP_CHECK(hipMemcpy(host, gpu, bytes, hipMemcpyDeviceToHost));
            explicit_copy_bytes += bytes;
        }
        const size_t errors_second = verify(host, count, 29u, 1);
        const std::string host_address = std::to_string(uintptr_t(host));
        const std::string gpu_address = std::to_string(uintptr_t(gpu));
        const bool same_address = host == gpu;
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        if (device_mode) { HIP_CHECK(hipFree(gpu)); std::free(host); }
        else if (registered) { HIP_CHECK(hipHostUnregister(host)); std::free(host); }
        else if (managed) HIP_CHECK(hipFree(host));
        else HIP_CHECK(hipHostFree(host));
        const std::string freed = snapshot(sysfs);
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const float median = sorted.size() % 2 ? sorted[sorted.size() / 2]
            : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2;
        std::cout << std::setprecision(9)
            << "{\"mode\":" << json_string(mode)
            << ",\"bytes\":" << bytes << ",\"samples\":" << samples
            << ",\"warmups\":" << warmups << ",\"allocation_and_registration_ms\":" << allocation_ms
            << ",\"device\":{\"name\":" << json_string(prop.name)
            << ",\"arch\":" << json_string(prop.gcnArchName)
            // HIP reports WGPs on this RDNA GPU (two CUs per WGP).
            << ",\"hip_multiprocessor_count\":" << prop.multiProcessorCount
            << ",\"integrated\":" << prop.integrated
            << ",\"can_map_host_memory\":" << prop.canMapHostMemory
            << ",\"pageable_memory_access\":" << pageable
            << ",\"managed_memory\":" << managed_support
            << ",\"concurrent_managed_access\":" << concurrent
            << ",\"hip_total_bytes\":" << hip_total << ",\"hip_free_bytes\":" << hip_free
            << ",\"vram_total_bytes\":" << read_number(sysfs + "/mem_info_vram_total")
            << ",\"gtt_total_bytes\":" << read_number(sysfs + "/mem_info_gtt_total") << "}"
            << ",\"pointer\":{\"host\":" << json_string(host_address)
            << ",\"gpu\":" << json_string(gpu_address)
            << ",\"same_virtual_address\":" << (same_address ? "true" : "false")
            << ",\"hip_memory_type\":" << int(attr.type)
            << ",\"is_managed\":" << attr.isManaged
            << ",\"allocation_flags\":" << attr.allocationFlags << "}"
            << ",\"snapshots\":{\"before\":" << before << ",\"allocated\":" << allocated
            << ",\"after_gpu\":" << after_gpu << ",\"freed\":" << freed << "}"
            << ",\"explicit_copy_bytes\":" << explicit_copy_bytes
            << ",\"errors_first_handoff\":" << errors_first
            << ",\"errors_second_handoff\":" << errors_second
            << ",\"elements_checked_per_handoff\":" << count
            << ",\"median_kernel_ms\":" << median
            // Logical traffic: one read + one write per element, not a DRAM counter.
            << ",\"effective_read_write_GB_s\":" << 2.0 * bytes / (median * 1e6)
            << ",\"kernel_ms\":[";
        for (size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
        std::cout << "],\"passed\":" << (!errors_first && !errors_second ? "true" : "false") << "}\n";
        return errors_first || errors_second ? 2 : 0;
    } catch (const std::exception& error) {
        std::cout << "{\"passed\":false,\"error\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
