#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaximumBlocks = 65535;

template <typename T, int Width>
struct alignas(sizeof(T) * Width) PackedValue {
    T values[Width];
};

template <typename T, typename Index, int Width>
__global__ __launch_bounds__(kThreads) void append_kernel(
    const T* key,
    const T* value,
    T* key_cache,
    T* value_cache,
    const Index* slot_mapping,
    int64_t tokens,
    int64_t elements_per_token,
    int64_t capacity) {
    using Pack = PackedValue<T, Width>;
    const int64_t packs_per_token = elements_per_token / Width;
    const int64_t total = tokens * packs_per_token;
    const auto* key_packed = reinterpret_cast<const Pack*>(key);
    const auto* value_packed = reinterpret_cast<const Pack*>(value);
    auto* key_cache_packed = reinterpret_cast<Pack*>(key_cache);
    auto* value_cache_packed = reinterpret_cast<Pack*>(value_cache);
    for(int64_t logical = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        logical < total;
        logical += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t token = logical / packs_per_token;
        const int64_t pack_index = logical - token * packs_per_token;
        const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
        if(slot < 0 || slot >= capacity) {
            continue;
        }
        const int64_t destination = slot * packs_per_token + pack_index;
        key_cache_packed[destination] = key_packed[logical];
        value_cache_packed[destination] = value_packed[logical];
    }
}

template <typename T, typename Index, int Width>
__global__ __launch_bounds__(kThreads) void copy_kernel(
    T* key_cache,
    T* value_cache,
    const Index* block_mapping,
    int64_t pairs,
    int64_t packs_per_block,
    int64_t num_blocks) {
    using Pack = PackedValue<T, Width>;
    auto* key_packed = reinterpret_cast<Pack*>(key_cache);
    auto* value_packed = reinterpret_cast<Pack*>(value_cache);
    const int64_t total = pairs * packs_per_block;
    for(int64_t logical = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        logical < total;
        logical += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t pair = logical / packs_per_block;
        const int64_t pack_index = logical - pair * packs_per_block;
        const int64_t source = static_cast<int64_t>(block_mapping[pair * 2]);
        const int64_t destination = static_cast<int64_t>(block_mapping[pair * 2 + 1]);
        if(source < 0 || destination < 0 || source >= num_blocks || destination >= num_blocks) {
            continue;
        }
        const int64_t source_index = source * packs_per_block + pack_index;
        const int64_t destination_index = destination * packs_per_block + pack_index;
        key_packed[destination_index] = key_packed[source_index];
        value_packed[destination_index] = value_packed[source_index];
    }
}

void check_data_tensor(const at::Tensor& tensor, const char* name, int64_t rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a ROCm tensor");
    TORCH_CHECK(tensor.dim() == rank, name, " must be rank ", rank);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16,
        name,
        " must use FP16 or BF16");
}

void check_index_tensor(const at::Tensor& tensor, const char* name, int64_t rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a ROCm tensor");
    TORCH_CHECK(tensor.dim() == rank, name, " must be rank ", rank);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
        name,
        " must use INT32 or INT64");
}

void check_same(const at::Tensor& first, const at::Tensor& second, const char* name) {
    TORCH_CHECK(first.device() == second.device(), name, " must use the same device");
    TORCH_CHECK(first.scalar_type() == second.scalar_type(), name, " must use the same dtype");
}

template <typename T, int Width>
void launch_append_index(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    hipStream_t stream,
    int64_t elements_per_token,
    int64_t capacity) {
    const int64_t total = key.size(0) * (elements_per_token / Width);
    dim3 block(kThreads);
    dim3 grid(std::min<int64_t>((total + kThreads - 1) / kThreads, kMaximumBlocks));
    if(slot_mapping.scalar_type() == at::kInt) {
        hipLaunchKernelGGL(
            (append_kernel<T, int32_t, Width>),
            grid,
            block,
            0,
            stream,
            key.data_ptr<T>(),
            value.data_ptr<T>(),
            key_cache.data_ptr<T>(),
            value_cache.data_ptr<T>(),
            slot_mapping.data_ptr<int32_t>(),
            key.size(0),
            elements_per_token,
            capacity);
    } else {
        hipLaunchKernelGGL(
            (append_kernel<T, int64_t, Width>),
            grid,
            block,
            0,
            stream,
            key.data_ptr<T>(),
            value.data_ptr<T>(),
            key_cache.data_ptr<T>(),
            value_cache.data_ptr<T>(),
            slot_mapping.data_ptr<int64_t>(),
            key.size(0),
            elements_per_token,
            capacity);
    }
}

template <int Width>
std::vector<at::Tensor> append_impl(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor key_cache,
    at::Tensor value_cache,
    const at::Tensor& slot_mapping) {
    check_data_tensor(key, "key", 3);
    check_data_tensor(value, "value", 3);
    check_data_tensor(key_cache, "key_cache", 4);
    check_data_tensor(value_cache, "value_cache", 4);
    check_index_tensor(slot_mapping, "slot_mapping", 1);
    check_same(key, value, "value");
    check_same(key, key_cache, "key_cache");
    check_same(key, value_cache, "value_cache");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "cache shapes must match");
    TORCH_CHECK(key.size(0) == slot_mapping.size(0), "slot_mapping must match tokens");
    TORCH_CHECK(
        key.size(1) == key_cache.size(2) && key.size(2) == key_cache.size(3),
        "key/value shape must match cache heads and head dimension");
    const int64_t elements_per_token = key.size(1) * key.size(2);
    TORCH_CHECK(elements_per_token % Width == 0, "token width must support packed copy");
    at::DeviceGuard guard(key.device());
    auto stream = c10::hip::getCurrentHIPStream(key.get_device()).stream();
    const int64_t capacity = key_cache.size(0) * key_cache.size(1);
    if(key.scalar_type() == at::kHalf) {
        launch_append_index<at::Half, Width>(
            key, value, key_cache, value_cache, slot_mapping, stream, elements_per_token, capacity);
    } else {
        launch_append_index<at::BFloat16, Width>(
            key, value, key_cache, value_cache, slot_mapping, stream, elements_per_token, capacity);
    }
    const auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "KV append launch failed: ", hipGetErrorString(error));
    return {key_cache, value_cache};
}

template <typename T, int Width>
void launch_copy_index(
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& block_mapping,
    hipStream_t stream,
    int64_t packs_per_block) {
    const int64_t total = block_mapping.size(0) * packs_per_block;
    dim3 block(kThreads);
    dim3 grid(std::min<int64_t>((total + kThreads - 1) / kThreads, kMaximumBlocks));
    if(block_mapping.scalar_type() == at::kInt) {
        hipLaunchKernelGGL(
            (copy_kernel<T, int32_t, Width>),
            grid,
            block,
            0,
            stream,
            key_cache.data_ptr<T>(),
            value_cache.data_ptr<T>(),
            block_mapping.data_ptr<int32_t>(),
            block_mapping.size(0),
            packs_per_block,
            key_cache.size(0));
    } else {
        hipLaunchKernelGGL(
            (copy_kernel<T, int64_t, Width>),
            grid,
            block,
            0,
            stream,
            key_cache.data_ptr<T>(),
            value_cache.data_ptr<T>(),
            block_mapping.data_ptr<int64_t>(),
            block_mapping.size(0),
            packs_per_block,
            key_cache.size(0));
    }
}

template <int Width>
std::vector<at::Tensor> copy_impl(
    at::Tensor key_cache,
    at::Tensor value_cache,
    const at::Tensor& block_mapping) {
    check_data_tensor(key_cache, "key_cache", 4);
    check_data_tensor(value_cache, "value_cache", 4);
    check_index_tensor(block_mapping, "block_mapping", 2);
    check_same(key_cache, value_cache, "value_cache");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "cache shapes must match");
    TORCH_CHECK(block_mapping.size(1) == 2, "block_mapping must have shape [pairs, 2]");
    const int64_t elements_per_block =
        key_cache.size(1) * key_cache.size(2) * key_cache.size(3);
    TORCH_CHECK(elements_per_block % Width == 0, "block width must support packed copy");
    at::DeviceGuard guard(key_cache.device());
    auto stream = c10::hip::getCurrentHIPStream(key_cache.get_device()).stream();
    const int64_t packs_per_block = elements_per_block / Width;
    if(key_cache.scalar_type() == at::kHalf) {
        launch_copy_index<at::Half, Width>(
            key_cache, value_cache, block_mapping, stream, packs_per_block);
    } else {
        launch_copy_index<at::BFloat16, Width>(
            key_cache, value_cache, block_mapping, stream, packs_per_block);
    }
    const auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "KV copy launch failed: ", hipGetErrorString(error));
    return {key_cache, value_cache};
}

std::vector<at::Tensor> append_scalar(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor key_cache,
    at::Tensor value_cache,
    const at::Tensor& slot_mapping) {
    return append_impl<1>(key, value, key_cache, value_cache, slot_mapping);
}

std::vector<at::Tensor> append_vec4(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor key_cache,
    at::Tensor value_cache,
    const at::Tensor& slot_mapping) {
    return append_impl<4>(key, value, key_cache, value_cache, slot_mapping);
}

std::vector<at::Tensor> append_vec8(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor key_cache,
    at::Tensor value_cache,
    const at::Tensor& slot_mapping) {
    return append_impl<8>(key, value, key_cache, value_cache, slot_mapping);
}

std::vector<at::Tensor> copy_scalar(
    at::Tensor key_cache, at::Tensor value_cache, const at::Tensor& block_mapping) {
    return copy_impl<1>(key_cache, value_cache, block_mapping);
}

std::vector<at::Tensor> copy_vec4(
    at::Tensor key_cache, at::Tensor value_cache, const at::Tensor& block_mapping) {
    return copy_impl<4>(key_cache, value_cache, block_mapping);
}

std::vector<at::Tensor> copy_vec8(
    at::Tensor key_cache, at::Tensor value_cache, const at::Tensor& block_mapping) {
    return copy_impl<8>(key_cache, value_cache, block_mapping);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("append_scalar", &append_scalar, "paged KV append with scalar copies");
    module.def("append_vec4", &append_vec4, "paged KV append with 64-bit copies");
    module.def("append_vec8", &append_vec8, "paged KV append with 128-bit copies");
    module.def("copy_scalar", &copy_scalar, "paged KV block copy with scalar copies");
    module.def("copy_vec4", &copy_vec4, "paged KV block copy with 64-bit copies");
    module.def("copy_vec8", &copy_vec8, "paged KV block copy with 128-bit copies");
}
