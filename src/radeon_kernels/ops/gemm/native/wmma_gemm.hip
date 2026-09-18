#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr uint32_t kWaveSize = 32;
constexpr uint32_t kWmmaM = 16;
constexpr uint32_t kWmmaN = 16;
constexpr uint32_t kWmmaK = 16;

template <typename InputT,
          uint32_t BlocksM,
          uint32_t BlocksN,
          uint32_t ThreadsX,
          uint32_t ThreadsY,
          uint32_t PanelK>
__global__ void wmma_gemm_kernel(uint32_t m,
                                 uint32_t n,
                                 uint32_t k,
                                 const InputT* a,
                                 const InputT* b,
                                 InputT* output) {
    using namespace rocwmma;

    constexpr uint32_t warp_tile_m = BlocksM * kWmmaM;
    constexpr uint32_t warp_tile_n = BlocksN * kWmmaN;
    constexpr uint32_t warps_m = ThreadsX / kWaveSize;
    constexpr uint32_t warps_n = ThreadsY;
    constexpr uint32_t macro_tile_m = warps_m * warp_tile_m;
    constexpr uint32_t macro_tile_n = warps_n * warp_tile_n;

    using InputLayoutA = row_major;
    using InputLayoutB = row_major;
    using OutputLayout = row_major;
    using LdsLayout = col_major;
    using MmaFragA = fragment<matrix_a,
                              warp_tile_m,
                              warp_tile_n,
                              kWmmaK,
                              InputT,
                              InputLayoutA>;
    using MmaFragB = fragment<matrix_b,
                              warp_tile_m,
                              warp_tile_n,
                              kWmmaK,
                              InputT,
                              InputLayoutB>;
    using MmaFragOutput = fragment<accumulator,
                                   warp_tile_m,
                                   warp_tile_n,
                                   kWmmaK,
                                   InputT,
                                   OutputLayout>;
    using MmaFragAcc = fragment<accumulator,
                                warp_tile_m,
                                warp_tile_n,
                                kWmmaK,
                                float32_t>;
    using CoopScheduler = fragment_scheduler::coop_row_major_2d<ThreadsX, ThreadsY>;
    using GlobalReadFragA = fragment<matrix_a,
                                     macro_tile_m,
                                     macro_tile_n,
                                     PanelK,
                                     InputT,
                                     InputLayoutA,
                                     CoopScheduler>;
    using GlobalReadFragB = fragment<matrix_b,
                                     macro_tile_m,
                                     macro_tile_n,
                                     PanelK,
                                     InputT,
                                     InputLayoutB,
                                     CoopScheduler>;
    using LocalWriteFragA = apply_data_layout_t<GlobalReadFragA, LdsLayout>;
    using LocalWriteFragB = apply_data_layout_t<apply_transpose_t<GlobalReadFragB>, LdsLayout>;
    using LocalReadFragA = apply_data_layout_t<MmaFragA, LdsLayout>;
    using LocalReadFragB = apply_data_layout_t<apply_transpose_t<MmaFragB>, LdsLayout>;

    constexpr auto warp_tile_size = make_coord2d(warp_tile_m, warp_tile_n);
    constexpr auto macro_tile_size = make_coord2d(macro_tile_m, macro_tile_n);
    auto local_warp_coord = make_coord2d(threadIdx.x / kWaveSize, threadIdx.y);
    auto local_warp_offset = local_warp_coord * warp_tile_size;
    auto macro_tile_coord = make_coord2d(blockIdx.x, blockIdx.y) * macro_tile_size;
    auto warp_tile_coord = macro_tile_coord + local_warp_offset;

    using GlobalReadMapA = GetDataLayout_t<GlobalReadFragA>;
    using GlobalReadMapB = GetDataLayout_t<GlobalReadFragB>;
    auto global_offset_a = GlobalReadMapA::fromMatrixCoord(
        make_coord2d(rocwmma::get<0>(macro_tile_coord), 0u), k);
    auto global_offset_b = GlobalReadMapB::fromMatrixCoord(
        make_coord2d(0u, rocwmma::get<1>(macro_tile_coord)), n);
    auto k_step_a = GlobalReadMapA::fromMatrixCoord(make_coord2d(0u, PanelK), k);
    auto k_step_b = GlobalReadMapB::fromMatrixCoord(make_coord2d(PanelK, 0u), n);

    GlobalReadFragA global_a;
    GlobalReadFragB global_b;
    load_matrix_sync(global_a, a + global_offset_a, k);
    load_matrix_sync(global_b, b + global_offset_b, n);
    global_offset_a += k_step_a;
    global_offset_b += k_step_b;

    HIP_DYNAMIC_SHARED(void*, shared_memory);
    using LocalWriteShapeA = GetIOShape_t<LocalWriteFragA>;
    using LocalWriteShapeB = GetIOShape_t<LocalWriteFragB>;
    using LocalWriteMapA = GetDataLayout_t<LocalWriteFragA>;
    using LocalWriteMapB = GetDataLayout_t<LocalWriteFragB>;
    constexpr uint32_t lds_height = LocalWriteShapeA::BlockHeight + LocalWriteShapeB::BlockHeight;
    constexpr uint32_t lds_leading_dimension = lds_height;
    constexpr uint32_t lds_elements = lds_height * PanelK;
    auto* lds_lo = reinterpret_cast<InputT*>(shared_memory);
    auto* lds_hi = lds_lo + lds_elements;
    constexpr uint32_t lds_write_a = 0;
    auto lds_write_b = LocalWriteMapA::fromMatrixCoord(
        make_coord2d(LocalWriteShapeA::BlockHeight, 0u), lds_leading_dimension);
    store_matrix_sync(
        lds_lo + lds_write_a, apply_data_layout<LdsLayout>(global_a), lds_leading_dimension);
    store_matrix_sync(lds_lo + lds_write_b,
                      apply_data_layout<LdsLayout>(apply_transpose(global_b)),
                      lds_leading_dimension);

    MmaFragAcc accumulator;
    fill_fragment(accumulator, 0.0f);
    synchronize_workgroup();

    for(uint32_t current_k = PanelK; current_k < k; current_k += PanelK) {
        load_matrix_sync(global_a, a + global_offset_a, k);
        load_matrix_sync(global_b, b + global_offset_b, n);
        global_offset_a += k_step_a;
        global_offset_b += k_step_b;

#pragma unroll
        for(uint32_t panel_k = 0; panel_k < PanelK; panel_k += kWmmaK) {
            auto lds_read_a = lds_write_a + LocalWriteMapA::fromMatrixCoord(
                make_coord2d(rocwmma::get<0>(local_warp_offset), panel_k),
                lds_leading_dimension);
            auto lds_read_b = lds_write_b + LocalWriteMapB::fromMatrixCoord(
                make_coord2d(rocwmma::get<1>(local_warp_offset), panel_k),
                lds_leading_dimension);
            LocalReadFragA local_a;
            LocalReadFragB local_b;
            load_matrix_sync(local_a, lds_lo + lds_read_a, lds_leading_dimension);
            load_matrix_sync(local_b, lds_lo + lds_read_b, lds_leading_dimension);
            mma_sync(accumulator,
                     apply_data_layout<InputLayoutA>(local_a),
                     apply_data_layout<InputLayoutB>(apply_transpose(local_b)),
                     accumulator);
        }
        store_matrix_sync(lds_hi + lds_write_a,
                          apply_data_layout<LdsLayout>(global_a),
                          lds_leading_dimension);
        store_matrix_sync(lds_hi + lds_write_b,
                          apply_data_layout<LdsLayout>(apply_transpose(global_b)),
                          lds_leading_dimension);
        synchronize_workgroup();
        auto* temporary = lds_lo;
        lds_lo = lds_hi;
        lds_hi = temporary;
    }

#pragma unroll
    for(uint32_t panel_k = 0; panel_k < PanelK; panel_k += kWmmaK) {
        auto lds_read_a = lds_write_a + LocalWriteMapA::fromMatrixCoord(
            make_coord2d(rocwmma::get<0>(local_warp_offset), panel_k), lds_leading_dimension);
        auto lds_read_b = lds_write_b + LocalWriteMapB::fromMatrixCoord(
            make_coord2d(rocwmma::get<1>(local_warp_offset), panel_k), lds_leading_dimension);
        LocalReadFragA local_a;
        LocalReadFragB local_b;
        load_matrix_sync(local_a, lds_lo + lds_read_a, lds_leading_dimension);
        load_matrix_sync(local_b, lds_lo + lds_read_b, lds_leading_dimension);
        mma_sync(accumulator,
                 apply_data_layout<InputLayoutA>(local_a),
                 apply_data_layout<InputLayoutB>(apply_transpose(local_b)),
                 accumulator);
    }

    MmaFragOutput result;
#pragma unroll
    for(uint32_t index = 0; index < result.num_elements; ++index) {
        result.x[index] = static_cast<InputT>(accumulator.x[index]);
    }
    using OutputMap = GetDataLayout_t<MmaFragOutput>;
    auto output_offset = OutputMap::fromMatrixCoord(warp_tile_coord, n);
    store_matrix_sync(output + output_offset, result, n);
}

template <typename InputT,
          uint32_t BlocksM,
          uint32_t BlocksN,
          uint32_t ThreadsX,
          uint32_t ThreadsY,
          uint32_t PanelK>
void launch_variant(const at::Tensor& a, const at::Tensor& b, at::Tensor& output, hipStream_t stream) {
    constexpr uint32_t macro_tile_m = (ThreadsX / kWaveSize) * BlocksM * kWmmaM;
    constexpr uint32_t macro_tile_n = ThreadsY * BlocksN * kWmmaN;
    auto m = static_cast<uint32_t>(a.size(0));
    auto n = static_cast<uint32_t>(b.size(1));
    auto k = static_cast<uint32_t>(a.size(1));
    TORCH_CHECK(m % macro_tile_m == 0,
                "M must be divisible by the selected macro tile: ",
                macro_tile_m);
    TORCH_CHECK(n % macro_tile_n == 0,
                "N must be divisible by the selected macro tile: ",
                macro_tile_n);
    TORCH_CHECK(k % PanelK == 0,
                "K must be divisible by the selected panel size: ",
                PanelK);
    dim3 grid(m / macro_tile_m, n / macro_tile_n);
    dim3 block(ThreadsX, ThreadsY);
    constexpr size_t shared_bytes =
        2 * (macro_tile_m + macro_tile_n) * PanelK * sizeof(InputT);
    hipLaunchKernelGGL(
        (wmma_gemm_kernel<InputT, BlocksM, BlocksN, ThreadsX, ThreadsY, PanelK>),
        grid,
        block,
        shared_bytes,
        stream,
        m,
        n,
        k,
        reinterpret_cast<const InputT*>(a.data_ptr()),
        reinterpret_cast<const InputT*>(b.data_ptr()),
        reinterpret_cast<InputT*>(output.data_ptr()));
}

template <typename InputT>
void launch(const at::Tensor& a,
            const at::Tensor& b,
            at::Tensor& output,
            int64_t variant,
            hipStream_t stream) {
    switch(variant) {
    case 0:
        launch_variant<InputT, 4, 2, 64, 2, 16>(a, b, output, stream);
        break;
    case 1:
        launch_variant<InputT, 2, 4, 64, 2, 16>(a, b, output, stream);
        break;
    case 2:
        launch_variant<InputT, 2, 2, 128, 2, 16>(a, b, output, stream);
        break;
    case 3:
        launch_variant<InputT, 4, 2, 128, 1, 16>(a, b, output, stream);
        break;
    case 4:
        launch_variant<InputT, 2, 4, 128, 1, 16>(a, b, output, stream);
        break;
    case 5:
        launch_variant<InputT, 4, 2, 64, 2, 32>(a, b, output, stream);
        break;
    case 6:
        launch_variant<InputT, 2, 4, 64, 2, 32>(a, b, output, stream);
        break;
    case 7:
        launch_variant<InputT, 2, 4, 64, 4, 32>(a, b, output, stream);
        break;
    case 8:
        launch_variant<InputT, 2, 4, 128, 2, 32>(a, b, output, stream);
        break;
    case 9:
        launch_variant<InputT, 2, 4, 64, 2, 64>(a, b, output, stream);
        break;
    case 10:
        launch_variant<InputT, 2, 4, 128, 2, 64>(a, b, output, stream);
        break;
    default:
        TORCH_CHECK(false, "unknown WMMA GEMM variant: ", variant);
    }
}

void check_inputs(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be ROCm tensors");
    TORCH_CHECK(a.device() == b.device(), "inputs must be on the same device");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be matrices");
    TORCH_CHECK(a.size(1) == b.size(0), "incompatible GEMM dimensions");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "input dtypes must match");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
                "only FP16 and BF16 are supported");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.size(0) % 16 == 0 && a.size(1) % 16 == 0 && b.size(1) % 16 == 0,
                "M, N and K must be divisible by 16");
    TORCH_CHECK(a.size(0) <= UINT32_MAX && a.size(1) <= UINT32_MAX && b.size(1) <= UINT32_MAX,
                "dimensions exceed the uint32 kernel ABI");
}

at::Tensor gemm_out(const at::Tensor& a,
                    const at::Tensor& b,
                    at::Tensor output,
                    int64_t variant) {
    check_inputs(a, b);
    TORCH_CHECK(output.is_cuda() && output.device() == a.device(),
                "output must be on the input device");
    TORCH_CHECK(output.scalar_type() == a.scalar_type(), "output dtype must match inputs");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == a.size(0) && output.size(1) == b.size(1),
                "output has the wrong shape");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    at::DeviceGuard guard(a.device());
    auto stream = c10::hip::getCurrentHIPStream(a.get_device()).stream();
    if(a.scalar_type() == at::kHalf) {
        launch<rocwmma::float16_t>(a, b, output, variant, stream);
    } else {
        launch<rocwmma::bfloat16_t>(a, b, output, variant, stream);
    }
    auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "WMMA GEMM launch failed: ", hipGetErrorString(error));
    return output;
}

at::Tensor gemm(const at::Tensor& a, const at::Tensor& b, int64_t variant) {
    check_inputs(a, b);
    auto output = at::empty({a.size(0), b.size(1)}, a.options());
    return gemm_out(a, b, output, variant);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gemm", &gemm, "Architecture-native ROCm WMMA GEMM");
    module.def("gemm_out", &gemm_out, "Architecture-native ROCm WMMA GEMM with fixed output");
}
