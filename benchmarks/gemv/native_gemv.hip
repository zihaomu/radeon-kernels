#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kWaveSize = 32;
constexpr int kWavesPerBlock = kThreads / kWaveSize;

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <typename T, int Width>
struct alignas(sizeof(T) * Width) PackedValue {
    T values[Width];
};

template <typename T, int OutputsPerWave>
__global__ __launch_bounds__(kThreads) void gemv_wave_kernel(
    const T* input,
    const T* weight,
    T* output,
    int64_t m,
    int64_t n,
    int64_t k) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int64_t row = blockIdx.y;
    const int64_t first_column =
        (static_cast<int64_t>(blockIdx.x) * kWavesPerBlock + wave) * OutputsPerWave;
    if(row >= m || first_column >= n) {
        return;
    }

    float sums[OutputsPerWave] = {};
    const int64_t input_base = row * k;
    for(int64_t inner = lane; inner < k; inner += kWaveSize) {
        const float input_value = as_float(input[input_base + inner]);
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            const int64_t column = first_column + output_index;
            if(column < n) {
                sums[output_index] += input_value * as_float(weight[column * k + inner]);
            }
        }
    }

#pragma unroll
    for(int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            sums[output_index] += __shfl_down(sums[output_index], offset, kWaveSize);
        }
    }
    if(lane == 0) {
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            const int64_t column = first_column + output_index;
            if(column < n) {
                output[row * n + column] = from_float<T>(sums[output_index]);
            }
        }
    }
}

template <typename T, int OutputsPerWave, int PackWidth>
__global__ __launch_bounds__(kThreads) void gemv_packed_kernel(
    const T* input,
    const T* weight,
    T* output,
    int64_t m,
    int64_t n,
    int64_t k) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int64_t row = blockIdx.y;
    const int64_t first_column =
        (static_cast<int64_t>(blockIdx.x) * kWavesPerBlock + wave) * OutputsPerWave;
    if(row >= m || first_column >= n) {
        return;
    }

    float sums[OutputsPerWave] = {};
    const int64_t input_base = row * k;
    // Every weight row must share the requested vector alignment. If K does
    // not preserve it, keep the generic scalar path for the whole row.
    const int64_t aligned_k = k % PackWidth == 0 ? k : 0;
    for(int64_t inner = static_cast<int64_t>(lane) * PackWidth;
        inner < aligned_k;
        inner += kWaveSize * PackWidth) {
        const auto input_value = *reinterpret_cast<const PackedValue<T, PackWidth>*>(
            input + input_base + inner);
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            const int64_t column = first_column + output_index;
            if(column < n) {
                const auto weight_value = *reinterpret_cast<const PackedValue<T, PackWidth>*>(
                    weight + column * k + inner);
#pragma unroll
                for(int pack_index = 0; pack_index < PackWidth; ++pack_index) {
                    sums[output_index] += as_float(input_value.values[pack_index])
                        * as_float(weight_value.values[pack_index]);
                }
            }
        }
    }
    for(int64_t inner = aligned_k + lane; inner < k; inner += kWaveSize) {
        const float input_value = as_float(input[input_base + inner]);
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            const int64_t column = first_column + output_index;
            if(column < n) {
                sums[output_index] += input_value * as_float(weight[column * k + inner]);
            }
        }
    }

#pragma unroll
    for(int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            sums[output_index] += __shfl_down(sums[output_index], offset, kWaveSize);
        }
    }
    if(lane == 0) {
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            const int64_t column = first_column + output_index;
            if(column < n) {
                output[row * n + column] = from_float<T>(sums[output_index]);
            }
        }
    }
}

template <typename T, int OutputsPerWave>
__global__ __launch_bounds__(kThreads) void gemv_lds_kernel(
    const T* input,
    const T* weight,
    T* output,
    int64_t m,
    int64_t n,
    int64_t k) {
    __shared__ T input_tile[kThreads];
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int64_t row = blockIdx.y;
    const int64_t first_column =
        (static_cast<int64_t>(blockIdx.x) * kWavesPerBlock + wave) * OutputsPerWave;
    const bool active = row < m && first_column < n;
    float sums[OutputsPerWave] = {};
    const int64_t input_base = row * k;

    for(int64_t tile_base = 0; tile_base < k; tile_base += kThreads) {
        const int64_t load_index = tile_base + threadIdx.x;
        input_tile[threadIdx.x] = load_index < k ? input[input_base + load_index] : from_float<T>(0.0f);
        __syncthreads();
        const int tile_size = static_cast<int>(k - tile_base < kThreads ? k - tile_base : kThreads);
        if(active) {
            for(int inner = lane; inner < tile_size; inner += kWaveSize) {
                const float input_value = as_float(input_tile[inner]);
#pragma unroll
                for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
                    const int64_t column = first_column + output_index;
                    if(column < n) {
                        sums[output_index] += input_value
                            * as_float(weight[column * k + tile_base + inner]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            sums[output_index] += __shfl_down(sums[output_index], offset, kWaveSize);
        }
    }
    if(active && lane == 0) {
#pragma unroll
        for(int output_index = 0; output_index < OutputsPerWave; ++output_index) {
            const int64_t column = first_column + output_index;
            if(column < n) {
                output[row * n + column] = from_float<T>(sums[output_index]);
            }
        }
    }
}

void check_tensor(const at::Tensor& value, const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a ROCm tensor");
    TORCH_CHECK(value.dim() == 2, name, " must be rank 2");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.scalar_type() == at::kHalf || value.scalar_type() == at::kBFloat16,
                name,
                " must use FP16 or BF16");
}

template <typename T, int OutputsPerWave>
void launch_gemv(const at::Tensor& input,
                 const at::Tensor& weight,
                 at::Tensor& output,
                 hipStream_t stream) {
    const int64_t columns_per_block = kWavesPerBlock * OutputsPerWave;
    dim3 block(kThreads);
    dim3 grid((weight.size(0) + columns_per_block - 1) / columns_per_block, input.size(0));
    hipLaunchKernelGGL((gemv_wave_kernel<T, OutputsPerWave>),
                       grid,
                       block,
                       0,
                       stream,
                       input.data_ptr<T>(),
                       weight.data_ptr<T>(),
                       output.data_ptr<T>(),
                       input.size(0),
                       weight.size(0),
                       input.size(1));
}

template <typename T, int OutputsPerWave, int PackWidth>
void launch_gemv_packed(const at::Tensor& input,
                        const at::Tensor& weight,
                        at::Tensor& output,
                        hipStream_t stream) {
    const int64_t columns_per_block = kWavesPerBlock * OutputsPerWave;
    dim3 block(kThreads);
    dim3 grid((weight.size(0) + columns_per_block - 1) / columns_per_block, input.size(0));
    hipLaunchKernelGGL((gemv_packed_kernel<T, OutputsPerWave, PackWidth>),
                       grid,
                       block,
                       0,
                       stream,
                       input.data_ptr<T>(),
                       weight.data_ptr<T>(),
                       output.data_ptr<T>(),
                       input.size(0),
                       weight.size(0),
                       input.size(1));
}

template <typename T, int OutputsPerWave>
void launch_gemv_lds(const at::Tensor& input,
                     const at::Tensor& weight,
                     at::Tensor& output,
                     hipStream_t stream) {
    const int64_t columns_per_block = kWavesPerBlock * OutputsPerWave;
    dim3 block(kThreads);
    dim3 grid((weight.size(0) + columns_per_block - 1) / columns_per_block, input.size(0));
    hipLaunchKernelGGL((gemv_lds_kernel<T, OutputsPerWave>),
                       grid,
                       block,
                       0,
                       stream,
                       input.data_ptr<T>(),
                       weight.data_ptr<T>(),
                       output.data_ptr<T>(),
                       input.size(0),
                       weight.size(0),
                       input.size(1));
}

template <int OutputsPerWave>
at::Tensor gemv_impl(const at::Tensor& input, const at::Tensor& weight) {
    check_tensor(input, "input");
    check_tensor(weight, "weight");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must use the same device");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must use the same dtype");
    TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 32, "input M must be in [1, 32]");
    TORCH_CHECK(input.size(1) > 0 && weight.size(0) > 0, "N and K must be positive");
    TORCH_CHECK(input.size(1) == weight.size(1), "input and weight must share K");
    at::DeviceGuard guard(input.device());
    auto output = at::empty({input.size(0), weight.size(0)}, input.options());
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    if(input.scalar_type() == at::kHalf) {
        launch_gemv<at::Half, OutputsPerWave>(input, weight, output, stream);
    } else {
        launch_gemv<at::BFloat16, OutputsPerWave>(input, weight, output, stream);
    }
    auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "GEMV launch failed: ", hipGetErrorString(error));
    return output;
}

template <int OutputsPerWave, int PackWidth>
at::Tensor gemv_packed_impl(const at::Tensor& input, const at::Tensor& weight) {
    check_tensor(input, "input");
    check_tensor(weight, "weight");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must use the same device");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must use the same dtype");
    TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 32, "input M must be in [1, 32]");
    TORCH_CHECK(input.size(1) > 0 && weight.size(0) > 0, "N and K must be positive");
    TORCH_CHECK(input.size(1) == weight.size(1), "input and weight must share K");
    at::DeviceGuard guard(input.device());
    auto output = at::empty({input.size(0), weight.size(0)}, input.options());
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    if(input.scalar_type() == at::kHalf) {
        launch_gemv_packed<at::Half, OutputsPerWave, PackWidth>(input, weight, output, stream);
    } else {
        launch_gemv_packed<at::BFloat16, OutputsPerWave, PackWidth>(input, weight, output, stream);
    }
    auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "packed GEMV launch failed: ", hipGetErrorString(error));
    return output;
}

template <int OutputsPerWave>
at::Tensor gemv_lds_impl(const at::Tensor& input, const at::Tensor& weight) {
    check_tensor(input, "input");
    check_tensor(weight, "weight");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must use the same device");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must use the same dtype");
    TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 32, "input M must be in [1, 32]");
    TORCH_CHECK(input.size(1) > 0 && weight.size(0) > 0, "N and K must be positive");
    TORCH_CHECK(input.size(1) == weight.size(1), "input and weight must share K");
    at::DeviceGuard guard(input.device());
    auto output = at::empty({input.size(0), weight.size(0)}, input.options());
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    if(input.scalar_type() == at::kHalf) {
        launch_gemv_lds<at::Half, OutputsPerWave>(input, weight, output, stream);
    } else {
        launch_gemv_lds<at::BFloat16, OutputsPerWave>(input, weight, output, stream);
    }
    auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "LDS GEMV launch failed: ", hipGetErrorString(error));
    return output;
}

at::Tensor gemv_wave1(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_impl<1>(input, weight);
}

at::Tensor gemv_wave2(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_impl<2>(input, weight);
}

at::Tensor gemv_wave4(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_impl<4>(input, weight);
}

at::Tensor gemv_packed2_wave1(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_packed_impl<1, 2>(input, weight);
}

at::Tensor gemv_packed2_wave2(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_packed_impl<2, 2>(input, weight);
}

at::Tensor gemv_packed4_wave1(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_packed_impl<1, 4>(input, weight);
}

at::Tensor gemv_packed4_wave2(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_packed_impl<2, 4>(input, weight);
}

at::Tensor gemv_lds_wave1(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_lds_impl<1>(input, weight);
}

at::Tensor gemv_lds_wave2(const at::Tensor& input, const at::Tensor& weight) {
    return gemv_lds_impl<2>(input, weight);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gemv_wave1", &gemv_wave1, "wave32 GEMV with one output per wave");
    module.def("gemv_wave2", &gemv_wave2, "wave32 GEMV with two outputs per wave");
    module.def("gemv_wave4", &gemv_wave4, "wave32 GEMV with four outputs per wave");
    module.def("gemv_packed2_wave1", &gemv_packed2_wave1, "packed2 GEMV with one output per wave");
    module.def("gemv_packed2_wave2", &gemv_packed2_wave2, "packed2 GEMV with two outputs per wave");
    module.def("gemv_packed4_wave1", &gemv_packed4_wave1, "packed4 GEMV with one output per wave");
    module.def("gemv_packed4_wave2", &gemv_packed4_wave2, "packed4 GEMV with two outputs per wave");
    module.def("gemv_lds_wave1", &gemv_lds_wave1, "LDS-tiled GEMV with one output per wave");
    module.def("gemv_lds_wave2", &gemv_lds_wave2, "LDS-tiled GEMV with two outputs per wave");
}
