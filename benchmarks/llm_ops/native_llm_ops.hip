#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

__device__ float reduce_sum(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float reduce_max(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__global__ void rms_norm_kernel(
    const T* input, const T* weight, T* output, int64_t hidden, float eps) {
    __shared__ float shared[kThreads];
    const int64_t row = blockIdx.x;
    const int64_t base = row * hidden;
    float square_sum = 0.0f;
    for(int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        float value = as_float(input[base + column]);
        square_sum += value * value;
    }
    float inverse_rms = rsqrtf(reduce_sum(square_sum, shared) / static_cast<float>(hidden) + eps);
    for(int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        float value = as_float(input[base + column]);
        output[base + column] = from_float<T>(
            value * inverse_rms * as_float(weight[column]));
    }
}

template <typename T>
__global__ void add_rms_norm_kernel(const T* input,
                                    const T* residual,
                                    const T* weight,
                                    T* output,
                                    T* residual_out,
                                    int64_t hidden,
                                    float eps) {
    __shared__ float shared[kThreads];
    const int64_t row = blockIdx.x;
    const int64_t base = row * hidden;
    float square_sum = 0.0f;
    for(int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        T combined = from_float<T>(as_float(input[base + column]) + as_float(residual[base + column]));
        residual_out[base + column] = combined;
        float value = as_float(combined);
        square_sum += value * value;
    }
    float inverse_rms = rsqrtf(reduce_sum(square_sum, shared) / static_cast<float>(hidden) + eps);
    for(int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        float value = as_float(residual_out[base + column]);
        output[base + column] = from_float<T>(
            value * inverse_rms * as_float(weight[column]));
    }
}

template <typename T>
__global__ void rope_kernel(const T* input,
                            const T* cos,
                            const T* sin,
                            T* output,
                            int64_t elements,
                            int64_t heads,
                            int64_t head_dim) {
    for(int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t dimension = index % head_dim;
        int64_t token = index / (heads * head_dim);
        int64_t vector_base = index - dimension;
        int64_t half = head_dim / 2;
        int64_t rotated = dimension < half
            ? vector_base + dimension + half
            : vector_base + dimension - half;
        float rotated_value = as_float(input[rotated]);
        if(dimension < half) {
            rotated_value = -rotated_value;
        }
        int64_t table_index = token * head_dim + dimension;
        float value = as_float(input[index]) * as_float(cos[table_index])
            + rotated_value * as_float(sin[table_index]);
        output[index] = from_float<T>(value);
    }
}

template <typename T>
__global__ void silu_mul_kernel(const T* gate, const T* up, T* output, int64_t elements) {
    for(int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float gate_value = as_float(gate[index]);
        float silu = gate_value / (1.0f + expf(-gate_value));
        output[index] = from_float<T>(silu * as_float(up[index]));
    }
}

template <typename T>
__global__ void softmax_three_pass_kernel(const T* input, T* output, int64_t width) {
    __shared__ float shared[kThreads];
    const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
    float local_max = -INFINITY;
    for(int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        local_max = fmaxf(local_max, as_float(input[base + column]));
    }
    float row_max = reduce_max(local_max, shared);
    float local_sum = 0.0f;
    for(int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        local_sum += expf(as_float(input[base + column]) - row_max);
    }
    float inverse_sum = 1.0f / reduce_sum(local_sum, shared);
    for(int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[base + column] = from_float<T>(
            expf(as_float(input[base + column]) - row_max) * inverse_sum);
    }
}

struct OnlineStats {
    float maximum;
    float denominator;
};

__device__ __forceinline__ OnlineStats online_update(OnlineStats current, float value) {
    if(current.denominator == 0.0f) {
        return {value, 1.0f};
    }
    if(value > current.maximum) {
        return {value, current.denominator * expf(current.maximum - value) + 1.0f};
    }
    return {current.maximum, current.denominator + expf(value - current.maximum)};
}

__device__ __forceinline__ OnlineStats online_combine(OnlineStats left, OnlineStats right) {
    if(left.denominator == 0.0f) {
        return right;
    }
    if(right.denominator == 0.0f) {
        return left;
    }
    float maximum = fmaxf(left.maximum, right.maximum);
    return {
        maximum,
        left.denominator * expf(left.maximum - maximum)
            + right.denominator * expf(right.maximum - maximum),
    };
}

__device__ OnlineStats reduce_online(OnlineStats value, float* maxima, float* denominators) {
    maxima[threadIdx.x] = value.maximum;
    denominators[threadIdx.x] = value.denominator;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            OnlineStats left{maxima[threadIdx.x], denominators[threadIdx.x]};
            OnlineStats right{maxima[threadIdx.x + stride], denominators[threadIdx.x + stride]};
            OnlineStats combined = online_combine(left, right);
            maxima[threadIdx.x] = combined.maximum;
            denominators[threadIdx.x] = combined.denominator;
        }
        __syncthreads();
    }
    return {maxima[0], denominators[0]};
}

template <typename T>
__global__ void softmax_online_kernel(const T* input, T* output, int64_t width) {
    __shared__ float maxima[kThreads];
    __shared__ float denominators[kThreads];
    const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
    OnlineStats local{-INFINITY, 0.0f};
    for(int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        local = online_update(local, as_float(input[base + column]));
    }
    OnlineStats row = reduce_online(local, maxima, denominators);
    float inverse_sum = 1.0f / row.denominator;
    for(int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[base + column] = from_float<T>(
            expf(as_float(input[base + column]) - row.maximum) * inverse_sum);
    }
}

template <typename T>
__global__ void softmax_online_partials_kernel(const T* input,
                                               float* partials,
                                               int64_t width,
                                               int64_t chunks,
                                               int64_t chunk_size) {
    __shared__ float maxima[kThreads];
    __shared__ float denominators[kThreads];
    const int64_t row = blockIdx.x / chunks;
    const int64_t chunk = blockIdx.x % chunks;
    const int64_t begin = chunk * chunk_size;
    const int64_t end = begin + chunk_size < width ? begin + chunk_size : width;
    const int64_t base = row * width;
    OnlineStats local{-INFINITY, 0.0f};
    for(int64_t column = begin + threadIdx.x; column < end; column += blockDim.x) {
        local = online_update(local, as_float(input[base + column]));
    }
    OnlineStats result = reduce_online(local, maxima, denominators);
    if(threadIdx.x == 0) {
        int64_t index = (row * chunks + chunk) * 2;
        partials[index] = result.maximum;
        partials[index + 1] = result.denominator;
    }
}

__global__ void softmax_online_finalize_kernel(
    const float* partials, float* row_stats, int64_t chunks) {
    __shared__ float maxima[kThreads];
    __shared__ float denominators[kThreads];
    const int64_t row = blockIdx.x;
    OnlineStats local{-INFINITY, 0.0f};
    for(int64_t chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        int64_t index = (row * chunks + chunk) * 2;
        local = online_combine(local, {partials[index], partials[index + 1]});
    }
    OnlineStats result = reduce_online(local, maxima, denominators);
    if(threadIdx.x == 0) {
        row_stats[row * 2] = result.maximum;
        row_stats[row * 2 + 1] = 1.0f / result.denominator;
    }
}

template <typename T>
__global__ void softmax_online_output_kernel(const T* input,
                                             const float* row_stats,
                                             T* output,
                                             int64_t width,
                                             int64_t chunks,
                                             int64_t chunk_size) {
    const int64_t row = blockIdx.x / chunks;
    const int64_t chunk = blockIdx.x % chunks;
    const int64_t begin = chunk * chunk_size;
    const int64_t end = begin + chunk_size < width ? begin + chunk_size : width;
    const int64_t base = row * width;
    const float maximum = row_stats[row * 2];
    const float inverse_sum = row_stats[row * 2 + 1];
    for(int64_t column = begin + threadIdx.x; column < end; column += blockDim.x) {
        output[base + column] = from_float<T>(
            expf(as_float(input[base + column]) - maximum) * inverse_sum);
    }
}

void check_tensor(const at::Tensor& value, const char* name, int64_t rank) {
    TORCH_CHECK(value.is_cuda(), name, " must be a ROCm tensor");
    TORCH_CHECK(value.dim() == rank, name, " has the wrong rank");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.scalar_type() == at::kHalf || value.scalar_type() == at::kBFloat16,
                name,
                " must use FP16 or BF16");
}

void check_same(const at::Tensor& left, const at::Tensor& right, const char* name) {
    TORCH_CHECK(left.device() == right.device(), name, " must use the input device");
    TORCH_CHECK(left.scalar_type() == right.scalar_type(), name, " must use the input dtype");
}

template <typename HalfCall, typename BFloatCall>
void dispatch_dtype(const at::Tensor& input, HalfCall half_call, BFloatCall bfloat_call) {
    if(input.scalar_type() == at::kHalf) {
        half_call();
    } else {
        bfloat_call();
    }
    auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "native LLM operator launch failed: ", hipGetErrorString(error));
}

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    check_tensor(input, "input", 2);
    check_tensor(weight, "weight", 1);
    check_same(input, weight, "weight");
    TORCH_CHECK(weight.numel() == input.size(1), "weight must match hidden size");
    at::DeviceGuard guard(input.device());
    auto output = at::empty_like(input);
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    dim3 grid(input.size(0));
    dim3 block(kThreads);
    dispatch_dtype(
        input,
        [&] { hipLaunchKernelGGL((rms_norm_kernel<at::Half>), grid, block, 0, stream,
                                input.data_ptr<at::Half>(), weight.data_ptr<at::Half>(),
                                output.data_ptr<at::Half>(), input.size(1), static_cast<float>(eps)); },
        [&] { hipLaunchKernelGGL((rms_norm_kernel<at::BFloat16>), grid, block, 0, stream,
                                input.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
                                output.data_ptr<at::BFloat16>(), input.size(1), static_cast<float>(eps)); });
    return output;
}

std::vector<at::Tensor> add_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight, double eps) {
    check_tensor(input, "input", 2);
    check_tensor(residual, "residual", 2);
    check_tensor(weight, "weight", 1);
    check_same(input, residual, "residual");
    check_same(input, weight, "weight");
    TORCH_CHECK(input.sizes() == residual.sizes(), "residual shape must match input");
    TORCH_CHECK(weight.numel() == input.size(1), "weight must match hidden size");
    at::DeviceGuard guard(input.device());
    auto output = at::empty_like(input);
    auto residual_out = at::empty_like(input);
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    dim3 grid(input.size(0));
    dim3 block(kThreads);
    dispatch_dtype(
        input,
        [&] { hipLaunchKernelGGL((add_rms_norm_kernel<at::Half>), grid, block, 0, stream,
                                input.data_ptr<at::Half>(), residual.data_ptr<at::Half>(),
                                weight.data_ptr<at::Half>(), output.data_ptr<at::Half>(),
                                residual_out.data_ptr<at::Half>(), input.size(1), static_cast<float>(eps)); },
        [&] { hipLaunchKernelGGL((add_rms_norm_kernel<at::BFloat16>), grid, block, 0, stream,
                                input.data_ptr<at::BFloat16>(), residual.data_ptr<at::BFloat16>(),
                                weight.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(),
                                residual_out.data_ptr<at::BFloat16>(), input.size(1), static_cast<float>(eps)); });
    return {output, residual_out};
}

std::vector<at::Tensor> rope(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& cos, const at::Tensor& sin) {
    check_tensor(q, "q", 3);
    check_tensor(k, "k", 3);
    check_tensor(cos, "cos", 2);
    check_tensor(sin, "sin", 2);
    check_same(q, k, "k");
    check_same(q, cos, "cos");
    check_same(q, sin, "sin");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2),
                "q and k must share tokens and head dimension");
    TORCH_CHECK(q.size(2) % 2 == 0, "head dimension must be even");
    TORCH_CHECK(cos.sizes() == sin.sizes() && cos.size(0) == q.size(0)
                && cos.size(1) == q.size(2), "cos/sin shape mismatch");
    at::DeviceGuard guard(q.device());
    auto q_out = at::empty_like(q);
    auto k_out = at::empty_like(k);
    auto stream = c10::hip::getCurrentHIPStream(q.get_device()).stream();
    dim3 block(kThreads);
    dim3 q_grid(std::min<int64_t>((q.numel() + kThreads - 1) / kThreads, 65535));
    dim3 k_grid(std::min<int64_t>((k.numel() + kThreads - 1) / kThreads, 65535));
    dispatch_dtype(
        q,
        [&] {
            hipLaunchKernelGGL((rope_kernel<at::Half>), q_grid, block, 0, stream,
                               q.data_ptr<at::Half>(), cos.data_ptr<at::Half>(), sin.data_ptr<at::Half>(),
                               q_out.data_ptr<at::Half>(), q.numel(), q.size(1), q.size(2));
            hipLaunchKernelGGL((rope_kernel<at::Half>), k_grid, block, 0, stream,
                               k.data_ptr<at::Half>(), cos.data_ptr<at::Half>(), sin.data_ptr<at::Half>(),
                               k_out.data_ptr<at::Half>(), k.numel(), k.size(1), k.size(2));
        },
        [&] {
            hipLaunchKernelGGL((rope_kernel<at::BFloat16>), q_grid, block, 0, stream,
                               q.data_ptr<at::BFloat16>(), cos.data_ptr<at::BFloat16>(), sin.data_ptr<at::BFloat16>(),
                               q_out.data_ptr<at::BFloat16>(), q.numel(), q.size(1), q.size(2));
            hipLaunchKernelGGL((rope_kernel<at::BFloat16>), k_grid, block, 0, stream,
                               k.data_ptr<at::BFloat16>(), cos.data_ptr<at::BFloat16>(), sin.data_ptr<at::BFloat16>(),
                               k_out.data_ptr<at::BFloat16>(), k.numel(), k.size(1), k.size(2));
        });
    return {q_out, k_out};
}

at::Tensor silu_mul(const at::Tensor& gate, const at::Tensor& up) {
    check_tensor(gate, "gate", 2);
    check_tensor(up, "up", 2);
    check_same(gate, up, "up");
    TORCH_CHECK(gate.sizes() == up.sizes(), "up shape must match gate");
    at::DeviceGuard guard(gate.device());
    auto output = at::empty_like(gate);
    auto stream = c10::hip::getCurrentHIPStream(gate.get_device()).stream();
    dim3 block(kThreads);
    dim3 grid(std::min<int64_t>((gate.numel() + kThreads - 1) / kThreads, 65535));
    dispatch_dtype(
        gate,
        [&] { hipLaunchKernelGGL((silu_mul_kernel<at::Half>), grid, block, 0, stream,
                                gate.data_ptr<at::Half>(), up.data_ptr<at::Half>(),
                                output.data_ptr<at::Half>(), gate.numel()); },
        [&] { hipLaunchKernelGGL((silu_mul_kernel<at::BFloat16>), grid, block, 0, stream,
                                gate.data_ptr<at::BFloat16>(), up.data_ptr<at::BFloat16>(),
                                output.data_ptr<at::BFloat16>(), gate.numel()); });
    return output;
}

at::Tensor softmax_three_pass(const at::Tensor& input) {
    check_tensor(input, "input", 2);
    at::DeviceGuard guard(input.device());
    auto output = at::empty_like(input);
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    dim3 grid(input.size(0));
    dim3 block(kThreads);
    dispatch_dtype(
        input,
        [&] { hipLaunchKernelGGL((softmax_three_pass_kernel<at::Half>), grid, block, 0, stream,
                                input.data_ptr<at::Half>(), output.data_ptr<at::Half>(), input.size(1)); },
        [&] { hipLaunchKernelGGL((softmax_three_pass_kernel<at::BFloat16>), grid, block, 0, stream,
                                input.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(), input.size(1)); });
    return output;
}

at::Tensor softmax(const at::Tensor& input) {
    check_tensor(input, "input", 2);
    at::DeviceGuard guard(input.device());
    auto output = at::empty_like(input);
    auto stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
    constexpr int64_t kChunkSize = 4096;
    constexpr int64_t kSegmentedThreshold = 16384;
    dim3 block(kThreads);

    if(input.size(1) < kSegmentedThreshold) {
        dim3 grid(input.size(0));
        dispatch_dtype(
            input,
            [&] { hipLaunchKernelGGL((softmax_online_kernel<at::Half>), grid, block, 0, stream,
                                    input.data_ptr<at::Half>(), output.data_ptr<at::Half>(), input.size(1)); },
            [&] { hipLaunchKernelGGL((softmax_online_kernel<at::BFloat16>), grid, block, 0, stream,
                                    input.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(), input.size(1)); });
        return output;
    }

    const int64_t chunks = (input.size(1) + kChunkSize - 1) / kChunkSize;
    auto float_options = input.options().dtype(at::kFloat);
    auto partials = at::empty({input.size(0), chunks, 2}, float_options);
    auto row_stats = at::empty({input.size(0), 2}, float_options);
    dim3 chunk_grid(input.size(0) * chunks);
    dim3 row_grid(input.size(0));
    dispatch_dtype(
        input,
        [&] {
            hipLaunchKernelGGL((softmax_online_partials_kernel<at::Half>), chunk_grid, block, 0, stream,
                               input.data_ptr<at::Half>(), partials.data_ptr<float>(), input.size(1),
                               chunks, kChunkSize);
            hipLaunchKernelGGL(softmax_online_finalize_kernel, row_grid, block, 0, stream,
                               partials.data_ptr<float>(), row_stats.data_ptr<float>(), chunks);
            hipLaunchKernelGGL((softmax_online_output_kernel<at::Half>), chunk_grid, block, 0, stream,
                               input.data_ptr<at::Half>(), row_stats.data_ptr<float>(),
                               output.data_ptr<at::Half>(), input.size(1), chunks, kChunkSize);
        },
        [&] {
            hipLaunchKernelGGL((softmax_online_partials_kernel<at::BFloat16>), chunk_grid, block, 0, stream,
                               input.data_ptr<at::BFloat16>(), partials.data_ptr<float>(), input.size(1),
                               chunks, kChunkSize);
            hipLaunchKernelGGL(softmax_online_finalize_kernel, row_grid, block, 0, stream,
                               partials.data_ptr<float>(), row_stats.data_ptr<float>(), chunks);
            hipLaunchKernelGGL((softmax_online_output_kernel<at::BFloat16>), chunk_grid, block, 0, stream,
                               input.data_ptr<at::BFloat16>(), row_stats.data_ptr<float>(),
                               output.data_ptr<at::BFloat16>(), input.size(1), chunks, kChunkSize);
        });
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rms_norm", &rms_norm, "RMSNorm");
    module.def("add_rms_norm", &add_rms_norm, "Add + RMSNorm");
    module.def("rope", &rope, "split-half RoPE");
    module.def("silu_mul", &silu_mul, "SiLU multiply");
    module.def("softmax", &softmax, "online last-dimension softmax");
    module.def("softmax_three_pass", &softmax_three_pass, "three-pass last-dimension softmax");
}
