#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kWaveSize = 32;
constexpr int kMaximumValuesPerLane = 8;

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <typename T, int HeadDim, int WavesPerBlock>
__global__ __launch_bounds__(WavesPerBlock * kWaveSize) void sdpa_online_kernel(
    const T* query,
    const T* key,
    const T* value,
    T* output,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t sequence,
    bool is_causal,
    float scale) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int64_t query_row = static_cast<int64_t>(blockIdx.x) * WavesPerBlock + wave;
    const int64_t total_rows = batch * query_heads * sequence;
    if(query_row >= total_rows) {
        return;
    }

    const int64_t query_position = query_row % sequence;
    const int64_t flattened_head = query_row / sequence;
    const int64_t query_head = flattened_head % query_heads;
    const int64_t batch_index = flattened_head / query_heads;
    const int64_t kv_head = query_head / (query_heads / kv_heads);
    const int64_t query_base = query_row * HeadDim;
    constexpr int values_per_lane = HeadDim / kWaveSize;
    float query_values[kMaximumValuesPerLane] = {};
    float accumulator[kMaximumValuesPerLane] = {};

#pragma unroll
    for(int item = 0; item < values_per_lane; ++item) {
        const int dimension = lane + item * kWaveSize;
        query_values[item] = as_float(query[query_base + dimension]);
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    const int64_t valid_tokens = is_causal ? query_position + 1 : sequence;
    for(int64_t token = 0; token < valid_tokens; ++token) {
        const int64_t kv_base =
            ((batch_index * kv_heads + kv_head) * sequence + token) * HeadDim;
        float score = 0.0f;
#pragma unroll
        for(int item = 0; item < values_per_lane; ++item) {
            const int dimension = lane + item * kWaveSize;
            score += query_values[item] * as_float(key[kv_base + dimension]);
        }
#pragma unroll
        for(int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
            score += __shfl_down(score, offset, kWaveSize);
        }
        score = __shfl(score, 0, kWaveSize) * scale;
        const float next_max = fmaxf(running_max, score);
        const float previous_weight = running_sum > 0.0f
            ? __expf(running_max - next_max)
            : 0.0f;
        const float current_weight = __expf(score - next_max);
        running_sum = running_sum * previous_weight + current_weight;
#pragma unroll
        for(int item = 0; item < values_per_lane; ++item) {
            const int dimension = lane + item * kWaveSize;
            accumulator[item] = accumulator[item] * previous_weight
                + current_weight * as_float(value[kv_base + dimension]);
        }
        running_max = next_max;
    }

    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for(int item = 0; item < values_per_lane; ++item) {
        const int dimension = lane + item * kWaveSize;
        output[query_base + dimension] = from_float<T>(accumulator[item] * inverse_sum);
    }
}

void check_tensor(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a ROCm tensor");
    TORCH_CHECK(tensor.dim() == 4, name, " must be rank 4");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16,
        name,
        " must use FP16 or BF16");
}

template <typename T, int HeadDim, int WavesPerBlock>
void launch_sdpa(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& output,
    bool is_causal,
    float scale,
    hipStream_t stream) {
    const int64_t total_rows = query.size(0) * query.size(1) * query.size(2);
    dim3 block(WavesPerBlock * kWaveSize);
    dim3 grid((total_rows + WavesPerBlock - 1) / WavesPerBlock);
    hipLaunchKernelGGL(
        (sdpa_online_kernel<T, HeadDim, WavesPerBlock>),
        grid,
        block,
        0,
        stream,
        query.data_ptr<T>(),
        key.data_ptr<T>(),
        value.data_ptr<T>(),
        output.data_ptr<T>(),
        query.size(0),
        query.size(1),
        key.size(1),
        query.size(2),
        is_causal,
        scale);
}

template <typename T, int WavesPerBlock>
void launch_head_dimension(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& output,
    bool is_causal,
    float scale,
    hipStream_t stream) {
    if(query.size(3) == 64) {
        launch_sdpa<T, 64, WavesPerBlock>(query, key, value, output, is_causal, scale, stream);
    } else if(query.size(3) == 128) {
        launch_sdpa<T, 128, WavesPerBlock>(query, key, value, output, is_causal, scale, stream);
    } else {
        launch_sdpa<T, 256, WavesPerBlock>(query, key, value, output, is_causal, scale, stream);
    }
}

template <int WavesPerBlock>
at::Tensor sdpa_impl(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool is_causal,
    double scale) {
    check_tensor(query, "query");
    check_tensor(key, "key");
    check_tensor(value, "value");
    TORCH_CHECK(query.device() == key.device() && key.device() == value.device(), "Q/K/V devices must match");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(), "Q/K/V dtypes must match");
    TORCH_CHECK(key.sizes() == value.sizes(), "K/V shapes must match");
    TORCH_CHECK(query.size(0) == key.size(0), "Q/K/V batch must match");
    TORCH_CHECK(query.size(2) == key.size(2), "SDPA 1.0 requires equal Q/KV sequence lengths");
    TORCH_CHECK(query.size(3) == key.size(3), "Q/K/V head dimensions must match");
    TORCH_CHECK(
        query.size(3) == 64 || query.size(3) == 128 || query.size(3) == 256,
        "head dimension must be 64, 128, or 256");
    TORCH_CHECK(query.size(1) % key.size(1) == 0, "query heads must divide into KV heads");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be positive and finite");

    at::DeviceGuard guard(query.device());
    auto output = at::empty_like(query);
    auto stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
    if(query.scalar_type() == at::kHalf) {
        launch_head_dimension<at::Half, WavesPerBlock>(
            query, key, value, output, is_causal, static_cast<float>(scale), stream);
    } else {
        launch_head_dimension<at::BFloat16, WavesPerBlock>(
            query, key, value, output, is_causal, static_cast<float>(scale), stream);
    }
    const auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "SDPA launch failed: ", hipGetErrorString(error));
    return output;
}

at::Tensor sdpa_wave1(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool is_causal,
    double scale) {
    return sdpa_impl<1>(query, key, value, is_causal, scale);
}

at::Tensor sdpa_wave4(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool is_causal,
    double scale) {
    return sdpa_impl<4>(query, key, value, is_causal, scale);
}

at::Tensor sdpa_wave8(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool is_causal,
    double scale) {
    return sdpa_impl<8>(query, key, value, is_causal, scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("sdpa_wave1", &sdpa_wave1, "online SDPA, one wave per block");
    module.def("sdpa_wave4", &sdpa_wave4, "online SDPA, four waves per block");
    module.def("sdpa_wave8", &sdpa_wave8, "online SDPA, eight waves per block");
}
