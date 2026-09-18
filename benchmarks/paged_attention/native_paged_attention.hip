#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cfloat>
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

template <typename T, typename Index, int HeadDim, int WavesPerBlock>
__global__ __launch_bounds__(WavesPerBlock * kWaveSize) void paged_attention_kernel(
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const Index* block_tables,
    const Index* context_lengths,
    T* output,
    int64_t batch,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t num_blocks,
    int64_t block_size,
    int64_t max_blocks_per_sequence,
    float scale) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int64_t logical_head = static_cast<int64_t>(blockIdx.x) * WavesPerBlock + wave;
    const int64_t total_heads = batch * query_heads;
    if(logical_head >= total_heads) {
        return;
    }

    const int64_t sequence = logical_head / query_heads;
    const int64_t query_head = logical_head - sequence * query_heads;
    const int64_t group_size = query_heads / kv_heads;
    const int64_t kv_head = query_head / group_size;
    const int64_t context_length = static_cast<int64_t>(context_lengths[sequence]);
    constexpr int values_per_lane = HeadDim / kWaveSize;
    float query_values[kMaximumValuesPerLane] = {};
    float accumulator[kMaximumValuesPerLane] = {};

#pragma unroll
    for(int item = 0; item < values_per_lane; ++item) {
        const int dimension = lane + item * kWaveSize;
        query_values[item] = as_float(query[logical_head * HeadDim + dimension]);
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    const int64_t maximum_context = max_blocks_per_sequence * block_size;
    const int64_t valid_context = context_length > 0 && context_length <= maximum_context
        ? context_length
        : 0;
    for(int64_t token = 0; token < valid_context; ++token) {
        const int64_t logical_block = token / block_size;
        const int64_t block_offset = token - logical_block * block_size;
        const int64_t physical_block = static_cast<int64_t>(
            block_tables[sequence * max_blocks_per_sequence + logical_block]);
        if(physical_block < 0 || physical_block >= num_blocks) {
            continue;
        }
        const int64_t cache_base =
            ((physical_block * block_size + block_offset) * kv_heads + kv_head) * HeadDim;
        float score = 0.0f;
#pragma unroll
        for(int item = 0; item < values_per_lane; ++item) {
            const int dimension = lane + item * kWaveSize;
            score += query_values[item] * as_float(key_cache[cache_base + dimension]);
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
                + current_weight * as_float(value_cache[cache_base + dimension]);
        }
        running_max = next_max;
    }

    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for(int item = 0; item < values_per_lane; ++item) {
        const int dimension = lane + item * kWaveSize;
        output[logical_head * HeadDim + dimension] =
            from_float<T>(accumulator[item] * inverse_sum);
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

template <typename T, typename Index, int HeadDim, int WavesPerBlock>
void launch_attention(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lengths,
    at::Tensor& output,
    float scale,
    hipStream_t stream) {
    const int64_t total_heads = query.size(0) * query.size(1);
    dim3 block(WavesPerBlock * kWaveSize);
    dim3 grid((total_heads + WavesPerBlock - 1) / WavesPerBlock);
    hipLaunchKernelGGL(
        (paged_attention_kernel<T, Index, HeadDim, WavesPerBlock>),
        grid,
        block,
        0,
        stream,
        query.data_ptr<T>(),
        key_cache.data_ptr<T>(),
        value_cache.data_ptr<T>(),
        block_tables.data_ptr<Index>(),
        context_lengths.data_ptr<Index>(),
        output.data_ptr<T>(),
        query.size(0),
        query.size(1),
        key_cache.size(2),
        key_cache.size(0),
        key_cache.size(1),
        block_tables.size(1),
        scale);
}

template <typename T, typename Index, int WavesPerBlock>
void launch_head_dimension(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lengths,
    at::Tensor& output,
    float scale,
    hipStream_t stream) {
    if(query.size(2) == 64) {
        launch_attention<T, Index, 64, WavesPerBlock>(
            query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
    } else if(query.size(2) == 128) {
        launch_attention<T, Index, 128, WavesPerBlock>(
            query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
    } else {
        launch_attention<T, Index, 256, WavesPerBlock>(
            query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
    }
}

template <int WavesPerBlock>
at::Tensor paged_attention_impl(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lengths,
    double scale) {
    check_data_tensor(query, "query", 3);
    check_data_tensor(key_cache, "key_cache", 4);
    check_data_tensor(value_cache, "value_cache", 4);
    check_index_tensor(block_tables, "block_tables", 2);
    check_index_tensor(context_lengths, "context_lengths", 1);
    TORCH_CHECK(query.device() == key_cache.device(), "query and cache must use the same device");
    TORCH_CHECK(key_cache.device() == value_cache.device(), "K/V cache must use the same device");
    TORCH_CHECK(query.device() == block_tables.device(), "indexes must use the query device");
    TORCH_CHECK(query.device() == context_lengths.device(), "indexes must use the query device");
    TORCH_CHECK(query.scalar_type() == key_cache.scalar_type(), "query and cache dtypes must match");
    TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(), "K/V cache dtypes must match");
    TORCH_CHECK(block_tables.scalar_type() == context_lengths.scalar_type(), "index dtypes must match");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "K/V cache shapes must match");
    TORCH_CHECK(query.size(0) == block_tables.size(0), "block table batch must match query");
    TORCH_CHECK(query.size(0) == context_lengths.size(0), "context length batch must match query");
    TORCH_CHECK(query.size(2) == key_cache.size(3), "query and cache head dimensions must match");
    TORCH_CHECK(
        query.size(2) == 64 || query.size(2) == 128 || query.size(2) == 256,
        "head dimension must be 64, 128, or 256");
    TORCH_CHECK(query.size(1) % key_cache.size(2) == 0, "query heads must divide into KV heads");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be positive and finite");

    at::DeviceGuard guard(query.device());
    auto output = at::empty_like(query);
    auto stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
    if(query.scalar_type() == at::kHalf) {
        if(block_tables.scalar_type() == at::kInt) {
            launch_head_dimension<at::Half, int32_t, WavesPerBlock>(
                query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
        } else {
            launch_head_dimension<at::Half, int64_t, WavesPerBlock>(
                query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
        }
    } else if(block_tables.scalar_type() == at::kInt) {
        launch_head_dimension<at::BFloat16, int32_t, WavesPerBlock>(
            query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
    } else {
        launch_head_dimension<at::BFloat16, int64_t, WavesPerBlock>(
            query, key_cache, value_cache, block_tables, context_lengths, output, scale, stream);
    }
    const auto error = hipGetLastError();
    TORCH_CHECK(error == hipSuccess, "paged attention launch failed: ", hipGetErrorString(error));
    return output;
}

at::Tensor attention_wave1(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lengths,
    double scale) {
    return paged_attention_impl<1>(
        query, key_cache, value_cache, block_tables, context_lengths, scale);
}

at::Tensor attention_wave4(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lengths,
    double scale) {
    return paged_attention_impl<4>(
        query, key_cache, value_cache, block_tables, context_lengths, scale);
}

at::Tensor attention_wave8(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lengths,
    double scale) {
    return paged_attention_impl<8>(
        query, key_cache, value_cache, block_tables, context_lengths, scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("attention_wave1", &attention_wave1, "online paged attention, one wave per block");
    module.def("attention_wave4", &attention_wave4, "online paged attention, four waves per block");
    module.def("attention_wave8", &attention_wave8, "online paged attention, eight waves per block");
}
