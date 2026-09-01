#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheAppendFullHeadDim = 256;

template <int KVHeadsValue>
struct KVCacheAppendFullGeometry {
    static_assert(KVHeadsValue == 4 || KVHeadsValue == 2);
    static constexpr int KVHeads = KVHeadsValue;
};

using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<4>;
using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;

struct KVCacheAppendDirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct KVCacheAppendBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

} // namespace ninfer::ops
