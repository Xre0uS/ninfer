#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

struct D256KVVectorProfile {
    DType code_dtype;
    std::int32_t code_leading_extent;
    DType scale_dtype;
    std::int32_t scale_leading_extent;
    std::int32_t quant_group;

    [[nodiscard]] constexpr bool scaled() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] static constexpr std::size_t logical_bytes() noexcept {
        return static_cast<std::size_t>(kD256KVCacheHeadDim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(code_leading_extent) * dtype_size(code_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }
};

struct D256KVCacheProfile {
    D256KVVectorProfile key;
    D256KVVectorProfile value;

    [[nodiscard]] static constexpr std::size_t logical_bytes_per_token_head() noexcept {
        return 2ULL * D256KVVectorProfile::logical_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }
};

inline D256KVCacheProfile d256_kv_cache_profile(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: {
        constexpr D256KVVectorProfile vector{DType::BF16, 256, DType::U8, 0, 0};
        return {vector, vector};
    }
    case KvCacheStorage::Int8Group64: {
        constexpr D256KVVectorProfile vector{DType::I8, 256, DType::FP16, 4, 64};
        return {vector, vector};
    }
    case KvCacheStorage::Fp8E4M3Row256: {
        constexpr D256KVVectorProfile vector{DType::FP8_E4M3FN, 256, DType::FP16, 1, 256};
        return {vector, vector};
    }
    case KvCacheStorage::Nvfp4Group16: {
        constexpr D256KVVectorProfile vector{DType::U8, 128, DType::U8, 16, 16};
        return {vector, vector};
    }
    case KvCacheStorage::K8V4: {
        constexpr D256KVVectorProfile key{DType::FP8_E4M3FN, 256, DType::FP16, 1, 256};
        constexpr D256KVVectorProfile value{DType::U8, 128, DType::U8, 16, 16};
        return {key, value};
    }
    default:
        throw std::invalid_argument("unsupported D256 KV-cache storage profile");
    }
}

} // namespace ninfer::ops
