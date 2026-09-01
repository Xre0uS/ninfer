#pragma once

// Per-state cache in front of build_json_token_mask.
//
// A full-vocabulary mask costs 4-8 ms to build against the 248,044-token vocabulary, against a
// 7.0 ms per-token budget at 143 tok/s. Rebuilt every token that roughly halves throughput and
// lands the lane BELOW the llama.cpp lane it exists to beat (measured: llama.cpp enforces a
// grammar at a 3.0% tax, 113.7 tok/s against 117.3 free). So the mask cannot be on the hot path.
//
// It does not need to be. A realistic document visits seven distinct constraint states, 89% of it
// in one -- see infra/ninfer/bench-state-census.cpp. Seven cold builds amortised over ~250 tokens
// is ~0.16 ms/token, about 2%, which is parity.
//
// The cache is keyed on JsonConstraint::state_key() and is sequence-independent: two requests at
// the same point in a document want the same mask, so one cache serves the whole engine.

#include "runtime/constraint/json_constraint.h"
#include "runtime/constraint/json_token_mask.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::constraint {

struct JsonMaskCacheStats {
    std::uint64_t hits      = 0;
    std::uint64_t misses    = 0;
    std::uint64_t evictions = 0;
};

class JsonMaskCache {
public:
    // Each entry is one bit per token -- ~31 KB at this vocabulary. The cap bounds a pathological
    // document (deeply nested, or one that walks many literal positions) rather than the ordinary
    // case, which needs single digits.
    static constexpr std::size_t kDefaultCapacity = 256;

    explicit JsonMaskCache(std::size_t capacity = kDefaultCapacity) : capacity_(capacity) {}

    // Returns a mask valid for `state`, building it only on a miss. The reference is stable until
    // the next call that misses.
    template <typename TokenBytes, typename IsTerminal>
    const std::vector<std::uint32_t>& mask_for(const JsonConstraint& state,
                                               std::size_t token_domain, TokenBytes&& token_bytes,
                                               IsTerminal&& is_terminal) {
        std::string key = state.state_key();
        if (const auto it = entries_.find(key); it != entries_.end()) {
            ++stats_.hits;
            return it->second;
        }
        ++stats_.misses;
        // Wholesale clear rather than LRU: reaching the cap at all means the document is unusual,
        // and an eviction policy is state to get wrong for a case that does not arise. The cost of
        // being wrong here is a rebuild, not a bad mask.
        if (entries_.size() >= capacity_) {
            entries_.clear();
            ++stats_.evictions;
        }
        std::vector<std::uint32_t> mask;
        build_json_token_mask(state, token_domain, token_bytes, is_terminal, mask);
        return entries_.emplace(std::move(key), std::move(mask)).first->second;
    }

    [[nodiscard]] const JsonMaskCacheStats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    void clear() noexcept { entries_.clear(); }

private:
    std::unordered_map<std::string, std::vector<std::uint32_t>> entries_;
    std::size_t capacity_;
    JsonMaskCacheStats stats_;
};

} // namespace ninfer::constraint
