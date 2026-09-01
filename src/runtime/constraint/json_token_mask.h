#pragma once

// Builds a sampler allow-mask from a JsonConstraint and a vocabulary.
//
// The sampler takes a device bitset over the token domain (ops::SamplingConfig::allow_mask); this
// produces the host side of it. For each token id, replay its bytes against a copy of the current
// constraint state and set the bit only if every byte is accepted.
//
// The vocabulary arrives as a callable rather than a Tokenizer reference so this stays testable
// without an artifact: any `std::string_view(int)` will do.
//
// Cost is O(vocab x token_length) per generated token. That is the honest naive shape and it is
// what makes constrained decoding slower than free decoding; llama.cpp pays the same and caches
// per-state. No caching here yet -- correctness first, and a cache keyed on a state that has to
// be compared for equality is its own piece of work.

#include "runtime/constraint/json_constraint.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::constraint {

struct JsonMaskStats {
    std::size_t allowed = 0;
    std::size_t total   = 0;
};

// words_for(n) u32 words cover n tokens.
[[nodiscard]] inline std::size_t json_mask_words(std::size_t token_domain) {
    return (token_domain + 31U) / 32U;
}

// `token_bytes(id)` returns the token's UTF-8 bytes; empty means "not usable" and is skipped.
// `is_terminal(id)` marks EOS-like tokens, which are legal only once the document is complete.
template <typename TokenBytes, typename IsTerminal>
JsonMaskStats build_json_token_mask(const JsonConstraint& state, std::size_t token_domain,
                                    TokenBytes&& token_bytes, IsTerminal&& is_terminal,
                                    std::vector<std::uint32_t>& out) {
    out.assign(json_mask_words(token_domain), 0U);
    JsonMaskStats stats;
    stats.total            = token_domain;
    const bool can_stop    = state.complete();

    for (std::size_t id = 0; id < token_domain; ++id) {
        const int token = static_cast<int>(id);
        if (is_terminal(token)) {
            // Stopping is legal exactly when the document is a finished value. Allowing EOS early
            // is how a constrained decode emits truncated JSON that still parses as a prefix.
            if (can_stop) {
                out[id >> 5U] |= 1U << (id & 31U);
                ++stats.allowed;
            }
            continue;
        }
        const std::string_view bytes = token_bytes(token);
        if (bytes.empty()) { continue; }
        if (state.can_accept_all(bytes)) {
            out[id >> 5U] |= 1U << (id & 31U);
            ++stats.allowed;
        }
    }

    // An empty mask is always a bug in the constraint: it means the automaton walked into a state
    // from which no continuation exists. It is not survivable as-is -- every logit becomes -inf,
    // and the sampler's tie-break over an all -inf candidate set returns token 0, so the model
    // emits whatever token 0 happens to decode to. Terminating is the least-wrong recovery, and
    // it keeps the failure visible as a truncated document rather than as garbage mid-string.
    if (stats.allowed == 0) {
        for (std::size_t id = 0; id < token_domain; ++id) {
            if (is_terminal(static_cast<int>(id))) {
                out[id >> 5U] |= 1U << (id & 31U);
                ++stats.allowed;
            }
        }
    }
    return stats;
}

} // namespace ninfer::constraint
