#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/registry.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

[[nodiscard]] ModelSamplingDefaults default_sampling(Architecture architecture);

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    Architecture architecture              = Architecture::Qwen3_5;
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

// What a token-level constraint needs to know about the vocabulary, in terms that do not expose
// the tokenizer. `token_bytes` is indexed by token id and holds each token's LITERAL bytes -- an
// empty view means the token cannot appear in constrained output at all, which covers both special
// tokens and ids that are padding in the token domain.
//
// The bytes matter: Qwen uses GPT-2 byte-level BPE, so a raw vocab key stores a space as U+0120 and
// is not the token's bytes. A mask built from raw keys admits almost nothing and deadlocks
// decoding. These are decoded.
struct ConstraintVocabulary {
    std::span<const std::string_view> token_bytes;
    std::span<const TokenId> terminal_tokens;
    // Tokens that end the thinking block. A response constraint starts at the token AFTER one of
    // these, and under speculative decoding that boundary can fall INSIDE a draft block, so the
    // caller must recognise it in a drafted token rather than waiting for the next round.
    //
    // By id, not by text: the closing marker is a SPECIAL token, and token_bytes above blanks
    // specials because they can never appear in constrained output. Matching on text found nothing
    // and let the first response token through unconstrained.
    std::span<const TokenId> reasoning_close_tokens;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    // Stable for the lifetime of this Frontend; the views point into tokenizer-owned storage.
    [[nodiscard]] ConstraintVocabulary constraint_vocabulary() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output            = {},
                        const ThinkingControlOptions& thinking = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::models::qwen3_5
