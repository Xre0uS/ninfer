#pragma once

// Incremental JSON validator for constrained decoding.
//
// Answers one question per byte: "can a valid JSON document continue with this?" That is what a
// token mask needs -- for each candidate token, replay its bytes and keep the token only if every
// byte is accepted.
//
// Prefix-validity, not validity: "{\"a\":" is accepted because some completion of it is a valid
// document. `complete()` reports whether the document could stop here.
//
// This is the structural half of `response_format: {"type":"json_object"}`. Schema constraints
// (properties, required, enum) build on top and are not modelled yet.
//
// Lives under runtime/ rather than serve/ because the ENGINE enforces the constraint: it is
// the layer that sees committed tokens and owns the per-lane mask upload. ninfer_serve links
// ninfer_engine, so a constraint under serve/ would be a dependency inversion.

#include "runtime/constraint/json_schema.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::constraint {

class JsonConstraint {
public:
    // What the top-level value is allowed to be.
    //
    // OpenAI's response_format {"type":"json_object"} means a JSON OBJECT, not merely a valid JSON
    // value. Allowing any value lets a model answer a prose question with a bare top-level string
    // -- '"The capital of France is Paris."' parses, and is not what the caller asked for.
    enum class RootKind : std::uint8_t {
        AnyValue,
        Object,
    };

    explicit JsonConstraint(RootKind root = RootKind::AnyValue) : root_(root) {
        stack_.push_back(FrameState{.kind = Frame::Root});
    }

    // Schema-constrained construction. The root kind is implied by the schema's own root type, so
    // it is not taken separately -- a schema saying "object" already says what json_object says.
    explicit JsonConstraint(std::shared_ptr<const CompiledSchema> schema)
        : schema_(std::move(schema)) {
        stack_.push_back(FrameState{.kind = Frame::Root, .node = CompiledSchema::root()});
        if (schema_ && !schema_->empty() &&
            schema_->node(CompiledSchema::root()).kind == SchemaNode::Kind::Object) {
            root_ = RootKind::Object;
        }
    }

    // Advance over one byte. Returns false and leaves the state untouched if the byte cannot
    // appear next in any valid document.
    bool accept(char byte) { return step(static_cast<unsigned char>(byte), true); }

    // Test one byte without advancing.
    [[nodiscard]] bool can_accept(char byte) const {
        JsonConstraint copy = *this;
        return copy.step(static_cast<unsigned char>(byte), true);
    }

    // Replay a whole token's bytes. All-or-nothing: on any rejection the state is unchanged, so
    // this doubles as the mask predicate.
    bool accept_all(std::string_view bytes) {
        const JsonConstraint saved = *this;
        for (const char c : bytes) {
            if (!accept(c)) {
                *this = saved;
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool can_accept_all(std::string_view bytes) const {
        JsonConstraint copy = *this;
        return copy.accept_all(bytes);
    }

    // A cache key over the states that matter to a mask.
    //
    // Two constraints with equal keys accept exactly the same byte set, so a mask built under one
    // is valid under the other. This is CONSERVATIVE rather than exact: identical stack implies
    // identical acceptance, but two states with different stacks can still accept the same bytes
    // and will get separate entries. That costs a cache miss, never a wrong mask.
    //
    // The exact equivalence is the accepted-byte set itself, but computing that costs 256 trial
    // steps -- more than the key is worth. See infra/ninfer/bench-state-census.cpp, which measures
    // a realistic document visiting seven distinct states with an 89% concentration in one.
    [[nodiscard]] std::string state_key() const {
        std::string key;
        key.reserve(stack_.size() * 8 + 8);
        const auto push32 = [&key](std::uint32_t value) {
            for (int shift = 0; shift < 32; shift += 8) {
                key.push_back(static_cast<char>((value >> shift) & 0xFFU));
            }
        };
        if (schema_ != nullptr) {
            // Which schema, not just which state. Node indices are per-schema, so two schemas
            // would otherwise share a key and the mask cache would serve one the other's masks.
            const std::uint64_t id = schema_->id();
            for (int shift = 0; shift < 64; shift += 8) {
                key.push_back(static_cast<char>((id >> shift) & 0xFFU));
            }
        }
        for (const FrameState& frame : stack_) {
            key.push_back(static_cast<char>(frame.kind));
            // Under a schema the frame's node, which properties it has already emitted, and how
            // many items an array holds all change which bytes come next, so all three are part
            // of the state. Without a schema they are constant and cost nothing.
            if (schema_ != nullptr) {
                push32(frame.node);
                push32(static_cast<std::uint32_t>(frame.emitted));
                push32(static_cast<std::uint32_t>(frame.emitted >> 32));
                push32(frame.count);
                push32(static_cast<std::uint32_t>(frame.property));
            }
        }
        key.push_back(static_cast<char>(key_ ? 1 : 0));
        key.push_back(static_cast<char>(number_complete_ ? 1 : 0));
        // literal_ is not cleared when a literal finishes, so it is only part of the state while
        // a Literal frame is actually open -- including it otherwise would split identical states.
        if (!stack_.empty() && stack_.back().kind == Frame::Literal) {
            key.push_back(static_cast<char>(literal_pos_));
            key.push_back(literal_.empty() ? '\0' : literal_.front());
        }
        // The partial key or enum member decides which bytes may follow -- but only under a
        // schema. Unconstrained, any key text accepts the same next bytes, so including it there
        // would split identical states and cost cache misses for nothing.
        if (schema_ != nullptr &&
            (key_ || (!stack_.empty() && stack_.back().kind == Frame::EnumLiteral))) {
            key.push_back('\x01');
            key.append(literal_text_);
        }
        return key;
    }

    // True when the document is a finished top-level value.
    [[nodiscard]] bool complete() const {
        if (stack_.size() == 1 && stack_.back().kind == Frame::Done) { return true; }
        // A number has no terminator, so at end of input a pending Number frame is a finished
        // top-level value: "123" is a whole document even though nothing closed it. But only when
        // it ends on a digit -- "1.", "1e" and "-" are all prefixes of a number and none of them
        // is a document. Reporting them complete lets a constrained decode emit "1." and stop,
        // which is what this originally did.
        if (stack_.size() != 2 || stack_.front().kind != Frame::Root) { return false; }
        // An enum member has no terminator either, and for the same reason: 1 is a prefix of 12.
        if (stack_.back().kind == Frame::EnumLiteral) { return enum_exact(); }
        return stack_.back().kind == Frame::Number && number_complete_;
    }

private:
    enum class Frame : std::uint8_t {
        Root,        // expecting the single top-level value
        Done,        // top-level value finished
        ObjectFirstKey, // expecting '"' of a key, or '}' -- only legal while the object is EMPTY
        ObjectKey,      // expecting '"' of a key AFTER a comma; '}' here is a trailing comma
        ObjectColon,    // expecting ':'
        ObjectValue,    // expecting a value
        ObjectNext,     // expecting ',' or '}'
        ArrayFirstValue,// expecting a value, or ']' -- only legal while the array is EMPTY
        ArrayValue,     // expecting a value AFTER a comma; ']' here is a trailing comma
        ArrayNext,      // expecting ',' or ']'
        String,      // inside a string
        StringEsc,   // just consumed '\'
        StringHex1,  // inside \uXXXX
        StringHex2,
        StringHex3,
        StringHex4,
        Number,      // inside a number literal
        Literal,     // inside true/false/null
        EnumLiteral, // matching one of a schema enum's exact JSON encodings
    };

    // One entry per open value. `node` is the schema governing this frame: for a container it is
    // the container's own node, so property_nodes/items stay reachable while the frame is open.
    struct FrameState {
        Frame kind             = Frame::Root;
        std::uint32_t node     = kNoSchemaNode;
        std::uint64_t emitted  = 0;  // Object: declared properties already seen
        std::uint32_t count    = 0;  // Array: completed items
        std::int32_t property  = -1; // Object: property whose value is being built
    };

    static bool is_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
    static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
    static bool is_hex(unsigned char c) {
        return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    static bool starts_value(unsigned char c) {
        return c == '{' || c == '[' || c == '"' || c == '-' || is_digit(c) || c == 't' ||
               c == 'f' || c == 'n';
    }

    // Push the frames a new value needs. Scalars finish immediately via finish_value().
    // `node` is the schema governing the value being opened, not the frame it opens inside.
    void open_value(unsigned char c, std::uint32_t node) {
        // An enum pins the value to a fixed set of literals, so it is matched as raw JSON text
        // rather than parsed -- which is why the compiler stores each member's exact encoding.
        if (node != kNoSchemaNode && schema_ != nullptr &&
            schema_->node(node).kind == SchemaNode::Kind::Enum) {
            literal_text_.assign(1, static_cast<char>(c));
            stack_.push_back(FrameState{.kind = Frame::EnumLiteral, .node = node});
            return;
        }
        switch (c) {
        case '{':
            stack_.push_back(FrameState{.kind = Frame::ObjectFirstKey, .node = node});
            return;
        case '[':
            stack_.push_back(FrameState{.kind = Frame::ArrayFirstValue, .node = node});
            return;
        case '"':
            stack_.push_back(FrameState{.kind = Frame::String, .node = node});
            return;
        case 't':
            literal_ = "true";  literal_pos_ = 1;
            stack_.push_back(FrameState{.kind = Frame::Literal, .node = node});
            return;
        case 'f':
            literal_ = "false"; literal_pos_ = 1;
            stack_.push_back(FrameState{.kind = Frame::Literal, .node = node});
            return;
        case 'n':
            literal_ = "null";  literal_pos_ = 1;
            stack_.push_back(FrameState{.kind = Frame::Literal, .node = node});
            return;
        default:
            // '-' alone is not yet a number; a digit must follow before the value
            // can be considered finished.
            number_complete_ = is_digit(c);
            stack_.push_back(FrameState{.kind = Frame::Number, .node = node});
            return;
        }
    }

    // The schema governing a value opened from the frame currently on top.
    [[nodiscard]] std::uint32_t child_node() const {
        if (schema_ == nullptr) { return kNoSchemaNode; }
        const FrameState& frame = stack_.back();
        switch (frame.kind) {
        case Frame::Root: return CompiledSchema::root();
        case Frame::ObjectValue: {
            if (frame.node == kNoSchemaNode) { return kNoSchemaNode; }
            const SchemaNode& object = schema_->node(frame.node);
            if (frame.property < 0 ||
                static_cast<std::size_t>(frame.property) >= object.property_nodes.size()) {
                return kNoSchemaNode; // an additionalProperties key: unconstrained value
            }
            return object.property_nodes[static_cast<std::size_t>(frame.property)];
        }
        case Frame::ArrayFirstValue:
        case Frame::ArrayValue:
            return frame.node == kNoSchemaNode ? kNoSchemaNode : schema_->node(frame.node).items;
        default: return kNoSchemaNode;
        }
    }

    // Which openers the schema permits for a value. Without a schema every JSON value opener is
    // legal, which is exactly the unconstrained behaviour.
    [[nodiscard]] bool schema_allows_opener(std::uint32_t node, unsigned char c) const {
        if (schema_ == nullptr || node == kNoSchemaNode) { return starts_value(c); }
        switch (schema_->node(node).kind) {
        case SchemaNode::Kind::Any:     return starts_value(c);
        case SchemaNode::Kind::Object:  return c == '{';
        case SchemaNode::Kind::Array:   return c == '[';
        case SchemaNode::Kind::String:  return c == '"';
        case SchemaNode::Kind::Number:
        case SchemaNode::Kind::Integer: return c == '-' || is_digit(c);
        case SchemaNode::Kind::Boolean: return c == 't' || c == 'f';
        case SchemaNode::Kind::Null:    return c == 'n';
        case SchemaNode::Kind::Enum:
            for (const std::string& value : schema_->node(node).enum_values) {
                if (!value.empty() && static_cast<unsigned char>(value.front()) == c) {
                    return true;
                }
            }
            return false;
        }
        return false;
    }

    // An array may take another item only while it is under maxItems.
    [[nodiscard]] bool array_accepts_item() const {
        const FrameState& frame = stack_.back();
        if (schema_ == nullptr || frame.node == kNoSchemaNode) { return true; }
        return frame.count < schema_->node(frame.node).max_items;
    }

    // An array may close only once it has met minItems.
    [[nodiscard]] bool array_may_close() const {
        const FrameState& frame = stack_.back();
        if (schema_ == nullptr || frame.node == kNoSchemaNode) { return true; }
        return frame.count >= schema_->node(frame.node).min_items;
    }

    // An object may take another key only if one is actually available. Without this, a comma is
    // legal after the last declared property under additionalProperties:false, and the state it
    // leads to accepts NOTHING -- no key is possible and '}' would be a trailing comma. A dead
    // end like that empties the token mask, and an all -inf candidate set makes the sampler's
    // tie-break return token 0, which is how this first showed up: a stray "!" key in the output.
    [[nodiscard]] bool object_may_take_key() const {
        const FrameState& frame = stack_.back();
        if (schema_ == nullptr || frame.node == kNoSchemaNode) { return true; }
        const SchemaNode& node = schema_->node(frame.node);
        if (node.additional_properties) { return true; }
        for (std::size_t i = 0; i < node.property_names.size(); ++i) {
            if ((frame.emitted & (1ULL << i)) == 0) { return true; }
        }
        return false;
    }

    // An object may close only once every required property has been emitted.
    [[nodiscard]] bool object_may_close() const {
        const FrameState& frame = stack_.back();
        if (schema_ == nullptr || frame.node == kNoSchemaNode) { return true; }
        const std::uint64_t required = schema_->node(frame.node).required_mask;
        return (frame.emitted & required) == required;
    }

    // Whether a key may still be extended by `next`, or closed as it stands when `next` is 0.
    // Undeclared keys are legal exactly when the schema allows additional properties, and a
    // declared key may not repeat.
    [[nodiscard]] bool key_viable(char next, bool closing) const {
        const FrameState& object = stack_[stack_.size() - 2];
        if (schema_ == nullptr || object.node == kNoSchemaNode) { return true; }
        const SchemaNode& node = schema_->node(object.node);
        if (node.additional_properties) { return true; }
        for (std::size_t i = 0; i < node.property_names.size(); ++i) {
            if ((object.emitted & (1ULL << i)) != 0) { continue; } // already used
            const std::string& name = node.property_names[i];
            if (closing) {
                if (name.size() == literal_text_.size() && name == literal_text_) { return true; }
                continue;
            }
            if (name.size() > literal_text_.size() &&
                name.compare(0, literal_text_.size(), literal_text_) == 0 &&
                name[literal_text_.size()] == next) {
                return true;
            }
        }
        return false;
    }

    // A value just ended; tell the enclosing frame.
    void finish_value() {
        stack_.pop_back();
        switch (stack_.back().kind) {
        case Frame::Root:        stack_.back().kind = Frame::Done; return;
        case Frame::ObjectValue: stack_.back().kind = Frame::ObjectNext;
                                 stack_.back().property = -1; return;
        case Frame::ArrayFirstValue:
        case Frame::ArrayValue:  stack_.back().kind = Frame::ArrayNext;
                                 ++stack_.back().count; return;
        default: return; // unreachable for well-formed states
        }
    }

    bool step(unsigned char c, bool commit) {
        (void) commit; // callers snapshot; kept for symmetry with a future non-committing path
        const Frame frame = stack_.back().kind;

        // Whitespace is legal anywhere except inside a string or a partial literal -- and,
        // deliberately, except at the two ends of the document.
        //
        // Trailing whitespace is valid JSON, but permitting it lets a constrained decode pad with
        // spaces forever instead of emitting EOS, burning the token budget on a document that was
        // already done. Once complete, stopping is the only legal move.
        //
        // Leading whitespace is the same trap from the other end, and a worse one: a model steered
        // off its preferred opening finds whitespace is the only thing legal besides '{' and can
        // sit there indefinitely. Observed -- a prose question under json_object generated nothing
        // but newlines until it hit the token limit. A generated document starts at its value.
        if (is_ws(c) && frame != Frame::Root && frame != Frame::Done && frame != Frame::String &&
            frame != Frame::StringEsc && frame != Frame::Literal && frame < Frame::StringHex1) {
            if (frame == Frame::Number) { finish_number(); }
            return true;
        }

        switch (frame) {
        case Frame::Root: {
            const std::uint32_t node = child_node();
            if (root_ == RootKind::Object ? c != '{' : !schema_allows_opener(node, c)) {
                return false;
            }
            open_value(c, node);
            return true;
        }

        case Frame::Done:
            return false; // nothing may follow the top-level value

        case Frame::ObjectFirstKey:
        case Frame::ObjectKey:
            // '}' closes only an EMPTY object; after a comma a key is mandatory. Under a schema it
            // also closes only once every required property has actually been emitted, which is
            // what turns `required` into an enforceable constraint rather than a hint.
            if (c == '}') {
                if (frame != Frame::ObjectFirstKey) { return false; }
                if (!object_may_close()) { return false; }
                finish_value();
                return true;
            }
            if (c != '"') { return false; }
            stack_.push_back(FrameState{.kind = Frame::String});
            key_ = true;
            literal_text_.clear();
            return true;

        case Frame::ObjectColon:
            if (c != ':') { return false; }
            stack_.back().kind = Frame::ObjectValue;
            return true;

        case Frame::ObjectValue: {
            const std::uint32_t node = child_node();
            if (!schema_allows_opener(node, c)) { return false; }
            open_value(c, node);
            return true;
        }

        case Frame::ObjectNext:
            if (c == ',') {
                if (!object_may_take_key()) { return false; }
                stack_.back().kind = Frame::ObjectKey;
                return true;
            }
            if (c == '}') {
                if (!object_may_close()) { return false; }
                finish_value();
                return true;
            }
            return false;

        case Frame::ArrayFirstValue:
        case Frame::ArrayValue:
            // ']' closes only an EMPTY array; after a comma a value is mandatory. Under a
            // schema it also closes only once minItems is met.
            if (c == ']') {
                if (frame != Frame::ArrayFirstValue || !array_may_close()) { return false; }
                finish_value();
                return true;
            }
            if (!array_accepts_item()) { return false; }
            {
                const std::uint32_t node = child_node();
                if (!schema_allows_opener(node, c)) { return false; }
                open_value(c, node);
            }
            return true;

        case Frame::ArrayNext:
            if (c == ',') {
                if (!array_accepts_item()) { return false; } // maxItems
                stack_.back().kind = Frame::ArrayValue;
                return true;
            }
            if (c == ']') {
                if (!array_may_close()) { return false; } // minItems
                finish_value();
                return true;
            }
            return false;

        case Frame::String:
            if (c == '"') {
                if (key_) { // that string was an object key
                    if (!key_viable('\0', true)) { return false; }
                    stack_.pop_back();
                    key_ = false;
                    resolve_key();
                    stack_.back().kind = Frame::ObjectColon;
                    return true;
                }
                finish_value(); // pops this String frame
                return true;
            }
            if (c < 0x20) { return false; } // raw control bytes are not legal in a JSON string
            if (key_ && !key_viable(static_cast<char>(c), false)) { return false; }
            if (c == '\\') { stack_.back().kind = Frame::StringEsc; return true; }
            if (key_) { literal_text_.push_back(static_cast<char>(c)); }
            return true;

        case Frame::StringEsc:
            if (c == 'u') { stack_.back().kind = Frame::StringHex1; return true; }
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
                c == 'r' || c == 't') {
                stack_.back().kind = Frame::String;
                return true;
            }
            return false;

        case Frame::StringHex1:
            if (!is_hex(c)) { return false; }
            stack_.back().kind = Frame::StringHex2;
            return true;
        case Frame::StringHex2:
            if (!is_hex(c)) { return false; }
            stack_.back().kind = Frame::StringHex3;
            return true;
        case Frame::StringHex3:
            if (!is_hex(c)) { return false; }
            stack_.back().kind = Frame::StringHex4;
            return true;
        case Frame::StringHex4:
            if (!is_hex(c)) { return false; }
            stack_.back().kind = Frame::String;
            return true;

        case Frame::Number:
            // An integer is a number that may not grow a fractional or exponent part.
            if ((c == '.' || c == 'e' || c == 'E') && schema_ != nullptr &&
                stack_.back().node != kNoSchemaNode &&
                schema_->node(stack_.back().node).kind == SchemaNode::Kind::Integer) {
                return false;
            }
            if (is_digit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                // A number may only END on a digit. Every other member of this set leaves it
                // mid-token, so it stops being a completable document until a digit arrives.
                number_complete_ = is_digit(c);
                return true; // deliberately permissive: shape is checked, not numeric grammar
            }
            if (!number_complete_) { return false; } // "-", "1." and "1e" cannot end here
            // The number ended; re-dispatch this byte to the enclosing frame.
            finish_number();
            return step(c, true);

        case Frame::Literal:
            if (literal_pos_ >= literal_.size() || c != literal_[literal_pos_]) { return false; }
            ++literal_pos_;
            if (literal_pos_ == literal_.size()) { finish_value(); }
            return true;

        case Frame::EnumLiteral: {
            // Enum members are matched as raw JSON text, so a string member carries its quotes and
            // a numeric one does not -- one code path covers both.
            if (schema_ == nullptr || stack_.back().node == kNoSchemaNode) { return false; }
            const SchemaNode& node = schema_->node(stack_.back().node);
            for (const std::string& value : node.enum_values) {
                if (value.size() > literal_text_.size() &&
                    value.compare(0, literal_text_.size(), literal_text_) == 0 &&
                    static_cast<unsigned char>(value[literal_text_.size()]) == c) {
                    literal_text_.push_back(static_cast<char>(c));
                    return true;
                }
            }
            // Nothing extends: the literal is finished if it exactly matches a member. Numeric
            // members make this necessary -- 1 is a prefix of 12, so completion cannot be eager.
            if (!enum_exact()) { return false; }
            finish_value();
            return step(c, true);
        }
        }
        return false;
    }

    void finish_number() { finish_value(); }

    // Records which declared property the just-closed key named, so the value that follows is
    // validated against that property's schema and `required` can be satisfied.
    void resolve_key() {
        FrameState& object = stack_.back();
        object.property    = -1;
        if (schema_ == nullptr || object.node == kNoSchemaNode) { return; }
        const SchemaNode& node = schema_->node(object.node);
        for (std::size_t i = 0; i < node.property_names.size(); ++i) {
            if (node.property_names[i] == literal_text_) {
                object.property = static_cast<std::int32_t>(i);
                object.emitted |= 1ULL << i;
                return;
            }
        }
    }

    [[nodiscard]] bool enum_exact() const {
        if (schema_ == nullptr || stack_.back().node == kNoSchemaNode) { return false; }
        for (const std::string& value : schema_->node(stack_.back().node).enum_values) {
            if (value == literal_text_) { return true; }
        }
        return false;
    }

    std::vector<FrameState> stack_;
    std::string_view literal_;
    std::size_t literal_pos_ = 0;
    bool key_                = false;
    bool number_complete_    = false;
    RootKind root_           = RootKind::AnyValue;
    // Doubles as the object-key buffer and the enum-literal buffer: a key and an enum member are
    // never open at the same time, since a key must close before its value opens.
    std::string literal_text_;
    std::shared_ptr<const CompiledSchema> schema_;
};

} // namespace ninfer::constraint
