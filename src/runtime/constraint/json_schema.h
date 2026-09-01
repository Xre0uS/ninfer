#pragma once

// A JSON Schema compiled down to what a token-level constraint can actually enforce.
//
// This is deliberately a SUBSET. Every keyword that reaches the compiler is either enforced or
// rejected by name -- there is no third category where a schema is accepted and quietly ignored,
// because that returns 200 for output that does not conform, which is the exact flaw in the
// upstream instruction-injection approach this replaces.
//
// Supported: type (object/array/string/number/integer/boolean/null), properties, required,
// additionalProperties, items, minItems, maxItems, enum, const, and arbitrary nesting of those.
//
// Rejected by name: $ref, anyOf, oneOf, allOf, not, patternProperties, pattern, dependencies,
// if/then/else, and numeric/string bounds that a prefix automaton cannot decide incrementally
// (minimum, maximum, minLength, maxLength, format).

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::constraint {

inline constexpr std::uint32_t kNoSchemaNode = 0xFFFFFFFFU;

// required_mask and the per-object emitted mask are both 64-bit, so an object is capped at 64
// declared properties. Beyond that the schema is rejected rather than partially enforced.
inline constexpr std::size_t kMaxSchemaProperties = 64;

struct SchemaNode {
    enum class Kind : std::uint8_t {
        Any,     // no type constraint; any JSON value
        Object,
        Array,
        String,
        Number,
        Integer, // a Number that may not take '.', 'e' or 'E'
        Boolean,
        Null,
        Enum,    // one of a fixed set of literals, matched as raw JSON text
    };

    Kind kind = Kind::Any;

    // Object. property_nodes is parallel to property_names; bit i of required_mask marks
    // property_names[i] as required.
    std::vector<std::string> property_names;
    std::vector<std::uint32_t> property_nodes;
    std::uint64_t required_mask = 0;
    bool additional_properties  = true;

    // Array.
    std::uint32_t items     = kNoSchemaNode;
    std::uint32_t min_items = 0;
    std::uint32_t max_items = 0xFFFFFFFFU;

    // Enum. Each entry is the exact JSON encoding the document must contain, quotes included for
    // strings, so matching is a plain byte comparison and non-string enums need no special case.
    std::vector<std::string> enum_values;
};

class CompiledSchema {
public:
    CompiledSchema();

    // Process-unique, never reused. The mask cache is keyed on constraint state, and a node index
    // means nothing without knowing WHICH schema it indexes -- two unrelated schemas both have a
    // root at node 0, so without this their root states produce byte-identical cache keys and one
    // request is served the other's masks. A pointer would not do: a freed schema's address can
    // be reused by a later one, which would resurrect exactly the same collision.
    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

    [[nodiscard]] const SchemaNode& node(std::uint32_t index) const { return nodes_[index]; }

    [[nodiscard]] static constexpr std::uint32_t root() noexcept { return 0; }

    [[nodiscard]] bool empty() const noexcept { return nodes_.empty(); }

    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }

    std::uint32_t add(SchemaNode value) {
        nodes_.push_back(std::move(value));
        return static_cast<std::uint32_t>(nodes_.size() - 1);
    }

    [[nodiscard]] SchemaNode& mutable_node(std::uint32_t index) { return nodes_[index]; }

private:
    std::vector<SchemaNode> nodes_;
    std::uint64_t id_ = 0;
};

// Compiles a JSON Schema document. Takes the raw text rather than a parsed object so this header
// stays free of a JSON library -- it is included by the engine's hot request path.
//
// Throws std::invalid_argument naming the offending keyword when the schema uses something that
// cannot be enforced incrementally.
[[nodiscard]] std::shared_ptr<const CompiledSchema> compile_json_schema(std::string_view document);

} // namespace ninfer::constraint
