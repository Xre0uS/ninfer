#include "runtime/constraint/json_schema.h"

#include "nlohmann/json.hpp"

#include <atomic>
#include <stdexcept>
#include <string>

namespace ninfer::constraint {

CompiledSchema::CompiledSchema() {
    static std::atomic<std::uint64_t> next_id{1};
    id_ = next_id.fetch_add(1, std::memory_order_relaxed);
}

namespace {

using Json = nlohmann::json;

[[noreturn]] void unsupported(const std::string& keyword, const std::string& where) {
    throw std::invalid_argument("json_schema: " + keyword + " is not supported (at " + where +
                                "). This engine enforces schemas with a token mask, and a keyword "
                                "it cannot decide from a prefix would be accepted and silently "
                                "ignored.");
}

[[noreturn]] void invalid(const std::string& detail, const std::string& where) {
    throw std::invalid_argument("json_schema: " + detail + " (at " + where + ")");
}

// Keywords that change what a document may contain and that this compiler cannot honour. Listed
// explicitly so a schema using one is refused by name rather than quietly under-enforced.
const char* const kRejected[] = {
    "$ref",     "$dynamicRef", "anyOf",   "oneOf",     "allOf",     "not",
    "patternProperties",       "pattern", "dependencies",           "dependentSchemas",
    "if",       "then",        "else",    "minimum",   "maximum",   "exclusiveMinimum",
    "exclusiveMaximum",        "multipleOf",           "minLength", "maxLength",
    "format",   "contains",    "uniqueItems",          "propertyNames",
};

SchemaNode::Kind kind_from_type(const std::string& type, const std::string& where) {
    if (type == "object") { return SchemaNode::Kind::Object; }
    if (type == "array") { return SchemaNode::Kind::Array; }
    if (type == "string") { return SchemaNode::Kind::String; }
    if (type == "number") { return SchemaNode::Kind::Number; }
    if (type == "integer") { return SchemaNode::Kind::Integer; }
    if (type == "boolean") { return SchemaNode::Kind::Boolean; }
    if (type == "null") { return SchemaNode::Kind::Null; }
    invalid("unknown type '" + type + "'", where);
}

std::uint32_t compile_node(const Json& schema, CompiledSchema& out, const std::string& where,
                           int depth) {
    // A prefix automaton with a frame stack has no trouble with depth, but an unbounded schema is
    // a denial-of-service shape rather than a real request.
    if (depth > 32) { invalid("schema nests deeper than 32 levels", where); }
    if (!schema.is_object()) { invalid("schema must be an object", where); }

    for (const char* keyword : kRejected) {
        if (schema.contains(keyword)) { unsupported(keyword, where); }
    }

    SchemaNode node;

    // enum and const both pin the value to a fixed set of literals, so they compile the same way.
    if (schema.contains("enum") || schema.contains("const")) {
        const Json values = schema.contains("enum")
                                ? schema.at("enum")
                                : Json::array({schema.at("const")});
        if (!values.is_array() || values.empty()) {
            invalid("enum must be a non-empty array", where);
        }
        node.kind = SchemaNode::Kind::Enum;
        for (const Json& value : values) {
            if (value.is_object() || value.is_array()) {
                unsupported("enum with object or array members", where);
            }
            // dump() is the exact encoding the document must contain, quotes included.
            node.enum_values.push_back(value.dump());
        }
        return out.add(std::move(node));
    }

    if (schema.contains("type")) {
        const Json& type = schema.at("type");
        if (type.is_array()) { unsupported("type as a union array", where); }
        if (!type.is_string()) { invalid("type must be a string", where); }
        node.kind = kind_from_type(type.get<std::string>(), where);
    } else if (schema.contains("properties") || schema.contains("required")) {
        node.kind = SchemaNode::Kind::Object; // implied, as most hand-written schemas assume
    } else if (schema.contains("items")) {
        node.kind = SchemaNode::Kind::Array;
    }

    if (node.kind == SchemaNode::Kind::Object) {
        // additionalProperties defaults to TRUE in JSON Schema, but a structured-output caller
        // almost always means "these keys and no others". Follow the spec rather than guessing,
        // and let the caller say so.
        if (schema.contains("additionalProperties")) {
            const Json& additional = schema.at("additionalProperties");
            if (additional.is_object()) { unsupported("additionalProperties as a schema", where); }
            if (!additional.is_boolean()) {
                invalid("additionalProperties must be a boolean", where);
            }
            node.additional_properties = additional.get<bool>();
        }
        if (schema.contains("properties")) {
            const Json& properties = schema.at("properties");
            if (!properties.is_object()) { invalid("properties must be an object", where); }
            if (properties.size() > kMaxSchemaProperties) {
                invalid("an object may declare at most " + std::to_string(kMaxSchemaProperties) +
                            " properties",
                        where);
            }
            for (const auto& [name, subschema] : properties.items()) {
                if (name.empty()) { invalid("property names must not be empty", where); }
                node.property_names.push_back(name);
                node.property_nodes.push_back(
                    compile_node(subschema, out, where + "/properties/" + name, depth + 1));
            }
        }
        if (schema.contains("required")) {
            const Json& required = schema.at("required");
            if (!required.is_array()) { invalid("required must be an array", where); }
            for (const Json& entry : required) {
                if (!entry.is_string()) { invalid("required entries must be strings", where); }
                const std::string name = entry.get<std::string>();
                bool found             = false;
                for (std::size_t i = 0; i < node.property_names.size(); ++i) {
                    if (node.property_names[i] == name) {
                        node.required_mask |= 1ULL << i;
                        found = true;
                        break;
                    }
                }
                // A required property with no schema is only enforceable if the object may
                // contain it at all.
                if (!found) {
                    if (!node.additional_properties) {
                        invalid("required property '" + name +
                                    "' is not declared and additionalProperties is false",
                                where);
                    }
                    if (node.property_names.size() >= kMaxSchemaProperties) {
                        invalid("too many properties once required entries are counted", where);
                    }
                    node.property_names.push_back(name);
                    node.property_nodes.push_back(kNoSchemaNode);
                    node.required_mask |= 1ULL << (node.property_names.size() - 1);
                }
            }
        }
    } else if (node.kind == SchemaNode::Kind::Array) {
        if (schema.contains("items")) {
            const Json& items = schema.at("items");
            if (items.is_array()) { unsupported("items as a tuple array", where); }
            node.items = compile_node(items, out, where + "/items", depth + 1);
        }
        if (schema.contains("minItems")) {
            const Json& value = schema.at("minItems");
            if (!value.is_number_unsigned()) { invalid("minItems must be a non-negative integer", where); }
            node.min_items = value.get<std::uint32_t>();
        }
        if (schema.contains("maxItems")) {
            const Json& value = schema.at("maxItems");
            if (!value.is_number_unsigned()) { invalid("maxItems must be a non-negative integer", where); }
            node.max_items = value.get<std::uint32_t>();
        }
        if (node.min_items > node.max_items) { invalid("minItems exceeds maxItems", where); }
    }

    return out.add(std::move(node));
}

} // namespace

std::shared_ptr<const CompiledSchema> compile_json_schema(std::string_view document) {
    Json parsed;
    try {
        parsed = Json::parse(document);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("json_schema: schema is not valid JSON: ") +
                                    error.what());
    }
    auto compiled = std::make_shared<CompiledSchema>();
    // compile_node appends children before their parent, so the root is not index 0 yet.
    const std::uint32_t root = compile_node(parsed, *compiled, "#", 0);
    if (root != CompiledSchema::root()) {
        // Reorder so root() is a compile-time constant for the constraint. Cheap: schemas are
        // tiny, and doing it here keeps an index indirection out of the per-token path.
        auto ordered = std::make_shared<CompiledSchema>();
        std::vector<std::uint32_t> remap(compiled->size(), kNoSchemaNode);
        // Depth-first from the root so the root lands at index 0.
        std::vector<std::uint32_t> stack{root};
        while (!stack.empty()) {
            const std::uint32_t index = stack.back();
            stack.pop_back();
            if (index == kNoSchemaNode || remap[index] != kNoSchemaNode) { continue; }
            remap[index] = ordered->add(compiled->node(index));
            const SchemaNode& node = compiled->node(index);
            for (const std::uint32_t child : node.property_nodes) {
                if (child != kNoSchemaNode) { stack.push_back(child); }
            }
            if (node.items != kNoSchemaNode) { stack.push_back(node.items); }
        }
        for (std::size_t i = 0; i < ordered->size(); ++i) {
            SchemaNode& node = ordered->mutable_node(static_cast<std::uint32_t>(i));
            for (std::uint32_t& child : node.property_nodes) {
                if (child != kNoSchemaNode) { child = remap[child]; }
            }
            if (node.items != kNoSchemaNode) { node.items = remap[node.items]; }
        }
        return ordered;
    }
    return compiled;
}

} // namespace ninfer::constraint
