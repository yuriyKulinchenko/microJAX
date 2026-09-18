//
// Created by Yuriy Kulinchenko on 01/07/2026.
//

#include "DCE_class.h"
#include "helper.h"

#include <unordered_set>

using namespace jax;

/*

Architecture of DCE pass:

The purpose of this pass is dead code elimination, which will remove
all unused equations in a jaxpr. This is an in-place transformation,
so it will overwrite the passed jaxpr.

DCE works by traversing backwards from the outputs of a given jaxpr,
and noting which equation results are reachable. Equation results
which are unreachable can be eliminated, which is done in the final pass

As the graph is traversed backwards, a 'keep' mapping can be maintained,
which tracks which equations need to be kept.

*/

DCE_class::DCE_class(expression& expr): input_expr(expr) {}

void DCE_class::apply_dead_code_elimination() {
    auto& equations = input_expr.equations;

    std::vector keep(equations.size(), false);
    std::unordered_set<size_t> used_ids {};

    for (auto& val: input_expr.outvals) {
        if (val.is<literal_t>()) continue;
        used_ids.insert(val.get_var().get_id());
    }


    for (size_t i = equations.size(); i--> 0;) {
        // Step 1: determine whether the equation should be kept,
        // based on whether it contains an output in used_ids:

        auto& equation = equations[i];
        bool keep_equation = false;

        for (auto& output: equation.get_output()) {
            if (used_ids.contains(output.get_id())) {
                keep_equation = true;
                keep[i] = true;
                break;
            }
        }

        // Step 2: if the equation should be kept, place all inputs
        // into used_ids:

        if (!keep_equation) continue;
        for (auto& input: equation.get_input()) {
            if (input.is<literal_t>()) continue;
            used_ids.insert(input.get_var().get_id());
        }

        // Step 3: if the equation has children, recursively apply DCE:

        switch (equation.get_op()) {
            using enum primitive_op;
            case COND: {
                for (auto& expr: std::get<cond_params>(equation.get_params()).branches) {
                    expr.eliminate_dead_code();
                }
                break;
            }

            case SCAN: {
                std::get<scan_params>(equation.get_params()).jaxpr.eliminate_dead_code();
                break;
            }
            default:
        }
    }

    // Finally, iterate through keep vector, and remove dead equations:

    size_t place_index = 0;
    for (size_t i = 0; i < equations.size(); i++) {
        if (!keep[i]) continue;
        if (place_index != i) equations[place_index] = std::move(equations[i]);
        place_index++;
    }

    // Shrink the vector appropriately:
    equations.erase(equations.begin() + place_index, equations.end());
}

// Efficiently represents a single equation key when performing CSE:

/*

Plan for common subexpression elimination:

CSE is performed bottom up, which is equivalent to a topological traversal order over the expression DAG.
Each equation in the jaxpr has a cannonical form, obtained by calling 'find'. For instance, the cannonical
form of %x is find(%x). If %x is unique, then find(%x) = %x: otherwise, find(%x) is the earliest occurence
of a variable equal to %x.

An equation of the form [%y = op %x1 ... %xn] is identified as a common subexpression based on whether
[op find(%x1) ... find(%xn)] has already been seen. The existence of [op find(%x1) ... find(%xn)] is determined
through a hash map lookup.

The hashmap lookup has to be quite detailed: If 'op' is commutative, the hash should reflect this. This culminates
in the 'equation_key' class, which provides hashing for all supported operations, based on these requirements.


*/

static bool is_commutative_op(const primitive_op op) {
    using enum primitive_op;
    return op == ADD || op == MUL || op == MAX || op == MIN || op == EQ || op == NE;
}

static bool value_less(const value& a, const value& b) {
    const bool a_literal = a.is<literal_t>();
    const bool b_literal = b.is<literal_t>();
    if (a_literal != b_literal) {
        return a_literal; // literals order before variables
    }
    if (a_literal) {
        const literal_t& la = a.get_literal();
        const literal_t& lb = b.get_literal();
        if (la.get_dtype() != lb.get_dtype()) {
            return static_cast<size_t>(la.get_dtype()) < static_cast<size_t>(lb.get_dtype());
        }
        return la.get_value() < lb.get_value();
    }
    return a.get_var().get_id() < b.get_var().get_id();
}

struct equation_key {
    equation_key(primitive_op op, const params_variant& params, std::span<value> operands):
    op(op), params(params), operands(operands) {

        if (is_commutative_op(op)) {
            std::ranges::sort(this->operands, value_less);
        }
    }

    bool operator==(const equation_key& other) const {
        return op == other.op && params == other.params && std::ranges::equal(operands, other.operands);
    }

    primitive_op op;
    const params_variant& params;
    std::span<value> operands;
};

namespace std {

    template<>
    struct std::hash<literal_t> {
        size_t operator()(const literal_t& literal) const noexcept {
            return std::hash<size_t>{}(static_cast<size_t>(literal.get_dtype()))
                ^ std::hash<double>{}(literal.get_value());
        }
    };

    template<>
    struct std::hash<value> {
        size_t operator()(const value& val) const noexcept {
            if (val.is<literal_t>()) {
                return std::hash<literal_t>{}(val.get_literal());
            }
            return std::hash<size_t>{}(val.get_var().get_id());
        }
    };

    template<>
    struct std::hash<equation_key> {
        size_t operator()(const equation_key& key) const noexcept {
            return ::multihash(key.op, key.operands, key.params);
        }
    };
}

CSE_class::CSE_class(expression& expr): input_expr(expr) {}

void CSE_class::apply_common_subexpression_elimination() {
    std::unordered_map<size_t, size_t> find_map {};
    std::unordered_map<equation_key, std::span<const var_t>> cse_map {};

    auto find = [&](const size_t var_id) -> size_t {
        auto it = find_map.find(var_id);
        if (it != find_map.end()) {
            return it->second;
        }
        return var_id;
    };

    auto path_compress = [&](std::span<value> values) -> void {
        for (auto& val: values) {
            if (val.is<var_t>()) {
                size_t& var_id = val.get_var().get_id();
                var_id = find(var_id);
            }
        }
    };

    for (auto& eq: input_expr.equations) {
        // Recursively hash, and then call 'find' on all input variables.
        // Goal is to have std::vector of operand_hashes:

        path_compress(eq.get_input());

        switch (eq.get_op()) {
            using enum primitive_op;
            case COND:
                for (auto& branch: std::get<cond_params>(eq.get_params()).branches) {
                    branch.eliminate_common_subexpressions();
                }
                break;
            case SCAN:
                std::get<scan_params>(eq.get_params()).jaxpr.eliminate_common_subexpressions();
                break;
            default: break;
        }

        equation_key key {eq.get_op(), eq.get_params(), eq.get_input()};
        if (auto it = cse_map.find(key); it == cse_map.end()) {
            // Subexpression is unique - register equation output in CSE map:
            cse_map[key] = eq.get_output();
        } else {
            // Subexpression already exists - map the outputs back:
            for (const auto& [original, replacement]: std::views::zip(eq.get_output(), it->second)) {
                find_map[original.get_id()] = replacement.get_id();
            }
        }
    }

    path_compress(input_expr.outvals);
}