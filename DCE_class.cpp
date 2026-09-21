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

bool DCE_class::apply_dead_code_elimination() {
    auto& equations = input_expr.equations;
    bool changed = false;

    std::vector keep(equations.size(), false);
    std::unordered_set<size_t> used_ids {};

    for (auto& val: input_expr.outvals) {
        if (val.is_literal()) continue;
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
            if (input.is_literal()) continue;
            used_ids.insert(input.get_var().get_id());
        }

        // Step 3: if the equation has children, recursively apply DCE:

        switch (equation.get_op()) {
            using enum primitive_op;
            case COND: {
                for (auto& expr: std::get<cond_params>(equation.get_params()).branches) {
                    changed |= expr.eliminate_dead_code();
                }
                break;
            }

            case SCAN: {
                changed |= std::get<scan_params>(equation.get_params()).jaxpr.eliminate_dead_code();
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
    changed |= (place_index != equations.size());
    equations.erase(equations.begin() + place_index, equations.end());

    return changed;
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
    const bool a_literal = a.is_literal();
    const bool b_literal = b.is_literal();
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
            if (val.is_literal()) {
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

bool CSE_class::apply_common_subexpression_elimination() {
    bool changed = false;
    std::unordered_map<size_t, size_t> find_map {};
    std::unordered_map<equation_key, std::span<const var_t>> cse_map {};

    auto find = [&](const size_t var_id) -> size_t {
        if (const auto it = find_map.find(var_id); it != find_map.end()) {
            return it->second;
        }
        return var_id;
    };

    auto path_compress = [&](std::span<value> values) -> void {
        for (auto& val: values) {
            if (val.is_var()) {
                size_t& var_id = val.get_var().get_id();
                size_t new_id = find(var_id);
                if (new_id != var_id) changed = true;
                var_id = new_id;
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
                    changed |= branch.eliminate_common_subexpressions();
                }
                break;
            case SCAN:
                changed |= std::get<scan_params>(eq.get_params()).jaxpr.eliminate_common_subexpressions();
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

    return changed;
}

TRS_class::TRS_class(expression& expr): input_expr(expr) {}

using value_variant = std::variant<size_t, literal_t>;
static value_variant to_value_variant(const value& value) {
    if (value.is_literal()) {
        return value.get_literal();
    } else {
        return value.get_var().get_id();
    }
}

/*

Like CSE, TRS will be done bottom up relative to the expression DAG. Unlike CSE, TRS is composed of multiple
entirely distinct cases, each with very different rewrite semantics. This means there is no unifying strucutre
for each rewrite.

In general, when an equation is visited, a term rewrite rule may be selected based on the corresponding primitive op.
At the end of each term rewrite, the semantics of the underlying jaxpr are entirely preserved. Unused variables are
NOT cleaned up, this is only ever done by the DCE pass.

Term rewrite rules are attempted based on a pre-defined order. If one term rewrite rule fails, the next one
is attempted.

*/

bool TRS_class::apply_term_rewrite(bool fast_math) {
    bool changed = false;
    std::unordered_map<size_t, size_t> id_equation_map {};
    // Unlike CSE, this maps var_id to var_id | literal
    std::unordered_map<size_t, value_variant> find_map {};
    std::vector<equation> new_equations {}; new_equations.reserve(input_expr.equations.size());
    std::vector<equation> equation_buffer {};

    size_t fresh_var_id = input_expr.var_id;
    auto fresh_var = [&](type_t type) -> var_t {
        return var_t{fresh_var_id++, std::move(type)};
    };

    auto emplace_equation_to_buffer = [&](
        std::vector<value> input, std::vector<var_t> output,
        primitive_op op, params_variant params=std::monostate{}) -> void {

        size_t new_equation_index = new_equations.size() + equation_buffer.size();
        for (var_t& outvar: output) {
            id_equation_map[outvar.get_id()] = new_equation_index;
        }

        equation_buffer.emplace_back(
            std::move(input), std::move(output), op, std::move(params)
        );
    };

    auto find = [&](value& val) -> value_variant {
        if (val.is_literal()) return val.get_literal();
        const var_t& val_var = val.get_var();
        if (const auto it = find_map.find(val_var.get_id()); it != find_map.end()) {
            return it->second;
        }
        return val.get_var().get_id();
    };

    auto bind = [&](var_t& var, const value& bound_val) -> void {
        changed = true;
        find_map[var.get_id()] = to_value_variant(bound_val);
    };

    // Broadcasts if necessary:
    auto bind_constant = [&](var_t& var, const double x) -> void {
        changed = true;
        if (var.get_shape().empty()) {
            find_map[var.get_id()] = literal_t{var.get_dtype(), x};
        } else {
            var_t broadcast_var = fresh_var(var.get_type());
            find_map[var.get_id()] = broadcast_var.get_id();
            value x_value {literal_t{var.get_dtype(), x}};

            emplace_equation_to_buffer(
                std::vector{std::move(x_value)},
                std::vector{std::move(broadcast_var)},
                primitive_op::BROADCAST_IN_DIM,
                broadcast_in_dim_params{var.get_shape(), {}}
            );

        }
    };

    auto path_compress = [&](std::span<value> values) -> void {
        for (value& val: values) {
            if (val.is_literal()) continue;
            auto find_result = find(val);
            var_t& val_var = val.get_var();
            if (std::holds_alternative<size_t>(find_result)) {
                // var_id case:
                val_var.set_id(*std::get_if<size_t>(&find_result));
            } else {
                val = value{*std::get_if<literal_t>(&find_result)};
            }
        }
    };

    // If %y = f %x, then trace(%y, predicate) := [%y = f %x] if equation satisfies predicate, otherwise nullptr
    // trace returns the equation index
    auto trace = [&]<typename F>(const size_t var_id, F&& predicate) -> std::optional<size_t>  {
        if (const auto it = id_equation_map.find(var_id); it != id_equation_map.end()) {
            auto& eq = new_equations[it->second];
            if (predicate(eq)) return it->second;
        }
        return std::nullopt;
    };


    auto trace_broadcast = [&](size_t var_id) ->
    std::optional<std::pair<value&, const broadcast_in_dim_params&>> {

        std::optional<size_t> eq_index = trace(var_id, [](equation& eq) -> bool {
            return eq.get_op() == primitive_op::BROADCAST_IN_DIM;
        });

        if (!eq_index) return std::nullopt;


        auto& eq = new_equations[*eq_index];
        const auto& params = std::get<broadcast_in_dim_params>(eq.get_params());
        return std::pair<value&, const broadcast_in_dim_params&> {
            eq.get_input(0), params};
    };

    // Special case of trace_broadcast:
    auto trace_literal_broadcast = [&](size_t var_id) ->
    std::optional<std::pair<literal_t, const broadcast_in_dim_params&>> {

        auto result = trace_broadcast(var_id);
        if (!result) return std::nullopt;
        if (result->first.is_var()) return std::nullopt;
        return std::pair<literal_t, const broadcast_in_dim_params&>(result->first.get_literal(), result->second);
    };


    /*

    If y = f(bc(x1), ..., bc(xN)), then y = bc(x1, ..., xN) provided that f is elementwise,
    and 'bc' is the same broadcast applied to each element.

    resolve_broadcast will be invoked on all elementwise ops if no other simplification is possible.
    resole_broadcast also assumes a single output.

    */

    auto resolve_broadcast = [&](equation& eq) -> bool {
        using namespace std::views;
        // First, check to make sure that the first argument is a broadcast:

        if (eq.get_input(0).is_literal()) return false;
        auto result = trace_broadcast(eq.get_input(0).get_var().get_id());
        if (!result) return false;

        broadcast_in_dim_params params = result->second;
        const std::vector<size_t>& pre_broadcast_shape = result->first.get_shape();

        std::vector pre_broadcast_invals {result->first};

        if (!std::ranges::all_of(eq.get_input() | drop(1), [&](const value& val) -> bool {
            if (val.is_literal()) return false;
            auto val_result = trace_broadcast(val.get_var().get_id());
            if (!val_result) return false;
            // If there exists a result, compare it against params:
            if (val_result->second.broadcast_dimensions != params.broadcast_dimensions) return false;
            if (val_result->first.get_shape() != pre_broadcast_shape) return false;
            pre_broadcast_invals.push_back(val_result->first);
            return true;
        })) {
            return false;
        }

        // y = bc(f(x1, ..., xN)):

        var_t& y = eq.get_output(0);

        var_t f_var = fresh_var(type_t{y.get_dtype(), pre_broadcast_shape});

        emplace_equation_to_buffer(
            std::move(pre_broadcast_invals),
            std::vector{f_var},
            eq.get_op(),
            eq.get_params()
        );

        var_t bc_var = fresh_var(y.get_type());
        bind(y, value{bc_var});

        emplace_equation_to_buffer(
            std::vector{value{std::move(f_var)}},
            std::vector{std::move(bc_var)},
            primitive_op::BROADCAST_IN_DIM,
            std::move(params)
        );

        return true;
    };

    auto fold_unary = [&](equation& eq, auto&& op) -> bool {
        const value& x = eq.get_input(0);
        if (!x.is_literal()) return false;
        bind_constant(eq.get_output(0), op(x.get_literal().get_value()));
        return true;
    };

    auto fold_binary = [&](equation& eq, auto&& op) -> bool {
        const value& x = eq.get_input(0);
        const value& y = eq.get_input(1);
        if (!x.is_literal() || !y.is_literal()) return false;
        bind_constant(eq.get_output(0), op(x.get_literal().get_value(), y.get_literal().get_value()));
        return true;
    };

    for (auto& old_eq: input_expr.equations) {
        for (auto& outvar: old_eq.get_output()) {
            id_equation_map[outvar.get_id()] = new_equations.size();
        }

        new_equations.push_back(old_eq);
        path_compress(new_equations.back().get_input());
        auto& eq = new_equations.back();

        switch (eq.get_op()) {
            using enum primitive_op;

            case ADD: {
                // %z = add %x %y

                // add(x, 0) -> x, add(0, x) -> x:

                auto rewrite_values = [&](const value& x, const value& y, var_t& z) -> bool {
                    if (x.is_literal()) {
                        if (x.get_literal().get_value() == 0) {
                            // If x is a zero literal, then bind z to y:
                            bind(z, y);
                            return true;
                        }
                        return false;
                    }

                    if (auto x_result = trace_literal_broadcast(x.get_var().get_id())) {
                        if (x_result->first.get_value() == 0) {
                            bind(z, y);
                            return true;
                        }
                    }
                    return false;
                };

                value& x = eq.get_input(0);
                value& y = eq.get_input(1);
                var_t& z = eq.get_output(0);

                if (fold_binary(eq, [](double a, double b) { return a + b; })) break;
                if (rewrite_values(x, y, z)) break;
                if (rewrite_values(y, x, z)) break;
                resolve_broadcast(eq);

                break;
            }

            case MUL: {
                // %z = mul %x %y

                // mul(x, 1) -> x, mul(1, x) -> x,
                // mul(x, 0) -> 0, mul(0, x) -> 0,
                // mul(x, -1) -> neg(x), mul(-1, x) -> neg(x):

                // Returns whether rewrite happened:
                auto rewrite_values = [&](const value& x, const value& y, var_t& z) -> bool {
                    if (x.is_literal()) {
                        if (x.get_literal().get_value() == 1) {
                            bind(z, y);
                            return true;
                        }

                        if (fast_math && x.get_literal().get_value() == 0) {
                            bind_constant(z, 0);
                            return true;
                        }

                        if (x.get_literal().get_value() == -1) {
                            // Have to emit a new instruction:
                            var_t neg_var = fresh_var(z.get_type());
                            bind(z, value{neg_var});

                            emplace_equation_to_buffer(
                                std::vector{y},
                                std::vector{std::move(neg_var)},
                                NEG
                            );

                            return true;
                        }
                        return false;
                    }

                    // x is a variable:
                    if (const auto x_result = trace_literal_broadcast(x.get_var().get_id())) {
                        if (x_result->first.get_value() == 1) {
                            bind(z, y);
                            return true;
                        }

                        if (fast_math && x_result->first.get_value() == 0) {
                            bind_constant(z, 0);
                            return true;
                        }

                        if (x_result->first.get_value() == -1) {
                            var_t neg_var = fresh_var(z.get_type());
                            bind(z, value{neg_var});

                            emplace_equation_to_buffer(
                                std::vector{y},
                                std::vector{std::move(neg_var)},
                                NEG
                            );
                            return true;
                        }
                    }

                    return false;
                };

                value& x = eq.get_input(0);
                value& y = eq.get_input(1);
                var_t& z = eq.get_output(0);

                if (fold_binary(eq, [](double a, double b) { return a * b; })) break;
                if (rewrite_values(x, y, z)) break;
                if (rewrite_values(y, x, z)) break;
                resolve_broadcast(eq);

                break;
            }

            case SUB : {
                // %z = sub %x %y

                // sub(x, 0) -> x,
                // sub(0, x) -> neg(x):

                value& x = eq.get_input(0);
                value& y = eq.get_input(1);
                var_t& z = eq.get_output(0);

                if (fold_binary(eq, [](double a, double b) { return a - b; })) break;

                if (y.is_literal()) {
                    if (y.get_literal().get_value() == 0) {
                        bind(z, x);
                        break;
                    }

                } else {
                    if (auto y_result = trace_literal_broadcast(y.get_var().get_id())) {
                        if (y_result->first.get_value() == 0) {
                            bind(z, x);
                            break;
                        }
                    }
                }

                if (x.is_literal()) {
                    if (x.get_literal().get_value() == 0) {
                        var_t neg_var = fresh_var(z.get_type());
                        bind(z, value{neg_var});
                        emplace_equation_to_buffer(
                            std::vector{y},
                            std::vector{std::move(neg_var)},
                            NEG
                        );
                        break;
                    }
                } else {
                    if (auto x_result = trace_literal_broadcast(x.get_var().get_id())) {
                        if (x_result->first.get_value() == 0) {
                            var_t neg_var = fresh_var(z.get_type());
                            bind(z, value{neg_var});
                            emplace_equation_to_buffer(
                                std::vector{y},
                                std::vector{std::move(neg_var)},
                                NEG
                            );
                            break;
                        }
                    }
                }

                resolve_broadcast(eq);
                break;
            }

            case DIV: {
                // %z = div %x %y

                // div(x, 1) -> x:

                value& x = eq.get_input(0);
                value& y = eq.get_input(1);
                var_t& z = eq.get_output(0);

                if (fold_binary(eq, [](double a, double b) { return a / b; })) break;

                if (y.is_literal()) {
                    if (y.get_literal().get_value() == 1) {
                        bind(z, x);
                        break;
                    }
                } else {
                    if (auto y_result = trace_literal_broadcast(y.get_var().get_id())) {
                        if (y_result->first.get_value() == 1) {
                            bind(z, x);
                            break;
                        }
                    }
                }

                resolve_broadcast(eq);

                break;
            }

            case NEG: {
                // %z = neg %x

                // neg(neg(x)) -> x:

                value& x = eq.get_input(0);
                var_t& z = eq.get_output(0);

                if (fold_unary(eq, [](double a) { return -a; })) break;

                if (x.is_var()) {
                    if (auto inner = trace(x.get_var().get_id(), [](equation& e) -> bool {
                        return e.get_op() == NEG;
                    })) {
                        bind(z, new_equations[*inner].get_input(0));
                        break;
                    }
                }

                resolve_broadcast(eq);
                break;
            }

            case INTEGER_POW: {
                // %z = integer_pow[y] %x

                // integer_pow(x, 1) -> x:

                value& x = eq.get_input(0);
                var_t& z = eq.get_output(0);

                if (x.is_literal()) {
                    double acc = 1;
                    const double base = x.get_literal().get_value();
                    for (size_t k = 0; k < std::get<integer_pow_params>(eq.get_params()).y; k++) acc *= base;
                    bind_constant(z, acc);
                    break;
                }

                if (std::get<integer_pow_params>(eq.get_params()).y == 1) {
                    bind(z, x);
                    break;
                }

                resolve_broadcast(eq);

                break;
            }

            case POW: {
                // %z = pow %x %y

                // pow(x, 1) -> x,
                // pow(x, n) -> integer_pow(x, n) if n is a non-negative integer literal:

                value& x = eq.get_input(0);
                value& y = eq.get_input(1);
                var_t& z = eq.get_output(0);

                if (fold_binary(eq, [](double a, double b) { return std::pow(a, b); })) break;

                std::optional<double> exponent;
                if (y.is_literal()) {
                    exponent = y.get_literal().get_value();
                } else if (auto y_result = trace_literal_broadcast(y.get_var().get_id())) {
                    exponent = y_result->first.get_value();
                }

                if (exponent) {
                    if (*exponent == 1) {
                        bind(z, x);
                        break;
                    }

                    if (*exponent >= 0) {
                        if (auto n = static_cast<size_t>(*exponent); static_cast<double>(n) == *exponent) {
                            var_t pow_var = fresh_var(z.get_type());
                            bind(z, value{pow_var});
                            emplace_equation_to_buffer(
                                std::vector{x},
                                std::vector{std::move(pow_var)},
                                INTEGER_POW,
                                integer_pow_params{n}
                            );
                            break;
                        }
                    }
                }

                resolve_broadcast(eq);

                break;
            }

            case LOG: {
                // %z = log %x

                // log(exp(x)) -> x (fast_math):

                value& x = eq.get_input(0);
                var_t& z = eq.get_output(0);

                if (fold_unary(eq, [](double a) { return std::log(a); })) break;

                if (fast_math && x.is_var()) {
                    if (auto inner = trace(x.get_var().get_id(), [](equation& e) -> bool {
                        return e.get_op() == EXP;
                    })) {
                        bind(z, new_equations[*inner].get_input(0));
                        break;
                    }
                }

                resolve_broadcast(eq);

                break;
            }

            case EXP: {
                // %z = exp %x

                // exp(log(x)) -> x (fast_math):

                value& x = eq.get_input(0);
                var_t& z = eq.get_output(0);

                if (fold_unary(eq, [](double a) { return std::exp(a); })) break;

                if (fast_math && x.is_var()) {
                    if (auto inner = trace(x.get_var().get_id(), [](equation& e) -> bool {
                        return e.get_op() == LOG;
                    })) {
                        bind(z, new_equations[*inner].get_input(0));
                        break;
                    }
                }

                resolve_broadcast(eq);

                break;
            }

            case SIN: {
                if (fold_unary(eq, [](double a) { return std::sin(a); })) break;
                resolve_broadcast(eq);
                break;
            }

            case COS: {
                if (fold_unary(eq, [](double a) { return std::cos(a); })) break;
                resolve_broadcast(eq);
                break;
            }

            case SQRT: {
                if (fold_unary(eq, [](double a) { return std::sqrt(a); })) break;
                resolve_broadcast(eq);
                break;
            }

            case RSQRT: {
                if (fold_unary(eq, [](double a) { return 1. / std::sqrt(a); })) break;
                resolve_broadcast(eq);
                break;
            }

            case TANH: {
                if (fold_unary(eq, [](double a) { return std::tanh(a); })) break;
                resolve_broadcast(eq);
                break;
            }

            case LOGISTIC: {
                if (fold_unary(eq, [](double a) { return 1. / (1. + std::exp(-a)); })) break;
                resolve_broadcast(eq);
                break;
            }

            case MAX: {
                if (fold_binary(eq, [](double a, double b) { return std::max(a, b); })) break;
                resolve_broadcast(eq);
                break;
            }

            case MIN: {
                if (fold_binary(eq, [](double a, double b) { return std::min(a, b); })) break;
                resolve_broadcast(eq);
                break;
            }

            case CONVERT_ELEMENT_TYPE: {
                // %y = convert_element_type[T] %x
                // convert_element_type(x, dtype(x)) -> x

                value& x = eq.get_input(0);
                var_t& y = eq.get_output(0);

                if (std::get<convert_element_type_params>(eq.get_params()).new_dtype
                    == x.get_dtype()) {
                    bind(y, x);
                    break;
                }

                if (fold_unary(eq, [](double a) { return a; })) break;
                resolve_broadcast(eq);
                break;
            }

            case LT: {
                if (fold_binary(eq, [](double a, double b) { return a < b; })) break;
                resolve_broadcast(eq);
                break;
            }

            case LE: {
                if (fold_binary(eq, [](double a, double b) { return a <= b; })) break;
                resolve_broadcast(eq);
                break;
            }

            case GT: {
                if (fold_binary(eq, [](double a, double b) { return a > b; })) break;
                resolve_broadcast(eq);
                break;
            }

            case GE: {
                if (fold_binary(eq, [](double a, double b) { return a >= b; })) break;
                resolve_broadcast(eq);
                break;
            }

            case EQ: {
                if (fold_binary(eq, [](double a, double b) { return double_eq(a, b); })) break;
                resolve_broadcast(eq);
                break;
            }

            case NE: {
                if (fold_binary(eq, [](double a, double b) { return !double_eq(a, b); })) break;
                resolve_broadcast(eq);
                break;
            }

            case SELECT: {
                value& pred = eq.get_input(0);
                var_t& z = eq.get_output(0);

                if (pred.is_literal()) {
                    size_t index = static_cast<size_t>(pred.get_literal().get_value());
                    bind(z, eq.get_input(1 + index));
                    break;
                }

                resolve_broadcast(eq);
                break;
            }

            case RESHAPE: {
                // %y = reshape[new_sizes] %x

                value& x = eq.get_input(0);
                var_t& y = eq.get_output(0);

                // reshape(x, shape(x)) -> x:

                if (x.get_shape() == std::get<reshape_params>(eq.get_params()).new_sizes) {
                    bind(y, x);
                    break;
                }

                // reshape(reshape(x, _), s) -> reshape(x, s)

                if (x.is_var()) {
                    if (auto result = trace(x.get_var().get_id(), [](equation& eq) -> bool {
                        return eq.get_op() == RESHAPE;
                    })) {
                        value& inner_x = new_equations[*result].get_input(0);
                        // Rewrite op:
                        x = inner_x;
                        changed = true;
                        break;
                    }
                }

                break;
            }

            case CONCATENATE: {
                // %y = concat %x1, ..., %xn

                // concatenate([x]) -> x (single operand)
                if (eq.get_input().size() == 1) {
                    bind(eq.get_output(0), eq.get_input(0)); // %y = %x
                    break;
                }

                break;
            }

            case TRANSPOSE: {
                // %y = transpose[permutation] %x
                value& x = eq.get_input(0);
                var_t& y = eq.get_output(0);
                std::vector<size_t>& permutation = std::get<transpose_params>(eq.get_params()).permutation;

                // transpose(x, identity) -> x

                if (is_identity_permutation(permutation)) {
                    bind(y, x);
                    break;
                }

                /*

                Composition of permutation:
                z = y[p], y = x[q], z = x[q][p]
                => y_i = x_q_i
                => z_i = y_p_i = x_q_p_i
                composition = q[p] => composition_i = q_p_i
                => z_i = x_composition_i => z = x[q[p]]

                */

                if (x.is_var()) {
                    if (auto result = trace(x.get_var().get_id(), [](equation& eq) -> bool {
                        return eq.get_op() == TRANSPOSE;
                    })) {
                        equation& inner_eq = new_equations[*result];
                        auto& inner_permutation = std::get<transpose_params>(inner_eq.get_params()).permutation;
                        std::vector<size_t> composed_permutation = permute(inner_permutation, permutation);

                        var_t permuted_var = fresh_var(y.get_type());
                        bind(y, value{permuted_var});

                        emplace_equation_to_buffer(
                            std::vector{inner_eq.get_input(0)},
                            std::vector{std::move(permuted_var)},
                            TRANSPOSE,
                            transpose_params{std::move(composed_permutation)}
                        );
                    }
                }

                break;
            }

            case REDUCE_SUM:
            case REDUCE_MIN:
            case REDUCE_MAX: {
                std::vector<size_t>& axes = std::invoke([&]() -> std::vector<size_t>& {
                    if (eq.get_op() == REDUCE_SUM)
                        return std::get<reduce_sum_params>(eq.get_params()).axes;
                    if (eq.get_op() == REDUCE_MIN)
                        return std::get<reduce_min_params>(eq.get_params()).axes;
                    return std::get<reduce_max_params>(eq.get_params()).axes;
                });

                if (axes.empty()) {
                    // reduce_op(x, {}) -> x (empty axis set)
                    bind(eq.get_output(0), eq.get_input(0));
                }
                break;
            }

            case BROADCAST_IN_DIM: {
                // %y = broadcast[shape, dims] %x

                value& x = eq.get_input(0);
                var_t& y = eq.get_output(0);
                auto& params = std::get<broadcast_in_dim_params>(eq.get_params());

                // broadcast_in_dim(x, shape(x), identity_dims) -> x

                if (x.get_shape() == params.shape && is_identity_permutation(params.broadcast_dimensions)) {
                    bind(y, x);
                    break;
                }

                /*

                Composition of broadcast:
                y = broadcast(x) => y_ij = x_i
                z = broadcast(y) => z_ijk = y_ij = x_i

                Therefore, all the NEW axes in y remain new axes in z when projecting from x to z.

                */

                if (x.is_var()) {
                    if (auto result = trace_broadcast(x.get_var().get_id())) {
                        value& inner_x = result->first;
                        const broadcast_in_dim_params& inner_params = result->second;
                        std::vector<size_t> new_broadcast_dimensions {};
                        new_broadcast_dimensions.reserve(inner_params.broadcast_dimensions.size());

                        for (size_t i: inner_params.broadcast_dimensions) {
                            size_t forwarded_i = params.broadcast_dimensions[i];
                            new_broadcast_dimensions.push_back(forwarded_i);
                        }

                        var_t rebroadcast_var = fresh_var(y.get_type());
                        bind(y, value{rebroadcast_var});

                        emplace_equation_to_buffer(
                            std::vector{inner_x},
                            std::vector{rebroadcast_var},
                            BROADCAST_IN_DIM,
                            broadcast_in_dim_params {
                                .shape = y.get_shape(),
                                .broadcast_dimensions = std::move(new_broadcast_dimensions)
                            }
                        );

                        break;
                    }
                }
                break;
            }

            case SLICE: {
                // %y = slice[start_indices, limit_indices, strides] %x

                // slice(x, full-range, stride 1) -> x:

                value& x = eq.get_input(0);
                var_t& y = eq.get_output(0);
                auto& params = std::get<slice_params>(eq.get_params());

                if (
                    std::ranges::all_of(params.start_indices, [](size_t x) {return x == 0;})
                    && params.limit_indices == x.get_shape()
                    && std::ranges::all_of(params.strides, [](size_t x) {return x == 1;})
                ) {
                    bind(y, x);
                    break;
                }

                break;
            }

            case PAD: {
                // %y = pad[padding_config] %x %k, padding_config = (low, high, interior) list

                value& x = eq.get_input(0);
                var_t& y = eq.get_output(0);
                auto& [padding_config] = std::get<pad_params>(eq.get_params());

                // pad(x, k, all-zero config) -> x

                if (std::ranges::all_of(padding_config, [](std::array<size_t, 3>& x) {
                    return x[0] == 0 && x[1] == 0 && x[2] == 0;
                })) {
                    bind(y, x);
                    break;
                }

                break;
            }

            default: break;
        }
        // Flush equation_buffer:
        for (auto& temp_eq: equation_buffer) {
            new_equations.push_back(std::move(temp_eq));
        }
        equation_buffer.clear();
    }

    input_expr.equations = std::move(new_equations);
    input_expr.var_id = fresh_var_id;
    path_compress(input_expr.outvals);

    return changed;
}

/*

The purpose of VDR_class is to convert a jaxpr with gaps in variable indices into one without. For instance:

%5 = add %0 %1
%7 = mul %5 %0

Will be normalized to:

%2 = add %0 %1
%3 = mul %2 %0

This is achieved by maintaining a simple variable map.

*/

VDN_class::VDN_class(expression& expr): input_expr{expr} {}

bool VDN_class::apply_variable_domain_normalisation() {
    bool changed = false;
    size_t var_id = 0;
    std::unordered_map<size_t, size_t> var_id_map {}; // Maps from old var_id space to new var_id space

    auto register_and_update = [&](var_t& var) -> void {
        if (var.get_id() != var_id) changed = true;
        var_id_map[var.get_id()] = var_id;
        var.get_id() = var_id++;
    };

    auto update = [&](value& val) -> void {
        if (val.is_var()) {
            size_t& id = val.get_var().get_id();
            id = var_id_map[id];
        }
    };

    for (auto& constvar: input_expr.constvars) register_and_update(constvar);
    for (auto& invar: input_expr.invars) register_and_update(invar);

    for (equation& eq: input_expr.equations) {
        for (value& inval: eq.get_input()) update(inval);
        for (var_t& outvar: eq.get_output()) register_and_update(outvar);

        switch (eq.get_op()) {
            using enum primitive_op;
            case COND:
                for (auto& branch: std::get<cond_params>(eq.get_params()).branches) {
                    changed |= VDN_class{branch}.apply_variable_domain_normalisation();
                }
                break;
            case SCAN:
                changed |= VDN_class{std::get<scan_params>(eq.get_params()).jaxpr}
                    .apply_variable_domain_normalisation();
                break;
            default: break;
        }
    }

    for (auto& outval: input_expr.outvals) update(outval);

    input_expr.var_id = var_id;

    return changed;
}
