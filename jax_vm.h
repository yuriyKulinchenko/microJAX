#ifndef MICROJAX_JAX_VM_H
#define MICROJAX_JAX_VM_H

#include "jax_types.h"

/*

The JAX VM is a virtual machine that can execute uncompiled JAX expressions.
The VM is inherently slower than JIT compilation, however it will act as
a ground truth that the JIT results can be compared against in testing.

*/

using namespace jax;


class jax_vm {
public:
    jax_vm(const expression& jaxpr):
    jaxpr(jaxpr),
    values(std::vector<std::unique_ptr<array_t>>(jaxpr.var_id)) {}

    std::vector<array_t> run(const std::vector<array_t>& input) {

        // TODO: Verify that size(input) == size(invars)

        reset();

        for (const auto& [invar, input_array]: std::views::zip(jaxpr.invars, input)) {
            values[invar.get_id()] = std::make_unique<array_t>(input_array);
        }

        for (const auto& [constvar, const_array]: std::views::zip(jaxpr.constvars, jaxpr.consts)) {
            values[constvar.get_id()] = std::make_unique<array_t>(const_array);
        }

        for (auto& eq: jaxpr.equations) {
            populate_literal_buffer(eq);
            switch (eq.get_op()) {
                using enum primitive_op;

                case SIN: execute_unary_op<(&array_t::sin)>(eq); break;
                case COS: execute_unary_op<(&array_t::cos)>(eq); break;
                case EXP: execute_unary_op<(&array_t::exp)>(eq); break;
                case LOG: execute_unary_op<(&array_t::log)>(eq); break;
                case NEG: execute_unary_op<(&array_t::operator-)>(eq); break;

                case ADD: execute_binary_op<(&array_t::operator+)>(eq); break;
                case SUB: execute_binary_op<(&array_t::operator-)>(eq); break;
                case MUL: execute_binary_op<(&array_t::operator*)>(eq); break;
                case DIV: execute_binary_op<(&array_t::operator/)>(eq); break;
                case LT: execute_binary_comparison_op<(&array_t::operator<)>(eq); break;
                case LE: execute_binary_comparison_op<(&array_t::operator<=)>(eq); break;
                case GT: execute_binary_comparison_op<(&array_t::operator>)>(eq); break;
                case GE: execute_binary_comparison_op<(&array_t::operator>=)>(eq); break;
                case EQ: execute_binary_comparison_op<(&array_t::elementwise_equal)>(eq); break;
                case NE: execute_binary_comparison_op<(&array_t::elementwise_not_equal)>(eq); break;

                case DOT_GENERAL: {
                    check_fixed_arity(eq, 2);
                    auto& [l_c, r_c, l_b, r_b] = std::get<dot_general_params>(eq.get_params());
                    emplace_variable(eq.get_output(0),
                        get_input(eq, 0).dot_general(get_input(eq, 1), l_c, r_c, l_b, r_b));
                    break;
                }

                case REDUCE_SUM: {
                    check_fixed_arity(eq, 1);
                    auto& axes = std::get<reduce_sum_params>(eq.get_params()).axes;
                    emplace_variable(eq.get_output(0), get_input(eq, 0).reduce_sum(axes));
                    break;
                }

                case TRANSPOSE: {
                    check_fixed_arity(eq, 1);
                    auto& permutation = std::get<transpose_params>(eq.get_params()).permutation;
                    emplace_variable(eq.get_output(0), get_input(eq, 0).transpose(permutation));
                    break;
                }

                case BROADCAST_IN_DIM: {
                    check_fixed_arity(eq, 1);
                    auto& [shape, broadcast_dimensions] = std::get<broadcast_in_dim_params>(eq.get_params());
                    emplace_variable(eq.get_output(0),
                        get_input(eq, 0).broadcast_in_dim(shape, broadcast_dimensions));
                    break;
                }

                case CONVERT_ELEMENT_TYPE: {
                    check_fixed_arity(eq, 1);
                    dtype_t new_dtype = std::get<convert_element_type_params>(eq.get_params()).new_dtype;
                    emplace_variable(eq.get_output(0), get_input(eq, 0).convert_element_type(new_dtype));
                    break;
                }

                case COND: {
                    // Switch on the first index:
                    const array_t& array = get_input(eq, 0);
                    if (!array.get_type().get_shape().empty()) {
                        throw std::logic_error(
                            "Error: expect first argument of 'cond' to be a scalar");
                    }

                    if (!is_integral(array.get_type().get_dtype())) {
                        throw std::logic_error(
                            "Error: expect first argument of 'cond' to have an integral dtype");
                    }

                    size_t index = static_cast<size_t>(array.get_value()[0]);
                    auto& branches = std::get<cond_params>(eq.get_params()).branches;

                    if (index >= branches.size()) {
                        throw formatted_error(
                            "Error: branch index in 'cond' is {}, but there are only {} branches",
                            index, branches.size());
                    }

                    jax_vm vm(branches[index]);
                    // Skip input 0 (the branch index): the branch's invars are the values only.
                    auto outputs = vm.run(std::views::iota(1ul, eq.get_input().size())
                        | std::views::transform([&](size_t i) -> array_t {
                        return get_input(eq, i);
                    }) | std::ranges::to<std::vector<array_t>>());


                    for (const auto& [output_var, output_array]: std::views::zip(eq.get_output(), outputs)) {
                        emplace_variable(output_var, std::move(output_array));
                    }
                    break;
                }

                case SCAN: {
                    /*

                     (c, Y) = scan(f)(k, c, X)
                     (c, y) = scan(f)(k, c, x) <- takes slices

                    */

                    const auto& params = std::get<scan_params>(eq.get_params());
                    const size_t input_count = eq.get_input().size();
                    const size_t output_count = eq.get_output().size();

                    const size_t num_xs = input_count - params.num_consts - params.num_carry;
                    const size_t num_ys = output_count - params.num_carry;

                    check_fixed_arity(eq, input_count, params.num_carry + num_xs);

                    // Make sure 'f' accepts the right input types:

                    // Iterate through (k, c)
                    for (const auto& [var, val]: std::views::zip(params.jaxpr.invars, eq.get_input())
                        | std::views::take(params.num_consts + params.num_carry)) {
                        if (var.get_type() != val.get_type()) {
                            throw formatted_error("Error: type mismatch for input parameters in SCAN");
                        }
                    }

                    // Iterate through X
                    for (const auto& [var, val]: std::views::zip(params.jaxpr.invars, eq.get_input())
                        | std::views::drop(params.num_consts + params.num_carry)) {
                        // Have to compare slice type:
                        if (var.get_dtype() != val.get_dtype()) {
                            throw formatted_error("Error: dtype mismatch for input paramaters in SCAN");
                        }

                        if (!std::ranges::equal(var.get_shape(), val.get_shape() | std::views::drop(1))) {
                            throw formatted_error("Error: shape mismatch for input parameters in SCAN");
                        }
                    }


                    // Iterate through c
                    for (const auto& [val, var]: std::views::zip(params.jaxpr.outvals, eq.get_output())
                        | std::views::take(params.num_carry)) {
                        if (var.get_type() != val.get_type()) {
                            throw formatted_error("Error: type mismatch for output parameters in SCAN");
                        }
                    }

                    // Iterate through y
                    for (const auto& [val, var]: std::views::zip(params.jaxpr.outvals, eq.get_output())
                        | std::views::drop(params.num_carry)) {
                        // Have to compare slice type:
                        if (var.get_dtype() != val.get_dtype()) {
                            throw formatted_error("Error: dtype mismatch for output parameters in SCAN");
                        }

                        if (!std::ranges::equal(val.get_shape(), var.get_shape() | std::views::drop(1))) {
                            throw formatted_error("Error: shape mismatch for output parameters in SCAN");
                        }
                    }

                    std::vector<array_t> Y {};
                    Y.reserve(num_ys);

                    for (const auto& var: eq.get_output() | std::views::drop(params.num_carry)) {
                        Y.push_back(array_t::build_fill(var.get_type(), 0.));
                    }

                    auto get_const = [&](const size_t const_index) -> array_t {
                        return get_input(eq, const_index);
                    };

                    auto get_carry = [&](const size_t carry_index) -> array_t {
                        return get_input(eq, params.num_consts + carry_index);
                    };

                    auto get_xs_slice = [&](const size_t x_index, const size_t slice_index) -> array_t {
                        return get_input(eq, params.num_consts + params.num_carry + x_index).slice({slice_index});
                    };

                    auto update_carry = [&](std::vector<array_t>& inputs, const std::span<array_t> carries) -> void {
                        for (size_t i = 0; i < carries.size(); i++) {
                            inputs[params.num_consts + i] = carries[i];
                        }
                    };

                    // Simultaneous update of multiple ys, across multiple slices:
                    auto update_ys = [&](const size_t slice_index, std::span<array_t> ys_slices) -> void {
                        for (size_t i = 0; i < ys_slices.size(); i++) {
                            Y[i].add_slice({slice_index}, ys_slices[i]);
                        }
                    };

                    auto update_xs_slice = [&](std::vector<array_t>& inputs, const size_t slice_index) -> void {
                        for (size_t i = 0; i < num_xs; i++) {
                            inputs[params.num_consts + params.num_carry + i] = get_xs_slice(i, slice_index);
                        }
                    };

                    // Maps a step number to the slice it touches:
                    auto slice_at = [&](size_t s) {
                        return params.reverse ? params.length - 1 - s : s;
                    };

                    std::vector<array_t> inputs {}; // The constants persist.
                    inputs.reserve(input_count);

                    for (size_t i = 0; i < params.num_consts; i++) inputs.push_back(get_const(i));
                    for (size_t i = 0; i < params.num_carry; i++) inputs.push_back(get_carry(i));
                    for (size_t i = 0; i < num_xs; i++) inputs.push_back(get_xs_slice(i, slice_at(0)));

                    jax_vm vm(params.jaxpr);

                    for (size_t s = 0; s < params.length; s++) {
                        std::vector<array_t> outputs = vm.run(inputs);
                        update_carry(inputs, std::span{outputs.begin(), outputs.begin() + params.num_carry});
                        update_ys(slice_at(s), std::span{outputs.begin() + params.num_carry, outputs.end()});
                        if (s + 1 < params.length) update_xs_slice(inputs, slice_at(s + 1));
                    }

                    // Gather carries first:
                    for (const auto& [var, array]:
                    std::views::zip(eq.get_output() | std::views::take(params.num_carry),
                            inputs | std::views::drop(params.num_consts) | std::views::take(params.num_carry))) {
                        emplace_variable(var, array);
                    }

                    // Gather ys next:
                    for (const auto& [var, array]:
                        std::views::zip(eq.get_output() | std::views::drop(params.num_carry), Y)) {
                        emplace_variable(var, array);
                    }

                    break;
                }

                default: {
                    throw std::logic_error("Error: not implemented");
                }
            }
        }

        return jaxpr.outvals | std::views::transform([this](const value& val) -> array_t {
            if (val.is<literal_t>()) {
                return literal_to_array(val.get_literal());
            }
            return fetch_value(val.get_var());
        }) | std::ranges::to<std::vector<array_t>>();
    }

    void reset() {
        for (auto& v: values) v.reset();
    }

private:

    const array_t& fetch_value(const var_t& var) const {
        if (values[var.get_id()] == nullptr) {
            throw formatted_error("Error: variable %{} does not exist", var.get_id());
        }
        return *values[var.get_id()];
    }

    void emplace_variable(const var_t& var, array_t array) {
        if (values[var.get_id()] != nullptr) {
            throw formatted_error(
                "Error: variable %{} already exists, it cannot be assigned to again", var.get_id());
        }
        values[var.get_id()] = std::make_unique<array_t>(std::move(array));
    }

    const array_t& get_input(const equation& eq, size_t i) {
        // Assumes a populated literal buffer
        if (eq.get_input(i).is<literal_t>()) {
            return *literal_buffer[i];
        }

        return fetch_value(eq.get_input(i).get_var());
    }

    static array_t literal_to_array(const literal_t& literal) {
        return array_t{type_t{literal.get_dtype()}, {literal.get_value()}};
    }

    void populate_literal_buffer(const equation& eq) {
        literal_buffer.assign(eq.get_input().size(), std::nullopt);
        for (size_t i = 0; i < eq.get_input().size(); i++) {
            if (eq.get_input(i).is<literal_t>()) {
                auto& literal = eq.get_input(i).get_literal();
                literal_buffer[i] = literal_to_array(literal);
            }
        }
    }

    static void check_fixed_arity(const equation& eq, size_t input_arity, size_t output_arity=1) {
        if (eq.get_input().size() != input_arity) {
            throw formatted_error("Error: input arity of equation is {}, expected {}",
                eq.get_input().size(), input_arity);
        }

        if (eq.get_output().size() != output_arity) {
            throw formatted_error("Error: output arity of equation is {}, expected {}",
                eq.get_output().size(), output_arity);
        }
    }

    template<array_t (array_t::*op)() const>
    void execute_unary_op(const equation& eq) {
        check_fixed_arity(eq, 1);
        if (eq.get_input(0).get_dtype() == dtype_t::BOOL) {
            throw std::logic_error("Error: cannot execute unary op on boolean argument");
        }
        emplace_variable(eq.get_output(0), (get_input(eq, 0).*op)());
    }

    template<array_t (array_t::*op)(const array_t&) const>
    void execute_binary_op(const equation& eq) {
        if (eq.get_input(0).get_dtype() != eq.get_input(1).get_dtype()) {
            throw std::logic_error("Error: can only execute binary op on arguments of the same type");
        }

        check_fixed_arity(eq, 2);
        emplace_variable(eq.get_output(0), (get_input(eq, 0).*op)(get_input(eq, 1)));
    }

    template<array_t (array_t::*op)(const array_t&) const>
    void execute_binary_comparison_op(const equation& eq) {
        check_fixed_arity(eq, 2);
        emplace_variable(eq.get_output(0), (get_input(eq, 0).*op)(get_input(eq, 1)));
    }

    const expression& jaxpr;
    std::vector<std::unique_ptr<array_t>> values;
    std::vector<std::optional<array_t>> literal_buffer;
};

template<bool single_output=true>
auto invoke_vm(const expression& jaxpr, const std::vector<array_t>& input) {
    jax_vm vm{jaxpr};
    if constexpr (single_output) {
        return vm.run(input)[0];
    } else {
        return vm.run(input);
    }
}

template<bool single_output=true>
auto invoke_vm(const expression& jaxpr, const array_t& input) {
    jax_vm vm{jaxpr};
    if constexpr (single_output) {
        return vm.run(std::vector{input})[0];
    } else {
        return vm.run(std::vector{input});
    }
}

#endif //MICROJAX_JAX_VM_H
