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
                    auto outputs = vm.run(std::views::iota(0ul, eq.get_input().size())
                        | std::views::transform([&](size_t i) -> array_t {
                        return get_input(eq, i);
                    }) | std::ranges::to<std::vector<array_t>>());


                    for (const auto& [output_var, output_array]: std::views::zip(eq.get_output(), outputs)) {
                        emplace_variable(output_var, std::move(output_array));
                    }
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

private:

    const array_t& fetch_value(const var_t& var) {
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

    void check_fixed_arity(const equation& eq, size_t input_arity, size_t output_arity=1) {
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
