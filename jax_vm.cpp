//
// Created by Yuriy Kulinchenko on 16/07/2026.
//

#include "jax_vm.h"
#include "helper.h"
#include "jax_functions.h"
#include "type_checker.h"

namespace {
    constexpr bool check_types = true;

    void verify(const equation& eq, const std::expected<type_t, std::string>& result) {
        if (!result) {
            throw formatted_error("Error: type check of {} failed: {}",
                to_string(eq.get_op()), result.error());
        }
        if (*result != eq.get_output(0).get_type()) {
            throw formatted_error("Error: {} output type does not match its declared output variable",
                to_string(eq.get_op()));
        }
    }

    void verify_multi(const equation& eq, const std::expected<std::vector<type_t>, std::string>& result) {
        if (!result) {
            throw formatted_error("Error: type check of {} failed: {}",
                to_string(eq.get_op()), result.error());
        }
        if (result->size() != eq.get_output().size()) {
            throw formatted_error("Error: {} produced {} output types but the equation declares {} outputs",
                to_string(eq.get_op()), result->size(), eq.get_output().size());
        }
        for (const auto& [expected, out]: std::views::zip(*result, eq.get_output())) {
            if (expected != out.get_type()) {
                throw formatted_error("Error: {} output type does not match its declared output variable",
                    to_string(eq.get_op()));
            }
        }
    }

    void check_equation_type(const equation& eq) {
        using enum primitive_op;
        auto t = [&](size_t i) { return eq.get_input(i).get_type(); };
        auto all_types = [&]() {
            std::vector<type_t> types;
            for (const value& v: eq.get_input()) types.push_back(v.get_type());
            return types;
        };
        auto tail_types = [&]() {
            std::vector<type_t> types;
            for (const value& v: eq.get_input() | std::views::drop(1)) types.push_back(v.get_type());
            return types;
        };

        switch (eq.get_op()) {
            case SIN: case COS: case EXP: case LOG: case NEG:
            case SQRT: case RSQRT: case TANH: case LOGISTIC:
                verify(eq, unary_elementwise_type(t(0))); break;

            case ADD: case SUB: case MUL: case DIV: case MAX: case MIN: case POW:
                verify(eq, binary_elementwise_type(t(0), t(1))); break;

            case LT: case LE: case GT: case GE: case EQ: case NE:
                verify(eq, comparison_type(t(0), t(1))); break;

            case INTEGER_POW:
                verify(eq, std::expected<type_t, std::string>{t(0)}); break;

            case REDUCE_SUM:
                verify(eq, reduce_type(t(0), std::get<reduce_sum_params>(eq.get_params()).axes)); break;
            case REDUCE_MAX:
                verify(eq, reduce_type(t(0), std::get<reduce_max_params>(eq.get_params()).axes)); break;
            case REDUCE_MIN:
                verify(eq, reduce_type(t(0), std::get<reduce_min_params>(eq.get_params()).axes)); break;

            case DOT_GENERAL: {
                auto& [l_c, r_c, l_b, r_b] = std::get<dot_general_params>(eq.get_params());
                verify(eq, dot_general_type(t(0), t(1), l_c, r_c, l_b, r_b));
                break;
            }

            case BROADCAST_IN_DIM: {
                auto& [shape, dims] = std::get<broadcast_in_dim_params>(eq.get_params());
                verify(eq, broadcast_in_dim_type(t(0), shape, dims));
                break;
            }

            case CONVERT_ELEMENT_TYPE:
                verify(eq, convert_element_type_type(t(0),
                    std::get<convert_element_type_params>(eq.get_params()).new_dtype)); break;

            case TRANSPOSE:
                verify(eq, transpose_type(t(0), std::get<transpose_params>(eq.get_params()).permutation)); break;

            case RESHAPE:
                verify(eq, reshape_type(t(0), std::get<reshape_params>(eq.get_params()).new_sizes)); break;

            case GATHER:
                verify(eq, gather_type(t(0), t(1))); break;

            case SCATTER_ADD: case SCATTER_MUL: case SCATTER_MAX: case SCATTER_MIN: case SCATTER:
                verify(eq, scatter_type(t(0), t(1), t(2))); break;

            case SLICE: {
                auto& [start, limit, strides] = std::get<slice_params>(eq.get_params());
                verify(eq, slice_type(t(0), start, limit, strides));
                break;
            }

            case PAD:
                verify(eq, pad_type(t(0), t(1), std::get<pad_params>(eq.get_params()).padding_config)); break;

            case CONCATENATE:
                verify(eq, concatenate_type(all_types(),
                    std::get<concatenate_params>(eq.get_params()).dimension)); break;

            case SELECT:
                verify(eq, select_type(t(0), tail_types())); break;

            case COND:
                verify_multi(eq, cond_type(t(0), tail_types(),
                    std::get<cond_params>(eq.get_params()).branches)); break;

            case SCAN: {
                auto& params = std::get<scan_params>(eq.get_params());
                verify_multi(eq, scan_type(all_types(), params.jaxpr,
                    params.length, params.num_consts, params.num_carry));
                break;
            }
        }
    }
}

jax_vm::jax_vm(const expression& jaxpr):
jaxpr(jaxpr),
values(std::vector<array_t>(jaxpr.var_id)) {}

std::vector<array_t> jax_vm::run(const std::vector<array_t>& input) {

    // TODO: Verify that size(input) == size(invars)

    reset();

    for (const auto& [invar, input_array]: std::views::zip(jaxpr.invars, input)) {
        values[invar.get_id()] = input_array;
    }

    for (const auto& [constvar, const_array]: std::views::zip(jaxpr.constvars, jaxpr.consts)) {
        values[constvar.get_id()] = const_array;
    }

    for (auto& eq: jaxpr.equations) {
        if constexpr (check_types) check_equation_type(eq);
        populate_literal_buffer(eq);
        switch (eq.get_op()) {
            using enum primitive_op;

            case SIN: execute_unary_op<[](const array_t& x) {return sin(x);}>(eq); break;
            case COS: execute_unary_op<[](const array_t& x) {return cos(x);}>(eq); break;
            case EXP: execute_unary_op<[](const array_t& x) {return exp(x);}>(eq); break;
            case LOG: execute_unary_op<[](const array_t& x) {return log(x);}>(eq); break;
            case NEG: execute_unary_op<[](const array_t& x) {return -x;}>(eq); break;
            case SQRT: execute_unary_op<[](const array_t& x) {return sqrt(x);}>(eq); break;
            case RSQRT: execute_unary_op<[](const array_t& x) {return rsqrt(x);}>(eq); break;
            case TANH: execute_unary_op<[](const array_t& x) {return tanh(x);}>(eq); break;
            case LOGISTIC: execute_unary_op<[](const array_t& x) {return logistic(x);}>(eq); break;

            case ADD: execute_binary_op<([](const array_t& x, const array_t& y) {return x + y;})>(eq); break;
            case SUB: execute_binary_op<([](const array_t& x, const array_t& y) {return x - y;})>(eq); break;
            case MUL: execute_binary_op<([](const array_t& x, const array_t& y) {return x * y;})>(eq); break;
            case DIV: execute_binary_op<([](const array_t& x, const array_t& y) {return x / y;})>(eq); break;
            case MAX: execute_binary_op<[](const array_t& x, const array_t& y) {return max(x, y);}>(eq); break;
            case MIN: execute_binary_op<[](const array_t& x, const array_t& y) {return min(x, y);}>(eq); break;
            case POW: execute_binary_op<[](const array_t& x, const array_t& y) {return pow(x, y);}>(eq); break;

            case LT: execute_binary_comparison_op<[](const array_t& x, const array_t& y) {return x < y;}>(eq); break;
            case LE: execute_binary_comparison_op<[](const array_t& x, const array_t& y) {return x <= y;}>(eq); break;
            case GT: execute_binary_comparison_op<[](const array_t& x, const array_t& y) {return x > y;}>(eq); break;
            case GE: execute_binary_comparison_op<[](const array_t& x, const array_t& y) {return x >= y;}>(eq); break;

            case EQ: execute_binary_comparison_op<[](const array_t& x, const array_t& y)
            {return elementwise_equal(x, y);}>(eq); break;
            case NE: execute_binary_comparison_op<[](const array_t& x, const array_t& y)
            {return elementwise_not_equal(x, y);}>(eq); break;

            case DOT_GENERAL: {
                check_fixed_arity(eq, 2);
                auto& [l_c, r_c, l_b, r_b] = std::get<dot_general_params>(eq.get_params());
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).dot_general(get_input(eq, 1), l_c, r_c, l_b, r_b));
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

            case INTEGER_POW: {
                check_fixed_arity(eq, 1);
                size_t y = std::get<integer_pow_params>(eq.get_params()).y;
                emplace_variable(eq.get_output(0), integer_pow(get_input(eq, 0), y));
                break;
            }

            case REDUCE_SUM: {
                check_fixed_arity(eq, 1);
                auto& axes = std::get<reduce_sum_params>(eq.get_params()).axes;
                emplace_variable(eq.get_output(0), get_input(eq, 0).reduce_sum(axes));
                break;
            }

            case REDUCE_MAX: {
                check_fixed_arity(eq, 1);
                auto& axes = std::get<reduce_max_params>(eq.get_params()).axes;
                emplace_variable(eq.get_output(0), get_input(eq, 0).reduce_max(axes));
                break;
            }

            case REDUCE_MIN: {
                check_fixed_arity(eq, 1);
                auto& axes = std::get<reduce_min_params>(eq.get_params()).axes;
                emplace_variable(eq.get_output(0), get_input(eq, 0).reduce_min(axes));
                break;
            }

            case RESHAPE: {
                check_fixed_arity(eq, 1);
                auto& new_sizes = std::get<reshape_params>(eq.get_params()).new_sizes;
                emplace_variable(eq.get_output(0), get_input(eq, 0).reshape(new_sizes));
                break;
            }

            case GATHER: {
                check_fixed_arity(eq, 2);
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).at(get_input(eq, 1)).get());
                break;
            }

            case SCATTER_ADD: {
                check_fixed_arity(eq, 3);
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).at(get_input(eq, 1)).add(get_input(eq, 2)));
                break;
            }

            case SCATTER_MUL: {
                check_fixed_arity(eq, 3);
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).at(get_input(eq, 1)).multiply(get_input(eq, 2)));
                break;
            }

            case SCATTER_MAX: {
                check_fixed_arity(eq, 3);
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).at(get_input(eq, 1)).max(get_input(eq, 2)));
                break;
            }

            case SCATTER_MIN: {
                check_fixed_arity(eq, 3);
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).at(get_input(eq, 1)).min(get_input(eq, 2)));
                break;
            }

            case SCATTER: {
                check_fixed_arity(eq, 3);
                emplace_variable(eq.get_output(0),
                    get_input(eq, 0).at(get_input(eq, 1)).set(get_input(eq, 2)));
                break;
            }

            case SLICE: {
                check_fixed_arity(eq, 1);
                auto& [start_indices, limit_indices, strides] = std::get<slice_params>(eq.get_params());
                emplace_variable(eq.get_output(0),
                    slice(get_input(eq, 0), start_indices, limit_indices, strides));
                break;
            }

            case PAD: {
                check_fixed_arity(eq, 2);
                auto& padding_config = std::get<pad_params>(eq.get_params()).padding_config;
                emplace_variable(eq.get_output(0),
                    pad(get_input(eq, 0), get_input(eq, 1), padding_config));
                break;
            }

            case CONCATENATE: {
                size_t axis = std::get<concatenate_params>(eq.get_params()).dimension;
                const size_t num_tensors = eq.get_input().size();

                const std::vector<size_t>& first_shape = get_input(eq, 0).get_type().get_shape();
                size_t rank = first_shape.size();

                size_t concat_size = 0;
                for (size_t t = 0; t < num_tensors; t++) {
                    concat_size += get_input(eq, t).get_type().get_shape()[axis];
                }

                std::vector new_shape {first_shape};
                new_shape[axis] = concat_size;
                array_t result {type_t{get_input(eq, 0).get_type().get_dtype(), new_shape},
                    std::vector<double>(num_elements(new_shape), 0)};

                size_t concatenate_dimension_offset = 0;
                for (size_t t = 0; t < num_tensors; t++) {
                    const array_t& tensor = get_input(eq, t);
                    const std::vector<size_t>& source_shape = tensor.get_type().get_shape();
                    std::vector<size_t> indicies (rank, 0);
                    for (auto& is: cartesian_product{indicies, source_shape}) {
                        double value = tensor[is];
                        indicies[axis] += concatenate_dimension_offset;
                        result[indicies] = value;
                        indicies[axis] -= concatenate_dimension_offset;
                    }
                    concatenate_dimension_offset += source_shape[axis];
                }

                emplace_variable(eq.get_output(0), std::move(result));
                break;
            }

            case SELECT: {
                const array_t& pred = get_input(eq, 0);
                array_t output = array_t::build_fill(get_input(eq, 1).get_type(), 0.);
                for (size_t i = 0; i < output.get_value().size(); i++) {
                    size_t index = static_cast<size_t>(pred.get_value()[i]);
                    output.get_value()[i] = get_input(eq, 1 + index).get_value()[i];
                }
                emplace_variable(eq.get_output(0), std::move(output));
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

                auto get_xs_slice = [&](const size_t x_index, const size_t slice_index) -> const_array_span_t {
                    return get_input(eq, params.num_consts + params.num_carry + x_index).index(slice_index);
                };

                auto update_carry = [&](std::vector<array_t>& inputs, const std::span<array_t> carries) -> void {
                    for (size_t i = 0; i < carries.size(); i++) {
                        inputs[params.num_consts + i] = carries[i];
                    }
                };

                // Simultaneous update of multiple ys, across multiple slices:
                auto update_ys = [&](const size_t slice_index, std::span<array_t> ys_slices) -> void {
                    for (size_t i = 0; i < ys_slices.size(); i++) {
                        Y[i].index(slice_index) += ys_slices[i];
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
        if (val.is_literal()) {
            return literal_to_array(val.get_literal());
        }
        return fetch_value(val.get_var());
    }) | std::ranges::to<std::vector<array_t>>();
}

void jax_vm::reset() {
    for (auto& v: values) v = array_t{};
}

const array_t& jax_vm::fetch_value(const var_t& var) const {
    return values[var.get_id()];
}

void jax_vm::emplace_variable(const var_t& var, array_t array) {
    values[var.get_id()] = std::move(array);
}

const array_t& jax_vm::get_input(const equation& eq, size_t i) {
    // Assumes a populated literal buffer
    if (eq.get_input(i).is_literal()) {
        return *literal_buffer[i];
    }

    return fetch_value(eq.get_input(i).get_var());
}

array_t jax_vm::literal_to_array(const literal_t& literal) {
    return array_t{type_t{literal.get_dtype()}, {literal.get_value()}};
}

void jax_vm::populate_literal_buffer(const equation& eq) {
    literal_buffer.assign(eq.get_input().size(), std::nullopt);
    for (size_t i = 0; i < eq.get_input().size(); i++) {
        if (eq.get_input(i).is_literal()) {
            auto& literal = eq.get_input(i).get_literal();
            literal_buffer[i] = literal_to_array(literal);
        }
    }
}

void jax_vm::check_fixed_arity(const equation& eq, size_t input_arity, size_t output_arity) {
    if (eq.get_input().size() != input_arity) {
        throw formatted_error("Error: input arity of equation is {}, expected {}",
            eq.get_input().size(), input_arity);
    }

    if (eq.get_output().size() != output_arity) {
        throw formatted_error("Error: output arity of equation is {}, expected {}",
            eq.get_output().size(), output_arity);
    }
}
