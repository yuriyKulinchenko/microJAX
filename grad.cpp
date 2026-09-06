#include "grad.h"
#include "helper.h"

/*
============================== High level architecture of grad ==============================

'grad' is intended to provide a jaxpr -> jaxpr transformation, which takes a computation,
and returns the grad / derivative of that computation. This is done through a forward pass
followed by backpropagation.

The forward pass will essentially be a copy of the original jaxpr, and it will also be used to
initialize the adjoints for each variable introduced.

In order to perform backpropagation, you need to track the adjoints of all intermediate nodes
in the original jaxpr. Adjoints are computed as a sum, so careful handling is required for this
given that the jaxpr language is SSA, meaning that I cannot simply mutate a single memory
location to accumulate the sum.

The solution will be to maintain a mapping from an intermediate in the original jaxpr to the
currently active adjoint. The active adjoint will continously change as it is added to,
so this mapping will also be continously changing. For now, this mapping will be an
unordered_map: this can be swapped out later for performance.

The core loop of backpropogation will essentially involve iterating through the original
expression in reverse, handling each equation individually. Each equation will be handled
by updating all necessary adjoints.

*/

using namespace jax;

expression grad(const expression& expr) {
    auto instance = grad_class{expr};
    return instance.find_grad();
}

expression grad_general(const expression& expr) {
    auto instance = grad_class{expr};
    return instance.find_grad_general();
}

grad_class::grad_class(const expression& expr): input_expr(expr) {
    output_expr.var_id = expr.var_id;
}

var_t grad_class::fresh_var(type_t type) {
    return var_t{output_expr.new_var_id(), std::move(type)};
}

void grad_class::introduce_adjoint(const var_t& var) {
    /*
    For each intermediate value in the calculation, an adjoint
    has to be produced which is added to. A simple way to introduce
    each of these adjoints is to initialize each adjoint to zero,
    and then 'modifying' each adjoint by creating a fresh intermediate,
    and tracking each of these intermediates.

    This would work, however it would be far less readable. Every
    emitted expression from grad would contain a sequence of zero
    initializations in the middle. A better solution is to keep track
    of whether an adjoint is 'active' or not. If it is not active,
    then it has not been used yet, and implicitly has a value of zero.
    Then, it can be cleanly introduced.
    */

    active_adjoint_map[var.get_id()] = false;
}

value* grad_class::get_adjoint(const var_t& var) {
    if (auto it = variable_adjoint_map.find(var.get_id());
        it != variable_adjoint_map.end()) {
        return &it->second;
    }
    return nullptr;
}

value grad_class::get_adjoint_value(const var_t& var) {
    value* val = get_adjoint(var);
    if (val) {
        return *val;
    }

    return broadcasted_value(var.get_type(), 0);
}

bool grad_class::adjoint_is_active(const var_t& var, bool apply_update) {
    const auto it = active_adjoint_map.find(var.get_id());
    if (it->second) return true;
    if (apply_update) it->second = true;
    return false;
}

void grad_class::update_adjoint(const var_t& input_adj_var, const value& product_val) {
    // update_adjoint no longer checks if the input is zero

    // Case 1: input_adj is tied to a constvar:
    if (!active_adjoint_map.contains(input_adj_var.get_id())) return;


    // Case 2: input_adj does not yet exist:
    if (!adjoint_is_active(input_adj_var)) {
        variable_adjoint_map.insert_or_assign(input_adj_var.get_id(), product_val);
    } else {
        // Case 3: it does exist - sum is required:
        auto sum_var = var_t{output_expr.new_var_id(), product_val.get_type()};
        const auto& input_adj = *get_adjoint(input_adj_var);

        output_expr.equations.emplace_back(
            std::vector{product_val, input_adj},
            std::vector{sum_var},
            primitive_op::ADD
        );

        // Carry through the updated adjoint:
        variable_adjoint_map.insert_or_assign(input_adj_var.get_id(), value{sum_var});

    }
}

void grad_class::update_adjoint(
    const var_t& input_adj_var,
    const value& f_prime_val,
    const value& adjoint_val) {

    auto product_var = fresh_var(input_adj_var.get_type());

    output_expr.equations.emplace_back(
        std::vector{f_prime_val, adjoint_val},
        std::vector{product_var},
        primitive_op::MUL
    );

    update_adjoint(input_adj_var, value{product_var});
}

value grad_class::negate(const value& val) {
    if (val.is<literal_t>()) {
        auto& literal = val.get_literal();
        if (literal.get_dtype() == dtype_t::BOOL) {
            throw std::logic_error{"Error: cannot negate boolean"};
        }
        return value{literal_t(literal.get_dtype(), -literal.get_value())};
    }

    // val is a var_t - emit a negation:

    auto& var = val.get_var();
    var_t negated_var = fresh_var(var.get_type());

    output_expr.equations.emplace_back(
        std::vector{val},
        std::vector{negated_var},
        primitive_op::NEG
    );

    return value{negated_var};
}

value grad_class::broadcasted_value(const type_t& type, double x) {
    using namespace jax;
    literal_t literal{type.get_dtype(), x};

    if (type.get_shape().size() == 0) {
        return value{literal};
    }

    var_t broadcast_var = fresh_var(type);

    output_expr.equations.emplace_back(
        std::vector{value{literal}},
        std::vector{broadcast_var},
        primitive_op::BROADCAST_IN_DIM,
        broadcast_in_dim_params{type.get_shape()}
    );

    return value{broadcast_var};
}

bool all_inputs_constant(const std::vector<value>& inputs) {
    for (auto& input: inputs) if (input.is<var_t>()) return false;
    return true;
}

bool grad_class::should_propagate(const std::vector<value>& input_vals,
        const std::vector<var_t>& output_vars) {
    // First, check if the output adjoint is non-zero,
    // and if any of the input values are variables

    bool active_adjoint_found = false;
    for (auto& output_var: output_vars) {
        if (adjoint_is_active(output_var, false)) {
            active_adjoint_found = true;
            break;
        }
    }

    if (!active_adjoint_found) {
        // Nothing to propagate backwards:
        return false;
    }

    // Check if any of the input values are differentiable variables:

    bool input_var_exists = false;
    for (auto& input_val: input_vals) {
        if (input_val.is<var_t>() && !is_integral(input_val.get_dtype())) {
            input_var_exists = true;
            break;
        }
    }

    if (!input_var_exists) {
        // Nothing to update:
        return false;
    }

    return true;
}

void grad_class::propagate_adjoints(equation& eq) {
    // Requires knowledge of output adjoints, and shape of eq.op to dispatch over
    // If an adjoint does not exist, take it to be zero

    // Check if there is anything to propagate to:
    if (all_inputs_constant(eq.get_input())) return;

    switch (eq.get_op()) {
        using enum primitive_op;
        case SIN: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            auto f_prime_var = fresh_var(output_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{value{input_var}},
                std::vector{f_prime_var},
                COS
            );

            update_adjoint(input_var, value{f_prime_var}, *output_adj);

            break;
        }

        case COS: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            auto var1 = fresh_var(output_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{value{input_var}},
                std::vector{var1},
                SIN
            );

            auto f_prime_var = fresh_var(output_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{value{var1}},
                std::vector{f_prime_var},
                NEG
            );

            update_adjoint(input_var, value{f_prime_var}, *output_adj);

            break;
        }

        case EXP: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            // Derivative of y = exp(x) is y
            update_adjoint(input_var, value{output_var}, *output_adj);
            break;
        }

        case LOG: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            // derivative of y = log(x) is 1/x

            value broadcasted_1 = broadcasted_value(input_var.get_type(), 1);
            auto f_prime_var = fresh_var(input_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{broadcasted_1, value{input_var}},
                std::vector{f_prime_var},
                DIV
            );

            update_adjoint(input_var, value{f_prime_var}, *output_adj);
            break;
        }

        case NEG: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();
            update_adjoint(input_var, negate(*output_adj));

            break;
        }

        case SQRT: break;
        case RSQRT: break;
        case TANH: break;
        case LOGISTIC: break;
        case INTEGER_POW: break;

        case ADD: {
            // z = x + y
            // x' += dL/dz dz/dx
            // dz/dx = 1 => x' += z'
            // Similarly, y' += z'

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& x_val = eq.get_input(0);
            auto& y_val = eq.get_input(1);

            if (x_val.is<var_t>()) {
                update_adjoint(x_val.get_var(), *output_adj);
            }

            if (y_val.is<var_t>()) {
                update_adjoint(y_val.get_var(), *output_adj);
            }
            break;
        }

        case SUB: {
            // z = x - y
            // x' += dL/dz dz/dx
            // dz/dx = 1 => x' += z'
            // Similarly, y' += -z'

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& x_val = eq.get_input(0);
            auto& y_val = eq.get_input(1);

            if (x_val.is<var_t>()) {
                update_adjoint(x_val.get_var(), *output_adj);
            }

            if (y_val.is<var_t>()) {
                update_adjoint(y_val.get_var(), negate(*output_adj));
            }
            break;
        }

        case MUL: {
            // z = x * y
            // x' += dL/dz dz/dx
            // dz/dx = y => x' += y * z'
            // Similarly, y' += x * z'

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& x_val = eq.get_input(0);
            auto& y_val = eq.get_input(1);

            if (x_val.is<var_t>()) {
                update_adjoint(x_val.get_var(), y_val, *output_adj);
            }

            if (y_val.is<var_t>()) {
                update_adjoint(y_val.get_var(), x_val, *output_adj);
            }

            break;
        }

        case DIV: {
            // z = x / y
            // x' += dL/dz dz/dx
            // dz/dx = 1 / y => x' += z' / y
            // y' += dL/dz dz/dy
            // dz/dy = - x / (y * y) => y' += z' * neg(z/y)

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& x_val = eq.get_input(0);
            auto& y_val = eq.get_input(1);

            if (x_val.is<var_t>()) {
                auto quotient_val = fresh_var(x_val.get_type());

                output_expr.equations.emplace_back(
                    std::vector{*output_adj, y_val},
                    std::vector{quotient_val},
                    DIV
                );

                update_adjoint(x_val.get_var(), value{quotient_val});
            }

            // dz/dy = - x / (y * y) => y' += z' * neg(z/y)

            if (y_val.is<var_t>()) {
                auto quotient_val = fresh_var(y_val.get_type());

                output_expr.equations.emplace_back(
                    std::vector{value{output_var}, y_val},
                    std::vector{quotient_val},
                    DIV
                );

                auto negated_quotient_val = fresh_var(y_val.get_type());

                output_expr.equations.emplace_back(
                    std::vector{value{quotient_val}},
                    std::vector{negated_quotient_val},
                    NEG
                );

                update_adjoint(y_val.get_var(), value{negated_quotient_val}, *output_adj);
            }

            break;
        }

        case MAX: break;
        case MIN: break;
        case POW: break;

        case REDUCE_SUM: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_val = eq.get_input(0);
            if (input_val.is<literal_t>()) break;
            auto& input_var = input_val.get_var();

            // The adjoint is a broadcast of the output_adj
            // The broadcast_dimensions are the complement of the axes:

            const auto& input_shape = input_var.get_type().get_shape();
            const auto& axes = std::get<reduce_sum_params>(eq.get_params()).axes;

            std::vector<size_t> axes_complement = complement(axes, input_shape.size());

            auto broadcast_adjoint = fresh_var(input_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{*output_adj},
                std::vector{broadcast_adjoint},
                BROADCAST_IN_DIM,
                broadcast_in_dim_params{input_var.get_type().get_shape(),
                    std::move(axes_complement)}
            );

            update_adjoint(input_var, value{broadcast_adjoint});
            break;
        }

        case REDUCE_MAX: break;
        case REDUCE_MIN: break;

        case BROADCAST_IN_DIM: {

            /*
            Suppose you perform a broadcast (5,1,3) -> (5,10,9,15,3),
            where 10 and 15 are newly inserted, and 1 is stretched to 9.

            The adjoint for the input X will be a summation over the ranks
            with sizes 10, 9 and 5 followed by another broadcast that re-inserts
            the size 1 rank.
            */

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_val = eq.get_input(0);
            if (input_val.is<literal_t>()) break;
            auto& input_var = input_val.get_var();

            // The adjoint is a summation over the newly added ranks
            // The newly added ranks are the complement of broadcast_dimensions

            const auto& params = std::get<broadcast_in_dim_params>(eq.get_params());
            const auto& output_shape = params.shape;
            const auto& broadcast_dims = params.broadcast_dimensions;

            std::vector<size_t> axes = complement(broadcast_dims, output_shape.size());

            // Add handling of stretching:

            size_t old_axes_size = axes.size();

            const auto& input_shape = input_var.get_type().get_shape();
            for (size_t i = 0; i < input_shape.size(); i++) {
                if (input_shape[i] == 1) {
                    size_t j = params.broadcast_dimensions[i];
                    if (output_shape[j] != 1) {
                        // A 1 has been stretched to output_shape[j]:
                        axes.push_back(j);
                    }
                }
            }

            std::ranges::inplace_merge(axes.begin(), axes.begin() + old_axes_size, axes.end());

            // If any stretching occured, an additional broadcast must also happen:

            if (old_axes_size != axes.size()) {
                // The new type shape is the complement of axes:

                std::vector<size_t> summed_shape = complement(axes, output_shape.size());
                for (auto& x: summed_shape) {
                    x = output_shape[x];
                }

                auto summed_type = type_t{input_var.get_type().get_dtype(), std::move(summed_shape)};

                auto summed_adjoint = fresh_var(std::move(summed_type));

                output_expr.equations.emplace_back(
                    std::vector{*output_adj},
                    std::vector{summed_adjoint},
                    REDUCE_SUM,
                    reduce_sum_params{std::move(axes)}
                );

                // Now, a re-broadcast has to happen:

                std::vector<size_t> broadcast_dimensions {};

                for (size_t i = 0; i < input_shape.size(); i++) {
                    bool was_stretched = (input_shape[i] == 1 && output_shape[broadcast_dims[i]] != 1);
                    if (!was_stretched) broadcast_dimensions.push_back(i);
                }

                auto broadcast_adjoint = fresh_var(input_var.get_type());

                output_expr.equations.emplace_back(
                    std::vector{value{summed_adjoint}},
                    std::vector{broadcast_adjoint},
                    BROADCAST_IN_DIM,
                    broadcast_in_dim_params{input_shape, broadcast_dimensions}
                );

                update_adjoint(input_var, value{broadcast_adjoint});
            } else {
                auto summed_adjoint = fresh_var(input_var.get_type());

                output_expr.equations.emplace_back(
                    std::vector{*output_adj},
                    std::vector{summed_adjoint},
                    REDUCE_SUM,
                    reduce_sum_params{std::move(axes)}
                );

                update_adjoint(input_var, value{summed_adjoint});
            }
            break;
        }

        case DOT_GENERAL: {

            // dot_general no longer has to support literals:

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& x_var = eq.get_input(0).get_var();
            auto& y_var = eq.get_input(1).get_var();
            auto& x_shape = x_var.get_type().get_shape();
            auto& y_shape = y_var.get_type().get_shape();

            const auto& params = std::get<dot_general_params>(eq.get_params());

            // Let b = batch indices, c = contraction indices,
            // f^(X) = free indices of X, f^(Y) = free indices of Y
            // Adjoint X is a contraction over f^(Y), and vice versa


            // Goal 1: free indices need to be found in X and Y

            auto free_indices_x = complement(
                params.left_contract, params.left_batch,
                x_shape.size()
            );

            auto free_indices_y = complement(
                params.right_contract, params.right_batch,
                y_shape.size()
            );

            // Goal 2: free indices need to be found in the product:
            // this should be a contiguous sequence at a reliable offset
            // (batch, x free, y free)

            size_t x_offset = params.left_batch.size();
            size_t y_offset = x_offset + free_indices_x.size();

            std::vector<size_t> free_indices_x_product(free_indices_x.size());
            for (size_t i = 0; i < free_indices_x.size(); i++) {
                free_indices_x_product[i] = x_offset + i;
            }

            std::vector<size_t> free_indices_y_product(free_indices_y.size());
            for (size_t i = 0; i < free_indices_y.size(); i++) {
                free_indices_y_product[i] = y_offset + i;
            }

            auto batch_indices_x = params.left_batch;
            auto batch_indices_y = params.right_batch;

            std::vector<size_t> batch_indices_x_product(batch_indices_x.size());
            for (size_t i = 0; i < batch_indices_x.size(); i++) {
                batch_indices_x_product[i] = i;
            }

            auto batch_indices_y_product = batch_indices_x_product;

            auto get_inverse_permutation = [](
                const std::vector<size_t>& batch,
                const std::vector<size_t>& contract,
                const std::vector<size_t>& free
                ) -> std::vector<size_t> {

                const size_t contract_offset = batch.size();
                const size_t free_offset = contract_offset + contract.size();
                const size_t total_size = free_offset + free.size();

                std::vector<size_t> inverse_permutation(total_size);

                std::ranges::copy(batch, inverse_permutation.begin());
                std::ranges::copy(contract, inverse_permutation.begin() + contract_offset);
                std::ranges::copy(free, inverse_permutation.begin() + free_offset);
                return inverse_permutation;
            };


            std::vector<size_t> inverse_permutation =
                get_inverse_permutation(params.left_batch, params.left_contract, free_indices_x);

            type_t new_type {x_var.get_type().get_dtype(), permute(x_shape, inverse_permutation)};

            auto dot_general_x = fresh_var(std::move(new_type));

            output_expr.equations.emplace_back(
                std::vector{value{y_var}, *output_adj},
                std::vector{dot_general_x},
                DOT_GENERAL,
                dot_general_params{
                    free_indices_y, std::move(free_indices_y_product),
                    std::move(batch_indices_y), std::move(batch_indices_y_product)
                }
            );

            // Transpose may be necessary:

            if (is_identity_permutation(inverse_permutation)) {
                update_adjoint(x_var, value{dot_general_x});
            } else {
                auto transpose_dot_general_x = fresh_var(x_var.get_type());

                output_expr.equations.emplace_back(
                    std::vector{value{dot_general_x}},
                    std::vector{transpose_dot_general_x},
                    TRANSPOSE,
                    transpose_params{invert_permutation(inverse_permutation)}
                );

                update_adjoint(x_var, value{transpose_dot_general_x});
            }



            inverse_permutation =
            get_inverse_permutation(params.right_batch, params.right_contract, free_indices_y);

            new_type = {y_var.get_type().get_dtype(), permute(y_shape, inverse_permutation)};

            auto dot_general_y = fresh_var(std::move(new_type));

            output_expr.equations.emplace_back(
                std::vector{value{x_var}, *output_adj},
                std::vector{dot_general_y},
                DOT_GENERAL,
                dot_general_params{
                    std::move(free_indices_x), std::move(free_indices_x_product),
                    std::move(batch_indices_x), std::move(batch_indices_x_product)
                }
            );


            // Transpose may be necessary:

            if (is_identity_permutation(inverse_permutation)) {
                update_adjoint(y_var, value{dot_general_y});
            } else {
                auto transpose_dot_general_y = fresh_var(y_var.get_type());

                output_expr.equations.emplace_back(
                    std::vector{value{dot_general_y}},
                    std::vector{transpose_dot_general_y},
                    TRANSPOSE,
                    transpose_params{invert_permutation(inverse_permutation)}
                );

                update_adjoint(y_var, value{transpose_dot_general_y});
            }

            break;
        }

        case TRANSPOSE: {
            // invert the transpose:
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();
            auto& params = std::get<transpose_params>(eq.get_params());

            auto transposed_adjoint = fresh_var(input_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{*output_adj},
                std::vector{transposed_adjoint},
                TRANSPOSE,
                transpose_params {invert_permutation(params.permutation)}
            );

            update_adjoint(input_var, value{transposed_adjoint});

            break;
        }

        case CONVERT_ELEMENT_TYPE: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            if (is_integral(input_var.get_dtype()) || is_integral(output_var.get_dtype())) {
                break;
            }

            // Cast the output adjoint to the type of the input:

            auto converted_adjoint = fresh_var(input_var.get_type());

            output_expr.equations.emplace_back(
                std::vector{*output_adj},
                std::vector{converted_adjoint},
                CONVERT_ELEMENT_TYPE,
                convert_element_type_params{input_var.get_dtype()}
            );

            update_adjoint(input_var, value{converted_adjoint});
            break;
        }

        case COND: {
            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;


            // Goal is to compute:
            // cond(i, f'0,...,f'm)(x0,...,xn,y'0,...,y'k)

            auto& params = std::get<cond_params>(eq.get_params());

            std::vector<expression> transformed_branches {};
            transformed_branches.reserve(params.branches.size());

            for (auto& expr: params.branches) {
                transformed_branches.push_back(grad_general(expr));
            }

            // Now, I have to specify a set of new inputs:
            // (i,x0,...,xn,y'0,...,y'k)

            size_t num_inputs = input_vals.size() + output_vars.size();
            std::vector inputs {input_vals};
            inputs.reserve(num_inputs);

            for (auto& output_var: output_vars) {
                // Find the adjoint of output_var:
                if (value* adjoint = get_adjoint(output_var)) {
                    inputs.push_back(*adjoint);
                } else {
                    inputs.push_back(broadcasted_value(output_var.get_type(), 0.));
                }
            }

            // Now, I have to specify a new set of outputs:

            std::vector<var_t> outputs {};
            outputs.reserve(input_vals.size() - 1); // exclude the index

            for (auto& input_val: input_vals | std::views::drop(1)) {
                outputs.push_back(fresh_var(input_val.get_type()));
            }

            // Gather the (input var, adjoint var) pairs before emplacing:
            // update_adjoint below emplaces into output_expr.equations, so no
            // reference into that vector may be held across the update loop.
            std::vector<std::pair<var_t, var_t>> adjoint_updates {};
            for (size_t i = 1; i < input_vals.size(); i++) {
                if (input_vals[i].is<var_t>()) {
                    adjoint_updates.emplace_back(input_vals[i].get_var(), outputs[i - 1]);
                }
            }

            output_expr.equations.emplace_back(
                std::move(inputs),
                std::move(outputs),
                COND,
                cond_params{std::move(transformed_branches)});

            for (auto& [input_var, adj_var] : adjoint_updates) {
                update_adjoint(input_var, value{adj_var});
            }

            break;
        }

        case SCAN: {
            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            // On a high level, the adjoint update will require emitting:
            // scan(f', reverse=true)(k, 0, c', C, X, Y')
            // where k is the constant, c' is a carry,
            // and C, X and Y are xs parameters.
            // The 0 parameter is the first value of k'_acc
            // this will require some re-arrangement of the parameters in f',
            // as well as the threading through of k'_acc

            auto& params = std::get<scan_params>(eq.get_params());
            expression f_prime = grad_general(params.jaxpr);

            // f_prime needs to be modified. First, the order of inputs must change.
            // They will be in the form f'(k, c, x, c', y')
            // They need to be in the form f'(k, k'_acc, c', c, x, y')

            size_t num_consts = params.num_consts;
            size_t num_carry = params.num_carry;
            size_t num_xs = eq.get_input().size() - num_consts - num_carry;
            size_t num_ys = eq.get_output().size() - num_carry;

            std::vector<var_t> permuted_invars {};
            permuted_invars.reserve(f_prime.invars.size());

            size_t offset = 0;
            for (size_t i = 0; i < num_consts; i++) {
                permuted_invars.push_back(std::move(f_prime.invars[offset + i]));
                // f(k)
            }

            for (size_t i = 0; i < num_consts; i++) {
                permuted_invars.push_back(f_prime.fresh_var(permuted_invars[i].get_type()));
                // f(k, k'_acc)
            }

            offset = num_consts + num_carry + num_xs;
            for (size_t i = 0; i < num_carry; i++) {
                permuted_invars.push_back(std::move(f_prime.invars[offset + i]));
                // f(k, k'_acc, c')
            }

            offset = num_consts;
            for (size_t i = 0; i < num_carry; i++) {
                permuted_invars.push_back(std::move(f_prime.invars[offset + i]));
                // f(k, k'_acc, c', c)
            }

            offset = num_consts + num_carry;
            for (size_t i = 0; i < num_xs; i++) {
                permuted_invars.push_back(std::move(f_prime.invars[offset + i]));
                // f(k, k'_acc, c', c, x)
            }

            offset = num_consts + num_carry + num_xs + num_carry;
            for (size_t i = 0; i < num_ys; i++) {
                permuted_invars.push_back(std::move(f_prime.invars[offset + i]));
                // f(k, k'_acc, c', c, x, y')
            }

            f_prime.invars = permuted_invars;

            // f_prime needs to be modified to correctly accumulate k'_acc
            // This will involve adding some addition instructions.

            for (size_t i = 0; i < num_consts; i++) {
                auto& const_outval = f_prime.outvals[i];
                // I need to add const_outval to its corresponding k'_acc
                auto& const_accumulator = f_prime.invars[num_consts + i];

                var_t updated_adjoint = f_prime.fresh_var(const_accumulator.get_type());

                f_prime.equations.emplace_back(
                    std::vector{value{const_accumulator}, const_outval},
                    std::vector {updated_adjoint},
                    ADD
                );

                f_prime.outvals[i] = value{updated_adjoint};
            }



            // The inputs to scan(f_prime, reverse=true) will be (k, 0, c', C, X, Y').
            // k, c', X and Y' are easily recoverable as output adjoints or inputs.
            // However, C will require performing the scan_carry_transform, which changes the signature
            // from (k, c, X) -> (c, Y) to (c, X) -> (c, Y, C):

            scan_carry_transform(eq);

            // Snapshot eq's operands and scan params before emitting equations
            // below: those calls emplace into output_expr.equations, which would
            // invalidate references into eq.
            std::vector eq_inputs {eq.get_input()};
            std::vector eq_outputs {eq.get_output()};
            size_t length = params.length;
            bool reverse = params.reverse;

            std::vector<value> inputs{};

            // Pass k:
            offset = 0;
            for (size_t i = 0; i < num_consts; i++) {
                auto& k_val = eq_inputs[offset + i];
                inputs.push_back(k_val);
            }

            // Pass initial value of k'_acc (0):
            offset = 0;
            for (size_t i = 0; i < num_consts; i++) {
                auto& k_val = eq_inputs[offset + i];
                inputs.push_back(broadcasted_value(k_val.get_type(), 0));
            }


            // Pass c':
            offset = 0;
            for (size_t i = 0; i < num_carry; i++) {
                auto& c_var = eq_outputs[offset + i];
                inputs.push_back(get_adjoint_value(c_var));
            }

            // Pass C:
            offset = num_carry + num_ys;
            for (size_t i = 0; i < num_carry; i++) {
                auto& C = eq_outputs[offset + i];
                inputs.push_back(value{C});
            }

            // Pass X:
            offset = num_consts + num_carry;
            for (size_t i = 0; i < num_xs; i++) {
                auto& X = eq_inputs[offset + i];
                inputs.push_back(X);
            }

            // Pass Y':
            offset = num_carry;
            for (size_t i = 0; i < num_ys; i++) {
                auto& y_var = eq_outputs[offset + i];
                inputs.push_back(get_adjoint_value(y_var));
            }

            // The output will be updates to (k', c_0', X'):

            std::vector<var_t> outputs {};
            for (auto& inval: eq_inputs) {
                outputs.push_back(fresh_var(inval.get_type()));
            }

            output_expr.equations.emplace_back(
                std::move(inputs),
                std::move(outputs),
                SCAN,
                scan_params{
                    .jaxpr = f_prime,
                    .length = length,
                    .num_consts = num_consts,
                    .num_carry = num_consts + num_carry,
                    .reverse = !reverse
                }
            );

            auto& updates = output_expr.equations[output_expr.equations.size() - 1].get_output();

            // Update adjoints:

            for (size_t i = 0; i < eq_inputs.size(); i++) {
                if (!eq_inputs[i].is<var_t>()) continue;
                update_adjoint(eq_inputs[i].get_var(), value{updates[i]});
            }

            break;
        }

        case SELECT: {
            // y = select(b, x1, x2)
            // => x1' += select(b, y', 0), x2' += select(b, 0, y')

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            // Snapshot the inputs before emplacing: broadcasted_value and the
            // update loop below emplace into output_expr.equations, so no
            // reference into that vector may be held across them.
            std::vector inputs {input_vals};

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null
            value zero_tensor = broadcasted_value(output_adj.get_type(), 0.);

            for (size_t i = 1; i < inputs.size(); i++) {

                if (!inputs[i].is<var_t>()) {
                    continue;
                }

                const var_t& input_var = inputs[i].get_var();

                std::vector input {inputs[0]}; // {b}
                input.reserve(inputs.size());
                for (size_t j = 0; j < inputs.size() - 1; j++) {
                    input.push_back(zero_tensor); // {b, 0, ..., 0}
                }

                input[i] = output_adj; // {b, 0, ..., y', ..., 0}

                var_t input_adj_update = fresh_var(output_adj.get_type());

                output_expr.equations.emplace_back(
                    std::move(input),
                    std::vector{input_adj_update},
                    SELECT
                );

                update_adjoint(input_var, value{input_adj_update});
            }

            break;
        }

        case RESHAPE: {
            // y = reshape(x, shape)
            // => x' += reshape(y', shape(x))

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& x = input_vals[0].get_var();
            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            var_t x_adjoint = fresh_var(x.get_type());

            output_expr.equations.emplace_back(
                std::vector{output_adj},
                std::vector{x_adjoint},
                RESHAPE,
                reshape_params {
                    .new_sizes = x.get_shape()
                }
            );

            update_adjoint(x, value{x_adjoint});
            break;
        }

        case GATHER: {
            // y = gather(x, idx)
            // => x' += scatter-add(0, idx, y')

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;
            if (!input_vals[0].is<var_t>()) break; // Nothing to propagate to x

            auto& x = input_vals[0];
            auto& idx = input_vals[1];

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null
            value zero_tensor = broadcasted_value(x.get_type(), 0.);

            var_t scatter_var = fresh_var(x.get_type());

            output_expr.equations.emplace_back(
                std::vector{zero_tensor, idx, output_adj},
                std::vector{scatter_var},
                SCATTER_ADD
            );

            update_adjoint(x.get_var(), value{scatter_var});
            break;
        }

        case SCATTER_ADD: {
            // y = scatter-add(x, idx, u)
            // => x' += y'
            // => u' += gather(y', idx)

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& x = input_vals[0];
            auto& idx = input_vals[1];
            auto& u = input_vals[2];

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            if (x.is<var_t>()) {
                update_adjoint(x.get_var(), output_adj);
            }

            if (u.is<var_t>()) {
                var_t u_adjoint = fresh_var(u.get_type());

                output_expr.equations.emplace_back(
                    std::vector{output_adj, idx},
                    std::vector{u_adjoint},
                    GATHER
                );

                update_adjoint(u.get_var(), value{u_adjoint});
            }
            break;
        }

        case SCATTER_MUL: {
            // y = scatter-mul(x, idx, u)
            // => x' += y' * y / x <=> x += scatter-mul(y', idx, u)
            // => u' += gather(y', idx) * gather(y, idx) / u (corner-case division-by-zero issue here)

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& x = input_vals[0];
            auto& idx = input_vals[1];
            auto& u = input_vals[2];
            auto& y = output_vars[0];

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            if (x.is<var_t>()) {
                // x += scatter-mul(y', idx, u)
                auto& x_type = x.get_var().get_type();

                var_t x_adjoint = fresh_var(x_type);

                output_expr.equations.emplace_back(
                    std::vector{output_adj, idx, u},
                    std::vector{x_adjoint},
                    SCATTER_MUL
                );

                update_adjoint(x.get_var(), value{x_adjoint});
            }

            if (u.is<var_t>()) {
                // u' += gather(y', idx) * gather(y, idx) / u
                auto& u_type = u.get_var().get_type();

                var_t gather_y = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{value{y}, idx},
                    std::vector{gather_y},
                    GATHER
                );

                var_t gather_y_adj = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{output_adj, idx},
                    std::vector{gather_y_adj},
                    GATHER
                );

                var_t product = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{value{gather_y}, value{gather_y_adj}},
                    std::vector{product},
                    MUL
                );

                var_t quotient = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{value{product}, u},
                    std::vector{quotient},
                    DIV
                );

                update_adjoint(u.get_var(), value{quotient});
            }
            break;
        }

        case SCATTER_MAX:
        case SCATTER_MIN: {
            // y = scatter-max(x, idx, u)
            // => x' += select(y > x, y', 0)
            // => u' += select(gather(y, idx) > u, gather(y', idx), 0)
            // Derivation is identical for scatter-min, with signs flipped.

            primitive_op comparison = eq.get_op() == SCATTER_MAX ? GT: LT;

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& x = input_vals[0];
            auto& idx = input_vals[1];
            auto& u = input_vals[2];
            auto& y = output_vars[0];

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            if (x.is<var_t>()) {
                // x' += select(y > x, y', 0)
                auto& x_type = x.get_var().get_type();

                var_t condition = fresh_var(type_t{dtype_t::BOOL, x_type.get_shape()});

                output_expr.equations.emplace_back(
                    std::vector{value{y}, x},
                    std::vector{condition},
                    comparison
                );

                value x_zero_tensor = broadcasted_value(x_type, 0.);
                var_t selected_x = fresh_var(x_type);

                output_expr.equations.emplace_back(
                    std::vector{value{condition}, output_adj, x_zero_tensor},
                    std::vector{selected_x},
                    SELECT
                );

                update_adjoint(x.get_var(), value{selected_x});
            }

            if (u.is<var_t>()) {
                // u' += select(gather(y, idx) > u, gather(y', idx), 0)
                auto& u_type = u.get_var().get_type();

                var_t gather_y = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{value{y}, idx},
                    std::vector{gather_y},
                    GATHER
                );

                var_t condition = fresh_var(type_t{dtype_t::BOOL, u_type.get_shape()});

                output_expr.equations.emplace_back(
                    std::vector{value{gather_y}, u},
                    std::vector{condition},
                    comparison
                );

                var_t gather_y_adj = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{output_adj, idx},
                    std::vector{gather_y_adj},
                    GATHER
                );

                value u_zero_tensor = broadcasted_value(u_type, 0.);
                var_t selected_u = fresh_var(u_type);

                output_expr.equations.emplace_back(
                    std::vector{value{condition}, value{gather_y_adj}, u_zero_tensor},
                    std::vector{selected_u},
                    SELECT
                );

                update_adjoint(u.get_var(), value{selected_u});
            }
            break;
        }

        case SCATTER: {
            // y = scatter(x, idx, u)
            // => x' += select(written_mask, 0, y'), where written_mask = scatter_add(0, idx, 1) > 0
            // => u' += gather(y', idx)
            // This assumes that idx provides a unique mapping: breaks otherwise

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& x = input_vals[0];
            auto& idx = input_vals[1];
            auto& u = input_vals[2];

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            if (x.is<var_t>()) {
                // written_mask = scatter_add(0, idx, 1) > 0
                // x' += select(written_mask, 0, y')

                auto& x_type = x.get_var().get_type();

                type_t counting_type {dtype_t::I32, x_type.get_shape()};
                value zero_tensor = broadcasted_value(counting_type, 0);
                value one_tensor = broadcasted_value(type_t{dtype_t::I32, u.get_type().get_shape()}, 1);

                var_t written_mask = fresh_var(counting_type);

                output_expr.equations.emplace_back(
                    std::vector{zero_tensor, idx, one_tensor},
                    std::vector{written_mask},
                    SCATTER_ADD
                );

                var_t clamped_written_mask = fresh_var(type_t{dtype_t::BOOL, x_type.get_shape()});

                output_expr.equations.emplace_back(
                    std::vector{value{written_mask}, zero_tensor},
                    std::vector{clamped_written_mask},
                    GT
                );

                var_t x_adjoint = fresh_var(x_type);

                output_expr.equations.emplace_back(
                    std::vector{value{clamped_written_mask}, output_adj, zero_tensor},
                    std::vector{x_adjoint},
                    SELECT
                );

                update_adjoint(x.get_var(), value{x_adjoint});
            }

            if (u.is<var_t>()) {
                // => u' += gather(y', idx)
                var_t u_adjoint = fresh_var(u.get_type());

                output_expr.equations.emplace_back(
                    std::vector{output_adj, idx},
                    std::vector{u_adjoint},
                    GATHER
                );

                update_adjoint(u.get_var(), value{u_adjoint});
            }
            break;
        }

        case CONCATENATE: {
            using namespace std::views;
            // y = concatenate(x1, ..., xN)
            // => xi' += slice(y', ...)

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            size_t concatenation_axis = std::get<concatenate_params>(eq.get_params()).dimension;

            if (!should_propagate(input_vals, output_vars)) break;

            var_t& y = output_vars[0];
            size_t rank = y.get_shape().size();

            auto& output_adj = *get_adjoint(y); // Guaranteed not null

            auto get_slice_params = [&, concatenation_axis](size_t start, size_t limit) -> slice_params {
                std::vector<size_t> start_indices (y.get_shape().size(), 0);
                std::vector limit_indices {y.get_shape()};

                start_indices[concatenation_axis] = start;
                limit_indices[concatenation_axis] = limit;

                return slice_params {
                    .start_indices = std::move(start_indices),
                    .limit_indices = std::move(limit_indices),
                    .strides = std::vector<size_t>(rank, 1)
                };
            };

            // Invariant: at the start of each loop, start == limit
            size_t start = 0, limit = 0;

            // All tensors that are concatenated are guaranteed to be variables:
            for (auto& x: input_vals) {
                auto& x_var = x.get_var();
                limit += x_var.get_shape()[concatenation_axis];

                var_t sliced_x = fresh_var(x.get_type());

                output_expr.equations.emplace_back(
                    std::vector{output_adj},
                    std::vector{sliced_x},
                    SLICE,
                    get_slice_params(start, limit)
                );

                update_adjoint(x_var, value{sliced_x});

                start = limit;
            }

            break;
        }

        case SLICE: {
            using namespace std::views;
            // y = slice(x, ...)
            // => x' += pad(y', 0, ...)

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& params = std::get<slice_params>(eq.get_params());
            auto& start_indices = params.start_indices;
            auto& strides = params.strides;

            auto& x = input_vals[0].get_var();
            auto& y = output_vars[0];
            size_t rank = x.get_shape().size();

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            value zero = broadcasted_value(type_t{x.get_dtype(), {}}, 0.);
            var_t padded_adjoint = fresh_var(x.get_type());

            // (low, high, interior) list:
            std::vector<std::array<size_t, 3>> padding_config {}; padding_config.reserve(rank);

            for (auto [start, stride, x_dim, y_dim]: zip(start_indices, strides, x.get_shape(), y.get_shape())) {
                size_t low = start;
                size_t high = x_dim - start - y_dim - (y_dim - 1) * (stride - 1);
                size_t interior = stride - 1;
                padding_config.push_back({low, high, interior});
            }

            output_expr.equations.emplace_back(
                std::vector{output_adj, zero},
                std::vector{padded_adjoint},
                PAD,
                pad_params {
                    .padding_config = std::move(padding_config)
                }
            );

            update_adjoint(x, value{padded_adjoint});
            break;
        }

        case PAD: {
            using namespace std::views;
            // y = pad(x, k, ...)
            // => x' += slice(y', ...)
            // => k' += reduce-sum(mask * y', all axes), where mask = pad(0, 1, ...)

            auto& input_vals = eq.get_input();
            auto& output_vars = eq.get_output();

            if (!should_propagate(input_vals, output_vars)) break;

            auto& padding_config = std::get<pad_params>(eq.get_params()).padding_config;

            auto& x = input_vals[0];
            auto& k = input_vals[1];
            auto& y = output_vars[0];

            auto& output_adj = *get_adjoint(output_vars[0]); // Guaranteed not null

            size_t rank = y.get_shape().size();

            if (x.is<var_t>()) {
               // x' += slice(y', ...)
                // Sort the slice params:

                auto& x_var = x.get_var();

                std::vector<size_t> start_indices {}; start_indices.reserve(rank);
                std::vector<size_t> limit_indices {}; limit_indices.reserve(rank);
                std::vector<size_t> strides {}; strides.reserve(rank);

                for (auto [padding, y_dim]: zip(padding_config, y.get_shape())) {
                    auto [low, high, interior] = padding;
                    start_indices.push_back(low);
                    limit_indices.push_back(y_dim - high);
                    strides.push_back(interior + 1);
                }

                var_t sliced_adjoint = fresh_var(x.get_type());

                output_expr.equations.emplace_back(
                    std::vector{output_adj},
                    std::vector{sliced_adjoint},
                    SLICE,
                    slice_params {
                        .start_indices = std::move(start_indices),
                        .limit_indices = std::move(limit_indices),
                        .strides = std::move(strides)
                    }
                );

                update_adjoint(x_var, value{sliced_adjoint});
            }

            if (k.is<var_t>()) {
                // k' += reduce-sum(mask * y', all axes), where mask = pad(0, 1, ...)

                value zero = broadcasted_value(x.get_type(), 0.);
                value one = broadcasted_value(type_t{x.get_dtype(), {}}, 1.);
                var_t mask = fresh_var(y.get_type());

                output_expr.equations.emplace_back(
                    std::vector{zero, one},
                    std::vector{mask},
                    PAD,
                    pad_params{padding_config}
                );

                var_t masked = fresh_var(y.get_type());

                output_expr.equations.emplace_back(
                    std::vector{value{mask}, output_adj},
                    std::vector{masked},
                    MUL
                );

                var_t reduction = fresh_var(k.get_type()); // Shape is scalar

                output_expr.equations.emplace_back(
                    std::vector{value{masked}},
                    std::vector{reduction},
                    REDUCE_SUM,
                    reduce_sum_params {
                        .axes = iota(rank) | std::ranges::to<std::vector<size_t>>()
                    }
                );

                update_adjoint(k.get_var(), value{reduction});
            }
            break;
        }

        case EQ:
        case NE:
        case LT:
        case LE:
        case GT:
        case GE:
        break;

        default: {
            throw std::logic_error{"Error: Not implemented"};
        }
    }
}

void grad_class::scan_carry_transform(equation& eq) {
    // If the equation is a scan, the type signature has to be modified.
    // scan(f): (k, c, X) -> (c, Y)
    // scan-carry(f): (k, c, X) -> (c, Y, C)

    auto& params = std::get<scan_params>(eq.get_params());
    auto& jaxpr = params.jaxpr;

    // Fetch the first 'num_carry' inputs to expr, push them to the output

    for (size_t i = 0; i < params.num_carry; i++) {
        jaxpr.outvals.push_back(value{jaxpr.invars[params.num_consts + i]});
    }

    // Modify the set of output variables:
    for (size_t i = 0; i < params.num_carry; i++) {
        auto& type = jaxpr.invars[params.num_consts + i].get_type();
        std::vector new_shape {params.length};
        new_shape.reserve(type.get_shape().size() + 1);

        for (size_t x: type.get_shape()) {
            new_shape.push_back(x);
        }

        eq.get_output().push_back(
            fresh_var(type_t{type.get_dtype(), std::move(new_shape)})
        );
    }
}

expression grad_class::find_grad() {
    if (input_expr.outvals.size() != 1) {
        throw formatted_error("Error: expected 1 output, received {}", input_expr.outvals.size());
    }

    auto& outval = input_expr.outvals[0];

    if (outval.get_shape().size() != 0) {
        throw std::logic_error("Error: shape of output variable must be scalar");
    }

    // Add constvars:

    output_expr.consts = input_expr.consts;
    for (auto& var: input_expr.constvars) {
        output_expr.add_constvar(var);
    }

    // Add inputs:

    for (auto& var: input_expr.invars) {
        output_expr.add_invar(var);
        introduce_adjoint(var);
    }

    // perform a forward pass:

    for (auto& eq: input_expr.equations) {
        output_expr.add_equation(eq);
        for (auto& var: eq.get_output()) {
            introduce_adjoint(var);
        }
    }

    // Seed the adjoint of the initial equation:

    if (!outval.is<literal_t>() && !is_integral(outval.get_dtype())) {
        const var_t& output_var = input_expr.outvals[0].get_var();
        update_adjoint(output_var, broadcasted_value(output_var.get_type(), 1));
    }

    // perform a backward pass:

    for (size_t i = output_expr.equations.size(); i--> 0;) {
        auto& eq = output_expr.equations[i];
        propagate_adjoints(eq);
    }

    // Populate outputs:

    for (auto& var: input_expr.invars) {
        if (value* val = get_adjoint(var)) {
            // The adjoint exists:
            output_expr.outvals.push_back(*val);
        } else {
            // The adjoint does not exist - replace it with 0:
            output_expr.outvals.push_back(broadcasted_value(var.get_type(), 0));
        }
    }

    return output_expr;
}

expression grad_class::find_grad_general() {
    // Add constvars:

    output_expr.consts = input_expr.consts;
    for (auto& var: input_expr.constvars) {
        output_expr.add_constvar(var);
    }

    // Add inputs (x0, ..., xn)

    for (auto& var: input_expr.invars) {
        output_expr.add_invar(var);
        introduce_adjoint(var);
    }

    // perform a forward pass:

    for (auto& eq: input_expr.equations) {
        output_expr.add_equation(eq);
        for (auto& var: eq.get_output()) {
            introduce_adjoint(var);
        }
    }

    // Add output adjoints (y'0, ..., y'm) and
    // Seed the adjoints of the initial equation:

    for (auto& output_val: input_expr.outvals) {
        var_t y_bar_param = fresh_var(output_val.get_type());
        output_expr.add_invar(y_bar_param);

        if (output_val.is<literal_t>()) continue;

        auto& output_var = output_val.get_var();
        update_adjoint(output_var, value{y_bar_param});
    }

    // perform a backward pass:

    for (size_t i = output_expr.equations.size(); i--> 0;) {
        auto& eq = output_expr.equations[i];
        propagate_adjoints(eq);
    }

    // Populate outputs:

    for (auto& var: input_expr.invars) {
        if (value* val = get_adjoint(var)) {
            // The adjoint exists:
            output_expr.outvals.push_back(*val);
        } else {
            // The adjoint does not exist - replace it with 0:
            output_expr.outvals.push_back(broadcasted_value(var.get_type(), 0));
        }
    }

    return output_expr;
}


