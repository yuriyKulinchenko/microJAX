#include "grad.h"
#include "helper.h"

/*
============================== High level architecture of grad ==============================

'grad' is intended to provide a jaxpr -> jaxpr transformation, which takes a computation,
and returns the grad / derivative of that computation. This is done through a forward pass
followed by backpropagation.

The forward pass will essentially be a copy of the original jaxpr, and it will also be used to
initialise the adjoints for each variable introduced.

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

The current implementation of grad will not be handling nested expressions, however it is
amenable to this extension.
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

value *grad_class::get_adjoint(const var_t& var) {
    if (auto it = variable_adjoint_map.find(var.get_id());
        it != variable_adjoint_map.end()) {
        return &it->second;
    }
    return nullptr;
}

bool grad_class::adjoint_is_active(var_t var, bool apply_update) {
    auto it = active_adjoint_map.find(var.get_id());
    if (it->second) return true;
    if (apply_update) it->second = true;
    return false;
}

void grad_class::update_adjoint(const var_t& input_adj_var, const value& product_val) {
    // First, check if product is zero:
    if (product_val.is<array_t>()) {
        if (product_val.get_array().has_single_value(0)) {
            return;
        }
    }

    // Case 1: input_adj does not yet exist:
    if (!adjoint_is_active(input_adj_var)) {
        variable_adjoint_map.insert_or_assign(input_adj_var.get_id(), value{product_val});
    } else {
        // Case 2: it does exist - sum is required:
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

    bool f_prime_is_array = f_prime_val.is<array_t>();
    bool adjoint_is_array = adjoint_val.is<array_t>();

    if (f_prime_is_array && adjoint_is_array) {
        // Fold constant:
        array_t product = f_prime_val.get_array() * adjoint_val.get_array();
        if (product.has_single_value(0)) return;
        update_adjoint(input_adj_var, value{std::move(product)});
    } else if (f_prime_is_array) {
        const array_t& f_prime = f_prime_val.get_array();
        if (f_prime.has_single_value(0)) {
            // If f_prime is 0, return early
            return;
        }
        if (f_prime.has_single_value(1)) {
            // If f_prime is 1, only propagate the adjoint
            update_adjoint(input_adj_var, adjoint_val);
        } else if (f_prime.has_single_value(-1)) {
            // If f_prime is -1, apply negation
            auto negated_var = fresh_var(input_adj_var.get_type());
            output_expr.equations.emplace_back(
                std::vector{adjoint_val},
                std::vector{negated_var},
                primitive_op::NEG
            );
            update_adjoint(input_adj_var, value{negated_var});
        } else {
            // Otherwise, actually multiply
            update_adjoint(input_adj_var, f_prime_val, adjoint_val);
        }

    } else if (adjoint_is_array) {
        const array_t& adjoint = adjoint_val.get_array();
        if (adjoint.has_single_value(0)) {
            return;
        }
        if (adjoint.has_single_value(1)) {
            update_adjoint(input_adj_var, f_prime_val);
        } else if (adjoint.has_single_value(-1)) {
            auto negated_var = fresh_var(input_adj_var.get_type());
            output_expr.equations.emplace_back(
                std::vector{f_prime_val},
                std::vector{negated_var},
                primitive_op::NEG
            );
            update_adjoint(input_adj_var, value{negated_var});
        } else {
            update_adjoint(input_adj_var, f_prime_val, adjoint_val);
        }

    } else {
        auto product_var = fresh_var(input_adj_var.get_type());

        output_expr.equations.emplace_back(
            std::vector{f_prime_val, adjoint_val},
            std::vector{product_var},
            primitive_op::MUL
        );

        update_adjoint(input_adj_var, value{product_var});
    }
}

bool all_inputs_constant(const std::vector<value>& inputs) {
    for (auto& input: inputs) if (input.is<var_t>()) return false;
    return true;
}

void grad_class::propagate_adjoints(const equation& eq) {
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

        case NEG: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();
            auto f_prime = array_t::build_fill(output_var.get_type(), -1); // -1

            update_adjoint(input_var, value{f_prime}, *output_adj);

            break;
        }

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
                if (y_val.is<array_t>() && y_val.get_array().has_single_value(1)) {

                    // If y_val is a 1 vector, the division can be omitted

                    update_adjoint(x_val.get_var(), *output_adj);
                } else {
                    auto quotient_val = fresh_var(x_val.get_type());

                    output_expr.equations.emplace_back(
                        std::vector{*output_adj, y_val},
                        std::vector{quotient_val},
                        DIV
                    );

                    update_adjoint(x_val.get_var(), value{quotient_val});
                }
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

                update_adjoint(y_val.get_var(), value{negated_quotient_val});
            }

            break;
        }

        case REDUCE_SUM: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            // If output_adj is 0, there will be no contribution to the adjoint:
            if (output_adj->is<array_t>()) {
                auto& output_adj_array = output_adj->get_array();
                if (output_adj_array.has_single_value(0)) break;
            }

            auto& input_val = eq.get_input(0);
            if (input_val.is<array_t>()) break;
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

            // If output_adj is 0, there will be no contribution to the adjoint:
            if (output_adj->is<array_t>()) {
                auto& output_adj_array = output_adj->get_array();
                if (output_adj_array.has_single_value(0)) break;
            }

            auto& input_val = eq.get_input(0);
            if (input_val.is<array_t>()) break;
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

            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& x_val = eq.get_input(0);
            auto& y_val = eq.get_input(1);
            auto& x_shape = x_val.get_type().get_shape();
            auto& y_shape = y_val.get_type().get_shape();

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

            if (x_val.is<var_t>()) {
                auto& x_var = x_val.get_var();

                std::vector<size_t> inverse_permutation =
                    get_inverse_permutation(params.left_batch, params.left_contract, free_indices_x);

                type_t new_type {x_var.get_type().get_dtype(), permute(x_shape, inverse_permutation)};

                auto dot_general_x = fresh_var(std::move(new_type));

                output_expr.equations.emplace_back(
                    std::vector{y_val, *output_adj},
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
            }

            if (y_val.is<var_t>()) {
                auto& y_var = y_val.get_var();

                std::vector<size_t> inverse_permutation =
                    get_inverse_permutation(params.right_batch, params.right_contract, free_indices_y);

                type_t new_type {y_var.get_type().get_dtype(), permute(y_shape, inverse_permutation)};

                auto dot_general_y = fresh_var(std::move(new_type));

                output_expr.equations.emplace_back(
                    std::vector{x_val, *output_adj},
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

        default: {
            throw std::logic_error{"Error: Not implemented"};
        }
    }
}

expression grad_class::find_grad() {
    if (input_expr.outvals.size() != 1) {
        throw formatted_error("Error: expected 1 output, received {}", input_expr.outvals.size());
    }

    if (input_expr.outvals[0].is<array_t>()) {
        throw formatted_error("Error: expected output to be variable, received array");
    }

    const var_t& output_var = input_expr.outvals[0].get_var();

    if (output_var.get_type().get_shape().size() != 0) {
        throw formatted_error("Error: shape of output variable must be scalar");
    }

    // Add inputs:

    for (auto& var: input_expr.invars) {
        output_expr.add_input(var);
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

    update_adjoint(output_var, value{array_t::build_fill(type_t{dtype_t::F32}, 1)});

    // perform a backward pass:

    for (auto& eq: input_expr.equations | std::views::reverse) {
        propagate_adjoints(eq);
    }

    // Populate outputs:

    for (auto& var: input_expr.invars) {
        if (value* val = get_adjoint(var)) {
            // The adjoint exists:
            output_expr.outvals.push_back(*val);
        } else {
            // The adjoint does not exist - replace it with 0:
            output_expr.outvals.push_back(value{array_t::build_fill(var.get_type(), 0)});
        }
    }

    return output_expr;
}

expression grad_class::find_grad_general() {
    // Add inputs (x0, ..., xn)

    for (auto& var: input_expr.invars) {
        output_expr.add_input(var);
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
        output_expr.add_input(y_bar_param);

        if (output_val.is<array_t>()) continue;

        auto& output_var = output_val.get_var();
        update_adjoint(output_var, value{y_bar_param});
    }

    // perform a backward pass:

    for (auto& eq: input_expr.equations | std::views::reverse) {
        propagate_adjoints(eq);
    }

    // Populate outputs:

    for (auto& var: input_expr.invars) {
        if (value* val = get_adjoint(var)) {
            // The adjoint exists:
            output_expr.outvals.push_back(*val);
        } else {
            // The adjoint does not exist - replace it with 0:
            output_expr.outvals.push_back(value{array_t::build_fill(var.get_type(), 0)});
        }
    }

    return output_expr;
}


