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

jax::expression grad(const jax::expression& expr) {
    // For now, take the seed to be f32:
    using namespace jax;
    auto instance = grad_class{expr};
    return instance.find_grad(value{array_t::build_fill(type_t{type_enum::F32}, 1)});
}

grad_class::grad_class(const jax::expression& expr): input_expr(expr) {
    output_expr.var_id = expr.var_id;
}

jax::var_t grad_class::fresh_var(jax::type_t type) {
    return jax::var_t{output_expr.new_var_id(), std::move(type)};
}


void grad_class::introduce_adjoint(const jax::var_t& var) {
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

jax::value *grad_class::get_adjoint(const jax::var_t& var) {
    if (auto it = variable_adjoint_map.find(var.get_id());
        it != variable_adjoint_map.end()) {
        return &it->second;
    }
    return nullptr;
}

bool grad_class::adjoint_is_active(jax::var_t var, bool apply_update) {
    auto it = active_adjoint_map.find(var.get_id());
    if (it->second) return true;
    if (apply_update) it->second = true;
    return false;
}

void grad_class::update_adjoint(const jax::var_t& input_adj_var, const jax::value& product_val) {
    // First, check if product is zero:
    if (product_val.is<jax::array_t>()) {
        if (product_val.get_array().has_single_value(0)) {
            return;
        }
    }

    // Case 1: input_adj does not yet exist:
    if (!adjoint_is_active(input_adj_var)) {
        variable_adjoint_map.insert_or_assign(input_adj_var.get_id(), jax::value{product_val});
    } else {
        // Case 2: it does exist - sum is required:
        auto sum_var = jax::var_t{output_expr.new_var_id(), product_val.get_type()};
        const auto& input_adj = *get_adjoint(input_adj_var);

        output_expr.equations.emplace_back(
            std::vector{product_val, input_adj},
            std::vector{sum_var},
            jax::primitive_op::ADD
        );

        // Carry through the updated adjoint:
        variable_adjoint_map.insert_or_assign(input_adj_var.get_id(), jax::value{sum_var});

    }
}

void grad_class::update_adjoint(
    const jax::var_t& input_adj_var,
    const jax::value& f_prime_val,
    const jax::value& adjoint_val) {

    bool f_prime_is_array = f_prime_val.is<jax::array_t>();
    bool adjoint_is_array = adjoint_val.is<jax::array_t>();

    if (f_prime_is_array && adjoint_is_array) {
        // Fold constant:
        jax::array_t product = f_prime_val.get_array() * adjoint_val.get_array();
        if (product.has_single_value(0)) return;
        update_adjoint(input_adj_var, jax::value{std::move(product)});
    } else if (f_prime_is_array) {
        const jax::array_t& f_prime = f_prime_val.get_array();
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
                jax::primitive_op::NEG
            );
            update_adjoint(input_adj_var, jax::value{negated_var});
        } else {
            // Otherwise, actually multiply
            update_adjoint(input_adj_var, f_prime_val, adjoint_val);
        }

    } else if (adjoint_is_array) {
        const jax::array_t& adjoint = adjoint_val.get_array();
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
                jax::primitive_op::NEG
            );
            update_adjoint(input_adj_var, jax::value{negated_var});
        } else {
            update_adjoint(input_adj_var, f_prime_val, adjoint_val);
        }

    } else {
        auto product_var = fresh_var(input_adj_var.get_type());

        output_expr.equations.emplace_back(
            std::vector{f_prime_val, adjoint_val},
            std::vector{product_var},
            jax::primitive_op::MUL
        );

        update_adjoint(input_adj_var, jax::value{product_var});
    }
}

bool all_inputs_constant(const std::vector<jax::value>& inputs) {
    for (auto& input: inputs) if (input.is<jax::var_t>()) return false;
    return true;
}

void grad_class::propagate_adjoints(const jax::equation& eq) {
    // Requires knowledge of output adjoints, and shape of eq.op to dispatch over
    // If an adjoint does not exist, take it to be zero

    // Check if there is anything to propagate to:
    if (all_inputs_constant(eq.get_input())) return;

    switch (eq.get_op()) {
        using enum jax::primitive_op;
        case SIN: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            auto f_prime_var = jax::var_t{output_expr.new_var_id(), output_var.get_type()};

            output_expr.equations.emplace_back(
                std::vector{jax::value{input_var}},
                std::vector{f_prime_var},
                COS
            );

            update_adjoint(input_var, jax::value{f_prime_var}, *output_adj);

            break;
        }

        case COS: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();

            auto var1 = jax::var_t{output_expr.new_var_id(), output_var.get_type()};

            output_expr.equations.emplace_back(
                std::vector{jax::value{input_var}},
                std::vector{var1},
                SIN
            );

            auto f_prime_var = jax::var_t{output_expr.new_var_id(), output_var.get_type()};

            output_expr.equations.emplace_back(
                std::vector{jax::value{var1}},
                std::vector{f_prime_var},
                NEG
            );

            update_adjoint(input_var, jax::value{f_prime_var}, *output_adj);

            break;
        }

        case NEG: {
            auto& output_var = eq.get_output(0);
            auto* output_adj = get_adjoint(output_var);
            if (!output_adj) break;

            auto& input_var = eq.get_input(0).get_var();
            auto f_prime = jax::array_t::build_fill(output_var.get_type(), -1); // -1

            update_adjoint(input_var, jax::value{f_prime}, *output_adj);

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

            if (x_val.is<jax::var_t>()) {
                update_adjoint(x_val.get_var(), *output_adj);
            }

            if (y_val.is<jax::var_t>()) {
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

            if (x_val.is<jax::var_t>()) {
                update_adjoint(x_val.get_var(), y_val, *output_adj);
            }

            if (y_val.is<jax::var_t>()) {
                update_adjoint(y_val.get_var(), x_val, *output_adj);
            }

            break;
        }

        default: {
            throw std::logic_error{"Error: Not implemented"};
        }
    }
}

jax::expression grad_class::find_grad(jax::value seed) {
    if (input_expr.outvals.size() != 1) {
        throw formatted_error("Error: expected 1 output, received {}", input_expr.outvals.size());
    }

    if (input_expr.outvals[0].is<jax::array_t>()) {
        throw formatted_error("Error: expected output to be variable, received array");
    }

    const jax::var_t& output_var = input_expr.outvals[0].get_var();

    if (output_var.get_type().get_dimension().size() != 0) {
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

    update_adjoint(output_var, seed);

    // perform a backward pass:

    for (auto& eq: input_expr.equations | std::views::reverse) {
        propagate_adjoints(eq);
    }

    // Populate outputs:

    for (auto& var: input_expr.invars) {
        jax::value* val = get_adjoint(var);
        if (val) {
            // The adjoint exists:
            output_expr.outvals.push_back(*val);
        } else {
            // The adjoint does not exist - replace it with 0:
            output_expr.outvals.push_back(jax::value{jax::array_t::build_fill(var.get_type(), 0)});
        }
    }

    return output_expr;
}


