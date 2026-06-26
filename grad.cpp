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
    auto instance = grad_class{expr};
    return instance.find_grad();
}

grad_class::grad_class(const jax::expression& expr): input_expr(expr) {
    output_expr.var_id = expr.var_id;
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

bool grad_class::adjoint_is_active(jax::var_t var, constexpr bool apply_update) {
    auto it = active_adjoint_map.find(var.get_id());
    if (it->second) return true;
    if (apply_update) it->second = true;
    return false;
}

void grad_class::update_adjoint(const jax::var_t& input_var, const jax::var_t& product_var) {
    // Case 1: input_adj does not yet exist:
    if (!adjoint_is_active(input_var)) {
        variable_adjoint_map[input_var.get_id()] = jax::value{product_var};
    } else {
        // Case 2: it does exist - sum is required:
        auto sum_var = jax::var_t{output_expr.new_var_id(), product_var.get_type()};
        const auto& input_adj = *get_adjoint(input_var);

        output_expr.equations.emplace_back(
            std::vector{jax::value{product_var}, input_adj},
            std::vector{sum_var},
            jax::primitive_op::ADD
        );

        // Carry through the updated adjoint:
        variable_adjoint_map[input_var.get_id()] = jax::value{sum_var};

    }
}


void grad_class::propagate_simple_elementwise_adjoints(const jax::equation& eq) {

}


void grad_class::propagate_adjoints(const jax::equation& eq) {
    // Requires knowledge of output adjoints, and shape of eq.op to dispatch over
    // If an adjoint does not exist, take it to be zero

    switch (eq.get_op()) {
        using enum jax::primitive_op;
        case SIN: {
            // Check if there is anything to propagate to:
            if (eq.get_input(0).is<jax::array_t>()) return;
            auto& input_var = eq.get_input(0).get_var();
            auto& output_var = eq.get_output(0);

            if (const auto* output_adj = get_adjoint(output_var)) {
                // First, create an equation that computes f'(input):

                auto f_prime_var = jax::var_t{output_expr.new_var_id(), output_var.get_type()};

                output_expr.equations.emplace_back(
                    std::vector{jax::value{input_var}},
                    std::vector{f_prime_var},
                    COS
                );

                // Then, compute f'(input) * output_adj:

                auto product_var = jax::var_t{output_expr.new_var_id(), output_var.get_type()};

                output_expr.equations.emplace_back(
                    std::vector{jax::value{f_prime_var}, *output_adj},
                    std::vector{product_var},
                    MUL
                );

                update_adjoint(input_var, product_var);
            }
            break;
        }

        case COS: {
            // Check if there is anything to propagate to:
            if (eq.get_input(0).is<jax::array_t>()) return;
            auto& input_var = eq.get_input(0).get_var();
            auto& output_var = eq.get_output(0);

            if (const auto* output_adj = get_adjoint(output_var)) {
                // First, create an equation that computes f'(input):

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

                // Then, compute f'(input) * output_adj:

                auto product_var = jax::var_t{output_expr.new_var_id(), output_var.get_type()};

                output_expr.equations.emplace_back(
                    std::vector{jax::value{f_prime_var}, *output_adj},
                    std::vector{product_var},
                    MUL
                );

                update_adjoint(input_var, product_var);
            }
            break;
        }

        case ADD: {
            break;
        }

        case MUL: {
            break;
        }

        default: {
            throw std::logic_error{"Error: Not implemented"};
        }
    }
}

jax::expression grad_class::find_grad() {
    if (input_expr.outvals.size() != 1) {
        throw formatted_error("Error: expected 1 output, received {}", input_expr.outvals.size());
    }

    if (input_expr.outvals[0].get_type().get_dimension().size() != 0) {
        throw formatted_error("Error: shape of output must be scalar");
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


    // perform a backward pass:
    // TODO: special handling is required to seed the initial equation

    for (auto& eq: input_expr.equations | std::views::reverse) {
        propagate_adjoints(eq);
    }

    return output_expr;
}




