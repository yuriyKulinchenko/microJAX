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
    variable_adjoint_map[var.get_id()] = output_expr.new_var_id();
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

    return output_expr;
}




