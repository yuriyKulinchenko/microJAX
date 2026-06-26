#ifndef GRAD_H
#define GRAD_H

#include <unordered_map>

#include "jax_types.h"

class grad_class {
    friend jax::expression grad(const jax::expression& expr);
    grad_class(const jax::expression& expr);

    jax::expression find_grad();
    void introduce_adjoint(const jax::var_t& var);
    jax::value* get_adjoint(const jax::var_t& var);
    void update_adjoint(const jax::var_t& input_var, const jax::var_t& product_var);

    void propagate_adjoints(const jax::equation& eq);
    void propagate_simple_elementwise_adjoints(const jax::equation& eq);

    bool adjoint_is_active(jax::var_t var, bool apply_update=true);

    const jax::expression& input_expr;
    jax::expression output_expr;

    std::unordered_map<size_t, bool> active_adjoint_map;
    std::unordered_map<size_t, jax::value> variable_adjoint_map;
};

jax::expression grad(const jax::expression& expr);


#endif //GRAD_H
