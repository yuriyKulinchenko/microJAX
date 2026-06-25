#ifndef GRAD_H
#define GRAD_H

#include <unordered_map>

#include "jax_types.h"

class grad_class {
    friend jax::expression grad(const jax::expression& expr);
    grad_class(const jax::expression& expr);

    jax::expression find_grad();
    void introduce_adjoint(const jax::var_t& var);

    const jax::expression& input_expr;
    jax::expression output_expr;

    std::unordered_map<u32, u32> variable_adjoint_map;
    u32 var_id_index;
};

jax::expression grad(const jax::expression& expr);


#endif //GRAD_H
