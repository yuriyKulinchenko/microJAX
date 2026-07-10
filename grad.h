#ifndef GRAD_H
#define GRAD_H

#include <unordered_map>

#include "jax_types.h"

class grad_class {
    friend jax::expression grad(const jax::expression& expr);
    friend jax::expression grad_general(const jax::expression& expr);
    grad_class(const jax::expression& expr);

    jax::expression find_grad();


    /*
    grad_general is the generalizable form of find_grad,
    which is able to handle multi-input, multi-output expressions
    and transform them into expressions which back-propagate adjoints
    given the output adjoints and input parameters.

    Unlike find_grad, grad_general does not bake the seed into the
    final expression. Instead, it takes it as an explicit parameter.

    For an expression mapping:
    (x0, ..., xn) -> (y0, ..., ym),

    grad_general will return a new expression mapping:
    (x0, ..., xn, y'0, ..., y'm) -> (x'0, ..., x'n)

    */

    jax::expression find_grad_general();

    void introduce_adjoint(const jax::var_t& var);
    jax::value* get_adjoint(const jax::var_t& var);

    // input_adj_var += val
    void update_adjoint(const jax::var_t& input_adj_var, const jax::value& val);

    // input_adj_var += f_prime * adjoint
    void update_adjoint(
        const jax::var_t& input_adj_var,
        const jax::value& f_prime_val,
        const jax::value& adjoint_val);

    void propagate_adjoints(const jax::equation& eq);
    bool adjoint_is_active(const jax::var_t& var, bool apply_update=true);

    jax::value negate(const jax::value& val);
    jax::value broadcasted_value(const jax::type_t& type, auto x) {
        using namespace jax;
        literal_t literal = std::invoke([&]() -> literal_t {
            switch (type.get_dtype()) {
                using enum dtype_t;
                case I32: return literal_t{I32, static_cast<i32>(x)};
                case I64: return literal_t{I64, static_cast<i64>(x)};
                case F32: return literal_t{F32, static_cast<f32>(x)};
                case F64: return literal_t{F64, static_cast<f64>(x)};
                default: return literal_t{BOOL, static_cast<b8>(x)};
            }
        });

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

    jax::var_t fresh_var(jax::type_t type);

    const jax::expression& input_expr;
    jax::expression output_expr;

    std::unordered_map<size_t, bool> active_adjoint_map;
    std::unordered_map<size_t, jax::value> variable_adjoint_map;
};

jax::expression grad(const jax::expression& expr);
jax::expression grad_general(const jax::expression& expr);


#endif //GRAD_H
