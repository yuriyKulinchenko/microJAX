#ifndef TRACER_H
#define TRACER_H

#include <type_traits>
#include <utility>

#include "jax_types.h"

class jaxpr_builder {
public:
    u32 new_var() {
        return var_count++;
    }

    jax::expression jaxpr;
    u32 var_count = 0;
};

class jaxpr_tracer {
public:
    jaxpr_tracer(jaxpr_builder& builder, jax::var_t var):
    var(std::move(var)),
    builder(builder) {}

    [[nodiscard]] const jax::type_t& get_type() const {
        return var.get_type();
    }

    friend jaxpr_tracer operator+(const jaxpr_tracer&, const jaxpr_tracer&);
    friend jaxpr_tracer operator+(const jaxpr_tracer&, const jax::array_t&);
    friend jaxpr_tracer operator+(const jax::array_t&, const jaxpr_tracer&);
    friend jaxpr_tracer operator-(const jaxpr_tracer&, const jaxpr_tracer&);
    friend jaxpr_tracer operator-(const jaxpr_tracer&, const jax::array_t&);
    friend jaxpr_tracer operator-(const jax::array_t&, const jaxpr_tracer&);
    friend jaxpr_tracer operator*(const jaxpr_tracer&, const jaxpr_tracer&);
    friend jaxpr_tracer operator*(const jaxpr_tracer&, const jax::array_t&);
    friend jaxpr_tracer operator*(const jax::array_t&, const jaxpr_tracer&);

    jaxpr_tracer sin();
    jaxpr_tracer cos();
    jaxpr_tracer exp();
private:
    jax::var_t var;
    jaxpr_builder& builder;
};

#endif //TRACER_H
