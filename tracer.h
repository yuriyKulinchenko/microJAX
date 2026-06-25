#ifndef TRACER_H
#define TRACER_H

#include <type_traits>
#include <utility>

#include "jax_types.h"

class jaxpr_tracer;
class jaxpr_builder;

class jaxpr_tracer {
public:
    jaxpr_tracer(jaxpr_builder& builder, jax::var_t var):
    var(std::move(var)),
    builder(builder) {}

    [[nodiscard]] const jax::type_t& get_type() const {
        return var.get_type();
    }

    [[nodiscard]] const jax::var_t& get_var() const {
        return var;
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

    [[nodiscard]] jaxpr_tracer sin() const;
    [[nodiscard]] jaxpr_tracer cos() const;
    [[nodiscard]] jaxpr_tracer exp() const;
private:
    jax::var_t var;
    jaxpr_builder& builder;
};

class jaxpr_builder {
public:
    u32 new_var_id();

    jaxpr_tracer register_tracer(jax::type_t type);
    void register_output(const jaxpr_tracer& tracer);
    void register_output(const jax::value& value);
    void register_output(const jax::array_t& array);
    void register_output(const jax::var_t& var);


    template<std::convertible_to<u32>... Args>
    jaxpr_tracer register_tracer(jax::type_enum base_type, Args... dimension) {
        return register_tracer(jax::type_t{base_type, dimension...});
    }

    [[nodiscard]] jax::expression&& get_jaxpr();

    jax::expression jaxpr;
    u32 var_count = 0;
};

#endif //TRACER_H
