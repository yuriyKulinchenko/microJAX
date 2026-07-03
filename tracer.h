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
    friend jaxpr_tracer operator/(const jaxpr_tracer&, const jaxpr_tracer&);
    friend jaxpr_tracer operator/(const jaxpr_tracer&, const jax::array_t&);
    friend jaxpr_tracer operator/(const jax::array_t&, const jaxpr_tracer&);

    [[nodiscard]] jaxpr_tracer sin() const;
    [[nodiscard]] jaxpr_tracer cos() const;
    [[nodiscard]] jaxpr_tracer exp() const;

    [[nodiscard]] jaxpr_tracer transpose(std::vector<size_t> permutation) const;
    [[nodiscard]] jaxpr_tracer reduce_sum(std::vector<size_t> axes) const;
    [[nodiscard]] jaxpr_tracer convert_element_type(jax::dtype_t dtype) const;

    [[nodiscard]] jaxpr_tracer broadcast_in_dim(
        std::vector<size_t> shape,
        std::vector<size_t> broadcast_dimensions) const;

    [[nodiscard]] jaxpr_tracer dot_general(
        const jaxpr_tracer& other,
        std::vector<size_t> left_contract,
        std::vector<size_t> right_contract,
        std::vector<size_t> left_batch,
        std::vector<size_t> right_batch) const;

private:
    jax::var_t var;
    jaxpr_builder& builder;
};

class jaxpr_builder {
public:
    jaxpr_tracer register_tracer(jax::type_t type);
    void register_output(const jaxpr_tracer& tracer);
    void register_output(const jax::value& value);
    void register_output(const jax::array_t& array);
    void register_output(const jax::var_t& var);


    template<std::convertible_to<size_t>... Args>
    jaxpr_tracer register_tracer(jax::dtype_t dtype, Args... shape) {
        return register_tracer(jax::type_t{dtype, shape...});
    }

    [[nodiscard]] jax::expression&& get_jaxpr();

    jax::expression jaxpr;
};

#endif //TRACER_H
