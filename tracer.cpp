#include "tracer.h"

jaxpr_tracer elementwise_binary_op(
    const jax::value& v1,
    const jax::value& v2,
    jax::primitive_op op,
    jaxpr_builder& builder
    ) {
    jax::var_t new_var {builder.new_var(), v1.get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{v1, v2},
        std::vector{new_var},
        op
    );

    return {builder, new_var};
}

jaxpr_tracer operator+(const jaxpr_tracer& t1, const jaxpr_tracer& t2) {
    if (t1.get_type() != t2.get_type()) {
        throw std::logic_error("ERROR: dimensionality mismatch when attempting elementwise operation");
    }
    using enum jax::primitive_op;
    return elementwise_binary_op(jax::value{t1.var}, jax::value{t2.var}, ADD, t1.builder);
}

jaxpr_tracer operator+(const jaxpr_tracer& t1, const jax::array_t& array) {
    if (t1.get_type() != array.get_type()) {
        throw std::logic_error("ERROR: dimensionality mismatch when attempting elementwise operation");
    }

    using enum jax::primitive_op;
    return elementwise_binary_op(jax::value{t1.var}, jax::value{array}, ADD, t1.builder);
}

jaxpr_tracer operator+(const jax::array_t& array, const jaxpr_tracer& t1) {
    if (t1.get_type() != array.get_type()) {
        throw std::logic_error("ERROR: dimensionality mismatch when attempting elementwise operation");
    }

    using enum jax::primitive_op;
    return elementwise_binary_op(jax::value{array}, jax::value{t1.var}, ADD, t1.builder);
}