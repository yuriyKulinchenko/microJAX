#include "tracer.h"

jaxpr_tracer elementwise_binary_op(
    const jax::value& v1,
    const jax::value& v2,
    jax::primitive_op op,
    jaxpr_builder& builder
    ) {
    jax::var_t new_var {builder.jaxpr.new_var_id(), v1.get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{v1, v2},
        std::vector{new_var},
        op
    );

    return {builder, new_var};
}

jaxpr_tracer unary_op(const jax::value& val, jax::primitive_op op, jaxpr_builder& builder) {
    jax::var_t new_var {builder.jaxpr.new_var_id(), val.get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{val},
        std::vector{new_var},
        op
    );

    return {builder, new_var};
}


#define DIMENSIONALITY_ERROR(v1, v2)                                                                \
if(v1.get_type() != v2.get_type()) {                                                                \
    throw std::logic_error("ERROR: dimensionality mismatch when attempting elementwise operation"); \
}                                                                                                   \

#define ELEMENTWISE_BINARY_OP_TRACER_TRACER(op, op_name)                                        \
jaxpr_tracer operator op (const jaxpr_tracer& t1, const jaxpr_tracer& t2) {                     \
    DIMENSIONALITY_ERROR(t1, t2);                                                               \
    return elementwise_binary_op(jax::value{t1.var}, jax::value{t2.var}, op_name, t1.builder);  \
}                                                                                               \

#define ELEMENTWISE_BINARY_OP_TRACER_ARRAY(op, op_name)                                         \
jaxpr_tracer operator op (const jaxpr_tracer& t1, const jax::array_t& array) {                  \
    DIMENSIONALITY_ERROR(t1, array);                                                            \
    return elementwise_binary_op(jax::value{t1.var}, jax::value{array}, op_name, t1.builder);   \
}                                                                                               \


#define ELEMENTWISE_BINARY_OP_ARRAY_TRACER(op, op_name)                                         \
jaxpr_tracer operator op (const jax::array_t& array, const jaxpr_tracer& t1) {                  \
    DIMENSIONALITY_ERROR(array, t1);                                                            \
    return elementwise_binary_op(jax::value{array}, jax::value{t1.var}, op_name, t1.builder);   \
}                                                                                               \

#define ELEMENTWISE_BINARY_OP(op, op_name)          \
ELEMENTWISE_BINARY_OP_TRACER_TRACER(op, op_name)    \
ELEMENTWISE_BINARY_OP_TRACER_ARRAY(op, op_name)     \
ELEMENTWISE_BINARY_OP_ARRAY_TRACER(op, op_name)     \

ELEMENTWISE_BINARY_OP(+, jax::primitive_op::ADD);
ELEMENTWISE_BINARY_OP(*, jax::primitive_op::MUL);
ELEMENTWISE_BINARY_OP(-, jax::primitive_op::SUB);

jaxpr_tracer jaxpr_tracer::sin() const {
    return unary_op(jax::value{var}, jax::primitive_op::SIN, builder);
}

jaxpr_tracer jaxpr_tracer::cos() const {
    return unary_op(jax::value{var}, jax::primitive_op::COS, builder);
}

jaxpr_tracer jaxpr_tracer::exp() const {
    return unary_op(jax::value{var}, jax::primitive_op::EXP, builder);
}

jaxpr_tracer jaxpr_builder::register_tracer(jax::type_t type) {
    auto var = jax::var_t{jaxpr.new_var_id(), std::move(type)};
    jaxpr.invars.push_back(var);
    return {*this, std::move(var)};
}

void jaxpr_builder::register_output(const jaxpr_tracer& tracer) {
    jaxpr.outvals.push_back(jax::value{tracer.get_var()});
}

void jaxpr_builder::register_output(const jax::value& value) {
    jaxpr.outvals.push_back(value);
}

void jaxpr_builder::register_output(const jax::array_t& array) {
    jaxpr.outvals.push_back(jax::value{array});
}

void jaxpr_builder::register_output(const jax::var_t& var) {
    jaxpr.outvals.push_back(jax::value{var});
}

jax::expression&& jaxpr_builder::get_jaxpr() {
    return std::move(jaxpr);
}
