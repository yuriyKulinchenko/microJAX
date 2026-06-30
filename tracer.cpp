#include "tracer.h"

#include <unordered_set>

#include "jax_functions.h"

using namespace jax;

jaxpr_tracer elementwise_binary_op(
    const value& v1,
    const value& v2,
    primitive_op op,
    jaxpr_builder& builder
    ) {
    var_t new_var {builder.jaxpr.new_var_id(), v1.get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{v1, v2},
        std::vector{new_var},
        op
    );

    return {builder, new_var};
}

jaxpr_tracer unary_op(const value& val, primitive_op op, jaxpr_builder& builder) {
    var_t new_var {builder.jaxpr.new_var_id(), val.get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{val},
        std::vector{new_var},
        op
    );

    return {builder, new_var};
}

jaxpr_tracer unary_op(const value& val, type_t type, primitive_op op,
    params_variant params, jaxpr_builder& builder) {
    var_t new_var {builder.jaxpr.new_var_id(), std::move(type)};

    builder.jaxpr.equations.emplace_back(
        std::vector{val},
        std::vector{new_var},
        op,
        params
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
    return unary_op(value{var}, primitive_op::SIN, builder);
}

jaxpr_tracer jaxpr_tracer::cos() const {
    return unary_op(value{var}, primitive_op::COS, builder);
}

jaxpr_tracer jaxpr_tracer::exp() const {
    return unary_op(value{var}, primitive_op::EXP, builder);
}

jaxpr_tracer jaxpr_tracer::transpose(std::vector<size_t> permutation) const {
    // The transpose must actually be possible:

    const auto& old_dimension = var.get_type().get_dimension();
    if (old_dimension.size() != permutation.size()) {
        throw std::logic_error("Error: shape of argument is invalid for transpose");
    }

    if (!valid_permutation(permutation)) {
        throw std::logic_error("Error: permutation in transpose is invalid");
    }

    // The new type dimensions have to be calculated:

    std::vector<size_t> new_dimension(old_dimension.size());
    for (size_t i = 0; i < permutation.size(); i++) {
        new_dimension[i] = old_dimension[permutation[i]];
    }

    type_t new_type {var.get_type().get_base_type(), std::move(new_dimension)};



    return unary_op(value{var}, std::move(new_type),  primitive_op::TRANSPOSE,
        transpose_params{std::move(permutation)}, builder);
}

jaxpr_tracer jaxpr_tracer::reduce_sum(std::vector<size_t> axes) const {
    // Axes must be valid:

    const auto& old_dimension = var.get_type().get_dimension();
    std::vector excluded(old_dimension.size(), false);
    for (auto axis: axes) {
        if (axis >= old_dimension.size()) {
            throw formatted_error(
                "Error: axis {} does not exist for the given argument in reduce_sum", axis);
        }
        if (excluded[axis]) {
            throw formatted_error(
                "Error: axis {} is included more than once in reduce_sum", axis);
        }
        excluded[axis] = true;
    }

    // Apply new shape:

    std::vector<size_t> new_dimension(old_dimension.size() - axes.size());
    for (size_t i = 0, j = 0; i < old_dimension.size(); i++) {
        if (excluded[i]) continue;
        new_dimension[j++] = old_dimension[i];
    }

    type_t new_type {var.get_type().get_base_type(), std::move(new_dimension)};

    return unary_op(value{var}, std::move(new_type),  primitive_op::REDUCE_SUM,
        reduce_sum_params{axes}, builder);

}

jaxpr_tracer jaxpr_tracer::dot_general(
    std::vector<size_t> left_contract, std::vector<size_t> right_contract,
    std::vector<size_t> left_batch, std::vector<size_t> right_batch) const {
    // TODO: implement this
    return {builder, var};
}






jaxpr_tracer jaxpr_builder::register_tracer(type_t type) {
    auto var = var_t{jaxpr.new_var_id(), std::move(type)};
    jaxpr.invars.push_back(var);
    return {*this, std::move(var)};
}

void jaxpr_builder::register_output(const jaxpr_tracer& tracer) {
    jaxpr.outvals.push_back(value{tracer.get_var()});
}

void jaxpr_builder::register_output(const value& value) {
    jaxpr.outvals.push_back(value);
}

void jaxpr_builder::register_output(const array_t& array) {
    jaxpr.outvals.push_back(value{array});
}

void jaxpr_builder::register_output(const var_t& var) {
    jaxpr.outvals.push_back(value{var});
}

expression&& jaxpr_builder::get_jaxpr() {
    return std::move(jaxpr);
}
