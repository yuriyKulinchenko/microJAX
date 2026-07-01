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

    return {builder, std::move(new_var)};
}

jaxpr_tracer unary_op(const value& val, primitive_op op, jaxpr_builder& builder) {
    var_t new_var {builder.jaxpr.new_var_id(), val.get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{val},
        std::vector{new_var},
        op
    );

    return {builder, std::move(new_var)};
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

    return {builder, std::move(new_var)};
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

    const auto& old_shape = var.get_type().get_shape();
    if (old_shape.size() != permutation.size()) {
        throw std::logic_error("Error: shape of argument is invalid for transpose");
    }

    if (!valid_permutation(permutation)) {
        throw std::logic_error("Error: permutation in transpose is invalid");
    }

    // The new type shape has to be calculated:

    std::vector<size_t> new_shape(old_shape.size());
    for (size_t i = 0; i < permutation.size(); i++) {
        new_shape[i] = old_shape[permutation[i]];
    }

    type_t new_type {var.get_type().get_base_type(), std::move(new_shape)};



    return unary_op(value{var}, std::move(new_type),  primitive_op::TRANSPOSE,
        transpose_params{std::move(permutation)}, builder);
}

jaxpr_tracer jaxpr_tracer::reduce_sum(std::vector<size_t> axes) const {
    // Axes must be valid:

    const auto& old_shape = var.get_type().get_shape();
    std::ranges::sort(axes);

    size_t max = 0;
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] >= old_shape.size()) {
            throw formatted_error(
                    "Error: axis {} does not exist for the given argument in reduce_sum", axes[i]);
        }
        if (i > 0 && axes[i] == max) {
            throw formatted_error(
                    "Error: axis {} is included more than once in reduce_sum", axes[i]);
        }
        max = axes[i];
    }

    // Apply new shape:

    std::vector<size_t> new_shape(old_shape.size() - axes.size());
    for (size_t i = 0, j = 0, k = 0; i < old_shape.size(); i++) {
        if (k < axes.size() && axes[k] == i) { k++; continue; }
        new_shape[j++] = old_shape[i];
    }

    type_t new_type {var.get_type().get_base_type(), std::move(new_shape)};

    return unary_op(value{var}, std::move(new_type), primitive_op::REDUCE_SUM,
        reduce_sum_params{std::move(axes)}, builder);
}

jaxpr_tracer jaxpr_tracer::broadcast_in_dim(std::vector<size_t> shape,
    std::vector<size_t> broadcast_dimensions) const {
    // If shape(x) = (2, 3)
    // if broadcast_dimensions=(0, 2), shape=(2,4,3) then:
    // y_i0,i1,i2 = x_i0,i2

    // broadcast_dimensions have to match:
    const auto& old_shape = var.get_type().get_shape();
    if (old_shape.size() != broadcast_dimensions.size()) {
        throw formatted_error("Error: expected {} axes in broadcast_dimensions, got {}",
            old_shape.size(), broadcast_dimensions.size());
    }

    // Enforce that broadcast_dimensions is strictly increasing, with no duplicates,
    // and that the sizes of mapped axes match

    if (broadcast_dimensions.size() != 0) {
        size_t max = broadcast_dimensions[0];
        for (size_t i = 1; i < broadcast_dimensions.size(); i++) {

            if (broadcast_dimensions[i] <= max) {
                throw std::logic_error(
                    "Error: broadcast_dimensions must be strictly increasing with no duplicates");
            }

            // The mapping must be valid:
            if (broadcast_dimensions[i] >= shape.size() ||
                !(old_shape[i] == shape[broadcast_dimensions[i]] || old_shape[i] == 1)) {
                throw std::logic_error("Error: axes are incompatible in broadcast_dim");
            }

            max = broadcast_dimensions[i];
        }
    }

    type_t new_type {var.get_type().get_base_type(), shape};
    return unary_op(
        value{var}, std::move(new_type), primitive_op::BROADCAST_IN_DIM,
        broadcast_in_dim_params{
            std::move(shape),
            std::move(broadcast_dimensions)},
            builder);
}

jaxpr_tracer jaxpr_tracer::dot_general(
    const jaxpr_tracer& other,
    std::vector<size_t> left_contract, std::vector<size_t> right_contract,
    std::vector<size_t> left_batch, std::vector<size_t> right_batch) const {

    const auto& this_shape = var.get_type().get_shape();
    const auto& other_shape = other.get_type().get_shape();

    // left and right lists have to match:

    if (left_contract.size() != right_contract.size()) {
        throw formatted_error(
            "Error: the left and right contraction lists should have equal length,"
            "Instead they have lengths {} and {} respectively",
            left_contract.size(), right_contract.size());
    }

    if (left_batch.size() != right_batch.size()) {
        throw formatted_error(
            "Error: the left and right batch lists should have equal length,"
            "Instead they have lengths {} and {} respectively",
            left_contract.size(), right_contract.size());
    }

    for (size_t i = 0; i < left_contract.size(); i++) {
        if (this_shape[left_contract[i]] != other_shape[right_contract[i]]) {
            throw std::logic_error(
                "Error: rank size mismatch in contraction indices"
            );
        }
    }

    for (size_t i = 0; i < left_contract.size(); i++) {
        if (this_shape[left_batch[i]] != other_shape[right_batch[i]]) {
            throw std::logic_error(
                "Error: rank size mismatch in batch indices"
            );
        }
    }

    // contract and batch indices cannot overlap
    // left_contract, left_batch have to be disjoint,
    // right_contract, right batch have to be disjoint:

    std::unordered_set<size_t> left_batch_set {left_batch.begin(), left_batch.end()};
    for (size_t contract_index: left_contract) {
        if (left_batch_set.contains(contract_index)) {
            throw formatted_error(
                "Error: left batch and left contract lists have shared index {}",
                contract_index
            );
        }
    }

    std::unordered_set<size_t> right_batch_set {right_batch.begin(), right_batch.end()};
    for (size_t contract_index: right_contract) {
        if (right_batch_set.contains(contract_index)) {
            throw formatted_error(
                "Error: right batch and right contract lists have shared index {}",
               contract_index
            );
        }
    }

    // Calculate the new shape, which will be of the form (batch, left free, right free):

    size_t new_size =
        this_shape.size() + other_shape.size() - left_batch.size() - 2 * left_contract.size();

    std::vector<size_t> new_shape {};
    new_shape.reserve(new_size);

    // Add batch:

    for (auto batch_dim: left_batch) {
        new_shape.push_back(this_shape[batch_dim]);
    }

    // Add left free:

    auto left_contract_complement = complement(left_contract, this_shape.size());
    for (auto free_dim: left_contract_complement) {
        if (!left_batch_set.contains(free_dim)) new_shape.push_back(this_shape[free_dim]);
    }

    // Add right free:

    auto right_contract_complement = complement(right_contract, other_shape.size());
    for (auto free_dim: right_contract_complement) {
        if (!right_batch_set.contains(free_dim)) new_shape.push_back(other_shape[free_dim]);
    }

    type_t new_type {var.get_type().get_base_type(), new_shape};

    var_t new_var {builder.jaxpr.new_var_id(), std::move(new_type)};

    builder.jaxpr.equations.emplace_back(
        std::vector{value{var}, value{other.var}},
        std::vector{new_var},
        primitive_op::DOT_GENERAL,
        dot_general_params{
            std::move(left_contract),
            std::move(right_contract),
            std::move(left_batch),
            std::move(right_batch)
        }
    );

    return {builder, std::move(new_var)};
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
