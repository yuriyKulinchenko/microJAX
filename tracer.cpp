#include "tracer.h"

#include <optional>
#include <unordered_set>

#include "jax_functions.h"

using namespace jax;


value array_value(const array_t& array, jaxpr_builder& builder) {
    if (const std::optional<literal_t> literal = array.get_literal()) {
        return value{*literal};
    }


    // Otherwise, add it to the array of consts:
    builder.jaxpr.consts.push_back(array);
    var_t fresh_var = builder.jaxpr.fresh_var(array.get_type());
    builder.jaxpr.constvars.push_back(fresh_var);
    return value{std::move(fresh_var)};
}


jaxpr_tracer elementwise_binary_op(
    const value& v1,
    const value& v2,
    primitive_op op,
    jaxpr_builder& builder,
    std::optional<dtype_t> output_dtype = std::nullopt
    ) {
    auto left_shape = v1.get_type().get_shape();
    auto right_shape = v2.get_type().get_shape();

    dtype_t result_dtype = output_dtype.value_or(v1.get_type().get_dtype());

    if (left_shape != right_shape) {
        // A reshape must occur:
        auto result = get_implicit_broadcast_result(left_shape, right_shape);
        type_t new_type{v1.get_type().get_dtype(), result.new_shape};
        type_t result_type{result_dtype, result.new_shape};

        auto get_left_broadcast_var = [&]() -> var_t {
            var_t v1_broadcast {builder.jaxpr.new_var_id(), new_type};

            builder.jaxpr.equations.emplace_back(
                std::vector{v1},
                std::vector{v1_broadcast},
                primitive_op::BROADCAST_IN_DIM,
                broadcast_in_dim_params{result.new_shape,
                    std::move(result.left_broadcast_dimensions)}
            );

            return v1_broadcast;
        };

        auto get_right_broadcast_var = [&]() -> var_t {
            var_t v2_broadcast {builder.jaxpr.new_var_id(), new_type};

            builder.jaxpr.equations.emplace_back(
                std::vector{v2},
                std::vector{v2_broadcast},
                primitive_op::BROADCAST_IN_DIM,
                broadcast_in_dim_params{result.new_shape,
                    std::move(result.right_broadcast_dimensions)}
            );

            return v2_broadcast;
        };

        if (left_shape == result.new_shape) {
            // The left shape has not changed, but the right shape has:
            var_t v2_broadcast = get_right_broadcast_var();
            var_t result_var {builder.jaxpr.new_var_id(), result_type};

            builder.jaxpr.equations.emplace_back(
                std::vector{v1, value{std::move(v2_broadcast)}},
                std::vector{result_var},
                op
            );

            return {builder, std::move(result_var)};
        }

        if (right_shape == result.new_shape) {
            // The right shape has not changed, but the left shape has:
            var_t v1_broadcast = get_left_broadcast_var();
            var_t result_var {builder.jaxpr.new_var_id(), result_type};

            builder.jaxpr.equations.emplace_back(
                std::vector{value{std::move(v1_broadcast)}, v2},
                std::vector{result_var},
                op
            );

            return {builder, std::move(result_var)};
        }

        // Both have changed:

        var_t v1_broadcast = get_left_broadcast_var();
        var_t v2_broadcast = get_right_broadcast_var();
        var_t result_var {builder.jaxpr.new_var_id(), result_type};

        builder.jaxpr.equations.emplace_back(
            std::vector{value{std::move(v1_broadcast)}, value{std::move(v2_broadcast)}},
            std::vector{result_var},
            op
        );

        return {builder, std::move(result_var)};
    }

    var_t result_var {builder.jaxpr.new_var_id(), type_t{result_dtype, v1.get_type().get_shape()}};

    builder.jaxpr.equations.emplace_back(
        std::vector{v1, v2},
        std::vector{result_var},
        op
    );

    return {builder, std::move(result_var)};
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

value promote(const value& val, dtype_t dtype, jaxpr_builder& builder) {
    // Precondition: dtype > dtype(val)
    const auto& old_type = val.get_type();
    if (old_type.get_dtype() != dtype) {
        type_t new_type {dtype, old_type.get_shape()};
        var_t cast_var {builder.jaxpr.new_var_id(), std::move(new_type)};

        builder.jaxpr.equations.emplace_back(
            std::vector{val},
            std::vector{cast_var},
            primitive_op::CONVERT_ELEMENT_TYPE,
            convert_element_type_params{dtype}
        );

        return value {cast_var};

    }

    return val;
}

#define ELEMENTWISE_BINARY_OP_TRACER_TRACER(op, op_name, output_dtype)                          \
jaxpr_tracer operator op (const jaxpr_tracer& t1, const jaxpr_tracer& t2) {                     \
    dtype_t dtype = resultant_type(t1.get_type().get_dtype(), t2.get_type().get_dtype());       \
    return elementwise_binary_op(                                                               \
        promote(value{t1.var}, dtype, t1.builder),                                              \
        promote(value{t2.var}, dtype, t1.builder),                                              \
        op_name, t1.builder, output_dtype);                                                     \
}

#define ELEMENTWISE_BINARY_OP_TRACER_ARRAY(op, op_name, output_dtype)                           \
jaxpr_tracer operator op (const jaxpr_tracer& t1, const jax::array_t& array) {                  \
    dtype_t dtype = resultant_type(t1.get_type().get_dtype(), array.get_type().get_dtype());    \
    return elementwise_binary_op(                                                               \
        promote(value{t1.var}, dtype, t1.builder),                                              \
        promote(array_value(array, t1.builder), dtype, t1.builder),                             \
        op_name, t1.builder, output_dtype);                                                     \
}

#define ELEMENTWISE_BINARY_OP_ARRAY_TRACER(op, op_name, output_dtype)                           \
jaxpr_tracer operator op (const jax::array_t& array, const jaxpr_tracer& t1) {                  \
    dtype_t dtype = resultant_type(t1.get_type().get_dtype(), array.get_type().get_dtype());    \
    return elementwise_binary_op(                                                               \
        promote(array_value(array, t1.builder), dtype, t1.builder),                             \
        promote(value{t1.var}, dtype, t1.builder),                                              \
        op_name, t1.builder, output_dtype);                                                     \
}


#define ELEMENTWISE_BINARY_OP(op, op_name, output_dtype)          \
ELEMENTWISE_BINARY_OP_TRACER_TRACER(op, op_name, output_dtype)    \
ELEMENTWISE_BINARY_OP_TRACER_ARRAY(op, op_name, output_dtype)     \
ELEMENTWISE_BINARY_OP_ARRAY_TRACER(op, op_name, output_dtype)     \

ELEMENTWISE_BINARY_OP(+, jax::primitive_op::ADD, std::nullopt);
ELEMENTWISE_BINARY_OP(*, jax::primitive_op::MUL, std::nullopt);
ELEMENTWISE_BINARY_OP(-, jax::primitive_op::SUB, std::nullopt);
ELEMENTWISE_BINARY_OP(/, jax::primitive_op::DIV, std::nullopt);

ELEMENTWISE_BINARY_OP(==, jax::primitive_op::EQ, dtype_t::BOOL);
ELEMENTWISE_BINARY_OP(!=, jax::primitive_op::NE, dtype_t::BOOL);
ELEMENTWISE_BINARY_OP(<, jax::primitive_op::LT, dtype_t::BOOL);
ELEMENTWISE_BINARY_OP(<=, jax::primitive_op::LE, dtype_t::BOOL);
ELEMENTWISE_BINARY_OP(>, jax::primitive_op::GT, dtype_t::BOOL);
ELEMENTWISE_BINARY_OP(>=, jax::primitive_op::GE, dtype_t::BOOL);

jaxpr_tracer jaxpr_tracer::operator-() const {
    return unary_op(value{var}, primitive_op::NEG, builder);
}

jaxpr_tracer jaxpr_tracer::sin() const {
    return unary_op(value{var}, primitive_op::SIN, builder);
}

jaxpr_tracer jaxpr_tracer::cos() const {
    return unary_op(value{var}, primitive_op::COS, builder);
}

jaxpr_tracer jaxpr_tracer::exp() const {
    return unary_op(value{var}, primitive_op::EXP, builder);
}

jaxpr_tracer jaxpr_tracer::log() const {
    return unary_op(value{var}, primitive_op::LOG, builder);
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

    type_t new_type {var.get_type().get_dtype(), std::move(new_shape)};

    return unary_op(value{var}, std::move(new_type),  primitive_op::TRANSPOSE,
        transpose_params{std::move(permutation)}, builder);
}

template<typename ParamType>
jaxpr_tracer jaxpr_tracer::reduce_monoid(std::vector<size_t> axes, primitive_op op) const {
    // Axes must be valid:

    const auto& old_shape = var.get_type().get_shape();
    std::ranges::sort(axes);

    size_t max = 0;
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] >= old_shape.size()) {
            throw formatted_error(
                    "Error: axis {} does not exist for the given argument in {}", axes[i], to_lower(to_string(op)));
        }
        if (i > 0 && axes[i] == max) {
            throw formatted_error(
                    "Error: axis {} is included more than once in {}", axes[i], to_lower(to_string(op)));
        }
        max = axes[i];
    }

    // Apply new shape:

    std::vector<size_t> new_shape(old_shape.size() - axes.size());
    for (size_t i = 0, j = 0, k = 0; i < old_shape.size(); i++) {
        if (k < axes.size() && axes[k] == i) { k++; continue; }
        new_shape[j++] = old_shape[i];
    }

    type_t new_type {var.get_type().get_dtype(), std::move(new_shape)};

    return unary_op(value{var}, std::move(new_type), op,
        ParamType{std::move(axes)}, builder);
}

jaxpr_tracer jaxpr_tracer::reduce_sum(std::vector<size_t> axes) const {
    return reduce_monoid<reduce_sum_params>(std::move(axes), primitive_op::REDUCE_SUM);
}

jaxpr_tracer jaxpr_tracer::reduce_max(std::vector<size_t> axes) const {
    return reduce_monoid<reduce_max_params>(std::move(axes), primitive_op::REDUCE_MAX);
}

jaxpr_tracer jaxpr_tracer::reduce_min(std::vector<size_t> axes) const {
    return reduce_monoid<reduce_min_params>(std::move(axes), primitive_op::REDUCE_MIN);
}

jaxpr_tracer jaxpr_tracer::convert_element_type(dtype_t dtype) const {
    type_t new_type {dtype, var.get_type().get_shape()};

    return unary_op(value{var}, std::move(new_type), primitive_op::CONVERT_ELEMENT_TYPE,
        convert_element_type_params {dtype}, builder);
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

    type_t new_type {var.get_type().get_dtype(), shape};
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
            left_batch.size(), right_batch.size());
    }

    for (size_t i = 0; i < left_contract.size(); i++) {
        if (this_shape[left_contract[i]] != other_shape[right_contract[i]]) {
            throw std::logic_error(
                "Error: rank size mismatch in contraction indices"
            );
        }
    }

    for (size_t i = 0; i < left_batch.size(); i++) {
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

    type_t new_type {var.get_type().get_dtype(), new_shape};

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

jaxpr_tracer_index jaxpr_tracer::at(jaxpr_tracer idx) {
    return jaxpr_tracer_index{*this, std::move(idx)};
}

jaxpr_tracer_index jaxpr_tracer::at(array_t idx) {
    return jaxpr_tracer_index{*this, std::move(idx)};
}

jaxpr_tracer_index::jaxpr_tracer_index(jaxpr_tracer x, jaxpr_tracer idx):
x{std::move(x)}, idx{std::move(idx)} {}

jaxpr_tracer_index::jaxpr_tracer_index(jaxpr_tracer x, array_t idx):
x{std::move(x)}, idx{std::move(idx)} {}

static jaxpr_tracer emit_gather(jaxpr_builder& builder, value x_val, value idx_val) {
    type_t new_type {x_val.get_dtype(), new_get_shape(x_val.get_shape(), idx_val.get_shape())};
    var_t new_var = builder.jaxpr.fresh_var(std::move(new_type));

    builder.jaxpr.equations.emplace_back(
        std::vector{std::move(x_val), std::move(idx_val)},
        std::vector{new_var},
        primitive_op::GATHER
    );

    return {builder, std::move(new_var)};
}

static jaxpr_tracer emit_scatter(jaxpr_builder& builder, value x_val, value idx_val,
    value update, primitive_op scatter_op) {
    validate_scatter_op_shapes(x_val.get_shape(), idx_val.get_shape(), update.get_shape());

    type_t new_type {x_val.get_dtype(), x_val.get_shape()};
    var_t new_var = builder.jaxpr.fresh_var(std::move(new_type));

    builder.jaxpr.equations.emplace_back(
        std::vector{std::move(x_val), std::move(idx_val), std::move(update)},
        std::vector{new_var},
        scatter_op
    );

    return {builder, std::move(new_var)};
}

value jaxpr_tracer_index::get_idx_value() {
    if (std::holds_alternative<jaxpr_tracer>(idx)) {
        return value{std::get<jaxpr_tracer>(idx).get_var()};
    }
    return array_value(std::get<array_t>(idx), x.get_builder());
}

jaxpr_tracer jaxpr_tracer_index::get() {
    return emit_gather(x.get_builder(), value{x.get_var()}, get_idx_value());
}

jaxpr_tracer jaxpr_tracer_index::op(value update, primitive_op scatter_op) {
    return emit_scatter(x.get_builder(), value{x.get_var()}, get_idx_value(), std::move(update), scatter_op);
}

#define INDEX_OP(cls, builder_expr, method_name, scatter_op)                        \
jaxpr_tracer cls::method_name(const jaxpr_tracer& update) {                         \
    return op(value{update.get_var()}, scatter_op);                                 \
}                                                                                   \
jaxpr_tracer cls::method_name(const array_t& update) {                              \
    return op(array_value(update, builder_expr), scatter_op);                       \
}

INDEX_OP(jaxpr_tracer_index, x.get_builder(), set, primitive_op::SCATTER)
INDEX_OP(jaxpr_tracer_index, x.get_builder(), add, primitive_op::SCATTER_ADD)
INDEX_OP(jaxpr_tracer_index, x.get_builder(), multiply, primitive_op::SCATTER_MUL)
INDEX_OP(jaxpr_tracer_index, x.get_builder(), max, primitive_op::SCATTER_MAX)
INDEX_OP(jaxpr_tracer_index, x.get_builder(), min, primitive_op::SCATTER_MIN)

namespace jax {

    array_tracer_index array_t::at(jaxpr_tracer idx) {
        return array_tracer_index{*this, std::move(idx)};
    }

    array_index array_t::at(array_t idx) {
        return array_index{*this, std::move(idx)};
    }

    array_tracer_index::array_tracer_index(array_t x, jaxpr_tracer idx):
    x{std::move(x)}, idx{std::move(idx)} {}

    array_index::array_index(array_t x, array_t idx):
    x{std::move(x)}, idx{std::move(idx)} {}

    jaxpr_tracer array_tracer_index::get() {
        auto& builder = idx.get_builder();
        return emit_gather(builder, array_value(x, builder), value{idx.get_var()});
    }

    jaxpr_tracer array_tracer_index::op(value update, primitive_op scatter_op) {
        auto& builder = idx.get_builder();
        return emit_scatter(builder, array_value(x, builder), value{idx.get_var()},
            std::move(update), scatter_op);
    }

    INDEX_OP(array_tracer_index, idx.get_builder(), set, primitive_op::SCATTER)
    INDEX_OP(array_tracer_index, idx.get_builder(), add, primitive_op::SCATTER_ADD)
    INDEX_OP(array_tracer_index, idx.get_builder(), multiply, primitive_op::SCATTER_MUL)
    INDEX_OP(array_tracer_index, idx.get_builder(), max, primitive_op::SCATTER_MAX)
    INDEX_OP(array_tracer_index, idx.get_builder(), min, primitive_op::SCATTER_MIN)

#define ARRAY_INDEX_TRACER_OP(method_name, scatter_op)                             \
    jaxpr_tracer array_index::method_name(const jaxpr_tracer& update) {            \
        auto& builder = update.get_builder();                                      \
        return emit_scatter(builder, array_value(x, builder),                      \
            array_value(idx, builder), value{update.get_var()}, scatter_op);       \
    }

    ARRAY_INDEX_TRACER_OP(set, primitive_op::SCATTER)
    ARRAY_INDEX_TRACER_OP(add, primitive_op::SCATTER_ADD)
    ARRAY_INDEX_TRACER_OP(multiply, primitive_op::SCATTER_MUL)
    ARRAY_INDEX_TRACER_OP(max, primitive_op::SCATTER_MAX)
    ARRAY_INDEX_TRACER_OP(min, primitive_op::SCATTER_MIN)

#undef ARRAY_INDEX_TRACER_OP

}

#undef INDEX_OP


jaxpr_tracer jaxpr_builder::register_tracer(type_t type) {
    auto var = var_t{jaxpr.new_var_id(), std::move(type)};
    jaxpr.invars.push_back(var);
    return {*this, std::move(var)};
}

// Note: this does NOT work for arrays with rank 0
jaxpr_tracer jaxpr_builder::register_tracer(const array_t& array) {
    if (array.get_type().get_shape().size() == 0) {
        throw std::logic_error("Error: cannot register scalar array as tracer");
    }
    return jaxpr_tracer{*this, array_value(array, *this).get_var()};
}

void jaxpr_builder::register_output(const jaxpr_tracer& tracer) {
    jaxpr.outvals.push_back(value{tracer.get_var()});
}

void jaxpr_builder::register_output(const value& value) {
    jaxpr.outvals.push_back(value);
}

void jaxpr_builder::register_output(const literal_t& literal) {
    jaxpr.outvals.push_back(value{literal});
}

void jaxpr_builder::register_output(const var_t& var) {
    jaxpr.outvals.push_back(value{var});
}

void jaxpr_builder::register_output(const array_t& array) {
    register_output(array_value(array, *this));
}

jaxpr_tracer jaxpr_builder::full(std::vector<size_t> shape, double val, dtype_t dtype) {
    var_t projected_var = jaxpr.fresh_var(type_t{dtype, shape});
    jaxpr.equations.emplace_back(
        std::vector{value{literal_t{dtype, val}}},
        std::vector{projected_var},
        primitive_op::BROADCAST_IN_DIM,
        broadcast_in_dim_params {
        .shape = std::move(shape),
        .broadcast_dimensions = {}
        }
    );
    return jaxpr_tracer{*this, std::move(projected_var)};
}

jaxpr_tracer jaxpr_builder::full(size_t shape, double val, dtype_t dtype) {
    return full(std::vector{shape}, val, dtype);
}

jaxpr_tracer jaxpr_builder::zeros(std::vector<size_t> shape, dtype_t dtype) {
    return full(std::move(shape), 0, dtype);
}

jaxpr_tracer jaxpr_builder::zeros(size_t shape, dtype_t dtype) {
    return full(shape, 0, dtype);
}

jaxpr_tracer jaxpr_builder::ones(std::vector<size_t> shape, dtype_t dtype) {
    return full(std::move(shape), 1, dtype);
}

jaxpr_tracer jaxpr_builder::ones(size_t shape, dtype_t dtype) {
    return full(shape, 1, dtype);
}

expression&& jaxpr_builder::get_jaxpr() {
    return std::move(jaxpr);
}
