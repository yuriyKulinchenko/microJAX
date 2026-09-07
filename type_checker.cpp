#include "type_checker.h"

// Validates that 'axes' are valid axis indices for an operand of the given rank,
// and that they are strictly increasing (which enforces both sorted-ascending and
// distinct in one pass). Returns an error message on failure, or nullopt on success.
static std::optional<std::string> validate_axes(
    const std::vector<size_t>& axes, size_t rank, std::string_view role) {
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] >= rank) {
            return std::format(
                "every {} axis should be less than the operand rank {}, "
                "instead the axis {} holds",
                role, rank, axes[i]);
        }
        if (i > 0 && axes[i] <= axes[i - 1]) {
            return std::format(
                "the {} axes should be strictly increasing, "
                "instead the axis {} follows the axis {}",
                role, axes[i], axes[i - 1]);
        }
    }
    return std::nullopt;
}

std::expected<jax::type_t, std::string> dot_general_type(const jax::type_t& left_type, const jax::type_t& right_type,
    const std::vector<size_t>& left_contract, const std::vector<size_t>& right_contract,
    const std::vector<size_t>& left_batch, const std::vector<size_t>& right_batch) {

    auto& left_shape = left_type.get_shape();
    auto& right_shape = right_type.get_shape();

    if (left_type.get_dtype() != right_type.get_dtype()) {
        return std::unexpected(std::format(
            "Operands should have the same type, "
            "instead they have types {} and {} respectively",
            to_string(left_type.get_dtype()), to_string(right_type.get_dtype())));
    }

    if (left_contract.size() != right_contract.size()) {
        return std::unexpected(std::format(
            "The left and right contraction lists should have equal length, "
            "instead they have lengths {} and {} respectively",
            left_contract.size(), right_contract.size()));
    }

    if (left_batch.size() != right_batch.size()) {
        return std::unexpected(std::format(
            "The left and right batch lists should have equal length, "
            "instead they have lengths {} and {} respectively",
            left_batch.size(), right_batch.size()));
    }

    // Every axis must be in range and strictly increasing before it is used to index
    // a shape below. This catches out-of-range, duplicate and unsorted axes.

    if (auto error = validate_axes(left_contract, left_shape.size(), "left contract")) {
        return std::unexpected(*error);
    }
    if (auto error = validate_axes(right_contract, right_shape.size(), "right contract")) {
        return std::unexpected(*error);
    }
    if (auto error = validate_axes(left_batch, left_shape.size(), "left batch")) {
        return std::unexpected(*error);
    }
    if (auto error = validate_axes(right_batch, right_shape.size(), "right batch")) {
        return std::unexpected(*error);
    }

    for (size_t i = 0; i < left_contract.size(); i++) {
        if (left_shape[left_contract[i]] != right_shape[right_contract[i]]) {
            return std::unexpected(std::format(
                "Dimension size at dimension {} for the first operand should be equal to "
                "dimension size at dimension {} for the second operand for contraction, "
                "instead the sizes are {} and {} respectively",
                left_contract[i], right_contract[i],
                left_shape[left_contract[i]], right_shape[right_contract[i]]
            ));
        }
    }

    for (size_t i = 0; i < left_batch.size(); i++) {
        if (left_shape[left_batch[i]] != right_shape[right_batch[i]]) {

            return std::unexpected(std::format(
                "Dimension size at dimension {} for the first operand should be equal to "
                "dimension size at dimension {} for the second operand for batching, "
                "instead the sizes are {} and {} respectively",
                left_batch[i], right_batch[i],
                left_shape[left_batch[i]], right_shape[right_batch[i]]
            ));
        }
    }

    // contract and batch indices cannot overlap
    // left_contract, left_batch have to be disjoint,
    // right_contract, right batch have to be disjoint:

    std::unordered_set<size_t> left_batch_set {left_batch.begin(), left_batch.end()};
    for (size_t contract_index: left_contract) {
        if (left_batch_set.contains(contract_index)) {
            return std::unexpected(std::format(
                "the left batch and left contract axes should be disjoint, "
                "instead the axis {} is shared",
                contract_index
            ));
        }
    }

    std::unordered_set<size_t> right_batch_set {right_batch.begin(), right_batch.end()};
    for (size_t contract_index: right_contract) {
        if (right_batch_set.contains(contract_index)) {
            return std::unexpected(std::format(
                "the right batch and right contract axes should be disjoint, "
                "instead the axis {} is shared",
                contract_index
            ));
        }
    }

    // Calculate the new shape, which will be of the form (batch, left free, right free):

    size_t new_size =
        left_shape.size() + right_shape.size() - left_batch.size() - 2 * left_contract.size();

    std::vector<size_t> new_shape {};
    new_shape.reserve(new_size);

    // Add batch:

    for (auto batch_dim: left_batch) {
        new_shape.push_back(left_shape[batch_dim]);
    }

    // Add left free:

    auto left_contract_complement = complement(left_contract, left_shape.size());
    for (auto free_dim: left_contract_complement) {
        if (!left_batch_set.contains(free_dim)) new_shape.push_back(left_shape[free_dim]);
    }

    // Add right free:

    auto right_contract_complement = complement(right_contract, right_shape.size());
    for (auto free_dim: right_contract_complement) {
        if (!right_batch_set.contains(free_dim)) new_shape.push_back(right_shape[free_dim]);
    }

    jax::type_t new_type {left_type.get_dtype(), std::move(new_shape)};

    return new_type;
}

// Formats a shape as "(1, 2, 3)" for error messages, consistently across every checker.
static std::string format_shape(std::span<const size_t> shape) {
    std::string out = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }
    out += ")";
    return out;
}

// Formats a full type as "F32(1, 2, 3)".
static std::string format_type(const jax::type_t& type) {
    return std::string{to_string(type.get_dtype())} + format_shape(type.get_shape());
}

// Right-aligned NumPy-style broadcasting: each aligned pair of dimensions must be equal
// or one of them must be 1; the result takes the larger. Shared by the binary elementwise
// and comparison checkers, which differ only in their output dtype.
static std::expected<std::vector<size_t>, std::string> broadcast_shapes(
    std::span<const size_t> left, std::span<const size_t> right) {
    size_t rank = left.size() > right.size() ? left.size() : right.size();
    size_t left_offset = rank - left.size();
    size_t right_offset = rank - right.size();

    std::vector<size_t> result(rank);
    for (size_t i = 0; i < rank; i++) {
        size_t l = i < left_offset ? 1 : left[i - left_offset];
        size_t r = i < right_offset ? 1 : right[i - right_offset];
        if (l == r || l == 1 || r == 1) {
            result[i] = l > r ? l : r;
        } else {
            return std::unexpected(std::format(
                "the operand shapes {} and {} should be broadcast compatible, "
                "instead the sizes {} and {} hold at aligned dimension {}",
                format_shape(left), format_shape(right), l, r, i));
        }
    }
    return result;
}

std::expected<jax::type_t, std::string> unary_elementwise_type(const jax::type_t& type) {
    return type;
}

std::expected<jax::type_t, std::string> binary_elementwise_type(
    const jax::type_t& left_type, const jax::type_t& right_type) {
    auto shape = broadcast_shapes(left_type.get_shape(), right_type.get_shape());
    if (!shape) return std::unexpected(shape.error());
    return jax::type_t{resultant_type(left_type.get_dtype(), right_type.get_dtype()), std::move(*shape)};
}

std::expected<jax::type_t, std::string> comparison_type(
    const jax::type_t& left_type, const jax::type_t& right_type) {
    auto shape = broadcast_shapes(left_type.get_shape(), right_type.get_shape());
    if (!shape) return std::unexpected(shape.error());
    return jax::type_t{jax::dtype_t::BOOL, std::move(*shape)};
}

std::expected<jax::type_t, std::string> reduce_type(
    const jax::type_t& type, const std::vector<size_t>& axes) {
    size_t rank = type.get_shape().size();
    if (auto error = validate_axes(axes, rank, "reduction")) {
        return std::unexpected(*error);
    }

    std::vector<size_t> new_shape;
    for (size_t axis: complement(axes, rank)) {
        new_shape.push_back(type.get_shape()[axis]);
    }
    return jax::type_t{type.get_dtype(), std::move(new_shape)};
}

// ============================ gather / scatter ============================

std::expected<jax::type_t, std::string> scatter_type(
    const jax::type_t& type, const jax::type_t& index_type, const jax::type_t& update_type) {
    // Convention: operand [n, ms...], index [ls...], update [ls..., ms...] -> operand type.

    if (!is_integral(index_type.get_dtype())) {
        return std::unexpected(std::format(
            "the scatter indices should have an integral dtype, instead the dtype {} holds",
            to_string(index_type.get_dtype())));
    }
    if (type.get_dtype() != update_type.get_dtype()) {
        return std::unexpected(std::format(
            "the scatter operand and update should share a dtype, "
            "instead the dtypes {} and {} hold respectively",
            to_string(type.get_dtype()), to_string(update_type.get_dtype())));
    }

    auto& x_shape = type.get_shape();
    auto& idx_shape = index_type.get_shape();
    auto& u_shape = update_type.get_shape();

    if (x_shape.empty()) {
        return std::unexpected(
            "the scatter operand should have rank at least 1, instead a scalar holds");
    }
    if (idx_shape.size() > u_shape.size()) {
        return std::unexpected(std::format(
            "the index rank should not exceed the update rank, "
            "instead the ranks {} and {} hold respectively",
            idx_shape.size(), u_shape.size()));
    }

    size_t num_ls = idx_shape.size();
    for (size_t i = 0; i < num_ls; i++) {
        if (idx_shape[i] != u_shape[i]) {
            return std::unexpected(std::format(
                "the update's leading dimensions should equal the index shape {}, "
                "instead the update has shape {}",
                format_shape(idx_shape), format_shape(u_shape)));
        }
    }
    if (x_shape.size() - 1 != u_shape.size() - num_ls) {
        return std::unexpected(std::format(
            "the update rank minus the index rank should equal the operand rank minus 1, "
            "instead the operand, index and update ranks {}, {} and {} hold",
            x_shape.size(), num_ls, u_shape.size()));
    }
    for (size_t i = 0; i + 1 < x_shape.size(); i++) {
        if (x_shape[i + 1] != u_shape[num_ls + i]) {
            return std::unexpected(std::format(
                "the operand's trailing dimensions should equal the update's trailing "
                "dimensions, instead operand shape {} and update shape {} hold",
                format_shape(x_shape), format_shape(u_shape)));
        }
    }

    return type;
}

std::expected<jax::type_t, std::string> gather_type(
    const jax::type_t& type, const jax::type_t& index_type) {
    // Convention: operand [n, ms...], index [ls...] -> [ls..., ms...].

    if (!is_integral(index_type.get_dtype())) {
        return std::unexpected(std::format(
            "the gather indices should have an integral dtype, instead the dtype {} holds",
            to_string(index_type.get_dtype())));
    }
    if (type.get_shape().empty()) {
        return std::unexpected(
            "the gather operand should have rank at least 1, instead a scalar holds");
    }

    return jax::type_t{type.get_dtype(),
        jax::new_get_shape(type.get_shape(), index_type.get_shape())};
}

std::expected<jax::type_t, std::string> reshape_type(
    const jax::type_t& type, const std::vector<size_t>& new_sizes) {
    size_t total = num_elements(type.get_shape());

    size_t known = 1;
    bool inferred = false;
    for (size_t dim: new_sizes) {
        if (dim == static_cast<size_t>(-1)) {
            if (inferred) {
                return std::unexpected(
                    "at most one reshape dimension should be inferred (-1), instead several hold");
            }
            inferred = true;
        } else {
            known *= dim;
        }
    }

    std::vector<size_t> result(new_sizes.begin(), new_sizes.end());
    if (inferred) {
        if (known == 0 || total % known != 0) {
            return std::unexpected(std::format(
                "the known reshape dimensions should divide the element count {}, "
                "instead they multiply to {}", total, known));
        }
        for (size_t& dim: result) {
            if (dim == static_cast<size_t>(-1)) dim = total / known;
        }
    } else if (known != total) {
        return std::unexpected(std::format(
            "the reshape should preserve the element count {}, instead it has {} elements",
            total, known));
    }

    return jax::type_t{type.get_dtype(), std::move(result)};
}

std::expected<jax::type_t, std::string> broadcast_in_dim_type(
    const jax::type_t& type, const std::vector<size_t>& shape,
    const std::vector<size_t>& broadcast_dimensions) {
    auto& in_shape = type.get_shape();

    if (broadcast_dimensions.size() != in_shape.size()) {
        return std::unexpected(std::format(
            "the number of broadcast dimensions should equal the operand rank {}, "
            "instead {} broadcast dimensions hold",
            in_shape.size(), broadcast_dimensions.size()));
    }
    // Broadcast dimensions must be in range and strictly increasing.
    if (auto error = validate_axes(broadcast_dimensions, shape.size(), "broadcast")) {
        return std::unexpected(*error);
    }
    for (size_t i = 0; i < in_shape.size(); i++) {
        size_t target = shape[broadcast_dimensions[i]];
        if (in_shape[i] != 1 && in_shape[i] != target) {
            return std::unexpected(std::format(
                "each broadcast operand dimension should be 1 or equal its target, "
                "instead operand dimension {} of size {} maps to target size {}",
                i, in_shape[i], target));
        }
    }

    return jax::type_t{type.get_dtype(), shape};
}

std::expected<jax::type_t, std::string> transpose_type(
    const jax::type_t& type, const std::vector<size_t>& permutation) {
    size_t rank = type.get_shape().size();

    if (permutation.size() != rank) {
        return std::unexpected(std::format(
            "the permutation length should equal the operand rank {}, "
            "instead the length {} holds", rank, permutation.size()));
    }

    std::unordered_set<size_t> seen;
    for (size_t axis: permutation) {
        if (axis >= rank) {
            return std::unexpected(std::format(
                "every permutation axis should be less than the operand rank {}, "
                "instead the axis {} holds", rank, axis));
        }
        if (!seen.insert(axis).second) {
            return std::unexpected(std::format(
                "the permutation should contain distinct axes, "
                "instead the axis {} is repeated", axis));
        }
    }

    return jax::type_t{type.get_dtype(), permute(type.get_shape(), permutation)};
}

std::expected<jax::type_t, std::string> concatenate_type(
    const std::vector<jax::type_t>& types, size_t dimension) {
    if (types.empty()) {
        return std::unexpected("concatenate should have at least one operand, instead none hold");
    }

    const jax::type_t& first = types[0];
    size_t rank = first.get_shape().size();

    if (dimension >= rank) {
        return std::unexpected(std::format(
            "the concatenation dimension should be less than the operand rank {}, "
            "instead the dimension {} holds", rank, dimension));
    }

    size_t concat_size = first.get_shape()[dimension];
    for (size_t t = 1; t < types.size(); t++) {
        const jax::type_t& operand = types[t];

        if (operand.get_dtype() != first.get_dtype()) {
            return std::unexpected(std::format(
                "all concatenation operands should share a dtype, "
                "instead the dtypes {} and {} hold",
                to_string(first.get_dtype()), to_string(operand.get_dtype())));
        }
        if (operand.get_shape().size() != rank) {
            return std::unexpected(std::format(
                "all concatenation operands should share the rank {}, "
                "instead the rank {} holds", rank, operand.get_shape().size()));
        }
        for (size_t d = 0; d < rank; d++) {
            if (d == dimension) continue;
            if (operand.get_shape()[d] != first.get_shape()[d]) {
                return std::unexpected(std::format(
                    "concatenation operands should match on every non-concatenated dimension, "
                    "instead the shapes {} and {} differ at dimension {}",
                    format_shape(first.get_shape()), format_shape(operand.get_shape()), d));
            }
        }
        concat_size += operand.get_shape()[dimension];
    }

    std::vector<size_t> new_shape = first.get_shape();
    new_shape[dimension] = concat_size;
    return jax::type_t{first.get_dtype(), std::move(new_shape)};
}

std::expected<jax::type_t, std::string> slice_type(
    const jax::type_t& type, const std::vector<size_t>& start_indices,
    const std::vector<size_t>& limit_indices, const std::vector<size_t>& strides) {
    auto& shape = type.get_shape();
    size_t rank = shape.size();

    if (start_indices.size() != rank || limit_indices.size() != rank || strides.size() != rank) {
        return std::unexpected(std::format(
            "the slice start, limit and stride lists should each have length equal to the "
            "operand rank {}, instead the lengths {}, {} and {} hold",
            rank, start_indices.size(), limit_indices.size(), strides.size()));
    }

    std::vector<size_t> new_shape;
    new_shape.reserve(rank);
    for (size_t d = 0; d < rank; d++) {
        if (strides[d] == 0) {
            return std::unexpected(std::format(
                "every slice stride should be at least 1, "
                "instead the stride {} holds at dimension {}", strides[d], d));
        }
        if (start_indices[d] > limit_indices[d] || limit_indices[d] > shape[d]) {
            return std::unexpected(std::format(
                "each slice should satisfy start <= limit <= dimension size ({}), "
                "instead the start {} and limit {} hold at dimension {}",
                shape[d], start_indices[d], limit_indices[d], d));
        }
        new_shape.push_back((limit_indices[d] - start_indices[d] + strides[d] - 1) / strides[d]);
    }

    return jax::type_t{type.get_dtype(), std::move(new_shape)};
}

std::expected<jax::type_t, std::string> pad_type(
    const jax::type_t& type, const jax::type_t& padding_value_type,
    const std::vector<std::array<size_t, 3>>& padding_config) {
    if (!padding_value_type.get_shape().empty()) {
        return std::unexpected(std::format(
            "the padding value should be a scalar, instead a rank {} value holds",
            padding_value_type.get_shape().size()));
    }
    if (padding_value_type.get_dtype() != type.get_dtype()) {
        return std::unexpected(std::format(
            "the padding value should share the operand dtype, "
            "instead the dtypes {} and {} hold respectively",
            to_string(type.get_dtype()), to_string(padding_value_type.get_dtype())));
    }

    auto& shape = type.get_shape();
    if (padding_config.size() != shape.size()) {
        return std::unexpected(std::format(
            "the padding config should have one entry per operand dimension ({}), "
            "instead {} entries hold", shape.size(), padding_config.size()));
    }

    std::vector<size_t> new_shape;
    new_shape.reserve(shape.size());
    for (size_t d = 0; d < shape.size(); d++) {
        auto [low, high, interior] = padding_config[d];
        size_t n = shape[d];
        size_t interior_total = n == 0 ? 0 : (n - 1) * interior;
        new_shape.push_back(low + high + n + interior_total);
    }

    return jax::type_t{type.get_dtype(), std::move(new_shape)};
}

std::expected<jax::type_t, std::string> convert_element_type_type(
    const jax::type_t& type, jax::dtype_t new_dtype) {
    return jax::type_t{new_dtype, type.get_shape()};
}

std::expected<jax::type_t, std::string> select_type(
    const jax::type_t& predicate_type, const std::vector<jax::type_t>& value_types) {
    if (value_types.empty()) {
        return std::unexpected("select should have at least one value operand, instead none hold");
    }
    if (!is_integral(predicate_type.get_dtype())) {
        return std::unexpected(std::format(
            "the select predicate should have an integral dtype, instead the dtype {} holds",
            to_string(predicate_type.get_dtype())));
    }

    const jax::type_t& first = value_types[0];
    if (predicate_type.get_shape() != first.get_shape()) {
        return std::unexpected(std::format(
            "the select predicate should share the value shape {}, instead the shape {} holds",
            format_shape(first.get_shape()), format_shape(predicate_type.get_shape())));
    }
    for (size_t i = 1; i < value_types.size(); i++) {
        if (value_types[i].get_shape() != first.get_shape()) {
            return std::unexpected(std::format(
                "all select values should share a shape, instead the shapes {} and {} hold",
                format_shape(first.get_shape()), format_shape(value_types[i].get_shape())));
        }
        if (value_types[i].get_dtype() != first.get_dtype()) {
            return std::unexpected(std::format(
                "all select values should share a dtype, instead the dtypes {} and {} hold",
                to_string(first.get_dtype()), to_string(value_types[i].get_dtype())));
        }
    }

    return first;
}

// ============================ structural (multi-output) ============================

std::expected<std::vector<jax::type_t>, std::string> cond_type(
    const jax::type_t& index_type, const std::vector<jax::type_t>& operand_types,
    const std::vector<jax::expression>& branches) {
    if (!index_type.get_shape().empty()) {
        return std::unexpected(std::format(
            "the cond index should be a scalar, instead a rank {} index holds",
            index_type.get_shape().size()));
    }
    if (!is_integral(index_type.get_dtype())) {
        return std::unexpected(std::format(
            "the cond index should have an integral dtype, instead the dtype {} holds",
            to_string(index_type.get_dtype())));
    }
    if (branches.empty()) {
        return std::unexpected("cond should have at least one branch, instead none hold");
    }

    // Every branch's inputs must match the operands.
    for (size_t b = 0; b < branches.size(); b++) {
        const jax::expression& branch = branches[b];
        if (branch.invars.size() != operand_types.size()) {
            return std::unexpected(std::format(
                "branch {} should take {} operands, instead it takes {}",
                b, operand_types.size(), branch.invars.size()));
        }
        for (size_t i = 0; i < operand_types.size(); i++) {
            if (branch.invars[i].get_type() != operand_types[i]) {
                return std::unexpected(std::format(
                    "branch {} input {} should match the operand type, instead the types {} "
                    "and {} hold", b, i,
                    format_type(operand_types[i]), format_type(branch.invars[i].get_type())));
            }
        }
    }

    // All branches must agree on their outputs; those are the cond's output types.
    std::vector<jax::type_t> output_types;
    for (const jax::value& outval: branches[0].outvals) {
        output_types.push_back(outval.get_type());
    }
    for (size_t b = 1; b < branches.size(); b++) {
        if (branches[b].outvals.size() != output_types.size()) {
            return std::unexpected(std::format(
                "all cond branches should return the same number of outputs, "
                "instead branches 0 and {} return {} and {}",
                b, output_types.size(), branches[b].outvals.size()));
        }
        for (size_t i = 0; i < output_types.size(); i++) {
            if (branches[b].outvals[i].get_type() != output_types[i]) {
                return std::unexpected(std::format(
                    "all cond branches should agree on output {}, instead the types {} and {} hold",
                    i, format_type(output_types[i]), format_type(branches[b].outvals[i].get_type())));
            }
        }
    }

    return output_types;
}

std::expected<std::vector<jax::type_t>, std::string> scan_type(
    const std::vector<jax::type_t>& operand_types, const jax::expression& body,
    size_t length, size_t num_consts, size_t num_carry) {
    // operands = [consts..., carry..., xs...]; body: (consts, carry, x_slice) -> (carry, y_slice).

    if (operand_types.size() < num_consts + num_carry) {
        return std::unexpected(std::format(
            "scan should receive at least {} const and carry operands, instead {} hold",
            num_consts + num_carry, operand_types.size()));
    }
    size_t num_xs = operand_types.size() - num_consts - num_carry;

    if (body.invars.size() != num_consts + num_carry + num_xs) {
        return std::unexpected(std::format(
            "the scan body should take {} inputs (consts + carry + xs), instead it takes {}",
            num_consts + num_carry + num_xs, body.invars.size()));
    }
    if (body.outvals.size() < num_carry) {
        return std::unexpected(std::format(
            "the scan body should return at least the {} carry outputs, instead it returns {}",
            num_carry, body.outvals.size()));
    }

    // Consts: operand type equals body input type.
    for (size_t i = 0; i < num_consts; i++) {
        if (operand_types[i] != body.invars[i].get_type()) {
            return std::unexpected(std::format(
                "scan const {} should match the body input, instead the types {} and {} hold",
                i, format_type(operand_types[i]), format_type(body.invars[i].get_type())));
        }
    }

    // Carry: operand == body input == body carry output (carry is invariant).
    for (size_t i = 0; i < num_carry; i++) {
        const jax::type_t& operand = operand_types[num_consts + i];
        const jax::type_t& invar = body.invars[num_consts + i].get_type();
        jax::type_t outval = body.outvals[i].get_type();
        if (operand != invar) {
            return std::unexpected(std::format(
                "scan carry {} should match the body input, instead the types {} and {} hold",
                i, format_type(operand), format_type(invar)));
        }
        if (outval != invar) {
            return std::unexpected(std::format(
                "scan carry {} should be invariant across iterations, "
                "instead the input type {} and output type {} hold",
                i, format_type(invar), format_type(outval)));
        }
    }

    // Xs: each operand is the per-iteration slice stacked along a leading length dimension.
    for (size_t i = 0; i < num_xs; i++) {
        const jax::type_t& xs_type = operand_types[num_consts + num_carry + i];
        const jax::type_t& slice = body.invars[num_consts + num_carry + i].get_type();
        auto& xs_shape = xs_type.get_shape();

        if (xs_shape.empty() || xs_shape[0] != length) {
            return std::unexpected(std::format(
                "scan xs input {} should have leading dimension equal to the scan length {}, "
                "instead the shape {} holds", i, length, format_shape(xs_shape)));
        }
        std::vector<size_t> trailing(xs_shape.begin() + 1, xs_shape.end());
        if (trailing != slice.get_shape() || xs_type.get_dtype() != slice.get_dtype()) {
            return std::unexpected(std::format(
                "scan xs input {} should be the per-iteration slice {} stacked along length {}, "
                "instead the shape {} holds", i, format_type(slice), length, format_shape(xs_shape)));
        }
    }

    // Output: the final carry, followed by each y-slice stacked along the length dimension.
    std::vector<jax::type_t> output_types;
    for (size_t i = 0; i < num_carry; i++) {
        output_types.push_back(body.outvals[i].get_type());
    }
    for (size_t i = num_carry; i < body.outvals.size(); i++) {
        jax::type_t y_slice = body.outvals[i].get_type();
        std::vector<size_t> stacked;
        stacked.reserve(y_slice.get_shape().size() + 1);
        stacked.push_back(length);
        for (size_t d: y_slice.get_shape()) stacked.push_back(d);
        output_types.push_back(jax::type_t{y_slice.get_dtype(), std::move(stacked)});
    }

    return output_types;
}
