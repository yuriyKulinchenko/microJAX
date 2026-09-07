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
