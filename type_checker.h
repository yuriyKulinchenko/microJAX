#ifndef MICROJAX_TYPE_CHECKER_H
#define MICROJAX_TYPE_CHECKER_H

#include <expected>
#include <unordered_set>
#include <optional>
#include <string>
#include <vector>
#include <array>
#include "jax_array.h"
#include "jax_types.h"

/*

The purpose of 'type_checker' is to enforce that operations between instances of array_t and jaxpr_tracer are
correctly type checked, whilst providing type deduction for complex operations, and useful error messages in
the event of incompatible types.

'type_checker' is completely location agnostic: it can only see the types that it is given, and produce error
messages accordingly. In the event of a type mismatch, it is the responsibility of the caller to provide location
information.

*/

std::expected<jax::type_t, std::string> unary_elementwise_type(const jax::type_t& type);

std::expected<jax::type_t, std::string> binary_elementwise_type(
    const jax::type_t& left_type, const jax::type_t& right_type);

std::expected<jax::type_t, std::string> comparison_type(
    const jax::type_t& left_type, const jax::type_t& right_type);

std::expected<jax::type_t, std::string> reduce_type(
    const jax::type_t& type, const std::vector<size_t>& axes);

std::expected<jax::type_t, std::string> dot_general_type(
    const jax::type_t& left_type, const jax::type_t& right_type,
    const std::vector<size_t>& left_contract, const std::vector<size_t>& right_contract,
    const std::vector<size_t>& left_batch, const std::vector<size_t>& right_batch);

std::expected<jax::type_t, std::string> scatter_type(
    const jax::type_t& type, const jax::type_t& index_type, const jax::type_t& update_type);

std::expected<jax::type_t, std::string> gather_type(
    const jax::type_t& type, const jax::type_t& index_type);

std::expected<jax::type_t, std::string> reshape_type(
    const jax::type_t& type, const std::vector<size_t>& new_sizes);

std::expected<jax::type_t, std::string> broadcast_in_dim_type(
    const jax::type_t& type, const std::vector<size_t>& shape,
    const std::vector<size_t>& broadcast_dimensions);

std::expected<jax::type_t, std::string> transpose_type(
    const jax::type_t& type, const std::vector<size_t>& permutation);

std::expected<jax::type_t, std::string> concatenate_type(
    const std::vector<jax::type_t>& types, size_t dimension);

std::expected<jax::type_t, std::string> slice_type(
    const jax::type_t& type, const std::vector<size_t>& start_indices,
    const std::vector<size_t>& limit_indices, const std::vector<size_t>& strides);

std::expected<jax::type_t, std::string> pad_type(
    const jax::type_t& type, const jax::type_t& padding_value_type,
    const std::vector<std::array<size_t, 3>>& padding_config);

std::expected<jax::type_t, std::string> convert_element_type_type(
    const jax::type_t& type, jax::dtype_t new_dtype);

std::expected<jax::type_t, std::string> select_type(
    const jax::type_t& predicate_type, const std::vector<jax::type_t>& value_types);

std::expected<std::vector<jax::type_t>, std::string> cond_type(
    const jax::type_t& index_type, const std::vector<jax::type_t>& operand_types,
    const std::vector<jax::expression>& branches);

std::expected<std::vector<jax::type_t>, std::string> scan_type(
    const std::vector<jax::type_t>& operand_types, const jax::expression& body,
    size_t length, size_t num_consts, size_t num_carry);


#endif //MICROJAX_TYPE_CHECKER_H
