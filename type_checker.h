#ifndef MICROJAX_TYPE_CHECKER_H
#define MICROJAX_TYPE_CHECKER_H

#include <expected>
#include <unordered_set>
#include "jax_array.h"

/*

The purpose of 'type_checker' is to enforce that operations between instances of array_t and jaxpr_tracer are
correctly type checked, whilst providing type deduction for complex operations, and useful error messages in
the event of incompatible types.

'type_checker' is completely location agnostic: it can only see the types that it is given, and produce error
messages accordingly. In the event of a type mismatch, it is the responsibility of the caller to provide location
information.

*/

std::expected<jax::type_t, std::string> dot_general_type(const jax::type_t& left_type, const jax::type_t& right_type,
    const std::vector<size_t>& left_contract, const std::vector<size_t>& right_contract,
    const std::vector<size_t>& left_batch, const std::vector<size_t>& right_batch);


#endif //MICROJAX_TYPE_CHECKER_H
