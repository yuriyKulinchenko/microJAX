#ifndef JAX_LOGGER_H
#define JAX_LOGGER_H
#include <iostream>

#include "jax_types.h"

std::ostream& operator<<(std::ostream& stream, const jax::type_t& type);
std::ostream& operator<<(std::ostream& stream, const jax::var_t& var);
std::ostream& operator<<(std::ostream& stream, const jax::array_t& array);
std::ostream& operator<<(std::ostream& stream, const jax::literal_t& literal);

std::ostream& emit_typed_var(std::ostream& stream, const jax::var_t& var);
std::ostream& emit_typed_array(std::ostream& stream, const jax::array_t& array);
std::ostream& emit_typed_literal(std::ostream& stream, const jax::literal_t& literal);

std::ostream& emit_value_vector(std::ostream& stream,
    const std::vector<jax::value>& values, const char* separator);

std::ostream& emit_tuple(std::ostream& stream, const std::vector<size_t>& vec);

std::ostream& emit_equation(std::ostream& stream, const jax::equation& eq, size_t tab_count);
std::ostream& operator<<(std::ostream& stream, const jax::equation& eq);

std::ostream& emit_expr(std::ostream& stream, const jax::expression& expr, size_t tab_count);
std::ostream& operator<<(std::ostream& stream, const jax::expression& expr);

#endif //JAX_LOGGER_H
