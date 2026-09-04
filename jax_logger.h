#ifndef JAX_LOGGER_H
#define JAX_LOGGER_H
#include <iostream>

#include "jax_types.h"

std::ostream& operator<<(std::ostream& stream, const jax::type_t& type);
std::ostream& operator<<(std::ostream& stream, const jax::var_t& var);
std::ostream& operator<<(std::ostream& stream, const jax::literal_t& literal);

// Emits the array (or any array_like, e.g. array_span_t) in nested-bracket form.
template<jax::array_like A>
std::ostream& operator<<(std::ostream& stream, const A& array) {
    const auto& shape = array.get_type().get_shape();
    const auto& flat_vector = array.get_value();

    auto index_vector = std::vector<size_t>(shape.size(), 0);
    size_t remaining_open_brackets = shape.size();

    for (double element: flat_vector) {
        // Emit sequence of [[[...
        while (remaining_open_brackets > 0) {
            stream << '[';
            remaining_open_brackets--;
        }

        stream << element;

        // Emit sequence of ]]]...
        for (size_t j = index_vector.size(); j--> 0;) {
            if (index_vector[j] < shape[j] - 1) {
                index_vector[j]++;
                stream << ", ";
                break;
            }

            index_vector[j] = 0;
            stream << ']';
            remaining_open_brackets++;
        }
    }
    return stream;
}

std::ostream& emit_typed_var(std::ostream& stream, const jax::var_t& var);

template<jax::array_like A>
std::ostream& emit_typed_array(std::ostream& stream, const A& array) {
    return stream << array << ':' << array.get_type();
}

std::ostream& emit_typed_literal(std::ostream& stream, const jax::literal_t& literal);

std::ostream& emit_value_vector(std::ostream& stream,
    const std::vector<jax::value>& values, const char* separator);

std::ostream& emit_tuple(std::ostream& stream, const std::vector<size_t>& vec);

std::ostream& emit_equation(std::ostream& stream, const jax::equation& eq, size_t tab_count);
std::ostream& operator<<(std::ostream& stream, const jax::equation& eq);

std::ostream& emit_expr(std::ostream& stream, const jax::expression& expr, size_t tab_count);
std::ostream& operator<<(std::ostream& stream, const jax::expression& expr);

#endif //JAX_LOGGER_H
