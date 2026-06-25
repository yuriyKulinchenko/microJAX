#ifndef JAX_LOGGER_H
#define JAX_LOGGER_H
#include <iostream>

#include "helper.h"
#include "jax_types.h"

// Need to be able to log equations, expressions, values

inline std::ostream& operator<<(std::ostream& stream, const jax::type_t& type) {
    stream << to_lower(jax::to_string(type.get_base_type()));
    stream << '[';
    const std::vector<u32>& dimension = type.get_dimension();
    for (int i = 0; i < dimension.size(); i++) {
        stream << dimension[i];
        if (i != dimension.size() - 1) stream << ", ";
    }
    stream << ']';
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const jax::var_t& var) {
    stream << "%" << var.get_id();
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const jax::array_t& array) {
    return std::visit([&](auto& variant) -> std::ostream& {
        return stream << variant;
    }, array.get_value());
}

// <var>':'<type>* '=' <op> (<var> | <array>':'<type>)

inline std::ostream& operator<<(std::ostream& stream, jax::equation& eq) {
    for (auto& output: eq.get_output()) {
        stream << output << ':' << output.get_type() << ' ';
    }

    stream << "= " << to_lower(jax::to_string(eq.get_op())) << ' ';

    for (int i = 0; i < eq.get_input().size(); i++) {
        const jax::value& input = eq.get_input()[i];
        if (input.is<jax::array_t>()) {
            const jax::array_t& array = input.get_array();
            stream << array << ':' << array.get_type();
        } else {
            // jax::var_t
            stream << input.get_var();
        }
        if (i != eq.get_input().size() - 1) stream << ' ';
    }

    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, jax::expression& expr) {
    stream << "{\n";
    for (auto& eq: expr.equations) {
        stream << "    " << eq << '\n';
    }
    stream << "}\n";
    return stream;
}

#endif //JAX_LOGGER_H
