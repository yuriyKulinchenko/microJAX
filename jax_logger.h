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
    return std::visit([&](auto& member) -> std::ostream& {
        if (member.size() == 1) return stream << member[0];
        return stream << member;
    }, array.get_value());
}

inline std::ostream& emit_typed_var(std::ostream& stream, const jax::var_t& var) {
    return stream << var << ':' << var.get_type();
}

inline std::ostream& emit_typed_array(std::ostream& stream, const jax::array_t& array) {
    return stream << array << ':' << array.get_type();
}

// <var>':'<type>* '=' <op> (<var> | <array>':'<type>)


inline std::ostream& emit_value_vector(std::ostream& stream,
    const std::vector<jax::value>& values, const char* separator) {
    for (int i = 0; i < values.size(); i++) {
        const jax::value& val = values[i];
        if (val.is<jax::array_t>()) {
            const jax::array_t& array = val.get_array();
            emit_typed_array(stream, array);
        } else {
            // jax::var_t
            stream << val.get_var();
        }
        if (i != values.size() - 1) stream << separator;
    }
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const jax::equation& eq) {
    for (auto& var: eq.get_output()) {
        emit_typed_var(stream, var) << ' ';
    }

    stream << "= " << to_lower(jax::to_string(eq.get_op())) << ' ';
    return emit_value_vector(stream, eq.get_input(), " ");
}

inline std::ostream& operator<<(std::ostream& stream, const jax::expression& expr) {
    stream << "{\n";

    stream << "    lambda ";

    for (size_t i = 0; i < expr.invars.size(); i++) {
        auto& var = expr.invars[i];
        emit_typed_var(stream, var);
        if (i != expr.invars.size() - 1) stream << ", ";
    }

    stream << ". let\n";

    for (auto& eq: expr.equations) {
        stream << "    " << eq << '\n';
    }

    stream << "    in (";
    emit_value_vector(stream, expr.outvals, ", ");
    if (expr.outvals.size() == 1) stream << ',';
    stream << ")\n";

    stream << "}\n";
    return stream;
}

#endif //JAX_LOGGER_H
