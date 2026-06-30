#ifndef JAX_LOGGER_H
#define JAX_LOGGER_H
#include <iostream>

#include "helper.h"
#include "jax_types.h"

// Need to be able to log equations, expressions, values

inline std::ostream& operator<<(std::ostream& stream, const jax::type_t& type) {
    stream << to_lower(jax::to_string(type.get_base_type()));
    stream << '[';
    const std::vector<size_t>& dimension = type.get_dimension();
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

    const auto& dimension = array.get_type().get_dimension();
    return std::visit([&](auto& flat_vector) -> std::ostream& {
        auto index_vector = std::vector<size_t>(dimension.size(), 0);
        size_t remaining_open_brackets = dimension.size();

        for (auto& element: flat_vector) {

            // Emit sequence of [[[...
            while (remaining_open_brackets > 0) {
                stream << '[';
                remaining_open_brackets--;
            }

            stream << element;

            // Emit sequence of ]]]...
            for (size_t j = index_vector.size(); j--> 0;) {
                if (index_vector[j] < dimension[j] - 1) {
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

inline std::ostream& emit_tuple(std::ostream& stream, const std::vector<size_t>& vec) {
    stream << '(';
    for (size_t i = 0; i < vec.size(); i++) {
        stream << vec[i];
        if (i != vec.size() - 1) stream << ", ";
        else if (vec.size() == 1) stream << ',';
    }
    stream << ')';
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const jax::equation& eq) {
    for (auto& var: eq.get_output()) {
        emit_typed_var(stream, var) << ' ';
    }

    stream << "= " << to_lower(jax::to_string(eq.get_op()));
    switch (eq.get_op()) {
        case jax::primitive_op::TRANSPOSE: {
            stream << "[permutation=";
            emit_tuple(stream, std::get<jax::transpose_params>(eq.get_params()).permutation);
            stream << ']';
            break;
        }
        default:
    }
    stream << ' ';
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
