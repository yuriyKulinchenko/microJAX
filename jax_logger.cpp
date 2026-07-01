#include "jax_logger.h"
#include "helper.h"

using namespace jax;

std::ostream& operator<<(std::ostream& stream, const type_t& type) {
    stream << to_lower(to_string(type.get_base_type()));
    stream << '[';
    const std::vector<size_t>& shape = type.get_shape();
    for (int i = 0; i < shape.size(); i++) {
        stream << shape[i];
        if (i != shape.size() - 1) stream << ", ";
    }
    stream << ']';
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const var_t& var) {
    stream << "%" << var.get_id();
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const array_t& array) {

    const auto& shape = array.get_type().get_shape();
    return std::visit([&](auto& flat_vector) -> std::ostream& {
        auto index_vector = std::vector<size_t>(shape.size(), 0);
        size_t remaining_open_brackets = shape.size();

        for (auto& element: flat_vector) {

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
    }, array.get_value());
}

std::ostream& emit_typed_var(std::ostream& stream, const var_t& var) {
    return stream << var << ':' << var.get_type();
}

std::ostream& emit_typed_array(std::ostream& stream, const array_t& array) {
    return stream << array << ':' << array.get_type();
}

// <var>':'<type>* '=' <op> (<var> | <array>':'<type>)


std::ostream& emit_value_vector(std::ostream& stream,
    const std::vector<value>& values, const char* separator) {
    for (int i = 0; i < values.size(); i++) {
        const value& val = values[i];
        if (val.is<array_t>()) {
            const array_t& array = val.get_array();
            emit_typed_array(stream, array);
        } else {
            // var_t
            stream << val.get_var();
        }
        if (i != values.size() - 1) stream << separator;
    }
    return stream;
}

std::ostream& emit_tuple(std::ostream& stream, const std::vector<size_t>& vec) {
    stream << '(';
    for (size_t i = 0; i < vec.size(); i++) {
        stream << vec[i];
        if (i != vec.size() - 1) stream << ", ";
        else if (vec.size() == 1) stream << ',';
    }
    stream << ')';
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const equation& eq) {
    for (auto& var: eq.get_output()) {
        emit_typed_var(stream, var) << ' ';
    }

    stream << "= " << to_lower(to_string(eq.get_op()));
    switch (eq.get_op()) {
        case primitive_op::TRANSPOSE: {
            stream << "[permutation=";
            emit_tuple(stream, std::get<transpose_params>(eq.get_params()).permutation);
            stream << ']';
            break;
        }

        case primitive_op::REDUCE_SUM: {
            stream << "[axes=";
            emit_tuple(stream, std::get<reduce_sum_params>(eq.get_params()).axes);
            stream << ']';
            break;
        }

        case primitive_op::BROADCAST_IN_DIM: {
            const auto& params = std::get<broadcast_in_dim_params>(eq.get_params());
            stream << "[shape=";
            emit_tuple(stream, params.shape);
            stream << ", broadcast_dimensions=";
            emit_tuple(stream, params.broadcast_dimensions);
            stream << ']';
            break;
        }

        case primitive_op::DOT_GENERAL: {
            const auto& params = std::get<dot_general_params>(eq.get_params());
            stream << "[dimension_numbers=(";
            emit_tuple(stream, params.left_contract);
            stream << ", ";
            emit_tuple(stream, params.right_contract);
            stream << "), (";
            emit_tuple(stream, params.left_batch);
            stream << ", ";
            emit_tuple(stream, params.right_batch);
            stream << ")]";
            break;
        }
        default:
    }
    stream << ' ';
    return emit_value_vector(stream, eq.get_input(), " ");
}

std::ostream& operator<<(std::ostream& stream, const expression& expr) {
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
