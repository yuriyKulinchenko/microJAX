#include "jax_logger.h"
#include "helper.h"

using namespace jax;

std::ostream& operator<<(std::ostream& stream, const type_t& type) {
    stream << to_lower(to_string(type.get_dtype()));
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

std::ostream& operator<<(std::ostream& stream, const literal_t& literal) {
    return stream << literal.get_value();
}

std::ostream& emit_typed_var(std::ostream& stream, const var_t& var) {
    return stream << var << ':' << var.get_type();
}

std::ostream& emit_typed_literal(std::ostream& stream, const literal_t& literal) {
    return stream << literal << ':' << to_lower(to_string(literal.get_dtype())) << "[]";
}

// <var>':'<type>* '=' <op> (<var> | <array>':'<type>)


std::ostream& emit_value_vector(std::ostream& stream,
    const std::vector<value>& values, const char* separator) {
    for (int i = 0; i < values.size(); i++) {
        const value& val = values[i];
        if (val.is<literal_t>()) {
            const literal_t& literal = val.get_literal();
            emit_typed_literal(stream, literal);
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

std::ostream& emit_equation(std::ostream& stream, const equation& eq, size_t tab_count) {
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

        case primitive_op::REDUCE_MAX: {
            stream << "[axes=";
            emit_tuple(stream, std::get<reduce_max_params>(eq.get_params()).axes);
            stream << ']';
            break;
        }

        case primitive_op::REDUCE_MIN: {
            stream << "[axes=";
            emit_tuple(stream, std::get<reduce_min_params>(eq.get_params()).axes);
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

        case primitive_op::CONVERT_ELEMENT_TYPE: {
            const auto& params = std::get<convert_element_type_params>(eq.get_params());
            stream << "[new_dtype=" << to_lower(to_string(params.new_dtype)) << ']';
            break;
        }

        case primitive_op::COND: {
            const auto& params = std::get<cond_params>(eq.get_params());
            stream << "[branches=(\n";
            for (auto& branch: params.branches) {
                emit_expr(stream, branch, tab_count + 1);
            }
            stream << std::string(4 * tab_count, ' ');
            stream << "    )]";
            break;
        }

        case primitive_op::SCAN: {
            std::string space(4 * tab_count, ' ');
            const auto& params = std::get<scan_params>(eq.get_params());
            stream << "[jaxpr=\n";
            emit_expr(stream, params.jaxpr, tab_count + 1);
            stream << space << "    , length=" << params.length
            << ", num_consts=" << params.num_consts
            << ", num_carry=" << params.num_carry
            << (params.reverse ? ", reverse=true": "") << ']';
            break;
        }

        case primitive_op::RESHAPE: {
            stream << "[new_sizes=";
            emit_tuple(stream, std::get<reshape_params>(eq.get_params()).new_sizes);
            stream << ']';
            break;
        }

        case primitive_op::CONCATENATE: {
            stream << "[dimension=" << std::get<concatenate_params>(eq.get_params()).dimension << ']';
            break;
        }

        case primitive_op::SLICE: {
            const auto& params = std::get<slice_params>(eq.get_params());
            stream << "[start_indices=";
            emit_tuple(stream, params.start_indices);
            stream << ", limit_indices=";
            emit_tuple(stream, params.limit_indices);
            stream << ", strides=";
            emit_tuple(stream, params.strides);
            stream << ']';
            break;
        }

        case primitive_op::PAD: {
            const auto& params = std::get<pad_params>(eq.get_params());
            stream << "[padding_config=(";
            for (size_t i = 0; i < params.padding_config.size(); i++) {
                auto [low, high, interior] = params.padding_config[i];
                stream << '(' << low << ", " << high << ", " << interior << ')';
                if (i != params.padding_config.size() - 1) stream << ", ";
            }
            stream << ")]";
            break;
        }

        case primitive_op::INTEGER_POW: {
            stream << "[y=" << std::get<integer_pow_params>(eq.get_params()).y << ']';
            break;
        }

        default:
    }
    stream << ' ';
    return emit_value_vector(stream, eq.get_input(), " ");
}

std::ostream& operator<<(std::ostream& stream, const equation& eq) {
    return emit_equation(stream, eq, 0);
}

std::ostream& emit_expr(std::ostream& stream, const expression& expr, size_t tab_count) {
    std::string space(4 * tab_count, ' ');

    stream << space << "{\n";
    stream << space << "    " << "lambda ";

    for (size_t i = 0; i < expr.constvars.size(); i++) {
        auto& var = expr.constvars[i];
        emit_typed_var(stream, var);
        if (i != expr.constvars. size() - 1) stream << ", ";
    }

    stream << "; ";

    for (size_t i = 0; i < expr.invars.size(); i++) {
        auto& var = expr.invars[i];
        emit_typed_var(stream, var);
        if (i != expr.invars.size() - 1) stream << ", ";
    }

    stream << ". let\n";

    for (auto& eq: expr.equations) {
        stream << space << "    ";
        emit_equation(stream, eq, tab_count) << '\n';
    }

    stream << space << "    " << "in (";
    emit_value_vector(stream, expr.outvals, ", ");
    if (expr.outvals.size() == 1) stream << ',';
    stream << ")\n";

    stream << space <<  "}\n";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const expression& expr) {
    return emit_expr(stream, expr, 0);
}
