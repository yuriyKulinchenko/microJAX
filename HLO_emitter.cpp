#include "HLO_emitter.h"

#include <sstream>
#include <format>
#include <limits>
#include <ranges>
#include <unordered_map>

#include "jax_array.h"

using namespace jax;

/*

StableHLO emission mirrors the structure of jax_logger: a jaxpr becomes a single
MLIR module holding a func.func @main. The invars become the function arguments,
each equation becomes one (or a few) stablehlo ops, and the outvals become the
returned values.

Unlike the internal jaxpr printer, SSA names cannot simply reuse variable ids:
nested sub-jaxprs (cond branches, scan bodies) have their own variable domains, so
ids collide across scopes. Instead every scope carries a var_id -> SSA name map, and
all names are drawn from a single global counter. Literal operands, which stablehlo
ops cannot take inline, are materialised as their own stablehlo.constant.

*/

namespace {

    using name_map = std::unordered_map<size_t, std::string>;

    std::string_view stablehlo_dtype(dtype_t dtype) {
        switch (dtype) {
            case dtype_t::F32: return "f32";
            case dtype_t::F64: return "f64";
            case dtype_t::I32: return "i32";
            case dtype_t::I64: return "i64";
            case dtype_t::BOOL: return "i1";
        }
        return "";
    }

    std::string type_string(const type_t& type) {
        std::string result = "tensor<";
        for (size_t dim: type.get_shape()) {
            result += std::to_string(dim);
            result += 'x';
        }
        result += stablehlo_dtype(type.get_dtype());
        result += '>';
        return result;
    }

    std::string int_list(const std::vector<size_t>& values) {
        std::string result = "[";
        for (size_t i = 0; i < values.size(); i++) {
            result += std::to_string(values[i]);
            if (i != values.size() - 1) result += ", ";
        }
        result += ']';
        return result;
    }

    std::string int_array(const std::vector<size_t>& values) {
        if (values.empty()) return "array<i64>";
        std::string result = "array<i64: ";
        for (size_t i = 0; i < values.size(); i++) {
            result += std::to_string(values[i]);
            if (i != values.size() - 1) result += ", ";
        }
        result += '>';
        return result;
    }

    std::string_view binary_op_name(primitive_op op) {
        using enum primitive_op;
        switch (op) {
            case ADD: return "stablehlo.add";
            case SUB: return "stablehlo.subtract";
            case MUL: return "stablehlo.multiply";
            case DIV: return "stablehlo.divide";
            case MAX: return "stablehlo.maximum";
            case MIN: return "stablehlo.minimum";
            case POW: return "stablehlo.power";
            default: return "";
        }
    }

    std::string_view unary_op_name(primitive_op op) {
        using enum primitive_op;
        switch (op) {
            case NEG: return "stablehlo.negate";
            case SIN: return "stablehlo.sine";
            case COS: return "stablehlo.cosine";
            case EXP: return "stablehlo.exponential";
            case LOG: return "stablehlo.log";
            case SQRT: return "stablehlo.sqrt";
            case RSQRT: return "stablehlo.rsqrt";
            case TANH: return "stablehlo.tanh";
            case LOGISTIC: return "stablehlo.logistic";
            default: return "";
        }
    }

    std::string_view compare_direction(primitive_op op) {
        using enum primitive_op;
        switch (op) {
            case EQ: return "EQ";
            case NE: return "NE";
            case LT: return "LT";
            case LE: return "LE";
            case GT: return "GT";
            case GE: return "GE";
            default: return "";
        }
    }

    std::string_view reduce_op_name(primitive_op op) {
        using enum primitive_op;
        switch (op) {
            case REDUCE_SUM: return "stablehlo.add";
            case REDUCE_MAX: return "stablehlo.maximum";
            case REDUCE_MIN: return "stablehlo.minimum";
            default: return "";
        }
    }

    std::string_view scatter_combine_name(primitive_op op) {
        using enum primitive_op;
        switch (op) {
            case SCATTER_ADD: return "stablehlo.add";
            case SCATTER_MUL: return "stablehlo.multiply";
            case SCATTER_MAX: return "stablehlo.maximum";
            case SCATTER_MIN: return "stablehlo.minimum";
            default: return "";
        }
    }

    // Identity element for a reduction: 0 for sum, and the dtype's extremes for
    // max/min. Finite extremes are safe: real data never exceeds them.
    double reduce_init(primitive_op op, dtype_t dtype) {
        using enum primitive_op;
        if (op == REDUCE_SUM) return 0;

        const bool is_max = (op == REDUCE_MAX);
        switch (dtype) {
            case dtype_t::F32:
                return is_max ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
            case dtype_t::F64:
                return is_max ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::max();
            case dtype_t::I32:
                return is_max ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
            case dtype_t::I64:
                return is_max ? static_cast<double>(std::numeric_limits<int64_t>::min())
                              : static_cast<double>(std::numeric_limits<int64_t>::max());
            default:
                return 0;
        }
    }

    class HLO_emitter {
    public:
        explicit HLO_emitter(const expression& jaxpr): top(jaxpr) {}

        std::string emit() {
            stream << "module {\n";

            name_map names;
            stream << "  func.func @main(";
            for (size_t i = 0; i < top.invars.size(); i++) {
                const var_t& invar = top.invars[i];
                std::string name = fresh();
                names[invar.get_id()] = name;
                stream << name << ": " << type_string(invar.get_type());
                if (i != top.invars.size() - 1) stream << ", ";
            }
            stream << ") -> " << result_types(top.outvals) << " {\n";

            std::vector<std::string> out_names = emit_equations(top, names, "    ");

            stream << "    return " << join(out_names) << " : " << join_types(top.outvals) << '\n';
            stream << "  }\n}\n";
            return stream.str();
        }

    private:
        const expression& top;
        std::ostringstream stream;
        size_t next_ssa = 0;

        std::string fresh() {
            return '%' + std::to_string(next_ssa++);
        }

        static std::string join(const std::vector<std::string>& names) {
            std::string result;
            for (size_t i = 0; i < names.size(); i++) {
                result += names[i];
                if (i != names.size() - 1) result += ", ";
            }
            return result;
        }

        std::string join_types(const std::vector<value>& values) {
            std::string result;
            for (size_t i = 0; i < values.size(); i++) {
                result += type_string(values[i].get_type());
                if (i != values.size() - 1) result += ", ";
            }
            return result;
        }

        std::string result_types(const std::vector<value>& values) {
            if (values.size() == 1) return type_string(values[0].get_type());
            return '(' + join_types(values) + ')';
        }

        std::string output_result_types(const std::vector<var_t>& outputs) {
            std::string joined;
            for (size_t i = 0; i < outputs.size(); i++) {
                joined += type_string(outputs[i].get_type());
                if (i != outputs.size() - 1) joined += ", ";
            }
            if (outputs.size() == 1) return joined;
            return '(' + joined + ')';
        }

        void emit_scalar_value(dtype_t dtype, double value) {
            if (dtype == dtype_t::BOOL) {
                stream << (value != 0 ? "true" : "false");
            } else if (is_integral(dtype)) {
                stream << static_cast<int64_t>(value);
            } else {
                stream << std::format("{:e}", value);
            }
        }

        void emit_dense_array(const array_t& array) {
            const std::vector<size_t>& shape = array.get_type().get_shape();
            dtype_t dtype = array.get_type().get_dtype();
            std::span<const double> values = array.get_value();

            if (shape.empty()) {
                emit_scalar_value(dtype, values[0]);
                return;
            }

            std::vector<size_t> index(shape.size(), 0);
            size_t open = shape.size();

            for (double element: values) {
                while (open > 0) { stream << '['; open--; }
                emit_scalar_value(dtype, element);
                for (size_t j = index.size(); j-- > 0;) {
                    if (index[j] < shape[j] - 1) {
                        index[j]++;
                        stream << ", ";
                        break;
                    }
                    index[j] = 0;
                    stream << ']';
                    open++;
                }
            }
        }

        // Emits a scalar (splat) constant and returns its name.
        std::string scalar_constant(dtype_t dtype, double value, const std::string& indent) {
            std::string name = fresh();
            stream << indent << name << " = stablehlo.constant dense<";
            emit_scalar_value(dtype, value);
            stream << "> : tensor<" << stablehlo_dtype(dtype) << ">\n";
            return name;
        }

        // Emits a splat constant of the given (possibly non-scalar) type.
        std::string splat_constant(const type_t& type, double value, const std::string& indent) {
            std::string name = fresh();
            stream << indent << name << " = stablehlo.constant dense<";
            emit_scalar_value(type.get_dtype(), value);
            stream << "> : " << type_string(type) << '\n';
            return name;
        }

        std::string resolve(const value& val, name_map& names, const std::string& indent) {
            if (val.is_var()) {
                return names.at(val.get_var().get_id());
            }
            const literal_t& literal = val.get_literal();
            return scalar_constant(literal.get_dtype(), literal.get_value(), indent);
        }

        // Emits the constvars and equations of an expression into the current scope,
        // returning the SSA names of its outvals. Callers emit the terminator.
        std::vector<std::string> emit_equations(
            const expression& expr, name_map& names, const std::string& indent) {

            for (const auto& [constvar, const_array]: std::views::zip(expr.constvars, expr.consts)) {
                std::string name = fresh();
                names[constvar.get_id()] = name;
                stream << indent << name << " = stablehlo.constant dense<";
                emit_dense_array(const_array);
                stream << "> : " << type_string(constvar.get_type()) << '\n';
            }

            for (const equation& eq: expr.equations) {
                emit_equation(eq, names, indent);
            }

            std::vector<std::string> out_names;
            for (const value& val: expr.outvals) out_names.push_back(resolve(val, names, indent));
            return out_names;
        }

        void emit_equation(const equation& eq, name_map& names, const std::string& indent) {
            using enum primitive_op;

            std::vector<std::string> operands;
            for (const value& val: eq.get_input()) operands.push_back(resolve(val, names, indent));

            std::vector<std::string> outputs;
            for (const var_t& var: eq.get_output()) {
                std::string name = fresh();
                names[var.get_id()] = name;
                outputs.push_back(name);
            }

            const std::string& out = outputs[0];
            const type_t& out_type = eq.get_output(0).get_type();
            const std::string out_type_string = type_string(out_type);

            auto in_type = [&](size_t i) { return type_string(eq.get_input(i).get_type()); };

            switch (eq.get_op()) {
                case ADD: case SUB: case MUL: case DIV: case MAX: case MIN: case POW:
                    stream << indent << out << " = " << binary_op_name(eq.get_op())
                        << ' ' << operands[0] << ", " << operands[1] << " : " << out_type_string << '\n';
                    break;

                case NEG: case SIN: case COS: case EXP: case LOG:
                case SQRT: case RSQRT: case TANH: case LOGISTIC:
                    stream << indent << out << " = " << unary_op_name(eq.get_op())
                        << ' ' << operands[0] << " : " << out_type_string << '\n';
                    break;

                case EQ: case NE: case LT: case LE: case GT: case GE:
                    stream << indent << out << " = stablehlo.compare "
                        << compare_direction(eq.get_op()) << ", " << operands[0] << ", " << operands[1]
                        << " : (" << in_type(0) << ", " << in_type(1) << ") -> " << out_type_string << '\n';
                    break;

                case CONVERT_ELEMENT_TYPE:
                    stream << indent << out << " = stablehlo.convert " << operands[0]
                        << " : (" << in_type(0) << ") -> " << out_type_string << '\n';
                    break;

                case TRANSPOSE:
                    stream << indent << out << " = stablehlo.transpose " << operands[0]
                        << ", dims = " << int_list(std::get<transpose_params>(eq.get_params()).permutation)
                        << " : (" << in_type(0) << ") -> " << out_type_string << '\n';
                    break;

                case RESHAPE:
                    stream << indent << out << " = stablehlo.reshape " << operands[0]
                        << " : (" << in_type(0) << ") -> " << out_type_string << '\n';
                    break;

                case BROADCAST_IN_DIM:
                    stream << indent << out << " = stablehlo.broadcast_in_dim " << operands[0]
                        << ", dims = " << int_list(std::get<broadcast_in_dim_params>(eq.get_params()).broadcast_dimensions)
                        << " : (" << in_type(0) << ") -> " << out_type_string << '\n';
                    break;

                case CONCATENATE:
                    stream << indent << out << " = stablehlo.concatenate " << join(operands)
                        << ", dim = " << std::get<concatenate_params>(eq.get_params()).dimension
                        << " : (" << join_types(eq.get_input()) << ") -> " << out_type_string << '\n';
                    break;

                case SLICE: {
                    const auto& params = std::get<slice_params>(eq.get_params());
                    stream << indent << out << " = stablehlo.slice " << operands[0] << " [";
                    for (size_t i = 0; i < params.start_indices.size(); i++) {
                        stream << params.start_indices[i] << ':' << params.limit_indices[i]
                            << ':' << params.strides[i];
                        if (i != params.start_indices.size() - 1) stream << ", ";
                    }
                    stream << "] : (" << in_type(0) << ") -> " << out_type_string << '\n';
                    break;
                }

                case PAD: {
                    const auto& config = std::get<pad_params>(eq.get_params()).padding_config;
                    std::vector<size_t> low, high, interior;
                    for (const auto& [l, h, i]: config) { low.push_back(l); high.push_back(h); interior.push_back(i); }
                    stream << indent << out << " = stablehlo.pad " << operands[0] << ", " << operands[1]
                        << ", low = " << int_list(low) << ", high = " << int_list(high)
                        << ", interior = " << int_list(interior)
                        << " : (" << in_type(0) << ", " << in_type(1) << ") -> " << out_type_string << '\n';
                    break;
                }

                case DOT_GENERAL: {
                    const auto& params = std::get<dot_general_params>(eq.get_params());
                    stream << indent << out << " = stablehlo.dot_general " << operands[0] << ", " << operands[1]
                        << ", contracting_dims = " << int_list(params.left_contract)
                        << " x " << int_list(params.right_contract)
                        << ", batching_dims = " << int_list(params.left_batch)
                        << " x " << int_list(params.right_batch)
                        << " : (" << in_type(0) << ", " << in_type(1) << ") -> " << out_type_string << '\n';
                    break;
                }

                case REDUCE_SUM: case REDUCE_MAX: case REDUCE_MIN: {
                    const std::vector<size_t>& axes = std::invoke([&]() -> const std::vector<size_t>& {
                        if (eq.get_op() == REDUCE_SUM) return std::get<reduce_sum_params>(eq.get_params()).axes;
                        if (eq.get_op() == REDUCE_MAX) return std::get<reduce_max_params>(eq.get_params()).axes;
                        return std::get<reduce_min_params>(eq.get_params()).axes;
                    });
                    dtype_t dtype = out_type.get_dtype();
                    std::string init = scalar_constant(dtype, reduce_init(eq.get_op(), dtype), indent);
                    stream << indent << out << " = stablehlo.reduce(" << operands[0] << " init: " << init
                        << ") applies " << reduce_op_name(eq.get_op())
                        << " across dimensions = " << int_list(axes)
                        << " : (" << in_type(0) << ", tensor<" << stablehlo_dtype(dtype) << ">) -> "
                        << out_type_string << '\n';
                    break;
                }

                case SELECT: {
                    // microJAX select is n-way: output[i] = case_{pred[i]}[i]. Only the 2-way
                    // (boolean) form maps onto stablehlo.select, with reversed cases:
                    // pred == 0 -> input(1), pred == 1 -> input(2).
                    if (eq.get_input().size() != 3) {
                        throw std::logic_error("Error: HLO emission only supports 2-way select");
                    }
                    stream << indent << out << " = stablehlo.select " << operands[0]
                        << ", " << operands[2] << ", " << operands[1]
                        << " : " << in_type(0) << ", " << out_type_string << '\n';
                    break;
                }

                case INTEGER_POW: {
                    size_t y = std::get<integer_pow_params>(eq.get_params()).y;
                    if (y == 0) {
                        stream << indent << out << " = stablehlo.constant dense<";
                        emit_scalar_value(out_type.get_dtype(), 1);
                        stream << "> : " << out_type_string << '\n';
                        break;
                    }
                    std::string acc = splat_constant(out_type, 1, indent);
                    for (size_t k = 1; k <= y; k++) {
                        std::string next = (k == y) ? out : fresh();
                        stream << indent << next << " = stablehlo.multiply " << acc << ", " << operands[0]
                            << " : " << out_type_string << '\n';
                        acc = next;
                    }
                    break;
                }

                case GATHER: emit_gather(eq, operands, out, out_type_string, indent); break;

                case SCATTER_ADD: case SCATTER_MUL: case SCATTER_MAX: case SCATTER_MIN: case SCATTER:
                    emit_scatter(eq, operands, out, out_type_string, indent); break;

                case COND: emit_cond(eq, operands, outputs, names, indent); break;

                case SCAN: emit_scan(eq, operands, outputs, names, indent); break;

                default:
                    throw std::logic_error(std::format(
                        "Error: HLO emission not implemented for {}", to_string(eq.get_op())));
            }
        }

        // gather: operand x [n, m...], indices [l...] into axis 0, result [l..., m...].
        void emit_gather(const equation& eq, const std::vector<std::string>& operands,
            const std::string& out, const std::string& out_type_string, const std::string& indent) {

            const type_t& operand_type = eq.get_input(0).get_type();
            const type_t& indices_type = eq.get_input(1).get_type();
            size_t operand_rank = operand_type.get_shape().size();
            size_t indices_rank = indices_type.get_shape().size();

            std::vector<size_t> offset_dims;
            for (size_t d = 1; d < operand_rank; d++) offset_dims.push_back(indices_rank + d - 1);

            std::vector<size_t> slice_sizes{1};
            for (size_t d = 1; d < operand_rank; d++) slice_sizes.push_back(operand_type.get_shape()[d]);

            stream << indent << out << " = \"stablehlo.gather\"(" << operands[0] << ", " << operands[1] << ") {\n";
            stream << indent << "  dimension_numbers = #stablehlo.gather<offset_dims = " << int_list(offset_dims)
                << ", collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = " << indices_rank << ">,\n";
            stream << indent << "  indices_are_sorted = false,\n";
            stream << indent << "  slice_sizes = " << int_array(slice_sizes) << "\n";
            stream << indent << "} : (" << type_string(operand_type) << ", " << type_string(indices_type)
                << ") -> " << out_type_string << '\n';
        }

        // scatter: operand [n, m...], indices [l...] into axis 0, updates [l..., m...], result [n, m...].
        void emit_scatter(const equation& eq, const std::vector<std::string>& operands,
            const std::string& out, const std::string& out_type_string, const std::string& indent) {

            const type_t& operand_type = eq.get_input(0).get_type();
            const type_t& indices_type = eq.get_input(1).get_type();
            const type_t& updates_type = eq.get_input(2).get_type();
            size_t operand_rank = operand_type.get_shape().size();
            size_t indices_rank = indices_type.get_shape().size();

            std::vector<size_t> update_window_dims;
            for (size_t d = 1; d < operand_rank; d++) update_window_dims.push_back(indices_rank + d - 1);

            dtype_t dtype = operand_type.get_dtype();
            std::string scalar = "tensor<" + std::string(stablehlo_dtype(dtype)) + ">";
            std::string a = fresh(), b = fresh();

            stream << indent << out << " = \"stablehlo.scatter\"(" << operands[0] << ", " << operands[1]
                << ", " << operands[2] << ") <{\n";
            stream << indent << "  scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = "
                << int_list(update_window_dims)
                << ", inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = "
                << indices_rank << ">,\n";
            stream << indent << "  indices_are_sorted = false,\n";
            stream << indent << "  unique_indices = false\n";
            stream << indent << "}> ({\n";
            stream << indent << "^bb0(" << a << ": " << scalar << ", " << b << ": " << scalar << "):\n";
            if (eq.get_op() == primitive_op::SCATTER) {
                stream << indent << "  stablehlo.return " << b << " : " << scalar << '\n';
            } else {
                std::string r = fresh();
                stream << indent << "  " << r << " = " << scatter_combine_name(eq.get_op())
                    << ' ' << a << ", " << b << " : " << scalar << '\n';
                stream << indent << "  stablehlo.return " << r << " : " << scalar << '\n';
            }
            stream << indent << "}) : (" << type_string(operand_type) << ", " << type_string(indices_type)
                << ", " << type_string(updates_type) << ") -> " << out_type_string << '\n';
        }

        // cond: input(0) is a scalar index, inputs 1.. are the branch operands. Each branch
        // is inlined as a case region whose invars are bound to those operands.
        void emit_cond(const equation& eq, const std::vector<std::string>& operands,
            const std::vector<std::string>& outputs, name_map&, const std::string& indent) {

            const auto& branches = std::get<cond_params>(eq.get_params()).branches;
            const std::string region_indent = indent + "  ";

            // stablehlo.case selects on an i32 index; convert a boolean/other index.
            std::string index = operands[0];
            const type_t& index_type = eq.get_input(0).get_type();
            if (index_type.get_dtype() != dtype_t::I32) {
                std::string converted = fresh();
                stream << indent << converted << " = stablehlo.convert " << index
                    << " : (" << type_string(index_type) << ") -> tensor<i32>\n";
                index = converted;
            }

            stream << indent << join(outputs) << " = \"stablehlo.case\"(" << index << ") (";
            for (size_t b = 0; b < branches.size(); b++) {
                const expression& branch = branches[b];
                stream << "{\n";

                name_map branch_names;
                for (size_t j = 0; j < branch.invars.size(); j++) {
                    branch_names[branch.invars[j].get_id()] = operands[1 + j];
                }
                std::vector<std::string> branch_out = emit_equations(branch, branch_names, region_indent);
                stream << region_indent << "stablehlo.return " << join(branch_out)
                    << " : " << join_types(branch.outvals) << '\n';

                stream << indent << "}";
                if (b != branches.size() - 1) stream << ", ";
            }
            stream << ") : (tensor<i32>) -> " << output_result_types(eq.get_output()) << '\n';
        }

        // scan lowers to stablehlo.while. Everything the body reads (consts, xs) is carried
        // through the loop unchanged to keep the loop self-contained; carries and stacked
        // y-accumulators are the values that actually evolve. The carried tuple is:
        //   (i, consts..., carries..., xs..., ys...)
        void emit_scan(const equation& eq, const std::vector<std::string>& operands,
            const std::vector<std::string>& outputs, name_map&, const std::string& indent) {

            const auto& params = std::get<scan_params>(eq.get_params());
            const expression& body = params.jaxpr;
            const size_t num_consts = params.num_consts;
            const size_t num_carry = params.num_carry;
            const size_t num_xs = eq.get_input().size() - num_consts - num_carry;
            const size_t num_ys = eq.get_output().size() - num_carry;
            const size_t length = params.length;

            const type_t i_type{dtype_t::I32};
            const std::string i_type_string = type_string(i_type);

            // Initial y-accumulators are zeros of the stacked output type.
            std::vector<std::string> y_init;
            for (size_t j = 0; j < num_ys; j++) {
                y_init.push_back(splat_constant(eq.get_output(num_carry + j).get_type(), 0, indent));
            }
            std::string i_init = scalar_constant(dtype_t::I32, 0, indent);
            std::string length_const = scalar_constant(dtype_t::I32, static_cast<double>(length), indent);

            // Loop-carried names and their types, in order: i, consts, carries, xs, ys.
            std::vector<std::string> iter_names;
            std::vector<std::string> init_names;
            std::vector<std::string> carried_types;
            std::vector<std::string> result_names;

            auto add_carried = [&](const std::string& init, const std::string& type, const std::string& result) {
                iter_names.push_back(fresh());
                init_names.push_back(init);
                carried_types.push_back(type);
                result_names.push_back(result);
            };

            add_carried(i_init, i_type_string, fresh());
            for (size_t j = 0; j < num_consts; j++) {
                add_carried(operands[j], type_string(eq.get_input(j).get_type()), fresh());
            }
            for (size_t j = 0; j < num_carry; j++) {
                add_carried(operands[num_consts + j],
                    type_string(eq.get_input(num_consts + j).get_type()), outputs[j]);
            }
            for (size_t j = 0; j < num_xs; j++) {
                add_carried(operands[num_consts + num_carry + j],
                    type_string(eq.get_input(num_consts + num_carry + j).get_type()), fresh());
            }
            for (size_t j = 0; j < num_ys; j++) {
                add_carried(y_init[j], type_string(eq.get_output(num_carry + j).get_type()),
                    outputs[num_carry + j]);
            }

            const std::string region_indent = indent + "  ";

            stream << indent << join(result_names) << " = stablehlo.while(";
            for (size_t k = 0; k < iter_names.size(); k++) {
                stream << iter_names[k] << " = " << init_names[k];
                if (k != iter_names.size() - 1) stream << ", ";
            }
            stream << ") : ";
            for (size_t k = 0; k < carried_types.size(); k++) {
                stream << carried_types[k];
                if (k != carried_types.size() - 1) stream << ", ";
            }
            stream << '\n';

            // cond: i < length
            const std::string& iter_i = iter_names[0];
            stream << indent << " cond {\n";
            std::string cmp = fresh();
            stream << region_indent << cmp << " = stablehlo.compare LT, " << iter_i << ", " << length_const
                << " : (" << i_type_string << ", " << i_type_string << ") -> tensor<i1>\n";
            stream << region_indent << "stablehlo.return " << cmp << " : tensor<i1>\n";
            stream << indent << " } do {\n";

            // Positions of each carried group within iter_names.
            const size_t consts_base = 1;
            const size_t carries_base = consts_base + num_consts;
            const size_t xs_base = carries_base + num_carry;
            const size_t ys_base = xs_base + num_xs;

            // Slice index si (reversed if requested).
            std::string si = iter_i;
            if (params.reverse) {
                std::string last = scalar_constant(dtype_t::I32, static_cast<double>(length - 1), region_indent);
                si = fresh();
                stream << region_indent << si << " = stablehlo.subtract " << last << ", " << iter_i
                    << " : " << i_type_string << '\n';
            }
            std::string zero_index = scalar_constant(dtype_t::I32, 0, region_indent);

            // x slices: dynamic_slice each xs at si along axis 0, then drop the leading 1.
            name_map body_names;
            for (size_t j = 0; j < num_consts; j++) {
                body_names[body.invars[j].get_id()] = iter_names[consts_base + j];
            }
            for (size_t j = 0; j < num_carry; j++) {
                body_names[body.invars[num_consts + j].get_id()] = iter_names[carries_base + j];
            }
            for (size_t j = 0; j < num_xs; j++) {
                const std::string& xs = iter_names[xs_base + j];
                const type_t& xs_type = eq.get_input(num_consts + num_carry + j).get_type();
                const type_t& slice_type = body.invars[num_consts + num_carry + j].get_type();
                size_t rank = xs_type.get_shape().size();

                std::vector<size_t> sizes{1};
                for (size_t d = 1; d < rank; d++) sizes.push_back(xs_type.get_shape()[d]);
                std::string stacked_type = type_string(type_t{xs_type.get_dtype(), sizes});

                std::string sliced = fresh();
                stream << region_indent << sliced << " = stablehlo.dynamic_slice " << xs << ", " << si;
                for (size_t d = 1; d < rank; d++) stream << ", " << zero_index;
                stream << ", sizes = " << int_list(sizes)
                    << " : (" << type_string(xs_type);
                for (size_t d = 0; d < rank; d++) stream << ", " << i_type_string;
                stream << ") -> " << stacked_type << '\n';

                std::string reshaped = fresh();
                stream << region_indent << reshaped << " = stablehlo.reshape " << sliced
                    << " : (" << stacked_type << ") -> " << type_string(slice_type) << '\n';
                body_names[body.invars[num_consts + num_carry + j].get_id()] = reshaped;
            }

            // Inline the body; its outvals are (new_carries..., y_slices...).
            std::vector<std::string> body_out = emit_equations(body, body_names, region_indent);

            // Write each y slice back into its accumulator via dynamic_update_slice.
            std::vector<std::string> new_ys;
            for (size_t j = 0; j < num_ys; j++) {
                const std::string& acc = iter_names[ys_base + j];
                const type_t& acc_type = eq.get_output(num_carry + j).get_type();
                const type_t& slice_type = body.outvals[num_carry + j].get_type();
                size_t rank = acc_type.get_shape().size();

                std::vector<size_t> stacked_shape{1};
                for (size_t d = 1; d < rank; d++) stacked_shape.push_back(acc_type.get_shape()[d]);
                std::string stacked_type = type_string(type_t{acc_type.get_dtype(), stacked_shape});

                std::string reshaped = fresh();
                stream << region_indent << reshaped << " = stablehlo.reshape " << body_out[num_carry + j]
                    << " : (" << type_string(slice_type) << ") -> " << stacked_type << '\n';

                std::string updated = fresh();
                stream << region_indent << updated << " = stablehlo.dynamic_update_slice " << acc
                    << ", " << reshaped << ", " << si;
                for (size_t d = 1; d < rank; d++) stream << ", " << zero_index;
                stream << " : (" << type_string(acc_type) << ", " << stacked_type;
                for (size_t d = 0; d < rank; d++) stream << ", " << i_type_string;
                stream << ") -> " << type_string(acc_type) << '\n';
                new_ys.push_back(updated);
            }

            // i_next = i + 1.
            std::string one = scalar_constant(dtype_t::I32, 1, region_indent);
            std::string i_next = fresh();
            stream << region_indent << i_next << " = stablehlo.add " << iter_i << ", " << one
                << " : " << i_type_string << '\n';

            // Return the evolved carried tuple.
            std::vector<std::string> returned;
            returned.push_back(i_next);
            for (size_t j = 0; j < num_consts; j++) returned.push_back(iter_names[consts_base + j]);
            for (size_t j = 0; j < num_carry; j++) returned.push_back(body_out[j]);
            for (size_t j = 0; j < num_xs; j++) returned.push_back(iter_names[xs_base + j]);
            for (size_t j = 0; j < num_ys; j++) returned.push_back(new_ys[j]);

            stream << region_indent << "stablehlo.return " << join(returned) << " : ";
            for (size_t k = 0; k < carried_types.size(); k++) {
                stream << carried_types[k];
                if (k != carried_types.size() - 1) stream << ", ";
            }
            stream << '\n';
            stream << indent << " }\n";
        }
    };
}

std::string emit_stablehlo(const expression& jaxpr) {
    HLO_emitter emitter{jaxpr};
    return emitter.emit();
}
