#ifndef MICROJAX_JAX_ARRAY_H
#define MICROJAX_JAX_ARRAY_H

#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <array>

#include "helper.h"

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float; // Temporary: these may not be true
using f64 = double;
using b8 = uint8_t; // byte-backed boolean

class jaxpr_tracer;

namespace jax {

#define TYPE_ENUM_LIST(X) \
    X(I32) X(I64) X(F32) X(F64) X(BOOL)

    enum class dtype_t {
    #define X(name) name,
        TYPE_ENUM_LIST(X)
    #undef X
    };

    std::string_view to_string(dtype_t t);

    bool is_floating(dtype_t t);
    bool is_integral(dtype_t t);
    dtype_t resultant_type(dtype_t t1, dtype_t t2);

    struct type_span;

    class type_t {
    public:

        type_t(dtype_t dtype);

        type_t(dtype_t dtype, std::vector<size_t> shape);

        type_t(const type_span& span);

        template<std::convertible_to<size_t>... Dims>
        explicit type_t(dtype_t dtype, Dims... shape)
        : dtype(dtype), shape{static_cast<size_t>(shape)...} {}


        bool operator==(const type_t& other) const;

        bool is_i32() const;
        bool is_i64() const;
        bool is_f32() const;
        bool is_f64() const;
        bool is_bool() const;

        [[nodiscard]] dtype_t get_dtype() const;
        [[nodiscard]] const std::vector<size_t>& get_shape() const;
        [[nodiscard]] std::vector<size_t>& get_shape();

        [[nodiscard]] operator type_span() const;

        void set_dtype(dtype_t new_dtype);

    private:
        dtype_t dtype;
        std::vector<size_t> shape;
    };

    std::vector<size_t> new_get_shape(const std::vector<size_t>& x_shape,
        const std::vector<size_t>& idx_shape);
    void validate_scatter_op_shapes(const std::vector<size_t>& x_shape,
        const std::vector<size_t>& idx_shape, const std::vector<size_t>& update_shape);

    class literal_t {
    public:
        literal_t(dtype_t dtype, double value);

        [[nodiscard]] dtype_t get_dtype() const;
        [[nodiscard]] double get_value() const;
        [[nodiscard]] static const std::vector<size_t>& get_shape();

    private:
        dtype_t dtype;
        double value;
        inline static std::vector<size_t> shape = {};
    };

    struct implicit_broadcast_result {
        std::vector<size_t> new_shape;
        std::vector<size_t> left_broadcast_dimensions;
        std::vector<size_t> right_broadcast_dimensions;
    };

    implicit_broadcast_result get_implicit_broadcast_result(std::span<const size_t> left_shape,
        std::span<const size_t> right_shape);

    template<typename A>
    concept array_like = requires(const A a, std::span<const size_t> indices, size_t i)
    {
        {a.index(indices)};
        {a.index(i)};
        {a.get_type()};
        {a.get_value()} -> std::convertible_to<std::span<const double>>;
        {a[indices]} -> std::convertible_to<double>;
        {a.get_strides()} -> std::convertible_to<std::span<const size_t>>;
    };

    class array_index;
    class array_tracer_index;

    template<bool Const>
    class array_span_t;

    using const_array_span_t = array_span_t<true>;
    using mutable_array_span_t = array_span_t<false>;

    class array_t {
    public:
        friend const_array_span_t;
        friend mutable_array_span_t;

        array_t();
        array_t(type_t type, std::vector<double> value);

        static array_t build(dtype_t dtype, std::vector<size_t> shape,
            const std::function<double(const std::vector<size_t>&)>& f);

        static array_t build_fill(type_t type, double value);
        static array_t zeros(std::vector<size_t> shape, dtype_t dtype = dtype_t::F32);

        // ReSharper disable once CppNonExplicitConvertingConstructor
        array_t(double value);

        double& operator[](std::span<const size_t> indices);
        const double& operator[](std::span<const size_t> indices) const;

        [[nodiscard]] bool operator==(const array_t& other) const;

        [[nodiscard]] array_t transpose(const std::vector<size_t>& permutation) const;
        [[nodiscard]] array_t reduce_sum(const std::vector<size_t>& axes) const;
        [[nodiscard]] array_t reduce_max(const std::vector<size_t>& axes) const;
        [[nodiscard]] array_t reduce_min(const std::vector<size_t>& axes) const;
        [[nodiscard]] array_t reshape(std::vector<size_t> shape) const;
        [[nodiscard]] array_t convert_element_type(dtype_t dtype) const;

        [[nodiscard]] array_t broadcast_in_dim(
            const std::vector<size_t>& shape,
            const std::vector<size_t>& broadcast_dimensions) const;

        [[nodiscard]] array_t dot_general(
            const array_t& other,
            const std::vector<size_t>& left_contract,
            const std::vector<size_t>& right_contract,
            const std::vector<size_t>& left_batch,
            const std::vector<size_t>& right_batch) const;

        [[nodiscard]] const type_t& get_type() const;
        [[nodiscard]] type_t& get_type();
        [[nodiscard]] const std::vector<double>& get_value() const;
        [[nodiscard]] std::vector<double>& get_value();
        [[nodiscard]] const std::vector<size_t>& get_strides() const;

        [[nodiscard]] std::optional<literal_t> get_literal() const;

        [[nodiscard]] array_tracer_index at(jaxpr_tracer idx);
        [[nodiscard]] array_index at(array_t idx);

        void set_type(type_t new_type);

        [[nodiscard]] mutable_array_span_t index(std::span<const size_t> indices);
        [[nodiscard]] const_array_span_t index(std::span<const size_t> indices) const;

        [[nodiscard]] mutable_array_span_t index(size_t i);
        [[nodiscard]] const_array_span_t index(size_t i) const;

        [[nodiscard]] bool has_single_value(f64 val) const;
        [[nodiscard]] bool has_single_value() const;
        void compute_strides();

    private:
        template<double Identity, double (*BinaryOp)(double, double)>
        [[nodiscard]] array_t reduce_monoid(const std::vector<size_t>& axes) const;
        void check_single_value();

        type_t type;
        std::vector<double> value;
        std::vector<size_t> strides;
        f64 single_value = 0;
        bool has_single_value_ = false;
    };

    class array_index {
    public:
        array_index(array_t x, array_t idx);

        array_t get();

        jaxpr_tracer set(const jaxpr_tracer& update);
        jaxpr_tracer add(const jaxpr_tracer& update);
        jaxpr_tracer multiply(const jaxpr_tracer& update);
        jaxpr_tracer max(const jaxpr_tracer& update);
        jaxpr_tracer min(const jaxpr_tracer& update);

        template<array_like A>
        array_t set(const A& update);
        template<array_like A>
        array_t add(const A& update);
        template<array_like A>
        array_t multiply(const A& update);
        template<array_like A>
        array_t max(const A& update);
        template<array_like A>
        array_t min(const A& update);

    private:
        array_t x;
        array_t idx;
    };

    struct type_span {
        dtype_t dtype;
        std::span<const size_t> shape;

        [[nodiscard]] dtype_t get_dtype() const;
        [[nodiscard]] std::span<const size_t> get_shape() const;
    };

    // Non-owning view, potentially to a subset of the array

    template<bool Const>
    class array_span_t {
    public:
        using element_t = std::conditional_t<Const, const double, double>;

        array_span_t(const array_t& array, std::span<const size_t> indices = {});

        element_t& operator[](std::span<const size_t> indices) const;
        array_span_t index(std::span<const size_t> indices) const; // Preserves element_t
        array_span_t index(size_t i) const;

        [[nodiscard]] const type_span& get_type() const;
        [[nodiscard]] std::span<element_t> get_value() const;
        [[nodiscard]] std::span<const size_t> get_strides() const;

        [[nodiscard]] operator array_t() const;

    private:
        type_span type;
        std::span<const size_t> stride;
        std::span<const double> value;
    };

    template<size_t N>
    using value_array = std::array<array_t, N>;

    template<typename F, size_t num_consts, size_t num_carry, size_t num_xs>
    auto array_scan(F f,
        value_array<num_consts> consts, value_array<num_carry> carry,
        value_array<num_xs> xs, size_t L, bool reverse) {
        // f must be able to take array_t parameters
        // (consts..., ys[:1]...) = f(consts..., carry..., xs[:1]...)

        // Expect that output_types_t is a tuple of array_t
        constexpr size_t num_inputs = num_consts + num_carry + num_xs;
        using output_types_t = apply_result_t<F, value_array<num_inputs>>;
        constexpr size_t num_outputs = std::tuple_size_v<output_types_t>;
        constexpr size_t num_ys = num_outputs - num_carry;

        // Expected dimension of ys is given by output to f.
        // Important: this consumes carry.
        // input_at_index takes an i parameter, ranging over
        // (consts..., carry..., xs...),
        // and a t parameter which xs[t]...

        auto input_at_index = [&]<size_t i>(size_t t) -> array_t {
            if constexpr (i < num_consts) {
                return consts[i];
            } else if constexpr(i < num_consts + num_carry) {
                return std::move(carry[i - num_consts]);
            } else {
                return xs[i - num_consts - num_carry].index(t);
            }
        };

        auto project_input = [&](size_t t) -> value_array<num_outputs> {
            // Return tuple containing (carry..., ys[t])
            value_array<num_inputs> input = std::invoke(
                [&]<size_t... Is>(std::index_sequence<Is...>) -> value_array<num_inputs> {
                return {input_at_index.template operator()<Is>(t)...};
            }, std::make_index_sequence<num_inputs>());

            // Next goal: pass the input into f:
            auto output = std::apply(f, input);
            // Expect output to be tuple-like:

            return std::invoke(
                [&]<size_t... Is>(std::index_sequence<Is...>)-> value_array<num_outputs> {
                return {std::get<Is>(output)...};
            }, std::make_index_sequence<num_outputs>());
        };

        // TODO: Handle reverse

        value_array<num_ys> ys {};

        for (size_t t = 0; t < L; t++) {
            value_array<num_outputs> output = project_input(t);
            // Update carry:
            for (size_t j = 0; j < num_carry; j++) {
                carry[j] = output[j];
            }

            // Update correct slice of ys:

            for (size_t j = 0; j < num_ys; j++) {
                array_t& output_instance = output[num_carry + j];
                array_t& y_instance = ys[j];

                if (t == 0) {
                    std::vector<size_t>& shape = y_instance.get_type().get_shape();
                    shape = {L};
                    for (size_t x: output_instance.get_type().get_shape()) shape.push_back(x);
                    y_instance.get_value().clear();
                }

                std::vector<double>& value {y_instance.get_value()};

                value.append_range(output_instance.get_value());
            }
        }

        for (array_t& array: ys) array.compute_strides();

        // Return final value:

        auto output_index = [&]<size_t i>() -> array_t {
            if constexpr (i < num_carry) {
                return std::move(carry[i]);
            } else {
                return std::move(ys[i - num_carry]);
            }
        };

        using output_tuple_t = array_to_tuple_t<value_array<num_outputs>>;
        return std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) -> output_tuple_t {
            return {(output_index.template operator()<Is>())...};
        }, std::make_index_sequence<num_outputs>());
    }

    template<size_t N>
    array_t array_select(const array_t& pred, std::array<std::reference_wrapper<const array_t>, N> values) {
        static_assert(N > 0);
        // Depends on the shape of pred, values.
        // For now, expect shape(pred) = shape(values)...
        type_t type = values[0].get().get_type();
        array_t output = array_t::build_fill(std::move(type), 0.);
        for (size_t i = 0; i < output.get_value().size(); i++) {
            const size_t index = pred.get_value()[i];
            output.get_value()[i] = values[index].get().get_value()[i];
        }

        return output;
    }

    template<size_t N>
    array_t array_concatenate(std::array<array_t, N> values, size_t axis) {
        // Dimensions have to match:
        array_t& first_value = values[0];
        auto& first_shape = first_value.get_type().get_shape();
        dtype_t dtype = first_value.get_type().get_dtype();
        size_t rank = first_shape.size();
        size_t concat_size = first_shape[axis];

        for (array_t& val: values | std::views::drop(1)) {
            if (val.get_type().get_dtype() != dtype) {
                // TODO: maybe perform implicit conversion?
                throw std::logic_error("Error: dtype mismatch when attempting concatenation");
            }

            auto& current_shape = val.get_type().get_shape();
            if (current_shape.size() != rank) {
                throw std::logic_error("Error: rank mismatch when attempting concatenation");
            }

            concat_size += current_shape[axis];
            for (size_t r = 0; r < rank; r++) {
                if (r == axis) continue;
                if (current_shape[r] != first_shape[r]) {
                    throw std::logic_error("Error: dimension mismatch when attempting concatenation");
                }
            }
        }

        std::vector new_shape {first_shape};
        new_shape[axis] = concat_size;

        array_t result {type_t{dtype, new_shape}, std::vector<double>(num_elements(new_shape), 0)};

        size_t concatenate_dimension_offset = 0;
        for (size_t tensor_index = 0; tensor_index < N; tensor_index++) {
            const auto& source_shape = values[tensor_index].get_type().get_shape();
            std::vector<size_t> indicies (rank, 0);
            for (auto& is: cartesian_product{indicies, source_shape}) {
                double value = values[tensor_index][is];
                indicies[axis] += concatenate_dimension_offset;
                result[indicies] = value;
                indicies[axis] -= concatenate_dimension_offset;
            }
            concatenate_dimension_offset += source_shape[axis];
        }

        return result;
    }

    // All arithmetic:

    template<double(*UnaryOp)(double), array_like A>
    array_t array_unary_elementwise_op(const A& a) {
        return array_t{a.get_type(), a.get_value()
            | std::views::transform(UnaryOp)
            | std::ranges::to<std::vector<double>>()};
    }

    template<double(*BinaryOp)(double, double), array_like A, array_like B>
    array_t array_binary_elementwise_op(const A& a, const B& b, std::optional<dtype_t> out_dtype = std::nullopt) {
        const std::vector<size_t>& a_shape = a.get_type().get_shape();
        const std::vector<size_t>& b_shape = b.get_type().get_shape();

        dtype_t dtype = out_dtype.value_or(resultant_type(a.get_type().get_dtype(), b.get_type().get_dtype()));

        // Fast path: identical shapes need no broadcasting, so zip the buffers directly.
        if (std::ranges::equal(a_shape, b_shape)) {
            std::span<const double> a_value = a.get_value();
            std::span<const double> b_value = b.get_value();
            std::vector<double> result_value(a_value.size());
            for (size_t i = 0; i < result_value.size(); i++) {
                result_value[i] = BinaryOp(a_value[i], b_value[i]);
            }
            return array_t{type_t{dtype, a_shape}, std::move(result_value)};
        }

        implicit_broadcast_result broadcast = get_implicit_broadcast_result(a_shape, b_shape);
        const std::vector<size_t>& new_shape = broadcast.new_shape;

        array_t result {type_t{dtype, new_shape}, std::vector<double>(num_elements(new_shape))};

        const bool a_broadcasts = a_shape != new_shape;
        const bool b_broadcasts = b_shape != new_shape;

        // Maps an output index to an operand index: an operand axis of extent 1
        // is pinned to 0, otherwise it reads the output axis it broadcasts onto.
        std::vector<size_t> a_index(a_shape.size());
        std::vector<size_t> b_index(b_shape.size());
        auto operand_index = [](const std::vector<size_t>& out_index,
            const std::vector<size_t>& operand_shape,
            const std::vector<size_t>& broadcast_dimensions,
            std::vector<size_t>& scratch) -> std::span<const size_t> {
            for (size_t k = 0; k < operand_shape.size(); k++) {
                scratch[k] = operand_shape[k] == 1 ? 0 : out_index[broadcast_dimensions[k]];
            }
            return scratch;
        };

        std::vector<size_t> out_index(new_shape.size(), 0);
        for (const auto& out: cartesian_product{out_index, new_shape}) {
            double a_value = a_broadcasts
                ? a[operand_index(out, a_shape, broadcast.left_broadcast_dimensions, a_index)]
                : a[out];
            double b_value = b_broadcasts
                ? b[operand_index(out, b_shape, broadcast.right_broadcast_dimensions, b_index)]
                : b[out];
            result[out] = BinaryOp(a_value, b_value);
        }

        return result;
    }

    // This assumes a matching dimension: no broadcast
    template<double(*BinaryOp)(double, double), array_like A, array_like B>
    A& array_binary_inplace_op(A& a, const B& b) {
        if (a.get_value().size() != b.get_value().size()) {
            throw std::logic_error("Error: in-place op requires operands of matching size (no broadcast)");
        }
        for (auto&& [x, y]: std::views::zip(a.get_value(), b.get_value())) {
            x = BinaryOp(x, y);
        }
        return a;
    }

    // Scalar broadcast: applies BinaryOp(x, b) to every element in place.
    template<double(*BinaryOp)(double, double), array_like A>
    A& array_binary_inplace_scalar_op(A& a, double b) {
        for (auto&& x: a.get_value()) x = BinaryOp(x, b);
        return a;
    }

    template<double(*BinaryOp)(double, double), array_like A>
    array_t scatter_op(const array_t& x, const array_t& idx,  const A& update) {
        // If x: [n, ms...], idx: [ls...], update: [ls..., ms...], then .op(): [n, ms...]
        // Iterate through each idx, fetch an update slice, set the appropriate x.

        auto& idx_shape = idx.get_type().get_shape();

        std::vector<size_t> idx_indicies (idx_shape.size(), 0);

        array_t updated_x {x};

        for (auto& ls: cartesian_product{idx_indicies, idx_shape}) {
            size_t i = static_cast<size_t>(idx[ls]);
            auto update_slice = update.index(ls);
            auto dest_slice = updated_x.index(i);

            array_binary_inplace_op<BinaryOp>(dest_slice, update_slice);
        }

        return updated_x;
    }

#define ELEMENTWISE_UNARY_OP(op_name, return_expr)                                  \
    template<array_like A>                                                          \
    array_t op_name(const A& a) {                                                   \
        return array_unary_elementwise_op<[](double x){return (return_expr);}>(a);  \
    }

    ELEMENTWISE_UNARY_OP(sin, std::sin(x));
    ELEMENTWISE_UNARY_OP(cos, std::cos(x));
    ELEMENTWISE_UNARY_OP(exp, std::exp(x));
    ELEMENTWISE_UNARY_OP(log, std::log(x));
    ELEMENTWISE_UNARY_OP(operator-, -x);

    ELEMENTWISE_UNARY_OP(sqrt, std::sqrt(x));
    ELEMENTWISE_UNARY_OP(rsqrt, 1. / std::sqrt(x));
    ELEMENTWISE_UNARY_OP(tanh, std::tanh(x));
    ELEMENTWISE_UNARY_OP(logistic, 1. / (1. + std::exp(-x)));

#undef ELEMENTWISE_UNARY_OP

#define ELEMENTWISE_BINARY_OP_DTYPE(op_name, return_expr, out_dtype)            \
    template<array_like A, array_like B>                                        \
    array_t op_name(const A& a, const B& b) {                                   \
        return array_binary_elementwise_op<[](double x, double y) -> double     \
        {return (return_expr);}>(a, b, out_dtype);                              \
    }                                                                           \
    template<array_like A>                                                      \
    array_t op_name(const A& a, double b) { return op_name(a, array_t{b}); }    \
    template<array_like B>                                                      \
    array_t op_name(double a, const B& b) { return op_name(array_t{a}, b); }

    // Arithmetic keeps the promoted (resultant) dtype, comparisons yield BOOL.
#define ELEMENTWISE_BINARY_OP(op_name, return_expr) ELEMENTWISE_BINARY_OP_DTYPE(op_name, return_expr, std::nullopt)
#define ELEMENTWISE_COMPARISON_OP(op_name, return_expr) ELEMENTWISE_BINARY_OP_DTYPE(op_name, return_expr, dtype_t::BOOL)

    ELEMENTWISE_BINARY_OP(operator+, x + y);
    ELEMENTWISE_BINARY_OP(operator-, x - y);
    ELEMENTWISE_BINARY_OP(operator*, x * y);
    ELEMENTWISE_BINARY_OP(operator/, x / y);

    ELEMENTWISE_COMPARISON_OP(elementwise_equal, double_eq(x, y));
    ELEMENTWISE_COMPARISON_OP(elementwise_not_equal, !double_eq(x, y));
    ELEMENTWISE_COMPARISON_OP(operator<, x < y);
    ELEMENTWISE_COMPARISON_OP(operator>, x > y);
    ELEMENTWISE_COMPARISON_OP(operator<=, x <= y);
    ELEMENTWISE_COMPARISON_OP(operator>=, x >= y);

    ELEMENTWISE_BINARY_OP(min, std::min(x, y));
    ELEMENTWISE_BINARY_OP(max, std::max(x, y));
    ELEMENTWISE_BINARY_OP(pow, std::pow(x, y));

#undef ELEMENTWISE_COMPARISON_OP
#undef ELEMENTWISE_BINARY_OP
#undef ELEMENTWISE_BINARY_OP_DTYPE

    // 'integer_pow' raises to a static, non-negative integer power (INTEGER_POW).
    template<array_like A>
    array_t integer_pow(const A& a, size_t y) {
        std::span<const double> a_value = a.get_value();
        std::vector<double> result_value(a_value.size());
        for (size_t i = 0; i < a_value.size(); i++) {
            double acc = 1;
            for (size_t k = 0; k < y; k++) acc *= a_value[i];
            result_value[i] = acc;
        }
        return array_t{a.get_type(), std::move(result_value)};
    }

#define ARRAY_INDEX_SCATTER_OP(op_name, return_expr)                                        \
    template<array_like A>                                                                  \
    array_t array_index::op_name(const A& update) {                                         \
        return scatter_op<[](double x, double y){return (return_expr);}>(x, idx, update);   \
    }

    ARRAY_INDEX_SCATTER_OP(set, y);
    ARRAY_INDEX_SCATTER_OP(add, x + y);
    ARRAY_INDEX_SCATTER_OP(multiply, x * y);
    ARRAY_INDEX_SCATTER_OP(max, std::max(x, y));
    ARRAY_INDEX_SCATTER_OP(min, std::min(x, y));

#undef ARRAY_INDEX_SCATTER_OP


#define ASSIGN_OP(op_name, return_expr)                                                         \
    template<array_like A, array_like B>                                                        \
    A&& op_name(A&& a, const B& b) {                                                            \
        array_binary_inplace_op<[](double x, double y){return (return_expr);}>(a, b);           \
        return std::forward<A>(a);                                                              \
    }                                                                                           \
    template<array_like A>                                                                      \
    A&& op_name(A&& a, double b) {                                                              \
        array_binary_inplace_scalar_op<[](double x, double y){return (return_expr);}>(a, b);    \
        return std::forward<A>(a);                                                              \
    }

    ASSIGN_OP(operator+=, x + y);
    ASSIGN_OP(operator-=, x - y);
    ASSIGN_OP(operator*=, x * y);
    ASSIGN_OP(operator/=, x / y);

}


#endif //MICROJAX_JAX_ARRAY_H
