#ifndef JAXPR_TYPES_H
#define JAXPR_TYPES_H
#include <any>
#include <string_view>
#include <variant>
#include <vector>
#include <span>

#include "helper.h"

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float; // Temporary: these may not be true
using f64 = double;
using b8 = uint8_t; // byte-backed boolean

namespace jax {

#define PRIMITIVE_OP_LIST(X)                    \
    X(ADD) X(SUB) X(MUL) X(DIV)                 \
    X(SIN) X(COS) X(EXP) X(LOG)                 \
    X(NEG) X(TRANSPOSE) X(REDUCE_SUM)           \
    X(DOT_GENERAL) X(BROADCAST_IN_DIM)          \
    X(CONVERT_ELEMENT_TYPE) X(COND)             \
    X(EQ) X(NE) X(LT) X(LE) X(GT) X(GE)

#define TYPE_ENUM_LIST(X) \
    X(I32) X(I64) X(F32) X(F64) X(BOOL)

enum class primitive_op {
#define X(name) name,
    PRIMITIVE_OP_LIST(X)
#undef X
};

enum class dtype_t {
#define X(name) name,
    TYPE_ENUM_LIST(X)
#undef X
};

std::string_view to_string(primitive_op op);
std::string_view to_string(dtype_t t);

bool is_floating(dtype_t t);
bool is_integral(dtype_t t);
dtype_t resultant_type(dtype_t t1, dtype_t t2);

class type_t {
public:

    explicit type_t(dtype_t dtype);

    type_t(dtype_t dtype, std::vector<size_t> shape);

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

private:
    dtype_t dtype;
    std::vector<size_t> shape;
};

class var_t {
public:
    explicit var_t(size_t id, type_t type);

    [[nodiscard]] size_t get_id() const;

    void set_id(size_t new_id);

    [[nodiscard]] const type_t& get_type() const;
    [[nodiscard]] const std::vector<size_t>& get_shape() const;
    [[nodiscard]] dtype_t get_dtype() const;

private:
    type_t type;
    size_t id;
};

#define check_type(T_real, T_enum)                          \
if constexpr (std::is_same_v<T, T_real>) {                  \
    if(type.get_dtype() != dtype_t::T_enum) {         \
        throw std::logic_error("Error: type mismatch");     \
    }                                                       \
}

using vector_variant = std::variant<
    std::vector<i32>, std::vector<i64>,
    std::vector<f32>,std::vector<f64>,
    std::vector<b8>
>;

using span_variant = std::variant<
    std::span<const i32>, std::span<const i64>,
    std::span<const f32>,std::span<const f64>,
    std::span<const b8>
>;

inline size_t num_elements(std::span<const size_t> shape) {
    size_t total = 1;
    for (size_t dim : shape) total *= dim;
    return total;
}

size_t flatten_index(std::span<const size_t> stride, std::same_as<size_t> auto... indices) {
    size_t idx = 0;
    size_t axis = 0;
    ((idx += indices * stride[axis++]), ...);
    return idx;
}

inline size_t flatten_index(std::span<const size_t> stride, const std::vector<size_t>& indices) {
    size_t idx = 0;
    for (size_t axis = 0; axis < indices.size(); axis++) {
        idx += indices[axis] * stride[axis];
    }
    return idx;
}

#define ACCESS_DISPATCH(...)                    \
switch (type.get_dtype()) {                     \
    using enum dtype_t;                         \
    case I32: return access<i32>(__VA_ARGS__);  \
    case I64: return access<i64>(__VA_ARGS__);  \
    case F32: return access<f32>(__VA_ARGS__);  \
    case F64: return access<f64>(__VA_ARGS__);  \
    case BOOL: return access<b8>(__VA_ARGS__);  \
    default: return 0;                          \
}

class array_t {
public:
    friend class array_span;
    template<typename T>
    array_t(type_t type, std::vector<T> value):
    type(std::move(type)),
    value(std::move(value)) {
        check_type(i32, I32);
        check_type(i64, I64);
        check_type(f32, F32);
        check_type(f64, F64);
        check_type(b8, BOOL);

        if (std::get<std::vector<T>>(this->value).size() != num_elements(this->type.get_shape())) {
            throw std::logic_error("Error: array value does not match the size of its dimensions");
        }

        compute_strides();
        check_single_value();
    }

    template<typename T>
    static array_t build(type_t type, invocable_r<T, const std::vector<size_t>&> auto f) {
        // f takes the std::vector
        const auto& shape_vector = type.get_shape();
        auto index_vector = std::vector<size_t>(type.get_shape().size(), 0);
        auto output_vector = std::vector<T>(num_elements(type.get_shape()));

        for(auto& output: output_vector) {
            output = f(index_vector);
            for (size_t j = index_vector.size(); j--> 0;) {
                if (index_vector[j] < shape_vector[j] - 1) {
                    index_vector[j]++;
                    break;
                }
                index_vector[j] = 0;
            }
        }

        return array_t{std::move(type), output_vector};
    }

    template<typename T>
    static array_t build_fill(type_t type, T value) {
        switch (type.get_dtype()) {
            using enum dtype_t;
            case I32:
            return array_t{std::move(type), std::vector<i32>(num_elements(type.get_shape()), value)};
            case I64:
            return array_t{std::move(type), std::vector<i64>(num_elements(type.get_shape()), value)};
            case F32:
            return array_t{std::move(type), std::vector<f32>(num_elements(type.get_shape()), value)};
            case F64:
            return array_t{std::move(type), std::vector<f64>(num_elements(type.get_shape()), value)};
            case BOOL:
            return array_t{std::move(type), std::vector<b8>(num_elements(type.get_shape()), value)};
            default:
            return array_t{0};
        }
    }

    // ReSharper disable once CppNonExplicitConvertingConstructor
    array_t(f32 value);

    // Quite inefficient, use sparingly
    std::variant<i32, i64, f32, f64, b8> operator[](std::same_as<size_t> auto... indices) {
        ACCESS_DISPATCH(indices...)
    }

    std::variant<i32, i64, f32, f64, b8> operator[](const std::vector<size_t>& indices) {
        ACCESS_DISPATCH(indices)
    }

    template<typename T>
    T& access(std::same_as<size_t> auto... indices) {
        return std::get<std::vector<T>>(value)[flatten_index(stride, indices...)];
    }

    template<typename T>
    T& access(const std::vector<size_t>& indices) {
        return std::get<std::vector<T>>(value)[flatten_index(stride, indices)];
    }

    array_t operator+(const array_t& other) const;
    array_t operator-(const array_t& other) const;
    array_t operator*(const array_t& other) const;
    array_t operator/(const array_t& other) const;

    [[nodiscard]] const type_t& get_type() const;
    [[nodiscard]] const vector_variant& get_value() const;
    [[nodiscard]] vector_variant& get_value();

    bool has_single_value(const f64 val) const {
        return has_single_value_ && val == single_value;
    }

    bool has_single_value() const {
        return has_single_value_;
    }

private:
    void compute_strides();
    void check_single_value();

    type_t type;
    vector_variant value;
    std::vector<size_t> stride;
    f64 single_value = 0;
    bool has_single_value_ = false;
};


struct type_span {
    dtype_t dtype;
    std::span<const size_t> shape;

    [[nodiscard]] dtype_t get_dtype() const { return dtype; }
    [[nodiscard]] std::span<const size_t> get_shape() const { return shape; }
};

// Non-owning view, potentially to a subset of the array
class array_span {
public:

    array_span(const array_t& array, const std::vector<size_t>& indices) {
        type.dtype = array.type.get_dtype();

        size_t i_0 = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            i_0 += array.stride[i] * indices[i];
        }

        const auto& array_shape = array.type.get_shape();
        type.shape = std::span{array_shape.begin() + indices.size(), array_shape.end()};

        const auto& array_stride = array.stride;
        stride = std::span {array_stride.begin() + indices.size(), array_stride.end()};

        // shape_size(i) = stride[i] * shape[i]
        size_t total_size = array.stride[indices.size()] * array.type.get_shape()[indices.size()];
        value = std::visit([=](const auto& vec) -> span_variant {
            return std::span{vec.begin() + i_0, total_size};
        }, array.value);
    }

    std::variant<i32, i64, f32, f64, b8> operator[](std::same_as<size_t> auto... indices) const {
        ACCESS_DISPATCH(indices...)
    }

    std::variant<i32, i64, f32, f64, b8> operator[](const std::vector<size_t>& indices) const {
        ACCESS_DISPATCH(indices)
    }

    template<typename T>
    const T& access(std::same_as<size_t> auto... indices) const {
        return std::get<std::span<const T>>(value)[flatten_index(stride, indices...)];
    }

    template<typename T>
    const T& access(const std::vector<size_t>& indices) const {
        return std::get<std::span<const T>>(value)[flatten_index(stride, indices)];
    }

    [[nodiscard]] const type_span& get_type() const;
    [[nodiscard]] const span_variant& get_value() const;
    [[nodiscard]] span_variant& get_value();

private:
    type_span type;
    std::span<const size_t> stride;
    span_variant value;
};


#undef check_type
#undef ACCESS_DISPATCH

class value {
public:
    explicit value(array_t array);
    explicit value(var_t var);

    template<typename T>
    [[nodiscard]] bool is() const {
        return std::holds_alternative<T>(variant_);
    }

    [[nodiscard]] const type_t& get_type() const;
    [[nodiscard]] const array_t& get_array() const;
    [[nodiscard]] const var_t& get_var() const;

private:
   std::variant<array_t, var_t> variant_;
};

struct implicit_broadcast_result {
    std::vector<size_t> new_shape;
    std::vector<size_t> left_broadcast_dimensions;
    std::vector<size_t> right_broadcast_dimensions;
};

implicit_broadcast_result get_implicit_broadcast_result(const std::vector<size_t>& left_shape,
    const std::vector<size_t>& right_shape);


struct expression;

struct transpose_params {
    std::vector<size_t> permutation;
};

struct dot_general_params {
    std::vector<size_t> left_contract;
    std::vector<size_t> right_contract;
    std::vector<size_t> left_batch;
    std::vector<size_t> right_batch;
};

struct reduce_sum_params {
    std::vector<size_t> axes;
};

struct broadcast_in_dim_params {
    std::vector<size_t> shape;
    std::vector<size_t> broadcast_dimensions;
};

struct convert_element_type_params {
    dtype_t new_dtype;
};

struct cond_params {
    std::vector<expression> branches;
};

using params_variant = std::variant<
    std::monostate,
    transpose_params,
    dot_general_params,
    reduce_sum_params,
    broadcast_in_dim_params,
    convert_element_type_params,
    cond_params
>;

class equation {
public:
    equation(std::vector<value> input, std::vector<var_t> output, primitive_op op);
    equation(std::vector<value> input, std::vector<var_t> output, primitive_op op, params_variant params);

    [[nodiscard]] const std::vector<value>& get_input() const;
    [[nodiscard]] const std::vector<var_t>& get_output() const;
    [[nodiscard]] const value& get_input(size_t i) const;
    [[nodiscard]] const var_t& get_output(size_t i) const;
    [[nodiscard]] primitive_op get_op() const;
    [[nodiscard]] const params_variant& get_params() const;


private:
    std::vector<value> input;
    std::vector<var_t> output;
    primitive_op op;
    params_variant params;
};

struct expression {
    std::vector<var_t> invars;
    std::vector<value> outvals;
    std::vector<equation> equations;
    size_t var_id = 0;

    void add_input(var_t var);
    void add_output(value val);
    void add_equation(equation eq);

    void eliminate_dead_code();

    size_t new_var_id();
};


}

#endif //JAXPR_TYPES_H
