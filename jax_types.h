#ifndef JAXPR_TYPES_H
#define JAXPR_TYPES_H
#include <any>
#include <functional>
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
    X(CONVERT_ELEMENT_TYPE) X(COND) X(SCAN)     \
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

    void set_dtype(dtype_t new_dtype);

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

class array_t {
public:
    friend class array_span;
    array_t(type_t type, std::vector<double> value);

    static array_t build(dtype_t dtype, std::vector<size_t> shape,
        const std::function<double(const std::vector<size_t>&)>& f);

    static array_t build_fill(type_t type, double value);

    // ReSharper disable once CppNonExplicitConvertingConstructor
    array_t(f32 value);

    // Quite inefficient, use sparingly
    double operator[](std::same_as<size_t> auto... indices) {
        return access(indices...);
    }

    double operator[](const std::vector<size_t>& indices);

    double& access(std::same_as<size_t> auto... indices) {
        return value[flatten_index(stride, indices...)];
    }

    double& access(const std::vector<size_t>& indices);
    const double& access(const std::vector<size_t>& indices) const;

    template<typename F>
    array_t binary_op(const array_t& other, F f) const;

    bool operator==(const array_t& other) const;

    array_t operator+(const array_t& other) const;
    array_t operator-(const array_t& other) const;
    array_t operator*(const array_t& other) const;
    array_t operator/(const array_t& other) const;

    array_t elementwise_equal(const array_t& other) const;
    array_t elementwise_not_equal(const array_t& other) const;
    array_t operator<(const array_t& other) const;
    array_t operator<=(const array_t& other) const;
    array_t operator>(const array_t& other) const;
    array_t operator>=(const array_t& other) const;


    [[nodiscard]] array_t sin() const;
    [[nodiscard]] array_t cos() const;
    [[nodiscard]] array_t exp() const;
    [[nodiscard]] array_t log() const;
    [[nodiscard]] array_t negate() const;

    [[nodiscard]] array_t transpose(const std::vector<size_t>& permutation) const;
    [[nodiscard]] array_t reduce_sum(const std::vector<size_t>& axes) const;
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
    [[nodiscard]] const std::vector<double>& get_value() const;
    [[nodiscard]] std::vector<double>& get_value();
    [[nodiscard]] std::optional<literal_t> get_literal() const;

    void set_type(type_t new_type);

    [[nodiscard]] bool has_single_value(f64 val) const;
    [[nodiscard]] bool has_single_value() const;

private:
    void compute_strides();
    void check_single_value();

    type_t type;
    std::vector<double> value;
    std::vector<size_t> stride;
    f64 single_value = 0;
    bool has_single_value_ = false;
};

struct type_span {
    dtype_t dtype;
    std::span<const size_t> shape;

    [[nodiscard]] dtype_t get_dtype() const;
    [[nodiscard]] std::span<const size_t> get_shape() const;
};

// Non-owning view, potentially to a subset of the array
class array_span {
public:

    array_span(const array_t& array, const std::vector<size_t>& indices);

    double operator[](std::same_as<size_t> auto... indices) const {
        return access(indices...);
    }

    double operator[](const std::vector<size_t>& indices) const;

    const double& access(std::same_as<size_t> auto... indices) const {
        return value[flatten_index(stride, indices...)];
    }

    const double& access(const std::vector<size_t>& indices) const;

    [[nodiscard]] const type_span& get_type() const;
    [[nodiscard]] std::span<const double> get_value() const;

private:
    type_span type;
    std::span<const size_t> stride;
    std::span<const double> value;
};

class value {
public:
    explicit value(literal_t literal);
    explicit value(var_t var);

    template<typename T>
    [[nodiscard]] bool is() const {
        return std::holds_alternative<T>(variant_);
    }

    // Types are NOT returned by const&, as in the event that
    // the value is a literal, it does not have an underlying
    // type_t. It only has a dtype_t, so the whole type_t has
    // to be created from scratch.

    [[nodiscard]] const literal_t& get_literal() const;
    [[nodiscard]] const var_t& get_var() const;
    [[nodiscard]] dtype_t get_dtype() const;
    [[nodiscard]] const std::vector<size_t>& get_shape() const;
    [[nodiscard]] type_t get_type() const;

private:
   std::variant<literal_t, var_t> variant_;
};

struct implicit_broadcast_result {
    std::vector<size_t> new_shape;
    std::vector<size_t> left_broadcast_dimensions;
    std::vector<size_t> right_broadcast_dimensions;
};

implicit_broadcast_result get_implicit_broadcast_result(const std::vector<size_t>& left_shape,
    const std::vector<size_t>& right_shape);

class equation;

struct expression {
    std::vector<var_t> constvars;
    std::vector<var_t> invars;
    std::vector<value> outvals;
    std::vector<equation> equations;
    std::vector<array_t> consts;
    size_t var_id = 0;

    void add_constvar(var_t var);
    void add_invar(var_t var);
    void add_output(value val);
    equation& add_equation(equation eq);
    var_t fresh_var(type_t type);

    void eliminate_dead_code();

    size_t new_var_id();
};

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

struct scan_params {
    expression jaxpr;
    size_t length;
    size_t num_carry;
    bool reverse;
};

using params_variant = std::variant<
    std::monostate,
    transpose_params,
    dot_general_params,
    reduce_sum_params,
    broadcast_in_dim_params,
    convert_element_type_params,
    cond_params,
    scan_params
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

    [[nodiscard]] std::vector<value>& get_input();
    [[nodiscard]] std::vector<var_t>& get_output();
    [[nodiscard]] value& get_input(size_t i);
    [[nodiscard]] var_t& get_output(size_t i);
    [[nodiscard]] params_variant& get_params();


private:
    std::vector<value> input;
    std::vector<var_t> output;
    primitive_op op;
    params_variant params;
};
}

#endif //JAXPR_TYPES_H
