#ifndef JAXPR_TYPES_H
#define JAXPR_TYPES_H
#include <any>
#include <string_view>
#include <variant>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float; // Temporary: these may not be true
using f64 = double;

namespace jax {

#define PRIMITIVE_OP_LIST(X) \
    X(ADD) X(SUB) X(MUL) X(SIN) X(COS) X(EXP) X(LOG)

#define TYPE_ENUM_LIST(X) \
    X(I32) X(I64) X(F32) X(F64)

enum class primitive_op {
#define X(name) name,
    PRIMITIVE_OP_LIST(X)
#undef X
};

enum class type_enum {
#define X(name) name,
    TYPE_ENUM_LIST(X)
#undef X
};

std::string_view to_string(primitive_op op);
std::string_view to_string(type_enum t);

class type_t {
public:

    explicit type_t(type_enum base_type);

    type_t(type_enum base_type, std::vector<u32> dimension);

    template<std::convertible_to<u32>... Dims>
    explicit type_t(type_enum base_type, Dims... dimension)
    : base_type(base_type), dimension{static_cast<u32>(dimension)...} {}


    bool operator==(const type_t& other) const;

    bool is_i32();
    bool is_i64();
    bool is_f32();
    bool is_f64();

    [[nodiscard]] type_enum get_base_type() const;
    [[nodiscard]] const std::vector<u32>& get_dimension() const;

private:
    type_enum base_type;
    std::vector<u32> dimension;
};

class var_t {
public:
    explicit var_t(u32 id, type_t type);

    [[nodiscard]] u32 get_id() const;

    void set_id(u32 new_id);

    [[nodiscard]] const type_t& get_type() const;

private:
    type_t type;
    u32 id;
};

#define check_type(T_real, T_enum)                          \
if constexpr (std::is_same_v<T, T_real>) {                  \
    if(type.get_base_type() != type_enum::T_enum) {         \
        throw std::logic_error("Error: type mismatch");     \
    }                                                       \
}

using vector_variant = std::variant<
    std::vector<i32>, std::vector<i64>,
    std::vector<f32>,std::vector<f64>
>;

class array_t {
public:
    template<typename T>
    array_t(type_t type, std::vector<T> value):
    type(std::move(type)),
    value(std::move(value)) {
        check_type(i32, I32);
        check_type(i64, I64);
        check_type(f32, F32);
        check_type(f64, F64);

        compute_strides();
    }

    // ReSharper disable once CppNonExplicitConvertingConstructor
    array_t(f32 value);

    // Quite inefficient, use sparingly
    std::variant<i32, i64, f32, f64> operator[](std::same_as<size_t> auto... indices) {
        switch (type.get_base_type()) {
            using enum type_enum;
            case I32: return access<i32>(indices...);
            case I64: return access<i64>(indices...);
            case F32: return access<f32>(indices...);
            case F64: return access<f64>(indices...);
            default: return 0;
        }
    }

    template<typename T>
    T& access(std::same_as<size_t> auto... indices) {
        auto& raw = std::get<std::vector<T>>(value);
        size_t idx = 0;
        size_t axis = 0;
        ((idx += indices * strides[axis++]), ...);
        return raw[idx];
    }

    [[nodiscard]] const type_t& get_type() const;
    [[nodiscard]] const vector_variant& get_value() const;
    [[nodiscard]] vector_variant& get_value();

private:
    void compute_strides();

    type_t type;
    vector_variant value;
    std::vector<u32> strides;
};

#undef check_type

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


// An equation will look something like:
// a:f32[8] = sin b
// c:f32[] = add a b
class equation {
public:
    equation(std::vector<value> input, std::vector<var_t> output, primitive_op op);

    [[nodiscard]] const std::vector<value>& get_input() const;
    [[nodiscard]] const std::vector<var_t>& get_output() const;
    [[nodiscard]] const value& get_input(size_t i) const;
    [[nodiscard]] const var_t& get_output(size_t i) const;
    [[nodiscard]] primitive_op get_op() const;


private:
    std::vector<value> input;
    std::vector<var_t> output;
    primitive_op op;
};

struct expression {
    std::vector<var_t> invars;
    std::vector<value> outvals;
    std::vector<equation> equations;
    u32 var_id = 0;

    void add_input(var_t var);
    void add_output(value val);
    void add_equation(equation eq);

    u32 new_var_id();
};


}

#endif //JAXPR_TYPES_H
