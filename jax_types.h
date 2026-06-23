#ifndef JAXPR_TYPES_H
#define JAXPR_TYPES_H
#include <any>
#include <variant>
#include <vector>

// TODO: re-use formatted exceptions from prolog interpreter

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float; // Temporary: these may not be true
using f64 = double;

namespace jax {

enum class primitive_op {
    ADD, SUB, MUL, SIN, COS, EXP, LOG
};

enum class type_enum {
    I32, I64, F32, F64
};

class type {
public:

    explicit type(type_enum base_type):
    base_type(base_type) {}

    type(type_enum base_type, std::vector<u32> dimension):
    base_type(base_type),
    dimension(std::move(dimension)) {}

    bool is_i32() {
        return base_type == type_enum::I32;
    }

    bool is_i64() {
        return base_type == type_enum::I64;
    }

    bool is_f32() {
        return base_type == type_enum::F32;
    }

    bool is_f64() {
        return base_type == type_enum::F64;
    }

    type_enum get_base_type() {
        return base_type;
    }

    const std::vector<u32>& get_dimension() {
        return dimension;
    }

private:
    type_enum base_type;
    std::vector<u32> dimension;
};

class var {
public:
    explicit var(type type_): id(global_id++), type_(std::move(type_)) {}
private:
    u32 id;
    type type_;
    // global_id is bumped on every new variable added
    inline static u32 global_id = 0;
};

#define check_type(T_real, T_enum)                          \
if constexpr (std::is_same_v<T, T_real>) {                  \
    if(type_.get_base_type() != type_enum::T_enum) {        \
        throw std::logic_error("Error: type mismatch");     \
    }                                                       \
}

class array {
public:
    template<typename T>
    explicit array(type type_, std::vector<T> value):
    type_(std::move(type_)),
    value(std::move(value)),
    strides(std::vector<u32>(type_.get_dimension().size())) {
        check_type(i32, I32);
        check_type(i64, I64);
        check_type(f32, F32);
        check_type(f64, F64);

        // Strides are calculated backwards:

        strides[strides.size() - 1] = 1;
        for (size_t i = strides.size() - 1; i-->0;) {
            strides[i] = strides[i + 1] * type_.get_dimension()[i + 1];
        }
    }

    // Quite inefficient, use sparingly
    std::variant<i32, i64, f32, f64> operator[](std::same_as<size_t> auto... indices) {
        switch (type_.get_base_type()) {
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

private:
    type type_;
    std::variant<
        std::vector<i32>,
        std::vector<i64>,
        std::vector<f32>,
        std::vector<f64>
    > value;
    std::vector<u32> strides;
};

#undef check_type

using value = std::variant<array, var>;

// An equation will look something like:
// a:f32[8] = sin b
// c:f32[] = add a b
class equation {
public:
    equation(std::vector<value> input, std::vector<var> output, primitive_op op):
    input(std::move(input)),
    output(std::move(output)),
    op(op) {}

private:
    std::vector<value> input;
    std::vector<var> output;
    primitive_op op;
};


}

#endif //JAXPR_TYPES_H
