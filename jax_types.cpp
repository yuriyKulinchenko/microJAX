#include "jax_types.h"
#include "DCE_class.h"

namespace jax {

std::string_view to_string(primitive_op op) {
    switch (op) {
#define X(name) case primitive_op::name: return #name;
        PRIMITIVE_OP_LIST(X)
#undef X
    }
    return "";
}

std::string_view to_string(dtype_t t) {
    switch (t) {
#define X(name) case dtype_t::name: return #name;
        TYPE_ENUM_LIST(X)
#undef X
    }
    return "";
}

bool is_floating(dtype_t t) {
    using enum dtype_t;
    return t == F32 || t == F64;
}

bool is_integral(dtype_t t) {
    using enum dtype_t;
    return t == I32 || t == I64 || t == BOOL;
}

dtype_t widest_float(dtype_t t1, dtype_t t2) {
    using enum dtype_t;
    return (t1 == F64 || t2 == F64) ? F64 : F32;
}

dtype_t widest_int(dtype_t t1, dtype_t t2) {
    using enum dtype_t;
    return (t1 == I64 || t2 == I64) ? I64 : I32;
}

dtype_t resultant_type(dtype_t t1, dtype_t t2) {
    using enum dtype_t;
    if (t1 == BOOL) return t2;
    if (t2 == BOOL) return t1;

    if (is_floating(t1)) {
        if (is_floating(t2)) {
            return widest_float(t1, t2);
        }
        return t1;
    }

    if (is_floating(t2)) {
        return t2;
    }

    return widest_int(t1, t2);
}

type_t::type_t(dtype_t dtype):
dtype(dtype) {}

type_t::type_t(dtype_t dtype, std::vector<size_t> shape):
dtype(dtype),
shape(std::move(shape)) {}

bool type_t::operator==(const type_t& other) const {
    return dtype == other.dtype && shape == other.shape;
}

bool type_t::is_i32() const {
    return dtype == dtype_t::I32;
}

bool type_t::is_i64() const {
    return dtype == dtype_t::I64;
}

bool type_t::is_f32() const {
    return dtype == dtype_t::F32;
}

bool type_t::is_f64() const {
    return dtype == dtype_t::F64;
}

bool type_t::is_bool() const {
    return dtype == dtype_t::BOOL;
}

dtype_t type_t::get_dtype() const {
    return dtype;
}

const std::vector<size_t>& type_t::get_shape() const {
    return shape;
}

var_t::var_t(size_t id, type_t type): id(id), type(std::move(type)) {}

size_t var_t::get_id() const {
    return id;
}

void var_t::set_id(size_t new_id) {
    id = new_id;
}

const type_t& var_t::get_type() const {
    return type;
}

const std::vector<size_t> &var_t::get_shape() const {
    return type.get_shape();
}

dtype_t var_t::get_dtype() const {
    return type.get_dtype();
}

array_t::array_t(f32 value)
: array_t(type_t{dtype_t::F32}, std::vector<f32>{value}) {
    check_single_value();
}

void array_t::compute_strides() {
    const std::vector<size_t>& shape = type.get_shape();
    stride.resize(shape.size());
    if (stride.empty()) {
        return;
    }

    // Strides are calculated backwards:

    stride[stride.size() - 1] = 1;
    for (size_t i = stride.size() - 1; i-->0;) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

#define DIMENSIONALITY_ERROR()                                                                      \
if(type != other.get_type()) {                                                                      \
    throw std::logic_error("ERROR: dimensionality mismatch when attempting elementwise operation"); \
}

#define BINARY_OP_OTHER(op, op_assign)                                      \
array_t array_t::operator op (const array_t& other) const {                 \
    DIMENSIONALITY_ERROR();                                                 \
    return std::visit([&](auto& vec) -> array_t {                           \
        auto return_vec = vec;                                              \
        auto other_vec = std::get<decltype(return_vec)>(other.value);       \
        for (size_t i = 0; i < return_vec.size(); i++) {                    \
            return_vec[i] op_assign other_vec[i];                           \
        }                                                                   \
        return array_t{type, return_vec};                                   \
    }, value);                                                              \
}

BINARY_OP_OTHER(+, +=);
BINARY_OP_OTHER(-, -=);
BINARY_OP_OTHER(*, *=);
BINARY_OP_OTHER(/, /=);

const type_t& array_t::get_type() const {
    return type;
}

const vector_variant &array_t::get_value() const {
    return value;
}

vector_variant &array_t::get_value() {
    return value;
}

std::optional<literal_t> array_t::get_literal() const {
    if (type.get_shape().size() != 0) return std::nullopt;
    switch (const dtype_t dtype = type.get_dtype()) {
        using enum dtype_t;
        case I32: return literal_t{dtype, std::get<std::vector<i32>>(value)[0]};
        case I64: return literal_t{dtype, std::get<std::vector<i64>>(value)[0]};
        case F32: return literal_t{dtype, std::get<std::vector<f32>>(value)[0]};
        case F64: return literal_t{dtype, std::get<std::vector<f64>>(value)[0]};
        default: return literal_t{dtype, std::get<std::vector<b8>>(value)[0]};
    }
}

const type_span& array_span::get_type() const {
    return type;
}

const span_variant& array_span::get_value() const {
    return value;
}

span_variant& array_span::get_value() {
    return value;
}

void array_t::check_single_value() {
    std::visit([this](auto& vec) {
        if (vec.size() == 0) {
            this->has_single_value_ = true;
            return;
        }

        auto val = vec[0];
        for (size_t i = 1; i < vec.size(); i++) {
            if (vec[i] != val) {
                this->has_single_value_ = false;
                return;
            }
        }

        this->has_single_value_ = true;
        this->single_value = static_cast<f64>(val);
    }, value);
}

dtype_t literal_t::get_dtype() const {
    return dtype;
}

const value_variant &literal_t::get_value() const {
    return value;
}

const std::vector<size_t> &literal_t::get_shape() {
    return shape;
}

value::value(literal_t literal): variant_(std::move(literal)) {}
value::value(var_t var): variant_(std::move(var)) {}

const literal_t& value::get_literal() const {
    return std::get<literal_t>(variant_);
}

const var_t& value::get_var() const {
    return std::get<var_t>(variant_);
}

dtype_t value::get_dtype() const {
    return std::visit([](auto& x){return x.get_dtype();}, variant_);
}

const std::vector<size_t> &value::get_shape() const {
    if (is<var_t>()) {
        return std::get<var_t>(variant_).get_shape();
    }
    return literal_t::get_shape();
}

type_t value::get_type() const {
    if (is<var_t>()) {
        return std::get<var_t>(variant_).get_type();
    }
    return type_t{std::get<literal_t>(variant_).get_dtype(), literal_t::get_shape()};
}

const std::vector<value> &equation::get_input() const {
    return input;
}

const std::vector<var_t> &equation::get_output() const {
    return output;
}

const value &equation::get_input(size_t i) const {
    return input[i];
}

const var_t &equation::get_output(size_t i) const {
    return output[i];
}

primitive_op equation::get_op() const {
    return op;
}

const params_variant& equation::get_params() const {
    return params;
}

params_variant& equation::get_params() {
    return params;
}




equation::equation(std::vector<value> input, std::vector<var_t> output, primitive_op op):
input(std::move(input)),
output(std::move(output)),
op(op)
{}

equation::equation(std::vector<value> input, std::vector<var_t> output, primitive_op op, params_variant params):
input(std::move(input)),
output(std::move(output)),
op(op),
params(params)
{}

void expression::add_invar(var_t var) {
    invars.push_back(std::move(var));
}

void expression::add_constvar(var_t var) {
    constvars.push_back(std::move(var));
}

void expression::add_output(value val) {
    outvals.push_back(std::move(val));
}

void expression::add_equation(equation eq) {
    equations.push_back(std::move(eq));
}

var_t expression::fresh_var(type_t type) {
    return var_t{new_var_id(), std::move(type)};
}

size_t expression::new_var_id() {
    return var_id++;
}

void expression::eliminate_dead_code() {
    DCE_class instance {*this};
    instance.apply_dead_code_elimination();
}

std::vector<size_t> get_implicit_broadcast_shape(const std::vector<size_t>& left_shape,
    const std::vector<size_t>& right_shape) {

    // Normalize so that left_shape.size() < right_shape.size():
    if (right_shape.size() < left_shape.size())
        return get_implicit_broadcast_shape(right_shape, left_shape);

    size_t new_size = right_shape.size();
    std::vector<size_t> new_shape(new_size);

    // For all places where the 1s prefix would have existed, populate with right_shape:

    size_t delta = right_shape.size() - left_shape.size();
    for (size_t i = 0; i < delta; i++) {
        new_shape[i] = right_shape[i];
    }

    // For all places where there is a shared prefix,
    // Ensure that one of them is 1, and take the max:

    for (size_t i = 0; i < left_shape.size(); i++) {
        if (left_shape[i] == 1) {
            new_shape[delta + i] = right_shape[i];
        } else if (right_shape[delta + i] == 1) {
            new_shape[delta + i] = left_shape[delta + i];
        } else if (left_shape[i] == right_shape[delta + i]) {
            new_shape[delta + i] = left_shape[i];
        } else {
            std::cerr << left_shape << '\n';
            std::cerr << right_shape << '\n';
            throw std::logic_error("Error: shape mismatch in implicit broadcast");
        }
    }

    return new_shape;
}

implicit_broadcast_result get_implicit_broadcast_result(const std::vector<size_t>& left_shape,
    const std::vector<size_t>& right_shape) {
    std::vector<size_t> new_shape = get_implicit_broadcast_shape(left_shape, right_shape);

    std::vector<size_t> broadcast_dimensions(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); i++) {
        broadcast_dimensions[i] = i;
    }

    // From the new_shape, infer the broadcast dimensions:
    if (left_shape.size() < right_shape.size()) {
        size_t delta = right_shape.size() - left_shape.size();
        std::vector<size_t> left_broadcast_dimensions(left_shape.size());
        for (size_t i = 0; i < left_shape.size(); i++) {
            left_broadcast_dimensions[i] = i + delta;
        }
        return {
            std::move(new_shape),
            std::move(left_broadcast_dimensions),
            std::move(broadcast_dimensions)
        };
    }

    size_t delta = left_shape.size() - right_shape.size();
    std::vector<size_t> right_broadcast_dimensions(right_shape.size());
    for (size_t i = 0; i < right_shape.size(); i++) {
        right_broadcast_dimensions[i] = i + delta;
    }
    return {
        std::move(new_shape),
        std::move(broadcast_dimensions),
        std::move(right_broadcast_dimensions)
    };
}



}
