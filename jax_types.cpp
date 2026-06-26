#include "jax_types.h"

namespace jax {

std::string_view to_string(primitive_op op) {
    switch (op) {
#define X(name) case primitive_op::name: return #name;
        PRIMITIVE_OP_LIST(X)
#undef X
    }
    return "";
}

std::string_view to_string(type_enum t) {
    switch (t) {
#define X(name) case type_enum::name: return #name;
        TYPE_ENUM_LIST(X)
#undef X
    }
    return "";
}

type_t::type_t(type_enum base_type):
base_type(base_type) {}

type_t::type_t(type_enum base_type, std::vector<size_t> dimension):
base_type(base_type),
dimension(std::move(dimension)) {}

bool type_t::operator==(const type_t& other) const {
    return base_type == other.base_type && dimension == other.dimension;
}

bool type_t::is_i32() {
    return base_type == type_enum::I32;
}

bool type_t::is_i64() {
    return base_type == type_enum::I64;
}

bool type_t::is_f32() {
    return base_type == type_enum::F32;
}

bool type_t::is_f64() {
    return base_type == type_enum::F64;
}

type_enum type_t::get_base_type() const {
    return base_type;
}

const std::vector<size_t>& type_t::get_dimension() const {
    return dimension;
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

array_t::array_t(f32 value)
: array_t(type_t{type_enum::F32}, std::vector<f32>{value}) {
    check_single_value();
}

void array_t::compute_strides() {
    const std::vector<size_t>& dimension = type.get_dimension();
    stride.resize(dimension.size());
    if (stride.empty()) {
        return;
    }

    // Strides are calculated backwards:

    stride[stride.size() - 1] = 1;
    for (size_t i = stride.size() - 1; i-->0;) {
        stride[i] = stride[i + 1] * dimension[i + 1];
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

const type_t& array_t::get_type() const {
    return type;
}

const vector_variant &array_t::get_value() const {
    return value;
}

vector_variant &array_t::get_value() {
    return value;
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

value::value(array_t array): variant_(std::move(array)) {}
value::value(var_t var): variant_(std::move(var)) {}

const type_t& value::get_type() const {
    return std::visit([](auto& v) -> const type_t& {return v.get_type();}, variant_);
}

const array_t& value::get_array() const {
    return std::get<array_t>(variant_);
}

const var_t& value::get_var() const {
    return std::get<var_t>(variant_);
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

equation::equation(std::vector<value> input, std::vector<var_t> output, primitive_op op):
input(std::move(input)),
output(std::move(output)),
op(op) {}

void expression::add_input(var_t var) {
    invars.push_back(std::move(var));
}

void expression::add_output(value val) {
    outvals.push_back(std::move(val));
}

void expression::add_equation(equation eq) {
    equations.push_back(std::move(eq));
}

size_t expression::new_var_id() {
    return var_id++;
}


}
