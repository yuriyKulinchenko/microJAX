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

type_t::type_t(type_enum base_type, std::vector<u32> dimension):
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

const std::vector<u32>& type_t::get_dimension() const {
    return dimension;
}

var_t::var_t(u32 id, type_t type): id(id), type(std::move(type)) {}

u32 var_t::get_id() const {
    return id;
}

void var_t::set_id(u32 new_id) {
    id = new_id;
}

const type_t& var_t::get_type() const {
    return type;
}

array_t::array_t(f32 value)
: array_t(type_t{type_enum::F32}, std::vector<f32>{value}) {}

void array_t::compute_strides() {
    const std::vector<u32>& dimension = type.get_dimension();
    strides.resize(dimension.size());
    if (strides.empty()) {
        return;
    }

    // Strides are calculated backwards:

    strides[strides.size() - 1] = 1;
    for (size_t i = strides.size() - 1; i-->0;) {
        strides[i] = strides[i + 1] * dimension[i + 1];
    }
}


const type_t& array_t::get_type() const {
    return type;
}

const vector_variant &array_t::get_value() const {
    return value;
}

vector_variant &array_t::get_value() {
    return value;
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

u32 expression::new_var_id() {
    return var_id;
}


}
