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

    std::vector<value> &equation::get_input() {
        return input;
    }

    const std::vector<var_t> &equation::get_output() const {
        return output;
    }

    std::vector<var_t> &equation::get_output() {
        return output;
    }

    const value &equation::get_input(size_t i) const {
        return input[i];
    }

    value &equation::get_input(size_t i) {
        return input[i];
    }

    const var_t &equation::get_output(size_t i) const {
        return output[i];
    }

    var_t &equation::get_output(size_t i) {
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

    equation& expression::add_equation(equation eq) {
        equations.push_back(std::move(eq));
        return equations[equations.size() - 1];
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

}
