#ifndef MICROJAX_JAX_VM_H
#define MICROJAX_JAX_VM_H

#include "jax_types.h"

/*

The JAX VM is a virtual machine that can execute uncompiled JAX expressions.
The VM is inherently slower than JIT compilation, however it will act as
a ground truth that the JIT results can be compared against in testing.

*/

using namespace jax;


class jax_vm {
public:
    jax_vm(const expression& jaxpr);

    std::vector<array_t> run(const std::vector<array_t>& input);

    void reset();

private:

    const array_t& fetch_value(const var_t& var) const;

    void emplace_variable(const var_t& var, array_t array);

    const array_t& get_input(const equation& eq, size_t i);

    static array_t literal_to_array(const literal_t& literal);

    void populate_literal_buffer(const equation& eq);

    static void check_fixed_arity(const equation& eq, size_t input_arity, size_t output_arity=1);

    template<array_t (array_t::*op)() const>
    void execute_unary_op(const equation& eq) {
        check_fixed_arity(eq, 1);
        if (eq.get_input(0).get_dtype() == dtype_t::BOOL) {
            throw std::logic_error("Error: cannot execute unary op on boolean argument");
        }
        emplace_variable(eq.get_output(0), (get_input(eq, 0).*op)());
    }

    template<array_t (array_t::*op)(const array_t&) const>
    void execute_binary_op(const equation& eq) {
        if (eq.get_input(0).get_dtype() != eq.get_input(1).get_dtype()) {
            throw std::logic_error("Error: can only execute binary op on arguments of the same type");
        }

        check_fixed_arity(eq, 2);
        emplace_variable(eq.get_output(0), (get_input(eq, 0).*op)(get_input(eq, 1)));
    }

    template<array_t (array_t::*op)(const array_t&) const>
    void execute_binary_comparison_op(const equation& eq) {
        check_fixed_arity(eq, 2);
        emplace_variable(eq.get_output(0), (get_input(eq, 0).*op)(get_input(eq, 1)));
    }

    const expression& jaxpr;
    std::vector<std::unique_ptr<array_t>> values;
    std::vector<std::optional<array_t>> literal_buffer;
};

template<bool single_output=true>
auto invoke_vm(const expression& jaxpr, const std::vector<array_t>& input) {
    jax_vm vm{jaxpr};
    if constexpr (single_output) {
        return vm.run(input)[0];
    } else {
        return vm.run(input);
    }
}

template<bool single_output=true>
auto invoke_vm(const expression& jaxpr, const array_t& input) {
    jax_vm vm{jaxpr};
    if constexpr (single_output) {
        return vm.run(std::vector{input})[0];
    } else {
        return vm.run(std::vector{input});
    }
}

#endif //MICROJAX_JAX_VM_H
