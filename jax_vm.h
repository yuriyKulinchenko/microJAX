#ifndef MICROJAX_JAX_VM_H
#define MICROJAX_JAX_VM_H

#include "jax_types.h"

/*

The JAX VM is a virtual machine that can execute uncompiled JAX expressions.
The VM is inherently slower than JIT compilation, however it will act as
a ground truth that the JIT results can be compared against in testing.

*/


class jax_vm {
public:
    jax_vm(const jax::expression& jaxpr): jaxpr(jaxpr) {}

    std::vector<jax::array_t> run(const std::vector<jax::array_t>& input) {
        // This is the core loop:
        return {};
    }



private:
    const jax::expression& jaxpr;
};


#endif //MICROJAX_JAX_VM_H
