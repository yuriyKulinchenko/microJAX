#include <iostream>

#include "jax_types.h"
#include "jax_functions.h"
#include "tracer.h"
#include "jax_logger.h"
#include "grad.h"


template<typename T>
T test_function(T x, T y) {
    return jax::sin(x) + jax::cos(x) * jax::cos(y) + 1;
}

int main() {
    using namespace jax;
    // Construct tracer:
    jaxpr_builder builder {};
    jaxpr_tracer tracer_x = builder.register_tracer(type_enum::F32);
    // jaxpr_tracer tracer_y = builder.register_tracer(type_enum::F32);

    auto output = jax::cos(jax::sin(tracer_x));
    builder.register_output(output);

    std::cout << "Original expression:\n";
    std::cout << builder.jaxpr;

    std::cout << "Grad expression:\n";
    std::cout << grad(grad(grad(builder.jaxpr)));

    return 0;
}
