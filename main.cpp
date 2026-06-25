#include <iostream>

#include "jax_types.h"
#include "jax_functions.h"
#include "tracer.h"
#include "jax_logger.h"

template<typename T>
T test_function(T x) {
    return jax::sin(x) + jax::cos(x) + 1;
}

int main() {
    // Construct tracer:
    jaxpr_builder builder {};
    jaxpr_tracer tracer = builder.register_tracer(jax::type_enum::F32);
    auto output = test_function<jaxpr_tracer>(tracer);
    builder.register_output(output);

    std::cout << builder.jaxpr;
    return 0;
}
