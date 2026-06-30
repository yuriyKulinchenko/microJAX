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

    jaxpr_tracer tracer_x = builder.register_tracer(type_enum::F32, 10, 15, 20);
    auto broadcasted = broadcast_in_dim(tracer_x, {10, 5, 15, 7, 20}, {0, 2, 4});
    auto output = reduce_sum(broadcasted, {0, 1, 2, 3, 4});

    builder.register_output(output);
    expression jaxpr = builder.get_jaxpr();

    std::cout << "Original expression:\n";
    std::cout << jaxpr;

    std::cout << "Grad expression:\n";
    std::cout << grad(jaxpr);

    return 0;
}
