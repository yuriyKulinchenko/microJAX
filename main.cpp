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

    auto tracer_x = builder.register_tracer(type_enum::F32, 5, 10, 15);
    auto tracer_y = builder.register_tracer(type_enum::F32, 5, 8, 10);

    auto product =
        dot_general(tracer_x, tracer_y, {{1}, {2}}, {{0}, {0}});

    // product: (5, 15, 8) -> sum all axes -> scalar
    auto output = reduce_sum(product, {0, 1, 2});

    builder.register_output(output);
    expression jaxpr = builder.get_jaxpr();

    std::cout << "Original expression:\n";
    std::cout << jaxpr;

    std::cout << "Grad expression:\n";
    std::cout << grad(jaxpr);

    return 0;
}
