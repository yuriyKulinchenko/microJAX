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

    auto output =
        dot_general(tracer_x, tracer_y, {{1}, {2}}, {{0}, {0}});

    builder.register_output(output);
    expression jaxpr = builder.get_jaxpr();

    std::cout << "Original expression:\n";
    std::cout << jaxpr;

    return 0;


    // jaxpr_tracer tracer_x = builder.register_tracer(type_enum::F32, 5, 1, 3);
    //
    // auto broadcasted = broadcast_in_dim(tracer_x, {5, 10, 9, 15, 3}, {0, 2, 4});
    // auto output = reduce_sum(broadcasted, {0, 1, 2, 3, 4});
    //
    // builder.register_output(output);
    // expression jaxpr = builder.get_jaxpr();
    //
    // std::cout << "Original expression:\n";
    // std::cout << jaxpr;
    //
    // expression grad_jaxpr = grad(jaxpr);
    // std::cout << "Grad expression:\n";
    // std::cout << grad_jaxpr;
    //
    // grad_jaxpr.eliminate_dead_code();
    // std::cout << "Grad expression (DCE):\n";
    // std::cout << grad_jaxpr;
}
