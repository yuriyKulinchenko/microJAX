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

    auto x = builder.register_tracer(type_enum::F32, 15, 10, 5);
    auto y = builder.register_tracer(type_enum::F32, 5, 8, 10);

    auto x_transpose = transpose(x, {2, 1, 0});

    auto product =
        dot_general(x_transpose, y, {{1}, {2}}, {{0}, {0}});

    auto output = reduce_sum(product, {0, 1, 2});

    builder.register_output(output);
    expression jaxpr = builder.get_jaxpr();

    std::cout << "Original expression:\n" << jaxpr;


    expression grad_jaxpr = grad(jaxpr);
    std::cout << "Grad expression:\n" << grad_jaxpr;

    grad_jaxpr.eliminate_dead_code();
    std::cout << "Grad expression (DCE):\n" << grad_jaxpr;

    return 0;
}
