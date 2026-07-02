#include <iostream>

#include "jax_types.h"
#include "jax_functions.h"
#include "tracer.h"
#include "jax_logger.h"
#include "grad.h"


template<typename T>
T softmax(T vec) {
    auto exp_vec = jax::exp(vec);
    auto sum = jax::reduce_sum(exp_vec, {0});
    return exp_vec / sum;
}

int main() {
    using namespace jax;
    // Construct tracer:
    jaxpr_builder builder {};

    auto x = builder.register_tracer(type_enum::F32, 10);
    builder.register_output(reduce_sum(softmax(x), {0}));

    expression jaxpr = builder.get_jaxpr();
    std::cout << "Original expression:\n" << jaxpr;


    expression grad_jaxpr = grad(jaxpr);
    std::cout << "Grad expression:\n" << grad_jaxpr;

    grad_jaxpr.eliminate_dead_code();
    std::cout << "Grad expression (DCE):\n" << grad_jaxpr;

    return 0;
}
