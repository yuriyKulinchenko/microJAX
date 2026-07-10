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
    using enum dtype_t;
    // Construct tracer:
    jaxpr_builder builder {};

    auto i = builder.register_tracer(I32);
    auto j = builder.register_tracer(I32);
    auto x = builder.register_tracer(F32, 2, 2);
    auto y = builder.register_tracer(F64, 2, 2);
    auto k = array_t::build<f32>({4, 2, 2}, [](auto& is) {
        return static_cast<f32>(is[0] + is[1] + is[2]);
    });

    auto z = switch_on(i < j, std::tuple{
        [&](auto& x, auto& y){return x + y - k;},
        [&](auto& x, auto& y){return x * y * k;}
    }, x, y);

    builder.register_output(reduce_sum(z, {0, 1}));

    auto jaxpr = builder.get_jaxpr();
    std::cout << "Original expression:\n" << jaxpr;

    auto grad_jaxpr = grad(jaxpr);
    std::cout << "Grad expression:\n" << grad_jaxpr;

    // auto grad_general_jaxpr = grad_general(jaxpr);
    // std::cout << "Grad general expression:\n" << grad_general_jaxpr;
}
