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

    array_t identity = array_t::build<f32>(type_t{type_enum::F32, 2, 2}, [](auto& i) {
       return i[0] == i[1];
    });

    std::cout << (identity + identity) * (identity + identity);

    return 0;
    // Construct tracer:
    jaxpr_builder builder {};
    jaxpr_tracer tracer_x = builder.register_tracer(type_enum::F32);
    jaxpr_tracer tracer_y = builder.register_tracer(type_enum::F32);

    auto output = test_function<jaxpr_tracer>(tracer_x, tracer_y);
    builder.register_output(output);

    std::cout << "Original expression:\n";
    std::cout << builder.jaxpr;

    std::cout << "Transformed expression:\n";
    std::cout << grad(builder.jaxpr);
}
