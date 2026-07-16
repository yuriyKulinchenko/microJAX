#include <iostream>

#include "jax_vm.h"
#include "jax_types.h"
#include "helper.h"
#include "jax_logger.h"
#include "jax_functions.h"
#include "tracer.h"
#include "grad.h"



template<typename T>
T softmax(T vec) {
    using std::exp;
    auto exp_vec = exp(vec);
    auto sum = jax::reduce_sum(exp_vec, {0});
    return exp_vec / sum;
}

void switch_example();
void scan_example();

int main() {
    using namespace jax;
    using enum dtype_t;
    array_t A = array_t{type_t{F32, 3, 3}, {
        5, 8, 9,
        1, 6, 7,
        4, 5, 3
    }};

    array_t B = array_t{type_t{F32, 3, 3}, {
        4, 8, 5,
        5, 1, 1,
        9, 6, 4
    }};

    array_t C = array_t{type_t{F32, 2, 3}, {
        1, 2, 3,
        4, 5, 6
    }};

    // std::cout << "Product: " << dot_general(A, B, {1}, {0}, {}, {}) << '\n';
    // std::cout << "Reduction of A: " << reduce_sum(A, {0}) << '\n';
    // std::cout << "Reduction of B along other axes: " << reduce_sum(B, {1}) << '\n';
    // std::cout << "C: " << C << '\n';
    // std::cout << "C transposed: " << transpose(C, {1, 0}) << '\n';

    jaxpr_builder builder {};
    auto x = builder.register_tracer(F32, 4);
    builder.register_output(softmax(x));
    auto jaxpr = builder.get_jaxpr();

    std::cout << "Softmax jaxpr: " << jaxpr;

    array_t input = array_t{type_t{F32, 4}, {1, 2, 3, 4}};

    jax_vm vm(jaxpr);
    std::cout << vm.run({input})[0];
    return 0;
}

void scan_example() {
    using namespace jax;
    using enum dtype_t;

    jaxpr_builder builder {};

    auto acc0 = builder.register_tracer(F32, 3);
    auto cnt0 = builder.register_tracer(F32);
    auto xs   = builder.register_tracer(F32, 5, 3);
    auto ws   = builder.register_tracer(F32, 5);

    auto f = [&](jaxpr_tracer acc, jaxpr_tracer cnt, jaxpr_tracer x, jaxpr_tracer w) {
        auto wx   = x * w;
        auto acc2 = acc + wx;
        auto cnt2 = cnt + w;
        auto y    = acc2 + x;      // per-step output: f32[3]
        return std::array{acc2, cnt2, y};
    };

    auto results = scan(
        builder,
        f,
        std::array{acc0, cnt0},
        std::array{xs, ws},
        5
    );

    auto z = results[0] * results[1] - results[2]; // f32[5,3]
    builder.register_output(reduce_sum(z, {0, 1}));

    auto jaxpr = builder.get_jaxpr();
    std::cout << "Original expression:\n" << jaxpr;

    auto grad_jaxpr = grad(jaxpr);
    std::cout << "Grad expression:\n" << grad_jaxpr;

    grad_jaxpr.eliminate_dead_code();
    std::cout << "DCE grad expression:\n" << grad_jaxpr;
}

void switch_example() {
     using namespace jax;
     using enum dtype_t;
     // Construct tracer:
     jaxpr_builder builder {};

     auto i = builder.register_tracer(I32);
     auto j = builder.register_tracer(I32);
     auto x = builder.register_tracer(F32, 2, 2);
     auto y = builder.register_tracer(F64, 2, 2);
     auto k = array_t::build(F32, {4, 2, 2}, [](auto& is) -> double {
         return is[0] + is[1] + is[2];
     });

     auto z = switch_on(i < j, std::tuple{
         [&](auto& x, auto& y){return x + y - k;},
         [&](auto& x, auto& y){return x * y * k;}
     }, x, y);

     /*

     auto z = scan(f, L, num_carry, std::tuple{carry}, std::tuple{xs})

     */

     builder.register_output(reduce_sum(z, {0, 1}));

     auto jaxpr = builder.get_jaxpr();
     std::cout << "Original expression:\n" << jaxpr;

     auto grad_jaxpr = grad(jaxpr);
     std::cout << "Grad expression:\n" << grad_jaxpr;

     grad_jaxpr.eliminate_dead_code();
     std::cout << "DCE grad expression:\n" << grad_jaxpr;
}
