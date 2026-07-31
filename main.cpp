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
    switch_example();
    return 0;
}

void scan_example() {
    using namespace jax;
    using enum dtype_t;

    jaxpr_builder builder {};
    auto k0 = builder.register_tracer(F32, 3);
    auto k1 = builder.register_tracer(F32, 2, 3);
    auto k2 = builder.register_tracer(F32);
    auto acc0 = builder.register_tracer(F32, 3);
    auto sum0 = builder.register_tracer(F32);
    auto mat0 = builder.register_tracer(F32, 2, 3);
    auto xs = builder.register_tracer(F32, 4, 3);
    auto ws = builder.register_tracer(F32, 4);
    auto ms = builder.register_tracer(F32, 4, 2, 3);

    auto f = [](auto k0, auto k1, auto k2, auto acc, auto sum, auto mat, auto x, auto w, auto m) {
        auto scaled = x * w;
        auto acc2 = acc + scaled * k0;
        auto sum2 = sum + w * k2;
        auto mat2 = mat * k1 + m;
        auto y0 = acc2 + x * k0;
        auto y1 = reduce_sum(mat2, {1}) + sum2;
        auto y2 = mat2 * k1;
        return std::tuple{acc2, sum2, mat2, y0, y1, y2};
    };

    auto results = scan(
        f,
        std::tuple{k0, k1, k2},
        std::tuple{acc0, sum0, mat0},
        std::tuple{xs, ws, ms},
        4
    );

    auto acc_f = std::get<0>(results);
    auto sum_f = std::get<1>(results);
    auto mat_f = std::get<2>(results);
    auto y0s = std::get<3>(results);
    auto y1s = std::get<4>(results);
    auto y2s = std::get<5>(results);
    auto a = reduce_sum(acc_f * k0, {0});
    auto b = sum_f * k2;
    auto c = reduce_sum(mat_f * k1, {0, 1});
    auto d = reduce_sum(y0s * xs, {0, 1});
    auto e = reduce_sum(y1s, {0, 1}) * reduce_sum(ws, {0});
    auto g = reduce_sum(y2s * ms, {0, 1, 2});
    builder.register_output(a + b + c + d + e + g);

    auto jaxpr = builder.get_jaxpr();
    std::cout << "Original expression:\n" << jaxpr;

    auto grad_jaxpr = grad(jaxpr);
    std::cout << "Grad expression:\n" << grad_jaxpr;

    // grad_jaxpr.eliminate_dead_code();
    // std::cout << "DCE grad expression:\n" << grad_jaxpr;
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
         [&k](auto&& x_, auto&& y_){return x_ + y_ - k;},
         [&k](const auto& x_, const auto& y_){return x_ * y_ * k;}
    }, x, y);

    /*

    auto z = scan(f, L, num_carry, std::tuple{carry}, std::tuple{xs})

    */

    builder.register_output(reduce_sum(z, {0, 1, 2}));

    auto jaxpr = builder.get_jaxpr();
    std::cout << "Original expression:\n" << jaxpr;

    auto grad_jaxpr = grad(jaxpr);

    std::cout << "Grad expression:\n" << grad_jaxpr;

    grad_jaxpr.eliminate_dead_code();
    std::cout << "DCE grad expression:\n" << grad_jaxpr;
}
