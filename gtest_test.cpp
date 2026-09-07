#include <gtest/gtest.h>
#include<vector>

#include "grad.h"
#include "jax_functions.h"
#include "jax_vm.h"
#include "jax_types.h"
#include "tracer.h"
#include "jax_format.h"

TEST(sanity, resultant_type_1) {
    using namespace jax;
    EXPECT_EQ(resultant_type(dtype_t::I32, dtype_t::F64), dtype_t::F64);
}

TEST(sanity, resultant_type_2) {
    using namespace jax;
    EXPECT_EQ(resultant_type(dtype_t::I32, dtype_t::I32), dtype_t::I32);
}

TEST(vm, matmul) {
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

    array_t expected_output = array_t{type_t{F32, 3, 3}, {
        141, 102, 69,
        97, 56, 39,
        68, 55, 37
    }};

    auto matmul = [](auto x, auto y) {
        return dot_general(x, y, {{1}, {0}}, {{}, {}});
    };

    expression jaxpr = get_jaxpr(matmul, type_t{F32, 3, 3}, type_t{F32, 3, 3});

    EXPECT_EQ(matmul(A, B), expected_output);
    EXPECT_EQ(invoke_vm(jaxpr, {A, B}), expected_output);
}

TEST(vm, matmul_batch) {
    using namespace jax;
    using enum dtype_t;
    array_t A {type_t{F32, 2, 3, 3}, {
        5, 8, 9,
        1, 6, 7,
        4, 5, 3,

        4, 8, 5,
        5, 1, 1,
        9, 6, 4
    }};

    array_t B {type_t{F32, 2, 3, 3}, {
        4, 8, 5,
        5, 1, 1,
        9, 6, 4,

        5, 8, 9,
        1, 6, 7,
        4, 5, 3,
    }};

    array_t expected_output {type_t{F32, 2, 3, 3}, {
        141, 102, 69,
        97, 56, 39,
        68, 55, 37,

        48, 105, 107,
        30, 51, 55,
        67, 128, 135
    }};

    auto matmul_batch = [](auto x, auto y) {
        return dot_general(x, y, {2}, {1}, {0}, {0});
    };

    expression jaxpr = get_jaxpr(matmul_batch, type_t{F32, 2, 3, 3}, type_t{F32, 2, 3, 3});

    jax_vm vm{jaxpr};

    EXPECT_EQ(matmul_batch(A, B), expected_output);
    EXPECT_EQ(invoke_vm(jaxpr, {A, B}), expected_output);
}

TEST(grad_closed_form, linear) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of y = 100 * x + 12

    array_t input {0.78}; // f32
    array_t expected_derivative {100}; // f32

    expression derivate_jaxpr {grad(get_jaxpr(
        [](const jaxpr_tracer& x) -> jaxpr_tracer {
            return 100 * x + 12;
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(derivate_jaxpr, input), 100);
}

TEST(grad_closed_form, trig) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of y = sin(x), cos(x):

    array_t input {0.78};

    expression sin_derivative_jaxpr {grad(get_jaxpr(
        [](auto x) {
            return sin(x);
        }, type_t{F32}))};

    expression cos_derivative_jaxpr {grad(get_jaxpr(
        [](auto x) {
            return cos(x);
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(sin_derivative_jaxpr, input), cos(input));
    EXPECT_EQ(invoke_vm(cos_derivative_jaxpr, input), -sin(input));
}

TEST(grad_closed_form, exp) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of y = exp(x):

    array_t input {0.78};

    expression exp_derivative_jaxpr {grad(get_jaxpr(
        [](auto x) {
            return exp(x);
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(exp_derivative_jaxpr, input), exp(input));
}

TEST(grad_closed_form, log) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of y = log(x):

    array_t input {0.78};

    expression log_derivative_jaxpr {grad(get_jaxpr([](auto x) {
        return log(x);
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(log_derivative_jaxpr, input), array_t{1.} / input);
};

TEST(grad_closed_form, arithmetic_add) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of z = x + y:

    array_t x {0.78};
    array_t y {5.32};

    expression sum_grad_jaxpr {grad(get_jaxpr([](auto x, auto y) {
        return x + y;
    }, type_t{F32}, type_t{F32}))};

    std::vector<array_t> expected_output {1., 1.};
    EXPECT_EQ(invoke_vm<false>(sum_grad_jaxpr, {x, y}), expected_output);
}

TEST(grad_closed_form, arithmetic_sub) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of z = x - y:

    array_t x {0.78};
    array_t y {5.32};

    expression sum_grad_jaxpr {grad(get_jaxpr([](auto x, auto y) {
        return x - y;
    }, type_t{F32}, type_t{F32}))};

    std::vector<array_t> expected_output {1., -1.};
    EXPECT_EQ(invoke_vm<false>(sum_grad_jaxpr, {x, y}), expected_output);
}

TEST(grad_closed_form, arithmetic_mul) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of z = x * y:

    array_t x {0.78};
    array_t y {5.32};

    expression sum_grad_jaxpr {grad(get_jaxpr([](auto x, auto y) {
        return x * y;
    }, type_t{F32}, type_t{F32}))};

    std::vector expected_output {y, x};
    EXPECT_EQ(invoke_vm<false>(sum_grad_jaxpr, {x, y}), expected_output);
}

TEST(grad_closed_form, arithmetic_div) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of z = x / y:

    array_t x {0.78};
    array_t y {5.32};

    expression sum_grad_jaxpr {grad(get_jaxpr([](auto x, auto y) {
        return x / y;
    }, type_t{F32}, type_t{F32}))};

    auto grad_result = invoke_vm<false>(sum_grad_jaxpr, {x, y});
    std::vector expected_output {array_t{1} / y, - x / (y * y)};

    EXPECT_EQ(grad_result, expected_output);
}

TEST(grad_closed_form, composition_1) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of y = exp(sin(x)):
    // dy/dx = cos(x) * exp(sin(x))

    array_t input {0.78};

    expression derivative_jaxpr {grad(get_jaxpr([](auto x) {
        return exp(sin(x));
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(derivative_jaxpr, input), cos(input) * exp(sin(input)));
}

TEST(grad_closed_form, composition_2) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of y = log(cos(x) + 2):
    // dy/dx = -sin(x) / (cos(x) + 2)

    array_t input {0.78};

    expression derivative_jaxpr {grad(get_jaxpr([](auto x) {
        return log(cos(x) + 2);
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(derivative_jaxpr, input),
              -sin(input) / (cos(input) + 2));
}

TEST(grad_closed_form, composition_3) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of z = sin(x * y) * exp(y):
    // dz/dx = y * cos(x * y) * exp(y)
    // dz/dy = (x * cos(x * y) + sin(x * y)) * exp(y)

    array_t x {0.78};
    array_t y {1.23};

    expression grad_jaxpr {grad(get_jaxpr([](auto x, auto y) {
        return sin(x * y) * exp(y);
    }, type_t{F32}, type_t{F32}))};

    std::vector expected_output {
        y * cos(x * y) * exp(y),
        (x * cos(x * y) + sin(x * y)) * exp(y)
    };

    EXPECT_EQ(invoke_vm<false>(grad_jaxpr, {x, y}), expected_output);
}

TEST(grad_closed_form, composition_4) {
    using namespace jax;
    using enum dtype_t;

    // Testing derivative of z = exp(x - y) / log(x + y):
    // let u = exp(x - y), v = log(x + y)
    // dz/dx = (u * v - u / (x + y)) / (v * v)
    // dz/dy = (-u * v - u / (x + y)) / (v * v)

    array_t x {2.34};
    array_t y {1.11};

    expression grad_jaxpr {grad(get_jaxpr([](auto x, auto y) {
        return exp(x - y) / log(x + y);
    }, type_t{F32}, type_t{F32}))};

    auto u = exp(x - y);
    auto v = log(x + y);
    std::vector expected_output {
        (u * v - u / (x + y)) / (v * v),
        (-u * v - u / (x + y)) / (v * v)
    };

    EXPECT_EQ(invoke_vm<false>(grad_jaxpr, {x, y}), expected_output);
}

std::vector<array_t> grad_fdm(const expression& jaxpr, std::vector<array_t>& x, double h) {
    if (jaxpr.outvals.size() != 1) {
        throw std::logic_error("Error: there must be only one differentiable output");
    }

    auto y = jax_vm{jaxpr}.run(x);

    std::vector<array_t> derivative {};

    // Compute the directional derivative for each input:

    for (size_t i = 0; i < x.size(); i++) {
        auto x_0 = x[i];
        if (x_0.get_type().get_shape().size() != 0) {
            throw std::logic_error("Error: grad_fdm only works for scalar inputs");
        }
        if (is_integral(x_0.get_type().get_dtype())) {
            derivative.emplace_back(array_t{x_0.get_type(), {0}});
        } else {
            x[i] = x_0 + h;
            auto y_h = jax_vm{jaxpr}.run(x);
            derivative.push_back((y_h[0] - y[0]) / h);
            x[i] = x_0;
        }
    }

    return derivative;
}

// ============================ broadcast_in_dim ============================
// Closed form: broadcast is fully determined, so we hand-write every result.

TEST(tensor_broadcast, scalar_to_vector) {
    using namespace jax; using enum dtype_t;
    array_t x {5.};
    EXPECT_EQ(x.broadcast_in_dim({3}, {}), (array_t{type_t{F32, 3}, {5, 5, 5}}));
}

TEST(tensor_broadcast, vector_to_matrix) {
    using namespace jax; using enum dtype_t;
    // src axis 0 maps to target axis 1: each row is a copy of the source vector.
    array_t x {type_t{F32, 3}, {1, 2, 3}};
    EXPECT_EQ(x.broadcast_in_dim({2, 3}, {1}),
              (array_t{type_t{F32, 2, 3}, {1, 2, 3, 1, 2, 3}}));
}

TEST(tensor_broadcast, stretch_size_one_vector) {
    using namespace jax; using enum dtype_t;
    // A leading dim of size 1 is stretched: [1] -> [4].
    array_t x {type_t{F32, 1}, {7}};
    EXPECT_EQ(x.broadcast_in_dim({4}, {0}),
              (array_t{type_t{F32, 4}, {7, 7, 7, 7}}));
}

TEST(tensor_broadcast, stretch_size_one_leading) {
    using namespace jax; using enum dtype_t;
    // [1,3] -> [2,3]: the leading 1 is stretched.
    array_t x {type_t{F32, 1, 3}, {1, 2, 3}};
    EXPECT_EQ(x.broadcast_in_dim({2, 3}, {0, 1}),
              (array_t{type_t{F32, 2, 3}, {1, 2, 3, 1, 2, 3}}));
}

TEST(tensor_broadcast, stretch_size_one_trailing) {
    using namespace jax; using enum dtype_t;
    // [3,1] -> [3,4]: the trailing 1 is stretched.
    array_t x {type_t{F32, 3, 1}, {1, 2, 3}};
    EXPECT_EQ(x.broadcast_in_dim({3, 4}, {0, 1}),
              (array_t{type_t{F32, 3, 4}, {1, 1, 1, 1,
                                           2, 2, 2, 2,
                                           3, 3, 3, 3}}));
}

TEST(tensor_broadcast, via_vm) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 3}, {1, 2, 3}};
    expression jaxpr = get_jaxpr([](auto v) {
        return v.broadcast_in_dim({2, 3}, {1});
    }, type_t{F32, 3});
    EXPECT_EQ(invoke_vm(jaxpr, x), (array_t{type_t{F32, 2, 3}, {1, 2, 3, 1, 2, 3}}));
}

// ============================ reduce_sum ============================

TEST(tensor_reduce, vector_to_scalar) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 4}, {1, 2, 3, 4}};
    EXPECT_EQ(x.reduce_sum({0}), (array_t{type_t{F32}, {10}}));
}

TEST(tensor_reduce, matrix_axis_0) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 2, 3}, {1, 2, 3, 4, 5, 6}};
    EXPECT_EQ(x.reduce_sum({0}), (array_t{type_t{F32, 3}, {5, 7, 9}}));
}

TEST(tensor_reduce, matrix_axis_1) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 2, 3}, {1, 2, 3, 4, 5, 6}};
    EXPECT_EQ(x.reduce_sum({1}), (array_t{type_t{F32, 2}, {6, 15}}));
}

TEST(tensor_reduce, matrix_all_axes) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 2, 3}, {1, 2, 3, 4, 5, 6}};
    EXPECT_EQ(x.reduce_sum({0, 1}), (array_t{type_t{F32}, {21}}));
}

// ============================ transpose ============================

TEST(tensor_transpose, matrix) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 2, 3}, {1, 2, 3, 4, 5, 6}};
    EXPECT_EQ(x.transpose({1, 0}), (array_t{type_t{F32, 3, 2}, {1, 4, 2, 5, 3, 6}}));
}

TEST(tensor_transpose, rank3_reverse) {
    using namespace jax; using enum dtype_t;
    array_t x {type_t{F32, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}};
    // out[a][b][c] = in[c][b][a]
    EXPECT_EQ(x.transpose({2, 1, 0}),
              (array_t{type_t{F32, 2, 2, 2}, {1, 5, 3, 7, 2, 6, 4, 8}}));
}

// ============================ dot_general ============================

TEST(tensor_dot, vector_dot) {
    using namespace jax; using enum dtype_t;
    array_t a {type_t{F32, 3}, {1, 2, 3}};
    array_t b {type_t{F32, 3}, {4, 5, 6}};
    // contract axis 0 of both, no free/batch -> scalar 1*4 + 2*5 + 3*6 = 32
    EXPECT_EQ(a.dot_general(b, {0}, {0}, {}, {}), (array_t{type_t{F32}, {32}}));
}

TEST(tensor_dot, matrix_vector) {
    using namespace jax; using enum dtype_t;
    array_t a {type_t{F32, 2, 3}, {1, 2, 3, 4, 5, 6}};
    array_t b {type_t{F32, 3}, {7, 8, 9}};
    // contract a's axis 1 with b's axis 0 -> [2]: [50, 122]
    EXPECT_EQ(a.dot_general(b, {1}, {0}, {}, {}), (array_t{type_t{F32, 2}, {50, 122}}));
}

// ============================ convert_element_type ============================

TEST(tensor_convert, f32_to_i32) {
    using namespace jax; using enum dtype_t;
    // convert only relabels the dtype; the stored values are unchanged.
    array_t x {type_t{F32, 2}, {2, 3}};
    EXPECT_EQ(x.convert_element_type(I32), (array_t{type_t{I32, 2}, {2, 3}}));
}

// ============================ elementwise (implicit broadcast) ============================

TEST(tensor_elementwise, matrix_plus_row) {
    using namespace jax; using enum dtype_t;
    array_t a {type_t{F32, 2, 3}, {1, 2, 3, 4, 5, 6}};
    array_t row {type_t{F32, 3}, {10, 20, 30}};
    EXPECT_EQ(a + row, (array_t{type_t{F32, 2, 3}, {11, 22, 33, 14, 25, 36}}));
}

TEST(tensor_elementwise, scalar_times_vector) {
    using namespace jax; using enum dtype_t;
    array_t v {type_t{F32, 3}, {1, 2, 3}};
    EXPECT_EQ(array_t{2.} * v, (array_t{type_t{F32, 3}, {2, 4, 6}}));
}

TEST(tensor_elementwise, both_operands_broadcast) {
    using namespace jax; using enum dtype_t;
    // [2,1] + [1,3] -> [2,3], both operands get stretched
    array_t col {type_t{F32, 2, 1}, {1, 2}};
    array_t row {type_t{F32, 1, 3}, {10, 20, 30}};
    EXPECT_EQ(col + row, (array_t{type_t{F32, 2, 3}, {11, 21, 31, 12, 22, 32}}));
}

TEST(tensor_elementwise, less_than) {
    using namespace jax; using enum dtype_t;
    array_t a {type_t{F32, 3}, {1, 2, 3}};
    array_t b {type_t{F32, 3}, {3, 2, 1}};
    EXPECT_EQ(a < b, (array_t{type_t{F32, 3}, {1, 0, 0}}));
}

// ============================ cond (both branches) ============================

TEST(vm_cond, two_branches) {
    using namespace jax; using enum dtype_t;

    auto f = [](auto idx, auto a, auto b) {
        return switch_on(idx, std::tuple{
            [](auto a, auto b){ return a + b; },   // branch 0
            [](auto a, auto b){ return a * b; }    // branch 1
        }, a, b);
    };

    expression jaxpr = get_jaxpr(f, type_t{I32}, type_t{F32, 2, 2}, type_t{F32, 2, 2});

    array_t a {type_t{F32, 2, 2}, {1, 2, 3, 4}};
    array_t b {type_t{F32, 2, 2}, {5, 6, 7, 8}};
    array_t idx0 {type_t{I32}, {0}};
    array_t idx1 {type_t{I32}, {1}};

    EXPECT_EQ(invoke_vm(jaxpr, {idx0, a, b}), a + b);
    EXPECT_EQ(invoke_vm(jaxpr, {idx1, a, b}), a * b);
}

TEST(vm_cond, three_branches) {
    using namespace jax; using enum dtype_t;

    auto f = [](auto idx, auto a, auto b) {
        return switch_on(idx, std::tuple{
            [](auto a, auto b){ return a - b; },   // branch 0
            [](auto a, auto b){ return a + b; },   // branch 1
            [](auto a, auto b){ return a * b; }    // branch 2
        }, a, b);
    };

    expression jaxpr = get_jaxpr(f, type_t{I32}, type_t{F32, 2}, type_t{F32, 2});

    array_t a {type_t{F32, 2}, {6, 8}};
    array_t b {type_t{F32, 2}, {2, 4}};

    EXPECT_EQ(invoke_vm(jaxpr, {array_t{type_t{I32}, {0}}, a, b}), a - b);
    EXPECT_EQ(invoke_vm(jaxpr, {array_t{type_t{I32}, {1}}, a, b}), a + b);
    EXPECT_EQ(invoke_vm(jaxpr, {array_t{type_t{I32}, {2}}, a, b}), a * b);
}

// ============================ grad of tensor ops ============================

TEST(grad_tensor, broadcast_reduce_is_linear) {
    using namespace jax; using enum dtype_t;
    // f(x) = reduce_sum(broadcast(x, [3])) = 3x, so df/dx = 3 everywhere.
    expression grad_jaxpr {grad(get_jaxpr([](auto x) {
        return jax::reduce_sum(x.broadcast_in_dim({3}, {}), {0});
    }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(grad_jaxpr, array_t{2.5}), array_t{3.});
    EXPECT_EQ(invoke_vm(grad_jaxpr, array_t{-4.}), array_t{3.});
}

// ============================ grad via finite differences ============================
// grad_fdm only handles scalar inputs, so we embed the tensor ops inside a
// scalar-in / scalar-out function (broadcast the scalar up, operate, reduce
// back down) and check the analytic gradient against the numeric one.

TEST(grad_fdm_check, tensor_pipeline) {
    using namespace jax; using enum dtype_t;

    auto f = [](auto x, auto y) {
        auto xv = x.broadcast_in_dim({4}, {});
        auto yv = y.broadcast_in_dim({4}, {});
        auto t = sin(xv * yv) + exp(xv);
        return jax::reduce_sum(t, {0});
    };

    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32});

    std::vector<array_t> inputs {array_t{type_t{F32}, {0.6}}, array_t{type_t{F32}, {1.3}}};

    auto analytic = invoke_vm<false>(grad(jaxpr), inputs);
    auto numeric = grad_fdm(jaxpr, inputs, 1e-6);

    ASSERT_EQ(analytic.size(), numeric.size());
    for (size_t i = 0; i < analytic.size(); i++) {
        EXPECT_NEAR(analytic[i].get_value()[0], numeric[i].get_value()[0], 1e-3);
    }
}

TEST(grad_fdm_check, div_and_trig) {
    using namespace jax; using enum dtype_t;

    // A messy scalar function where hand-deriving is annoying: validate against FDM.
    auto f = [](auto x, auto y) {
        return log(cos(x * y) + 2) * exp(x) - y / (x * x + 1);
    };

    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32});

    std::vector<array_t> inputs {array_t{type_t{F32}, {1.1}}, array_t{type_t{F32}, {0.4}}};

    auto analytic = invoke_vm<false>(grad(jaxpr), inputs);
    auto numeric = grad_fdm(jaxpr, inputs, 1e-6);

    std::cout << grad(jaxpr);

    ASSERT_EQ(analytic.size(), numeric.size());
    for (size_t i = 0; i < analytic.size(); i++) {
        EXPECT_NEAR(analytic[i].get_value()[0], numeric[i].get_value()[0], 1e-3);
    }
}

// ============================ grad of new unary ops ============================
// Closed form: single scalar in / scalar out, checked against the hand-derivative.

TEST(grad_closed_form, sqrt) {
    using namespace jax; using enum dtype_t;
    expression d {grad(get_jaxpr([](auto x) { return sqrt(x); }, type_t{F32}))};
    // d/dx sqrt(x) = 1 / (2 sqrt(x))
    EXPECT_EQ(invoke_vm(d, array_t{2.5}), array_t{0.5} / sqrt(array_t{2.5}));
    EXPECT_EQ(invoke_vm(d, array_t{0.7}), array_t{0.5} / sqrt(array_t{0.7}));
}

TEST(grad_closed_form, rsqrt) {
    using namespace jax; using enum dtype_t;
    expression d {grad(get_jaxpr([](auto x) { return rsqrt(x); }, type_t{F32}))};
    // d/dx x^{-1/2} = -1/2 x^{-3/2} = -1/2 rsqrt(x)^3
    EXPECT_EQ(invoke_vm(d, array_t{2.5}), array_t{-0.5} * integer_pow(rsqrt(array_t{2.5}), 3));
    EXPECT_EQ(invoke_vm(d, array_t{0.7}), array_t{-0.5} * integer_pow(rsqrt(array_t{0.7}), 3));
}

TEST(grad_closed_form, tanh) {
    using namespace jax; using enum dtype_t;
    expression d {grad(get_jaxpr([](auto x) { return tanh(x); }, type_t{F32}))};
    // d/dx tanh(x) = 1 - tanh(x)^2
    EXPECT_EQ(invoke_vm(d, array_t{0.4}), array_t{1.} - tanh(array_t{0.4}) * tanh(array_t{0.4}));
    EXPECT_EQ(invoke_vm(d, array_t{-1.2}), array_t{1.} - tanh(array_t{-1.2}) * tanh(array_t{-1.2}));
}

TEST(grad_closed_form, logistic) {
    using namespace jax; using enum dtype_t;
    expression d {grad(get_jaxpr([](auto x) { return logistic(x); }, type_t{F32}))};
    // d/dx sigma(x) = sigma(x) (1 - sigma(x))
    EXPECT_EQ(invoke_vm(d, array_t{0.4}), logistic(array_t{0.4}) * (array_t{1.} - logistic(array_t{0.4})));
    EXPECT_EQ(invoke_vm(d, array_t{-1.2}), logistic(array_t{-1.2}) * (array_t{1.} - logistic(array_t{-1.2})));
}

TEST(grad_closed_form, integer_pow) {
    using namespace jax; using enum dtype_t;
    array_t input {1.7};
    // n == 1 and n == 2 are carved out; n == 3, 5 exercise the general path.
    expression d1 {grad(get_jaxpr([](auto x) { return integer_pow(x, 1); }, type_t{F32}))};
    expression d2 {grad(get_jaxpr([](auto x) { return integer_pow(x, 2); }, type_t{F32}))};
    expression d3 {grad(get_jaxpr([](auto x) { return integer_pow(x, 3); }, type_t{F32}))};
    expression d5 {grad(get_jaxpr([](auto x) { return integer_pow(x, 5); }, type_t{F32}))};

    EXPECT_EQ(invoke_vm(d1, input), array_t{1.});
    EXPECT_EQ(invoke_vm(d2, input), array_t{2.} * input);
    EXPECT_EQ(invoke_vm(d3, input), array_t{3.} * integer_pow(input, 2));
    EXPECT_EQ(invoke_vm(d5, input), array_t{5.} * integer_pow(input, 4));
}

// ============================ grad of new binary ops ============================

TEST(grad_closed_form, pow) {
    using namespace jax; using enum dtype_t;
    array_t x {1.8};
    array_t y {2.3};
    expression d {grad(get_jaxpr([](auto x, auto y) { return pow(x, y); }, type_t{F32}, type_t{F32}))};
    // dz/dx = y x^{y-1}, dz/dy = x^y ln(x)
    std::vector expected { y * pow(x, y - array_t{1.}), pow(x, y) * log(x) };
    EXPECT_EQ(invoke_vm<false>(d, {x, y}), expected);
}

TEST(grad_closed_form, max) {
    using namespace jax; using enum dtype_t;
    expression d {grad(get_jaxpr([](auto x, auto y) { return max(x, y); }, type_t{F32}, type_t{F32}))};
    // Gradient flows to the larger operand; ties are routed to the second (y).
    EXPECT_EQ(invoke_vm<false>(d, {array_t{5.}, array_t{2.}}), (std::vector<array_t>{1., 0.}));
    EXPECT_EQ(invoke_vm<false>(d, {array_t{2.}, array_t{5.}}), (std::vector<array_t>{0., 1.}));
    EXPECT_EQ(invoke_vm<false>(d, {array_t{3.}, array_t{3.}}), (std::vector<array_t>{0., 1.}));
}

TEST(grad_closed_form, min) {
    using namespace jax; using enum dtype_t;
    expression d {grad(get_jaxpr([](auto x, auto y) { return min(x, y); }, type_t{F32}, type_t{F32}))};
    // Gradient flows to the smaller operand; ties are routed to the second (y).
    EXPECT_EQ(invoke_vm<false>(d, {array_t{2.}, array_t{5.}}), (std::vector<array_t>{1., 0.}));
    EXPECT_EQ(invoke_vm<false>(d, {array_t{5.}, array_t{2.}}), (std::vector<array_t>{0., 1.}));
    EXPECT_EQ(invoke_vm<false>(d, {array_t{3.}, array_t{3.}}), (std::vector<array_t>{0., 1.}));
}

// ============================ grad of tensor ops via FDM ============================
// The tensor ops need vector inputs, but grad_fdm only perturbs scalars, so each
// test stacks a handful of scalar inputs into a vector, applies the op, and reduces
// back to a scalar. Distinct values keep max/min/argmax away from ties.

template <typename... Ts>
auto stack_scalars(const Ts&... xs) {
    using namespace jax;
    return concatenate(std::tuple{xs.broadcast_in_dim({1}, {})...}, 0);
}

void check_grad_fdm(const jax::expression& jaxpr, std::vector<jax::array_t> inputs, double tol = 1e-3) {
    using namespace jax;
    auto analytic = invoke_vm<false>(grad(jaxpr), inputs);
    auto numeric = grad_fdm(jaxpr, inputs, 1e-6);
    ASSERT_EQ(analytic.size(), numeric.size());
    for (size_t i = 0; i < analytic.size(); i++) {
        EXPECT_NEAR(analytic[i].get_value()[0], numeric[i].get_value()[0], tol);
    }
}

TEST(grad_tensor_fdm, reduce_max) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c) { return reduce_max(stack_scalars(a, b, c), {0}); };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{1.}, array_t{3.}, array_t{2.}});
    check_grad_fdm(jaxpr, {array_t{4.}, array_t{-1.}, array_t{0.5}});
}

TEST(grad_tensor_fdm, reduce_min) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c) { return reduce_min(stack_scalars(a, b, c), {0}); };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{3.}, array_t{1.}, array_t{2.}});
    check_grad_fdm(jaxpr, {array_t{-1.}, array_t{4.}, array_t{0.5}});
}

TEST(grad_tensor_fdm, concatenate) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c) { return reduce_sum(sin(stack_scalars(a, b, c)), {0}); };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}});
}

TEST(grad_tensor_fdm, reshape) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c, auto d) {
        return reduce_sum(sin(reshape(stack_scalars(a, b, c, d), {2, 2})), {0, 1});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}, array_t{0.9}});
}

TEST(grad_tensor_fdm, slice) {
    using namespace jax; using enum dtype_t;
    // Length-4 vector, slice [1, 4) with stride 2 -> indices {1, 3}.
    auto f = [](auto a, auto b, auto c, auto d) {
        return reduce_sum(sin(slice(stack_scalars(a, b, c, d), {1}, {4}, {2})), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}, array_t{0.9}});
}

TEST(grad_tensor_fdm, pad) {
    using namespace jax; using enum dtype_t;
    // Length-2 vector, (low, high, interior) = (1, 1, 1) -> length 5: [k, a, k, b, k].
    // Exercises both the operand (slice) and pad-value (masked sum) branches.
    auto f = [](auto a, auto b, auto k) {
        return reduce_sum(sin(pad(stack_scalars(a, b), k, {{1, 1, 1}})), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});

    std::cout << jaxpr;
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}});
}

TEST(grad_tensor_fdm, gather) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c) {
        auto v = stack_scalars(a, b, c);
        array_t idx {type_t{I32, 3}, {2, 0, 2}};
        return reduce_sum(sin(v.at(idx).get()), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}});
}

TEST(grad_tensor_fdm, scatter_add) {
    using namespace jax; using enum dtype_t;
    // operand {a, b}, add update {c} at index 0 -> [a + c, b].
    auto f = [](auto a, auto b, auto c) {
        auto operand = stack_scalars(a, b);
        array_t idx {type_t{I32, 1}, {0}};
        return reduce_sum(sin(operand.at(idx).add(c.broadcast_in_dim({1}, {}))), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}});
}

TEST(grad_tensor_fdm, scatter_mul) {
    using namespace jax; using enum dtype_t;
    // operand {a, b}, multiply update {c} at index 0 -> [a * c, b]. Nonzero c avoids /0.
    auto f = [](auto a, auto b, auto c) {
        auto operand = stack_scalars(a, b);
        array_t idx {type_t{I32, 1}, {0}};
        return reduce_sum(operand.at(idx).multiply(c.broadcast_in_dim({1}, {})), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{1.5}, array_t{1.1}, array_t{2.3}});
}

TEST(grad_tensor_fdm, scatter_max) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c) {
        auto operand = stack_scalars(a, b);
        array_t idx {type_t{I32, 1}, {0}};
        return reduce_sum(operand.at(idx).max(c.broadcast_in_dim({1}, {})), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{3.0}, array_t{1.1}, array_t{1.0}}); // operand wins at index 0
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{2.0}}); // update wins at index 0
}

TEST(grad_tensor_fdm, scatter_min) {
    using namespace jax; using enum dtype_t;
    auto f = [](auto a, auto b, auto c) {
        auto operand = stack_scalars(a, b);
        array_t idx {type_t{I32, 1}, {0}};
        return reduce_sum(operand.at(idx).min(c.broadcast_in_dim({1}, {})), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{2.0}}); // operand wins at index 0
    check_grad_fdm(jaxpr, {array_t{3.0}, array_t{1.1}, array_t{1.0}}); // update wins at index 0
}

TEST(grad_tensor_fdm, scatter) {
    using namespace jax; using enum dtype_t;
    // Overwrite index 0 with the update -> [c, b]; the operand's index 0 gets no gradient.
    auto f = [](auto a, auto b, auto c) {
        auto operand = stack_scalars(a, b);
        array_t idx {type_t{I32, 1}, {0}};
        return reduce_sum(sin(operand.at(idx).set(c.broadcast_in_dim({1}, {}))), {0});
    };
    expression jaxpr = get_jaxpr(f, type_t{F32}, type_t{F32}, type_t{F32});
    check_grad_fdm(jaxpr, {array_t{0.5}, array_t{1.1}, array_t{-0.3}});
}