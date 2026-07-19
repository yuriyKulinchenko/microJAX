#include <gtest/gtest.h>

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
        [](jaxpr_tracer x) -> jaxpr_tracer {
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

    EXPECT_EQ(invoke_vm(log_derivative_jaxpr, input), array_t{1}/input);
};

TEST(grad_closed_form, arithmetic) {
    using namespace jax;
    using enum dtype_t;
    // TODO: test for arithmetic operations
}

