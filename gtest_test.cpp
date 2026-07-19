#include <gtest/gtest.h>

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

    jaxpr_builder builder {};
    auto x = builder.register_tracer(F32, 3, 3);
    auto y = builder.register_tracer(F32, 3, 3);
    builder.register_output(matmul(x, y));
    auto jaxpr = builder.get_jaxpr();

    jax_vm vm{jaxpr};

    EXPECT_EQ(matmul(A, B), expected_output);
    EXPECT_EQ(vm.run({A, B})[0], expected_output);
}

TEST(vm, matmul_batch) {
    using namespace jax;
    using enum dtype_t;
    array_t A = array_t{type_t{F32, 2, 3, 3}, {
        5, 8, 9,
        1, 6, 7,
        4, 5, 3,

        4, 8, 5,
        5, 1, 1,
        9, 6, 4
    }};

    array_t B = array_t{type_t{F32, 2, 3, 3}, {
        4, 8, 5,
        5, 1, 1,
        9, 6, 4,

        5, 8, 9,
        1, 6, 7,
        4, 5, 3,
    }};

    array_t expected_output = array_t{type_t{F32, 2, 3, 3}, {
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

    jaxpr_builder builder {};
    auto x = builder.register_tracer(F32, 3, 3);
    auto y = builder.register_tracer(F32, 3, 3);
    builder.register_output(matmul_batch(x, y));
    auto jaxpr = builder.get_jaxpr();

    jax_vm vm{jaxpr};

    EXPECT_EQ(matmul_batch(A, B), expected_output);
    EXPECT_EQ(vm.run({A, B})[0], expected_output);
}