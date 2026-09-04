cmake_minimum_required(VERSION 3.31)
project(microJAX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(microJAX main.cpp
        jax_types.cpp
        jax_types.h
        tracer.h
        jax_functions.h
        tracer.cpp
        helper.h
        jax_logger.cpp
        jax_logger.h
        tagged_variant.h
        grad.h
        grad.cpp
        DCE_class.cpp
        DCE_class.h
        jax_format.h
        jax_vm.h
        jax_vm.cpp
        jax_array.cpp
        jax_array.h
)

add_executable(microJAX_tests gtest_test.cpp
        jax_types.cpp
        tracer.cpp
        jax_logger.cpp
        grad.cpp
        DCE_class.cpp
        jax_vm.cpp
        jax_array.cpp
)

target_link_libraries(microJAX_tests GTest::gtest_main)

enable_testing()
include(GoogleTest)
gtest_discover_tests(microJAX_tests)