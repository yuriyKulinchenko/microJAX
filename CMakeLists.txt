cmake_minimum_required(VERSION 3.31)
project(microJAX)

set(CMAKE_CXX_STANDARD 26)

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
        grad.cpp)
