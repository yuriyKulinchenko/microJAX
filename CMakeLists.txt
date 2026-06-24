cmake_minimum_required(VERSION 3.31)
project(microJAX)

set(CMAKE_CXX_STANDARD 23)

add_executable(microJAX main.cpp
        jaxpr_types.cpp
        jax_types.h
        tracer.h
        jax_functions.h
        tracer.cpp
        helper.h)
