cmake_minimum_required(VERSION 3.31)
project(microJAX)

set(CMAKE_CXX_STANDARD 23)

add_executable(microJAX main.cpp
        jaxpr_types.cpp
        jaxpr_types.h)
