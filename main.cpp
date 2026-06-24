#include <iostream>

#include "jax_types.h"
#include "jax_functions.h"

template<typename T>
T test_function(T x) {
    return jax::sin(x) + jax::cos(x);
}


int main() {
    std::cout << "Function evaluated: " << test_function(0.5) << std::endl;
    return 0;
}
