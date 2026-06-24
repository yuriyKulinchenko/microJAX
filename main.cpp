#include <iostream>

#include "jax_types.h"
#include "jax_functions.h"
#include "tagged_variant.h"

template<typename T>
T test_function(T x) {
    return jax::sin(x) + jax::cos(x);
}

int main() {
    test_fn();
    return 0;
}
