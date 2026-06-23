#ifndef JAX_FUNCTIONS_H
#define JAX_FUNCTIONS_H
#include <type_traits>
#include <valarray>

namespace jax {

template<typename T> requires(std::is_scalar_v<T>)
T sin(T x) {
    return std::sin(x);
}

template<typename T> requires(!std::is_scalar_v<T>)
T sin(T x) {
    return x.sin();
}

template<typename T> requires(std::is_scalar_v<T>)
T cos(T x) {
    return std::cos(x);
}

template<typename T> requires(!std::is_scalar_v<T>)
T cos(T x) {
    return x.cos();
}

}

#endif //JAX_FUNCTIONS_H
