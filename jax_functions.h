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

inline bool valid_permutation(const std::vector<size_t>& permutation) {
    // A permutation of N elements is valid if it is a permutation of the sequence (0, ..., N-1)
    auto exists = std::vector(permutation.size(), false);
    for (auto x: permutation) {
        if (x >= permutation.size()) return false;
        if (exists[x]) return false;
        exists[x] = true;
    }
    return true;
}

template<typename T>
T transpose(T x, std::vector<size_t> permutation) {
    return x.transpose(permutation);
}

}

#endif //JAX_FUNCTIONS_H
