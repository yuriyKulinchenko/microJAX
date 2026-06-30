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
    return x.transpose(std::move(permutation));
}

template<typename T>
T reduce_sum(T x, std::vector<size_t> axes) {
    return x.reduce_sum(std::move(axes));
}

template<typename T>
T dot_general(T x,
    std::vector<size_t> left_contract, std::vector<size_t> right_contract,
    std::vector<size_t> left_batch, std::vector<size_t> right_batch) {
    return x.dot_general(std::move(left_contract), std::move(right_contract),
        std::move(left_batch), std::move(right_batch));
}

template<typename T>
T broadcast_in_dim(T x, std::vector<size_t> shape, std::vector<size_t> broadcast_dimensions) {
    return x.broadcast_in_dim(std::move(shape), std::move(broadcast_dimensions));
}

}

#endif //JAX_FUNCTIONS_H
