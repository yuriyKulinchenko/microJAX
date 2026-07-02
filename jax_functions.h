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

template<typename T> requires(std::is_scalar_v<T>)
T exp(T x) {
    return std::exp(x);
}

template<typename T> requires(!std::is_scalar_v<T>)
T exp(T x) {
    return x.exp();
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
T dot_general(T x, T y,
    std::vector<size_t> left_contract, std::vector<size_t> right_contract,
    std::vector<size_t> left_batch, std::vector<size_t> right_batch) {
    return x.dot_general(y, std::move(left_contract), std::move(right_contract),
        std::move(left_batch), std::move(right_batch));
}

template<typename T>
T dot_general(T x, T y,
        std::pair<std::vector<size_t>, std::vector<size_t>> contract,
        std::pair<std::vector<size_t>, std::vector<size_t>> batch) {
    return x.dot_general(y, std::move(contract.first), std::move(contract.second),
        std::move(batch.first), std::move(batch.second));
}

template<typename T>
T broadcast_in_dim(T x, std::vector<size_t> shape, std::vector<size_t> broadcast_dimensions) {
    return x.broadcast_in_dim(std::move(shape), std::move(broadcast_dimensions));
}

}

#endif //JAX_FUNCTIONS_H
