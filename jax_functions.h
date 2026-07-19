#ifndef JAX_FUNCTIONS_H
#define JAX_FUNCTIONS_H

#include <tuple>
#include <vector>

#include "tracer.h"

inline jaxpr_tracer sin(const jaxpr_tracer& x) { return x.sin(); }
inline jaxpr_tracer cos(const jaxpr_tracer& x) { return x.cos(); }
inline jaxpr_tracer exp(const jaxpr_tracer& x) { return x.exp(); }
inline jaxpr_tracer log(const jaxpr_tracer& x) { return x.log(); }

inline jax::array_t sin(const jax::array_t& x) { return x.sin(); }
inline jax::array_t cos(const jax::array_t& x) { return x.cos(); }
inline jax::array_t exp(const jax::array_t& x) { return x.exp(); }
inline jax::array_t log(const jax::array_t& x) { return x.log(); }

namespace jax {

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

template<typename T>
T convert_element_type(T x, dtype_t new_dtype) {
    return x.convert_element_type(new_dtype);
}

template<typename T, typename... Us, typename... Fs>
T switch_on(T index, std::tuple<Fs...> branches, Us... values) {
    return index.switch_on(branches, values...);
}

}

#endif //JAX_FUNCTIONS_H
