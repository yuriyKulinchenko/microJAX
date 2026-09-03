#ifndef JAX_FUNCTIONS_H
#define JAX_FUNCTIONS_H

#include <tuple>
#include <vector>

#include "jax_logger.h"
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
    T reduce_max(T x, std::vector<size_t> axes) {
        return x.reduce_max(std::move(axes));
    }

    template<typename T>
    T reduce_min(T x, std::vector<size_t> axes) {
        return x.reduce_min(std::move(axes));
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
    T switch_on(T index, std::tuple<Fs...> branches, const Us&... values) {
        return index.switch_on(branches, values...);
    }

    /*

    What should the interface of switch_on be?
    switch_on(tracer_t, {F1, F2, F3}, tracer_t, tracer_t) -> tracer_t | tuple<tracer_t, ...>

    switch_on takes as input a combination of tracers and array_t instances.
    It returns the result of applying the sequence of inputs to F1, F2 or F3
    These should all match!

    What if index is an array_t, but 'values' contains a tracer?
    */



    template<typename... Ts>
    struct tracer_or_array_tuple_struct {
        static constexpr bool value =
        ((std::convertible_to<Ts, jaxpr_tracer>
            || std::convertible_to<Ts, array_t>) && ...);
    };

    template<typename... Ts>
    concept tracer_or_array_tuple = tracer_or_array_tuple_struct<Ts...>::value;

    // This exists for finding the tracer element

    template<typename Tuple, typename T, size_t i = 0>
    struct tuple_find_type_instance;

    template<typename T, size_t i>
    struct tuple_find_type_instance<std::tuple<>, T, i> {
        static constexpr bool exists = false;
        static constexpr size_t index = i;
    };

    template<typename T, size_t i, typename U, typename... Us>
    struct tuple_find_type_instance<std::tuple<U, Us...>, T, i> {
        static constexpr bool exists = std::convertible_to<T, U>
        || tuple_find_type_instance<std::tuple<Us...>, T, i + 1>::exists;

        static constexpr size_t index = std::convertible_to<T, U> ? i
        : tuple_find_type_instance<std::tuple<Us...>, T, i + 1>::index;
    };

    template<typename... Ts>
    struct first_type_struct;

    template<>
    struct first_type_struct<> {
        using type = array_t;
    };

    template<typename T, typename... Ts>
    struct first_type_struct<T, Ts...> {
        using type = T;
    };

    template<typename... Ts>
    using first_type = first_type_struct<Ts...>::type;

    template<typename... Ts>
    std::array<first_type<Ts...>, sizeof...(Ts)> tuple_to_array(std::tuple<Ts...> tuple) {

        auto populate_array = [&]<size_t... Is>(std::index_sequence<Is...>)
        -> std::array<first_type<array_t, Ts...>, sizeof...(Ts)>{
            return {std::move(std::get<Is>(tuple))...};
        };

        return populate_array(std::make_index_sequence<sizeof...(Ts)>());
    }

    template<typename F, typename... Consts, typename... Carry, typename... Xs>
    requires tracer_or_array_tuple<Consts...>
    && tracer_or_array_tuple<Carry...>
    && tracer_or_array_tuple<Xs...>
    auto scan(
        F f, std::tuple<Consts...> consts, std::tuple<Carry...> carry,
        std::tuple<Xs...> xs, size_t L, bool reverse=false) {
        // Find the tracer element, if it exists:

        using consts_find_tracer = tuple_find_type_instance<std::tuple<Consts...>, jaxpr_tracer>;
        using carry_find_tracer = tuple_find_type_instance<std::tuple<Carry...>, jaxpr_tracer>;
        using xs_find_tracer = tuple_find_type_instance<std::tuple<Xs...>, jaxpr_tracer>;


        if constexpr (consts_find_tracer::exists || carry_find_tracer::exists || xs_find_tracer::exists) {
            // Case 1: if there is a jaxpr_tracer somewhere, delagate to tracer_scan
            auto get_builder = [&]() -> jaxpr_builder& {
                if constexpr(consts_find_tracer::exists) {
                    return std::get<consts_find_tracer::index>(consts).get_builder();
                } else if constexpr (carry_find_tracer::exists) {
                    return std::get<carry_find_tracer::index>(carry).get_builder();
                } else {
                    return std::get<xs_find_tracer::index>(xs).get_builder();
                }
            };
            return tracer_scan(get_builder(), std::move(f),
                std::move(consts), std::move(carry), std::move(xs), L, reverse);
        } else {
            // Otherwise, convert to array
            return array_scan(std::move(f), tuple_to_array(std::move(consts)), tuple_to_array(std::move(carry)),
                tuple_to_array(std::move(xs)), L, reverse);
        }
    }

    template<typename Pred, typename... Ts>
    auto select(const Pred& pred, const Ts&... values) {
        // 'pred' and 'values' are instances of array_t or jaxpr_tracer
        using values_find_tracer = tuple_find_type_instance<std::tuple<Ts...>, jaxpr_tracer>;

        if constexpr(std::convertible_to<Pred, jaxpr_tracer>) {
            // Tracer belongs to 'pred'
            auto& builder = pred.get_builder();
            return tracer_select(builder, pred, values...);
        } else if constexpr(values_find_tracer::exists) {
            // Tracer exists somewhere inside 'values'
            constexpr size_t I = values_find_tracer::index;
            auto& tracer = std::get<I>(std::forward_as_tuple(std::forward<Ts>(values)...));
            auto& builder = tracer.get_builder();
            return tracer_select(builder, pred, values...);
        } else {
            return array_select(pred, values...);
        }
    }
}



#endif //JAX_FUNCTIONS_H
