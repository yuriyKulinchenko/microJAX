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
    std::array<first_type<Ts...>, sizeof...(Ts)> array_to_tuple(std::tuple<Ts...> tuple) {

        auto populate_array = [&]<size_t... Is>(std::index_sequence<Is...>)
        -> std::array<first_type<Ts...>, sizeof...(Ts)>{
            return {std::move(std::get<Is>(tuple))...};
        };

        return populate_array(std::make_index_sequence<sizeof...(Ts)>());
    }

    // tracer implementation:
    // TODO: move back to correct location

    inline value array_value(const array_t& array, jaxpr_builder& builder) {
        if (const std::optional<literal_t> literal = array.get_literal()) {
            return value{*literal};
        }


        // Otherwise, add it to the array of consts:
        builder.jaxpr.consts.push_back(array);
        var_t fresh_var = builder.jaxpr.fresh_var(array.get_type());
        builder.jaxpr.constvars.push_back(fresh_var);
        return value{std::move(fresh_var)};
    }

    template<size_t N>
    using value_array = std::array<array_t, N>;


    template<typename F, size_t num_consts, size_t num_carry, size_t num_xs>
    auto array_scan(F f,
        value_array<num_consts> consts, value_array<num_carry> carry,
        value_array<num_xs> xs, size_t L, bool reverse) {
        // f must be able to take array_t parameters
        // (consts..., ys[:1]...) = f(consts..., carry..., xs[:1]...)

        // Expect that output_types_t is a tuple of array_t
        constexpr size_t num_inputs = num_consts + num_carry + num_xs;
        using output_types_t = apply_result_t<F, value_array<num_inputs>>;
        constexpr size_t num_outputs = std::tuple_size_v<output_types_t>;
        constexpr size_t num_ys = num_outputs - num_carry;

        // Expected dimension of ys is given by output to f.
        // Important: this consumes carry.
        // input_at_index takes an i parameter, ranging over
        // (consts..., carry..., xs...),
        // and a t parameter which xs[t]...

        auto input_at_index = [&]<size_t i>(size_t t) -> array_t {
            if constexpr (i < num_consts) {
                return consts[i];
            } else if constexpr(i < num_consts + num_carry) {
                return std::move(carry[i - num_consts]);
            } else {
                return xs[i - num_consts - num_carry].slice({t});
            }
        };

        auto project_input = [&](size_t t) -> value_array<num_outputs> {
            // Return tuple containing (carry..., ys[t])
            value_array<num_inputs> input = std::invoke(
                [&]<size_t... Is>(std::index_sequence<Is...>) -> value_array<num_inputs> {
                return {input_at_index.template operator()<Is>(t)...};
            }, std::make_index_sequence<num_inputs>());

            // Next goal: pass the input into f:
            auto output = std::apply(f, input);
            // Expect output to be tuple-like:

            return std::invoke(
                [&]<size_t... Is>(std::index_sequence<Is...>)-> value_array<num_outputs> {
                return {std::get<Is>(output)...};
            }, std::make_index_sequence<num_outputs>());
        };

        // TODO: Handle reverse

        value_array<num_ys> ys {};

        for (size_t t = 0; t < L; t++) {
            value_array<num_outputs> output = project_input(t);
            // Update carry:
            for (size_t j = 0; j < num_carry; j++) {
                carry[j] = output[j];
            }

            // Update correct slice of ys:

            for (size_t j = 0; j < num_ys; j++) {
                array_t& output_instance = output[num_carry + j];
                array_t& y_instance = ys[j];

                if (t == 0) {
                    std::vector<size_t>& shape = y_instance.get_type().get_shape();
                    shape = {L};
                    for (size_t x: output_instance.get_type().get_shape()) shape.push_back(x);
                    y_instance.get_value().clear();
                }

                std::vector<double>& value {y_instance.get_value()};

                value.append_range(output_instance.get_value());
            }
        }

        for (array_t& array: ys) array.compute_strides();

        // Return final value:

        auto output_index = [&]<size_t i>() -> array_t {
            if constexpr (i < num_carry) {
                return std::move(carry[i]);
            } else {
                return std::move(ys[i - num_carry]);
            }
        };

        using output_tuple_t = array_to_tuple_t<value_array<num_outputs>>;
        return std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) -> output_tuple_t {
            return {(output_index.template operator()<Is>())...};
        }, std::make_index_sequence<num_outputs>());
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
            return array_scan(std::move(f), array_to_tuple(std::move(consts)), array_to_tuple(std::move(carry)),
                array_to_tuple(std::move(xs)), L, reverse);
        }

        // Otherwise, convert everything to a simplified array representation:

    }
}



#endif //JAX_FUNCTIONS_H
