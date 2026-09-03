#ifndef TRACER_H
#define TRACER_H

#include <type_traits>
#include <utility>

#include "jax_types.h"

class jaxpr_builder;

class jaxpr_tracer {
public:
    jaxpr_tracer(jaxpr_builder& builder, jax::var_t var):
    var(std::move(var)),
    builder(builder) {}

    [[nodiscard]] const jax::type_t& get_type() const {
        return var.get_type();
    }

    [[nodiscard]] const jax::var_t& get_var() const {
        return var;
    }

    [[nodiscard]] jaxpr_builder& get_builder() const {
        return builder;
    }

#define BINARY_OP(op)                                                               \
    friend jaxpr_tracer operator op (const jaxpr_tracer&, const jaxpr_tracer&);     \
    friend jaxpr_tracer operator op (const jaxpr_tracer&, const jax::array_t&);     \
    friend jaxpr_tracer operator op (const jax::array_t&, const jaxpr_tracer&);

    BINARY_OP(+); BINARY_OP(-);
    BINARY_OP(*); BINARY_OP(/);
    BINARY_OP(==); BINARY_OP(!=);
    BINARY_OP(<); BINARY_OP(<=);
    BINARY_OP(>); BINARY_OP(>=);

#undef BINARY_OP

    [[nodiscard]] jaxpr_tracer sin() const;
    [[nodiscard]] jaxpr_tracer cos() const;
    [[nodiscard]] jaxpr_tracer exp() const;
    [[nodiscard]] jaxpr_tracer log() const;
    [[nodiscard]] jaxpr_tracer operator-() const;

    [[nodiscard]] jaxpr_tracer transpose(std::vector<size_t> permutation) const;
    [[nodiscard]] jaxpr_tracer reduce_sum(std::vector<size_t> axes) const;
    [[nodiscard]] jaxpr_tracer convert_element_type(jax::dtype_t dtype) const;

    [[nodiscard]] jaxpr_tracer broadcast_in_dim(
        std::vector<size_t> shape,
        std::vector<size_t> broadcast_dimensions) const;

    [[nodiscard]] jaxpr_tracer dot_general(
        const jaxpr_tracer& other,
        std::vector<size_t> left_contract,
        std::vector<size_t> right_contract,
        std::vector<size_t> left_batch,
        std::vector<size_t> right_batch) const;

    [[nodiscard]] jaxpr_tracer_index at(jaxpr_tracer idx);
    [[nodiscard]] jaxpr_tracer_index at(jax::array_t idx);

    template<typename... Fs, typename... Ts>
    [[nodiscard]] auto switch_on(std::tuple<Fs...> branches, const Ts&... vals) const;

private:
    jax::var_t var;
    jaxpr_builder& builder;
};

class jaxpr_builder {
public:
    jaxpr_tracer register_tracer(jax::type_t type);
    jaxpr_tracer register_tracer(const jax::array_t& array);
    void register_output(const jaxpr_tracer& tracer);
    void register_output(const jax::value& value);
    void register_output(const jax::literal_t& literal);
    void register_output(const jax::var_t& var);
    void register_output(const jax::array_t& array);

    template<typename TupleContainer>
    void register_output(const TupleContainer& container) {
        constexpr size_t tuple_size = std::tuple_size_v<TupleContainer>;
        std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) {
            ((register_output(std::get<Is>(container))), ...);
        }, std::make_index_sequence<tuple_size>());
    }


    template<std::convertible_to<size_t>... Args>
    jaxpr_tracer register_tracer(jax::dtype_t dtype, Args... shape) {
        return register_tracer(jax::type_t{dtype, shape...});
    }

    [[nodiscard]] jax::expression&& get_jaxpr();
    jax::expression jaxpr;
};

/*

The class jaxpr_tracer_index is a proxy object required when performing scatter-reduce like operations:
scatter_add, scatter_mul, etc.

For a tracer 't' of type jaxpr_tracer, t.at(idx).op(u) produces a jaxpr_tracer_index
on the invocation of the 'at(idx)' method, and a jaxpr_tracer on the invocation of the 'op(u)'
method. jaxpr_tracer_index holds a reference to the underlying tensor object,

*/

class jaxpr_tracer_index {
public:
    jaxpr_tracer_index(jaxpr_tracer x, jaxpr_tracer idx);
    jaxpr_tracer_index(jaxpr_tracer x, jax::array_t idx);

    jaxpr_tracer get();

    // Overwrite:
    jaxpr_tracer set(const jaxpr_tracer& update);
    jaxpr_tracer set(const jax::array_t& update);

    jaxpr_tracer add(const jaxpr_tracer& update);
    jaxpr_tracer add(const jax::array_t& update);

    jaxpr_tracer multiply(const jaxpr_tracer& update);
    jaxpr_tracer multiply(const jax::array_t& update);

    jaxpr_tracer max(const jaxpr_tracer& update);
    jaxpr_tracer max(const jax::array_t& update);

    jaxpr_tracer min(const jaxpr_tracer& update);
    jaxpr_tracer min(const jax::array_t& update);
private:

    jax::value get_idx_value();
    jaxpr_tracer op(jax::value update, jax::primitive_op scatter_op);

    jaxpr_tracer x;
    std::variant<jaxpr_tracer, jax::array_t> idx;
};

namespace jax {
    class array_tracer_index {
    public:
        array_tracer_index(array_t x, jaxpr_tracer idx);

        jaxpr_tracer get();

        jaxpr_tracer set(const jaxpr_tracer& update);
        jaxpr_tracer set(const array_t& update);

        jaxpr_tracer add(const jaxpr_tracer& update);
        jaxpr_tracer add(const array_t& update);

        jaxpr_tracer multiply(const jaxpr_tracer& update);
        jaxpr_tracer multiply(const array_t& update);

        jaxpr_tracer max(const jaxpr_tracer& update);
        jaxpr_tracer max(const array_t& update);

        jaxpr_tracer min(const jaxpr_tracer& update);
        jaxpr_tracer min(const array_t& update);

    private:
        jaxpr_tracer op(value update, primitive_op scatter_op);

        array_t x;
        jaxpr_tracer idx;
    };
}

jax::value array_value(const jax::array_t& array, jaxpr_builder& builder);

template<std::convertible_to<jax::type_t>... Types, typename F>
jax::expression get_jaxpr(F&& f, Types&&... types) {
    jaxpr_builder builder {};
    builder.register_output(f(builder.register_tracer(types)...));
    auto jaxpr = builder.get_jaxpr();
    jaxpr.eliminate_dead_code();
    return jaxpr;
}

jax::value promote(const jax::value& val, jax::dtype_t dtype, jaxpr_builder& builder);

template<typename... Fs, typename... Ts>
auto jaxpr_tracer::switch_on(std::tuple<Fs...> branches, const Ts&... vals) const {
    using namespace jax;
    static_assert(sizeof...(Fs) > 0, "There must be a branch provided to the switch statement");

    // What are the inputs? They are the index parameter followed by the provided vals.
    // The provided vals are tracers or array_t instances.
    // The outputs have a shape derived from something
    // The jaxpr of each function in 'branches' is recursively constructed
    // This is done with create_jaxpr

    // type(vals) = jaxpr_tracer | array_t

    auto transform_to_value = []<typename T>(const T& val) -> value {
        if constexpr(std::convertible_to<T, jaxpr_tracer>) {
            return value{val.var};
        } else if constexpr(std::convertible_to<T, array_t>) {
            // TODO: incorrect - array_value requires a builder argument; rewrite this.
            return array_value(val);
        }
        std::unreachable();
    };

    auto transform_to_type = []<typename T>(const T& val) -> type_t {
        return val.get_type();
    };

    std::array<value, sizeof...(vals)> non_index_inputs = {transform_to_value(vals)...};
    std::array<type_t, sizeof...(vals)> input_types = {transform_to_type(vals)...};

    // I need to pass the array of input types to each of the branches:

    auto apply_to_branch = [&]<typename F>(F&& f) -> expression {
        return std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) -> expression {
            return get_jaxpr(f, std::get<Is>(input_types)...);
        }, std::make_index_sequence<sizeof...(vals)>());
    };

    // For each element of branches, get its jaxpr, and put it in a list:

    std::vector<expression> expression_branches
    = std::apply([&](auto&&... fs) -> std::vector<expression> {
        return std::vector{apply_to_branch(fs)...};
    }, branches);

    // No matter what, switch_on will ALWAYS assume the output is a jaxpr_tracer if there exists even
    // one input that is a jaxpr_tracer. Here, this is trivially the case, as the index is a jaxpr_tracer

    using F1 = first_type<nullptr_t, Fs...>;
    static_assert(!std::same_as<F1, nullptr_t>);
    using output_type = std::invoke_result_t<F1, const Ts&...>;

    // There are 3 options: output_type_t is an array_t, jaxpr_tracer, or std::tuple<...>


    std::vector inputs {value{var}}; // First input is the index itself
    inputs.append_range(non_index_inputs);

    // Outputs are interesting
    // They have to be created with builder.jaxpr.fresh_var()
    // However, the types have to be coherent
    // The types are fetched from the output of expression_branches

    // TODO: For now, I am assuming that all passed branches have the same output
    // Obviously, this might not be the case

    std::vector<var_t> outputs = expression_branches[0].outvals
    | std::views::transform([&](const value& val) {
        return builder.jaxpr.fresh_var(val.get_type());
    }) | std::ranges::to<std::vector<var_t>>();

    builder.jaxpr.equations.emplace_back(
        std::move(inputs), std::move(outputs),
        primitive_op::COND, cond_params{std::move(expression_branches)});

    std::vector<var_t>& new_outputs = builder.jaxpr.equations[builder.jaxpr.equations.size() - 1].get_output();

    if constexpr(std::convertible_to<output_type, array_t> || std::convertible_to<output_type, jaxpr_tracer>) {
        // Simply return the single output.
        return jaxpr_tracer{builder, new_outputs[0]};
    } else {
        // Return multiple outputs.
        constexpr size_t N = std::tuple_size_v<output_type>;
        using output_tuple_t = array_to_tuple_t<std::array<jaxpr_tracer, N>>;
        return std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) {
            return output_tuple_t {jaxpr_tracer{builder, new_outputs[Is]}...};
        }, std::make_index_sequence<N>());
    }
}

namespace jax {
    template<typename F, typename... Consts, typename... Carry, typename...Xs>
    auto tracer_scan(jaxpr_builder& builder, F f, std::tuple<Consts...> consts,
        std::tuple<Carry...> carry, std::tuple<Xs...> xs, size_t L, bool reverse) {

        // elements of xs, ys are stacked
        // consts, carry pass through

        // f(carry[i]..., xs[i]...) -> carry, ys[i]...
        // f : (consts..., carry..., xs[1:]...) -> (carry..., ys[1:]...)
        // scan(f) : (consts..., carry..., xs...) -> (carry..., ys...)


        std::vector<value> input {};
        std::vector<var_t> output {};

        auto get_value = [&]<typename T>(const T& x) -> value {
            // If tuple[i] is an array, call array_value
            // if tuple[i] is a tracer, wrap with a value
            if constexpr(std::convertible_to<T, array_t>) {
                return array_value(x, builder);
            } else {
                // jaxpr_tracer
                return value{x.get_var()};
            }
        };

        // Populate consts, carry, xs:

        constexpr size_t input_count = sizeof...(Consts) + sizeof...(Carry) + sizeof...(Xs);
        input.reserve(input_count);

        std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) {
            ((input.push_back(get_value(std::get<Is>(consts)))), ...);

        }, std::make_index_sequence<sizeof...(Consts)>());

        std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) {
            ((input.push_back(get_value(std::get<Is>(carry)))), ...);

        }, std::make_index_sequence<sizeof...(Carry)>());

        std::invoke([&]<size_t... Is>(std::index_sequence<Is...>) {
            ((input.push_back(get_value(std::get<Is>(xs)))), ...);

        }, std::make_index_sequence<sizeof...(Xs)>());

        // Here, I need to fetch and reduce the types. This will be done with a tuple:

        constexpr size_t carry_offset = sizeof...(Consts);
        constexpr size_t xs_offset = carry_offset + sizeof...(Carry);

        auto unstack = [](const type_t& type) -> type_t {
            return type_t{type.get_dtype(),
                std::vector<size_t>{type.get_shape().begin() + 1, type.get_shape().end()}};
        };

        auto stack = [L](const type_t& type) -> type_t {
            std::vector new_shape {L};
            new_shape.reserve(type.get_shape().size() + 1);
            for (size_t x: type.get_shape()) new_shape.push_back(x);
            return type_t{type.get_dtype(), std::move(new_shape)};
        };

        auto get_input_type = [&]<size_t i>() -> type_t {
            if constexpr(i < carry_offset) {
                return std::get<i>(consts).get_type();
            } else if constexpr(i < xs_offset) {
                return std::get<i - carry_offset>(carry).get_type();
            } else {
                return unstack(std::get<i - xs_offset>(xs).get_type());
            }
        };

        auto input_types = std::invoke([&]<size_t... Is>(std::index_sequence<Is...>)
            -> std::array<type_t, input_count> {
            return {get_input_type.template operator()<Is>()...};
        }, std::make_index_sequence<input_count>());

        using output_types_t = apply_result_t<F, std::array<jaxpr_tracer, input_count>>;

        constexpr size_t output_count = std::tuple_size_v<output_types_t>;

        // The final outputs of inner_jaxpr inform the output type signature:
        // f : (consts..., carry..., xs[1:]...) -> (carry..., ys[1:]...)
        // scan(f) : (consts..., carry..., xs...) -> (carry..., ys...)

        expression inner_jaxpr = std::apply([&](auto&... types) {
            return get_jaxpr(f, types...);
        }, input_types);

        std::array<type_t, output_count> output_types
        = std::invoke([&]<size_t... Is>(std::index_sequence<Is...>)
            -> std::array<type_t, output_count> {
            return {(
                Is < sizeof...(Carry) ? inner_jaxpr.outvals[Is].get_type():
                stack(inner_jaxpr.outvals[Is].get_type())
            )...};
        }, std::make_index_sequence<output_count>());

        // Populate the output:

        output.reserve(output_count);

        for (const type_t& type: output_types) {
            output.emplace_back(std::move(builder.jaxpr.fresh_var(type)));
        }

        using output_tuple_t = array_to_tuple_t<std::array<jaxpr_tracer, output_count>>;

        output_tuple_t output_tracers
        = std::invoke([&]<size_t... Is>(std::index_sequence<Is...>)
            -> output_tuple_t {
            return {(jaxpr_tracer{builder, output[Is]})...};
        }, std::make_index_sequence<output_count>());

         builder.jaxpr.equations.emplace_back(
         std::move(input), std::move(output),
         primitive_op::SCAN,
         scan_params{
             .jaxpr = std::move(inner_jaxpr),
             .length = L,
             .num_consts = sizeof...(Consts),
             .num_carry = sizeof...(Carry),
             .reverse = reverse}
        );

        return output_tracers;
    }

    template<typename Pred, typename... Values>
    jaxpr_tracer tracer_select(jaxpr_builder& builder, const Pred& pred, const Values&... values) {
        // Convert everything into an array of tracers:

        auto to_value = [&builder]<typename T>(const T& x) -> value {
            if constexpr(std::convertible_to<T, array_t>) {
                return array_value(x, builder);
            } else {
                // jaxpr_tracer
                return value{x.get_var()};
            }
        };

        std::vector<value> processed_values = {to_value(values)...};

        // The predicate is the switching operand: like COND, it is an integer
        // index. A boolean predicate is auto-converted to I32.
        value predicate = promote(to_value(pred), dtype_t::I32, builder);

        std::vector<value> inputs {predicate};
        inputs.reserve(processed_values.size() + 1);
        inputs.append_range(processed_values);

        // Get input variables, the single output variable:

        var_t output = builder.jaxpr.fresh_var(processed_values[0].get_type());

        builder.jaxpr.equations.emplace_back(
            std::move(inputs),
            std::vector{output},
            primitive_op::SELECT);

        return jaxpr_tracer{builder, output};
    }
}

#endif //TRACER_H
