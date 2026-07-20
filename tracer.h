#ifndef TRACER_H
#define TRACER_H

#include <type_traits>
#include <utility>

#include "jax_types.h"

class jaxpr_tracer;
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

    [[nodiscard]] jaxpr_builder& get_builder() {
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

    template<typename... Fs, typename... Ts>
    [[nodiscard]] jaxpr_tracer switch_on(std::tuple<Fs...> branches, const Ts&... vals) const;

private:
    jax::var_t var;
    jaxpr_builder& builder;
};

class jaxpr_builder {
public:
    jaxpr_tracer register_tracer(jax::type_t type);
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
jaxpr_tracer jaxpr_tracer::switch_on(
        std::tuple<Fs...> branches,
        const Ts&... vals) const {
    using namespace jax;

    // propagate_branch will construct a jaxpr corresponding to each branch:
    auto propagate_branch = [&](const auto& f) -> expression {
        jaxpr_builder builder {};

        std::array<jaxpr_tracer, sizeof...(Ts)> tracer_inputs
            {builder.register_tracer(vals.get_type())...};

        builder.register_output(std::apply(f, tracer_inputs));
        return builder.get_jaxpr();
    };

    auto get_branch_expressions = [&](Fs... branches_) -> std::vector<expression> {
        return {propagate_branch(branches_)...};
    };

    std::vector<expression> branch_expressions = std::apply(get_branch_expressions, branches);


    if (branch_expressions.size() == 0) {
        throw std::logic_error("Error: expect at least one branch in switch expression");
    }

    auto& outvals = branch_expressions[0].outvals;

    for (size_t i = 1; i < branch_expressions.size(); i++) {
        // Outputs have to match:
        auto& branch_expression = branch_expressions[i];
        for (size_t j = 0; j < branch_expression.outvals.size(); j++) {
            if (outvals[j].get_type() != branch_expression.outvals[j].get_type()) {
                throw std::logic_error(
                    "Error: expect outputs to have consistent type signature in branches");
            }
        }
    }


    var_t output_var {builder.jaxpr.new_var_id(),
        branch_expressions[0].equations[0].get_output(0).get_type()};

    builder.jaxpr.equations.emplace_back(
        std::vector{promote(value{var}, dtype_t::I32, builder), value{vals.get_var()}...},
        std::vector{output_var},
        primitive_op::COND,
        cond_params{std::move(branch_expressions)}
    );

    return {builder, std::move(output_var)};
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

        std::array<type_t, output_count> output_types = std::invoke([&]<size_t... Is>(std::index_sequence<Is...>)
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

        std::array<jaxpr_tracer, output_count> output_tracers
        = std::invoke([&]<size_t... Is>(std::index_sequence<Is...>)
            -> std::array<jaxpr_tracer, output_count> {
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
}

#endif //TRACER_H
