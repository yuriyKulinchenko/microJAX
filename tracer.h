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
    // TODO: register_output should be able to handle tuples
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

template<size_t L>
using jaxpr_array = std::array<jaxpr_tracer, L>;

template<size_t num_carry, size_t num_xs, typename F>
auto scan(
    jaxpr_builder& builder,
    F f,
    jaxpr_array<num_carry> carry,
    jaxpr_array<num_xs> xs,
    size_t L, bool reverse=false) {

    jaxpr_builder scan_builder {};

    auto project = [L](const jax::type_t& type) -> jax::type_t {
        auto& old_shape = type.get_shape();
        if (old_shape.size() == 0) {
            throw std::logic_error("Error: 'xs' in scan cannot be a literal, it must be a tensor");
        }
        if (old_shape[0] != L) {
            throw formatted_error(
                "Error: expected leading rank size to be {}, instead got {} in 'xs'",
                L, old_shape[0]
                );
        }

        std::vector<size_t> new_shape
        {old_shape.begin() + 1, old_shape.end()};
        return jax::type_t{type.get_dtype(), std::move(new_shape)};
    };

    auto project_inverse = [L](const jax::type_t& type) -> jax::type_t {
        std::vector new_shape {L};
        new_shape.reserve(type.get_shape().size() + 1);
        for (size_t x: type.get_shape()) new_shape.push_back(x);
        return jax::type_t{type.get_dtype(), std::move(new_shape)};
    };

    // Build each input tracer directly via register_tracer, selecting the source
    // with if constexpr so there is no ternary common-type materialization.
    auto make_input_tracer = [&]<size_t I>() -> jaxpr_tracer {
        if constexpr (I < num_carry) {
            return scan_builder.register_tracer(carry[I].get_type());
        } else {
            return scan_builder.register_tracer(project(xs[I - num_carry].get_type()));
        }
    };

    jaxpr_array<num_carry + num_xs> input_tracers = std::invoke(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return jaxpr_array<num_carry + num_xs> {
                make_input_tracer.template operator()<Is>()...
            };
    }, std::make_index_sequence<num_carry + num_xs>{});

    auto output_tracers = std::apply(f, input_tracers);

    constexpr size_t num_total_out = std::tuple_size_v<decltype(output_tracers)>;
    static_assert(num_total_out >= num_carry,
        "scan body must return at least num_carry outputs (the updated carries)");
    constexpr size_t num_ys = num_total_out - num_carry;

    for (auto& output_tracer: output_tracers) {
        scan_builder.register_output(output_tracer);
    }

    std::vector<jax::var_t> output {};
    output.reserve(num_carry + num_ys);

    for (size_t i = 0; i < num_carry; i++) {
        output.push_back(builder.jaxpr.fresh_var(output_tracers[i].get_type()));
    }

    for (size_t i = 0; i < num_ys; i++) {
        output.push_back(builder.jaxpr.fresh_var(
            project_inverse(output_tracers[num_carry + i].get_type())));
    }

    std::vector<jax::value> input {};
    input.reserve(num_carry + num_xs);

    for (auto& tracer: carry) {
        input.push_back(jax::value{tracer.get_var()});
    }

    for (auto& tracer: xs) {
       input.push_back(jax::value{tracer.get_var()});
    }

    jaxpr_array<num_carry + num_ys> global_output_tracers = std::invoke(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return jaxpr_array<num_carry + num_ys> {
                    jaxpr_tracer{builder, output[Is]}...
            };
    }, std::make_index_sequence<num_carry + num_ys>{});

    builder.jaxpr.equations.emplace_back(
        std::move(input), std::move(output),
        jax::primitive_op::SCAN,
        jax::scan_params{
            .jaxpr = scan_builder.get_jaxpr(),
            .length = L,
            .num_carry = num_carry,
            .reverse = reverse}
    );

    return global_output_tracers;
}
#endif //TRACER_H
