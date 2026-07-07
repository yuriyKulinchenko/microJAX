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

    [[nodiscard]] jaxpr_tracer sin() const;
    [[nodiscard]] jaxpr_tracer cos() const;
    [[nodiscard]] jaxpr_tracer exp() const;

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
    void register_output(const jax::array_t& array);
    void register_output(const jax::var_t& var);


    template<std::convertible_to<size_t>... Args>
    jaxpr_tracer register_tracer(jax::dtype_t dtype, Args... shape) {
        return register_tracer(jax::type_t{dtype, shape...});
    }

    [[nodiscard]] jax::expression&& get_jaxpr();

    jax::expression jaxpr;
};

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

#endif //TRACER_H
