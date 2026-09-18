#ifndef JAXPR_TYPES_H
#define JAXPR_TYPES_H

#include <string_view>
#include <variant>
#include <vector>

#include "helper.h"
#include "jax_array.h"

class jaxpr_tracer;
class jaxpr_tracer_index;

namespace jax {

#define PRIMITIVE_OP_LIST(X)                        \
    X(ADD) X(SUB) X(MUL) X(DIV)                     \
    X(SIN) X(COS) X(EXP) X(LOG)                     \
    X(NEG) X(TRANSPOSE) X(REDUCE_SUM)               \
    X(REDUCE_MAX) X(REDUCE_MIN)                     \
    X(DOT_GENERAL) X(BROADCAST_IN_DIM)              \
    X(CONVERT_ELEMENT_TYPE) X(COND) X(SCAN)         \
    X(SELECT) X(EQ) X(NE) X(LT) X(LE) X(GT) X(GE)   \
    X(GATHER) X(SCATTER_ADD) X(SCATTER_MUL)         \
    X(SCATTER_MAX) X(SCATTER_MIN) X(SCATTER)        \
    X(RESHAPE) X(SQRT) X(RSQRT) X(TANH) X(LOGISTIC) \
    X(MAX) X(MIN) X(INTEGER_POW) X(POW)             \
    X(CONCATENATE) X(SLICE) X(PAD)

    enum class primitive_op {
    #define X(name) name,
        PRIMITIVE_OP_LIST(X)
    #undef X
    };

    std::string_view to_string(primitive_op op);

    class var_t {
    public:
        explicit var_t(size_t id, type_t type);

        [[nodiscard]] const size_t& get_id() const;
        [[nodiscard]] size_t& get_id();

        void set_id(size_t new_id);

        [[nodiscard]] const type_t& get_type() const;
        [[nodiscard]] const std::vector<size_t>& get_shape() const;
        [[nodiscard]] dtype_t get_dtype() const;

        bool operator==(const var_t& other) const;

    private:
        type_t type;
        size_t id;
    };

    class value {
    public:
        explicit value(literal_t literal);
        explicit value(var_t var);

        template<typename T>
        [[nodiscard]] bool is() const {
            return std::holds_alternative<T>(variant_);
        }

        // Types are NOT returned by const&, as in the event that
        // the value is a literal, it does not have an underlying
        // type_t. It only has a dtype_t, so the whole type_t has
        // to be created from scratch.

        [[nodiscard]] const literal_t& get_literal() const;
        [[nodiscard]] const var_t& get_var() const;
        [[nodiscard]] var_t& get_var();
        [[nodiscard]] dtype_t get_dtype() const;
        [[nodiscard]] const std::vector<size_t>& get_shape() const;
        [[nodiscard]] type_t get_type() const;

        bool operator==(const value& other) const;

    private:
       std::variant<literal_t, var_t> variant_;
    };

    class equation;

    struct expression {
        std::vector<var_t> constvars;
        std::vector<var_t> invars;
        std::vector<value> outvals;
        std::vector<equation> equations;
        std::vector<array_t> consts;
        size_t var_id = 0;

        void add_constvar(var_t var);
        void add_invar(var_t var);
        void add_output(value val);
        equation& add_equation(equation eq);
        var_t fresh_var(type_t type);

        void eliminate_dead_code();
        void eliminate_common_subexpressions();

        size_t new_var_id();

        bool operator==(const expression& other) const;
    };

    struct transpose_params {
        std::vector<size_t> permutation;
        bool operator==(const transpose_params& other) const = default;
    };

    struct dot_general_params {
        std::vector<size_t> left_contract;
        std::vector<size_t> right_contract;
        std::vector<size_t> left_batch;
        std::vector<size_t> right_batch;
        bool operator==(const dot_general_params& other) const = default;
    };

    struct reduce_sum_params {
        std::vector<size_t> axes;
        bool operator==(const reduce_sum_params& other) const = default;
    };

    struct reduce_max_params {
        std::vector<size_t> axes;
        bool operator==(const reduce_max_params& other) const = default;
    };

    struct reduce_min_params {
        std::vector<size_t> axes;
        bool operator==(const reduce_min_params& other) const = default;
    };

    struct broadcast_in_dim_params {
        std::vector<size_t> shape;
        std::vector<size_t> broadcast_dimensions;
        bool operator==(const broadcast_in_dim_params& other) const = default;
    };

    struct convert_element_type_params {
        dtype_t new_dtype;
        bool operator==(const convert_element_type_params& other) const = default;
    };

    struct cond_params {
        std::vector<expression> branches;
        bool operator==(const cond_params& other) const = default;
    };

    struct scan_params {
        expression jaxpr;
        size_t length;
        size_t num_consts;
        size_t num_carry;
        bool reverse;
        bool operator==(const scan_params& other) const = default;
    };

    struct integer_pow_params {
        size_t y;
        bool operator==(const integer_pow_params& other) const = default;
    };

    struct concatenate_params {
        size_t dimension;
        bool operator==(const concatenate_params& other) const = default;
    };

    struct reshape_params {
        std::vector<size_t> new_sizes;
        bool operator==(const reshape_params& other) const = default;
    };

    struct slice_params {
        std::vector<size_t> start_indices;
        std::vector<size_t> limit_indices;
        std::vector<size_t> strides;
        bool operator==(const slice_params& other) const = default;
    };

    struct pad_params {
        std::vector<std::array<size_t, 3>> padding_config; // (low, high, interior) list
        bool operator==(const pad_params& other) const = default;
    };

    using params_variant = std::variant<
        std::monostate,
        transpose_params,
        dot_general_params,
        reduce_sum_params,
        reduce_max_params,
        reduce_min_params,
        broadcast_in_dim_params,
        convert_element_type_params,
        cond_params,
        scan_params,
        integer_pow_params,
        concatenate_params,
        reshape_params,
        slice_params,
        pad_params
    >;

    class equation {
    public:
        equation(std::vector<value> input, std::vector<var_t> output, primitive_op op);
        equation(std::vector<value> input, std::vector<var_t> output, primitive_op op, params_variant params);

        [[nodiscard]] const std::vector<value>& get_input() const;
        [[nodiscard]] const std::vector<var_t>& get_output() const;
        [[nodiscard]] const value& get_input(size_t i) const;
        [[nodiscard]] const var_t& get_output(size_t i) const;
        [[nodiscard]] primitive_op get_op() const;
        [[nodiscard]] const params_variant& get_params() const;

        [[nodiscard]] std::vector<value>& get_input();
        [[nodiscard]] std::vector<var_t>& get_output();
        [[nodiscard]] value& get_input(size_t i);
        [[nodiscard]] var_t& get_output(size_t i);
        [[nodiscard]] params_variant& get_params();

        bool operator==(const equation& other) const = default;

    private:
        std::vector<value> input;
        std::vector<var_t> output;
        primitive_op op;
        params_variant params;
    };
}

#define PARAMS_HASH_LIST(X) \
    X(transpose_params, p.permutation) \
    X(dot_general_params, p.left_contract, p.right_contract, p.left_batch, p.right_batch) \
    X(reduce_sum_params, p.axes) \
    X(reduce_max_params, p.axes) \
    X(reduce_min_params, p.axes) \
    X(broadcast_in_dim_params, p.shape, p.broadcast_dimensions) \
    X(convert_element_type_params, p.new_dtype) \
    X(cond_params, p.branches.size()) \
    X(scan_params, p.length, p.num_consts, p.num_carry, p.reverse) \
    X(integer_pow_params, p.y) \
    X(concatenate_params, p.dimension) \
    X(reshape_params, p.new_sizes) \
    X(slice_params, p.start_indices, p.limit_indices, p.strides) \
    X(pad_params, p.padding_config)

namespace std {
#define X(T, ...) template<> struct hash<jax::T> { size_t operator()(const jax::T&) const noexcept; };
    PARAMS_HASH_LIST(X)
#undef X
}

#endif //JAXPR_TYPES_H
