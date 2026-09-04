#ifndef JAXPR_TYPES_H
#define JAXPR_TYPES_H
#include <any>
#include <functional>
#include <string_view>
#include <variant>
#include <vector>
#include <span>

#include "helper.h"
#include "jax_array.h"

/*
TODO:
REDUCE_MAX, REDUCE_MIN // Done
GATHER // Done
SCATTER_ADD, SCATTER_MUL, SCATTER_MAX, SCATTER // Done
RESHAPE // Done
SQRT, RSQRT
TANH
LOGISTIC
MAX, MIN
INTEGER_POW
POW
CONCATENATE // Done
SLICE
PAD
*/

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

        [[nodiscard]] size_t get_id() const;

        void set_id(size_t new_id);

        [[nodiscard]] const type_t& get_type() const;
        [[nodiscard]] const std::vector<size_t>& get_shape() const;
        [[nodiscard]] dtype_t get_dtype() const;

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
        [[nodiscard]] dtype_t get_dtype() const;
        [[nodiscard]] const std::vector<size_t>& get_shape() const;
        [[nodiscard]] type_t get_type() const;

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

        size_t new_var_id();
    };

    struct transpose_params {
        std::vector<size_t> permutation;
    };

    struct dot_general_params {
        std::vector<size_t> left_contract;
        std::vector<size_t> right_contract;
        std::vector<size_t> left_batch;
        std::vector<size_t> right_batch;
    };

    struct reduce_sum_params {
        std::vector<size_t> axes;
    };

    struct reduce_max_params {
        std::vector<size_t> axes;
    };

    struct reduce_min_params {
        std::vector<size_t> axes;
    };

    struct broadcast_in_dim_params {
        std::vector<size_t> shape;
        std::vector<size_t> broadcast_dimensions;
    };

    struct convert_element_type_params {
        dtype_t new_dtype;
    };

    struct cond_params {
        std::vector<expression> branches;
    };

    struct scan_params {
        expression jaxpr;
        size_t length;
        size_t num_consts;
        size_t num_carry;
        bool reverse;
    };

    struct integer_pow_params {
        size_t y;
    };

    struct concatenate_params {
        size_t dimension;
    };

    struct reshape_params {
        std::vector<size_t> new_sizes;
    };

    struct slice_params {
        std::vector<size_t> start_indices;
        std::vector<size_t> limit_indices;
        std::vector<size_t> strides;
    };

    struct pad_params {
        std::vector<std::array<size_t, 3>> padding_config; // (low, high, interior) list
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


    private:
        std::vector<value> input;
        std::vector<var_t> output;
        primitive_op op;
        params_variant params;
    };
}

#endif //JAXPR_TYPES_H
