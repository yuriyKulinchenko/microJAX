#include "jax_types.h"
#include "DCE_class.h"

namespace jax {

    size_t num_elements(std::span<const size_t> shape) {
        size_t total = 1;
        for (size_t dim : shape) total *= dim;
        return total;
    }

    std::string_view to_string(primitive_op op) {
        switch (op) {
    #define X(name) case primitive_op::name: return #name;
            PRIMITIVE_OP_LIST(X)
    #undef X
        }
        return "";
    }

    std::string_view to_string(dtype_t t) {
        switch (t) {
    #define X(name) case dtype_t::name: return #name;
            TYPE_ENUM_LIST(X)
    #undef X
        }
        return "";
    }

    bool is_floating(dtype_t t) {
        using enum dtype_t;
        return t == F32 || t == F64;
    }

    bool is_integral(dtype_t t) {
        using enum dtype_t;
        return t == I32 || t == I64 || t == BOOL;
    }

    dtype_t widest_float(dtype_t t1, dtype_t t2) {
        using enum dtype_t;
        return (t1 == F64 || t2 == F64) ? F64 : F32;
    }

    dtype_t widest_int(dtype_t t1, dtype_t t2) {
        using enum dtype_t;
        return (t1 == I64 || t2 == I64) ? I64 : I32;
    }

    dtype_t resultant_type(dtype_t t1, dtype_t t2) {
        using enum dtype_t;
        if (t1 == BOOL) return t2;
        if (t2 == BOOL) return t1;

        if (is_floating(t1)) {
            if (is_floating(t2)) {
                return widest_float(t1, t2);
            }
            return t1;
        }

        if (is_floating(t2)) {
            return t2;
        }

        return widest_int(t1, t2);
    }

    type_t::type_t(dtype_t dtype):
    dtype(dtype) {}

    type_t::type_t(dtype_t dtype, std::vector<size_t> shape):
    dtype(dtype),
    shape(std::move(shape)) {}

    bool type_t::operator==(const type_t& other) const {
        return dtype == other.dtype && shape == other.shape;
    }

    bool type_t::is_i32() const {
        return dtype == dtype_t::I32;
    }

    bool type_t::is_i64() const {
        return dtype == dtype_t::I64;
    }

    bool type_t::is_f32() const {
        return dtype == dtype_t::F32;
    }

    bool type_t::is_f64() const {
        return dtype == dtype_t::F64;
    }

    bool type_t::is_bool() const {
        return dtype == dtype_t::BOOL;
    }

    dtype_t type_t::get_dtype() const {
        return dtype;
    }

    const std::vector<size_t>& type_t::get_shape() const {
        return shape;
    }

    std::vector<size_t> &type_t::get_shape() {
        return shape;
    }

    void type_t::set_dtype(const dtype_t new_dtype) {
        dtype = new_dtype;
    }

    var_t::var_t(size_t id, type_t type): id(id), type(std::move(type)) {}

    size_t var_t::get_id() const {
        return id;
    }

    void var_t::set_id(size_t new_id) {
        id = new_id;
    }

    const type_t& var_t::get_type() const {
        return type;
    }

    const std::vector<size_t> &var_t::get_shape() const {
        return type.get_shape();
    }

    dtype_t var_t::get_dtype() const {
        return type.get_dtype();
    }

    array_t::array_t(): type(dtype_t::F32), value({0.}){}

    array_t::array_t(type_t type, std::vector<double> value):
    type(std::move(type)),
    value(std::move(value)) {
        if (this->value.size() != num_elements(this->type.get_shape())) {
            throw std::logic_error("Error: array value does not match the size of its shape");
        }

        compute_strides();
        check_single_value();
    }

    array_t::array_t(double value)
    : array_t(type_t{dtype_t::F32}, std::vector{value}) {}

    array_t array_t::build(dtype_t dtype, std::vector<size_t> shape,
        const std::function<double(const std::vector<size_t>&)>& f) {
        type_t type {dtype, std::move(shape)};

        const auto& shape_vector = type.get_shape();
        auto index_vector = std::vector<size_t>(shape_vector.size(), 0);
        auto output_vector = std::vector<double>(num_elements(shape_vector));

        for (auto& output: output_vector) {
            output = f(index_vector);
            for (size_t j = index_vector.size(); j--> 0;) {
                if (index_vector[j] < shape_vector[j] - 1) {
                    index_vector[j]++;
                    break;
                }
                index_vector[j] = 0;
            }
        }

        return array_t{std::move(type), std::move(output_vector)};
    }

    array_t array_t::build_fill(type_t type, double value) {
        size_t count = num_elements(type.get_shape());
        return array_t{std::move(type), std::vector(count, value)};
    }

    void array_t::compute_strides() {
        const std::vector<size_t>& shape = type.get_shape();
        stride.resize(shape.size());
        if (stride.empty()) {
            return;
        }

        // Strides are calculated backwards:

        stride[stride.size() - 1] = 1;
        for (size_t i = stride.size() - 1; i-->0;) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }

    template<double(*f)(double)>
    array_t elementwise_unary_array_op(const array_t& array) {
        array_t new_array{array};
        for (double& x: new_array.get_value()) x = f(x);
        return new_array;
    }

    array_t array_t::operator-() const {
        return elementwise_unary_array_op<[](double x){return -x;}>(*this);
    }

    array_t array_t::sin() const {
        return elementwise_unary_array_op<std::sin>(*this);
    }

    array_t array_t::cos() const {
        return elementwise_unary_array_op<std::cos>(*this);
    }

    array_t array_t::exp() const {
        return elementwise_unary_array_op<std::exp>(*this);
    }

    array_t array_t::log() const {
        return elementwise_unary_array_op<std::log>(*this);
    }

    std::vector<size_t> extract_shape(const std::vector<size_t>& indices, const std::vector<size_t>& shape) {
        std::vector<size_t> new_shape {};
        new_shape.reserve(indices.size());
        std::ranges::transform(indices, std::back_inserter(new_shape), [&](size_t i) {
           return shape[i];
        });
        return new_shape;
    }

    using Is = const std::vector<size_t>&;

    array_t array_t::dot_general(
        const array_t& other,
        const std::vector<size_t>& left_contract,
        const std::vector<size_t>& right_contract,
        const std::vector<size_t>& left_batch,
        const std::vector<size_t>& right_batch) const {
        // new dimensions are (batch, free_left, free_right)
        std::vector<size_t> left_free = complement(left_batch, left_contract, type.get_shape().size());
        std::vector<size_t> right_free = complement(right_batch, right_contract, other.type.get_shape().size());

        // For dynamic looping:

        const std::vector<size_t> batch_sizes = extract_shape(left_batch, type.get_shape());
        const std::vector<size_t> contract_sizes = extract_shape(left_contract, type.get_shape());
        const std::vector<size_t> left_free_sizes = extract_shape(left_free, type.get_shape());
        const std::vector<size_t> right_free_sizes = extract_shape(right_free, other.type.get_shape());

        std::vector<size_t> shape {};
        shape.reserve(left_batch.size() + left_free.size() + right_free.size());

        for (size_t x: batch_sizes) shape.push_back(x);
        for (size_t x: left_free_sizes) shape.push_back(x);
        for (size_t x: right_free_sizes) shape.push_back(x);

        auto tensor_access = [](
            Is batch_indices, Is free_indices, Is contract_indices,
            Is batch, Is free, Is contract,
            const array_t& array, std::vector<size_t>& indices
        ) -> double {
            for (size_t i = 0; i < batch_indices.size(); i++) indices[batch[i]] = batch_indices[i];
            for (size_t i = 0; i < free_indices.size(); i++) indices[free[i]] = free_indices[i];
            for (size_t i = 0; i < contract_indices.size(); i++) indices[contract[i]] = contract_indices[i];
            return array.access(indices);
        };

        std::vector<size_t> left_scratch_indices(type.get_shape().size());
        auto left_access = [&](Is batch_indices, Is free_indices, Is contract_indices) -> double {
            return tensor_access(batch_indices, free_indices, contract_indices,
                left_batch, left_free, left_contract, *this, left_scratch_indices);
        };

        std::vector<size_t> right_scratch_indices(other.type.get_shape().size());
        auto right_access = [&](Is batch_indices, Is free_indices, Is contract_indices) -> double {
            return tensor_access(batch_indices, free_indices, contract_indices,
                right_batch, right_free, right_contract, other, right_scratch_indices);
        };

        std::vector<double> new_value
        (num_elements(batch_sizes) * num_elements(left_free_sizes) * num_elements(right_free_sizes), 0);
        size_t i = 0;

        std::vector<size_t> batch_indices(batch_sizes.size(), 0);
        std::vector<size_t> left_free_indices(left_free_sizes.size(), 0);
        std::vector<size_t> right_free_indices(right_free_sizes.size(), 0);
        std::vector<size_t> contract_indices(contract_sizes.size(), 0);

        for (auto& b: cartesian_product{batch_indices, batch_sizes}) {
            for (auto& f_l: cartesian_product{left_free_indices, left_free_sizes}) {
                for (auto& f_r: cartesian_product{right_free_indices, right_free_sizes}) {
                    // product[b, f_l, f_r] = sum_c { left[b, f_l, c] * right[b, f_r, c] }
                    for (auto& c: cartesian_product{contract_indices, contract_sizes}) {
                        new_value[i] += left_access(b, f_l, c) * right_access(b, f_r, c);
                    }
                    i++;
                }
            }
        }

        return array_t{type_t{type.get_dtype(), std::move(shape)}, std::move(new_value)};
    }

    array_t array_t::reduce_sum(const std::vector<size_t>& axes) const {
        std::vector<size_t> remaining_axes = complement(axes, type.get_shape().size());
        std::vector<size_t> axes_sizes = extract_shape(axes, type.get_shape());
        std::vector<size_t> new_shape = extract_shape(remaining_axes, type.get_shape());

        std::vector<size_t> scratch_indices(type.get_shape().size());
        auto tensor_access = [&](Is remaining_indices, Is sum_indices) -> double {
            for (size_t i = 0; i < remaining_indices.size(); i++)
                scratch_indices[remaining_axes[i]] = remaining_indices[i];
            for (size_t i = 0; i < sum_indices.size(); i++)
                scratch_indices[axes[i]] = sum_indices[i];
            return access(scratch_indices);
        };

        std::vector<size_t> remaining_indices(remaining_axes.size(), 0);
        std::vector<size_t> sum_indices(axes.size(), 0);


        std::vector<double> new_value(num_elements(new_shape), 0);
        size_t i = 0;

        for (auto& r: cartesian_product{remaining_indices, new_shape}) {
            // reduce[r] = sum_s {x[r, s]}
            for (auto& s: cartesian_product{sum_indices, axes_sizes}) {
                new_value[i] += tensor_access(r, s);
            }
            i++;
        }

        return array_t{type_t{type.get_dtype(), std::move(new_shape)}, std::move(new_value)};
    }

    array_t array_t::transpose(const std::vector<size_t>& permutation) const {
        // permuted_indices[i] = indices[transpose[i]]
        std::vector<size_t> indices(type.get_shape().size(), 0);
        std::vector new_shape(permute(type.get_shape(), permutation));

        std::vector<double> new_value {};
        new_value.reserve(num_elements(new_shape));

        for (auto& is: cartesian_product{indices, new_shape}) {
            new_value.push_back(access(permute(is, permutation)));
        }

        return array_t{type_t{type.get_dtype(), std::move(new_shape)}, std::move(new_value)};
    }

    array_t array_t::convert_element_type(dtype_t dtype) const {
        return array_t{type_t{dtype, type.get_shape()}, value};
    }

    array_t array_t::broadcast_in_dim(
        const std::vector<size_t>& shape,
        const std::vector<size_t>& broadcast_dimensions) const {
        std::vector<size_t> indices(shape.size(), 0);

        std::vector<double> new_value {};
        new_value.reserve(num_elements(shape));

        const std::vector<size_t>& src_shape = type.get_shape();

        std::vector<size_t> scratch_indices(broadcast_dimensions.size());
        auto select = [&](Is is) -> double {
            for (size_t i = 0; i < broadcast_dimensions.size(); i++) {
                size_t coord = is[broadcast_dimensions[i]];
                scratch_indices[i] = (src_shape[i] == 1) ? 0 : coord;
            }
            return access(scratch_indices);
        };

        for (auto& is: cartesian_product{indices, shape}) {
            new_value.push_back(select(is));
        }

        return array_t{type_t{type.get_dtype(), shape}, std::move(new_value)};
    }

    template<typename F>
    array_t array_t::binary_op(const array_t& other, F f) const {
        implicit_broadcast_result result = get_implicit_broadcast_result(type.get_shape(), other.type.get_shape());

        auto apply_op = [&](array_t left, const array_t& right) -> array_t {
            left.type.set_dtype(resultant_type(left.type.get_dtype(), right.type.get_dtype()));
            for (const auto& [x, y]: std::views::zip(left.value, right.value)) x = f(x, y);
            return left;
        };

        if (type.get_shape() != result.new_shape) {
            array_t left_broadcast = broadcast_in_dim(result.new_shape, result.left_broadcast_dimensions);
            if (other.type.get_shape() != result.new_shape) {
                array_t right_broadcast = other.broadcast_in_dim(result.new_shape, result.right_broadcast_dimensions);
                return apply_op(std::move(left_broadcast), right_broadcast);
            }
            return apply_op(std::move(left_broadcast), other);
        }

        if (other.type.get_shape() != result.new_shape) {
            array_t right_broadcast = other.broadcast_in_dim(result.new_shape, result.right_broadcast_dimensions);
            return apply_op(*this, right_broadcast);
        }

        return apply_op(*this, other);
    }

    #define BINARY_OP(op)                                                   \
    array_t array_t::operator op (const array_t& other) const {             \
        return binary_op(other, [](double x, double y){return x op y;});    \
    }

    #define BINARY_COMPARE_OP(op)                                                   \
    array_t array_t::operator op (const array_t& other) const {                     \
        return binary_op(other, [](double x, double y) {return x op y ? 1. : 0.;}); \
    }                                                                               \

    BINARY_OP(+);
    BINARY_OP(-);
    BINARY_OP(*);
    BINARY_OP(/);

    BINARY_COMPARE_OP(<);
    BINARY_COMPARE_OP(<=);
    BINARY_COMPARE_OP(>);
    BINARY_COMPARE_OP(>=);

    // TODO: expose this globally somehow
    static double epsilon = 1. / static_cast<double>(1 << 10);

    static bool double_eq(double x, double y) {
        return std::abs(x - y) < epsilon;
    }

    array_t array_t::elementwise_equal(const array_t& other) const {
        return binary_op(other, [](double x, double y) -> double {
            return double_eq(x, y) ? 1 : 0;
        });
    }

    array_t array_t::elementwise_not_equal(const array_t& other) const {
        return binary_op(other, [](double x, double y) -> double {
            return !double_eq(x, y) >= epsilon ? 1 : 0;
        });
    }

    bool array_t::operator==(const array_t& other) const {
        if (type != other.type) return false;
        if (is_integral(type.get_dtype())) return value == other.value;

        for (auto [x, y]: std::views::zip(value, other.value)) {
            if (!double_eq(x, y)) return false;
        }

        return true;
    }

    const type_t& array_t::get_type() const {
        return type;
    }

    type_t& array_t::get_type() {
        return type;
    }

    const std::vector<double>& array_t::get_value() const {
        return value;
    }

    std::vector<double>& array_t::get_value() {
        return value;
    }

    std::optional<literal_t> array_t::get_literal() const {
        if (type.get_shape().size() != 0) return std::nullopt;
        return literal_t{type.get_dtype(), value[0]};
    }

    void array_t::set_type(type_t new_type) {
        type = new_type;
    }

    dtype_t type_span::get_dtype() const {
        return dtype;
    }

    std::span<const size_t> type_span::get_shape() const {
        return shape;
    }

    array_span_t::array_span_t(const array_t& array, const std::vector<size_t>& indices) {
        type.dtype = array.type.get_dtype();

        size_t i_0 = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            i_0 += array.stride[i] * indices[i];
        }

        const auto& array_shape = array.type.get_shape();
        type.shape = std::span{array_shape.begin() + indices.size(), array_shape.end()};

        const auto& array_stride = array.stride;
        stride = std::span {array_stride.begin() + indices.size(), array_stride.end()};

        // shape_size(i) = stride[i] * shape[i]
        size_t total_size = array.stride[indices.size()] * array.type.get_shape()[indices.size()];
        value = std::span{array.value.begin() + i_0, total_size};
    }

    double array_span_t::operator[](const std::vector<size_t>& indices) const {
        return access(indices);
    }

    const double& array_span_t::access(const std::vector<size_t>& indices) const {
        return value[flatten_index(stride, indices)];
    }

    const type_span& array_span_t::get_type() const {
        return type;
    }

    std::span<const double> array_span_t::get_value() const {
        return value;
    }

    void array_t::check_single_value() {
        if (value.empty()) {
            has_single_value_ = true;
            return;
        }

        double val = value[0];
        for (size_t i = 1; i < value.size(); i++) {
            if (value[i] != val) {
                has_single_value_ = false;
                return;
            }
        }

        has_single_value_ = true;
        single_value = val;
    }

    double array_t::operator[](const std::vector<size_t>& indices) {
        return access(indices);
    }

    double& array_t::access(const std::vector<size_t>& indices) {
        return value[flatten_index(stride, indices)];
    }

    const double& array_t::access(const std::vector<size_t>& indices) const {
        return value[flatten_index(stride, indices)];
    }

    array_t array_t::slice(const std::vector<size_t>& indices) const {
        if (indices.empty()) return *this;

        if (indices.size() > type.get_shape().size()) {
            throw formatted_error("Error: cannot access indices given by {}", indices);
        }

        std::vector<size_t> new_shape{type.get_shape().begin() + indices.size(), type.get_shape().end()};
        size_t start_index = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            start_index += stride[i] * indices[i];
        }

        // Now, calculate the size:
        size_t new_size = stride[indices.size() - 1];

        std::vector<double> new_value {value.begin() + start_index, value.begin() + start_index + new_size};

        return array_t{type_t{type.get_dtype(), std::move(new_shape)}, std::move(new_value)};
    }

    // Precondition is that 'slice' and 'indices' are both valid.
    void array_t::add_slice(const std::vector<size_t>& indices, const array_t& slice) {
        size_t start_index = 0;
        has_single_value_ = false;
        for (size_t i = 0; i < indices.size(); i++) {
            start_index += stride[i] * indices[i];
        }
        for (size_t i = 0; i < slice.value.size(); i++) {
            value[start_index + i] += slice.value[i];
        }
    }

    bool array_t::has_single_value(f64 val) const {
        return has_single_value_ && val == single_value;
    }

    bool array_t::has_single_value() const {
        return has_single_value_;
    }

    literal_t::literal_t(dtype_t dtype, double value): dtype(dtype), value(value) {}

    dtype_t literal_t::get_dtype() const {
        return dtype;
    }

    double literal_t::get_value() const {
        return value;
    }

    const std::vector<size_t> &literal_t::get_shape() {
        return shape;
    }

    value::value(literal_t literal): variant_(std::move(literal)) {}
    value::value(var_t var): variant_(std::move(var)) {}

    const literal_t& value::get_literal() const {
        return std::get<literal_t>(variant_);
    }

    const var_t& value::get_var() const {
        return std::get<var_t>(variant_);
    }

    dtype_t value::get_dtype() const {
        return std::visit([](auto& x){return x.get_dtype();}, variant_);
    }

    const std::vector<size_t> &value::get_shape() const {
        if (is<var_t>()) {
            return std::get<var_t>(variant_).get_shape();
        }
        return literal_t::get_shape();
    }

    type_t value::get_type() const {
        if (is<var_t>()) {
            return std::get<var_t>(variant_).get_type();
        }
        return type_t{std::get<literal_t>(variant_).get_dtype(), literal_t::get_shape()};
    }

    const std::vector<value> &equation::get_input() const {
        return input;
    }

    std::vector<value> &equation::get_input() {
        return input;
    }

    const std::vector<var_t> &equation::get_output() const {
        return output;
    }

    std::vector<var_t> &equation::get_output() {
        return output;
    }

    const value &equation::get_input(size_t i) const {
        return input[i];
    }

    value &equation::get_input(size_t i) {
        return input[i];
    }

    const var_t &equation::get_output(size_t i) const {
        return output[i];
    }

    var_t &equation::get_output(size_t i) {
        return output[i];
    }

    primitive_op equation::get_op() const {
        return op;
    }

    const params_variant& equation::get_params() const {
        return params;
    }

    params_variant& equation::get_params() {
        return params;
    }




    equation::equation(std::vector<value> input, std::vector<var_t> output, primitive_op op):
    input(std::move(input)),
    output(std::move(output)),
    op(op)
    {}

    equation::equation(std::vector<value> input, std::vector<var_t> output, primitive_op op, params_variant params):
    input(std::move(input)),
    output(std::move(output)),
    op(op),
    params(params)
    {}

    void expression::add_invar(var_t var) {
        invars.push_back(std::move(var));
    }

    void expression::add_constvar(var_t var) {
        constvars.push_back(std::move(var));
    }

    void expression::add_output(value val) {
        outvals.push_back(std::move(val));
    }

    equation& expression::add_equation(equation eq) {
        equations.push_back(std::move(eq));
        return equations[equations.size() - 1];
    }

    var_t expression::fresh_var(type_t type) {
        return var_t{new_var_id(), std::move(type)};
    }

    size_t expression::new_var_id() {
        return var_id++;
    }

    void expression::eliminate_dead_code() {
        DCE_class instance {*this};
        instance.apply_dead_code_elimination();
    }

    std::vector<size_t> get_implicit_broadcast_shape(const std::vector<size_t>& left_shape,
        const std::vector<size_t>& right_shape) {

        // Normalize so that left_shape.size() < right_shape.size():
        if (right_shape.size() < left_shape.size())
            return get_implicit_broadcast_shape(right_shape, left_shape);

        size_t new_size = right_shape.size();
        std::vector<size_t> new_shape(new_size);

        // For all places where the 1s prefix would have existed, populate with right_shape:

        size_t delta = right_shape.size() - left_shape.size();
        for (size_t i = 0; i < delta; i++) {
            new_shape[i] = right_shape[i];
        }

        // For all places where there is a shared prefix,
        // Ensure that one of them is 1, and take the max:

        for (size_t i = 0; i < left_shape.size(); i++) {
            if (left_shape[i] == 1) {
                new_shape[delta + i] = right_shape[i];
            } else if (right_shape[delta + i] == 1) {
                new_shape[delta + i] = left_shape[delta + i];
            } else if (left_shape[i] == right_shape[delta + i]) {
                new_shape[delta + i] = left_shape[i];
            } else {
                std::cerr << left_shape << '\n';
                std::cerr << right_shape << '\n';
                throw std::logic_error("Error: shape mismatch in implicit broadcast");
            }
        }

        return new_shape;
    }

    implicit_broadcast_result get_implicit_broadcast_result(const std::vector<size_t>& left_shape,
        const std::vector<size_t>& right_shape) {
        std::vector<size_t> new_shape = get_implicit_broadcast_shape(left_shape, right_shape);

        std::vector<size_t> broadcast_dimensions(new_shape.size());
        for (size_t i = 0; i < new_shape.size(); i++) {
            broadcast_dimensions[i] = i;
        }

        // From the new_shape, infer the broadcast dimensions:
        if (left_shape.size() < right_shape.size()) {
            size_t delta = right_shape.size() - left_shape.size();
            std::vector<size_t> left_broadcast_dimensions(left_shape.size());
            for (size_t i = 0; i < left_shape.size(); i++) {
                left_broadcast_dimensions[i] = i + delta;
            }
            return {
                std::move(new_shape),
                std::move(left_broadcast_dimensions),
                std::move(broadcast_dimensions)
            };
        }

        size_t delta = left_shape.size() - right_shape.size();
        std::vector<size_t> right_broadcast_dimensions(right_shape.size());
        for (size_t i = 0; i < right_shape.size(); i++) {
            right_broadcast_dimensions[i] = i + delta;
        }
        return {
            std::move(new_shape),
            std::move(broadcast_dimensions),
            std::move(right_broadcast_dimensions)
        };
    }
}
