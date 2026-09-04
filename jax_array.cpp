#include "jax_array.h"

#include <cmath>
#include <limits>

namespace jax {

    std::vector<size_t> new_get_shape(const std::vector<size_t>& x_shape, const std::vector<size_t>& idx_shape) {
        // If x: [n, ms...], idx: [ls...], then .get(): [ls..., ms...]
        std::vector<size_t> new_shape {};
        new_shape.reserve(x_shape.size() - 1 + idx_shape.size());

        for (auto dim: idx_shape) new_shape.push_back(dim);
        for (auto dim: x_shape | std::views::drop(1)) new_shape.push_back(dim);

        return new_shape;
    }

    void validate_scatter_op_shapes(const std::vector<size_t>& x_shape,
        const std::vector<size_t>& idx_shape, const std::vector<size_t>& update_shape) {
        // If x: [n, ms...], idx: [ls...], update: [ls..., ms...], then .op(): [n, ms...]

        using namespace std::views;

        if (idx_shape.size() > update_shape.size()) {
            throw std::logic_error("Error: idx rank cannot be greater than update rank");
        }

        for (auto [d1, d2]: zip(idx_shape, update_shape)) {
            if (d1 != d2) {
                throw std::logic_error("Error: dimension mismatch between idx and update");
            }
        }

        size_t num_ls = idx_shape.size();

        if (x_shape.size() - 1 != update_shape.size() - num_ls) {
            throw std::logic_error("Error: x and update have incompatible ranks");
        }

        for (auto [d1, d2]: zip(x_shape | drop(1), update_shape | drop(num_ls))) {
            if (d1 != d2) {
                throw std::logic_error("Error: dimension mismatch between x and update");
            }
        }
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

    type_t::type_t(const type_span& span):
    dtype(span.dtype),
    shape(span.shape.begin(), span.shape.end()) {}

    type_t::operator type_span() const {
        return type_span{dtype, shape};
    }

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

    array_t array_t::zeros(std::vector<size_t> shape, dtype_t dtype) {
        return build_fill(type_t{dtype, std::move(shape)}, 0.);
    }

    void array_t::compute_strides() {
        const std::vector<size_t>& shape = type.get_shape();
        strides.resize(shape.size());
        if (strides.empty()) {
            return;
        }

        // Strides are calculated backwards:

        strides[strides.size() - 1] = 1;
        for (size_t i = strides.size() - 1; i-->0;) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
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
            return array[indices];
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

    template<double Identity, double (*BinaryOp)(double, double)>
    array_t array_t::reduce_monoid(const std::vector<size_t>& axes) const {
        std::vector<size_t> remaining_axes = complement(axes, type.get_shape().size());
        std::vector<size_t> axes_sizes = extract_shape(axes, type.get_shape());
        std::vector<size_t> new_shape = extract_shape(remaining_axes, type.get_shape());

        std::vector<size_t> scratch_indices(type.get_shape().size());
        auto tensor_access = [&](Is remaining_indices, Is sum_indices) -> double {
            for (size_t i = 0; i < remaining_indices.size(); i++)
                scratch_indices[remaining_axes[i]] = remaining_indices[i];
            for (size_t i = 0; i < sum_indices.size(); i++)
                scratch_indices[axes[i]] = sum_indices[i];
            return operator[](scratch_indices);
        };

        std::vector<size_t> remaining_indices(remaining_axes.size(), 0);
        std::vector<size_t> op_indices(axes.size(), 0);


        std::vector new_value(num_elements(new_shape), Identity);
        size_t i = 0;

        for (auto& r: cartesian_product{remaining_indices, new_shape}) {
            // reduce[r] = op_s {x[r, s]}
            for (auto& s: cartesian_product{op_indices, axes_sizes}) {
                new_value[i] = BinaryOp(new_value[i], tensor_access(r, s));
            }
            i++;
        }

        return array_t{type_t{type.get_dtype(), std::move(new_shape)}, std::move(new_value)};
    }

    array_t array_t::reduce_sum(const std::vector<size_t>& axes) const {
        return reduce_monoid<0.,
        [](double x, double y){return x + y;}>(axes);
    }

    array_t array_t::reduce_max(const std::vector<size_t>& axes) const {
        return reduce_monoid<std::numeric_limits<double>::lowest(),
        [](double x, double y) {return std::max(x, y);}>(axes);
    }

    array_t array_t::reduce_min(const std::vector<size_t>& axes) const {
        return reduce_monoid<std::numeric_limits<double>::max(),
        [](double x, double y) {return std::min(x, y);}>(axes);
    }

    array_t array_t::reshape(std::vector<size_t> shape) const {
        std::vector<size_t> new_sizes = deduced_shape(shape, num_elements(type.get_shape()));
        return array_t {type_t{type.get_dtype(), std::move(new_sizes)}, value};
    }

    array_t array_t::transpose(const std::vector<size_t>& permutation) const {
        // permuted_indices[i] = indices[transpose[i]]
        std::vector<size_t> indices(type.get_shape().size(), 0);
        std::vector new_shape(permute(type.get_shape(), permutation));

        std::vector<double> new_value {};
        new_value.reserve(num_elements(new_shape));

        for (auto& is: cartesian_product{indices, new_shape}) {
            new_value.push_back(operator[](permute(is, permutation)));
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
            return operator[](scratch_indices);
        };

        for (auto& is: cartesian_product{indices, shape}) {
            new_value.push_back(select(is));
        }

        return array_t{type_t{type.get_dtype(), shape}, std::move(new_value)};
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

    const std::vector<size_t> &array_t::get_strides() const {
        return strides;
    }

    std::optional<literal_t> array_t::get_literal() const {
        if (type.get_shape().size() != 0) return std::nullopt;
        return literal_t{type.get_dtype(), value[0]};
    }

    void array_t::set_type(type_t new_type) {
        type = new_type;
    }

    mutable_array_span_t array_t::index(std::span<const size_t> indices) {
        return mutable_array_span_t{*this, indices};
    }

    const_array_span_t array_t::index(std::span<const size_t> indices) const {
        return const_array_span_t{*this, indices};
    }

    mutable_array_span_t array_t::index(size_t i) {
        return mutable_array_span_t{*this, std::span{&i, 1}};
    }

    const_array_span_t array_t::index(size_t i) const {
        return const_array_span_t{*this, std::span{&i, 1}};
    }

    dtype_t type_span::get_dtype() const {
        return dtype;
    }

    std::span<const size_t> type_span::get_shape() const {
        return shape;
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

    double &array_t::operator[](std::span<const size_t> indices) {
        return value[flatten_index(strides, indices)];
    }

    const double &array_t::operator[](std::span<const size_t> indices) const {
        return value[flatten_index(strides, indices)];
    }

    template<bool Const>
    array_span_t<Const>::array_span_t(const array_t& array, std::span<const size_t> indices) {
        type.dtype = array.type.get_dtype();

        const std::vector<size_t>& array_shape = array.type.get_shape();
        const std::vector<size_t>& array_strides = array.strides;

        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            offset += array_strides[i] * indices[i];
        }

        type.shape = std::span{array_shape.begin() + indices.size(), array_shape.end()};
        stride = std::span{array_strides.begin() + indices.size(), array_strides.end()};

        // The remaining dimensions form a contiguous block; its size is the
        // stride of the last fixed axis (or the whole buffer if nothing is fixed).
        size_t total = indices.empty() ? array.value.size() : array_strides[indices.size() - 1];
        value = std::span{array.value.begin() + offset, total};
    }

    template<bool Const>
    array_span_t<Const>::element_t& array_span_t<Const>::operator[](std::span<const size_t> indices) const {
        const double& element = value[flatten_index(stride, indices)];
        if constexpr (Const) {
            return element;
        } else {
            return const_cast<double&>(element);
        }
    }

    template<bool Const>
    array_span_t<Const> array_span_t<Const>::index(std::span<const size_t> indices) const {
        array_span_t result = *this;

        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            offset += stride[i] * indices[i];
        }

        result.type.shape = std::span{type.shape.begin() + indices.size(), type.shape.end()};
        result.stride = std::span{stride.begin() + indices.size(), stride.end()};

        size_t total = indices.empty() ? value.size() : stride[indices.size() - 1];
        result.value = std::span{value.begin() + offset, total};
        return result;
    }

    template<bool Const>
    array_span_t<Const> array_span_t<Const>::index(size_t i) const {
        return index(std::span{&i, 1});
    }

    template<bool Const>
    const type_span& array_span_t<Const>::get_type() const {
        return type;
    }

    template<bool Const>
    std::span<typename array_span_t<Const>::element_t> array_span_t<Const>::get_value() const {
        if constexpr (Const) {
            return value;
        } else {
            return std::span<double>{const_cast<double*>(value.data()), value.size()};
        }
    }

    template<bool Const>
    std::span<const size_t> array_span_t<Const>::get_strides() const {
        return stride;
    }

    template<bool Const>
    array_span_t<Const>::operator array_t() const {
        std::vector<double> buffer(value.begin(), value.end());
        std::vector<size_t> shape(type.shape.begin(), type.shape.end());
        return array_t{type_t{type.dtype, std::move(shape)}, std::move(buffer)};
    }

    template class array_span_t<true>;
    template class array_span_t<false>;



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
    std::vector<size_t> get_implicit_broadcast_shape(std::span<const size_t> left_shape,
        std::span<const size_t> right_shape) {

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
                std::cerr << std::vector<size_t>(left_shape.begin(), left_shape.end()) << '\n';
                std::cerr << std::vector<size_t>(right_shape.begin(), right_shape.end()) << '\n';
                throw std::logic_error("Error: shape mismatch in implicit broadcast");
            }
        }

        return new_shape;
    }

    implicit_broadcast_result get_implicit_broadcast_result(std::span<const size_t> left_shape,
        std::span<const size_t> right_shape) {
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
