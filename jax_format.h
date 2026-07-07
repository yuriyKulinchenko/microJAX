#ifndef JAX_FORMAT_H
#define JAX_FORMAT_H

#include <format>
#include <sstream>

#include "jax_logger.h"

// Bridges std::format to the existing operator<< logging for jax types.


template<typename T>
struct ostream_formatter {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const T& value, std::format_context& ctx) const {
        std::ostringstream oss;
        oss << value;
        return std::format_to(ctx.out(), "{}", oss.str());
    }
};

template<> struct std::formatter<jax::type_t> : ostream_formatter<jax::type_t> {};
template<> struct std::formatter<jax::var_t> : ostream_formatter<jax::var_t> {};
template<> struct std::formatter<jax::array_t> : ostream_formatter<jax::array_t> {};
template<> struct std::formatter<jax::equation> : ostream_formatter<jax::equation> {};
template<> struct std::formatter<jax::expression> : ostream_formatter<jax::expression> {};

#endif //JAX_FORMAT_H
