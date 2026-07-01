#ifndef HELPER_H
#define HELPER_H

#include <algorithm>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <format>
#include <iostream>
#include <functional>

#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define RESET   "\033[0m"

inline std::string to_lower(std::string_view s) {
    std::string result(s);
    std::ranges::transform(result, result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
    stream << '[';
    if (vector.size() != 0) {
        stream << vector[0];
        for (int i = 1; i < vector.size(); i++) {
            stream << ", " << vector[i];
        }
    }
    return stream << ']';
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& stream, std::unordered_map<T, U>& map) {
    stream << '{';

    auto it = map.begin();
    if (it != map.end()) {
        stream << it->first << " -> " << it->second;
        ++it;
    }

    for (; it != map.end(); ++it) {
        stream << ", " << it->first << " -> " << it->second;
    }

    return stream << '}';
}

inline std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file");

    std::streamsize size = file.tellg();
    file.seekg(0);

    std::string buffer(size, '\0');
    file.read(buffer.data(), size);

    return buffer;
}

template <typename... Args>
std::logic_error formatted_error(std::format_string<Args...> fmt,
                                 Args&&... args) {
    return std::logic_error(std::format(fmt, std::forward<Args>(args)...));
}

inline void print_green(const std::string& s) {
    std::cout << GREEN << s << RESET;
}

inline void print_red(const std::string& s) {
    std::cout << RED << s << RESET;
}

template <typename F, typename R, typename... Args>
concept invocable_r = std::is_invocable_r_v<R, F, Args...>;

inline std::vector<size_t> complement(const std::vector<size_t>& vec, const size_t n) {
    // Given that vec is a subsequence of (0, ..., n-1),
    // complement(vec, n) is the complement of that subsequence

    std::vector<size_t> out_vec {};
    out_vec.reserve(n - vec.size());

    for (size_t i = 0, j = 0; i < n; i++) {
        if (j < vec.size() && vec[j] == i) {
            j++;
            continue;
        }
        out_vec.push_back(i);
    }

    return out_vec;
}

#endif //HELPER_H
