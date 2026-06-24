#ifndef TAGGED_VARIANT_H
#define TAGGED_VARIANT_H

#include <type_traits>
#include <variant>
#include <concepts>
#include <utility>

// tagged_variant is a wrapper around std::variant which uses enum tags to discriminate between members

template<typename E>
concept Enum = std::is_enum_v<E>;

template<Enum auto tag, typename Type>
class entry;

namespace impl_t_var {

    // type_vector utilities

    template<typename... Ts>
    struct type_vector;

    template<typename T, typename Vector>
    struct cons_class;

    template<typename T, typename... Ts>
    struct cons_class<T, type_vector<Ts...>> {
        using vector = type_vector<T, Ts...>;
    };

    template<typename T, typename Vector>
    using cons = typename cons_class<T, Vector>::vector;

    // Extract tag type:

    template<typename... Entries>
    struct tag_type_class;

    template<Enum auto tag, typename T, typename... Rest>
    struct tag_type_class<entry<tag, T>, Rest...> {
        using type = decltype(tag);
    };

    // 'only_entries' enforces that all entries passed are indeed entry types:

    template<typename T>
    struct is_entry : std::false_type {};

    template<Enum auto tag, typename T>
    struct is_entry<entry<tag, T>> : std::true_type {};

    // 'safe_entries' enforces that all entry tags come from the same enum:

    template<typename... Entries>
    struct safe_entries_class;

    template<>
    struct safe_entries_class<> {
        static constexpr bool value = true;
    };

    template<Enum auto tag, typename Type>
    struct safe_entries_class<entry<tag, Type>> {
        static constexpr bool value = true;
    };

    template<Enum auto t1, Enum auto t2, typename T1, typename  T2, typename... Entries>
    struct safe_entries_class<entry<t1, T1>, entry<t2, T2>, Entries...> {
        static constexpr bool value = std::same_as<decltype(t1), decltype(t2)>
            && safe_entries_class<entry<t2,T2>, Entries...>::value;
    };

    // 'duplicate_tags' checks whether a duplicate tag exists - this shouldn't be allowed:

    template<Enum auto tag, typename... Entries>
    struct tag_occurrence;

    template<Enum auto tag>
    struct tag_occurrence<tag> {
        static constexpr bool value = false;
    };

    template<Enum auto tag, Enum auto other, typename T, typename... Rest>
    struct tag_occurrence<tag, entry<other, T>, Rest...> {
        static constexpr bool value = (tag == other) || tag_occurrence<tag, Rest...>::value;
    };

    template<typename... Entries>
    struct duplicate_tags_class;

    template<>
    struct duplicate_tags_class<> {
        static constexpr bool value = false;
    };

    template<Enum auto tag, typename T, typename... Rest>
    struct duplicate_tags_class<entry<tag, T>, Rest...> {
        static constexpr bool value = tag_occurrence<tag, Rest...>::value
            || duplicate_tags_class<Rest...>::value;
    };

    // Convert 'Entries' to a corresponding variant:

    template<typename... Entries>
    struct entries_to_vector;

    template<>
    struct entries_to_vector<> {
        using vector = type_vector<>;
    };

    template<Enum auto tag, typename Type, typename... Rest>
    struct entries_to_vector<entry<tag, Type>, Rest...> {
        using vector = cons<Type, typename entries_to_vector<Rest...>::vector>;
    };

    template<typename Vector>
    struct vector_to_variant_class;

    template<typename... Entries>
    struct vector_to_variant_class<type_vector<Entries...>> {
        using type = std::variant<Entries...>;
    };

    template<typename Vector>
    struct vector_to_common_type_class;

    template<typename... Entries>
    struct vector_to_common_type_class<type_vector<Entries...>> {
        using type = std::common_type_t<Entries...>;
    };

    // Core functionality:

    template<Enum auto tag, size_t accumulator, typename... Entries>
    struct find_tag_index_class;

    template<Enum auto tag, Enum auto other, size_t accumulator, typename  T, typename... Rest>
    struct find_tag_index_class<tag, accumulator, entry<other, T>, Rest...> {
        static constexpr size_t value = std::conditional_t<
                (tag == other),
                std::integral_constant<size_t, accumulator>,
                find_tag_index_class<tag, accumulator + 1, Rest...>
        >::value;
    };

    template<Enum auto tag, typename... Entries>
    struct find_tag_type_class;

    template<Enum auto tag, Enum auto other, typename T, typename... Rest>
    struct find_tag_type_class<tag, entry<other, T>, Rest...> {
        // Lazy evaluation, to prevent recursing beyond the found solution:
        using type = typename std::conditional_t<
            (tag == other),
            std::type_identity<T>,
            find_tag_type_class<tag, Rest...>
        >::type;

    };

    // Match expression:

    template<typename Entry, typename Function>
    struct entry_matches_function;

    template<Enum auto tag, typename Type, typename Function>
    struct entry_matches_function<entry<tag, Type>, Function> {
        static constexpr bool value = std::invocable<Function, Type&>;
    };

    template<typename EntryVector, typename FunctionVector>
    struct entries_match_functions_class;

    template<typename... Entries, typename... Functions>
    struct entries_match_functions_class<type_vector<Entries...>, type_vector<Functions...>> {
        static constexpr bool value = (entry_matches_function<Entries, Functions>::value && ...);
    };

    // All outputs of a match expression must have a shared type:

    template<typename Entry, typename Function>
    struct output_type_class;

    template<Enum auto tag, typename Type, typename Function>
    struct output_type_class<entry<tag, Type>, Function> {
        using type = std::invoke_result_t<Function, Type&>;
    };

    template<typename EntryVector, typename FunctionVector>
    struct outputs_type_class;

    template<typename... Entries, typename... Functions>
    struct outputs_type_class<type_vector<Entries...>, type_vector<Functions...>> {
        using type = std::common_type_t<typename output_type_class<Entries, Functions>::type...>;
    };

}

// ===================================================== CONCEPTS =====================================================

template<typename... Entries>
concept only_entries = (impl_t_var::is_entry<Entries>::value && ...);

template<typename... Entries>
concept safe_entries = impl_t_var::safe_entries_class<Entries...>::value;

template<auto tag, typename... Entries>
concept tag_occurs = impl_t_var::tag_occurrence<tag, Entries...>::value;

template<typename... Entries>
concept duplicate_tags = impl_t_var::duplicate_tags_class<Entries...>::value;

template<typename EntryVector, typename FunctionVector>
concept entries_match_functions = impl_t_var::entries_match_functions_class<EntryVector, FunctionVector>::value;

// ====================================================== TYPES =======================================================

template<typename... Entries>
using entries_to_variant_t = typename impl_t_var::vector_to_variant_class<
    typename impl_t_var::entries_to_vector<Entries...>::vector>::type;

template<typename... Entries>
using tag_type_t = typename impl_t_var::tag_type_class<Entries...>::type;

template<typename EntryVector, typename FunctionVector>
using output_type_t = typename impl_t_var::outputs_type_class<EntryVector, FunctionVector>::type;

// ====================================================== VALUES ======================================================

template<Enum auto tag, typename... Entries>
static constexpr size_t tag_index_v = impl_t_var::find_tag_index_class<tag, 0, Entries...>::value;

template<typename... Entries>
requires only_entries<Entries...> && safe_entries<Entries...> && (!duplicate_tags<Entries...>)
class tagged_variant {
    static_assert(sizeof...(Entries) > 0, "tagged_variant cannot be instantiated with no entries");
public:
    using raw_variant = entries_to_variant_t<Entries...>;
    using tag_type = tag_type_t<Entries...>;

    template<Enum auto tag, typename... Entries_>
    requires tag_occurs<tag, Entries_...>
    friend auto& get(tagged_variant<Entries_...>& var);

    template<Enum auto tag, typename... Entries_>
    requires tag_occurs<tag, Entries_...>
    friend const auto& get(const tagged_variant<Entries_...>& var);


    template<size_t index, typename Ret_, typename... Entries_, typename F_>
    friend Ret_ match_(tagged_variant<Entries_...>& var, F_ f);

    template<size_t index, typename Ret_, typename... Entries_, typename F_, typename... Fs_>
    friend Ret_ match_(tagged_variant<Entries_...>& var, F_ f, Fs_... fs);

    template<tag_type tag, typename... Args>
    requires tag_occurs<tag, Entries...>
    static tagged_variant make(Args&&... value) {
        return tagged_variant( std::in_place_index<tag_index_v<tag, Entries...>>,
            std::forward<Args>(value)...
        );
    }

private:
    template<size_t I, typename... Args>
    explicit tagged_variant(std::in_place_index_t<I> tag, Args&&... args)
    : variant_(tag, std::forward<Args>(args)...) {}
    raw_variant variant_;
};

template<Enum auto tag, typename... Entries>
requires tag_occurs<tag, Entries...>
auto& get(tagged_variant<Entries...>& var) {
    return std::get<tag_index_v<tag, Entries...>>(var.variant_);
}

template<Enum auto tag, typename... Entries>
requires tag_occurs<tag, Entries...>
const auto& get(const tagged_variant<Entries...>& var) {
    return std::get<tag_index_v<tag, Entries...>>(var.variant_);
}

// TESTING:

enum class test {
    Circle, Rectangle, Triangle
};

template<size_t index, typename Ret, typename... Entries, typename F>
Ret match_(tagged_variant<Entries...>& var, F f) {
    if (var.variant_.index() == index) {
        return f(std::get<index>(var.variant_));
    }
    throw std::bad_variant_access{};
}

template<size_t index, typename Ret, typename... Entries, typename F, typename... Fs>
Ret match_(tagged_variant<Entries...>& var, F f, Fs... fs) {
    if (var.variant_.index() == index) {
        return f(std::get<index>(var.variant_));
    }
    return match_<index + 1, Ret>(var, fs...);
}

template<typename... Entries, typename... Fs>
requires entries_match_functions<impl_t_var::type_vector<Entries...>, impl_t_var::type_vector<Fs...>>
&& requires {typename output_type_t< impl_t_var::type_vector<Entries...>, impl_t_var::type_vector<Fs...>>;}

auto match(tagged_variant<Entries...>& var, Fs... fs) {
    using Ret = output_type_t<impl_t_var::type_vector<Entries...>, impl_t_var::type_vector<Fs...>>;
    return match_<0, Ret>(var, fs...);
}

inline void test_fn() {
    using enum test;
    using shape = tagged_variant<
        entry<Circle, double>,
        entry<Rectangle, std::pair<double, double>>,
        entry<Triangle, std::tuple<double, double, double>>
    >;

    static_assert(std::same_as<shape::raw_variant,
        std::variant<double, std::pair<double,double>, std::tuple<double, double, double>>>);
    static_assert(std::same_as<shape::tag_type, test>);

    shape x = shape::make<Rectangle>(1, 5);
    shape y = shape::make<Circle>(1);

    std::pair<double, double> dimensions = get<Rectangle>(x);
    double radius = get<Circle>(y);

    std::println("Rectangle dimensions: {} x {}", dimensions.first, dimensions.second);
    std::println("circle radius: {}", radius);

    std::variant<std::string, double> h {"Hello world!"};

    auto z = match(x,
        [] (double& v) {
            return "Circle";
        },
        [](std::pair<double, double>& v) {
            return "Rectangle";
        },
        [](std::tuple<double, double, double>& v) {
            return "Triangle";
        }
    );
    std::puts(z);
}


#endif //TAGGED_VARIANT_H
