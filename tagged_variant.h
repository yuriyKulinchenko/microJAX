#ifndef TAGGED_VARIANT_H
#define TAGGED_VARIANT_H
#include <type_traits>

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

}

template<typename... Entries>
concept only_entries = (impl_t_var::is_entry<Entries>::value && ...);

template<typename... Entries>
concept safe_entries = impl_t_var::safe_entries_class<Entries...>::value;

template<typename... Entries>
concept duplicate_tags = impl_t_var::duplicate_tags_class<Entries...>::value;

template<typename... Entries>
using entries_to_variant = typename impl_t_var::vector_to_variant_class<
    typename impl_t_var::entries_to_vector<Entries...>::vector>::type;

template<typename... Entries>
using tag_type_v = typename impl_t_var::tag_type_class<Entries...>::type;

// TODO: Implement Compatible_tag properly
// template<typename T, Enum auto tag, typename... Entries>
// concept compatible_tag = true;

template<Enum auto tag, typename... Entries>
static constexpr size_t tag_index_v = impl_t_var::find_tag_index_class<tag, 0, Entries...>::value;

template<typename... Entries>
requires only_entries<Entries...> && safe_entries<Entries...> && (!duplicate_tags<Entries...>)
class tagged_variant {
public:
    using raw_variant = entries_to_variant<Entries...>;
    using tag_type = tag_type_v<Entries...>;

    // I need to think about the constructor:
    // For now, consider the simple case of by-value, with the tag explicitly provided:

    // Need to be able to fetch the associated value
    // Ought to be associated with a given tag, explicitly

    template<Enum auto tag, typename T>
    static tagged_variant make(T&& value) {
        return tagged_variant(raw_variant{std::in_place_index<tag_index_v<tag, Entries...>>, std::forward<T>(value)});
    }

private:
    explicit tagged_variant(raw_variant variant_): variant_(variant_) {}
    raw_variant variant_;
};


// TESTING:

enum class test {
    Circle, Rectangle
};

enum class test_other {
    Some, None
};

inline void test_fn() {
    using enum test;
    using shape = tagged_variant<entry<Circle, double>, entry<Rectangle, std::pair<double, double>>>;
    static_assert(std::same_as<shape::raw_variant, std::variant<double, std::pair<double,double>>>);
    static_assert(std::same_as<shape::tag_type, test>);
}


#endif //TAGGED_VARIANT_H
