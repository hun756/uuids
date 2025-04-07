// generate inlude guard with random 6 char engin UUIDV4_HPP_[random6char]
#ifndef UUIDV4_HPP_xir2zk
#define UUIDV4_HPP_xir2zk

#include <array>
#include <bit>
#include <concepts>
#include <span>

namespace uuids::inline v1
{
template <typename T>
concept RandomNumberEngine = requires(T& eng) {
    typename T::result_type;
    { eng() } -> std::convertible_to<typename T::result_type>;
    { T::min() } -> std::same_as<typename T::result_type>;
    { T::max() } -> std::same_as<typename T::result_type>;
};

inline constexpr bool is_little_endian =
    std::endian::native == std::endian::little;

struct alignas(16) uuid_bytes final
{
    std::array<std::uint8_t, 16> data;

    constexpr uuid_bytes() noexcept : data{} {}
    constexpr explicit uuid_bytes(
        const std::array<std::uint8_t, 16>& bytes) noexcept
        : data{bytes}
    {
    }
    constexpr explicit uuid_bytes(
        std::span<const std::uint8_t, 16> bytes) noexcept
    {
        std::copy(bytes.begin(), bytes.end(), data.begin());
    }
};
} // namespace uuids::inline v1

#endif /* End of include guard: UUIDV4_HPP_xir2zk */