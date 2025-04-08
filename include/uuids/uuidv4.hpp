// generate inlude guard with random 6 char engin UUIDV4_HPP_[random6char]
#ifndef UUIDV4_HPP_xir2zk
#define UUIDV4_HPP_xir2zk

#include <array>
#include <bit>
#include <concepts>
#include <span>

#include <simd/feature_check.hpp>

#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace uuids::inline v1
{

namespace detail
{

template <typename T>
concept RandomNumberEngine = requires(T& eng) {
    typename T::result_type;
    { eng() } -> std::convertible_to<typename T::result_type>;
    { T::min() } -> std::same_as<typename T::result_type>;
    { T::max() } -> std::same_as<typename T::result_type>;
};

inline constexpr bool is_little_endian = std::endian::native == std::endian::little;

struct alignas(16) uuid_bytes final
{
    std::array<std::uint8_t, 16> data;

    constexpr uuid_bytes() noexcept : data{} {}

    constexpr explicit uuid_bytes(const std::array<std::uint8_t, 16>& bytes) noexcept : data{bytes}
    {
    }

    constexpr explicit uuid_bytes(std::span<const std::uint8_t, 16> bytes) noexcept
    {
        std::copy(bytes.begin(), bytes.end(), data.begin());
    }
};

class hardware_rng final
{
public:
    [[nodiscard]] static bool rdrand_supported() noexcept
    {
        static const bool supported = simd::has_feature(simd::Feature::RDRND);
        return supported;
    }

    [[nodiscard]] static bool rdseed_supported() noexcept
    {
        static const bool supported = simd::has_feature(simd::Feature::RDSEED);
        return supported;
    }

    [[nodiscard]] static bool aesni_supported() noexcept
    {
        static const bool supported = simd::has_feature(simd::Feature::AES);
        return supported;
    }

    [[nodiscard]] static std::uint64_t rdrand() noexcept
    {
        if (!rdrand_supported())
        {
            return 0;
        }

#if defined(__x86_64__) || defined(_M_X64)
        std::uint64_t value = 0;

        if (simd::compile_time::has<simd::Feature::RDRND>())
        {
#ifdef _MSC_VER
            _rdrand64_step(&value);
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_ia32_rdrand64_step(&value);
#endif
        }

        return value;
#else
        return 0;
#endif
    }

    [[nodiscard]] static std::uint64_t rdseed() noexcept
    {
        if (!rdseed_supported())
        {
            return 0;
        }

#if defined(__x86_64__) || defined(_M_X64)
        std::uint64_t value = 0;

        if (simd::compile_time::has<simd::Feature::RDSEED>())
        {
#ifdef _MSC_VER
            _rdseed64_step(&value);
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_ia32_rdseed64_step(&value);
#endif
        }

        return value;
#else
        return 0;
#endif
    }

    [[nodiscard]] static __m128i aesni_enc(__m128i key, __m128i data) noexcept
    {
        if (!aesni_supported())
        {
            return data;
        }

#if defined(__x86_64__) || defined(_M_X64)
        if (simd::compile_time::has<simd::Feature::AES>())
        {
            return _mm_aesenc_si128(data, key);
        }
#endif
        return data;
    }
};

} // namespace detail
} // namespace uuids::inline v1

#endif /* End of include guard: UUIDV4_HPP_xir2zk */