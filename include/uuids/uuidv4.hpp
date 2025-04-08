// generate inlude guard with random 6 char engin UUIDV4_HPP_[random6char]
#ifndef UUIDV4_HPP_xir2zk
#define UUIDV4_HPP_xir2zk

#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <random>
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

template <typename PRNG = std::mt19937_64>
    requires RandomNumberEngine<PRNG>
class optimized_generator final
{
public:
    using result_type = uuid_bytes;

    optimized_generator() noexcept : rng_(std::random_device{}()), use_hw_rng_(setup_hw_rng()) {}

    explicit optimized_generator(typename PRNG::result_type seed) noexcept
        : rng_(seed), use_hw_rng_(setup_hw_rng())
    {
    }

    [[nodiscard]] result_type operator()() noexcept
    {
        return use_hw_rng_ ? generate_hw() : generate_sw();
    }

private:
    [[nodiscard]] static bool setup_hw_rng() noexcept
    {
        return simd::has_feature(simd::Feature::RDRND) || simd::has_feature(simd::Feature::RDSEED);
    }

    [[nodiscard]] result_type generate_hw() noexcept
    {
        uuid_bytes uuid;
        std::span<std::uint8_t, 16> uuid_span(uuid.data);

        bool used_hw_rng = false;

        if (hardware_rng::rdrand_supported())
        {
            std::uint64_t v1 = hardware_rng::rdrand();
            std::uint64_t v2 = hardware_rng::rdrand();

            if (v1 != 0 || v2 != 0)
            {
                std::memcpy(uuid_span.data(), &v1, 8);
                std::memcpy(uuid_span.subspan(8).data(), &v2, 8);
                used_hw_rng = true;
            }
        }

        if (!used_hw_rng && hardware_rng::rdseed_supported())
        {
            std::uint64_t v1 = hardware_rng::rdseed();
            std::uint64_t v2 = hardware_rng::rdseed();

            if (v1 != 0 || v2 != 0)
            {
                std::memcpy(uuid_span.data(), &v1, 8);
                std::memcpy(uuid_span.subspan(8).data(), &v2, 8);
                used_hw_rng = true;
            }
        }

        if (!used_hw_rng)
        {
            return generate_sw();
        }

#if defined(__x86_64__) || defined(_M_X64)
        if (hardware_rng::aesni_supported() && simd::compile_time::has<simd::Feature::AES>())
        {
            __m128i data = _mm_load_si128(reinterpret_cast<const __m128i*>(uuid.data.data()));
            __m128i key = _mm_set_epi64x(0x1b873593, 0x9e3779b9);
            data = hardware_rng::aesni_enc(key, data);
            _mm_store_si128(reinterpret_cast<__m128i*>(uuid.data.data()), data);
        }
#endif

        uuid.data[6] = (uuid.data[6] & 0x0F) | 0x40;
        uuid.data[8] = (uuid.data[8] & 0x3F) | 0x80;

        return uuid;
    }

    [[nodiscard]] result_type generate_sw() noexcept
    {
        uuid_bytes uuid;
        if constexpr (sizeof(typename PRNG::result_type) == 8)
        {
            const auto v1 = rng_();
            const auto v2 = rng_();
            std::span<std::uint8_t, 16> uuid_span(uuid.data);
            std::memcpy(uuid_span.data(), &v1, 8);
            std::memcpy(uuid_span.subspan(8).data(), &v2, 8);
        }
        else
        {
            for (std::size_t i = 0; i < 16; i += sizeof(typename PRNG::result_type))
            {
                const auto value = rng_();
                std::memcpy(uuid.data.data() + i, &value, sizeof(typename PRNG::result_type));
            }
        }

        uuid.data[6] = (uuid.data[6] & 0x0F) | 0x40;
        uuid.data[8] = (uuid.data[8] & 0x3F) | 0x80;

        return uuid;
    }

    PRNG rng_;
    bool use_hw_rng_;

    static_assert(std::is_trivially_copyable_v<PRNG>);
    static_assert(std::is_trivially_destructible_v<PRNG>);
};

} // namespace detail

template <typename PRNG = std::mt19937_64>
class basic_uuid final
{
public:
    using bytes_type = std::array<std::uint8_t, 16>;

    static constexpr std::size_t size() noexcept { return 16; }

    constexpr basic_uuid() noexcept = default;

    explicit constexpr basic_uuid(const bytes_type& bytes) noexcept : data_{bytes} {}

    explicit constexpr basic_uuid(std::span<const std::uint8_t, 16> bytes) noexcept : data_{bytes}
    {
    }

    explicit constexpr basic_uuid(detail::uuid_bytes bytes) noexcept : data_{std::move(bytes)} {}

    [[nodiscard]] constexpr const bytes_type& bytes() const noexcept { return data_.data; }

    [[nodiscard]] constexpr std::span<const std::uint8_t, 16> span() const noexcept
    {
        return std::span<const std::uint8_t, 16>(data_.data);
    }

    [[nodiscard]] std::string str() const
    {
        static constexpr std::string_view hex = "0123456789abcdef";
        std::string result(36, '\0');

        std::size_t i = 0;
        for (std::size_t j = 0; j < 16; ++j)
        {
            if (j == 4 || j == 6 || j == 8 || j == 10)
            {
                result[i++] = '-';
            }
            result[i++] = hex[data_.data[j] >> 4];
            result[i++] = hex[data_.data[j] & 0x0F];
        }

        return result;
    }

    [[nodiscard]] constexpr auto operator<=>(const basic_uuid&) const noexcept = default;

private:
    detail::uuid_bytes data_{};
};

template <typename PRNG = std::mt19937_64>
class basic_uuid_generator final
{
public:
    using uuid_type = basic_uuid<PRNG>;
    using result_type = detail::uuid_bytes;

    constexpr basic_uuid_generator() noexcept = default;

    explicit constexpr basic_uuid_generator(typename PRNG::result_type seed) noexcept : gen_(seed)
    {
    }

    [[nodiscard]] uuid_type operator()() noexcept { return uuid_type(gen_()); }

private:
    detail::optimized_generator<PRNG> gen_;
};

using uuid = basic_uuid<>;
using uuid_generator = basic_uuid_generator<>;

template <typename CharT, typename Traits, typename PRNG>
inline std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                                     const basic_uuid<PRNG>& uuid)
{
    return os << uuid.str();
}

} // namespace uuids::inline v1

#endif /* End of include guard: UUIDV4_HPP_xir2zk */