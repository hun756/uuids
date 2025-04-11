#ifndef SIMD_HPP_al9nn6
#define SIMD_HPP_al9nn6

#include <array>
#include <bit>
#include <cmath>
#include <simd/common.hpp>
#include <simd/feature_check.hpp>

namespace vector_simd
{

constexpr int kVersionMajor = 0;
constexpr int kVersionMinor = 1;
constexpr int kVersionPatch = 0;

constexpr size_t kSSEAlignment = 16;
constexpr size_t kAVXAlignment = 32;
constexpr size_t kAVX512Alignment = 64;

constexpr size_t kDefaultAlignment =
    simd::compile_time::has<simd::Feature::AVX512F>() ? kAVX512Alignment
    : simd::compile_time::has<simd::Feature::AVX>()   ? kAVXAlignment
                                                      : kSSEAlignment;

template <typename T>
concept SimdArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int8_t> ||
    std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, uint64_t>;

template <typename T>
concept SimdFloat = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept SimdInteger = SimdArithmetic<T> && !SimdFloat<T>;

template <typename T>
concept SimdSignedInteger = std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                            std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

template <typename T>
concept SimdUnsignedInteger = std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                              std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

// Forward declarations
template <SimdArithmetic T, size_t N>
class Vector;

template <SimdArithmetic T, size_t N>
class Mask;

namespace detail
{

struct generic_tag
{
};

struct sse2_tag : generic_tag
{
};

struct sse3_tag : sse2_tag
{
};

struct ssse3_tag : sse3_tag
{
};

struct sse4_1_tag : ssse3_tag
{
};

struct sse4_2_tag : sse4_1_tag
{
};

struct avx_tag : sse4_2_tag
{
};

struct avx2_tag : avx_tag
{
};

struct avx512_tag : avx2_tag
{
};

struct neon_tag : generic_tag
{
};

struct wasm_simd_tag : generic_tag
{
};

struct best_available_tag
{
    using type = std::conditional_t<
        simd::compile_time::has<simd::Feature::AVX512F>(), avx512_tag,
        std::conditional_t<
            simd::compile_time::has<simd::Feature::AVX2>(), avx2_tag,
            std::conditional_t<
                simd::compile_time::has<simd::Feature::AVX>(), avx_tag,
                std::conditional_t<
                    simd::compile_time::has<simd::Feature::SSE42>(), sse4_2_tag,
                    std::conditional_t<
                        simd::compile_time::has<simd::Feature::SSE41>(), sse4_1_tag,
                        std::conditional_t<
                            simd::compile_time::has<simd::Feature::SSSE3>(), ssse3_tag,
                            std::conditional_t<
                                simd::compile_time::has<simd::Feature::SSE3>(), sse3_tag,
                                std::conditional_t<simd::compile_time::has<simd::Feature::SSE2>(),
                                                   sse2_tag, generic_tag
                                >
                            >
                        >
                    >
                >
            >
        >
    >;
};

using current_isa = typename best_available_tag::type;

template <typename T, typename ISA>
struct simd_width;

template <>
struct simd_width<float, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<double, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<int8_t, sse2_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint8_t, sse2_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int16_t, sse2_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint16_t, sse2_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int32_t, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint32_t, sse2_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int64_t, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<uint64_t, sse2_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<float, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<double, avx_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int8_t, avx_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<uint8_t, avx_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<int16_t, avx_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint16_t, avx_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int32_t, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint32_t, avx_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int64_t, avx_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint64_t, avx_tag>
{
    static constexpr size_t value = 4;
};

template <typename T>
struct simd_width<T, avx2_tag> : simd_width<T, avx_tag>
{
};

template <>
struct simd_width<float, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<double, avx512_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int8_t, avx512_tag>
{
    static constexpr size_t value = 64;
};

template <>
struct simd_width<uint8_t, avx512_tag>
{
    static constexpr size_t value = 64;
};

template <>
struct simd_width<int16_t, avx512_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<uint16_t, avx512_tag>
{
    static constexpr size_t value = 32;
};

template <>
struct simd_width<int32_t, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint32_t, avx512_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int64_t, avx512_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint64_t, avx512_tag>
{
    static constexpr size_t value = 8;
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template <>
struct simd_width<float, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int8_t, neon_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<uint8_t, neon_tag>
{
    static constexpr size_t value = 16;
};

template <>
struct simd_width<int16_t, neon_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<uint16_t, neon_tag>
{
    static constexpr size_t value = 8;
};

template <>
struct simd_width<int32_t, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<uint32_t, neon_tag>
{
    static constexpr size_t value = 4;
};

template <>
struct simd_width<int64_t, neon_tag>
{
    static constexpr size_t value = 2;
};

template <>
struct simd_width<uint64_t, neon_tag>
{
    static constexpr size_t value = 2;
};

#endif

// Generic fallback for unsupported architectures

template <typename T>
struct simd_width<T, generic_tag>
{
    static constexpr size_t value = 1;
};

template <typename T>
struct native_width
{
    static constexpr size_t value = simd_width<T, current_isa>::value;
};

template <typename T, typename ISA>
struct register_type;

template <>
struct register_type<float, sse2_tag>
{
    using type = __m128;
};

template <>
struct register_type<double, sse2_tag>
{
    using type = __m128d;
};

template <>
struct register_type<int8_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint8_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int16_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint16_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int32_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint32_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<int64_t, sse2_tag>
{
    using type = __m128i;
};

template <>
struct register_type<uint64_t, sse2_tag>
{
    using type = __m128i;
};

// AVX register type mappings

template <>
struct register_type<float, avx_tag>
{
    using type = __m256;
};

template <>
struct register_type<double, avx_tag>
{
    using type = __m256d;
};

template <>
struct register_type<int8_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint8_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int16_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint16_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int32_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint32_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<int64_t, avx_tag>
{
    using type = __m256i;
};

template <>
struct register_type<uint64_t, avx_tag>
{
    using type = __m256i;
};

// Same register types for AVX2

template <typename T>
struct register_type<T, avx2_tag> : register_type<T, avx_tag>
{
};

#ifdef __AVX512F__

template <>
struct register_type<float, avx512_tag>
{
    using type = __m512;
};

template <>
struct register_type<double, avx512_tag>
{
    using type = __m512d;
};

template <>
struct register_type<int8_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint8_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int16_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint16_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int32_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint32_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<int64_t, avx512_tag>
{
    using type = __m512i;
};

template <>
struct register_type<uint64_t, avx512_tag>
{
    using type = __m512i;
};

#endif

// NEON register type mappings (if needed)

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

template <>
struct register_type<float, neon_tag>
{
    using type = float32x4_t;
};

template <>
struct register_type<int8_t, neon_tag>
{
    using type = int8x16_t;
};

template <>
struct register_type<uint8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct register_type<int16_t, neon_tag>
{
    using type = int16x8_t;
};

template <>
struct register_type<uint16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct register_type<int32_t, neon_tag>
{
    using type = int32x4_t;
};

template <>
struct register_type<uint32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct register_type<int64_t, neon_tag>
{
    using type = int64x2_t;
};

template <>
struct register_type<uint64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct register_type<double, neon_tag>
{
#ifdef __aarch64__
    using type = float64x2_t;
#else
    using type = double;
#endif
};

#endif

template <typename T>
struct register_type<T, generic_tag>
{
    using type = T;
};

template <typename T, typename ISA>
struct mask_register_type;

template <typename T>
struct mask_register_type<T, sse2_tag>
{
    using type = typename register_type<T, sse2_tag>::type;
};

template <typename T>
struct mask_register_type<T, avx_tag>
{
    using type = typename register_type<T, avx_tag>::type;
};

template <typename T>
struct mask_register_type<T, avx2_tag> : mask_register_type<T, avx_tag>
{
};

#ifdef __AVX512F__

template <>
struct mask_register_type<float, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct mask_register_type<int8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct mask_register_type<uint8_t, neon_tag>
{
    using type = uint8x16_t;
};

template <>
struct mask_register_type<int16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct mask_register_type<uint16_t, neon_tag>
{
    using type = uint16x8_t;
};

template <>
struct mask_register_type<int32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct mask_register_type<uint32_t, neon_tag>
{
    using type = uint32x4_t;
};

template <>
struct mask_register_type<int64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct mask_register_type<uint64_t, neon_tag>
{
    using type = uint64x2_t;
};

template <>
struct mask_register_type<double, neon_tag>
{
#ifdef __aarch64__
    using type = uint64x2_t;
#else
    using type = bool;
#endif
};

#endif

template <typename T>
struct mask_register_type<T, generic_tag>
{
    using type = bool;
};

// forward declarations for vector and mask operations

template <typename T, size_t N, typename ISA = current_isa>
struct vector_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct mask_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct memory_ops;

template <typename T, size_t N, typename ISA = current_isa>
struct math_ops;

template <typename Derived, typename T, size_t N>
class vector_base
{
protected:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

public:
    using value_type = T;
    using size_type = size_t;
    static constexpr size_t size_value = N;

    SIMD_INLINE T operator[](size_t i) const
    {
        assert(i < N && "Index out of bounds");
        return derived().extract(i);
    }

    class reference
    {
    private:
        Derived& vec;
        size_t idx;

    public:
        reference(Derived& v, size_t i) : vec(v), idx(i) {}

        operator T() const { return vec.extract(idx); }

        reference& operator=(T value)
        {
            vec.insert(idx, value);
            return *this;
        }

        reference& operator+=(T value)
        {
            vec.insert(idx, vec.extract(idx) + value);
            return *this;
        }

        reference& operator-=(T value)
        {
            vec.insert(idx, vec.extract(idx) - value);
            return *this;
        }

        reference& operator*=(T value)
        {
            vec.insert(idx, vec.extract(idx) * value);
            return *this;
        }

        reference& operator/=(T value)
        {
            vec.insert(idx, vec.extract(idx) / value);
            return *this;
        }
    };

    SIMD_INLINE reference operator[](size_t i)
    {
        assert(i < N && "Index out of bounds");
        return reference(derived(), i);
    }

    auto _data() { return derived().data(); }
    auto _data() const { return derived().data(); }
};

} // namespace detail

template <SimdArithmetic T, size_t N>
class alignas(kDefaultAlignment) Vector : public detail::vector_base<Vector<T, N>, T, N>
{
private:
    using ops = detail::vector_ops<T, N>;
    using m_ops = detail::mask_ops<T, N>;
    using mem_ops = detail::memory_ops<T, N>;
    using math = detail::math_ops<T, N>;

    static constexpr size_t native_width = detail::native_width<T>::value;
    static constexpr size_t storage_size = (N + native_width - 1) / native_width * native_width;

    using register_t = typename detail::register_type<T, detail::current_isa>::type;
    static constexpr size_t num_registers =
        (storage_size + detail::simd_width<T, detail::current_isa>::value - 1) /
        detail::simd_width<T, detail::current_isa>::value;

    alignas(kDefaultAlignment) std::array<register_t, num_registers> registers;

public:
    using value_type = T;
    using mask_type = Mask<T, N>;
    using size_type = size_t;
    static constexpr size_t size_value = N;

    Vector() = default;

    explicit Vector(T value) { ops::set1(registers.data(), value); }

    explicit Vector(const T* ptr) { mem_ops::load(registers.data(), ptr); }

    Vector(const Vector&) = default;
    Vector(Vector&&) = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) = default;

    Vector(std::initializer_list<T> values)
    {
        std::array<T, storage_size> tmp{};
        size_t count = std::min(values.size(), storage_size);
        std::copy_n(values.begin(), count, tmp.begin());
        mem_ops::load(registers.data(), tmp.data());
    }

    register_t* data() { return registers.data(); }
    const register_t* data() const { return registers.data(); }

    SIMD_INLINE T extract(size_t i) const
    {
        assert(i < N && "Index out of bounds");
        return ops::extract(registers.data(), i);
    }

    SIMD_INLINE void insert(size_t i, T value)
    {
        assert(i < N && "Index out of bounds");
        ops::insert(registers.data(), i, value);
    }

    static Vector load(const T* ptr)
    {
        Vector result;
        mem_ops::load(result.data(), ptr);
        return result;
    }

    static Vector load_aligned(const T* ptr)
    {
        Vector result;
        mem_ops::load_aligned(result.data(), ptr);
        return result;
    }

    static Vector load_unaligned(const T* ptr)
    {
        Vector result;
        mem_ops::load_unaligned(result.data(), ptr);
        return result;
    }

    void store(T* ptr) const { mem_ops::store(registers.data(), ptr); }

    void store_aligned(T* ptr) const { mem_ops::store_aligned(registers.data(), ptr); }

    void store_unaligned(T* ptr) const { mem_ops::store_unaligned(registers.data(), ptr); }

    std::array<T, N> to_array() const
    {
        std::array<T, N> result;
        store(result.data());
        return result;
    }

    template <typename IndexT>
    static Vector gather(const T* base, const Vector<IndexT, N>& indices)
    {
        Vector result;
        mem_ops::gather(result.data(), base, indices.data());
        return result;
    }

    template <typename IndexT>
    void scatter(T* base, const Vector<IndexT, N>& indices) const
    {
        mem_ops::scatter(registers.data(), base, indices.data());
    }

    Vector operator+(const Vector& rhs) const
    {
        Vector result;
        ops::add(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator-(const Vector& rhs) const
    {
        Vector result;
        ops::sub(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator*(const Vector& rhs) const
    {
        Vector result;
        ops::mul(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator/(const Vector& rhs) const
    {
        Vector result;
        ops::div(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector& operator+=(const Vector& rhs)
    {
        ops::add(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator-=(const Vector& rhs)
    {
        ops::sub(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator*=(const Vector& rhs)
    {
        ops::mul(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator/=(const Vector& rhs)
    {
        ops::div(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector operator&(const Vector& rhs) const
    {
        Vector result;
        ops::bitwise_and(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator|(const Vector& rhs) const
    {
        Vector result;
        ops::bitwise_or(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator^(const Vector& rhs) const
    {
        Vector result;
        ops::bitwise_xor(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector operator~() const
    {
        Vector result;
        ops::bitwise_not(result.data(), registers.data());
        return result;
    }

    Vector& operator&=(const Vector& rhs)
    {
        ops::bitwise_and(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator|=(const Vector& rhs)
    {
        ops::bitwise_or(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    Vector& operator^=(const Vector& rhs)
    {
        ops::bitwise_xor(registers.data(), registers.data(), rhs.data());
        return *this;
    }

    mask_type operator==(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_eq(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator!=(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_neq(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator<(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_lt(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator<=(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_le(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator>(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_gt(result._data(), registers.data(), rhs.data());
        return result;
    }

    mask_type operator>=(const Vector& rhs) const
    {
        mask_type result;
        m_ops::cmp_ge(result._data(), registers.data(), rhs.data());
        return result;
    }

    Vector abs() const
    {
        Vector result;
        math::abs(result.data(), registers.data());
        return result;
    }

    Vector sqrt() const
    {
        Vector result;
        math::sqrt(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector sin() const
    {
        Vector result;
        math::sin(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector cos() const
    {
        Vector result;
        math::cos(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector tan() const
    {
        Vector result;
        math::tan(result.data(), registers.data());
        return result;
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vector exp() const
    {
        Vector result;
        math::exp(result.data(), registers.data());
        return result;
    }

    Vector log() const
    {
        Vector result;
        math::log(result.data(), registers.data());
        return result;
    }

    Vector min(const Vector& rhs) const
    {
        Vector result;
        ops::min(result.data(), registers.data(), rhs.data());
        return result;
    }

    Vector max(const Vector& rhs) const
    {
        Vector result;
        ops::max(result.data(), registers.data(), rhs.data());
        return result;
    }

    T hsum() const { return ops::horizontal_sum(registers.data()); }
    
    T hmin() const { return ops::horizontal_min(registers.data()); }

    T hmax() const { return ops::horizontal_max(registers.data()); }

    Vector shuffle(const std::array<int, N>& indices) const
    {
        Vector result;
        ops::shuffle(result.data(), registers.data(), indices.data());
        return result;
    }

    Vector blend(const Vector& rhs, const mask_type& mask) const
    {
        Vector result;
        ops::blend(result.data(), registers.data(), rhs.data(), mask._data());
        return result;
    }

    static Vector select(const mask_type& mask, const Vector& a, const Vector& b)
    {
        Vector result;
        ops::select(result.data(), mask._data(), a.data(), b.data());
        return result;
    }

    Vector fmadd(const Vector& a, const Vector& b) const
    {
        Vector result;
        math::fmadd(result.data(), registers.data(), a.data(), b.data());
        return result;
    }

    Vector fmsub(const Vector& a, const Vector& b) const
    {
        Vector result;
        math::fmsub(result.data(), registers.data(), a.data(), b.data());
        return result;
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<T, U>, int> = 0>
    Vector<U, N> convert() const
    {
        Vector<U, N> result;
        ops::convert(result.data(), registers.data());
        return result;
    }

    static void prefetch(const T* ptr, int hint = 0) { mem_ops::prefetch(ptr, hint); }
};

template <SimdArithmetic T, size_t N>
class alignas(kDefaultAlignment) Mask
{
private:
    using m_ops = detail::mask_ops<T, N>;
    friend class Vector<T, N>;

    using mask_register_t = typename detail::mask_register_type<T, detail::current_isa>::type;
    static constexpr size_t native_width = detail::native_width<T>::value;
    static constexpr size_t storage_size = (N + native_width - 1) / native_width * native_width;
    static constexpr size_t num_registers =
        (storage_size + detail::simd_width<T, detail::current_isa>::value - 1) /
        detail::simd_width<T, detail::current_isa>::value;

    alignas(kDefaultAlignment) std::array<mask_register_t, num_registers> registers;

public:
    using value_type = bool;
    using size_type = size_t;
    static constexpr size_t size_value = N;

    Mask() { m_ops::set_false(registers.data()); }

    explicit Mask(bool value)
    {
        if (value)
        {
            m_ops::set_true(registers.data());
        }
        else
        {
            m_ops::set_false(registers.data());
        }
    }

    explicit Mask(const bool* ptr) { m_ops::load(registers.data(), ptr); }

    Mask(const Mask&) = default;
    Mask(Mask&&) = default;
    Mask& operator=(const Mask&) = default;
    Mask& operator=(Mask&&) = default;

    mask_register_t* _data() { return registers.data(); }
    const mask_register_t* _data() const { return registers.data(); }

    SIMD_INLINE bool operator[](size_t i) const
    {
        assert(i < N && "Index out of bounds");
        return m_ops::extract(registers.data(), i);
    }

    Mask operator&(const Mask& rhs) const
    {
        Mask result;
        m_ops::logical_and(result._data(), registers.data(), rhs._data());
        return result;
    }

    Mask operator|(const Mask& rhs) const
    {
        Mask result;
        m_ops::logical_or(result._data(), registers.data(), rhs._data());
        return result;
    }

    Mask operator^(const Mask& rhs) const
    {
        Mask result;
        m_ops::logical_xor(result._data(), registers.data(), rhs._data());
        return result;
    }

    Mask operator~() const
    {
        Mask result;
        m_ops::logical_not(result._data(), registers.data());
        return result;
    }

    Mask& operator&=(const Mask& rhs)
    {
        m_ops::logical_and(registers.data(), registers.data(), rhs._data());
        return *this;
    }

    Mask& operator|=(const Mask& rhs)
    {
        m_ops::logical_or(registers.data(), registers.data(), rhs._data());
        return *this;
    }

    Mask& operator^=(const Mask& rhs)
    {
        m_ops::logical_xor(registers.data(), registers.data(), rhs._data());
        return *this;
    }

    uint64_t to_bitmask() const { return m_ops::to_bitmask(registers.data()); }

    bool any() const { return m_ops::any(registers.data()); }

    bool all() const { return m_ops::all(registers.data()); }

    bool none() const { return !any(); }

    int count() const { return m_ops::count(registers.data()); }

    void store(bool* ptr) const { m_ops::store(registers.data(), ptr); }

    std::array<bool, N> to_array() const
    {
        std::array<bool, N> result;
        store(result.data());
        return result;
    }
};

// Type aliases for common vector types
// Fixed-size vector types
template <size_t N>
using float_v = Vector<float, N>;
template <size_t N>
using double_v = Vector<double, N>;
template <size_t N>
using int8_v = Vector<int8_t, N>;
template <size_t N>
using uint8_v = Vector<uint8_t, N>;
template <size_t N>
using int16_v = Vector<int16_t, N>;
template <size_t N>
using uint16_v = Vector<uint16_t, N>;
template <size_t N>
using int32_v = Vector<int32_t, N>;
template <size_t N>
using uint32_v = Vector<uint32_t, N>;
template <size_t N>
using int64_v = Vector<int64_t, N>;
template <size_t N>
using uint64_v = Vector<uint64_t, N>;

// Native-width vector types (width depends on best available instruction set)
using float_vn = float_v<detail::native_width<float>::value>;
using double_vn = double_v<detail::native_width<double>::value>;
using int8_vn = int8_v<detail::native_width<int8_t>::value>;
using uint8_vn = uint8_v<detail::native_width<uint8_t>::value>;
using int16_vn = int16_v<detail::native_width<int16_t>::value>;
using uint16_vn = uint16_v<detail::native_width<uint16_t>::value>;
using int32_vn = int32_v<detail::native_width<int32_t>::value>;
using uint32_vn = uint32_v<detail::native_width<uint32_t>::value>;
using int64_vn = int64_v<detail::native_width<int64_t>::value>;
using uint64_vn = uint64_v<detail::native_width<uint64_t>::value>;

#if SIMD_ARCH_X86 && SIMD_HAS_SSE2

namespace detail
{

template <typename T, size_t N>
struct vector_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_set1_ps(value);
        }

        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_set1_pd(value);
        }

        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_set1_epi8(static_cast<char>(value));
        }

        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_set1_epi16(static_cast<short>(value));
        }

        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_set1_epi32(static_cast<int>(value));
        }

        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_set1_epi64x(static_cast<long long>(value));
        }
    }

    static SIMD_INLINE T extract(const register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            return tmp[index % 4];
        }

        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            return tmp[index % 2];
        }

        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            return static_cast<int8_t>(_mm_extract_epi8(*src, index % 16));
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 16];
#endif
        }

        else if constexpr (std::is_same_v<T, uint8_t>)
        {
#if SIMD_SSE4_1
            return static_cast<uint8_t>(_mm_extract_epi8(*src, index % 16));
#else
            alignas(16) uint8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 16];
#endif
        }

        else if constexpr (std::is_same_v<T, int16_t>)
        {
            alignas(16) int16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 8];
        }

        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            alignas(16) uint16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 8];
        }

        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            return static_cast<int32_t>(_mm_extract_epi32(*src, index % 4));
#else
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }

        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            return static_cast<uint32_t>(_mm_extract_epi32(*src, index % 4));
#else
            alignas(16) uint32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 4];
#endif
        }

        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_SSE4_1
            return static_cast<T>(_mm_extract_epi64(*src, index % 2));
#else
            alignas(16) T tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % 2];
#endif
        }
    }

    static SIMD_INLINE void insert(register_t* dst, size_t index, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *dst);
            tmp[index % 4] = value;
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *dst);
            tmp[index % 2] = value;
            *dst = _mm_load_pd(tmp);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_insert_epi8(*dst, static_cast<int>(value), index % 16);
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 16] = value;
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            alignas(16) int16_t tmp[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 8] = static_cast<int16_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_insert_epi32(*dst, static_cast<int>(value), index % 4);
#else
            alignas(16) int32_t tmp[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 4] = static_cast<int32_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_insert_epi64(*dst, static_cast<long long>(value), index % 2);
#else
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *dst);
            tmp[index % 2] = static_cast<int64_t>(value);
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
    }

    static SIMD_INLINE void add(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_add_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_add_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_add_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_add_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_add_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_add_epi64(*a, *b);
        }
    }

    static SIMD_INLINE void sub(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_sub_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_sub_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_sub_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_sub_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm_sub_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_sub_epi64(*a, *b);
        }
    }

    static SIMD_INLINE void mul(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_mul_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_mul_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_mullo_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_mullo_epi32(*a, *b);
#else
            __m128i tmp1 = _mm_mul_epu32(*a, *b);
            __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(*a, 4), _mm_srli_si128(*b, 4));
            *dst = _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
                                      _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
#endif
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] *= b_arr[i];
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void div(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_div_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_div_pd(*a, *b);
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] /= b_arr[i];
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }
    
    static SIMD_INLINE void min(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_min_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_min_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epi8(*a, *b);
#else
            alignas(16) int8_t a_arr[16], b_arr[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 16; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_min_epu8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_min_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epu16(*a, *b);
#else
            alignas(16) uint16_t a_arr[8], b_arr[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epi32(*a, *b);
#else
            alignas(16) int32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_min_epu32(*a, *b);
#else
            alignas(16) uint32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] = std::min(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void max(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_max_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_max_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epi8(*a, *b);
#else
            alignas(16) int8_t a_arr[16], b_arr[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 16; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_max_epu8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_max_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epu16(*a, *b);
#else
            alignas(16) uint16_t a_arr[8], b_arr[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 8; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epi32(*a, *b);
#else
            alignas(16) int32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
#if SIMD_SSE4_1
            *dst = _mm_max_epu32(*a, *b);
#else
            alignas(16) uint32_t a_arr[4], b_arr[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (int i = 0; i < 4; ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
#endif
        }
        else
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                a_arr[i] = std::max(a_arr[i], b_arr[i]);
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(a_arr));
        }
    }

    static SIMD_INLINE void bitwise_and(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_and_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_and_pd(*a, *b);
        }
        else
        {
            *dst = _mm_and_si128(*a, *b);
        }
    }

    static SIMD_INLINE void bitwise_or(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_or_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_or_pd(*a, *b);
        }
        else
        {
            *dst = _mm_or_si128(*a, *b);
        }
    }

    static SIMD_INLINE void bitwise_xor(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_xor_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_xor_pd(*a, *b);
        }
        else
        {
            *dst = _mm_xor_si128(*a, *b);
        }
    }

    static SIMD_INLINE void bitwise_not(register_t* dst, const register_t* a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const __m128 ones = _mm_cmpeq_ps(*a, *a);
            *dst = _mm_xor_ps(*a, ones);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            const __m128d ones = _mm_cmpeq_pd(*a, *a);
            *dst = _mm_xor_pd(*a, ones);
        }
        else
        {
            const __m128i ones = _mm_cmpeq_epi32(*a, *a);
            *dst = _mm_xor_si128(*a, ones);
        }
    }

    static SIMD_INLINE void blend(register_t* dst, const register_t* a, const register_t* b,
                                  const typename mask_register_type<T, sse2_tag>::type* mask)
    {
#if SIMD_SSE4_1
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_blendv_ps(*a, *b, *mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_blendv_pd(*a, *b, *mask);
        }
        else
        {
            *dst = _mm_blendv_epi8(*a, *b, *mask);
        }
#else
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_or_ps(_mm_and_ps(*mask, *b), _mm_andnot_ps(*mask, *a));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_or_pd(_mm_and_pd(*mask, *b), _mm_andnot_pd(*mask, *a));
        }
        else
        {
            *dst = _mm_or_si128(_mm_and_si128(*mask, *b), _mm_andnot_si128(*mask, *a));
        }
#endif
    }

    static SIMD_INLINE void select(register_t* dst,
                                   const typename mask_register_type<T, sse2_tag>::type* mask,
                                   const register_t* a, const register_t* b)
    {
        // !Note: operands reversed because masks are different semantics
        blend(dst, b, a, mask);
    }

    static SIMD_INLINE T horizontal_sum(const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m128 sum = *src;
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m128d sum = *src;
            sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
            return _mm_cvtsd_f64(sum);
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            T sum = 0;
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                sum += tmp[i];
            }
            return sum;
        }
    }

    static SIMD_INLINE T horizontal_min(const register_t* src)
    {
        alignas(16) T tmp[16 / sizeof(T)];
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(tmp, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(tmp, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
        }

        T min_val = tmp[0];
        for (size_t i = 1; i < 16 / sizeof(T); ++i)
        {
            min_val = std::min(min_val, tmp[i]);
        }
        return min_val;
    }

    static SIMD_INLINE T horizontal_max(const register_t* src)
    {
        alignas(16) T tmp[16 / sizeof(T)];
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(tmp, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(tmp, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
        }

        T max_val = tmp[0];
        for (size_t i = 1; i < 16 / sizeof(T); ++i)
        {
            max_val = std::max(max_val, tmp[i]);
        }
        return max_val;
    }

    static SIMD_INLINE void shuffle(register_t* dst, const register_t* src, const int* indices)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);

            alignas(16) float result[4];
            for (size_t i = 0; i < 4; ++i)
            {
                result[i] = tmp[indices[i] % 4];
            }

            *dst = _mm_load_ps(result);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);

            alignas(16) double result[2];
            for (size_t i = 0; i < 2; ++i)
            {
                result[i] = tmp[indices[i] % 2];
            }

            *dst = _mm_load_pd(result);
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);

            alignas(16) T result[16 / sizeof(T)];
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                result[i] = tmp[indices[i] % (16 / sizeof(T))];
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(result));
        }
    }

    template <typename U>
    static SIMD_INLINE void convert(typename register_type<U, sse2_tag>::type* dst,
                                    const register_t* src)
    {
        if constexpr (std::is_same_v<T, float> && std::is_same_v<U, int32_t>)
        {
            *dst = _mm_cvtps_epi32(*src);
        }
        else if constexpr (std::is_same_v<T, int32_t> && std::is_same_v<U, float>)
        {
            *dst = _mm_cvtepi32_ps(*src);
        }
        else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, double>)
        {
            __m128 s = *src;
            __m128d lo = _mm_cvtps_pd(s);
            __m128d hi = _mm_cvtps_pd(_mm_movehl_ps(s, s));
            // dst should be an array of two __m128d registers
            dst[0] = lo;
            dst[1] = hi;
        }
        else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, float>)
        {
            __m128 lo = _mm_cvtpd_ps(src[0]);
            __m128 hi = _mm_cvtpd_ps(src[1]);
            *dst = _mm_movelh_ps(lo, hi);
        }
        else
        {
            // Generic conversion for other types
            alignas(16) T src_arr[16 / sizeof(T)];
            if constexpr (std::is_same_v<T, float>)
            {
                _mm_store_ps(src_arr, *src);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                _mm_store_pd(src_arr, *src);
            }
            else
            {
                _mm_store_si128(reinterpret_cast<__m128i*>(src_arr), *src);
            }

            alignas(16) U dst_arr[16 / sizeof(U)];
            for (size_t i = 0; i < std::min(16 / sizeof(T), 16 / sizeof(U)); ++i)
            {
                dst_arr[i] = static_cast<U>(src_arr[i]);
            }

            if constexpr (std::is_same_v<U, float>)
            {
                *dst = _mm_load_ps(dst_arr);
            }
            else if constexpr (std::is_same_v<U, double>)
            {
                *dst = _mm_load_pd(dst_arr);
            }
            else
            {
                *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(dst_arr));
            }
        }
    }
};

template <typename T, size_t N>
struct mask_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using mask_register_t = typename mask_register_type<T, sse2_tag>::type;
    using register_t = typename register_type<T, sse2_tag>::type;
    
    static SIMD_INLINE void set_true(mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *mask = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *mask = _mm_cmpeq_pd(_mm_setzero_pd(), _mm_setzero_pd());
        }
        else
        {
            *mask = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
        }
    }

    static SIMD_INLINE void set_false(mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *mask = _mm_setzero_ps();
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *mask = _mm_setzero_pd();
        }
        else
        {
            *mask = _mm_setzero_si128();
        }
    }

    static SIMD_INLINE void load(mask_register_t* dst, const bool* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) int32_t tmp[4] = {0};
            for (size_t i = 0; i < 4; ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm_load_ps(reinterpret_cast<const float*>(tmp));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) int64_t tmp[2] = {0};
            for (size_t i = 0; i < 2; ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm_load_pd(reinterpret_cast<const double*>(tmp));
        }
        else
        {
            alignas(16) int tmp[16 / sizeof(T)] = {0};
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                tmp[i] = src[i] ? -1 : 0;
            }
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
    }

    static SIMD_INLINE void store(const mask_register_t* src, bool* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (size_t i = 0; i < 4; ++i)
            {
                dst[i] = reinterpret_cast<const int32_t*>(tmp)[i] != 0;
            }
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (size_t i = 0; i < 2; ++i)
            {
                dst[i] = reinterpret_cast<const int64_t*>(tmp)[i] != 0;
            }
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                dst[i] = tmp[i] != 0;
            }
        }
    }

    static SIMD_INLINE bool extract(const mask_register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            return reinterpret_cast<const int32_t*>(tmp)[index % 4] != 0;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            return reinterpret_cast<const int64_t*>(tmp)[index % 2] != 0;
        }
        else
        {
            alignas(16) T tmp[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            return tmp[index % (16 / sizeof(T))] != 0;
        }
    }

    static SIMD_INLINE void logical_and(mask_register_t* dst, const mask_register_t* a,
                                        const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_and_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_and_pd(*a, *b);
        }
        else
        {
            *dst = _mm_and_si128(*a, *b);
        }
    }

    static SIMD_INLINE void logical_or(mask_register_t* dst, const mask_register_t* a,
                                       const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_or_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_or_pd(*a, *b);
        }
        else
        {
            *dst = _mm_or_si128(*a, *b);
        }
    }

    static SIMD_INLINE void logical_xor(mask_register_t* dst, const mask_register_t* a,
                                        const mask_register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_xor_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_xor_pd(*a, *b);
        }
        else
        {
            *dst = _mm_xor_si128(*a, *b);
        }
    }

    static SIMD_INLINE void logical_not(mask_register_t* dst, const mask_register_t* a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            __m128 ones = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
            *dst = _mm_xor_ps(*a, ones);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            __m128d ones = _mm_cmpeq_pd(_mm_setzero_pd(), _mm_setzero_pd());
            *dst = _mm_xor_pd(*a, ones);
        }
        else
        {
            __m128i ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
            *dst = _mm_xor_si128(*a, ones);
        }
    }

    static SIMD_INLINE void cmp_eq(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmpeq_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmpeq_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm_cmpeq_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm_cmpeq_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                           std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm_cmpeq_epi32(*a, *b);
        }
    }

    static SIMD_INLINE void cmp_neq(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        cmp_eq(dst, a, b);
        logical_not(dst, dst);
    }

    static SIMD_INLINE void cmp_lt(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmplt_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmplt_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            *dst = _mm_cmplt_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_cmplt_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            *dst = _mm_cmplt_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                           std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                           std::is_same_v<T, uint64_t>)
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            alignas(16) int tmp[16 / sizeof(T)] = {0};
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] < b_arr[i] ? -1 : 0;
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
    }

    static SIMD_INLINE void cmp_le(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmple_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmple_pd(*a, *b);
        }
        else
        {
            mask_register_t lt, eq;
            cmp_lt(&lt, a, b);
            cmp_eq(&eq, a, b);
            logical_or(dst, &lt, &eq);
        }
    }

    static SIMD_INLINE void cmp_gt(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmpgt_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmpgt_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
            *dst = _mm_cmpgt_epi8(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
            *dst = _mm_cmpgt_epi16(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            *dst = _mm_cmpgt_epi32(*a, *b);
        }
        else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                           std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                           std::is_same_v<T, uint64_t>)
        {
            alignas(16) T a_arr[16 / sizeof(T)];
            alignas(16) T b_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(a_arr), *a);
            _mm_store_si128(reinterpret_cast<__m128i*>(b_arr), *b);

            alignas(16) int tmp[16 / sizeof(T)] = {0};
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                tmp[i] = a_arr[i] > b_arr[i] ? -1 : 0;
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
    }

    static SIMD_INLINE void cmp_ge(mask_register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_cmpge_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_cmpge_pd(*a, *b);
        }
        else
        {
            mask_register_t gt, eq;
            cmp_gt(&gt, a, b);
            cmp_eq(&eq, a, b);
            logical_or(dst, &gt, &eq);
        }
    }

    static SIMD_INLINE uint64_t to_bitmask(const mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return _mm_movemask_ps(*mask);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return _mm_movemask_pd(*mask);
        }
        else
        {
            return _mm_movemask_epi8(*mask);
        }
    }

    static SIMD_INLINE bool any(const mask_register_t* mask) { return to_bitmask(mask) != 0; }

    static SIMD_INLINE bool all(const mask_register_t* mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return to_bitmask(mask) == 0xF;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return to_bitmask(mask) == 0x3;
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            return to_bitmask(mask) == 0xFFFF;
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            constexpr uint64_t mask_bits = (1ULL << (16 / shift)) - 1;
            constexpr uint64_t full_mask = shift == 1   ? 0xFFFF
                                           : shift == 2 ? 0x5555
                                           : shift == 4 ? 0x1111
                                           : shift == 8 ? 0x0101
                                                        : 0;
            return (to_bitmask(mask) & full_mask) == full_mask;
        }
        else
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            constexpr uint64_t mask_bits = (1ULL << (16 / shift)) - 1;
            constexpr uint64_t full_mask = shift == 1   ? 0xFFFF
                                           : shift == 2 ? 0x5555
                                           : shift == 4 ? 0x1111
                                           : shift == 8 ? 0x0101
                                                        : 0;
            return (to_bitmask(mask) & full_mask) == full_mask;
        }
    }

    static SIMD_INLINE int count(const mask_register_t* mask)
    {
        uint64_t bits = to_bitmask(mask);
        if constexpr (std::is_same_v<T, float>)
        {
            return std::popcount(bits & 0xF);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return std::popcount(bits & 0x3);
        }
        else
        {
            constexpr int shift = sizeof(T) / sizeof(int8_t);
            if constexpr (shift == 1)
            {
                return std::popcount(bits & 0xFFFF);
            }
            else
            {
                int count = 0;
                constexpr int elements = 16 / sizeof(T);
                for (int i = 0; i < elements; ++i)
                {
                    if (bits & (1ULL << (i * shift)))
                    {
                        count++;
                    }
                }
                return count;
            }
        }
    }

};

template <typename T, size_t N>
struct memory_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void load(register_t* dst, const T* src)
    {
        load_unaligned(dst, src);
    }

    static SIMD_INLINE void load_aligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_load_ps(src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_load_pd(src);
        }
        else
        {
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(src));
        }
    }

    static SIMD_INLINE void load_unaligned(register_t* dst, const T* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_loadu_ps(src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_loadu_pd(src);
        }
        else
        {
            *dst = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        }
    }

    static SIMD_INLINE void store(const register_t* src, T* dst)
    {
        store_unaligned(src, dst);
    }

    static SIMD_INLINE void store_aligned(const register_t* src, T* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(dst, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(dst, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), *src);
        }
    }

    static SIMD_INLINE void store_unaligned(const register_t* src, T* dst)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            _mm_storeu_ps(dst, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_storeu_pd(dst, *src);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), *src);
        }
    }

    static SIMD_INLINE void prefetch(const T* ptr, int hint)
    {
#if SIMD_SSE3
        _mm_prefetch(reinterpret_cast<const char*>(ptr), hint);
#else
        (void)ptr;
        (void)hint;
#endif
    }

    template <typename IndexT>
    static SIMD_INLINE void gather(register_t* dst, const T* base,
                                   const typename register_type<IndexT, sse2_tag>::type* indices)
    {
        alignas(16) IndexT idx_arr[16 / sizeof(IndexT)];
        _mm_store_si128(reinterpret_cast<__m128i*>(idx_arr), *indices);

        alignas(16) T result[16 / sizeof(T)];
        for (size_t i = 0; i < 16 / sizeof(T); ++i)
        {
            if (i < N)
            {
                result[i] = base[idx_arr[i]];
            }
            else
            {
                result[i] = 0;
            }
        }

        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_load_ps(result);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_load_pd(result);
        }
        else
        {
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(result));
        }
    }

    template <typename IndexT>
    static SIMD_INLINE void scatter(const register_t* src, T* base,
                                    const typename register_type<IndexT, sse2_tag>::type* indices)
    {
        alignas(16) T src_arr[16 / sizeof(T)];
        alignas(16) IndexT idx_arr[16 / sizeof(IndexT)];

        if constexpr (std::is_same_v<T, float>)
        {
            _mm_store_ps(src_arr, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            _mm_store_pd(src_arr, *src);
        }
        else
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(src_arr), *src);
        }

        _mm_store_si128(reinterpret_cast<__m128i*>(idx_arr), *indices);

        for (size_t i = 0; i < std::min(N, 16 / sizeof(T)); ++i)
        {
            base[idx_arr[i]] = src_arr[i];
        }
    }
};

template <typename T, size_t N>
struct math_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>
{
    using register_t = typename register_type<T, sse2_tag>::type;

    static SIMD_INLINE void abs(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const __m128 sign_mask = _mm_set1_ps(-0.0f);
            *dst = _mm_andnot_ps(sign_mask, *src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            const __m128d sign_mask = _mm_set1_pd(-0.0);
            *dst = _mm_andnot_pd(sign_mask, *src);
        }
        else if constexpr (std::is_same_v<T, int8_t>)
        {
#if SIMD_SSSE3
            *dst = _mm_abs_epi8(*src);
#else
            alignas(16) int8_t tmp[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            for (int i = 0; i < 16; ++i)
            {
                tmp[i] = std::abs(tmp[i]);
            }
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t>)
        {
#if SIMD_SSSE3
            *dst = _mm_abs_epi16(*src);
#else
            __m128i sign = _mm_srai_epi16(*src, 15);
            __m128i inv = _mm_xor_si128(*src, sign);
            *dst = _mm_sub_epi16(inv, sign);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
#if SIMD_SSSE3
            *dst = _mm_abs_epi32(*src);
#else
            // Manual abs using SSE2 instructions
            __m128i sign = _mm_srai_epi32(*src, 31);
            __m128i inv = _mm_xor_si128(*src, sign);
            *dst = _mm_sub_epi32(inv, sign);
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            alignas(16) int64_t tmp[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp), *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::abs(tmp[i]);
            }
            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
        }
        else
        {
            *dst = *src;
        }
    }

    static SIMD_INLINE void sqrt(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_sqrt_ps(*src);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_sqrt_pd(*src);
        }
        else
        {
            alignas(16) T src_arr[16 / sizeof(T)];
            _mm_store_si128(reinterpret_cast<__m128i*>(src_arr), *src);

            alignas(16) T result[16 / sizeof(T)];
            for (size_t i = 0; i < 16 / sizeof(T); ++i)
            {
                result[i] = static_cast<T>(std::sqrt(static_cast<double>(src_arr[i])));
            }

            *dst = _mm_load_si128(reinterpret_cast<const __m128i*>(result));
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void sin(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::sin(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::sin(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void cos(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::cos(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::cos(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void tan(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::tan(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::tan(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void exp(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::exp(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::exp(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    template <typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    static SIMD_INLINE void log(register_t* dst, const register_t* src)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, *src);
            for (int i = 0; i < 4; ++i)
            {
                tmp[i] = std::log(tmp[i]);
            }
            *dst = _mm_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, *src);
            for (int i = 0; i < 2; ++i)
            {
                tmp[i] = std::log(tmp[i]);
            }
            *dst = _mm_load_pd(tmp);
        }
    }

    static SIMD_INLINE void fmadd(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
#if SIMD_FMA
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_fmadd_ps(*a, *b, *c);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_fmadd_pd(*a, *b, *c);
        }
        else
        {
            register_t tmp;
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                mul(&tmp, a, b);
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                add(dst, &tmp, c);
        }
#else
        register_t tmp;
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::mul(&tmp,
                                                                                             a, b);
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::add(dst,
                                                                                             &tmp,
                                                                                             c);
#endif
    }

    static SIMD_INLINE void fmsub(register_t* dst, const register_t* a, const register_t* b,
                                  const register_t* c)
    {
#if SIMD_FMA
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm_fmsub_ps(*a, *b, *c);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm_fmsub_pd(*a, *b, *c);
        }
        else
        {
            register_t tmp;
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                mul(&tmp, a, b);
            vector_ops<T, N,
                       std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::
                sub(dst, &tmp, c);
        }
#else
        // No FMA instructions available, use separate multiply and subtract
        register_t tmp;
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::mul(&tmp,
                                                                                             a, b);
        vector_ops<
            T, N,
            std::enable_if_t<simd::FeatureDetector<simd::Feature::SSE2>::compile_time>>::sub(dst,
                                                                                             &tmp,
                                                                                             c);
#endif
    }
};

#if SIMD_AVX

template <typename T, size_t N>
struct vector_ops<T, N, std::enable_if_t<simd::FeatureDetector<simd::Feature::AVX>::compile_time>>
{
    using register_t = typename register_type<T, avx_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_set1_ps(value);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_set1_pd(value);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm256_set1_epi8(static_cast<char>(value));
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm256_set1_epi16(static_cast<short>(value));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm256_set1_epi32(static_cast<int>(value));
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm256_set1_epi64x(static_cast<long long>(value));
        }
    }

    static SIMD_INLINE T extract(const register_t* src, size_t index)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *src);
            return tmp[index % 8];
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *src);
            return tmp[index % 4];
        }
        else
        {
            alignas(32) T tmp[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *src);
            return tmp[index % (32 / sizeof(T))];
        }
    }

    static SIMD_INLINE void insert(register_t* dst, size_t index, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, *dst);
            tmp[index % 8] = value;
            *dst = _mm256_load_ps(tmp);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, *dst);
            tmp[index % 4] = value;
            *dst = _mm256_load_pd(tmp);
        }
        else
        {
            alignas(32) T tmp[32 / sizeof(T)];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), *dst);
            tmp[index % (32 / sizeof(T))] = value;
            *dst = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
        }
    }

    static SIMD_INLINE void add(register_t* dst, const register_t* a, const register_t* b)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm256_add_ps(*a, *b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm256_add_pd(*a, *b);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi8(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_add_epi8(a_lo, b_lo);
            __m128i res_hi = _mm_add_epi8(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi16(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_add_epi16(a_lo, b_lo);
            __m128i res_hi = _mm_add_epi16(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi32(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_add_epi32(a_lo, b_lo);
            __m128i res_hi = _mm_add_epi32(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
#if SIMD_AVX2
            *dst = _mm256_add_epi64(*a, *b);
#else
            __m128i a_lo = _mm256_extractf128_si256(*a, 0);
            __m128i a_hi = _mm256_extractf128_si256(*a, 1);
            __m128i b_lo = _mm256_extractf128_si256(*b, 0);
            __m128i b_hi = _mm256_extractf128_si256(*b, 1);

            __m128i res_lo = _mm_add_epi64(a_lo, b_lo);
            __m128i res_hi = _mm_add_epi64(a_hi, b_hi);

            *dst = _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
#endif
        }
    }

    // Additional AVX operations would be implemented here...
    // For brevity, not all operations are shown, but they would follow
    // a similar pattern to the SSE2 implementations with AVX instructions
};

#endif // SIMD_AVX

#if SIMD_ARCH_X86 && SIMD_AVX512

template <typename T, size_t N>
struct vector_ops<T, N,
                  std::enable_if_t<simd::FeatureDetector<simd::Feature::AVX512F>::compile_time>>
{
    using register_t = typename register_type<T, avx512_tag>::type;
    using mask_t = typename mask_register_type<T, avx512_tag>::type;

    static SIMD_INLINE void set1(register_t* dst, T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            *dst = _mm512_set1_ps(value);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            *dst = _mm512_set1_pd(value);
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
        {
            *dst = _mm512_set1_epi8(static_cast<char>(value));
        }
        else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
        {
            *dst = _mm512_set1_epi16(static_cast<short>(value));
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
        {
            *dst = _mm512_set1_epi32(static_cast<int>(value));
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        {
            *dst = _mm512_set1_epi64(static_cast<long long>(value));
        }
    }

    // Additional AVX-512 operations would be implemented here...
    // For brevity, not all operations are shown
};

#endif // SIMD_AVX512 

template <typename T, size_t N, typename Func>
SIMD_INLINE auto dispatch_simd_function(Func&& sse2_impl, Func&& avx_impl, Func&& avx2_impl,
                                        Func&& avx512_impl, Func&& fallback_impl)
{
    if constexpr (simd::FeatureDetector<simd::Feature::AVX512F>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX512F>())
        {
            return std::forward<Func>(avx512_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::AVX2>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX2>())
        {
            return std::forward<Func>(avx2_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::AVX>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::AVX>())
        {
            return std::forward<Func>(avx_impl);
        }
    }

    if constexpr (simd::FeatureDetector<simd::Feature::SSE2>::compile_time)
    {
        if (simd::runtime::has<simd::Feature::SSE2>())
        {
            return std::forward<Func>(sse2_impl);
        }
    }

    return std::forward<Func>(fallback_impl);
}

} // namespace detail
} // namespace vector_simd

#endif
#endif /* End of include guard: SIMD_HPP_al9nn6 */
