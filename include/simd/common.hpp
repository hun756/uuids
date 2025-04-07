#ifndef COMMON_HPP_2J3K4L5
#define COMMON_HPP_2J3K4L5

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || \
    defined(__i386__) || defined(_M_IX86)
#define SIMD_ARCH_X86 1
#else
#define SIMD_ARCH_X86 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_ARCH_ARM_NEON 1
#else
#define SIMD_ARCH_ARM_NEON 0
#endif

#if defined(__wasm_simd128__)
#define SIMD_ARCH_WASM_SIMD 1
#else
#define SIMD_ARCH_WASM_SIMD 0
#endif

#if defined(_MSC_VER)
#define SIMD_COMPILER_MSVC 1
#define SIMD_COMPILER_GCC 0
#define SIMD_COMPILER_CLANG 0
#include <intrin.h>
#elif defined(__clang__)
#define SIMD_COMPILER_MSVC 0
#define SIMD_COMPILER_GCC 0
#define SIMD_COMPILER_CLANG 1
#if SIMD_ARCH_X86
#include <cpuid.h>
#include <x86intrin.h>
#endif
#include <immintrin.h>
#elif defined(__GNUC__)
#define SIMD_COMPILER_MSVC 0
#define SIMD_COMPILER_GCC 1
#define SIMD_COMPILER_CLANG 0
#if SIMD_ARCH_X86
#include <cpuid.h>
#include <x86intrin.h>
#endif
#include <immintrin.h>
#else
#define SIMD_COMPILER_MSVC 0
#define SIMD_COMPILER_GCC 0
#define SIMD_COMPILER_CLANG 0
#endif

#if SIMD_ARCH_ARM_NEON
#include <arm_neon.h>
#endif

#if SIMD_ARCH_WASM_SIMD
#include <wasm_simd128.h>
#endif

#if SIMD_COMPILER_MSVC
#define SIMD_INLINE __forceinline
#define SIMD_NOINLINE __declspec(noinline)
#define SIMD_ALIGNED(x) __declspec(align(x))
#define SIMD_RESTRICT __restrict
#define SIMD_LIKELY(x) (x)
#define SIMD_UNLIKELY(x) (x)
#define SIMD_VECTORCALL __vectorcall
#define SIMD_UNROLL_LOOPS
#define SIMD_ALWAYS_INLINE SIMD_INLINE
#define SIMD_NEVER_INLINE SIMD_NOINLINE
#elif SIMD_COMPILER_GCC || SIMD_COMPILER_CLANG
#define SIMD_INLINE inline __attribute__((always_inline))
#define SIMD_NOINLINE __attribute__((noinline))
#define SIMD_ALIGNED(x) __attribute__((aligned(x)))
#define SIMD_RESTRICT __restrict__
#define SIMD_LIKELY(x) __builtin_expect(!!(x), 1)
#define SIMD_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define SIMD_VECTORCALL
#define SIMD_ALWAYS_INLINE SIMD_INLINE
#define SIMD_NEVER_INLINE SIMD_NOINLINE

#if SIMD_COMPILER_CLANG
#define SIMD_UNROLL_LOOPS _Pragma("clang loop unroll(enable)")
#elif SIMD_COMPILER_GCC
#define SIMD_UNROLL_LOOPS _Pragma("GCC unroll 16")
#endif

#else
#define SIMD_INLINE inline
#define SIMD_NOINLINE
#define SIMD_ALIGNED(x)
#define SIMD_RESTRICT
#define SIMD_LIKELY(x) (x)
#define SIMD_UNLIKELY(x) (x)
#define SIMD_VECTORCALL
#define SIMD_UNROLL_LOOPS
#define SIMD_ALWAYS_INLINE SIMD_INLINE
#define SIMD_NEVER_INLINE SIMD_NOINLINE
#endif

#define SIMD_SUPPORT_VERSION_MAJOR 0
#define SIMD_SUPPORT_VERSION_MINOR 1
#define SIMD_SUPPORT_VERSION_PATCH 0

#endif // COMMON_HPP_2J3K4L5