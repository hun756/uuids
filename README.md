<p align="center">
  <img src="https://raw.githubusercontent.com/assets/uuid-logo.svg" height="128" alt="UUIDv4 Logo">
</p>
<h1 align="center">UUIDv4</h1>
<p align="center">
  <strong>Ultra-high performance hardware-accelerated UUID generation for modern C++</strong>
</p>
<p align="center">
  <a href="https://en.wikipedia.org/wiki/C%2B%2B20"><img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square" alt="C++ 20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="MIT License"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-x86__64%20%7C%20ARM-lightgrey.svg?style=flat-square" alt="Platforms"></a>
  <a href="#"><img src="https://img.shields.io/badge/header--only-yes-brightgreen.svg?style=flat-square" alt="Header Only"></a>
  <a href="#"><img src="https://img.shields.io/badge/CPU-SIMD%20Optimized-orange.svg?style=flat-square" alt="SIMD Optimized"></a>
</p>
<p align="center">
  <a href="#key-features">Features</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#integration">Integration</a> •
  <a href="#usage">Usage</a> •
  <a href="#advanced-usage">Advanced</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#api-reference">API</a> •
  <a href="#license">License</a>
</p>

---

## Introduction

**UUIDv4** is a state-of-the-art UUID generation library designed for mission-critical applications where performance and reliability are non-negotiable. Using advanced SIMD instructions, hardware intrinsics, and modern C++20 features, it delivers unparalleled UUID generation speed while maintaining strict RFC 4122 compliance.

Whether you're building high-throughput distributed systems, performance-sensitive databases, or real-time applications, UUIDv4 provides the rock-solid foundation you need for unique identifier generation at scale.

## Key Features

- **📈 Exceptional Performance**: Leverages CPU hardware acceleration for UUID generation that's up to **~10x faster** than standard implementations
- **🔒 Enterprise-Grade Security**: Uses hardware RNG capabilities (RDRAND/RDSEED) for cryptographically secure random generation
- **⚡ SIMD Acceleration**: Utilizes AES-NI instructions for enhanced throughput and performance
- **🔍 Zero Overhead Abstractions**: Meticulously crafted to compile down to optimal machine code
- **📦 Header-Only**: Single-file inclusion with zero dependencies for trivial integration
- **🛡️ Type Safety**: Leverages C++20 concepts for compile-time validation and type safety
- **🧠 Intelligent Fallbacks**: Seamlessly adapts to available hardware capabilities
- **🔧 Highly Configurable**: Template-based architecture supports custom PRNG engines and configuration
- **📊 Thread-Safe**: Lock-free design ensures thread safety without performance penalties
- **✅ Fully RFC 4122 Compliant**: Generates standard-compliant UUIDs (version 4, variant 1)

## Benchmarks

Performance comparison against leading UUID libraries (lower is better):

| Library                     | Time to generate 10M UUIDs | Relative Performance |
| --------------------------- | -------------------------- | -------------------- |
| **UUIDv4 (HW accelerated)** | **40 ms**                  | **1.0x (baseline)**  |
| UUIDv4 (SW fallback)        | 102 ms                     | 2.5x slower          |
| boost::uuid                 | 736 ms                     | 18.4x slower         |
| libuuid                     | -                          | -                    |

*Benchmarked on Ryzen 7 7800x3d, compiled with clang 19.1.6 with -O3 optimization (hw accelaration with -march=native flag)*

## Integration

### Basic Integration (Header-Only)

Simply include the header in your project:

```cpp
#include <uuids/uuidv4.hpp>
```

### Package Managers

#### CMake FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
  uuidv4
  GIT_REPOSITORY https://github.com/korbolkoinc/uuids.git
  GIT_TAG v1.0.0
)

FetchContent_MakeAvailable(uuidv4)

target_link_libraries(your_target PRIVATE uuidv4::uuidv4)
```
