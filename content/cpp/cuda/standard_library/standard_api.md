---
title: "CUDA C++ Standard Library standard API"
description: "The cuda::std API surface: CUDA forms of selected C++ Standard Library headers."
source_path: "cpp/cuda/standard_library/standard_api"
---

The libcu++ Standard API exposes C++ Standard Library facilities under `cuda::std` and `cuda/std/` headers, including common headers such as `<cuda/std/span>`, `<cuda/std/array>`, `<cuda/std/tuple>`, `<cuda/std/atomic>`, and `<cuda/std/type_traits>`. The intent is to make standard-style vocabulary types, type traits, numerics, atomics, synchronization primitives, and small utilities available to code compiled for CUDA host and device contexts.

## Include and namespace pattern

```cpp
#include <cuda/std/atomic>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

cuda::std::atomic<int> counter;
cuda::std::tuple<int, float> pair_like;
static_assert(cuda::std::is_trivially_copyable_v<int>);
```

Do not include `<atomic>` and then expect `std::atomic` to be the CUDA device-capable implementation. Use the `cuda/std/` header and `cuda::std::` name when the object is intended to be usable in CUDA C++ device code.

## Header families

| Family | Representative headers | Notes |
| --- | --- | --- |
| Algorithms | `<cuda/std/algorithm>` | Standard algorithms and algorithm utilities where supported |
| Concepts and type support | `<cuda/std/concepts>`, `<cuda/std/type_traits>`, `<cuda/std/compare>`, `<cuda/std/version>` | Useful for generic host/device libraries |
| Containers and views | `<cuda/std/array>`, `<cuda/std/span>`, `<cuda/std/mdspan>` | Prefer fixed-size or non-owning views in device code |
| Utility vocabulary | `<cuda/std/optional>`, `<cuda/std/tuple>`, `<cuda/std/variant>`, `<cuda/std/utility>` | Familiar vocabulary types with CUDA-aware implementation constraints |
| Numerics | `<cuda/std/complex>`, `<cuda/std/numeric>`, `<cuda/std/numbers>`, `<cuda/std/random>` | Random support is not identical to a full host library surface |
| Ranges | `<cuda/std/ranges>` | Requires C++20; NVIDIA documents that range algorithms and views are not provided |
| Synchronization | `<cuda/std/atomic>`, `<cuda/std/barrier>`, `<cuda/std/latch>`, `<cuda/std/semaphore>` | For CUDA-specific thread scopes, use the `cuda::` extension forms |
| Time | `<cuda/std/chrono>` | Standard-style time utilities where available |
| C library wrappers | `<cuda/std/cstddef>`, `<cuda/std/cstdint>`, `<cuda/std/cmath>`, `<cuda/std/cstring>` | CUDA forms of selected C library headers |

## Backports

NVIDIA documents some Standard Library facilities as available in earlier C++ dialects than the ISO standard version that originally introduced them. That means the `cuda::std` API table is the source of truth for practical availability, not only the ISO standard label.

For example, a project compiling as C++17 may still see some library facilities that are standard in C++20 or later if libcu++ backports them. Code that relies on that should treat it as a CUDA/libcu++ portability decision, not a portable ISO C++ guarantee.

## Common choices

Use `cuda::std::array` for fixed-size aggregate storage that must be valid in device code.

Use `cuda::std::span` to pass pointer-and-size views into kernels without inventing a custom pair type.

Use `cuda::std::tuple`, `cuda::std::pair`, `cuda::std::optional`, and `cuda::std::variant` for small vocabulary types in generic host/device utilities.

Use `cuda::std::atomic` when standard atomic semantics are enough. Use `cuda::atomic` or `cuda::atomic_ref` when you need explicit CUDA thread scope.

## Limits to remember

The Standard API is not a promise that every standard facility is sensible inside a kernel. Dynamic allocation, exceptions, blocking behavior, and large stateful objects can be poor fits or unavailable depending on the facility and target.

Device code should still be written with CUDA execution in mind: thousands of threads, SIMT execution, explicit synchronization, separate memory spaces, and architecture-specific capabilities.

## External references

- [NVIDIA libcu++ Standard API](https://nvidia.github.io/cccl/libcudacxx/standard_api.html)
- [NVIDIA libcu++ overview](https://nvidia.github.io/cccl/libcudacxx/)
