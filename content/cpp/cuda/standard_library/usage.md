---
title: "Using the CUDA C++ Standard Library"
description: "Availability, requirements, namespace choices, and practical rules for libcu++."
source_path: "cpp/cuda/standard_library/usage"
---

libcu++ is included with the CUDA Toolkit and the NVIDIA HPC SDK. NVIDIA documents it as header-only and on the default include path, so ordinary CUDA Toolkit use does not require linking a separate libcu++ shared library.

## Requirements

| Requirement area | Practical note |
| --- | --- |
| System software | Use a supported CUDA Toolkit or NVIDIA HPC SDK release |
| First availability | NVIDIA documents libcu++ as first released in NVHPC 20.3 and CUDA 10.2 |
| C++ dialect | Supported dialects follow CCCL; some features are backported to earlier dialects |
| Host architecture | NVIDIA documents support for `aarch64` and `x86-64` hosts |
| Host operating systems | NVIDIA documents Linux, Windows, Android, and QNX support |
| Device architecture | Support follows the CUDA Toolkit, with partial support notes for older Maxwell and Pascal architectures |

For feature-by-feature availability, use NVIDIA's Standard API and Extended API pages rather than assuming that a header exists because the corresponding ISO C++ feature exists.

## Namespace chooser

| Need | Use |
| --- | --- |
| Ordinary host-only C++ Standard Library | `std::` with ordinary C++ headers |
| Standard-like facility in CUDA host/device code | `cuda::std::` with `cuda/std/` headers |
| CUDA extension to a standard-like facility | `cuda::` with `cuda/` headers |
| Device-only CUDA extension | `cuda::device` |
| PTX instruction wrapper | `cuda::ptx` |

## Host and device compilation

`__host__ __device__` code is compiled for both sides. That makes `cuda::std` useful, but it does not erase the difference between host and device environments.

In shared host/device code:

- Avoid unguarded calls to host-only APIs.
- Keep abstractions small enough that both host and device compilation errors remain understandable.
- Prefer non-owning views and fixed-size storage in kernels.
- Keep dynamic allocation, exceptions, locale, I/O, and heavyweight runtime behavior out of hot device code unless NVIDIA explicitly documents the facility and the performance tradeoff is acceptable.
- Use explicit CUDA synchronization and memory scopes when communication between threads matters.

## Build shape

```cpp
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/atomic>

__host__ __device__ int clamp_nonnegative(int value)
{
    return value < 0 ? 0 : value;
}
```

Compile CUDA source with `nvcc` or another CUDA-capable compiler configuration appropriate for the project. The CUDA headers come from the toolkit or SDK installation.

## Version-sensitive code

The CCCL/libcu++ documentation evolves with CUDA releases. When writing portable library code:

- Record the minimum CUDA Toolkit or CCCL version in project documentation.
- Prefer stable `cuda::std` facilities over `cuda::experimental` APIs unless the experimental API is central to the design.
- Check release notes before depending on newer runtime wrappers, memory resource facilities, TMA support, or PTX wrappers.
- Test the exact GPU architectures and CUDA Toolkit versions you claim to support.

## External references

- [NVIDIA libcu++ requirements](https://nvidia.github.io/cccl/libcudacxx/setup/requirements.html)
- [NVIDIA getting libcu++](https://nvidia.github.io/cccl/libcudacxx/setup/getting.html)
- [NVIDIA libcu++ Standard API](https://nvidia.github.io/cccl/libcudacxx/standard_api.html)
- [NVIDIA libcu++ Extended API](https://nvidia.github.io/cccl/libcudacxx/extended_api.html)
