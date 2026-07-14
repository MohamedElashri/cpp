---
title: "CUDA C++ Standard Library"
description: "libcu++, cuda::std, CUDA C++ Standard Library extensions, and the CCCL library model."
source_path: "cpp/cuda/standard_library"
---

The CUDA C++ Standard Library is NVIDIA's libcu++ library: a CUDA-aware C++ library implementation distributed with the CUDA Core Compute Libraries (CCCL). It provides Standard Library facilities that can be used in CUDA C++ source, plus CUDA-specific extensions for the parts of GPU programming that the ISO C++ library does not model directly.

The central distinction is namespace and include path:

| Form | Header shape | Namespace | Where it is meant to work |
| --- | --- | --- | --- |
| Host Standard Library | `<atomic>`, `<tuple>`, `<type_traits>` | `std::` | Host code from the host compiler's C++ library |
| CUDA Standard API | `<cuda/std/atomic>`, `<cuda/std/tuple>` | `cuda::std::` | Host and device code where libcu++ provides the facility |
| CUDA extensions | `<cuda/atomic>`, `<cuda/barrier>`, `<cuda/pipeline>` | `cuda::` | Host and device code with CUDA-specific semantics |
| Device-only extensions | `<cuda/device/*>` | `cuda::device` | Device code only |
| PTX wrappers | `<cuda/ptx>` | `cuda::ptx` | Device code using C++ wrappers around PTX instructions |

Use `cuda::std` when you want a familiar Standard Library type or function in code that may run on the GPU. Use `cuda::` when you need a CUDA extension such as scoped atomics, device-side asynchronous copy, CUDA memory resources, fancy iterators, or runtime-aware wrappers.

## First example

```cpp
#include <cuda/std/array>
#include <cuda/std/span>

__global__ void add_one(cuda::std::span<int> values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < values.size())
        values[i] += 1;
}
```

This is the same idea as using `std::span` in ordinary C++, but the include and namespace are the CUDA forms. A `cuda::std` facility should still be chosen with GPU constraints in mind: device code has different execution, allocation, synchronization, and exception-handling constraints than host code.

## CUDA extension example

```cpp
#include <cuda/atomic>

__global__ void count_hits(int* counter, int* values, int n)
{
    cuda::atomic_ref<int, cuda::thread_scope_device> hits(*counter);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && values[i] != 0)
        hits.fetch_add(1, cuda::std::memory_order_relaxed);
}
```

The extension form adds CUDA thread scopes. A block-scoped atomic, device-scoped atomic, and system-scoped atomic do not express the same visibility or cost.

## Main areas

| Area | Start here | What it covers |
| --- | --- | --- |
| Standard API | [Standard headers](standard_api/) | `cuda::std` headers corresponding to ISO C++ library headers |
| CUDA extensions | [Extensions](extensions/) | Scoped synchronization, async copy, memory resources, iterators, runtime wrappers, PTX wrappers |
| Usage and availability | [Usage requirements](usage/) | Toolkit/HPC SDK availability, header-only use, architecture notes, and namespace selection |
| CUDA memory model | [Memory model](../memory/) | CUDA memory spaces and runtime allocation/copy patterns |
| Runtime API | [Runtime API basics](../runtime_api/) | Host-side allocation, copy, launch, synchronization, and error checks |

## What libcu++ is not

libcu++ does not replace the host compiler's `std::` library. Ordinary `<vector>`, `<string>`, `<thread>`, and other host headers still come from the host toolchain and are primarily host facilities. When device code needs a supported Standard Library facility, include the `cuda/std/` header and use `cuda::std::`.

It is also not the whole CUDA platform. CCCL also contains CUB and Thrust, and CUDA Toolkit includes domain libraries such as cuBLAS, cuFFT, cuDNN, cuSOLVER, and many more. libcu++ is the C++ library layer for writing CUDA-aware C++ abstractions, not a replacement for every high-level numeric library.

## External references

- [NVIDIA libcu++ overview](https://nvidia.github.io/cccl/libcudacxx/)
- [NVIDIA libcu++ Standard API](https://nvidia.github.io/cccl/libcudacxx/standard_api.html)
- [NVIDIA libcu++ Extended API](https://nvidia.github.io/cccl/libcudacxx/extended_api.html)
- [NVIDIA CCCL documentation](https://nvidia.github.io/cccl/)
