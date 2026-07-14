---
title: "CUDA C++ Standard Library extensions"
description: "CUDA-specific libcu++ facilities in cuda::, cuda::device, and cuda::ptx."
source_path: "cpp/cuda/standard_library/extensions"
---

The libcu++ Extended API adds CUDA-specific facilities that cannot be expressed by a purely conforming ISO C++ Standard Library API, including `cuda::atomic_ref`, `cuda::thread_scope_device`, `cuda::memcpy_async`, CUDA fancy iterators, memory resources, runtime wrappers, and PTX wrappers. These facilities usually live in `cuda::`, with some device-only APIs in `cuda::device` and PTX wrappers in `cuda::ptx`.

## Extension families

| Family | Typical headers or names | Use it for |
| --- | --- | --- |
| Scoped atomics | `<cuda/atomic>`, `cuda::atomic`, `cuda::atomic_ref` | Atomic operations with block, device, or system visibility |
| Synchronization | `<cuda/barrier>`, `<cuda/latch>`, `<cuda/semaphore>` | CUDA-aware barriers, latches, semaphores, and cooperative coordination |
| Asynchronous operations | `<cuda/pipeline>`, `cuda::memcpy_async` | Device-side asynchronous data movement and pipelined shared-memory staging |
| Memory model utilities | Thread scopes and memory-order integration | Express CUDA visibility and ordering more directly than host-only C++ APIs |
| Fancy iterators | `cuda::counting_iterator`, `cuda::transform_iterator`, `cuda::zip_iterator` | Compose index spaces and data views without materializing temporary arrays |
| Memory resources | CUDA memory resources and resource references | Allocate or adapt pinned, managed, device, and pool-backed memory resources |
| Runtime wrappers | Streams, events, devices, buffers, launch utilities | C++ wrappers around selected CUDA Runtime API concepts |
| Math, numeric, random | CUDA extension numeric utilities | CUDA-specific additions around numeric and math use cases |
| Mdspan and vector protocols | `mdspan`, vector tuple protocol | Multidimensional views and structured access to CUDA vector-like types |
| PTX wrappers | `cuda::ptx` | C++ convenience wrappers for inline PTX instructions in device code |

## Thread scope

Standard C++ atomics describe memory ordering, but CUDA also needs scope: which threads can observe and participate in the synchronization.

| Scope | Typical meaning |
| --- | --- |
| `cuda::thread_scope_block` | Threads in the same CUDA block |
| `cuda::thread_scope_device` | Threads on the same CUDA device |
| `cuda::thread_scope_system` | System-visible synchronization where supported |

Choose the narrowest scope that is correct. Wider scope can imply stronger visibility requirements and higher cost.

```cpp
#include <cuda/atomic>

__global__ void publish(int* flag)
{
    cuda::atomic_ref<int, cuda::thread_scope_device> ready(*flag);
    ready.store(1, cuda::std::memory_order_release);
}
```

## Asynchronous copy and pipelines

Device-side asynchronous copy exists because high-performance kernels often stage data from global memory into shared memory while other work continues. NVIDIA documents `cuda::memcpy_async` as a key abstraction for asynchronous movement between global and shared memory, including hardware-backed paths on newer architectures.

Use this family when a kernel has a clear producer/consumer staging pattern. For a simple first CUDA program, ordinary runtime copies and shared-memory loads are easier to reason about.

## Fancy iterators

Fancy iterators represent generated, transformed, zipped, permuted, or strided sequences. They are useful when writing generic algorithms or when interoperating with CCCL components such as Thrust and CUB.

Examples include:

- `cuda::counting_iterator` for generated integer sequences.
- `cuda::transform_iterator` for applying a function on dereference.
- `cuda::zip_iterator` for traversing multiple ranges together.
- `cuda::strided_iterator` for non-contiguous traversal.

## Runtime abstractions

The Runtime API section in libcu++ provides C++ wrappers around CUDA Runtime concepts such as streams, events, devices, launches, buffers, and memory pools. These do not replace knowing the underlying CUDA Runtime API; they provide typed C++ building blocks for codebases that want resource wrappers and generic composition.

## Experimental APIs

Some libcu++ APIs are under `cuda::experimental`. Treat them as NVIDIA-provided experimental interfaces rather than stable ISO C++ facilities. They can be useful for advanced CUDA work, but they deserve tighter version pinning and more careful upgrade review than ordinary standard-like utilities.

## External references

- [NVIDIA libcu++ Extended API](https://nvidia.github.io/cccl/libcudacxx/extended_api.html)
- [NVIDIA libcu++ Runtime API](https://nvidia.github.io/cccl/libcudacxx/extended_api/runtime.html)
- [NVIDIA libcu++ API reference](https://nvidia.github.io/cccl/libcudacxx/api.html)
