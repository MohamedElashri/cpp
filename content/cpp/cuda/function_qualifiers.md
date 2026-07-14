---
title: "CUDA function qualifiers"
description: "CUDA C++ qualifiers that mark whether a function runs on the host, the device, or both."
source_path: "cpp/cuda/function_qualifiers"
---

CUDA C++ uses function qualifiers to describe where a function is called from and where it executes.

## Qualifiers

| Qualifier | Called from | Executes on | Common use |
| --- | --- | --- | --- |
| `__global__` | Host code | Device | Kernel entry point |
| `__device__` | Device code | Device | Helper function for kernels |
| `__host__` | Host code | Host | Explicit host-side function marker |
| `__host__ __device__` | Host or device code | Host or device | Small utility usable in both contexts |

## Kernel function

```cpp
__global__ void fill(int* data, int n, int value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = value;
}
```

A `__global__` function is launched with an execution configuration and must return `void`.

## Device helper

```cpp
__device__ int clamp_nonnegative(int value)
{
    return value < 0 ? 0 : value;
}

__global__ void clamp_all(int* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = clamp_nonnegative(data[i]);
}
```

A `__device__` function is not a kernel launch target. It is called by device code.

## Host and device utility

```cpp
__host__ __device__ int square(int x)
{
    return x * x;
}
```

A host/device function is compiled for both sides. Its body must be valid in both compilation contexts, so it should avoid APIs or language features that are available only on one side unless guarded by conditional compilation.

## Notes

- Ordinary unqualified functions are host functions.
- A `__global__` function represents a kernel entry point, not a normal device helper.
- Host/device functions are useful for small math and indexing helpers, but large shared abstractions can become hard to maintain if host and device constraints diverge.

## See also

- [Kernels](../kernel/)
- [Execution configuration](../execution_configuration/)
- [Runtime API basics](../runtime_api/)
