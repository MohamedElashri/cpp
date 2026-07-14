---
title: "CUDA kernel"
description: "A CUDA C++ function launched by host code and executed by many GPU threads."
source_path: "cpp/cuda/kernel"
---

A CUDA kernel is a function that runs on the GPU device. Host code launches a kernel with an execution configuration, and the CUDA runtime creates many device threads that all execute the same kernel body.

Kernel functions are declared with the [`__global__`](../function_qualifiers/) qualifier.

## Declaration

```cpp
__global__ void kernel_name(parameter-list);
```

A `__global__` function:

- is callable from host code;
- executes on the device;
- must return `void`;
- is launched with CUDA's execution configuration syntax.

## Launch

```cpp
kernel_name<<<grid_dim, block_dim>>>(arguments);
```

The launch is asynchronous with respect to the host in the usual case. The host thread may continue before the kernel has finished, so host code normally uses `cudaDeviceSynchronize()`, stream synchronization, events, or a later blocking operation when it must observe completion.

## Per-thread work

Every thread runs the same function body. The kernel usually maps the current thread to a data element by combining built-in indices:

```cpp
__global__ void scale(float* values, int n, float factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        values[i] *= factor;
}
```

The bounds check is part of the normal pattern. Launch dimensions are often rounded up to a convenient block size, so some threads may not correspond to valid input elements.

## Notes

- Kernel parameters are copied from host code into the launch.
- Kernel code cannot directly access ordinary host stack or heap objects.
- Device-side failures are reported through the CUDA runtime error state and should be checked after launches.
- A kernel launch only describes work; completion is a separate synchronization question.

## See also

- [Execution configuration](../execution_configuration/)
- [Thread hierarchy](../thread_hierarchy/)
- [Function qualifiers](../function_qualifiers/)
- [Runtime API basics](../runtime_api/)
