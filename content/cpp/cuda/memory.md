---
title: "CUDA memory model"
description: "The common CUDA C++ memory spaces used by host code, device code, and GPU threads."
source_path: "cpp/cuda/memory"
---

CUDA programs usually deal with memory owned by the host CPU, memory owned by the GPU device, and several device-side memory spaces with different visibility and lifetime rules.

## Common memory spaces

| Memory space | Visible to | Typical use |
| --- | --- | --- |
| Host memory | CPU host code | Ordinary C++ objects and buffers |
| Global memory | Device threads, through device pointers | Large arrays and long-lived device data |
| Shared memory | Threads in the same block | Block-local cooperation and data reuse |
| Local memory | One device thread | Thread-private spills or arrays not kept in registers |
| Constant memory | Device threads | Read-only data with specialized caching behavior |

Host code cannot normally dereference a device pointer directly, and device code cannot normally dereference an arbitrary host pointer. Data movement and allocation are coordinated through the CUDA runtime or driver APIs.

## Runtime allocation and transfer

```cpp
float* device_values = nullptr;

cudaMalloc(&device_values, n * sizeof(float));
cudaMemcpy(device_values, host_values,
           n * sizeof(float),
           cudaMemcpyHostToDevice);

scale<<<blocks, threads>>>(device_values, n, factor);

cudaMemcpy(host_values, device_values,
           n * sizeof(float),
           cudaMemcpyDeviceToHost);
cudaFree(device_values);
```

This explicit allocation-and-copy pattern is the baseline model. Unified memory and other allocation APIs can reduce manual transfer management, but they do not remove the need to reason about where data is used and when work completes.

## Shared memory

Shared memory is allocated per block and is visible only to threads in that block.

```cpp
__global__ void use_shared(float* out)
{
    __shared__ float scratch[256];

    scratch[threadIdx.x] = 1.0f;
    __syncthreads();

    out[threadIdx.x] = scratch[threadIdx.x];
}
```

`__syncthreads()` is a block-level barrier. It is commonly needed after one set of threads writes shared memory and before another set of threads reads those values.

## Notes

- Global memory is large but has higher latency than registers or shared memory.
- Coalesced memory accesses are important for throughput.
- Shared memory can improve locality, but it is limited and can reduce occupancy if overused.
- Synchronization is part of memory correctness. A copy back to the host before a kernel has completed can observe incomplete results.

## See also

- [Runtime API basics](../runtime_api/)
- [Thread hierarchy](../thread_hierarchy/)
- [Execution configuration](../execution_configuration/)
