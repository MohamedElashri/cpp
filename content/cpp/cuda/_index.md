---
title: "CUDA C++"
description: "CUDA C++ language extensions and programming model basics for writing GPU kernels from C++."
source_path: "cpp/cuda"
---

CUDA C++ extends C++ with syntax and APIs for launching work on NVIDIA GPUs. It keeps a split execution model: host code runs on the CPU, while device code runs on the GPU as many lightweight threads organized into blocks and grids.

This section starts with the core concepts needed to read and write simple CUDA C++ code. It is not a replacement for NVIDIA's full documentation; it is a compact reference layer that connects CUDA vocabulary to familiar C++ ideas.

## Start here

| Topic | Use it for |
| --- | --- |
| [Kernels](kernel/) | Functions launched from host code and executed by GPU threads |
| [Execution configuration](execution_configuration/) | The `<<<grid, block>>>` launch syntax and launch dimensions |
| [Thread hierarchy](thread_hierarchy/) | Grids, blocks, threads, warps, and built-in index variables |
| [Memory model](memory/) | Host memory, device memory, shared memory, local memory, and transfers |
| [Function qualifiers](function_qualifiers/) | `__global__`, `__device__`, and `__host__` |
| [Runtime API basics](runtime_api/) | Allocation, copies, synchronization, and error checking |

## Minimal program shape

```cpp
#include <cuda_runtime.h>

__global__ void add_one(int* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] += 1;
}

int main()
{
    int* device_data = nullptr;
    int n = 1024;

    cudaMalloc(&device_data, n * sizeof(int));
    add_one<<<(n + 255) / 256, 256>>>(device_data, n);
    cudaDeviceSynchronize();
    cudaFree(device_data);
}
```

The important pieces are the kernel declaration, the launch configuration, the per-thread index calculation, and the explicit synchronization before observing completion on the host.

## Scope

The starter pages focus on CUDA C++ language and runtime fundamentals:

- how kernels are declared and launched;
- how GPU work is divided into grids, blocks, and threads;
- how common memory spaces affect visibility and lifetime;
- how host code coordinates allocation, transfer, launch, and synchronization.

Advanced CUDA features such as streams, events, graphs, cooperative groups, dynamic parallelism, texture memory, and library-specific APIs can be added later as separate pages.

## External references

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [NVIDIA CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
