---
title: "CUDA execution configuration"
description: "The CUDA C++ launch syntax that selects grid and block dimensions for a kernel."
source_path: "cpp/cuda/execution_configuration"
---

An execution configuration specifies how many GPU threads are created for a kernel launch and how those threads are grouped.

## Syntax

```cpp
kernel<<<grid_dim, block_dim>>>(args);
kernel<<<grid_dim, block_dim, shared_memory_bytes, stream>>>(args);
```

The first two arguments are the common form:

| Argument | Meaning |
| --- | --- |
| `grid_dim` | Number of blocks in the grid |
| `block_dim` | Number of threads in each block |

The optional third argument requests dynamically allocated shared memory per block. The optional fourth argument selects the CUDA stream used for the launch.

## Dimensions

`grid_dim` and `block_dim` may be integers for one-dimensional launches or `dim3` values for one-, two-, or three-dimensional launches.

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);

kernel2d<<<grid, block>>>(image, width, height);
```

Inside the kernel, each thread can inspect its location through [`blockIdx`, `blockDim`, and `threadIdx`](../thread_hierarchy/).

## One-dimensional indexing

```cpp
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;

scale<<<blocks, threads_per_block>>>(values, n, factor);
```

The division rounds up so that every element has at least one possible thread. The kernel still checks `i < n` because the final block may contain extra threads.

## Choosing a starter block size

For simple element-wise kernels, a block size such as 128, 256, or 512 threads is a common starting point. The best value depends on the GPU, register use, shared memory use, occupancy, memory access pattern, and instruction mix.

## See also

- [Kernels](../kernel/)
- [Thread hierarchy](../thread_hierarchy/)
- [Memory model](../memory/)
