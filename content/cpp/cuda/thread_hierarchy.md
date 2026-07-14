---
title: "CUDA thread hierarchy"
description: "How CUDA organizes GPU work into grids, blocks, threads, and warps."
source_path: "cpp/cuda/thread_hierarchy"
---

CUDA kernels execute over a hierarchy of threads. A kernel launch creates a grid. A grid contains blocks. A block contains threads. Every thread executes the same kernel function and uses its indices to choose which data to operate on.

## Hierarchy

| Level | Description |
| --- | --- |
| Grid | The full set of blocks launched for one kernel call |
| Block | A group of threads that can cooperate through shared memory and block-level synchronization |
| Thread | One execution instance of the kernel body |
| Warp | A hardware scheduling group of threads within a block |

Threads in the same block can use shared memory and synchronize with `__syncthreads()`. Threads in different blocks cannot synchronize with each other inside an ordinary kernel launch.

## Built-in variables

| Variable | Meaning |
| --- | --- |
| `gridDim` | Dimensions of the grid, in blocks |
| `blockIdx` | Index of the current block within the grid |
| `blockDim` | Dimensions of each block, in threads |
| `threadIdx` | Index of the current thread within its block |
| `warpSize` | Number of threads in a warp |

The dimension variables are `dim3`-like values with `.x`, `.y`, and `.z` members.

## One-dimensional index

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This is the standard mapping for a one-dimensional array. The block index selects the block-sized chunk, and the thread index selects an element inside the chunk.

## Two-dimensional index

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

This mapping is common for images, matrices, and grids. The kernel should still check that `x` and `y` are inside the actual data bounds.

## Notes

- Blocks are scheduled independently. Code should not assume an ordering between blocks.
- A block runs on one streaming multiprocessor, which allows threads in that block to share block-local resources.
- Warp-level behavior matters for performance, especially branch divergence and memory access patterns, but correctness should first be expressed at the thread and block level.

## See also

- [Execution configuration](../execution_configuration/)
- [Memory model](../memory/)
- [Kernels](../kernel/)
