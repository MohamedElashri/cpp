---
title: "CUDA Runtime API basics"
description: "The basic CUDA Runtime API calls used to allocate device memory, copy data, launch kernels, and check errors."
source_path: "cpp/cuda/runtime_api"
---

The CUDA Runtime API is the common C/C++ API used by host code to manage device memory, launch kernels, synchronize work, and inspect errors.

## Basic flow

```cpp
cudaMalloc(&device_ptr, bytes);
cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);

kernel<<<blocks, threads>>>(device_ptr, n);
cudaDeviceSynchronize();

cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
cudaFree(device_ptr);
```

This sequence allocates device memory, copies input data to the device, launches work, waits for completion, copies results back, and releases the allocation.

## Common calls

| Function | Purpose |
| --- | --- |
| `cudaMalloc` | Allocate device memory |
| `cudaFree` | Release device memory |
| `cudaMemcpy` | Copy memory between host and device, or between device locations |
| `cudaMemset` | Fill device memory with a byte value |
| `cudaDeviceSynchronize` | Wait until preceding work on the device has completed |
| `cudaGetLastError` | Return and clear the last runtime error |
| `cudaPeekAtLastError` | Return the last runtime error without clearing it |
| `cudaGetErrorString` | Convert a runtime error code to a string |

## Error checking

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Handle or report cudaGetErrorString(err)
}
```

Kernel launch errors can be reported separately from execution errors. A common starter pattern is to check the launch with `cudaGetLastError()` and then check `cudaDeviceSynchronize()` when the host needs to wait for the kernel to finish.

```cpp
kernel<<<blocks, threads>>>(device_ptr, n);

cudaError_t launch_error = cudaGetLastError();
if (launch_error != cudaSuccess) {
    // Invalid launch configuration or argument problem.
}

cudaError_t execution_error = cudaDeviceSynchronize();
if (execution_error != cudaSuccess) {
    // Runtime failure while executing the kernel.
}
```

## Notes

- Many runtime calls return `cudaError_t`; production code should check it.
- `cudaMemcpy` with pageable host memory is usually synchronous for the host in the common blocking form.
- Asynchronous copies, streams, events, and pinned host memory are important next steps, but they are separate concepts from this minimal runtime flow.

## See also

- [Memory model](../memory/)
- [Kernels](../kernel/)
- [Execution configuration](../execution_configuration/)
