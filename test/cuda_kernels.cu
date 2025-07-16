#include <cuda_runtime.h>

// Simple CUDA kernel to demonstrate pinned memory access
__global__ void access_pinned_memory_kernel(const int* host_data, int* device_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // This demonstrates that we can read from pinned host memory directly
        device_result[idx] = host_data[idx] * 2;
    }
} 