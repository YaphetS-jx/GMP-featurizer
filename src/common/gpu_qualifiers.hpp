#pragma once

// CUDA qualifiers for cross-platform compatibility
#ifdef __CUDA_ARCH__
    #define GPU_HOST_DEVICE __host__ __device__
    #define GPU_DEVICE __device__
    #define GPU_HOST __host__
#else
    #define GPU_HOST_DEVICE
    #define GPU_DEVICE
    #define GPU_HOST
#endif 