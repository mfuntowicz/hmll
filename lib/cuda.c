//
// Created by mfuntowicz on 12/18/25.
//

#include "hmll/cuda.h"
#include <cuda_runtime.h>

int hmll_cuda_device_count(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) return count;
    return 0;
}

int hmll_cuda_allocation_flags(void)
{
    int device;
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int flags = cudaHostAllocDefault;
    if (prop.canMapHostMemory || prop.unifiedAddressing) flags = cudaHostAllocMapped;
    if (hmll_cuda_device_count() > 0) flags |= cudaHostAllocPortable;

    return flags;
}