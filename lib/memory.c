//
// Created by mfuntowicz on 12/2/25.
//
#include <linux/mman.h>
#include <sys/mman.h>
#include <stdlib.h>
#include "hmll/types.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>

// We use Write Combined as we don't want the CPU tready of these bytes at any time
#define CUDA_HOST_ALLOC_FLAGS (cudaHostAllocMapped | cudaHostAllocPortable | cudaHostAllocWriteCombined)

#endif

void *hmll_get_buffer(struct hmll_context *ctx, const enum hmll_device device, const size_t size)
{
    void* ptr = NULL;

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
    switch (device)
    {
    case HMLL_DEVICE_CPU:
        ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            ctx->error = HMLL_ERR_ALLOCATION_FAILED;
            return NULL;
        }
        break;

#if defined(__HMLL_CUDA_ENABLED__)
    case HMLL_DEVICE_CUDA:
        ;
        enum cudaError error = {0};
        if ((error = cudaHostAlloc(&ptr, size, CUDA_HOST_ALLOC_FLAGS)) != cudaSuccess)
#if defined(DEBUG)
            printf("Failed to allocate CUDA paged-locked memory: %s\n", cudaGetErrorString(error));
#endif
        break;
#endif
    }

#else
    ctx->error = HMLL_ERR_UNSUPPORTED_PLATFORM;
#endif

    return ptr;
}

void *hmll_get_io_buffer(struct hmll_context *ctx, const enum hmll_device device, const size_t size)
{
    void *ptr = NULL;
    switch (device)
    {
    case HMLL_DEVICE_CPU:
        ;
        // MAP_POPULATE to avoid page fault on the first IO wri
        int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
        if (size > 2U * 1024 * 1024)  flags |= MAP_HUGETLB | MAP_HUGE_2MB;

        ptr = mmap(0, size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (ptr == MAP_FAILED) ptr = hmll_get_buffer(ctx, device, size);
        return ptr;

#if defined(__HMLL_CUDA_ENABLED__)
    case HMLL_DEVICE_CUDA:
        ;
        enum cudaError error = 0;
        if ((error = cudaHostAlloc(&ptr, size, cudaHostAllocMapped | cudaHostAllocWriteCombined)) != cudaSuccess)
#if defined(DEBUG)
            printf("Failed to allocate CUDA paged-locked memory: %s", cudaGetErrorString(error));
#endif
        return ptr;
#endif
    }

    ctx->error = HMLL_ERR_ALLOCATION_FAILED;
    return ptr;
}
