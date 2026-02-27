//
// Created by mfuntowicz on 1/25/26.
//

#ifdef _WIN32
#include <windows.h>
#include "hmll/hmll.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#endif


void *hmll_alloc(const size_t size, const struct hmll_device device, const int flags)
{
    void *ptr = 0;
#if !defined(__HMLL_CUDA_ENABLED__)
    HMLL_UNUSED(flags);
#endif
    if (hmll_device_is_cpu(device)) {
        // Use VirtualAlloc with MEM_COMMIT | MEM_RESERVE for Windows
        ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (ptr == NULL) {
            return 0;
        }
        return ptr;
    }

#if defined(__HMLL_CUDA_ENABLED__)
    if (hmll_device_is_cuda(device) && flags == HMLL_MEM_DEVICE)
        cudaMalloc(&ptr, size);

    if (hmll_device_is_cuda(device) && flags == HMLL_MEM_STAGING)
        cudaHostAlloc(&ptr, size, cudaHostAllocDefault | cudaHostAllocPortable);

#endif

    return ptr;
}

void hmll_free_buffer(struct hmll_iobuf *buffer)
{
    if (!buffer) return;

#if defined(__HMLL_CUDA_ENABLED__)
    if (hmll_device_is_cuda(buffer->device)) {
        cudaFreeHost(buffer->ptr);
    }
    else
#endif
    if (hmll_device_is_cpu(buffer->device)) {
        VirtualFree(buffer->ptr, 0, MEM_RELEASE);
    }

    buffer->ptr = NULL;
    buffer->size = 0;
}

struct hmll_iobuf hmll_get_buffer(struct hmll *ctx, const size_t size, const int flags)
{
    struct hmll_device device = ctx->fetcher->device;

    void* ptr = NULL;
#if !defined(__HMLL_CUDA_ENABLED__)
    HMLL_UNUSED(flags);
#endif

    switch (device.kind)
    {
    case HMLL_DEVICE_CPU:
        ptr = hmll_alloc(size, device, HMLL_MEM_DEVICE);
        break;

    case HMLL_DEVICE_CUDA:
        {
#if defined(__HMLL_CUDA_ENABLED__)
            cudaError_t cuda_err = cudaSetDevice(device.idx);
            if (cuda_err != cudaSuccess) {
                ctx->error = HMLL_ERR(HMLL_ERR_CUDA_SET_DEVICE_FAILED);
                return (struct hmll_iobuf){0};
            }

            ptr = hmll_alloc(size, device, flags);
            if (!ptr) {
                ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
                return (struct hmll_iobuf){0};
            }

            break;
#else
            ctx->error = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
            return (struct hmll_iobuf){0};
#endif
        }
    }

    return (struct hmll_iobuf){size, ptr, device};
}

#endif
