//
// Created by mfuntowicz on 1/25/26.
//

#ifdef _WIN32
#include <windows.h>
#include "hmll/hmll.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#endif


void *hmll_alloc(const size_t size, const enum hmll_device device, const int flags)
{
    void *ptr = 0;
    if (device == HMLL_DEVICE_CPU) {
        // Use VirtualAlloc with MEM_COMMIT | MEM_RESERVE for Windows
        ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (ptr == NULL) {
            return 0;
        }
        return ptr;
    }

#if defined(__HMLL_CUDA_ENABLED__)
    if (device == HMLL_DEVICE_CUDA && flags == HMLL_MEM_DEVICE)
        cudaMalloc(&ptr, size);

    if (device == HMLL_DEVICE_CUDA && flags == HMLL_MEM_STAGING)
        cudaHostAlloc(&ptr, size, cudaHostAllocDefault | cudaHostAllocPortable);

#endif

    return ptr;
}

void hmll_free_buffer(struct hmll_iobuf *buffer)
{
    if (!buffer) return;

#if defined(__HMLL_CUDA_ENABLED__)
    if (buffer->device == HMLL_DEVICE_CUDA) {
        cudaFreeHost(buffer->ptr);
    }
    else
#endif
    if (buffer->device == HMLL_DEVICE_CPU) {
        VirtualFree(buffer->ptr, 0, MEM_RELEASE);
    }

    buffer->ptr = NULL;
    buffer->size = 0;
}

struct hmll_iobuf hmll_get_buffer(struct hmll *ctx, const enum hmll_device device, const size_t size, const int flags)
{
    void* ptr = NULL;

    switch (device)
    {
    case HMLL_DEVICE_CPU:
        ptr = hmll_alloc(size, device, HMLL_MEM_DEVICE);
        break;

    case HMLL_DEVICE_CUDA:
#if defined(__HMLL_CUDA_ENABLED__)
        ptr = hmll_alloc(size, device, flags);
        if (!ptr) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            return (struct hmll_iobuf){0};
        }

        break;
#else
        ctx->error = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
#endif
    }

    return (struct hmll_iobuf){size, ptr, device};
}

#endif