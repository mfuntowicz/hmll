//
// Created by mfuntowicz on 1/13/26.
//

#ifdef __unix__
#include <linux/mman.h>
#include <sys/mman.h>
#include "hmll/hmll.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#endif


void *hmll_alloc(const size_t size, const enum hmll_device device, const int flags)
{
#define HMLL_MAP_DEFAULT (MAP_PRIVATE | MAP_ANONYMOUS)

    void *ptr = 0;
    if (device == HMLL_DEVICE_CPU) {
        if ((ptr = mmap(0, size, PROT_READ | PROT_WRITE, HMLL_MAP_DEFAULT | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0)) == MAP_FAILED) {
            if ((ptr = mmap(0, size, PROT_READ | PROT_WRITE, HMLL_MAP_DEFAULT, -1, 0)) != MAP_FAILED)
                madvise(ptr, size, MADV_HUGEPAGE);
            else
                ptr = 0;
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
    if (buffer->device == HMLL_DEVICE_CUDA) cudaFreeHost(buffer->ptr);
#endif

    if (buffer->device == HMLL_DEVICE_CPU) munmap(buffer->ptr, buffer->size);

    buffer->ptr = NULL;
    buffer->size = 0;
}

struct hmll_iobuf hmll_get_buffer(struct hmll *ctx, const enum hmll_device device, const size_t size, const int flags)
{
    void* ptr = NULL;

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
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
#endif
    return (struct hmll_iobuf){size, ptr, device};
}

#endif