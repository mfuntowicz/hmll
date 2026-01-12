//
// Created by mfuntowicz on 12/2/25.
//
#include "hmll/memory.h"
#include "hmll/types.h"
#include <linux/mman.h>
#include <sys/mman.h>

#include "hmll/hmll.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#endif


static inline void *hmll_get_buffer_with_flags(const size_t size, const int flags)
{
    return mmap(0, size, PROT_READ | PROT_WRITE, flags, -1, 0);
}


void hmll_free_buffer(struct hmll_iobuf *buffer)
{
    if (buffer == NULL) return;

#if defined(__HMLL_CUDA_ENABLED__)
    if (buffer->device == HMLL_DEVICE_CUDA) cudaFreeHost(buffer->ptr);
#endif

    if (buffer->device == HMLL_DEVICE_CPU) munmap(buffer->ptr, buffer->size);

    buffer->ptr = NULL;
    buffer->size = 0;
}

void *hmll_get_buffer(struct hmll *ctx, const enum hmll_device device, const size_t size, const int flags)
{
    void* ptr = NULL;

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
    switch (device)
    {
    case HMLL_DEVICE_CPU:
        if ((ptr = hmll_get_buffer_with_flags(size, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE| MAP_HUGETLB | MAP_HUGE_8MB)) == MAP_FAILED) {
            if((ptr = hmll_get_buffer_with_flags(size, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE)) == MAP_FAILED) {
                ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
                return NULL;
            }
        }
        break;

    case HMLL_DEVICE_CUDA:
#if defined(__HMLL_CUDA_ENABLED__)
        ;
        enum cudaError error;
        if (flags == HMLL_MEM_DEVICE)
            error = cudaMalloc(&ptr, size);
        else
            error = cudaHostAlloc(&ptr, size, cudaHostAllocDefault | cudaHostAllocPortable);

        if (error != cudaSuccess) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            return NULL;
        }

        break;
#else
        ctx->error = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
#endif
    }
#endif
    return ptr;
}

struct hmll_iobuf hmll_get_buffer_for_range(struct hmll *ctx, const enum hmll_device device, const struct hmll_range range)
{
    if (hmll_check(ctx->error))
        return (struct hmll_iobuf) {0};

    const size_t size = hmll_range_size(range);
    void *ptr = hmll_get_buffer(ctx, device, size, HMLL_MEM_DEVICE);
    if (hmll_check(ctx->error))
        return (struct hmll_iobuf) {0};

    return (struct hmll_iobuf) {size, ptr, device};
}
