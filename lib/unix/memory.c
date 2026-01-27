//
// Created by mfuntowicz on 1/13/26.
//
#include <sys/mman.h>
#include "hmll/hmll.h"

#if defined(__linux__)
#include "linux/mman.h"
#elif defined(__APPLE__)
#include "mach/mach_vm.h"
#endif
#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#endif

#if defined(__APPLE__)
#define HMLL_MAP_ANONYMOUS MAP_ANON
#define HMLL_MAP_HUGETLB VM_FLAGS_SUPERPAGE_SIZE_2MB
#else
#define HMLL_MAP_ANONYMOUS MAP_ANONYMOUS
#define HMLL_MAP_HUGETLB (MAP_HUGETLB | MAP_HUGE_2MB)
#endif

void *hmll_alloc(const size_t size, const enum hmll_device device, const int flags)
{
    void *ptr = 0;
    if (device == HMLL_DEVICE_CPU) {
        HMLL_UNUSED(flags);
        // Try huge pages first, fall back to regular mmap
        if ((ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | HMLL_MAP_ANONYMOUS | HMLL_MAP_HUGETLB, -1, 0)) == MAP_FAILED) {
            if ((ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | HMLL_MAP_ANONYMOUS, -1, 0)) != MAP_FAILED) {
#if defined(MADV_HUGEPAGE)
                madvise(ptr, size, MADV_HUGEPAGE);
#endif
            } else {
                ptr = 0;
            }
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
    switch (device)
    {
    case HMLL_DEVICE_CPU:
        HMLL_UNUSED(flags);
        ptr = hmll_alloc(size, device, HMLL_MEM_DEVICE);
        break;

    case HMLL_DEVICE_CUDA:
#if defined(__HMLL_CUDA_ENABLED__)
        ptr = hmll_alloc(size, device, flags);
        if (!ptr) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            return (struct hmll_iobuf){0};
        }
#else
        ctx->error = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
#endif
        break;
    }
    return (struct hmll_iobuf){size, ptr, device};
}
