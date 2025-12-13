//
// Created by mfuntowicz on 12/2/25.
//
#include <linux/mman.h>
#include <sys/mman.h>
#include <stdlib.h>
#include "hmll/types.h"


void *hmll_get_buffer(struct hmll_context *ctx, const size_t size)
{
    void* ptr = 0;
    ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (ptr == MAP_FAILED) {
        ctx->error = HMLL_ERR_ALLOCATION_FAILED;
        return NULL;
    }

    return ptr;
}


void *hmll_get_io_buffer(struct hmll_context *ctx, const enum hmll_device device, const size_t size)
{
    void* ptr = 0;
    switch (device)
    {
    case HMLL_DEVICE_CPU:
        // Try to allocate with 2MB huge pages first
        ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);

        // If huge pages fail, fall back to regular anonymous mmap
        if (ptr == MAP_FAILED)
            ptr = hmll_get_buffer(ctx, size);

        return ptr;

    default:
        ctx->error = HMLL_ERR_ALLOCATION_FAILED;
        return NULL;
    }
}