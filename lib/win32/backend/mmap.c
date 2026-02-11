#include "hmll/win32/backend/mmap.h"
#include <windows.h>
#include <memoryapi.h>
#include <stdlib.h>
#include <string.h>

// PrefetchVirtualMemory requires Windows 8+ (_WIN32_WINNT >= 0x0602)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif

#ifdef __HMLL_CUDA_ENABLED__
#include "cuda_runtime.h"
#endif

static ssize_t hmll_mmap_fetch_range_impl(
    struct hmll *ctx, const int iofile, const struct hmll_iobuf *dst, const size_t offset)
{
    if (hmll_check(ctx->error)) return -1;
    if (dst->size == 0) return 0;

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    const unsigned char *m_buf = fetcher->m_content[iofile];

#ifdef __HMLL_CUDA_ENABLED__
    if (ctx->fetcher->device == HMLL_DEVICE_CUDA) {
        cudaMemcpy(dst->ptr, m_buf + offset, dst->size, cudaMemcpyHostToDevice);
    } else {
        memcpy(dst->ptr, m_buf + offset, dst->size);
    }
#else
    memcpy(dst->ptr, m_buf + offset, dst->size);
#endif

    return (ssize_t) dst->size;
}

static inline void hmll_mmap_free(void *ptr)
{
    if (ptr) {
        struct hmll_mmap *backend = ptr;
        if (backend->m_content) {
            // Note: backend doesn't own the memory, sources do
            free(backend->m_content);
        }
        free(ptr);
    }
}

struct hmll_error hmll_mmap_init(struct hmll *ctx, const enum hmll_device device)
{
    if (hmll_check(ctx->error)) goto exit;
    if (ctx->num_sources <= 0 || !ctx->sources) {
        ctx->error = HMLL_ERR(HMLL_ERR_NO_SOURCE_PROVIDED);
        goto exit;
    }

    struct hmll_mmap *backend = calloc(1, sizeof(struct hmll_mmap));
    if (!backend) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        goto exit;
    }

    backend->m_content = calloc(ctx->num_sources, sizeof(const unsigned char *));
    if (!backend->m_content) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        free(backend);
        goto exit;
    }

    for (size_t i = 0; i < ctx->num_sources; i++) {
        const struct hmll_source src = ctx->sources[i];
        // Simply reference the already-mapped content from the source
        backend->m_content[i] = src.content;
    }

    backend->n = ctx->num_sources;

    ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
    ctx->fetcher->kind = HMLL_FETCHER_MMAP;
    ctx->fetcher->device = device;
    ctx->fetcher->backend_free = hmll_mmap_free;
    ctx->fetcher->backend_impl_ = backend;
    ctx->fetcher->fetch_range_impl_ = hmll_mmap_fetch_range_impl;
    ctx->fetcher->fetchv_range_impl_ = NULL;

exit:
    return ctx->error;
}

struct hmll_error hmll_mmap_get_view(struct hmll *ctx, const int iofile, const struct hmll_range range, struct hmll_iobuf *out_view)
{
    if (hmll_check(ctx->error)) return ctx->error;

    // Only supported for mmap backend
    if (!ctx->fetcher || ctx->fetcher->kind != HMLL_FETCHER_MMAP) {
        return HMLL_ERR(HMLL_ERR_UNSUPPORTED_PLATFORM);
    }

    // Only supported for CPU device (GPU needs to copy)
    if (ctx->fetcher->device != HMLL_DEVICE_CPU) {
        return HMLL_ERR(HMLL_ERR_UNSUPPORTED_DEVICE);
    }

    // Validate file index
    if (iofile < 0 || (size_t)iofile >= ctx->num_sources) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    unsigned char *m_buf = fetcher->m_content[iofile];
    const size_t n_bytes = range.end - range.start;

    // Prefetch the memory range (Windows 8+ equivalent of madvise MADV_WILLNEED)
    WIN32_MEMORY_RANGE_ENTRY range_entry;
    range_entry.VirtualAddress = m_buf + range.start;
    range_entry.NumberOfBytes = n_bytes;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range_entry, 0);

    // Return a view directly into the mmap'd region
    out_view->ptr = m_buf + range.start;
    out_view->size = n_bytes;
    out_view->device = HMLL_DEVICE_CPU;

    return HMLL_OK;
}

#ifdef __HMLL_CUDA_ENABLED__
ssize_t hmll_mmap_fetch_async(
    struct hmll *ctx,
    int iofile,
    struct hmll_iobuf *dst,
    size_t offset,
    cudaStream_t stream,
    cudaEvent_t done_event
) {
    if (hmll_check(ctx->error)) return -1;
    if (!dst || dst->size == 0) return 0;

    // Validate file index
    if (iofile < 0 || (size_t)iofile >= ctx->num_sources) {
        ctx->error = HMLL_ERR(HMLL_ERR_INVALID_RANGE);
        return -1;
    }

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    const unsigned char *m_buf = fetcher->m_content[iofile];

    // Prefetch the memory range (Windows equivalent of madvise MADV_WILLNEED)
    WIN32_MEMORY_RANGE_ENTRY range_entry;
    range_entry.VirtualAddress = (PVOID)(m_buf + offset);
    range_entry.NumberOfBytes = dst->size;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range_entry, 0);

    // Async copy from mmap to GPU
    cudaError_t err = cudaMemcpyAsync(
        dst->ptr,
        m_buf + offset,
        dst->size,
        cudaMemcpyHostToDevice,
        stream
    );
    if (err != cudaSuccess) {
        ctx->error = HMLL_ERR(HMLL_ERR_CUDA_ERROR);
        return -1;
    }

    // Record completion event if provided
    if (done_event) {
        err = cudaEventRecord(done_event, stream);
        if (err != cudaSuccess) {
            ctx->error = HMLL_ERR(HMLL_ERR_CUDA_ERROR);
            return -1;
        }
    }

    return (ssize_t)dst->size;
}

const void* hmll_mmap_get_content_ptr(struct hmll *ctx, int iofile) {
    if (!ctx || !ctx->fetcher || ctx->fetcher->kind != HMLL_FETCHER_MMAP) {
        return NULL;
    }

    if (iofile < 0 || (size_t)iofile >= ctx->num_sources) {
        return NULL;
    }

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    return fetcher->m_content[iofile];
}
#endif // __HMLL_CUDA_ENABLED__