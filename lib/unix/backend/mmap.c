//
// Created by mfuntowicz on 1/23/26.
//

#include "hmll/unix/backend/mmap.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif

#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif

#ifdef __HMLL_CUDA_ENABLED__
#include "cuda_runtime.h"
#endif

static ssize_t
hmll_mmap_fetch_range_impl(struct hmll *ctx, const int iofile, const struct hmll_iobuf *dst, const size_t offset)
{
    if (hmll_check(ctx->error)) return -1;
    if (dst->size == 0) return 0;

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    const unsigned char *m_buf = fetcher->m_content[iofile];
    madvise((void *)(m_buf + offset), dst->size, MADV_WILLNEED | MADV_SEQUENTIAL);

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

static ssize_t
hmll_mmap_fetchv_range_impl(struct hmll *ctx, const int iofile, const struct hmll_iobuf *dsts, const size_t *offsets, const size_t n)
{
    if (hmll_check(ctx->error)) return -1;

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    const unsigned char *m_buf = fetcher->m_content[iofile];
    ssize_t total_bytes = 0;

    for (size_t i = 0; i < n; i++) {
        const struct hmll_iobuf *dst = &dsts[i];
        const size_t offset = offsets[i];
        const size_t n_bytes = dst->size;

        madvise((void *)(m_buf + offset), n_bytes, MADV_WILLNEED | MADV_SEQUENTIAL);

#ifdef __HMLL_CUDA_ENABLED__
        if (ctx->fetcher->device == HMLL_DEVICE_CUDA) {
            const void *p_src = (void *) ((uintptr_t)m_buf + offset);
            cudaMemcpy(dst->ptr, p_src, n_bytes, cudaMemcpyHostToDevice);
        } else {
            memcpy(dst->ptr, m_buf + offset, n_bytes);
        }
#else
        memcpy(dst->ptr, m_buf + offset, n_bytes);
#endif

        total_bytes += (ssize_t) n_bytes;
    }

    return total_bytes;
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
        goto exit;
    }

    for (size_t i = 0; i < ctx->num_sources; i++) {
        const struct hmll_source src = ctx->sources[i];
        // Simply reference the already-mapped content from the source
        backend->m_content[i] = src.content;
    }

    ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
    ctx->fetcher->kind = HMLL_FETCHER_MMAP;
    ctx->fetcher->device = device;
    ctx->fetcher->backend_impl_ = backend;
    ctx->fetcher->fetch_range_impl_ = hmll_mmap_fetch_range_impl;
    ctx->fetcher->fetchv_range_impl_ = hmll_mmap_fetchv_range_impl;

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

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;
    const unsigned char *m_buf = fetcher->m_content[iofile];
    const size_t n_bytes = range.end - range.start;

    // Issue madvise for the range
    madvise((void*)(m_buf + range.start), n_bytes, MADV_WILLNEED | MADV_SEQUENTIAL);

    // Return a view directly into the mmap'd region
    out_view->ptr = (void *)(m_buf + range.start);  // todo(mfuntowicz) change that const ellision ...
    out_view->size = n_bytes;
    out_view->device = HMLL_DEVICE_CPU;

    return HMLL_OK;
}
