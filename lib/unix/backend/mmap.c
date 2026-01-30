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
    unsigned char *m_buf = fetcher->m_content[iofile];
    madvise(m_buf + offset, dst->size, MADV_WILLNEED | MADV_SEQUENTIAL);

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

    backend->m_content = calloc(sizeof(unsigned char *), ctx->num_sources);
    if (!backend->m_content) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        goto exit;
    }

    for (size_t i = 0; i < ctx->num_sources; i++) {
        const struct hmll_source src = ctx->sources[i];
        unsigned char *buf;
        if ((buf = mmap(0, src.size, PROT_READ, MAP_PRIVATE, src.fd, 0)) == MAP_FAILED) {
            free(backend->m_content);
            ctx->error = HMLL_ERR(HMLL_ERR_MMAP_FAILED);
            goto exit;
        }

        madvise(buf, src.size, MADV_WILLNEED | MADV_HUGEPAGE);
        backend->m_content[i] = buf;
    }

    ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
    ctx->fetcher->kind = HMLL_FETCHER_MMAP;
    ctx->fetcher->device = device;
    ctx->fetcher->backend_impl_ = backend;
    ctx->fetcher->fetch_range_impl_ = hmll_mmap_fetch_range_impl;
    ctx->fetcher->fetchv_range_impl_ = NULL;

exit:
    return ctx->error;
}
