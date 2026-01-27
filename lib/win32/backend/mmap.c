#include "hmll/win32/backend/mmap.h"
#include <memoryapi.h>
#include <windows.h>
#include <stdlib.h>
#include <string.h>

#ifdef __HMLL_CUDA_ENABLED__
#include "cuda_runtime.h"
#endif

static ssize_t hmll_mmap_fetch_range_impl(
    struct hmll *ctx, const int iofile, const struct hmll_iobuf *dst, const struct hmll_range range)
{
    if (hmll_check(ctx->error)) return -1;

    const struct hmll_mmap *fetcher = ctx->fetcher->backend_impl_;

    unsigned char *m_buf = fetcher->m_content[iofile];
    const size_t n_bytes = range.end - range.start;

    if (dst->size < n_bytes) {
        ctx->error = HMLL_ERR(HMLL_ERR_BUFFER_TOO_SMALL);
        return -1;
    }

#ifdef __HMLL_CUDA_ENABLED__
    if (ctx->fetcher->device == HMLL_DEVICE_CUDA) {
        const void *p_src = (void *) ((uintptr_t)m_buf + range.start);
        cudaMemcpy(dst->ptr, p_src, n_bytes, cudaMemcpyHostToDevice);
    } else {
        memcpy(dst->ptr, m_buf + range.start, n_bytes);
    }
#else
    memcpy(dst->ptr, m_buf + range.start, n_bytes);
#endif

    return (ssize_t) n_bytes;
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
        free(backend);
        goto exit;
    }

    backend->n = 0;

    for (size_t i = 0; i < ctx->num_sources; i++) {
        const struct hmll_source src = ctx->sources[i];

        // Create file mapping using older, more compatible API
        const HANDLE h_mapping = CreateFileMappingA(
            src.handle,
            NULL,
            PAGE_READONLY,
            0,
            0,
            NULL
        );

        if (!h_mapping) {
            ctx->error = HMLL_SYS_ERR(GetLastError());
            goto cleanup_mappings;
        }

        unsigned char *buf = MapViewOfFile(
            h_mapping,
            FILE_MAP_READ,
            0,
            0,
            (SIZE_T)src.size
        );

        CloseHandle(h_mapping);

        if (buf == NULL) {
            ctx->error = HMLL_SYS_ERR(GetLastError());
            goto cleanup_mappings;
        }

        backend->m_content[i] = buf;
        backend->n++;
    }

    ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
    ctx->fetcher->kind = HMLL_FETCHER_MMAP;
    ctx->fetcher->device = device;
    ctx->fetcher->backend_impl_ = backend;
    ctx->fetcher->fetch_range_impl_ = hmll_mmap_fetch_range_impl;
    ctx->fetcher->fetchv_range_impl_ = NULL;
    goto exit;

cleanup_mappings:
    hmll_mmap_free(backend);

exit:
    return ctx->error;
}