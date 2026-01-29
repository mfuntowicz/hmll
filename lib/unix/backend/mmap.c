//
// Created by mfuntowicz on 1/23/26.
//

#include "hmll/unix/backend/mmap.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif

#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif

#ifdef __HMLL_CUDA_ENABLED__
#include "cuda_runtime.h"
#endif

// Threshold for parallel memcpy (64 MiB)
#define HMLL_PARALLEL_MEMCPY_THRESHOLD (64ULL * 1024 * 1024)

struct memcpy_chunk {
    void *dst;
    const void *src;
    size_t size;
};

static void *memcpy_thread(void *arg)
{
    struct memcpy_chunk *chunk = (struct memcpy_chunk *)arg;
    memcpy(chunk->dst, chunk->src, chunk->size);
    return NULL;
}

static void parallel_memcpy(void *dst, const void *src, size_t n_bytes)
{
    const int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads <= 1 || n_bytes < HMLL_PARALLEL_MEMCPY_THRESHOLD) {
        memcpy(dst, src, n_bytes);
        return;
    }

    const int actual_threads = num_threads > 16 ? 16 : num_threads;
    const size_t chunk_size = n_bytes / actual_threads;
    const size_t remainder = n_bytes % actual_threads;

    pthread_t threads[16];
    struct memcpy_chunk chunks[16];

    for (int i = 0; i < actual_threads; i++) {
        const size_t offset = i * chunk_size;
        const size_t size = (i == actual_threads - 1) ? (chunk_size + remainder) : chunk_size;

        chunks[i].dst = (unsigned char *)dst + offset;
        chunks[i].src = (const unsigned char *)src + offset;
        chunks[i].size = size;

        pthread_create(&threads[i], NULL, memcpy_thread, &chunks[i]);
    }

    for (int i = 0; i < actual_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

static ssize_t
hmll_mmap_fetch_range_impl(struct hmll *ctx, const int iofile, const struct hmll_iobuf *dst, const struct hmll_range range)
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
        parallel_memcpy(dst->ptr, m_buf + range.start, n_bytes);
    }
#else
    parallel_memcpy(dst->ptr, m_buf + range.start, n_bytes);
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
    backend->n = ctx->num_sources;
    if (!backend->m_content) {
        free(backend->m_content);
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
