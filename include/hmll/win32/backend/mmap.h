#ifndef HMLL_WIN32_BACKEND_MMAP_H
#define HMLL_WIN32_BACKEND_MMAP_H
#include "hmll/hmll.h"
#include <windows.h>
#include <stdlib.h>

struct hmll_mmap {
    const unsigned char **m_content;
    size_t n;
};

struct hmll_error hmll_mmap_init(struct hmll *ctx, enum hmll_device device);
ssize_t hmll_mmap_fetch_range(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, struct hmll_range range);

/**
 * Get a zero-copy view into the mmap'd region.
 */
struct hmll_error hmll_mmap_get_view(struct hmll *ctx, int iofile, struct hmll_range range, struct hmll_iobuf *out_view);

#ifdef __HMLL_CUDA_ENABLED__
#include <cuda_runtime_api.h>

/**
 * Asynchronously copy data from mmap'd region to a CUDA buffer.
 */
ssize_t hmll_mmap_fetch_async(
    struct hmll *ctx,
    int iofile,
    struct hmll_iobuf *dst,
    size_t offset,
    cudaStream_t stream,
    cudaEvent_t done_event
);

/**
 * Get the raw mmap'd content pointer for a source file.
 */
const void* hmll_mmap_get_content_ptr(struct hmll *ctx, int iofile);
#endif // __HMLL_CUDA_ENABLED__

#endif // HMLL_WIN32_BACKEND_MMAP_H
