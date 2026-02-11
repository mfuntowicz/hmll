#ifndef HMLL_UNIX_BACKEND_MMAP_H
#define HMLL_UNIX_BACKEND_MMAP_H
#include <stdlib.h>

#include "hmll/hmll.h"

struct hmll_mmap {
    const unsigned char **m_content;
    size_t n;
};

struct hmll_error hmll_mmap_init(struct hmll *ctx, enum hmll_device device);
ssize_t hmll_mmap_fetch_range(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, struct hmll_range range);

/**
 * Get a zero-copy view into the mmap'd region.
 *
 * This returns a pointer directly into the mmap'd file without any memory allocation
 * or copying. The returned pointer is valid as long as the hmll context remains valid.
 *
 * Note: This function is only available for the mmap backend with CPU device.
 * For GPU or other backends, this will return an error.
 *
 * @param ctx The hmll context (must be initialized with mmap backend)
 * @param iofile Index of the source file
 * @param range The byte range to get a view of
 * @param out_view Output parameter for the view (ptr and size will be set)
 * @return HMLL_OK on success, error on failure
 */
struct hmll_error hmll_mmap_get_view(struct hmll *ctx, int iofile, struct hmll_range range, struct hmll_iobuf *out_view);

#ifdef __HMLL_CUDA_ENABLED__
#include <cuda_runtime_api.h>

/**
 * Asynchronously copy data from mmap'd region to a CUDA buffer.
 *
 * This function initiates an async memcpy from the mmap'd source to the
 * destination GPU buffer, using the provided CUDA stream. The copy completes
 * asynchronously; use the provided event to track completion.
 *
 * @param ctx The hmll context (must be initialized with mmap backend)
 * @param iofile Index of the source file
 * @param dst Destination GPU buffer (must already be allocated)
 * @param offset Byte offset into the source file
 * @param stream CUDA stream to use for the async copy
 * @param done_event Optional: event to record after copy is queued (can be NULL)
 * @return Number of bytes queued for copy, or -1 on error
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
 * Useful for direct access when implementing custom async copy patterns.
 *
 * @param ctx The hmll context (must be initialized with mmap backend)
 * @param iofile Index of the source file
 * @return Pointer to the mmap'd content, or NULL on error
 */
const void* hmll_mmap_get_content_ptr(struct hmll *ctx, int iofile);
#endif // __HMLL_CUDA_ENABLED__

#endif // HMLL_UNIX_BACKEND_MMAP_H
