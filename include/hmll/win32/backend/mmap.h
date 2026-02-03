#ifndef HMLL_WIN32_BACKEND_MMAP_H
#define HMLL_WIN32_BACKEND_MMAP_H
#include "hmll/hmll.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <windows.h>

struct hmll_mmap {
  unsigned char **m_content;
  size_t *m_sizes; // Size of each mmap'd region (needed for cleanup)
  size_t n;
  atomic_size_t refcount;
};

struct hmll_error hmll_mmap_init(struct hmll *ctx, enum hmll_device device);
ssize_t hmll_mmap_fetch_range(struct hmll *ctx, int iofile,
                              const struct hmll_iobuf *dst,
                              struct hmll_range range);

/**
 * Get a zero-copy view into the mmap'd region.
 *
 * This returns a pointer directly into the mmap'd file without any memory
 * allocation or copying. The returned iobuf holds a reference to the mmap
 * (refcount incremented). Call hmll_free_buffer on the view when done to
 * decrement the refcount.
 *
 * Note: This function is only available for the mmap backend with CPU device.
 * For GPU or other backends, this will return an error.
 *
 * @param ctx The hmll context (must be initialized with mmap backend)
 * @param iofile Index of the source file
 * @param range The byte range to get a view of
 * @param out_view Output parameter for the view (ptr, size, and mmap_ref will
 * be set)
 * @return HMLL_OK on success, error on failure
 */
struct hmll_error hmll_mmap_get_view(struct hmll *ctx, int iofile,
                                     struct hmll_range range,
                                     struct hmll_iobuf *out_view);

/**
 * Increment the reference count on an mmap region.
 * @param mmap The mmap structure to retain
 */
void hmll_mmap_retain(struct hmll_mmap *mmap);

/**
 * Decrement the reference count on an mmap region.
 * When refcount reaches 0, the mmap'd memory is unmapped and freed.
 * @param mmap The mmap structure to release
 */
void hmll_mmap_release(struct hmll_mmap *mmap);

#endif // HMLL_WIN32_BACKEND_MMAP_H
