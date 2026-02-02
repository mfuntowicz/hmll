#ifndef HMLL_UNIX_BACKEND_MMAP_H
#define HMLL_UNIX_BACKEND_MMAP_H
#include <stdlib.h>

#include "hmll/hmll.h"

struct hmll_mmap {
    const unsigned char **m_content;
    size_t n;
};

static inline void hmll_mmap_free(struct hmll_mmap *backend)
{
    if (backend) {
        if (backend->m_content) {
            // Note: backend doesn't own the memory, sources do
            free(backend->m_content);
        }
        free(backend);
    }
}

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

#endif // HMLL_UNIX_BACKEND_MMAP_H
