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

#endif // HMLL_WIN32_BACKEND_MMAP_H
