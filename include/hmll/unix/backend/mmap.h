#ifndef HMLL_UNIX_BACKEND_MMAP_H
#define HMLL_UNIX_BACKEND_MMAP_H
#include "hmll/hmll.h"

struct hmll_mmap {
    unsigned char **m_content;
    size_t *m_sizes;  // Size of each mmap'd region (needed for munmap)
    size_t n;
};

struct hmll_error hmll_mmap_init(struct hmll *ctx, enum hmll_device device);
ssize_t hmll_mmap_fetch_range(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, struct hmll_range range);

/**
 * Free all mmap resources (internal implementation).
 *
 * Unmaps all memory regions and frees the mmap structure.
 * This should only be called when all views into the mmap are done.
 * The Rust wrapper manages this via Arc reference counting.
 *
 * @param mmap The mmap structure to free (can be NULL)
 */
void hmll_mmap_free_impl(struct hmll_mmap *mmap);

#endif // HMLL_UNIX_BACKEND_MMAP_H
