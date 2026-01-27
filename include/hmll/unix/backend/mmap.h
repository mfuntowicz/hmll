#ifndef HMLL_UNIX_BACKEND_MMAP_H
#define HMLL_UNIX_BACKEND_MMAP_H
#include "hmll/hmll.h"

struct hmll_mmap {
    unsigned char **m_content;
    size_t n;
};

struct hmll_error hmll_mmap_init(struct hmll *ctx, enum hmll_device device);
ssize_t hmll_mmap_fetch_range(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, struct hmll_range range);

#endif // HMLL_UNIX_BACKEND_MMAP_H
