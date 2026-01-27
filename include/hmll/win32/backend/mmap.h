#ifndef HMLL_WIN32_BACKEND_MMAP_H
#define HMLL_WIN32_BACKEND_MMAP_H
#include "hmll/hmll.h"
#include <windows.h>
#include <stdlib.h>

struct hmll_mmap {
    unsigned char **m_content;
    size_t n;
};

static inline void hmll_mmap_free(struct hmll_mmap *backend)
{
    if (backend) {
        if (backend->m_content) {
            for (size_t i = 0; i < backend->n; i++) {
                if (backend->m_content[i]) {
                    UnmapViewOfFile(backend->m_content[i]);
                }
            }
            free(backend->m_content);
        }
        free(backend);
    }
}

struct hmll_error hmll_mmap_init(struct hmll *ctx, enum hmll_device device);
ssize_t hmll_mmap_fetch_range(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, struct hmll_range range);

#endif // HMLL_WIN32_BACKEND_MMAP_H
