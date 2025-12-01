#ifndef HMLL_UNIX_MMAP_H
#define HMLL_UNIX_MMAP_H

#include "hmll/status.h"
#include "hmll/types.h"

hmll_status_t hmll_open_mmap(const char *path, hmll_context_t *ctx);
hmll_status_t hmll_close_mmap(hmll_context_t *ctx);

#endif // HMLL_UNIX_MMAP_H
