#ifndef HMLL_UNIX_MMAP_H
#define HMLL_UNIX_MMAP_H

#include "hmll/types.h"

enum hmll_error_code hmll_open_mmap(const char *path, hmll_context_t *ctx);
void hmll_close_mmap(hmll_context_t *ctx);

#endif // HMLL_UNIX_MMAP_H
