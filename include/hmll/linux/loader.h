#ifndef HMLL_UNIX_FETCHER_H
#define HMLL_UNIX_FETCHER_H

#include "hmll/types.h"

enum hmll_loader_kind
{
    HMLL_FETCHER_AUTO,
    HMLL_FETCHER_IO_URING
};
typedef enum hmll_loader_kind hmll_fetcher_kind_t;

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, enum hmll_device device, enum hmll_loader_kind kind);

#endif // HMLL_UNIX_FETCHER_H

