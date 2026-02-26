//
// Created by mfuntowicz on 1/23/26.
//

#ifndef HMLL_UNIX_LOADER_H
#define HMLL_UNIX_LOADER_H

#include "hmll/types.h"

enum hmll_loader_kind
{
    HMLL_FETCHER_AUTO,
    HMLL_FETCHER_MMAP
};
typedef enum hmll_loader_kind hmll_fetcher_kind_t;

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, struct hmll_device device, enum hmll_loader_kind kind);

#endif //HMLL_UNIX_LOADER_H
