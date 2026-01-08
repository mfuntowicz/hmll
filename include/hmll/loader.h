#ifndef HMLL_FETCHER_H
#define HMLL_FETCHER_H

#include "hmll/types.h"

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/loader.h"
#endif

struct hmll_loader
{
    enum hmll_loader_kind kind;
    enum hmll_device device;
    void *backend_impl_;
    struct hmll_range (*fetch_range_impl_)(struct hmll *, void *, struct hmll_iobuf *, struct hmll_range, unsigned short);
};
typedef struct hmll_loader hmll_loader_t;

#endif // HMLL_FETCHER_H
