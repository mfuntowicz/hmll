#ifndef HMLL_FETCHER_H
#define HMLL_FETCHER_H

#include "hmll/types.h"

#if defined(__linux__)
#include "hmll/linux/loader.h"
#elif defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/loader.h"
#elif defined(_WIN32)
#include "hmll/win32/loader.h"
#endif

struct hmll_loader
{
    enum hmll_loader_kind kind;
    enum hmll_device device;
    void *backend_impl_;
    ssize_t(*fetch_range_impl_)(struct hmll *, int, const struct hmll_iobuf *, size_t);
    ssize_t(*fetchv_range_impl_)(struct hmll *, int, const struct hmll_iobuf *, const size_t *, size_t);
};
typedef struct hmll_loader hmll_loader_t;

#endif // HMLL_FETCHER_H
