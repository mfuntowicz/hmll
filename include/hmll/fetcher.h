#ifndef HMLL_FETCHER_H
#define HMLL_FETCHER_H

#include "hmll/types.h"


#define PAGE_ALIGNED_UP(x, align) (((x) + align - 1) & ~(align - 1))
#define PAGE_ALIGNED_DOWN(x, align) ((x) & ~(align - 1))


struct hmll_fetch_range {
    size_t start;
    size_t end;
};
typedef struct hmll_fetch_range hmll_fetch_range_t;

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/fetcher.h"
#endif


struct hmll_range {
    size_t start;
    size_t end;
};
typedef struct hmll_range hmll_range_t;


#if defined(__HMLL_CUDA_ENABLED__)
struct hmll_fetcher_cuda_meta
{
    int device_count;
    int alloc_flags;
};
typedef struct hmll_fetcher_cuda_meta hmll_fetcher_cuda_meta_t;
#endif

struct hmll_fetcher
{
    enum hmll_fetcher_kind kind;
    enum hmll_device device;
    void *backend_impl_;
    struct hmll_fetch_range (*fetch_range_impl_)(struct hmll_context *, void *, struct hmll_range, struct hmll_device_buffer);

#if defined(__HMLL_CUDA_ENABLED__)
    union
    {
        hmll_fetcher_cuda_meta_t cuda;
    } meta;
#endif
};
typedef struct hmll_fetcher hmll_fetcher_t;

#endif // HMLL_FETCHER_H
