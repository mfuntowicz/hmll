#include "hmll/types.h"
#include "hmll/fetcher.h"

#include "hmll/hmll.h"

#if defined(__linux)
#include "hmll/unix/fetcher.h"
#include "hmll/unix/iouring.h"
#endif

#if defined(__HMLL_CUDA_ENABLED__)
#include "hmll/cuda.h"
#endif

struct hmll_fetcher hmll_fetcher_init(struct hmll_context *ctx, const enum hmll_device device, const enum hmll_fetcher_kind kind)
{
    struct hmll_fetcher fetcher = {0};

    if (hmll_has_error(hmll_get_error(ctx)))
        return fetcher;

#if defined(__HMLL_CUDA_ENABLED__)
    int device_count = 0;
    if (device == HMLL_DEVICE_CUDA && (device_count = hmll_cuda_device_count()) == 0) {
        ctx->error = HMLL_ERR_CUDA_NO_DEVICE;
        return fetcher;
    }

    const int cudaAllocFlags = hmll_cuda_allocation_flags();
    if (hmll_has_error(hmll_get_error(ctx)))
        return fetcher;

    const struct hmll_fetcher_cuda_meta meta = {device_count, cudaAllocFlags};
    fetcher.meta.cuda = meta;
#endif

#if defined(__linux)
    if (kind == HMLL_FETCHER_AUTO || kind == HMLL_FETCHER_IO_URING)
        hmll_io_uring_init(ctx, &fetcher, device);
#endif
    return fetcher;
}

struct hmll_fetch_range hmll_fetch_range(struct hmll_context *ctx, struct hmll_fetcher fetcher, struct hmll_range range, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_fetch_range){0};

    if (range.start >= range.end) {
        ctx->error = HMLL_ERR_INVALID_RANGE;
        return (struct hmll_fetch_range){0};
    }

    if (dst.size < range.end - range.start) {
        ctx->error = HMLL_ERR_BUFFER_TOO_SMALL;
        return (struct hmll_fetch_range){0};
    }

    return fetcher.fetch_range_impl_(ctx, fetcher.backend_impl_, range, dst);
}

struct hmll_fetch_range hmll_fetch_tensor(struct hmll_context *ctx, struct hmll_fetcher fetcher, const char *name, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_fetch_range){0};

    const struct hmll_tensor_lookup_result lookup = hmll_get_tensor_specs(ctx, name);
    if (lookup.found == HMLL_FALSE) {
        ctx->error = HMLL_ERR_TENSOR_NOT_FOUND;
        return (struct hmll_fetch_range){0};
    }

    const struct hmll_tensor_specs specs = lookup.specs;
    const struct hmll_range range = (struct hmll_range){specs.start, specs.end};
    return hmll_fetch_range(ctx, fetcher, range, dst);
}
