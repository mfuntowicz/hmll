#include "hmll/fetcher.h"
#include "hmll/tracing.h"
#include "hmll/types.h"

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

    HMLL_ZONE_START(hmll_fetcher_init);
#if defined(__HMLL_CUDA_ENABLED__)
    int device_count = 0;
    if (device == HMLL_DEVICE_CUDA && (device_count = hmll_cuda_device_count()) == 0) {
        HMLL_ZONE_END(hmll_fetcher_init);
        ctx->error = HMLL_ERR_CUDA_NO_DEVICE;
        return fetcher;
    }

    const int cudaAllocFlags = hmll_cuda_allocation_flags();
    if (hmll_has_error(hmll_get_error(ctx))) {
        HMLL_ZONE_END(hmll_fetcher_init);
        return fetcher;
    }

    const struct hmll_fetcher_cuda_meta meta = {device_count, cudaAllocFlags};
    fetcher.meta.cuda = meta;
#endif

#if defined(__linux)
    if (kind == HMLL_FETCHER_AUTO || kind == HMLL_FETCHER_IO_URING)
        hmll_iouring_init(ctx, &fetcher, device);
#endif
    HMLL_ZONE_END(hmll_fetcher_init);
    return fetcher;
}

struct hmll_range hmll_fetch_range(struct hmll_context *ctx, struct hmll_fetcher fetcher, struct hmll_range range, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_range){0};

    HMLL_ZONE_START(hmll_fetch_range)
    if (range.start >= range.end) {
        HMLL_ZONE_END(hmll_fetch_range)
        ctx->error = HMLL_ERR_INVALID_RANGE;
        return (struct hmll_range){0};
    }

    if (dst.size < range.end - range.start) {
        HMLL_ZONE_END(hmll_fetch_range)
        ctx->error = HMLL_ERR_BUFFER_TOO_SMALL;
        return (struct hmll_range){0};
    }

    struct hmll_range offsets = fetcher.fetch_range_impl_(ctx, fetcher.backend_impl_, range, dst);
    HMLL_ZONE_END(hmll_fetch_range)
    return offsets;
}

struct hmll_range hmll_fetch_tensor(struct hmll_context *ctx, struct hmll_fetcher fetcher, const char *name, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_range){0};

    HMLL_ZONE_START(hmll_fetch_tensor)
    const struct hmll_tensor_lookup_result lookup = hmll_get_tensor_specs(ctx, name);
    if (lookup.found == HMLL_FALSE) {
        HMLL_ZONE_END(hmll_fetch_tensor)
        ctx->error = HMLL_ERR_TENSOR_NOT_FOUND;
        return (struct hmll_range){0};
    }

    const struct hmll_tensor_specs specs = lookup.specs;
    const struct hmll_range range = (struct hmll_range){specs.start, specs.end};
    struct hmll_range offsets = hmll_fetch_range(ctx, fetcher, range, dst);
    HMLL_ZONE_END(hmll_fetch_tensor)
    return offsets;
}
