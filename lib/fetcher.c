#include "hmll/types.h"
#include "hmll/fetcher.h"

#include "hmll/hmll.h"

#if defined(__linux)
#include "hmll/unix/fetcher.h"
#include "hmll/unix/iouring.h"
#endif

struct hmll_fetcher hmll_fetcher_init(struct hmll_context *ctx, const enum hmll_device device, const enum hmll_fetcher_kind kind)
{
    struct hmll_fetcher fetcher = {0};

    if (hmll_has_error(hmll_get_error(ctx)))
        return fetcher;

#if defined(__linux)
    if (kind == HMLL_FETCHER_AUTO || kind == HMLL_FETCHER_IO_URING)
    {
        hmll_io_uring_init(ctx, &fetcher, device);
        if (hmll_has_error(hmll_get_error(ctx)))
            return fetcher;
    }

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

    const struct hmll_tensor_specs specs = hmll_get_tensor_specs(ctx, name);
    const struct hmll_range range = (struct hmll_range){specs.start, specs.end};
    return hmll_fetch_range(ctx, fetcher, range, dst);
}