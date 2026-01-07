#include "hmll/types.h"
#include "hmll/loader.h"
#include "hmll/hmll.h"

#if defined(__linux)
#include "hmll/unix/loader.h"
#endif


struct hmll_error hmll_loader_init(
    struct hmll *ctx,
    const struct hmll_source *srcs,
    const size_t n,
    const enum hmll_device device,
    const enum hmll_loader_kind kind)
{
    if (n > 0 && srcs) {
        ctx->num_sources = n;
        ctx->sources = srcs;
        return hmll_fetcher_init_impl(ctx, device, kind);;
    }

    return HMLL_ERR(HMLL_ERR_FILE_EMPTY);
}

struct hmll_range hmll_fetch(struct hmll *ctx, struct hmll_iobuf *dst, const struct hmll_range range, const size_t iofile)
{
    if (hmll_check(ctx->error))
        goto fail;

    if (range.start >= range.end) {
        ctx->error = HMLL_ERR(HMLL_ERR_INVALID_RANGE);
        goto fail;
    }

    if (dst->size < range.end - range.start) {
        ctx->error.code = HMLL_ERR_BUFFER_TOO_SMALL;
        goto fail;
    }

    const struct hmll_fetcher *fetcher = ctx->fetcher;
    return fetcher->fetch_range_impl_(ctx, fetcher->backend_impl_, dst, range, iofile);

fail:
    return (struct hmll_range){0};
}

// struct hmll_range hmll_fetch_tensor(struct hmll_context *ctx, struct hmll_fetcher fetcher, const char *name, const struct hmll_device_buffer dst)
// {
//     if (hmll_has_error(hmll_get_error(ctx)))
//         return (struct hmll_range){0};
//
//     const struct hmll_tensor_lookup_result lookup = hmll_get_tensor_specs(ctx, name);
//     if (lookup.found == HMLL_FALSE) {
//         ctx->error = HMLL_ERR_TENSOR_NOT_FOUND;
//         return (struct hmll_range){0};
//     }
//
//     const struct hmll_tensor_specs specs = lookup.specs;
//     const struct hmll_range range = (struct hmll_range){specs.start, specs.end};
//     return hmll_fetch(ctx, fetcher, dst, range, lookup.file);
// }
