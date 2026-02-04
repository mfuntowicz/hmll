#include "hmll/types.h"
#include "hmll/loader.h"
#include "hmll/hmll.h"

// Platform-specific loader headers (loader.h handles the include logic)


struct hmll_error hmll_loader_init(
    struct hmll *ctx,
    struct hmll_source *srcs,
    const size_t n,
    const enum hmll_device device,
    const enum hmll_loader_kind kind)
{
    if (n > 0 && srcs) {
        ctx->num_sources = n;
        ctx->sources = srcs;
        return hmll_fetcher_init_impl(ctx, device, kind);
    }

    return HMLL_ERR(HMLL_ERR_FILE_EMPTY);
}

ssize_t hmll_fetch(struct hmll *ctx, const int iofile, const struct hmll_iobuf *dst, const size_t offset)
{
    if (hmll_check(ctx->error))
        goto fail;

    if (dst->size == 0) return 0;

    const struct hmll_loader *fetcher = ctx->fetcher;
    return fetcher->fetch_range_impl_(ctx, iofile, dst, offset);

fail:
    return -1;
}

ssize_t hmll_fetchv(struct hmll *ctx, const int iofile, const struct hmll_iobuf *dsts, const size_t *offsets, const size_t n)
{
    if (hmll_check(ctx->error))
        goto fail;

    const struct hmll_loader *fetcher = ctx->fetcher;
    return fetcher->fetchv_range_impl_(ctx, iofile, dsts, offsets, n);

fail:
    return -1;
}

#ifdef __HMLL_TENSORS_ENABLED__
ssize_t hmll_fetch_tensor(struct hmll *ctx, const struct hmll_registry *registry, struct hmll_iobuf *dst, const char *name)
{
    if (hmll_check(ctx->error))
        return -1;

    const struct hmll_lookup_result lookup = hmll_lookup_tensor(ctx, registry, name);
    if (lookup.specs == NULL) {
        ctx->error = HMLL_ERR(HMLL_ERR_TENSOR_NOT_FOUND);
        return -1;
    }

    const struct hmll_tensor_specs *specs = lookup.specs;
    const struct hmll_range range = (struct hmll_range){specs->start, specs->end};
    return hmll_fetch(ctx, lookup.file, dst, range.start);
}
#endif
