//
// Created by mfuntowicz on 1/23/26.
//
#include "hmll/unix/loader.h"
#include "hmll/unix/backend/mmap.h"

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, const struct hmll_device device, const enum hmll_loader_kind kind)
{
    // On non-Linux Unix systems (macOS, BSD), only mmap backend is available
    if (kind == HMLL_FETCHER_AUTO || kind == HMLL_FETCHER_MMAP)
        return hmll_mmap_init(ctx, device);

    // Unsupported backend requested
    ctx->error = HMLL_ERR(HMLL_ERR_UNSUPPORTED_PLATFORM);
    return ctx->error;
}
