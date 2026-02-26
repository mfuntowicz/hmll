#include "hmll/loader.h"
#include "hmll/hmll.h"
#include "hmll/linux/backend/iouring.h"
#include "hmll/unix/backend/mmap.h"

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, const struct hmll_device device, const enum hmll_loader_kind kind)
{
    if (kind == HMLL_FETCHER_IO_URING)
        return hmll_io_uring_init(ctx, device);

    if (kind == HMLL_FETCHER_MMAP || kind == HMLL_FETCHER_AUTO)
        return hmll_mmap_init(ctx, device);

    // Unsupported backend requested
    ctx->error = HMLL_ERR(HMLL_ERR_UNSUPPORTED_PLATFORM);
    return ctx->error;
}
