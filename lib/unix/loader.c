#include "hmll/loader.h"
#include "hmll/types.h"
#include "hmll/unix/backend/iouring.h"
#include "hmll/hmll.h"

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, const enum hmll_device device, const enum hmll_loader_kind kind)
{
    if (kind == HMLL_FETCHER_AUTO || kind == HMLL_FETCHER_IO_URING)
        return hmll_io_uring_init(ctx, device);

    return HMLL_OK;
}
