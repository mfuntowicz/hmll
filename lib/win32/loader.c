#include "hmll/loader.h"
#include "hmll/hmll.h"
#include "hmll/win32/backend/mmap.h"

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, const enum hmll_device device, const enum hmll_loader_kind kind)
{
    HMLL_UNUSED(kind);
    return hmll_mmap_init(ctx, device);
}
