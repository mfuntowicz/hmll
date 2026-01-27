#include "hmll/loader.h"
#include "hmll/hmll.h"
#include "hmll/linux/backend/iouring.h"
#include "hmll/unix/backend/mmap.h"
#ifdef __HMLL_SPDK_ENABLED__
#include "hmll/linux/backend/spdk.h"
#endif

struct hmll_error hmll_fetcher_init_impl(struct hmll *ctx, const enum hmll_device device, const enum hmll_loader_kind kind)
{
    // Explicit backend selection
    if (kind == HMLL_FETCHER_IO_URING)
        return hmll_io_uring_init(ctx, device);

#ifdef __HMLL_SPDK_ENABLED__
    if (kind == HMLL_FETCHER_SPDK)
        return hmll_spdk_init(ctx, device);
#endif

    if (kind == HMLL_FETCHER_MMAP)
        return hmll_mmap_init(ctx, device);

    // Auto-selection: try backends in order of performance
    if (kind == HMLL_FETCHER_AUTO) {
#ifdef __HMLL_SPDK_ENABLED__
        // Try SPDK first (fastest if NVMe available)
        struct hmll_error spdk_err = hmll_spdk_init(ctx, device);
        if (hmll_success(spdk_err))
            return spdk_err;
        // If SPDK failed due to no hardware, fall through
        // Reset error state for next attempt
        ctx->error = HMLL_OK;
#endif
        // Try io_uring next (async I/O)
        struct hmll_error uring_err = hmll_io_uring_init(ctx, device);
        if (hmll_success(uring_err))
            return uring_err;
        ctx->error = HMLL_OK;

        // Fall back to mmap (always works)
        return hmll_mmap_init(ctx, device);
    }

    return HMLL_OK;
}
