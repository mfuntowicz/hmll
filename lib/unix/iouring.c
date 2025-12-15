#include "hmll/unix/iouring.h"

#include <stdlib.h>

#include "hmll/hmll.h"

int hmll_io_uring_slot_available(const long mask)
{
    const int pos = __builtin_ffsl(mask);
    return pos == 0 ? 0 : pos - 1;
}

size_t hmll_io_uring_fetch_range(struct hmll_context *ctx, struct hmll_fetcher_io_uring *fetcher, struct hmll_range range, const struct hmll_device_buffer *dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return 0;

    if (!fetcher) return 0;
    if (!dst) return 0;
    if (range.start >= range.end) return 0;

    return 0;
}

size_t hmll_io_uring_fetch_range_impl_(struct hmll_context *ctx, void *fetcher, struct hmll_range range, const struct hmll_device_buffer *dst)
{
    return hmll_io_uring_fetch_range(ctx, fetcher, range, dst);
}

enum hmll_error_code hmll_io_uring_init(struct hmll_context *ctx, struct hmll_fetcher *fetcher, const enum hmll_device device)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return ctx->error;

    if (device != HMLL_DEVICE_CPU) {
        ctx->error = HMLL_ERR_UNSUPPORTED_DEVICE;
        return ctx->error;
    }

    struct hmll_fetcher_io_uring *backend = calloc(1, sizeof(struct hmll_fetcher_io_uring));
    backend->iobusy = 0xFFFFFF;

    struct io_uring_params params = {0};
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 250;

    int iofiles[1];
    iofiles[0] = ctx->source.fd;

    io_uring_queue_init_params(HMLL_URING_NUM_IOVECS, &backend->ioring, &params);
    io_uring_register_files(&backend->ioring, iofiles, 1);

    fetcher->backend_impl_ = backend;
    fetcher->fetch_range_impl_ = hmll_io_uring_fetch_range_impl_;

    return HMLL_ERR_SUCCESS;
}
