#include "hmll/unix/iouring.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "hmll/hmll.h"

bool hmll_io_uring_is_aligned(const uintptr_t addr)
{
    return (addr & 4095) == 0;
}

int hmll_io_uring_slot_find_available(const long mask)
{
    const int pos = __builtin_ffsl(~mask);
    return pos == 0 ? -1 : pos - 1;
}

void hmll_io_uring_set_slot_busy(long *mask, const unsigned int slot)
{
    *mask |= 1 << slot;
}

void hmll_io_uring_set_slot_available(long *mask, const unsigned int slot)
{
    *mask &= ~(1 << slot);
}

void hmll_io_uring_set_payload(struct hmll_io_uring_user_payload *pyld, const unsigned int slot, const unsigned int discard, const hmll_io_uring_discard_direction_t direction)
{
    const ssize_t to_discard = direction == HMLL_DISCARD_FROM_START ? (ssize_t)discard : -discard;
    pyld->discard = to_discard;
    pyld->slot = slot;
}

void hmll_io_uring_clear_payload(struct hmll_io_uring_user_payload *pyld)
{
    pyld->discard = 0;
    pyld->slot = UINT_MAX;
}

struct hmll_fetch_range hmll_io_uring_fetch_range_to_cpu(struct hmll_context *ctx, struct hmll_fetcher_io_uring *fetcher, struct hmll_range range, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_fetch_range) {0};

    if (!hmll_io_uring_is_aligned((uintptr_t) dst.ptr)) {
        ctx->error = HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED;
        return (struct hmll_fetch_range) {0};
    }

    const size_t a_start = PAGE_ALIGNED_DOWN(range.start);
    const size_t a_end = PAGE_ALIGNED_UP(range.end);
    const size_t a_size = a_end - a_start;

    // size_t n_read = 0;
    size_t n_submitted = 0;

    // Initial submission: fill the queue
    int slot;
    while ((slot = hmll_io_uring_slot_find_available(fetcher->iobusy)) != -1 && n_submitted < a_size) {
        struct io_uring_sqe *sqe;
        if ((sqe = io_uring_get_sqe(&fetcher->ioring)) != NULL) {
            hmll_io_uring_set_slot_busy(&fetcher->iobusy, slot);

            const size_t remaining = a_size - n_submitted;
            const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t offset = a_start + n_submitted;

            struct hmll_io_uring_user_payload *payload = fetcher->iopylds + slot;
            hmll_io_uring_set_payload(payload, slot, 0, HMLL_DISCARD_FROM_START);

            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data(sqe, payload);
            io_uring_prep_read(sqe, 0, (char *)dst.ptr + n_submitted, to_read, offset);

            n_submitted += to_read;
        } else {
            break;
        }
    }

    io_uring_submit(&fetcher->ioring);

    // Process completions and resubmit as slots become available
    while (n_submitted > 0)
    {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&fetcher->ioring, &cqe) < 0) {
            ctx->error = HMLL_ERR_IO_ERROR;
            return (struct hmll_fetch_range) {0};
        }

        n_submitted -= cqe->res;
    }

    return (struct hmll_fetch_range){ range.start - a_start, a_start + (range.end - range.start) };
}

struct hmll_fetch_range hmll_io_uring_fetch_range(struct hmll_context *ctx, struct hmll_fetcher_io_uring *fetcher, const struct hmll_range range, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_fetch_range){0};

    if (dst.device != HMLL_DEVICE_CPU) {
        ctx->error = HMLL_ERR_UNSUPPORTED_DEVICE;
        return (struct hmll_fetch_range){0};
    }

    return hmll_io_uring_fetch_range_to_cpu(ctx, fetcher, range, dst);
}

struct hmll_fetch_range hmll_io_uring_fetch_range_impl_(struct hmll_context *ctx, void *fetcher, struct hmll_range range, const struct hmll_device_buffer dst)
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
    backend->iobusy = 0;

    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i)
        backend->iopylds[i].slot = UINT_MAX;

    struct io_uring_params params = {0};
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 250;

    int iofiles[1];
    iofiles[0] = ctx->source.fd;

    io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
    io_uring_register_files(&backend->ioring, iofiles, 1);

    fetcher->backend_impl_ = backend;
    fetcher->fetch_range_impl_ = hmll_io_uring_fetch_range_impl_;

    return HMLL_ERR_SUCCESS;
}
