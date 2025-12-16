#include "hmll/unix/iouring.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "hmll/hmll.h"

int hmll_io_uring_is_aligned(const uintptr_t addr)
{
    return (addr & 4095) == 0;
}

int hmll_io_uring_slot_find_available(const long long mask)
{
    const int pos = __builtin_ffsll(~mask);
    return pos == 0 ? -1 : pos - 1;
}

void hmll_io_uring_slot_set_busy(long long *mask, const unsigned int slot)
{
    *mask |= 1LL << slot;
}

void hmll_io_uring_slot_set_available(long long *mask, const unsigned int slot)
{
    *mask &= ~(1LL << slot);
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

    const size_t n_bytes = range.end - range.start;
    char *ptr = dst.ptr;
    size_t b_read = 0;
    size_t b_submitted = 0;

    // Initial submission: fill the queue
    int slot;
    while ((slot = hmll_io_uring_slot_find_available(fetcher->iobusy)) != -1 && b_submitted < a_size) {
        struct io_uring_sqe *sqe;
        if ((sqe = io_uring_get_sqe(&fetcher->ioring)) != NULL) {
            hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);

            const size_t remaining = a_size - b_submitted;
            const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? PAGE_ALIGNED_UP(remaining) : HMLL_URING_BUFFER_SIZE;
            const size_t offset = a_start + b_submitted;

            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data64(sqe, slot);
            io_uring_prep_read(sqe, 0, ptr, to_read, offset);

            ptr += to_read;
            b_submitted += to_read;
        } else {
            break;
        }
    }

    io_uring_submit(&fetcher->ioring);

    // Process completions and resubmit as slots become available
    while (b_read < n_bytes)
    {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&fetcher->ioring, &cqe) < 0)
            goto return_io_error;

        io_uring_cqe_seen(&fetcher->ioring, cqe);

        if (cqe->res <= 0) {
#ifdef DEBUG
            printf("Error reading %i -> %s\n", cqe->res, strerror(-cqe->res));
#endif
            goto return_io_error;
        }

        const __u64 cb_slot = cqe->user_data;
        b_read += cqe->res;

        // Resubmit a new chunk directly
        if (b_submitted < n_bytes) {
            struct io_uring_sqe *sqe;
            if ((sqe = io_uring_get_sqe(&fetcher->ioring)) != NULL) {
                hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);

                const size_t remaining = a_size - b_submitted;
                const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? PAGE_ALIGNED_UP(remaining) : HMLL_URING_BUFFER_SIZE;
                const size_t offset = a_start + b_submitted;

                io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                io_uring_sqe_set_data64(sqe, cb_slot);
                io_uring_prep_read(sqe, 0, ptr, to_read, offset);

                ptr += to_read;
                b_submitted += to_read;

                io_uring_submit(&fetcher->ioring);
            } else {
                // Should not happen, we just released a slot?
                break;
            }
        } else {
            hmll_io_uring_slot_set_available(&fetcher->iobusy, cb_slot);
        }

#ifdef DEBUG
        printf("Read %zu / %zu (%f)\n", b_read, n_bytes, (float)b_read / n_bytes * 100.0f);
#endif
    }

    return (struct hmll_fetch_range){ range.start - a_start, a_start + (range.end - range.start) };

return_io_error:
    ctx->error = HMLL_ERR_IO_ERROR;
    return (struct hmll_fetch_range) {0};
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
    params.sq_thread_idle = 500;

    int iofiles[1];
    iofiles[0] = ctx->source.fd;

    io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
    io_uring_register_files(&backend->ioring, iofiles, 1);

    fetcher->backend_impl_ = backend;
    fetcher->fetch_range_impl_ = hmll_io_uring_fetch_range_impl_;

    return HMLL_ERR_SUCCESS;
}
