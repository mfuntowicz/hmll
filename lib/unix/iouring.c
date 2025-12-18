#include "hmll/unix/iouring.h"

#include <stdlib.h>
#include "hmll/hmll.h"

static struct hmll_fetch_range hmll_io_uring_fetch_range_to_cpu(struct hmll_context *ctx, struct hmll_fetcher_io_uring *fetcher, struct hmll_range range, const struct hmll_device_buffer dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_fetch_range) {0};

    const size_t a_start = PAGE_ALIGNED_DOWN(range.start, ALIGNMENT);
    const size_t a_end = PAGE_ALIGNED_UP(range.end, ALIGNMENT);
    const size_t a_size = a_end - a_start;

    if (!hmll_io_uring_is_aligned((uintptr_t)dst.ptr)) {
         ctx->error = HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED;
         return (struct hmll_fetch_range){0};
    }

    size_t b_read      = 0;
    size_t b_submitted = 0;
    int    inflight    = 0;

    while (b_read < a_size) {
        while (b_submitted < a_size) {
            const int slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
            if (slot == -1) break; // No slots left

            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);

            const size_t remaining = a_size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = a_start + b_submitted;

            char *req_addr = (char *)dst.ptr + b_submitted;

            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data64(sqe, slot);
            io_uring_prep_read(sqe, 0, req_addr, to_read, file_offset);

            b_submitted += to_read;
            ++inflight;
        }

        if (inflight > 0) {
            const int to_wait = (inflight < 8) ? inflight : 8;

            // Use submit_and_wait to flush the SQEs we just added AND wait in one syscall.
            const int ret = io_uring_submit_and_wait(&fetcher->ioring, to_wait);
            if (ret < 0)
                goto return_io_error;
        }

        struct io_uring_cqe *cqe;
        const int ret = io_uring_submit_and_wait(&fetcher->ioring, 1);
        if (ret < 0)
            goto return_io_error;

        unsigned head, count = 0;
        io_uring_for_each_cqe(&fetcher->ioring, head, cqe) {
            count++;
            --inflight;

            if (cqe->res < 0)
            {
#include <string.h>
                const char *err = strerror(-cqe->res);
                printf("Error: %s", err);
                goto return_io_error;
            }
            b_read += cqe->res;

            const uint64_t cb_slot = cqe->user_data;
            hmll_io_uring_slot_set_available(&fetcher->iobusy, cb_slot);
        }

        // Advance the CQ ring by the number of processed events
        io_uring_cq_advance(&fetcher->ioring, count);
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
    struct io_uring_params params = {0};
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 500;

    int iofiles[1];
    iofiles[0] = ctx->source.fd;

    io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
    io_uring_register_files(&backend->ioring, iofiles, 1);

    fetcher->device = device;
    fetcher->backend_impl_ = backend;
    fetcher->fetch_range_impl_ = hmll_io_uring_fetch_range_impl_;

    return HMLL_ERR_SUCCESS;
}
