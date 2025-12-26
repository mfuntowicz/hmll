#include "hmll/unix/iouring.h"

#include <stdlib.h>
#include "hmll/hmll.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "hmll/cuda.h"
#endif

static enum hmll_error_code hmll_iouring_register_staging_buffers(
    struct hmll_context *ctx,
    struct hmll_iouring *fetcher,
    const enum hmll_device device
) {
    fetcher->iovecs = hmll_get_io_buffer(ctx, HMLL_DEVICE_CPU, HMLL_URING_QUEUE_DEPTH * sizeof(struct iovec));
    if (hmll_has_error(hmll_get_error(ctx))) return ctx->error;

#ifdef DEBUG
    printf("Registering %u IO staging buffers for io_uring\n", HMLL_URING_QUEUE_DEPTH);
#endif

    void *arena = hmll_get_io_buffer(ctx, device, HMLL_URING_QUEUE_DEPTH * HMLL_URING_BUFFER_SIZE);
    if (hmll_has_error(hmll_get_error(ctx))) return ctx->error;

    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
        fetcher->iovecs[i].iov_base = (char *)arena + i * HMLL_URING_BUFFER_SIZE;
        fetcher->iovecs[i].iov_len = HMLL_URING_BUFFER_SIZE;
    }

    int err = 0;
    if ((err = io_uring_register_buffers(&fetcher->ioring, fetcher->iovecs, HMLL_URING_QUEUE_DEPTH)) != 0) {
#ifdef DEBUG
#include <string.h>
        printf("Failed to register IO buffer for io_uring: %s", strerror(-err));
#endif
        return ctx->error = HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED;
    }

    return HMLL_ERR_SUCCESS;
}

static struct hmll_range hmll_iouring_fetch_range_cpu(
    struct hmll_context *ctx,
    struct hmll_iouring *fetcher,
    const struct hmll_range range,
    const struct hmll_device_buffer dst
){
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_range) {0};

    const size_t a_start = ALIGN_DOWN(range.start, ALIGN_PAGE);
    const size_t a_end = ALIGN_UP(range.end, ALIGN_PAGE);
    const size_t a_size = a_end - a_start;

    if (!hmll_is_aligned((uintptr_t)dst.ptr, ALIGN_PAGE)) {
         ctx->error = HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED;
         return (struct hmll_range){0};
    }

    size_t b_read      = 0;
    size_t b_submitted = 0;
    int    inflight    = 0;

    while (b_read < a_size) {
        while (b_submitted < a_size) {
            const int slot = hmll_iouring_slot_find_available(fetcher->iobusy);
            if (slot == -1) break; // No slots left

            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            hmll_iouring_slot_set_busy(&fetcher->iobusy, slot);

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

        unsigned head, count = 0;
        struct io_uring_cqe *cqe;
        io_uring_for_each_cqe(&fetcher->ioring, head, cqe) {
            count++;
            --inflight;

            if (cqe->res < 0) {
                goto return_io_error;
            }
            b_read += cqe->res;

            const uint64_t cb_slot = cqe->user_data;
            hmll_iouring_slot_set_available(&fetcher->iobusy, cb_slot);
        }

        // Advance the CQ ring by the number of processed events
        io_uring_cq_advance(&fetcher->ioring, count);
    }

    return (struct hmll_range){ range.start - a_start, a_start + (range.end - range.start) };

return_io_error:
    ctx->error = HMLL_ERR_IO_ERROR;
    return (struct hmll_range) {0};
}

static struct hmll_range hmll_iouring_fetch_range_cuda(
    struct hmll_context *ctx,
    struct hmll_iouring *fetcher,
    const struct hmll_range range,
    const struct hmll_device_buffer dst
) {
#ifdef __HMLL_CUDA_ENABLED__
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_range) {0};

    const size_t a_start = ALIGN_DOWN(range.start, ALIGN_PAGE);
    const size_t a_end = ALIGN_UP(range.end, ALIGN_PAGE);
    const size_t a_size = a_end - a_start;

    size_t b_read      = 0;
    size_t b_submitted = 0;
    int    n_dma       = 0;

    while (b_read < a_size)
    {
        // reclaim GPU buffers if possible before sending read requests (i.e., maximize the number of submission slots)
        struct hmll_iouring_cuda_context *dctx = fetcher->device_ctx;
        for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH ; ++i) {
            struct hmll_iouring_cuda_context *cd = dctx + i;
            if (hmll_iouring_slot_is_busy(fetcher->iobusy, i) && cd->state == HMLL_CUDA_STREAM_MEMCPY) {
                if (cudaEventQuery(cd->done) == cudaSuccess) {
                    hmll_iouring_cuda_stream_set_idle(&cd->state);
                    hmll_iouring_slot_set_available(&fetcher->iobusy, cd->slot);
                }
            }
        }

        while (b_submitted < a_size) {
            const int slot = hmll_iouring_slot_find_available(fetcher->iobusy);
            if (slot == -1) break; // No slots left

            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            hmll_iouring_slot_set_busy(&fetcher->iobusy, slot);

            const size_t remaining = a_size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = a_start + b_submitted;

            void *buf = fetcher->iovecs[slot].iov_base;

            dctx[slot].offset = file_offset;
            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data(sqe, dctx + slot);
            io_uring_prep_read_fixed(sqe, 0, buf, to_read, file_offset, dctx[slot].slot);

            b_submitted += to_read;
            ++n_dma;
        }

        if (n_dma > 0 && io_uring_submit_and_wait(&fetcher->ioring, 1) < 0)
            goto return_io_error;

        unsigned head, count = 0;
        struct io_uring_cqe *cqe;
        io_uring_for_each_cqe(&fetcher->ioring, head, cqe) {
            if (cqe->res < 0)
                goto return_io_error;

            if (cqe->res > 0) {
                b_read += cqe->res;

                struct hmll_iouring_cuda_context *cctx = (struct hmll_iouring_cuda_context *) cqe->user_data;
                void *to = (char *)dst.ptr + (cctx->offset - a_start);
                void *from = fetcher->iovecs[cctx->slot].iov_base;

                cudaMemcpyAsync(to, from, cqe->res, cudaMemcpyHostToDevice, cctx->stream);
                cudaEventRecord(cctx->done, cctx->stream);
                hmll_iouring_cuda_stream_set_memcpy(&cctx->state);
            } else {
                b_read = a_size;
            }

            --n_dma;
            count++;
        }

        // Advance the CQ ring by the number of processed events
        io_uring_cq_advance(&fetcher->ioring, count);
    }

    return (struct hmll_range){ range.start - a_start, a_start + (range.end - range.start) };

return_io_error:
    ctx->error = HMLL_ERR_IO_ERROR;
    return (struct hmll_range) {0};

#else
    HMLL_UNUSED(fetcher);
    HMLL_UNUSED(range);
    HMLL_UNUSED(dst);
    ctx->error = HMLL_ERR_CUDA_NOT_ENABLED;
    return (struct hmll_range) {0};
#endif
}

struct hmll_range hmll_iouring_fetch_range(
    struct hmll_context *ctx,
    struct hmll_iouring *fetcher,
    const struct hmll_range range,
    const struct hmll_device_buffer dst
) {
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_range){0};

    switch (dst.device) {
    case HMLL_DEVICE_CPU:
        return hmll_iouring_fetch_range_cpu(ctx, fetcher, range, dst);
    case HMLL_DEVICE_CUDA:
        return hmll_iouring_fetch_range_cuda(ctx, fetcher, range, dst);
    }

    ctx->error = HMLL_ERR_UNSUPPORTED_DEVICE;
    return (struct hmll_range){0};
}

struct hmll_range hmll_iouring_fetch_range_impl_(
    struct hmll_context *ctx,
    void *fetcher,
    const struct hmll_range range,
    const struct hmll_device_buffer dst
) {
    return hmll_iouring_fetch_range(ctx, fetcher, range, dst);
}

enum hmll_error_code hmll_iouring_init(
    struct hmll_context *ctx,
    struct hmll_fetcher *fetcher,
    const enum hmll_device device
) {
    if (hmll_has_error(hmll_get_error(ctx)))
        return ctx->error;

    struct hmll_iouring *backend = calloc(1, sizeof(struct hmll_iouring));
    struct io_uring_params params = {0};
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 500;

    if (device == HMLL_DEVICE_CUDA) {
#if defined(__HMLL_CUDA_ENABLED__)
        struct hmll_iouring_cuda_context *data = calloc(HMLL_URING_QUEUE_DEPTH, sizeof(struct hmll_iouring_cuda_context));
        backend->device_ctx = (void *)data;

        for (int i = 0; i < (int)HMLL_URING_QUEUE_DEPTH; ++i) {
            data[i].slot = i;
            CHECK_CUDA(cudaStreamCreateWithFlags(&data[i].stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(&data[i].done, cudaEventDisableTiming));
        }

        io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
        hmll_iouring_register_staging_buffers(ctx, backend, device);
#else
        ctx->error = HMLL_ERR_CUDA_NOT_ENABLED;
        return ctx->error;
#endif
    } else {
        io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
    }

    // register file descriptors to avoid lookups
    int iofiles[1];
    iofiles[0] = ctx->source.fd;
    io_uring_register_files(&backend->ioring, iofiles, 1);

    fetcher->device = device;
    fetcher->backend_impl_ = backend;
    fetcher->fetch_range_impl_ = hmll_iouring_fetch_range_impl_;

    return HMLL_ERR_SUCCESS;
}
