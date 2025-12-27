#include "hmll/unix/iouring.h"

#include <stdlib.h>
#include "hmll/hmll.h"

#define HMLL_IOURING_BAIL(call, err) if ((call) < 0) { ctx->error = err; goto cleanup; }
#define HMLL_IOURING_CHECK(call) HMLL_IOURING_BAIL(call, HMLL_ERR_IO_ERROR)

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

    void *arena = hmll_get_io_buffer(ctx, device, HMLL_URING_QUEUE_DEPTH * HMLL_URING_BUFFER_SIZE);
    if (hmll_has_error(hmll_get_error(ctx))) return ctx->error;

    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
        fetcher->iovecs[i].iov_base = (char *)arena + i * HMLL_URING_BUFFER_SIZE;
        fetcher->iovecs[i].iov_len = HMLL_URING_BUFFER_SIZE;
    }

    if (io_uring_register_buffers(&fetcher->ioring, fetcher->iovecs, HMLL_URING_QUEUE_DEPTH) != 0)
        return ctx->error = HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED;

    return HMLL_ERR_SUCCESS;
}

/**
 * Checks for completed CUDA events and reclaims the associated io_uring slots.
 * If CUDA is disabled or device is CPU, this is a no-op.
 */
static inline void hmll_iouring_reclaim_slots(
    struct hmll_iouring *fetcher,
    const enum hmll_device device
) {
#if defined(__HMLL_CUDA_ENABLED__)
    if (device != HMLL_DEVICE_CUDA) return;

    struct hmll_iouring_cuda_context *dctx = fetcher->device_ctx;

    // TODO(mfuntowicz): Should we directly store `slots` which are doing memcpy currently to avoid full scan?
    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
        struct hmll_iouring_cuda_context *cd = dctx + i;
        if (hmll_iouring_slot_is_busy(fetcher->iobusy, i)) {
            if (cd->state == HMLL_CUDA_STREAM_MEMCPY && cudaEventQuery(cd->done) == cudaSuccess) {
                hmll_iouring_cuda_stream_set_idle(&cd->state);
                hmll_iouring_slot_set_available(&fetcher->iobusy, cd->slot);
            }
        }
    }
#endif
}

/**
 * Prepares a single SQE (Submission Queue Entry).
 * Handles the difference between direct CPU buffer reads and CUDA staging buffer reads.
 */
static inline void hmll_iouring_prep_sqe(
    struct hmll_iouring *fetcher,
    enum hmll_device device,
    struct io_uring_sqe *sqe,
    void *dst,
    const size_t offset,
    const size_t len,
    const int slot
) {
    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);

    if (device == HMLL_DEVICE_CPU) {
        // CPU: Read directly into user memory
        io_uring_sqe_set_data64(sqe, slot);
        io_uring_prep_read(sqe, 0, dst, len, offset);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (device == HMLL_DEVICE_CUDA) {
        // CUDA: Read into registered staging buffers
        struct hmll_iouring_cuda_context *dctx = fetcher->device_ctx;
        void *buf = fetcher->iovecs[slot].iov_base;

        dctx[slot].offset = offset;
        io_uring_sqe_set_data(sqe, dctx + slot);
        io_uring_prep_read_fixed(sqe, 0, buf, len, offset, slot);
    }
#endif
}

/**
 * Handles the completion of an IO request (CQE).
 * For CPU: just marks a slot available.
 * For CUDA: Dispatches the Async Memcpy from staging to GPU.
 */
static inline void hmll_iouring_handle_completion(
    struct hmll_iouring *fetcher,
    const struct io_uring_cqe *cqe,
    const struct hmll_device_buffer *dst,
    const size_t offset,
    const int32_t len
) {
    if (dst->device == HMLL_DEVICE_CPU) {
        const uint64_t cb_slot = cqe->user_data;
        hmll_iouring_slot_set_available(&fetcher->iobusy, cb_slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (dst->device == HMLL_DEVICE_CUDA) {
        struct hmll_iouring_cuda_context *cctx = (struct hmll_iouring_cuda_context *)cqe->user_data;

        void *to = (char *)dst->ptr + (cctx->offset - offset);
        void *from = fetcher->iovecs[cctx->slot].iov_base;

        cudaMemcpyAsync(to, from, len, cudaMemcpyHostToDevice, cctx->stream);
        cudaEventRecord(cctx->done, cctx->stream);
        hmll_iouring_cuda_stream_set_memcpy(&cctx->state);
    }
#endif
}

// --- Main Unified Logic ---------------------------------------------------

static struct hmll_range hmll_iouring_fetch_range_impl(
    struct hmll_context *ctx,
    struct hmll_iouring *fetcher,
    const struct hmll_range range,
    const struct hmll_device_buffer dst
) {
    if (hmll_has_error(hmll_get_error(ctx))) return (struct hmll_range) {0};

    const size_t a_start = ALIGN_DOWN(range.start, ALIGN_PAGE);
    const size_t a_end = ALIGN_UP(range.end, ALIGN_PAGE);
    const size_t a_size = a_end - a_start;

    if (dst.device == HMLL_DEVICE_CPU && !hmll_is_aligned((uintptr_t)dst.ptr, ALIGN_PAGE)) {
        ctx->error = HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED;
        return (struct hmll_range){0};
    }

    size_t b_read = 0;
    size_t b_submitted = 0;
    unsigned int n_dma = 0;

    while (b_read < a_size) {
        hmll_iouring_reclaim_slots(fetcher, dst.device);

        while (b_submitted < a_size) {
            const int slot = hmll_iouring_slot_find_available(fetcher->iobusy);
            if (slot == -1) break;

            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            hmll_iouring_slot_set_busy(&fetcher->iobusy, slot);

            const size_t remaining = a_size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = a_start + b_submitted;

            hmll_iouring_prep_sqe(fetcher, dst.device, sqe, (char *)dst.ptr + b_submitted, file_offset, to_read, slot);

            b_submitted += to_read;
            ++n_dma;
        }

        if (n_dma > 0) {
            const int to_wait = (dst.device == HMLL_DEVICE_CPU && n_dma >= 8) ? 8 : 1;
            if (io_uring_submit_and_wait(&fetcher->ioring, to_wait) < 0) {
                ctx->error = HMLL_ERR_IO_ERROR;
                return (struct hmll_range) {0};
            }
        }

        unsigned head, count = 0;
        struct io_uring_cqe *cqe;

        io_uring_for_each_cqe(&fetcher->ioring, head, cqe) {
            count++;
            --n_dma;

            if (cqe->res < 0) {
                ctx->error = HMLL_ERR_IO_ERROR;
                return (struct hmll_range) {0};
            }

            b_read += cqe->res;
            hmll_iouring_handle_completion(fetcher, cqe, &dst, a_start, cqe->res);
        }

        io_uring_cq_advance(&fetcher->ioring, count);
    }

    return (struct hmll_range){ range.start - a_start, a_start + (range.end - range.start) };
}


static struct hmll_range hmll_iouring_fetch_range(
    struct hmll_context *ctx,
    void *fetcher,
    const struct hmll_range range,
    const struct hmll_device_buffer dst
) {
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_range){0};

    return hmll_iouring_fetch_range_impl(ctx, fetcher, range, dst);
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


        HMLL_IOURING_BAIL(io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params), HMLL_ERR_IO_ERROR);
        HMLL_IOURING_CHECK(hmll_iouring_register_staging_buffers(ctx, backend, device));

#else
        ctx->error = HMLL_ERR_CUDA_NOT_ENABLED;
        return ctx->error;
#endif
    } else {
        io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
    }

    int iofiles[1];
    iofiles[0] = ctx->source.fd;
    HMLL_IOURING_BAIL(io_uring_register_files(&backend->ioring, iofiles, 1), HMLL_ERR_IO_ERROR);

    fetcher->device = device;
    fetcher->backend_impl_ = backend;
    fetcher->fetch_range_impl_ = hmll_iouring_fetch_range;

    return HMLL_ERR_SUCCESS;

cleanup:
    if (backend->ioring.ring_fd > 0) io_uring_queue_exit(&backend->ioring);
#if defined(__HMLL_CUDA_ENABLED__)
    if (backend->device_ctx)
        free(backend->device_ctx);

#endif
    free(backend);
    return ctx->error;
}
