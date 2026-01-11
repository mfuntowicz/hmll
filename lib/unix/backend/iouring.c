#include <stdlib.h>
#include <string.h>
#include "hmll/hmll.h"
#include "hmll/cuda.h"
#include "hmll/memory.h"
#include "hmll/unix/backend/iouring.h"

#define HMLL_IO_URING_ADVISORY_FLAG UINT64_MAX

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif

static struct hmll_error hmll_io_uring_register_staging_buffers(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const enum hmll_device device
) {
    fetcher->iovecs = hmll_get_io_buffer(ctx, device, HMLL_URING_QUEUE_DEPTH * sizeof(struct iovec));
    if (hmll_check(ctx->error)) return ctx->error;

    void *arena = hmll_get_io_buffer(ctx, device, HMLL_URING_QUEUE_DEPTH * HMLL_URING_BUFFER_SIZE);
    if (hmll_check(ctx->error)) return ctx->error;

    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
        fetcher->iovecs[i].iov_base = (char *)arena + i * HMLL_URING_BUFFER_SIZE;
        fetcher->iovecs[i].iov_len = HMLL_URING_BUFFER_SIZE;
    }

    int res;
    if ((res = io_uring_register_buffers(&fetcher->ioring, fetcher->iovecs, HMLL_URING_QUEUE_DEPTH)) < 0)
        return ctx->error = HMLL_SYS_ERR(res);

    return HMLL_OK;
}

/**
 * Checks for completed CUDA events and reclaims the associated io_uring slots.
 * If CUDA is disabled or the device is CPU, this is a no-op.
 */
static inline void hmll_io_uring_reclaim_slots(
    struct hmll_io_uring *fetcher,
    const enum hmll_device device
) {
#ifdef __HMLL_CUDA_ENABLED__
    if (device != HMLL_DEVICE_CUDA) return;

    struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;

    // TODO(mfuntowicz): Should we directly store `slots` which are doing memcpy currently to avoid full scan?
    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
        struct hmll_io_uring_cuda_context *cd = dctx + i;
        if (hmll_io_uring_slot_is_busy(fetcher->iobusy, i)) {
            if (cd->state == HMLL_CUDA_STREAM_MEMCPY && cudaEventQuery(cd->done) == cudaSuccess) {
                hmll_io_uring_cuda_stream_set_idle(&cd->state);
                hmll_io_uring_slot_set_available(&fetcher->iobusy, cd->slot);
            }
        }
    }
#else
    HMLL_UNUSED(fetcher);
    HMLL_UNUSED(device);
#endif
}

/**
 * Prepares a single SQE (Submission Queue Entry).
 * Handles the difference between direct CPU buffer reads and CUDA staging buffer reads.
 */
static inline void hmll_io_uring_prep_sqe(
    const struct hmll_io_uring *fetcher,
    const enum hmll_device device,
    struct io_uring_sqe *sqe,
    void *dst,
    const size_t offset,
    const size_t len,
    const unsigned short iofile,
    const int slot
) {
    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);

    if (device == HMLL_DEVICE_CPU) {
        // CPU: Read directly into user memory
        io_uring_sqe_set_data64(sqe, slot);
        io_uring_prep_read(sqe, iofile, dst, len, offset);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (device == HMLL_DEVICE_CUDA) {
        // CUDA: Read into registered staging buffers
        struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
        void *buf = fetcher->iovecs[slot].iov_base;

        dctx[slot].offset = offset;
        io_uring_sqe_set_data(sqe, dctx + slot);
        io_uring_prep_read_fixed(sqe, iofile, buf, len, offset, slot);
    }
#else
    HMLL_UNUSED(fetcher);
#endif
}

static inline int hmll_io_uring_get_sqe(struct hmll_io_uring *fetcher, struct io_uring_sqe **sqe)
{
    const int slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
    if (slot == -1) return -1;

    *sqe = io_uring_get_sqe(&fetcher->ioring);
    if (*sqe == NULL) return -1;

    hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);
    return slot;
}

/**
 * Handles the completion of an IO request (CQE).
 * For CPU: just marks a slot available.
 * For CUDA: Dispatches the Async Memcpy from staging to GPU.
 */
static inline void hmll_io_uring_handle_completion(
    struct hmll_io_uring *fetcher,
    const struct io_uring_cqe *cqe,
    const struct hmll_iobuf *dst,
    const size_t offset,
    const int32_t len
) {
    if (dst->device == HMLL_DEVICE_CPU) {
        const uint64_t cb_slot = cqe->user_data;
        hmll_io_uring_slot_set_available(&fetcher->iobusy, cb_slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (dst->device == HMLL_DEVICE_CUDA) {
        struct hmll_io_uring_cuda_context *cctx = (struct hmll_io_uring_cuda_context *)cqe->user_data;

        void *to = (char *)dst->ptr + (cctx->offset - offset);
        void *from = fetcher->iovecs[cctx->slot].iov_base;

        cudaMemcpyAsync(to, from, len, cudaMemcpyHostToDevice, cctx->stream);
        cudaEventRecord(cctx->done, cctx->stream);
        hmll_io_uring_cuda_stream_set_memcpy(&cctx->state);
    }
#else
    HMLL_UNUSED(offset);
    HMLL_UNUSED(len);
#endif
}

static struct hmll_range hmll_io_uring_fetch_range_impl(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct hmll_iobuf *dst,
    const struct hmll_range range,
    const int iofile
) {
    if (hmll_check(ctx->error)) return (struct hmll_range) {0};

    size_t n_dma = 0;
    size_t b_read = 0;
    size_t b_submitted = 0;
    struct io_uring_cqe *cqes[HMLL_URING_CQE_BATCH_SIZE];

    const size_t size = hmll_range_size(range);
    struct io_uring_sqe *sqe = NULL;
    int slot;
    if ((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) >= 0) {
        io_uring_prep_fadvise(sqe, iofile, range.start, size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
        io_uring_sqe_set_data64(sqe, HMLL_IO_URING_ADVISORY_FLAG);
    }

    while (b_read < size) {
        hmll_io_uring_reclaim_slots(fetcher, dst->device);

        while (b_submitted < size) {
            if ((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) < 0)
                break;

            const size_t remaining = size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = range.start + b_submitted;

            hmll_io_uring_prep_sqe(fetcher, dst->device, sqe, (char *)dst->ptr + b_submitted, file_offset, to_read, iofile, slot);

            b_submitted += to_read;
            ++n_dma;
        }

        // update congestion control algorithm
        if (n_dma > 0) {
             const size_t nwait = MIN(n_dma, fetcher->iocca.window);

            struct timespec ts_start, ts_end;
            clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_start);

            if (io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0) {
                // todo: do we need to reset the cca? hmll_io_uring_cca_init(&fetcher->iocca)
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                return (struct hmll_range) {0};
            }
            clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_end);

            // todo: approximated version of the number of bytes actually reads because it assumes full reads
            hmll_io_uring_cca_update(&fetcher->iocca, HMLL_URING_BUFFER_SIZE * nwait, ts_start, ts_end);
        }

        unsigned count = 0;
        while ((count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, HMLL_URING_CQE_BATCH_SIZE)) > 0) {
            for (unsigned i = 0; i < count; i++) {
                --n_dma;

                const struct io_uring_cqe *cqe = cqes[i];
                if (cqe->user_data == HMLL_IO_URING_ADVISORY_FLAG) continue;
                if (cqe->res < 0) {
                    ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                    io_uring_cq_advance(&fetcher->ioring, i + 1);
                    return (struct hmll_range) {0};
                }

                b_read += cqe->res;
                hmll_io_uring_handle_completion(fetcher, cqe, dst, range.start, cqe->res);
            }

            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    return (struct hmll_range){0};
}

static struct hmll_range *hmll_io_uring_fetchv_range_impl(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct hmll_iobuf *dsts,
    const struct hmll_range *ranges,
    const int iofile,
    const size_t n
) {
    if (hmll_check(ctx->error)) return NULL;

    // Allocate result array
    struct hmll_range *results = calloc(n, sizeof(struct hmll_range));
    if (!results) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        return NULL;
    }

    // Use stack for small batches (common in tensor chunking)
    struct fetch_state {
        size_t submitted;
        size_t read;
        size_t size;
        unsigned char fadvise_sent;
    };

    struct fetch_state *states;
    struct fetch_state stack_states[64];
    if (n <= 64) {
        states = stack_states;
        memset(states, 0, sizeof(struct fetch_state) * n);
    } else {
        states = calloc(n, sizeof(struct fetch_state));
        if (!states) {
            free(results);
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            return NULL;
        }
    }

    // Pre-calculate sizes
    for (size_t i = 0; i < n; ++i) {
        states[i].size = ranges[i].end - ranges[i].start;
    }

    // Bitmask Constants for packed user_data
    // [1 bit: Fadvise] [31 bits: Range Index] [32 bits: Slot Index]
    const uint64_t MASK_FADVISE   = 1ULL << 63;
    const uint64_t MASK_RANGE     = 0x7FFFFFFF00000000ULL;
    const uint64_t MASK_SLOT      = 0x00000000FFFFFFFFULL;
    const uint64_t SHIFT_RANGE    = 32;

    size_t n_in_flight_data = 0;
    size_t n_completed = 0;
    size_t submit_cursor = 0;

    struct io_uring_cqe *cqes[HMLL_URING_CQE_BATCH_SIZE];

    while (n_completed < n) {

        while (1) {
            // 1. Find work (Round-Robin)
            size_t current_idx = -1;
            size_t checked = 0;
            size_t idx = submit_cursor;

            while (checked < n) {
                if ((states[idx].read < states[idx].size && states[idx].submitted < states[idx].size) ||
                    (!states[idx].fadvise_sent && states[idx].size > 0)) {
                    current_idx = idx;
                    break;
                }
                idx++;
                if (idx == n) idx = 0;
                checked++;
            }

            if (current_idx == (size_t)-1) break;

            submit_cursor = current_idx + 1;
            if (submit_cursor == n) submit_cursor = 0;

            // 2. Get SQE
            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            // 3. Prepare SQE
            if (!states[current_idx].fadvise_sent) {
                io_uring_prep_fadvise(
                    sqe, iofile, ranges[current_idx].start, states[current_idx].size,
                    POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
                io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);

                const uint64_t data = MASK_FADVISE | current_idx << SHIFT_RANGE;
                io_uring_sqe_set_data64(sqe, data);

                states[current_idx].fadvise_sent = true;
                continue;
            }

            // Data Read
            int slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
            if (slot == -1) {
                hmll_io_uring_reclaim_slots(fetcher, dsts[0].device);
                slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
            }

            if (slot == -1) break; // No slots, must wait

            hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);

            const size_t remaining = states[current_idx].size - states[current_idx].submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = ranges[current_idx].start + states[current_idx].submitted;

            hmll_io_uring_prep_sqe(
                fetcher,
                dsts[current_idx].device,
                sqe,
                (char *)dsts[current_idx].ptr + states[current_idx].submitted,
                file_offset,
                to_read,
                iofile,
                slot
            );

            const uint64_t data = current_idx << SHIFT_RANGE | slot;
            io_uring_sqe_set_data64(sqe, data);

            states[current_idx].submitted += to_read;
            n_in_flight_data++;
        }

        // --- WAITING PHASE ---
        size_t nwait = 0;
        if (n_in_flight_data > 0) {
            nwait = MIN(n_in_flight_data, fetcher->iocca.window);
        }

        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_start);

        if (io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0) {
            ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_end);

        if (nwait > 0) {
            hmll_io_uring_cca_update(&fetcher->iocca, HMLL_URING_BUFFER_SIZE * nwait, ts_start, ts_end);
        }

        // --- COMPLETION PHASE ---
        unsigned count = 0;
        while ((count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, HMLL_URING_CQE_BATCH_SIZE)) > 0) {
            for (unsigned i = 0; i < count; i++) {
                const struct io_uring_cqe *cqe = cqes[i];
                const uint64_t data = cqe->user_data;

                if (data & MASK_FADVISE) continue; // Ignore fadvise completion

                --n_in_flight_data;

                if (cqe->res < 0) {
                    ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                    io_uring_cq_advance(&fetcher->ioring, count);
                    goto cleanup;
                }

                const uint32_t r_idx = (data & MASK_RANGE) >> SHIFT_RANGE;
                const uint32_t s_idx = (data & MASK_SLOT);

                if (dsts[0].device == HMLL_DEVICE_CPU) {
                    hmll_io_uring_slot_set_available(&fetcher->iobusy, s_idx);
                }
#if defined(__HMLL_CUDA_ENABLED__)
                else if (dsts[0].device == HMLL_DEVICE_CUDA) {
                    struct hmll_io_uring_cuda_context *cctx = &((struct hmll_io_uring_cuda_context *)fetcher->device_ctx)[s_idx];

                    // Calc offset relative to destination buffer start
                    const size_t rel_offset = cctx->offset - ranges[r_idx].start;

                    void *to = (char *)dsts[r_idx].ptr + rel_offset;
                    void *from = fetcher->iovecs[s_idx].iov_base;

                    cudaMemcpyAsync(to, from, cqe->res, cudaMemcpyHostToDevice, cctx->stream);
                    cudaEventRecord(cctx->done, cctx->stream);
                    hmll_io_uring_cuda_stream_set_memcpy(&cctx->state);
                }
#endif

                states[r_idx].read += cqe->res;
                if (states[r_idx].read >= states[r_idx].size) {
                    results[r_idx] = (struct hmll_range){ 0, states[r_idx].size };
                    n_completed++;
                }
            }
            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    if (n > 64) free(states);
    return results;

cleanup:
    if (n > 64) free(states);
    free(results);
    return NULL;
}

static struct hmll_range hmll_io_uring_fetch_range(
    struct hmll *ctx,
    void *fetcher,
    const struct hmll_iobuf *dst,
    const struct hmll_range range,
    const int iofile
) {
    if (hmll_check(ctx->error))
        return (struct hmll_range){0};

    return hmll_io_uring_fetch_range_impl(ctx, fetcher, dst, range, iofile);
}

static struct hmll_range *hmll_io_uring_fetchv_range(
    struct hmll *ctx,
    void *fetcher,
    const struct hmll_iobuf *dsts,
    const struct hmll_range *ranges,
    const int iofile,
    const size_t n
) {
    if (hmll_check(ctx->error))
        return NULL;

    return hmll_io_uring_fetchv_range_impl(ctx, fetcher, dsts, ranges, iofile, n);
}

struct hmll_error hmll_io_uring_init(struct hmll *ctx, const enum hmll_device device) {
    if (hmll_check(ctx->error))
        return ctx->error;

    struct hmll_io_uring *backend = calloc(1, sizeof(struct hmll_io_uring));
    hmll_io_uring_cca_init(&backend->iocca);

    struct io_uring_params params = {
        .flags = IORING_SETUP_SQPOLL | IORING_SETUP_SINGLE_ISSUER,
        .sq_thread_idle = 500
    };

    if (device == HMLL_DEVICE_CUDA) {
#if defined(__HMLL_CUDA_ENABLED__)
        struct hmll_io_uring_cuda_context *data = calloc(HMLL_URING_QUEUE_DEPTH, sizeof(struct hmll_io_uring_cuda_context));
        backend->device_ctx = (void *)data;

        for (int i = 0; i < (int)HMLL_URING_QUEUE_DEPTH; ++i) {
            data[i].slot = i;
            CHECK_CUDA(cudaStreamCreateWithFlags(&data[i].stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(&data[i].done, cudaEventDisableTiming));
        }


        int res = 0;
        if ((res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params)) < 0) {
            ctx->error = HMLL_SYS_ERR(res);
            return ctx->error;
        }

        ctx->error = hmll_io_uring_register_staging_buffers(ctx, backend, device);
        if (hmll_check(ctx->error)) {
            return ctx->error;
        }

#else
        ctx->error = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
        return ctx->error;
#endif
    } else {
        io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
    }

    {
        int *iofiles = calloc(ctx->num_sources, sizeof(int));
        for (size_t i = 0; i < ctx->num_sources; ++i)
            iofiles[i] = ctx->sources[i].fd;

        const int res = io_uring_register_files(&backend->ioring, iofiles, ctx->num_sources);
        free(iofiles);

        if (res != 0) {
            ctx->error = HMLL_ERR(HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED);
            goto cleanup;
        }
    }

    if (ctx->fetcher == NULL) {
        ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
        ctx->fetcher->device = device;
        ctx->fetcher->backend_impl_ = backend;
        ctx->fetcher->fetch_range_impl_ = hmll_io_uring_fetch_range;
        ctx->fetcher->fetchv_range_impl_ = hmll_io_uring_fetchv_range;
    }


    return HMLL_OK;

cleanup:
    if (backend->ioring.ring_fd > 0)
        io_uring_queue_exit(&backend->ioring);

#if defined(__HMLL_CUDA_ENABLED__)
    if (backend->device_ctx)
        free(backend->device_ctx);
#endif

    free(backend);
    return ctx->error;
}

