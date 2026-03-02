#include <stdlib.h>
#include "hmll/hmll.h"
#include "hmll/memory.h"
#include "hmll/linux/backend/iouring.h"
#include "sys/mman.h"

#define HMLL_IO_URING_FADVISE_TAG (1ULL << 63)

#if defined(__HMLL_CUDA_ENABLED__)
#include "hmll/cuda.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif


static inline int hmll_io_uring_get_setup_flags(void) { return IORING_SETUP_SQPOLL; }

static struct hmll_error hmll_io_uring_register_staging_buffers(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct hmll_device device
) {
    fetcher->iovecs = hmll_alloc(HMLL_URING_QUEUE_DEPTH * sizeof(struct iovec), hmll_device_cpu(), HMLL_MEM_DEVICE);
    if (!fetcher->iovecs) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        return ctx->error;
    }

    unsigned char *arena = hmll_alloc(HMLL_URING_QUEUE_DEPTH * HMLL_URING_BUFFER_SIZE, device, HMLL_MEM_STAGING);
    if (!arena) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        return ctx->error;
    }

    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
        fetcher->iovecs[i].iov_base = arena + i * HMLL_URING_BUFFER_SIZE;
        fetcher->iovecs[i].iov_len = HMLL_URING_BUFFER_SIZE;
    }

    int res;
    if ((res = io_uring_register_buffers(&fetcher->ioring, fetcher->iovecs, HMLL_URING_QUEUE_DEPTH)) < 0) {
        ctx->error = HMLL_SYS_ERR(-res);
        return ctx->error;
    }

    return HMLL_OK;
}

static inline void hmll_io_uring_sync(const struct hmll_device device, const struct hmll_io_uring *fetcher)
{
    if (hmll_device_is_cuda(device)) {
#ifdef __HMLL_CUDA_ENABLED__
        // Wait for all pending CUDA operations to complete
        for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
            if (hmll_io_uring_slot_is_busy(fetcher->iobusy, i)) {
                const struct hmll_io_uring_cuda_context *cd = (struct hmll_io_uring_cuda_context *)fetcher->device_ctx + i;
                if (cd->state == HMLL_CUDA_STREAM_MEMCPY)
                    cudaEventSynchronize(cd->done);
            }
        }
#endif
    }

    HMLL_UNUSED(fetcher);
}

/**
 * Checks for completed CUDA events and reclaims the associated io_uring slots.
 * If CUDA is disabled or the device is CPU, this is a no-op.
 */
static inline void hmll_io_uring_reclaim_slots(
    struct hmll_io_uring *fetcher,
    const struct hmll_device device
) {
#ifdef __HMLL_CUDA_ENABLED__
    if (!hmll_device_is_cuda(device)) return;

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
    const struct hmll_device device,
    struct io_uring_sqe *sqe,
    void *dst,
    const size_t offset,
    const size_t len,
    const unsigned short iofile,
    const int slot
) {
    if (hmll_device_is_cpu(device)) {
        // CPU: Read directly into user memory
        io_uring_prep_read(sqe, iofile, dst, len, offset);
        io_uring_sqe_set_data64(sqe, slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (hmll_device_is_cuda(device)) {
        // CUDA: Read into registered staging buffers
        struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
        void *buf = fetcher->iovecs[slot].iov_base;

        dctx[slot].offset = offset;
        io_uring_prep_read_fixed(sqe, iofile, buf, len, offset, slot);
        io_uring_sqe_set_data(sqe, dctx + slot);
    }
#else
    HMLL_UNUSED(fetcher);
#endif

    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
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
    if (hmll_device_is_cpu(dst->device)) {
        const uint64_t cb_slot = cqe->user_data;
        hmll_io_uring_slot_set_available(&fetcher->iobusy, cb_slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (hmll_device_is_cuda(dst->device)) {
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

static ssize_t hmll_io_uring_fetch_impl(
    struct hmll *ctx,
    const int iofile,
    const struct hmll_iobuf *dst,
    const size_t offset
) {
    if (hmll_check(ctx->error)) return -1;

    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;

    size_t n_dma = 0;
    size_t b_read = 0;
    size_t b_submitted = 0;
    struct io_uring_cqe *cqes[HMLL_URING_QUEUE_DEPTH];

    struct io_uring_sqe *sqe = NULL;
    int slot;
    if ((sqe = io_uring_get_sqe(&fetcher->ioring))) {
        io_uring_prep_fadvise(sqe, iofile, offset, dst->size, POSIX_FADV_WILLNEED);
        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
        io_uring_sqe_set_data64(sqe, HMLL_IO_URING_FADVISE_TAG);
    }

    while (b_read < dst->size) {
        hmll_io_uring_reclaim_slots(fetcher, dst->device);

        while (b_submitted < dst->size) {
            if (unlikely((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) < 0))
                break;

            const size_t remaining = dst->size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = offset + b_submitted;

            hmll_io_uring_prep_sqe(fetcher, dst->device, sqe, (char *)dst->ptr + b_submitted, file_offset, to_read, iofile, slot);

            b_submitted += to_read;
            ++n_dma;
        }

        // update congestion control algorithm
        if (likely(n_dma > 0)) {
             const size_t nwait = n_dma < fetcher->iocca.window ? n_dma : fetcher->iocca.window;

            struct timespec ts_start, ts_end;
            clock_gettime(CLOCK_MONOTONIC, &ts_start);

            if (unlikely(io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0)) {
                // todo: do we need to reset the cca? hmll_io_uring_cca_init(&fetcher->iocca)
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                return -1;
            }
            clock_gettime(CLOCK_MONOTONIC, &ts_end);

            // todo: approximated version of the number of bytes actually reads because it assumes full reads
            hmll_io_uring_cca_update(&fetcher->iocca, HMLL_URING_BUFFER_SIZE * nwait, ts_start, ts_end);
        }

        unsigned count = 0;
        while ((count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, fetcher->iocca.window)) > 0) {
            for (unsigned i = 0; i < count; i++) {

                const struct io_uring_cqe *cqe = cqes[i];
                if (unlikely(cqe->user_data == HMLL_IO_URING_FADVISE_TAG))
                    continue;

                --n_dma;
                if (unlikely(cqe->res < 0)) {
                    ctx->error = HMLL_SYS_ERR(-cqe->res);
                    io_uring_cq_advance(&fetcher->ioring, count);
                    return -1;
                }

                b_read += cqe->res;
                hmll_io_uring_handle_completion(fetcher, cqe, dst, offset, cqe->res);
            }

            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    hmll_io_uring_sync(dst->device, fetcher);
    return (ssize_t)b_read;
}

static ssize_t hmll_io_uring_fetchv_impl(
    struct hmll *ctx,
    const int iofile,
    const struct hmll_iobuf *dsts,
    const size_t *offsets,
    const size_t n
) {
    if (hmll_check(ctx->error)) return -1;
    if (unlikely(n == 0)) return 0;

    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;
    const int is_cuda = (dsts[0].device == HMLL_DEVICE_CUDA);

    /* user_data encoding for CQEs: high bit = fadvise (skip), else (bidx << 8) | slot */
    static const unsigned FETCHV_BIDX_SHIFT = 8;
    static const uint64_t FETCHV_SLOT_MASK = HMLL_URING_QUEUE_DEPTH - 1;

    struct fetchv_buf_state {
        size_t submitted;
        size_t size;
        unsigned char fadvise_sent;
    };

    /* Scratch layout: [buf_states][active_indices][slot_offsets] */
    const size_t sz_state = (sizeof(struct fetchv_buf_state) * n + _Alignof(uint32_t) - 1) & ~(_Alignof(uint32_t) - 1);
    const size_t sz_idx   = (sizeof(uint32_t) * n + _Alignof(size_t) - 1) & ~(_Alignof(size_t) - 1);
    const size_t sz_slot  = sizeof(size_t) * HMLL_URING_QUEUE_DEPTH;
    const size_t scratch_size = sz_state + sz_idx + sz_slot;

    _Alignas(16) uint8_t stack_scratch[8192];
    struct fetchv_buf_state *buf_states;
    uint32_t *active_indices;
    size_t *slot_offsets;
    void *scratch_to_free = NULL;

    if (scratch_size <= sizeof(stack_scratch)) {
        uint8_t *p = stack_scratch;
        buf_states     = (struct fetchv_buf_state *)p; p += sz_state;
        active_indices = (uint32_t *)p;                p += sz_idx;
        slot_offsets   = (size_t *)p;
    } else {
        void *p = calloc(1, scratch_size);
        if (!p) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            return -1;
        }
        scratch_to_free = p;
        buf_states     = (struct fetchv_buf_state *)p; p = (char *)p + sz_state;
        active_indices = (uint32_t *)p;                p = (char *)p + sz_idx;
        slot_offsets   = (size_t *)p;
    }

    /* Build list of buffers that have bytes to read */
    size_t n_active = 0;
    for (size_t i = 0; i < n; i++) {
        buf_states[i].submitted = 0;
        buf_states[i].size      = dsts[i].size;
        buf_states[i].fadvise_sent = 0;
        if (dsts[i].size > 0)
            active_indices[n_active++] = (uint32_t)i;
    }

    const unsigned char is_cuda = hmll_device_is_cuda(dsts[0].device);
    size_t n_in_flight = 0, nbytes = 0, active_cursor = 0;
    struct io_uring_cqe *cqes[HMLL_URING_QUEUE_DEPTH];

    while (n_active > 0 || n_in_flight > 0) {
        hmll_io_uring_reclaim_slots(fetcher, dsts[0].device);

        /* Submit: round-robin over active buffers, send fadvise then chunked reads */
        while (n_active > 0) {
            if (active_cursor >= n_active) active_cursor = 0;
            const uint32_t bidx = active_indices[active_cursor];
            struct fetchv_buf_state *st = &buf_states[bidx];

            if (!st->fadvise_sent) {
                struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
                if (!sqe) break;
                io_uring_prep_fadvise(sqe, iofile, offsets[bidx], st->size, POSIX_FADV_WILLNEED);
                io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                io_uring_sqe_set_data64(sqe, HMLL_IO_URING_FADVISE_TAG);
                st->fadvise_sent = 1;
                continue;
            }

            const int slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
            if (slot < 0) break;
            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);
            slot_offsets[slot] = st->submitted;

            const size_t remaining = st->size - st->submitted;
            const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_off = offsets[bidx] + st->submitted;
            void *read_dst = (char *)dsts[bidx].ptr + st->submitted;

#if defined(__HMLL_CUDA_ENABLED__)
            if (is_cuda)
                io_uring_prep_read_fixed(sqe, iofile, fetcher->iovecs[slot].iov_base, to_read, file_off, slot);
            else
                io_uring_prep_read(sqe, iofile, read_dst, to_read, file_off);
#else
            io_uring_prep_read(sqe, iofile, read_dst, to_read, file_off);
#endif
            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data64(sqe, ((uint64_t)bidx << FETCHV_BIDX_SHIFT) | (uint64_t)slot);

            st->submitted += to_read;
            n_in_flight++;

            if (st->submitted >= st->size) {
                n_active--;
                active_indices[active_cursor] = active_indices[n_active];
            } else {
                active_cursor++;
            }
        }

        if (n_in_flight == 0) {
            if (n_active == 0) break;
            io_uring_submit(&fetcher->ioring);
#if defined(__HMLL_CUDA_ENABLED__)
            if (is_cuda && hmll_io_uring_slot_find_available(fetcher->iobusy) < 0) {
                struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
                for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; i++) {
                    if (hmll_io_uring_slot_is_busy(fetcher->iobusy, i)) {
                        struct hmll_io_uring_cuda_context *cd = &dctx[i];
                        if (cd->state == HMLL_CUDA_STREAM_MEMCPY) {
                            cudaEventSynchronize(cd->done);
                            hmll_io_uring_cuda_stream_set_idle(&cd->state);
                            hmll_io_uring_slot_set_available(&fetcher->iobusy, (unsigned)i);
                            break;
                        }
                    }
                }
            }
#endif
            continue;
        }

        const size_t nwait = n_in_flight < fetcher->iocca.window ? n_in_flight : fetcher->iocca.window;
        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        if (io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0) {
            ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        hmll_io_uring_cca_update(&fetcher->iocca, HMLL_URING_BUFFER_SIZE * nwait, ts_start, ts_end);

        unsigned count;
        while ((count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, fetcher->iocca.window)) > 0) {
            for (unsigned i = 0; i < count; i++) {
                const struct io_uring_cqe *cqe = cqes[i];
                const uint64_t data = cqe->user_data;

                if (data == HMLL_IO_URING_FADVISE_TAG) continue;

                n_in_flight--;
                if (cqe->res < 0) {
                    ctx->error = HMLL_SYS_ERR(-cqe->res);
                    io_uring_cq_advance(&fetcher->ioring, count);
                    goto cleanup;
                }
                nbytes += (size_t)cqe->res;

                const uint32_t slot = (uint32_t)(data & FETCHV_SLOT_MASK);
                const uint32_t bidx = (uint32_t)(data >> FETCHV_BIDX_SHIFT);

                if (!is_cuda) {
                    hmll_io_uring_slot_set_available(&fetcher->iobusy, slot);
                }
#if defined(__HMLL_CUDA_ENABLED__)
                else {
                    struct hmll_io_uring_cuda_context *cctx = &((struct hmll_io_uring_cuda_context *)fetcher->device_ctx)[slot];
                    void *to = (char *)dsts[bidx].ptr + slot_offsets[slot];
                    void *from = fetcher->iovecs[slot].iov_base;
                    cudaMemcpyAsync(to, from, (size_t)cqe->res, cudaMemcpyHostToDevice, cctx->stream);
                    cudaEventRecord(cctx->done, cctx->stream);
                    hmll_io_uring_cuda_stream_set_memcpy(&cctx->state);
                }
#else
                (void)bidx;
#endif
            }
            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    hmll_io_uring_sync(dsts[0].device, fetcher);
    if (scratch_to_free) free(scratch_to_free);
    return (ssize_t)nbytes;

cleanup:
    if (scratch_to_free) free(scratch_to_free);
    return -1;
}

struct hmll_error hmll_io_uring_init(struct hmll *ctx, const struct hmll_device device) {
    if (hmll_check(ctx->error))
        return ctx->error;

    struct hmll_io_uring *backend = calloc(1, sizeof(struct hmll_io_uring));
    hmll_io_uring_cca_init(&backend->iocca);

    struct io_uring_params params = {
        .flags = hmll_io_uring_get_setup_flags(),
        .sq_thread_idle = 500
    };

    if (hmll_device_is_cuda(device)) {
#if defined(__HMLL_CUDA_ENABLED__)
        cudaError_t cuda_err = cudaSetDevice(device.idx);
        if (cuda_err != cudaSuccess) {
            ctx->error = HMLL_ERR(HMLL_ERR_CUDA_SET_DEVICE_FAILED);
            return ctx->error;
        }

        struct hmll_io_uring_cuda_context *data = calloc(HMLL_URING_QUEUE_DEPTH, sizeof(struct hmll_io_uring_cuda_context));
        backend->device_ctx = (void *)data;

        for (int i = 0; i < (int)HMLL_URING_QUEUE_DEPTH; ++i) {
            data[i].slot = i;
            CHECK_CUDA(cudaStreamCreateWithFlags(&data[i].stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(&data[i].done, cudaEventDisableTiming));
        }

        int res = 0;
        if ((res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params)) < 0) {
            ctx->error = HMLL_SYS_ERR(-res);
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
        int res;
        if ((res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params)) < 0) {
            ctx->error = HMLL_SYS_ERR(-res);
            goto cleanup;
        }
    }

    int *iofiles = calloc(ctx->num_sources, sizeof(int));
    for (size_t i = 0; i < ctx->num_sources; ++i)
        iofiles[i] = ctx->sources[i].fd;

    const int res = io_uring_register_files(&backend->ioring, iofiles, ctx->num_sources);
    free(iofiles);

    if (res != 0) {
        ctx->error = HMLL_ERR(HMLL_ERR_FILE_REGISTRATION_FAILED);
        goto cleanup;
    }

    if (ctx->fetcher == NULL) {
        ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
        ctx->fetcher->kind = HMLL_FETCHER_IO_URING;
        ctx->fetcher->device = device;
        ctx->fetcher->backend_impl_ = backend;
        ctx->fetcher->fetch_range_impl_ = hmll_io_uring_fetch_impl;
        ctx->fetcher->fetchv_range_impl_ = hmll_io_uring_fetchv_impl;
        ctx->fetcher->backend_free = hmll_io_uring_destroy;
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


void hmll_io_uring_destroy(void *ptr)
{
    if (!ptr) return;

    struct hmll_io_uring *backend = ptr;
    io_uring_unregister_buffers(&backend->ioring);

#if defined(__HMLL_CUDA_ENABLED__)
    if (backend->device_ctx) {
        struct hmll_io_uring_cuda_context *cuda_ctx = backend->device_ctx;
        for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; ++i) {
            if (cuda_ctx[i].done) {
                cudaEventDestroy(cuda_ctx[i].done);
            }
            if (cuda_ctx[i].stream) {
                cudaStreamDestroy(cuda_ctx[i].stream);
            }
        }

        munmap(backend->iovecs[0].iov_base, HMLL_URING_QUEUE_DEPTH * sizeof(struct iovec));
        free(backend->device_ctx);
        backend->device_ctx = NULL;
    }
#endif

    io_uring_queue_exit(&backend->ioring);
    free(ptr);
}
