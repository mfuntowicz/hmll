#include <stdlib.h>
#include <string.h>
#include "hmll/hmll.h"
#include "hmll/cuda.h"
#include "hmll/memory.h"
#include "hmll/unix/backend/iouring.h"
#include "sys/mman.h"
#include <sys/utsname.h>

#define HMLL_IO_URING_ADVISORY_FLAG UINT64_MAX

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif


static inline int hmll_io_uring_get_setup_flags(void)
{
    int flags = IORING_SETUP_SQPOLL;

    // retrieve the current kernel version so we can adjust io_uring flags
    struct utsname unamedata;
    uname(&unamedata);

    int major, minor, revision = 0;
    if (sscanf(unamedata.release, "%d.%d.%d", &major, &minor, &revision)) {
        if (major >= 6) flags |= IORING_SETUP_SINGLE_ISSUER;
    }

    return flags;
}

static struct hmll_error hmll_io_uring_register_staging_buffers(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const enum hmll_device device
) {
    fetcher->iovecs = hmll_alloc(HMLL_URING_QUEUE_DEPTH * sizeof(struct iovec), HMLL_DEVICE_CPU, HMLL_MEM_DEVICE);
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
        ctx->error = HMLL_SYS_ERR(res);
        return ctx->error;
    }

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

static ssize_t hmll_io_uring_fetch_range_impl(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct hmll_iobuf *dst,
    const struct hmll_range range,
    const int iofile
) {
    if (hmll_check(ctx->error)) return -1;

    size_t n_dma = 0;
    size_t b_read = 0;
    size_t b_submitted = 0;
    struct io_uring_cqe *cqes[HMLL_URING_CQE_BATCH_SIZE];

    const size_t size = hmll_range_size(range);
    struct io_uring_sqe *sqe = NULL;
    int slot;
    if (likely((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) >= 0)) {
        io_uring_prep_fadvise(sqe, iofile, range.start, size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
        io_uring_sqe_set_data64(sqe, HMLL_IO_URING_ADVISORY_FLAG);
    }

    while (b_read < size) {
        hmll_io_uring_reclaim_slots(fetcher, dst->device);

        while (b_submitted < size) {
            if (unlikely((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) < 0))
                break;

            const size_t remaining = size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = range.start + b_submitted;

            hmll_io_uring_prep_sqe(fetcher, dst->device, sqe, (char *)dst->ptr + b_submitted, file_offset, to_read, iofile, slot);

            b_submitted += to_read;
            ++n_dma;
        }

        // update congestion control algorithm
        if (likely(n_dma > 0)) {
             const size_t nwait = MIN(n_dma, fetcher->iocca.window);

            struct timespec ts_start, ts_end;
            clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_start);

            if (unlikely(io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0)) {
                // todo: do we need to reset the cca? hmll_io_uring_cca_init(&fetcher->iocca)
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                return -1;
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
                if (unlikely(cqe->user_data == HMLL_IO_URING_ADVISORY_FLAG)) continue;
                if (unlikely(cqe->res < 0)) {
                    ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                    io_uring_cq_advance(&fetcher->ioring, i + 1);
                    return -1;
                }

                b_read += cqe->res;
                hmll_io_uring_handle_completion(fetcher, cqe, dst, range.start, cqe->res);
            }

            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    return (ssize_t)b_read;
}

static ssize_t hmll_io_uring_fetchv_range_impl(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct hmll_iobuf *dsts,
    const struct hmll_range *ranges,
    const int iofile,
    const size_t n
) {
    if (unlikely(hmll_check(ctx->error))) return -1;

    struct fetch_state {
        size_t submitted;
        size_t size;
        unsigned char fadvise_sent;
    };

    struct fetch_state *states;
    uint32_t *active_indices;
    size_t *slot_offsets;

    _Alignas(16) uint8_t stack_mem[8192];

    const size_t state_mem_req = sizeof(struct fetch_state) * n;
    const size_t idx_mem_req   = sizeof(uint32_t) * n;
    const size_t slot_mem_req  = sizeof(size_t) * HMLL_URING_QUEUE_DEPTH;
    const size_t total_req     = state_mem_req + idx_mem_req + slot_mem_req;

    if (likely(total_req <= sizeof(stack_mem))) {
        uint8_t *ptr = stack_mem;

        states = (struct fetch_state *)ptr;
        ptr += state_mem_req;

        active_indices = (uint32_t *)ptr;
        ptr += idx_mem_req;

        slot_offsets = (size_t *)ptr;
        memset(stack_mem, 0, total_req);
    } else {
        states = calloc(1, total_req);
        if (unlikely(!states)) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            return -1;
        }
        active_indices = (uint32_t *)((char *)states + state_mem_req);
        slot_offsets = (size_t *)((char *)active_indices + idx_mem_req);
    }

    size_t n_active = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t sz = ranges[i].end - ranges[i].start;
        states[i].submitted = 0;
        states[i].size = sz;
        states[i].fadvise_sent = false;

        if (sz > 0) {
            active_indices[n_active++] = i;
        }
    }

    const uint64_t BIT_FADVISE    = 1ULL << 63;
    const uint64_t SHIFT_RANGE    = 32;
    const uint64_t MASK_SLOT      = 0xFFFFFFFFULL;
    const unsigned char is_cuda = dsts[0].device == HMLL_DEVICE_CUDA;

    size_t n_in_flight = 0, nbytes = 0, active_cursor = 0;
    struct io_uring_cqe *cqes[HMLL_URING_CQE_BATCH_SIZE];

    while (n_active > 0 || n_in_flight > 0) {
        while (n_active > 0) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            if (active_cursor >= n_active) active_cursor = 0;
            const uint32_t current_idx = active_indices[active_cursor];
            struct fetch_state *st = &states[current_idx];

            if (unlikely(!st->fadvise_sent)) {
                io_uring_prep_fadvise(sqe, iofile, ranges[current_idx].start, st->size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
                io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                io_uring_sqe_set_data64(sqe, BIT_FADVISE);
                st->fadvise_sent = 1;
                continue;
            }

            int slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
            if (slot == -1) {
                hmll_io_uring_reclaim_slots(fetcher, dsts[0].device);
                slot = hmll_io_uring_slot_find_available(fetcher->iobusy);
                if (slot == -1) break;
            }

            hmll_io_uring_slot_set_busy(&fetcher->iobusy, slot);

            slot_offsets[slot] = st->submitted;
            const size_t remaining = st->size - st->submitted;
            const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = ranges[current_idx].start + st->submitted;

            hmll_io_uring_prep_sqe(
                fetcher,
                dsts[current_idx].device,
                sqe,
                (char *)dsts[current_idx].ptr + st->submitted,
                file_offset,
                to_read,
                iofile,
                slot
            );

            io_uring_sqe_set_data64(sqe, ((uint64_t)current_idx << SHIFT_RANGE) | slot);

            st->submitted += to_read;
            n_in_flight++;

            if (st->submitted >= st->size) {
                n_active--;
                active_indices[active_cursor] = active_indices[n_active];
            } else {
                active_cursor++;
            }
        }

        size_t nwait = 0;
        if (n_in_flight > 0) {
            nwait = (n_in_flight < fetcher->iocca.window) ? n_in_flight : fetcher->iocca.window;
        } else if (n_active == 0) {
            break;
        }

        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_start);

        if (unlikely(io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0)) {
            ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC_COARSE, &ts_end);

        if (nwait > 0) hmll_io_uring_cca_update(&fetcher->iocca, HMLL_URING_BUFFER_SIZE * nwait, ts_start, ts_end);

        unsigned count;
        while ((count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, HMLL_URING_CQE_BATCH_SIZE)) > 0) {
            for (unsigned i = 0; i < count; i++) {
                const struct io_uring_cqe *cqe = cqes[i];
                const uint64_t data = cqe->user_data;

                if (unlikely(data & BIT_FADVISE)) continue;

                n_in_flight--;

                if (unlikely(cqe->res < 0)) {
                    ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                    io_uring_cq_advance(&fetcher->ioring, count);
                    goto cleanup;
                }

                nbytes += cqe->res;

                const uint32_t r_idx = (uint32_t)(data >> SHIFT_RANGE);
                const uint32_t s_idx = (uint32_t)(data & MASK_SLOT);

                if (!is_cuda) {
                    hmll_io_uring_slot_set_available(&fetcher->iobusy, s_idx);
                }
#if defined(__HMLL_CUDA_ENABLED__)
                else {
                    struct hmll_io_uring_cuda_context *cctx = &((struct hmll_io_uring_cuda_context *)fetcher->device_ctx)[s_idx];

                    void *to = (char *)dsts[r_idx].ptr + slot_offsets[s_idx];
                    void *from = fetcher->iovecs[s_idx].iov_base;

                    cudaMemcpyAsync(to, from, cqe->res, cudaMemcpyHostToDevice, cctx->stream);
                    cudaEventRecord(cctx->done, cctx->stream);
                    hmll_io_uring_cuda_stream_set_memcpy(&cctx->state);
                }
#endif
            }
            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    if ((unsigned char*)states != stack_mem) free(states);
    return (ssize_t)nbytes;

cleanup:
    if ((unsigned char*)states != stack_mem) free(states);
    return -1;
}

static ssize_t hmll_io_uring_fetch_range(
    struct hmll *ctx,
    void *fetcher,
    const int iofile,
    const struct hmll_iobuf *dst,
    const struct hmll_range range
) {
    if (hmll_check(ctx->error))
        return -1;

    return hmll_io_uring_fetch_range_impl(ctx, fetcher, dst, range, iofile);
}

static ssize_t hmll_io_uring_fetchv_range(
    struct hmll *ctx,
    void *fetcher,
    const int iofile,
    const struct hmll_iobuf *dsts,
    const struct hmll_range *ranges,
    const size_t n
) {
    if (hmll_check(ctx->error))
        return -1;

    return hmll_io_uring_fetchv_range_impl(ctx, fetcher, dsts, ranges, iofile, n);
}

struct hmll_error hmll_io_uring_init(struct hmll *ctx, const enum hmll_device device) {
    if (hmll_check(ctx->error))
        return ctx->error;

    struct hmll_io_uring *backend = calloc(1, sizeof(struct hmll_io_uring));
    hmll_io_uring_cca_init(&backend->iocca);

    struct io_uring_params params = {
        .flags = hmll_io_uring_get_setup_flags(),
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
        int res;
        if ((res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params)) < 0) {
            ctx->error = HMLL_SYS_ERR(res);
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

