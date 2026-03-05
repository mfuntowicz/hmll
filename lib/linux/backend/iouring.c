#include <stdlib.h>
#include <string.h>
#include "hmll/hmll.h"
#include "hmll/cuda.h"
#include "hmll/memory.h"
#include "hmll/linux/backend/iouring.h"
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
 * `reg_fd` is the registered file index (already mapped through HMLL_IOFILE_*).
 */
static inline void hmll_io_uring_prep_sqe(
    const struct hmll_io_uring *fetcher,
    const struct hmll_device device,
    struct io_uring_sqe *sqe,
    void *dst,
    const size_t offset,
    const size_t len,
    const int reg_fd,
    const int slot
) {
    if (hmll_device_is_cpu(device)) {
        io_uring_prep_read(sqe, reg_fd, dst, len, offset);
        io_uring_sqe_set_data64(sqe, slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (hmll_device_is_cuda(device)) {
        struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
        void *buf = fetcher->iovecs[slot].iov_base;

        dctx[slot].offset = offset;
        io_uring_prep_read_fixed(sqe, reg_fd, buf, len, offset, slot);
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

/**
 * Submits chunked read SQEs and drains CQEs until `total` bytes are read.
 * `buf_offset` is the offset into dst->ptr where data starts being placed.
 * `file_offset` is the starting offset in the file.
 * `reg_fd` is the registered file index to target.
 */
static ssize_t hmll_io_uring_drain_reads(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct hmll_iobuf *dst,
    const size_t buf_offset,
    const size_t file_offset,
    const size_t total,
    const int reg_fd
) {
    size_t n_dma = 0, b_read = 0, b_submitted = 0;
    struct io_uring_cqe *cqes[HMLL_URING_CQE_BATCH_SIZE];
    struct io_uring_sqe *sqe;
    int slot;

    while (b_read < total) {
        hmll_io_uring_reclaim_slots(fetcher, dst->device);

        while (b_submitted < total) {
            if (unlikely((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) < 0))
                break;

            const size_t remaining = total - b_submitted;
            const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? remaining : HMLL_URING_BUFFER_SIZE;

            hmll_io_uring_prep_sqe(
                fetcher, dst->device, sqe,
                (char *)dst->ptr + buf_offset + b_submitted,
                file_offset + b_submitted,
                to_read, reg_fd, slot
            );

            b_submitted += to_read;
            ++n_dma;
        }

        if (likely(n_dma > 0)) {
            const size_t nwait = n_dma < fetcher->iocca.window ? n_dma : fetcher->iocca.window;

            struct timespec ts_start, ts_end;
            clock_gettime(CLOCK_MONOTONIC, &ts_start);

            if (unlikely(io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0)) {
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                return -1;
            }
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            hmll_io_uring_cca_update(&fetcher->iocca, HMLL_URING_BUFFER_SIZE * nwait, ts_start, ts_end);
        }

        unsigned count = 0;
        while ((count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, HMLL_URING_CQE_BATCH_SIZE)) > 0) {
            for (unsigned i = 0; i < count; i++) {
                const struct io_uring_cqe *cqe = cqes[i];
                if (unlikely(cqe->user_data == HMLL_IO_URING_ADVISORY_FLAG))
                    continue;

                --n_dma;
                if (unlikely(cqe->res < 0)) {
                    ctx->error = HMLL_SYS_ERR(-cqe->res);
                    io_uring_cq_advance(&fetcher->ioring, i + 1);
                    return -1;
                }

                b_read += cqe->res;
                hmll_io_uring_handle_completion(fetcher, cqe, dst, file_offset, cqe->res);
            }
            io_uring_cq_advance(&fetcher->ioring, count);
        }
    }

    return (ssize_t)b_read;
}

static ssize_t hmll_io_uring_fetch_range_impl(
    struct hmll *ctx,
    const int iofile,
    const struct hmll_iobuf *dst,
    const size_t offset
) {
    if (hmll_check(ctx->error)) return -1;

    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;
    const int fd_buf = HMLL_IOFILE_BUFFERED(iofile);
    const int fd_dir = HMLL_IOFILE_DIRECT(iofile);
    const size_t end = offset + dst->size;

    const size_t aligned_start = ALIGN_UP(offset, ALIGN_PAGE);
    const size_t aligned_end   = ALIGN_DOWN(end, ALIGN_PAGE);

    const unsigned char use_direct =
        fetcher->has_direct
        && hmll_device_is_cuda(dst->device)
        && aligned_end > aligned_start;

    const size_t head_size = use_direct ? (aligned_start - offset) : 0;
    const size_t core_size = use_direct ? (aligned_end - aligned_start) : 0;
    const size_t tail_size = use_direct ? (end - aligned_end) : 0;

    struct io_uring_sqe *sqe;
    ssize_t total_read = 0;

    if (!use_direct) {
        // Pure buffered path: fadvise + chunked reads through buffered fd
        if ((sqe = io_uring_get_sqe(&fetcher->ioring))) {
            io_uring_prep_fadvise(sqe, fd_buf, offset, dst->size,
                                  POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data64(sqe, HMLL_IO_URING_ADVISORY_FLAG);
        }

        total_read = hmll_io_uring_drain_reads(
            ctx, fetcher, dst, 0, offset, dst->size, fd_buf);
        if (total_read < 0) return -1;

        hmll_io_uring_sync(dst->device, fetcher);
        return total_read;
    }

    // --- Hybrid O_DIRECT path (CUDA only) ---

    // 1) fadvise head and tail on the buffered fd so the kernel prefetches
    if (head_size > 0 && (sqe = io_uring_get_sqe(&fetcher->ioring))) {
        io_uring_prep_fadvise(sqe, fd_buf, offset, head_size, POSIX_FADV_WILLNEED);
        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
        io_uring_sqe_set_data64(sqe, HMLL_IO_URING_ADVISORY_FLAG);
    }
    if (tail_size > 0 && (sqe = io_uring_get_sqe(&fetcher->ioring))) {
        io_uring_prep_fadvise(sqe, fd_buf, aligned_end, tail_size, POSIX_FADV_WILLNEED);
        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
        io_uring_sqe_set_data64(sqe, HMLL_IO_URING_ADVISORY_FLAG);
    }

    // 2) Read aligned core via O_DIRECT fd
    ssize_t n = hmll_io_uring_drain_reads(
        ctx, fetcher, dst, head_size, aligned_start, core_size, fd_dir);
    if (n < 0) return -1;
    total_read += n;

    // 3) Read head via buffered fd
    if (head_size > 0) {
        n = hmll_io_uring_drain_reads(
            ctx, fetcher, dst, 0, offset, head_size, fd_buf);
        if (n < 0) return -1;
        total_read += n;
    }

    // 4) Read tail via buffered fd
    if (tail_size > 0) {
        n = hmll_io_uring_drain_reads(
            ctx, fetcher, dst, head_size + core_size, aligned_end, tail_size, fd_buf);
        if (n < 0) return -1;
        total_read += n;
    }

    hmll_io_uring_sync(dst->device, fetcher);
    return total_read;
}

static ssize_t hmll_io_uring_fetchv_range_impl(
    struct hmll *ctx,
    const int iofile,
    const struct hmll_iobuf *dsts,
    const size_t *offsets,
    const size_t n
) {
    if (hmll_check(ctx->error)) return -1;

    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;
    const int fd_buf = HMLL_IOFILE_BUFFERED(iofile);
    const int fd_dir = HMLL_IOFILE_DIRECT(iofile);
    const unsigned char is_cuda = hmll_device_is_cuda(dsts[0].device);
    const unsigned char use_direct = fetcher->has_direct && is_cuda;

    enum fetch_phase { PHASE_CORE, PHASE_FRINGE };

    struct fetch_state {
        size_t submitted;
        size_t size;

        size_t head_size;
        size_t core_offset;
        size_t core_size;
        size_t tail_offset;
        size_t tail_size;

        unsigned char fadvise_sent;
        unsigned char phase;
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
        states = (struct fetch_state *)ptr; ptr += state_mem_req;
        active_indices = (uint32_t *)ptr;   ptr += idx_mem_req;
        slot_offsets = (size_t *)ptr;
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
        if (dsts[i].size == 0) continue;

        struct fetch_state *st = &states[i];
        const size_t off = offsets[i];
        const size_t end = off + dsts[i].size;
        const size_t a_start = ALIGN_UP(off, ALIGN_PAGE);
        const size_t a_end   = ALIGN_DOWN(end, ALIGN_PAGE);

        st->size = dsts[i].size;
        st->submitted = 0;
        st->fadvise_sent = 0;

        if (use_direct && a_end > a_start) {
            st->head_size    = a_start - off;
            st->core_offset  = a_start;
            st->core_size    = a_end - a_start;
            st->tail_offset  = a_end;
            st->tail_size    = end - a_end;
            st->phase        = PHASE_CORE;
        } else {
            st->head_size    = 0;
            st->core_offset  = 0;
            st->core_size    = 0;
            st->tail_offset  = 0;
            st->tail_size    = 0;
            st->phase        = PHASE_FRINGE;
        }

        active_indices[n_active++] = i;
    }

    const uint64_t BIT_FADVISE = 1ULL << 63;
    const uint64_t SHIFT_RANGE = 32;
    const uint64_t MASK_SLOT   = 0xFFFFFFFFULL;

    size_t n_in_flight = 0, nbytes = 0, active_cursor = 0;
    struct io_uring_cqe *cqes[HMLL_URING_CQE_BATCH_SIZE];

    // Tracks ranges whose core is done and need fringe (head+tail) reads
    uint32_t fringe_indices[128];
    size_t n_fringe = 0;

    while (n_active > 0 || n_in_flight > 0 || n_fringe > 0) {

        // Enqueue fringe ranges that just finished their core phase
        while (n_fringe > 0 && n_active < n) {
            active_indices[n_active++] = fringe_indices[--n_fringe];
        }

        while (n_active > 0) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            if (!sqe) break;

            if (active_cursor >= n_active) active_cursor = 0;
            const uint32_t current_idx = active_indices[active_cursor];
            struct fetch_state *st = &states[current_idx];

            // Submit fadvise before first read of this range
            if (unlikely(!st->fadvise_sent)) {
                if (st->phase == PHASE_CORE) {
                    // Prefetch head + tail on buffered fd while core goes through O_DIRECT
                    if (st->head_size > 0) {
                        io_uring_prep_fadvise(sqe, fd_buf, offsets[current_idx],
                                              st->head_size, POSIX_FADV_WILLNEED);
                        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                        io_uring_sqe_set_data64(sqe, BIT_FADVISE);

                        if (st->tail_size > 0) {
                            struct io_uring_sqe *sqe2 = io_uring_get_sqe(&fetcher->ioring);
                            if (sqe2) {
                                io_uring_prep_fadvise(sqe2, fd_buf, st->tail_offset,
                                                      st->tail_size, POSIX_FADV_WILLNEED);
                                io_uring_sqe_set_flags(sqe2, IOSQE_FIXED_FILE);
                                io_uring_sqe_set_data64(sqe2, BIT_FADVISE);
                            }
                        }
                    } else if (st->tail_size > 0) {
                        io_uring_prep_fadvise(sqe, fd_buf, st->tail_offset,
                                              st->tail_size, POSIX_FADV_WILLNEED);
                        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                        io_uring_sqe_set_data64(sqe, BIT_FADVISE);
                    } else {
                        // Core-only, no head/tail -- issue fadvise for full range on direct fd
                        io_uring_prep_fadvise(sqe, fd_dir, st->core_offset,
                                              st->core_size, POSIX_FADV_SEQUENTIAL);
                        io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                        io_uring_sqe_set_data64(sqe, BIT_FADVISE);
                    }
                } else {
                    // PHASE_FRINGE (buffered-only): fadvise the whole range
                    io_uring_prep_fadvise(sqe, fd_buf, offsets[current_idx],
                                          st->size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
                    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                    io_uring_sqe_set_data64(sqe, BIT_FADVISE);
                }
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

            size_t file_offset, buf_offset, region_size;
            int target_fd;

            if (st->phase == PHASE_CORE) {
                // Submit core reads via O_DIRECT fd
                region_size = st->core_size;
                file_offset = st->core_offset + st->submitted;
                buf_offset  = st->head_size + st->submitted;
                target_fd   = fd_dir;
            } else {
                // PHASE_FRINGE: submit head, then tail, via buffered fd
                if (st->core_size > 0) {
                    // We had a core phase; now reading head then tail
                    if (st->submitted < st->head_size) {
                        region_size = st->head_size;
                        file_offset = offsets[current_idx] + st->submitted;
                        buf_offset  = st->submitted;
                    } else {
                        region_size = st->head_size + st->tail_size;
                        const size_t fringe_done = st->submitted - st->head_size;
                        file_offset = st->tail_offset + fringe_done;
                        buf_offset  = st->head_size + st->core_size + fringe_done;
                    }
                } else {
                    // Pure buffered (no O_DIRECT for this range)
                    region_size = st->size;
                    file_offset = offsets[current_idx] + st->submitted;
                    buf_offset  = st->submitted;
                }
                target_fd = fd_buf;
            }

            const size_t remaining = region_size - st->submitted;
            const size_t to_read = remaining < HMLL_URING_BUFFER_SIZE ? remaining : HMLL_URING_BUFFER_SIZE;

            slot_offsets[slot] = buf_offset;

            hmll_io_uring_prep_sqe(
                fetcher, dsts[current_idx].device, sqe,
                (char *)dsts[current_idx].ptr + buf_offset,
                file_offset, to_read, target_fd, slot
            );

            io_uring_sqe_set_data64(sqe, ((uint64_t)current_idx << SHIFT_RANGE) | slot);

            st->submitted += to_read;
            n_in_flight++;

            if (st->submitted >= region_size) {
                if (st->phase == PHASE_CORE && (st->head_size > 0 || st->tail_size > 0)) {
                    // Core done, transition to fringe reads
                    st->phase = PHASE_FRINGE;
                    st->submitted = 0;
                    st->fadvise_sent = 1;

                    // Remove from active and queue for fringe
                    n_active--;
                    active_indices[active_cursor] = active_indices[n_active];
                    fringe_indices[n_fringe++] = current_idx;
                } else {
                    // Fully done
                    n_active--;
                    active_indices[active_cursor] = active_indices[n_active];
                }
            } else {
                active_cursor++;
            }
        }

        size_t nwait = 0;
        if (n_in_flight > 0) {
            nwait = (n_in_flight < fetcher->iocca.window) ? n_in_flight : fetcher->iocca.window;
        } else if (n_active == 0 && n_fringe == 0) {
            break;
        }

        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        if (unlikely(io_uring_submit_and_wait(&fetcher->ioring, nwait) < 0)) {
            ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_end);

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

                const uint32_t s_idx = (uint32_t)(data & MASK_SLOT);

                if (!is_cuda) {
                    hmll_io_uring_slot_set_available(&fetcher->iobusy, s_idx);
                }
#if defined(__HMLL_CUDA_ENABLED__)
                else {
                    const uint32_t r_idx = (uint32_t)(data >> SHIFT_RANGE);
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

    hmll_io_uring_sync(dsts[0].device, fetcher);
    if ((unsigned char*)states != stack_mem) free(states);
    return (ssize_t)nbytes;

cleanup:
    if ((unsigned char*)states != stack_mem) free(states);
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

    const size_t n_iofiles = ctx->num_sources * 2;
    int *iofiles = calloc(n_iofiles, sizeof(int));
    unsigned char any_direct = 0;
    for (size_t i = 0; i < ctx->num_sources; ++i) {
        iofiles[i * 2] = ctx->sources[i].b_fd;
        if (ctx->sources[i].d_fd > 0) {
            iofiles[i * 2 + 1] = ctx->sources[i].d_fd;
            any_direct = 1;
        } else {
            iofiles[i * 2 + 1] = ctx->sources[i].b_fd;
        }
    }
    backend->has_direct = any_direct;

    const int res = io_uring_register_files(&backend->ioring, iofiles, (unsigned)n_iofiles);
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
        ctx->fetcher->fetch_range_impl_ = hmll_io_uring_fetch_range_impl;
        ctx->fetcher->fetchv_range_impl_ = hmll_io_uring_fetchv_range_impl;
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
