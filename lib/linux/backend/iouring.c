#include <stdlib.h>
#include <sys/utsname.h>
#include "hmll/hmll.h"
#include "hmll/memory.h"
#include "hmll/linux/backend/iouring.h"
#include "sys/mman.h"

#define HMLL_IO_URING_FADVISE_TAG (1ULL << 63)
#define HMLL_IO_URING_DEFAULT_QUEUE_PARAMS IORING_SETUP_SQPOLL

#if defined(__HMLL_CUDA_ENABLED__)
#include "hmll/cuda.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif

/* ── runtime kernel version detection ───────────────────────────────── */
static inline unsigned hmll_kernel_version_internal(unsigned maj, unsigned min)
{
    return (maj << 16) | min;
}

static unsigned hmll_kernel_version(void)
{
    static unsigned cached = 0;
    if (cached) return cached;

    struct utsname u;
    if (uname(&u) != 0) return 0;

    unsigned maj = 0, min = 0;
    if (sscanf(u.release, "%u.%u", &maj, &min) < 2) return 0;

    cached = hmll_kernel_version_internal(maj, min);
    return cached;
}

/**
 * Build io_uring setup flags based on the running kernel version.
 *
 *  SQPOLL          (always)  — kernel thread polls SQ, eliminates submit syscalls
 *  COOP_TASKRUN    (>= 6.1)  — no IPI for CQE delivery, process on next syscall
 *  TASKRUN_FLAG    (>= 6.1)  — companion: sets SQ flag when CQEs are pending
 */
static inline int hmll_io_uring_get_setup_flags(void)
{
    const unsigned kversion = hmll_kernel_version();

    int flags = HMLL_IO_URING_DEFAULT_QUEUE_PARAMS;
    if (kversion >= hmll_kernel_version_internal(6, 1)) {
        flags |= IORING_SETUP_COOP_TASKRUN | IORING_SETUP_TASKRUN_FLAG;
    }

    return flags;
}

/* ── scratch allocator (stack with heap fallback) ───────────────────── */

static inline void *hmll_scratch_alloc(
    uint8_t *stack, const size_t size, const size_t need, void **to_free
) {
    *to_free = NULL;
    if (need <= size) return stack;
    void *p = calloc(1, need);
    *to_free = p;
    return p;
}

/* ── staging buffer registration ────────────────────────────────────── */

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

/* ── fadvise helper ─────────────────────────────────────────────────── */

static inline void hmll_io_uring_queue_fadvise(
    struct hmll_io_uring *fetcher, const unsigned iofd, const size_t off, const size_t len
) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
    if (!sqe) return;
    io_uring_prep_fadvise(sqe, (int)iofd, off, len, POSIX_FADV_WILLNEED);
    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
    io_uring_sqe_set_data64(sqe, HMLL_IO_URING_FADVISE_TAG);
}

/* ── CUDA helpers ───────────────────────────────────────────────────── */

static inline void hmll_io_uring_sync(const struct hmll_device device, const struct hmll_io_uring *fetcher)
{
    if (hmll_device_is_cuda(device)) {
#ifdef __HMLL_CUDA_ENABLED__
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

static inline void hmll_io_uring_reclaim_slots(
    struct hmll_io_uring *fetcher,
    const struct hmll_device device
) {
#ifdef __HMLL_CUDA_ENABLED__
    if (!hmll_device_is_cuda(device)) return;

    struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
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
 * When all slots are busy with CUDA memcpy, synchronously wait on one event
 * to free a slot.  No-op when a slot is already available.
 */
static inline void hmll_io_uring_cuda_relieve_pressure(struct hmll_io_uring *fetcher)
{
#ifdef __HMLL_CUDA_ENABLED__
    if (hmll_io_uring_slot_find_available(fetcher->iobusy) >= 0) return;

    struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
    for (size_t i = 0; i < HMLL_URING_QUEUE_DEPTH; i++) {
        if (!hmll_io_uring_slot_is_busy(fetcher->iobusy, i)) continue;
        struct hmll_io_uring_cuda_context *cd = &dctx[i];
        if (cd->state == HMLL_CUDA_STREAM_MEMCPY) {
            cudaEventSynchronize(cd->done);
            hmll_io_uring_cuda_stream_set_idle(&cd->state);
            hmll_io_uring_slot_set_available(&fetcher->iobusy, (unsigned)i);
            return;
        }
    }
#else
    HMLL_UNUSED(fetcher);
#endif
}

/* ── SQE / CQE primitives ──────────────────────────────────────────── */

static inline void hmll_io_uring_prep_sqe(
    const struct hmll_io_uring *fetcher,
    const struct hmll_device device,
    struct io_uring_sqe *sqe,
    void *dst,
    const size_t offset,
    const size_t len,
    const int iofile,
    const int slot
) {
    if (hmll_device_is_cpu(device)) {
        io_uring_prep_read(sqe, iofile, dst, len, offset);
        io_uring_sqe_set_data64(sqe, slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (hmll_device_is_cuda(device)) {
        struct hmll_io_uring_cuda_context *dctx = fetcher->device_ctx;
        dctx[slot].offset = offset;
        io_uring_prep_read_fixed(sqe, iofile, fetcher->iovecs[slot].iov_base, len, offset, slot);
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

static inline void hmll_io_uring_handle_completion(
    struct hmll_io_uring *fetcher,
    const struct io_uring_cqe *cqe,
    const struct hmll_iobuf *dst,
    const size_t offset,
    const int32_t len
) {
    if (hmll_device_is_cpu(dst->device)) {
        hmll_io_uring_slot_set_available(&fetcher->iobusy, cqe->user_data);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else if (hmll_device_is_cuda(dst->device)) {
        struct hmll_io_uring_cuda_context *cctx = (struct hmll_io_uring_cuda_context *)cqe->user_data;
        void *to   = (char *)dst->ptr + (cctx->offset - offset);
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
 * Generic fetch loop for a single buffer.  When @p fadvise is non-zero an
 * initial POSIX_FADV_WILLNEED is queued; suppress it (pass 0) when the caller
 * has already arranged readahead or when using an O_DIRECT fd.
 */
static ssize_t hmll_io_uring_fetch_loop(
    struct hmll *ctx,
    const int iofd,
    const struct hmll_iobuf *dst,
    const size_t offset,
    const int fadvise
) {
    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;

    size_t n_inflight = 0;
    size_t b_read = 0;
    size_t b_submitted = 0;
    struct io_uring_cqe *cqes[HMLL_URING_QUEUE_DEPTH];

    if (fadvise)
        hmll_io_uring_queue_fadvise(fetcher, iofd, offset, dst->size);

    while (b_read < dst->size) {
        hmll_io_uring_reclaim_slots(fetcher, dst->device);

        /* ── submit ── */
        while (b_submitted < dst->size && n_inflight < HMLL_URING_QUEUE_DEPTH) {
            struct io_uring_sqe *sqe = NULL;
            int slot;
            if (unlikely((slot = hmll_io_uring_get_sqe(fetcher, &sqe)) < 0))
                break;

            const size_t remaining = dst->size - b_submitted;
            const size_t to_read = (remaining < HMLL_URING_BUFFER_SIZE) ? remaining : HMLL_URING_BUFFER_SIZE;
            const size_t file_offset = offset + b_submitted;

            hmll_io_uring_prep_sqe(fetcher, dst->device, sqe, (char *)dst->ptr + b_submitted, file_offset, to_read, iofd, slot);

            b_submitted += to_read;
            n_inflight++;
        }

        io_uring_submit(&fetcher->ioring);

        /* if slots are exhausted by CUDA memcpy, relieve pressure before blocking */
        if (b_submitted < dst->size)
            hmll_io_uring_cuda_relieve_pressure(fetcher);

        /* ── complete: non-blocking peek first, block only when pipeline full ── */
        unsigned count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, HMLL_URING_QUEUE_DEPTH);
        if (count == 0 && n_inflight > 0) {
            struct io_uring_cqe *cqe;
            if (unlikely(io_uring_wait_cqe(&fetcher->ioring, &cqe) < 0)) {
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                return -1;
            }
            cqes[0] = cqe;
            count = 1;
        }

        for (unsigned i = 0; i < count; i++) {
            const struct io_uring_cqe *cqe = cqes[i];
            if (unlikely(cqe->user_data == HMLL_IO_URING_FADVISE_TAG))
                continue;

            if (unlikely(cqe->res < 0)) {
                ctx->error = HMLL_SYS_ERR(-cqe->res);
                io_uring_cq_advance(&fetcher->ioring, count);
                return -1;
            }

            b_read += cqe->res;
            n_inflight--;
            hmll_io_uring_handle_completion(fetcher, cqe, dst, offset, cqe->res);
        }
        io_uring_cq_advance(&fetcher->ioring, count);
    }

    hmll_io_uring_sync(dst->device, fetcher);
    return (ssize_t)b_read;
}

#if defined(__HMLL_CUDA_ENABLED__)
/**
 * CUDA split-I/O fetch: O_DIRECT for the page-aligned core, buffered I/O for
 * the unaligned head/tail.  fadvise(WILLNEED) is issued for the edge regions
 * before reading the core, giving the kernel time to readahead.
 */
static ssize_t hmll_io_uring_fetch_cuda_split(
    struct hmll *ctx,
    const int bfd,
    const int dfd,
    const struct hmll_iobuf *dst,
    const size_t offset
) {
    const size_t end = offset + dst->size;
    const size_t aligned_start = ALIGN_UP(offset, ALIGN_PAGE);
    const size_t aligned_end   = ALIGN_DOWN(end, ALIGN_PAGE);

    const size_t head_size = aligned_start - offset;
    const size_t tail_size = end - aligned_end;
    const size_t core_size = (aligned_end > aligned_start) ? aligned_end - aligned_start : 0;

    if (core_size < ALIGN_PAGE * 2)
        return hmll_io_uring_fetch_loop(ctx, bfd, dst, offset, 1);

    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;
    ssize_t total = 0;

    if (head_size > 0) hmll_io_uring_queue_fadvise(fetcher, bfd, offset, head_size);
    if (tail_size > 0) hmll_io_uring_queue_fadvise(fetcher, bfd, aligned_end, tail_size);

    /* aligned core via O_DIRECT → staging → GPU */
    struct hmll_iobuf core_dst = {
        .size = core_size, .ptr = (char *)dst->ptr + head_size, .device = dst->device,
    };
    ssize_t n = hmll_io_uring_fetch_loop(ctx, dfd, &core_dst, aligned_start, 0);
    if (n < 0) return -1;
    total += n;

    /* head edge from buffered fd (fadvise already queued) */
    if (head_size > 0) {
        struct hmll_iobuf head_dst = {
            .size = head_size, .ptr = dst->ptr, .device = dst->device,
        };
        n = hmll_io_uring_fetch_loop(ctx, bfd, &head_dst, offset, 0);
        if (n < 0) return -1;
        total += n;
    }

    /* tail edge from buffered fd (fadvise already queued) */
    if (tail_size > 0) {
        struct hmll_iobuf tail_dst = {
            .size = tail_size, .ptr = (char *)dst->ptr + head_size + core_size, .device = dst->device,
        };
        n = hmll_io_uring_fetch_loop(ctx, bfd, &tail_dst, aligned_end, 0);
        if (n < 0) return -1;
        total += n;
    }

    return total;
}
#endif /* __HMLL_CUDA_ENABLED__ */

static ssize_t hmll_io_uring_fetch_impl(
    struct hmll *ctx,
    const unsigned iofile,
    const struct hmll_iobuf *dst,
    const size_t offset
) {
    if (hmll_check(ctx->error)) return -1;

    const unsigned bfd = hmll_io_uring_buffered_fd(iofile);

#if defined(__HMLL_CUDA_ENABLED__)
    if (hmll_device_is_cuda(dst->device)) {
        const int dfd = hmll_io_uring_direct_fd(iofile);
        return hmll_io_uring_fetch_cuda_split(ctx, bfd, dfd, dst, offset);
    }
#endif

    return hmll_io_uring_fetch_loop(ctx, bfd, dst, offset, 1);
}

/* ── fetchv CQE handler ─────────────────────────────────────────────── */

/* user_data encoding for fetchv CQEs: (bidx << 8) | slot */
#define FETCHV_BIDX_SHIFT 8
#define FETCHV_SLOT_MASK  ((uint64_t)HMLL_URING_QUEUE_DEPTH - 1)

/**
 * Handle a single fetchv CQE.  Frees the slot (CPU) or kicks off the
 * staging→GPU memcpy (CUDA).  Returns bytes read, or -1 on I/O error.
 */
static inline ssize_t hmll_io_uring_fetchv_handle_cqe(
    struct hmll *ctx,
    struct hmll_io_uring *fetcher,
    const struct io_uring_cqe *cqe,
    const struct hmll_iobuf *dsts,
    const size_t *slot_offsets,
    const int is_cuda
) {
    if (cqe->res < 0) {
        ctx->error = HMLL_SYS_ERR(-cqe->res);
        return -1;
    }

    const uint64_t data = cqe->user_data;
    const uint32_t slot = (uint32_t)(data & FETCHV_SLOT_MASK);
    const uint32_t bidx = (uint32_t)(data >> FETCHV_BIDX_SHIFT);

    if (!is_cuda) {
        HMLL_UNUSED(dsts);
        HMLL_UNUSED(slot_offsets);
        hmll_io_uring_slot_set_available(&fetcher->iobusy, slot);
    }
#if defined(__HMLL_CUDA_ENABLED__)
    else {
        struct hmll_io_uring_cuda_context *cctx = &((struct hmll_io_uring_cuda_context *)fetcher->device_ctx)[slot];
        void *to   = (char *)dsts[bidx].ptr + slot_offsets[slot];
        void *from = fetcher->iovecs[slot].iov_base;
        cudaMemcpyAsync(to, from, (size_t)cqe->res, cudaMemcpyHostToDevice, cctx->stream);
        cudaEventRecord(cctx->done, cctx->stream);
        hmll_io_uring_cuda_stream_set_memcpy(&cctx->state);
    }
#else
    (void)bidx;
#endif

    return (ssize_t)cqe->res;
}

/* ── fetchv loop ────────────────────────────────────────────────────── */

struct fetchv_buf_state {
    size_t submitted;
    size_t size;
    unsigned char fadvise_sent;
};

/**
 * Generic fetchv loop: reads all buffers through a single registered file index.
 * Used for the CPU path and as a building block for the CUDA split-I/O path.
 * When @p fadvise is 0, per-buffer POSIX_FADV_WILLNEED is suppressed (useful
 * when the caller already issued readahead or when reading via O_DIRECT).
 */
static ssize_t hmll_io_uring_fetchv_loop(
    struct hmll *ctx,
    const unsigned iofd,
    const struct hmll_iobuf *dsts,
    const size_t *offsets,
    const size_t n,
    const int fadvise
) {
    if (hmll_check(ctx->error)) return -1;
    if (unlikely(n == 0)) return 0;

    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;
    const int is_cuda = hmll_device_is_cuda(dsts[0].device);

    /* scratch: [buf_states | active_indices | slot_offsets] */
    const size_t sz_state = (sizeof(struct fetchv_buf_state) * n + _Alignof(uint32_t) - 1) & ~(_Alignof(uint32_t) - 1);
    const size_t sz_idx   = (sizeof(uint32_t) * n + _Alignof(size_t) - 1) & ~(_Alignof(size_t) - 1);
    const size_t sz_slot  = sizeof(size_t) * HMLL_URING_QUEUE_DEPTH;

    _Alignas(16) uint8_t stack_scratch[8192];
    void *scratch_to_free;
    uint8_t *scratch = hmll_scratch_alloc(
        stack_scratch, sizeof(stack_scratch), sz_state + sz_idx + sz_slot, &scratch_to_free);
    if (!scratch) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        return -1;
    }

    struct fetchv_buf_state *buf_states = (struct fetchv_buf_state *)scratch;
    uint32_t *active_indices            = (uint32_t *)(scratch + sz_state);
    size_t   *slot_offsets              = (size_t   *)(scratch + sz_state + sz_idx);

    size_t n_active = 0;
    for (size_t i = 0; i < n; i++) {
        buf_states[i] = (struct fetchv_buf_state){ .size = dsts[i].size };
        if (dsts[i].size > 0)
            active_indices[n_active++] = (uint32_t)i;
    }

    size_t n_in_flight = 0, nbytes = 0, active_cursor = 0;
    struct io_uring_cqe *cqes[HMLL_URING_QUEUE_DEPTH];

    while (n_active > 0 || n_in_flight > 0) {
        hmll_io_uring_reclaim_slots(fetcher, dsts[0].device);

        /* ── submit: round-robin across buffers ── */
        while (n_active > 0 && n_in_flight < HMLL_URING_QUEUE_DEPTH) {
            if (active_cursor >= n_active) active_cursor = 0;
            const uint32_t bidx = active_indices[active_cursor];
            struct fetchv_buf_state *st = &buf_states[bidx];

            if (!st->fadvise_sent) {
                st->fadvise_sent = 1;
                if (fadvise)
                    hmll_io_uring_queue_fadvise(fetcher, iofd, offsets[bidx], st->size);
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

#if defined(__HMLL_CUDA_ENABLED__)
            if (is_cuda)
                io_uring_prep_read_fixed(sqe, iofd, fetcher->iovecs[slot].iov_base, to_read, file_off, slot);
            else
                io_uring_prep_read(sqe, iofd, (char *)dsts[bidx].ptr + st->submitted, to_read, file_off);

#else
            io_uring_prep_read(sqe, (int)iofd, (char *)dsts[bidx].ptr + st->submitted, to_read, file_off);
#endif
            io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_sqe_set_data64(sqe, ((uint64_t)bidx << FETCHV_BIDX_SHIFT) | (uint64_t)slot);

            st->submitted += to_read;
            n_in_flight++;

            if (st->submitted >= st->size)
                active_indices[active_cursor] = active_indices[--n_active];
            else
                active_cursor++;
        }

        /* ── nothing in flight but buffers remain: relieve CUDA pressure ── */
        if (n_in_flight == 0 && n_active > 0) {
            io_uring_submit(&fetcher->ioring);
            if (is_cuda)
                hmll_io_uring_cuda_relieve_pressure(fetcher);
        }

        io_uring_submit(&fetcher->ioring);

        /* relieve pressure if submission stalled on busy slots */
        if (n_active > 0 && n_in_flight < HMLL_URING_QUEUE_DEPTH) {
            if (is_cuda) hmll_io_uring_cuda_relieve_pressure(fetcher);
        }

        /* ── complete: non-blocking peek first, block only when the pipeline full ── */
        unsigned count = io_uring_peek_batch_cqe(&fetcher->ioring, cqes, HMLL_URING_QUEUE_DEPTH);
        if (count == 0 && n_in_flight > 0) {
            struct io_uring_cqe *cqe;
            if (unlikely(io_uring_wait_cqe(&fetcher->ioring, &cqe) < 0)) {
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                goto cleanup;
            }
            cqes[0] = cqe;
            count = 1;
        }

        for (unsigned i = 0; i < count; i++) {
            if (cqes[i]->user_data == HMLL_IO_URING_FADVISE_TAG) continue;

            n_in_flight--;
            const ssize_t r = hmll_io_uring_fetchv_handle_cqe(ctx, fetcher, cqes[i], dsts, slot_offsets, is_cuda);
            if (r < 0) {
                io_uring_cq_advance(&fetcher->ioring, count);
                goto cleanup;
            }
            nbytes += (size_t)r;
        }
        io_uring_cq_advance(&fetcher->ioring, count);
    }

    hmll_io_uring_sync(dsts[0].device, fetcher);
    if (scratch_to_free) free(scratch_to_free);
    return (ssize_t)nbytes;

cleanup:
    if (scratch_to_free) free(scratch_to_free);
    return -1;
}

#if defined(__HMLL_CUDA_ENABLED__)
/**
 * Decompose buffer @p i into an aligned core and up to 2 edge fragments.
 * Issues fadvise for all edge regions on the buffered fd.  Returns the number
 * of new core/edge entries appended.
 */
static void hmll_io_uring_decompose_buf(
    struct hmll_io_uring *fetcher,
    const int bfd,
    const struct hmll_iobuf *dst,
    const size_t off,
    struct hmll_iobuf *core_dsts, size_t *core_offs, size_t *n_core,
    struct hmll_iobuf *edge_dsts, size_t *edge_offs, size_t *n_edge
) {
    const size_t end      = off + dst->size;
    const size_t a_start  = ALIGN_UP(off, ALIGN_PAGE);
    const size_t a_end    = ALIGN_DOWN(end, ALIGN_PAGE);
    const size_t head_sz  = a_start - off;
    const size_t tail_sz  = end - a_end;
    const size_t core_sz  = (a_end > a_start) ? a_end - a_start : 0;

    if (core_sz < ALIGN_PAGE * 2) {
        edge_dsts[*n_edge] = *dst;
        edge_offs[*n_edge] = off;
        (*n_edge)++;
        hmll_io_uring_queue_fadvise(fetcher, bfd, off, dst->size);
        return;
    }

    core_dsts[*n_core] = (struct hmll_iobuf){
        .size = core_sz, .ptr = (char *)dst->ptr + head_sz, .device = dst->device,
    };
    core_offs[*n_core] = a_start;
    (*n_core)++;

    if (head_sz > 0) {
        hmll_io_uring_queue_fadvise(fetcher, bfd, off, head_sz);
        edge_dsts[*n_edge] = (struct hmll_iobuf){
            .size = head_sz, .ptr = dst->ptr, .device = dst->device,
        };
        edge_offs[*n_edge] = off;
        (*n_edge)++;
    }

    if (tail_sz > 0) {
        hmll_io_uring_queue_fadvise(fetcher, bfd, a_end, tail_sz);
        edge_dsts[*n_edge] = (struct hmll_iobuf){
            .size = tail_sz, .ptr = (char *)dst->ptr + head_sz + core_sz, .device = dst->device,
        };
        edge_offs[*n_edge] = a_end;
        (*n_edge)++;
    }
}

/**
 * CUDA split-I/O fetchv: decomposes each buffer into an aligned core (O_DIRECT)
 * and unaligned head/tail edges (buffered).  fadvise(WILLNEED) is issued for all
 * edge regions before core I/O begins.
 */
static ssize_t hmll_io_uring_fetchv_cuda_split(
    struct hmll *ctx,
    const int bfd,
    const int dfd,
    const struct hmll_iobuf *dsts,
    const size_t *offsets,
    const size_t n
) {
    struct hmll_io_uring *fetcher = ctx->fetcher->backend_impl_;

    /* scratch: up to 3 sub-requests per buffer (1 core + 2 edges) */
    const size_t max_parts    = n * 3;
    const size_t scratch_need = max_parts * (sizeof(struct hmll_iobuf) + sizeof(size_t));

    _Alignas(16) uint8_t stack_scratch[16384];
    void *scratch_to_free;
    uint8_t *scratch = hmll_scratch_alloc(stack_scratch, sizeof(stack_scratch),
                                          scratch_need, &scratch_to_free);
    if (!scratch) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        return -1;
    }

    struct hmll_iobuf *core_dsts = (struct hmll_iobuf *)scratch;
    size_t *core_offs            = (size_t *)(scratch + max_parts * sizeof(struct hmll_iobuf));
    struct hmll_iobuf *edge_dsts = core_dsts + n;
    size_t *edge_offs            = core_offs + n;
    size_t n_core = 0, n_edge = 0;

    for (size_t i = 0; i < n; i++) {
        if (dsts[i].size == 0) continue;
        hmll_io_uring_decompose_buf(fetcher, bfd, &dsts[i], offsets[i],
                                    core_dsts, core_offs, &n_core,
                                    edge_dsts, edge_offs, &n_edge);
    }

    io_uring_submit(&fetcher->ioring);

    ssize_t total = 0;

    if (n_core > 0) {
        ssize_t r = hmll_io_uring_fetchv_loop(ctx, dfd, core_dsts, core_offs, n_core, 0);
        if (r < 0) goto cleanup;
        total += r;
    }

    if (n_edge > 0) {
        ssize_t r = hmll_io_uring_fetchv_loop(ctx, bfd, edge_dsts, edge_offs, n_edge, 0);
        if (r < 0) goto cleanup;
        total += r;
    }

    if (scratch_to_free) free(scratch_to_free);
    return total;

cleanup:
    if (scratch_to_free) free(scratch_to_free);
    return -1;
}
#endif /* __HMLL_CUDA_ENABLED__ */

static ssize_t hmll_io_uring_fetchv_impl(
    struct hmll *ctx,
    const unsigned iofile,
    const struct hmll_iobuf *dsts,
    const size_t *offsets,
    const size_t n
) {
    if (hmll_check(ctx->error)) return -1;
    if (unlikely(n == 0)) return 0;

    const unsigned bfd = hmll_io_uring_buffered_fd(iofile);

#if defined(__HMLL_CUDA_ENABLED__)
    if (hmll_device_is_cuda(dsts[0].device)) {
        const int dfd = hmll_io_uring_direct_fd(iofile);
        return hmll_io_uring_fetchv_cuda_split(ctx, bfd, dfd, dsts, offsets, n);
    }
#endif

    return hmll_io_uring_fetchv_loop(ctx, bfd, dsts, offsets, n, 1);
}

static struct hmll_error hmll_io_uring_queue_init(
    struct hmll *ctx,
    struct hmll_io_uring *backend,
    const struct hmll_device device
) {
    (void)ctx;
    struct io_uring_params params = {
        .sq_thread_cpu = 0,
        .flags = hmll_io_uring_get_setup_flags(),
        .sq_thread_idle = 500
    };

    if (hmll_device_is_cuda(device)) {
#if defined(__HMLL_CUDA_ENABLED__)
        cudaError_t cuda_err = cudaSetDevice(device.idx);
        if (cuda_err != cudaSuccess) {
            return HMLL_ERR(HMLL_ERR_CUDA_SET_DEVICE_FAILED);
        }

        struct hmll_io_uring_cuda_context *data = calloc(HMLL_URING_QUEUE_DEPTH, sizeof(struct hmll_io_uring_cuda_context));
        backend->device_ctx = (void *)data;

        for (int i = 0; i < (int)HMLL_URING_QUEUE_DEPTH; ++i) {
            data[i].slot = i;
            CHECK_CUDA(cudaStreamCreateWithFlags(&data[i].stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(&data[i].done, cudaEventDisableTiming));
        }

        // we get the "optimal" set of flags we would like to enable and attempt to init there
        int res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
        if (res < 0) {
            if (res == -EINVAL && (params.flags & IORING_SETUP_COOP_TASKRUN)) {
                params.flags = HMLL_IO_URING_DEFAULT_QUEUE_PARAMS;
                res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
            }
            if (res < 0) {
                return HMLL_SYS_ERR(-res);
            }
        }

        struct hmll_error err = hmll_io_uring_register_staging_buffers(ctx, backend, device);
        if (hmll_check(err)) {
            return err;
        }

        return HMLL_OK;
#else
        return HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
#endif
    } else {
        int res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
        if (res < 0) {
            if (res == -EINVAL && (params.flags & IORING_SETUP_COOP_TASKRUN)) {
                params.flags = IORING_SETUP_SQPOLL;
                res = io_uring_queue_init_params(HMLL_URING_QUEUE_DEPTH, &backend->ioring, &params);
            }
            if (res < 0) {
                return HMLL_SYS_ERR(-res);
            }
        }
        return HMLL_OK;
    }
}

struct hmll_error hmll_io_uring_init(struct hmll *ctx, const struct hmll_device device) {
    if (hmll_check(ctx->error))
        return ctx->error;

    struct hmll_io_uring *backend = calloc(1, sizeof(struct hmll_io_uring));

    ctx->error = hmll_io_uring_queue_init(ctx, backend, device);
    if (hmll_check(ctx->error))
        goto cleanup;

    const size_t n_iofiles = ctx->num_sources * 2;
    int *iofiles = calloc(n_iofiles, sizeof(int));
    for (unsigned i = 0; i < ctx->num_sources; ++i) {
        iofiles[hmll_io_uring_buffered_fd(i)] = ctx->sources[i].fd;
        const int dfd = ctx->sources[i].d_fd;
        iofiles[hmll_io_uring_direct_fd(i)] = (dfd > 0) ? dfd : ctx->sources[i].fd;
    }

    const int res = io_uring_register_files(&backend->ioring, iofiles, n_iofiles);
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
