#ifndef HMLL_FETCHER_IOURING_H
#define HMLL_FETCHER_IOURING_H

// The queue-depth matchs the number of slot (bits) allocable through iobusy var
#ifndef HMLL_URING_QUEUE_DEPTH
#define HMLL_URING_QUEUE_DEPTH 128U
#endif

#ifndef HMLL_URING_BUFFER_SIZE
#define HMLL_URING_BUFFER_SIZE (8U * 1024 * 1024)
#endif

#ifndef HMLL_URING_CQE_BATCH_SIZE
#define HMLL_URING_CQE_BATCH_SIZE 16
#endif

#include <liburing.h>
#include "hmll/types.h"

struct hmll_iouring_iobusy
{
    long long msb;
    long long lsb;
};

struct hmll_iouring_cca
{
    unsigned throughput;
    unsigned window;
};

#if defined(__HMLL_CUDA_ENABLED__)
#include <driver_types.h>

enum hmll_iouring_cuda_state {
    HMLL_CUDA_STREAM_IDLE = 0,
    HMLL_CUDA_STREAM_MEMCPY = 1,
};

struct hmll_io_uring_cuda_context
{
    cudaStream_t stream;
    cudaEvent_t done;
    size_t offset;
    int slot;
    enum hmll_iouring_cuda_state state;
};

static inline void hmll_io_uring_cuda_stream_set_idle(enum hmll_iouring_cuda_state *state)
{
    *state = HMLL_CUDA_STREAM_IDLE;
}

static inline void hmll_io_uring_cuda_stream_set_memcpy(enum hmll_iouring_cuda_state *state)
{
    *state = HMLL_CUDA_STREAM_MEMCPY;
}

#endif


static inline size_t hmll_iouring_throughput(const size_t nbytes, const size_t elapsed)
{
    return nbytes * 1000000L / elapsed;
}

static inline void hmll_io_uring_cca_init(struct hmll_iouring_cca *cca)
{
    cca->throughput = 0;
    cca->window = 1;
}

static inline unsigned hmll_io_uring_cca_update(
    struct hmll_iouring_cca *cca, const size_t bytes, const struct timespec ts_start, const struct timespec ts_end)
{
    const unsigned current = cca->window;

    const size_t elapsed = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000L + (ts_end.tv_nsec - ts_start.tv_nsec);
    const size_t throughput = hmll_iouring_throughput(bytes, elapsed);
    const size_t smoothed = ((throughput * 3) + cca->throughput) >> 2;

    if (cca->throughput < throughput)
        cca->window = cca->window + 1 >= HMLL_URING_CQE_BATCH_SIZE ? HMLL_URING_CQE_BATCH_SIZE : cca->window + 1;
    else
        cca->window = cca->window - 1 < 1 ? 1 : cca->window - 1;

    cca->throughput = smoothed;
    return current;
}

struct hmll_io_uring {
    struct io_uring ioring;
    struct iovec *iovecs;
    struct hmll_iouring_iobusy iobusy;
    struct hmll_iouring_cca iocca;  // congestion control

    // store optional device data
    void *device_ctx;
};

static inline unsigned int hmll_io_uring_slot_is_busy(const struct hmll_iouring_iobusy iobusy, const unsigned int slot)
{
    if (slot < 64)
        return iobusy.lsb & (1LL << slot);
    return iobusy.msb & (1LL << (slot - 64));
}

static inline int hmll_io_uring_slot_find_available(const struct hmll_iouring_iobusy iobusy)
{
    // First check LSB
    const int pos_lsb = __builtin_ffsll(~iobusy.lsb);
    if (pos_lsb > 0)
        return pos_lsb - 1;

    // Then check MSB
    const int pos_msb = __builtin_ffsll(~iobusy.msb);
    if (pos_msb > 0)
        return 64 + pos_msb - 1;

    return -1;
}

static inline void hmll_io_uring_slot_set_busy(struct hmll_iouring_iobusy *iobusy, const unsigned int slot)
{
    if (slot < 64) {
        iobusy->lsb |= 1LL << slot;
    } else {
        iobusy->msb |= 1LL << (slot - 64);
    }
}

static inline void hmll_io_uring_slot_set_available(struct hmll_iouring_iobusy *iobusy, const unsigned int slot)
{
    if (slot < 64) {
        iobusy->lsb &= ~(1LL << slot);
    } else {
        iobusy->msb &= ~(1LL << (slot - 64));
    }
}

struct hmll_error hmll_io_uring_init(struct hmll *, enum hmll_device);
#endif // HMLL_FETCHER_IOURING_H
