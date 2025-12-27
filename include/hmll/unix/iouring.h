#ifndef HMLL_FETCHER_IOURING_H
#define HMLL_FETCHER_IOURING_H

// The queue-depth matchs the number of slot (bits) allocable through iobusy var
#define HMLL_URING_QUEUE_DEPTH 128U
#define HMLL_URING_BUFFER_SIZE (64U * 1024)

#include <liburing.h>
#include "hmll/fetcher.h"
#include "hmll/types.h"

#if defined(__HMLL_CUDA_ENABLED__)
#include <driver_types.h>

enum hmll_iouring_cuda_state {
    HMLL_CUDA_STREAM_IDLE = 0,
    HMLL_CUDA_STREAM_MEMCPY = 1,
};

struct hmll_iouring_cuda_context
{
    cudaStream_t stream;
    cudaEvent_t done;
    size_t offset;
    int slot;
    enum hmll_iouring_cuda_state state;
};

static inline void hmll_iouring_cuda_stream_set_idle(enum hmll_iouring_cuda_state *state)
{
    *state = HMLL_CUDA_STREAM_IDLE;
}

static inline void hmll_iouring_cuda_stream_set_memcpy(enum hmll_iouring_cuda_state *state)
{
    *state = HMLL_CUDA_STREAM_MEMCPY;
}

#endif

struct hmll_iouring_iobusy
{
    long long msb;
    long long lsb;
};

struct hmll_iouring {
    struct io_uring ioring;
    struct iovec *iovecs;
    struct hmll_iouring_iobusy iobusy;

    // Store optional device data
    void *device_ctx;
};

static inline unsigned int hmll_iouring_slot_is_busy(const struct hmll_iouring_iobusy iobusy, const unsigned int slot)
{
    if (slot < 64)
        return iobusy.lsb & (1LL << slot);
    return iobusy.msb & (1LL << (slot - 64));
}

static inline int hmll_iouring_slot_find_available(const struct hmll_iouring_iobusy iobusy)
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

static inline void hmll_iouring_slot_set_busy(struct hmll_iouring_iobusy *iobusy, const unsigned int slot)
{
    if (slot < 64) {
        iobusy->lsb |= 1LL << slot;
    } else {
        iobusy->msb |= 1LL << (slot - 64);
    }
}

static inline void hmll_iouring_slot_set_available(struct hmll_iouring_iobusy *iobusy, const unsigned int slot)
{
    if (slot < 64) {
        iobusy->lsb &= ~(1LL << slot);
    } else {
        iobusy->msb &= ~(1LL << (slot - 64));
    }
}

enum hmll_error_code hmll_iouring_init(struct hmll_context *, struct hmll_fetcher *, enum hmll_device);
#endif // HMLL_FETCHER_IOURING_H
