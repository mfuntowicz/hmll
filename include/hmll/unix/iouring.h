#ifndef HMLL_FETCHER_IOURING_H
#define HMLL_FETCHER_IOURING_H

// The queue-depth matchs the number of slot (bits) allocable through iobusy var
#define HMLL_URING_QUEUE_DEPTH sizeof(long long) * 8
#define HMLL_URING_BUFFER_SIZE (128U * 1024)

#include <liburing.h>
#include "hmll/fetcher.h"
#include "hmll/types.h"

static inline int hmll_io_uring_slot_find_available(const long long mask)
{
    const int pos = __builtin_ffsll(~mask);
    return pos == 0 ? -1 : pos - 1;
}

static inline void hmll_io_uring_slot_set_busy(long long *mask, const unsigned int slot)
{
    *mask |= 1LL << slot;
}

static inline void hmll_io_uring_slot_set_available(long long *mask, const unsigned int slot)
{
    *mask &= ~(1LL << slot);
}

struct hmll_fetcher_io_uring {
    struct io_uring ioring;
    long long iobusy;
};
typedef struct hmll_fetcher_io_uring hmll_fetcher_io_uring_t;


enum hmll_error_code hmll_io_uring_init(struct hmll_context *, struct hmll_fetcher *, enum hmll_device);
struct hmll_fetch_range hmll_io_uring_fetch_range(struct hmll_context *, struct hmll_fetcher_io_uring *, struct hmll_range, struct hmll_device_buffer);
int hmll_io_uring_slot_available(long);

#endif // HMLL_FETCHER_IOURING_H
