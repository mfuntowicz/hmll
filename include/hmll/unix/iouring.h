#ifndef HMLL_FETCHER_IOURING_H
#define HMLL_FETCHER_IOURING_H

#define HMLL_URING_NUM_IOVECS 64U
#define HMLL_URING_BUFFER_SIZE (256U * 1024U)

#include <hmll/types.h>
#include <hmll/fetcher.h>
#include <liburing.h>

struct hmll_fetcher_io_uring {
    struct io_uring ioring;
    struct iovec iovs[HMLL_URING_NUM_IOVECS];
    // struct hmll_fetcher_io_uring_payload iopylds[HMLL_URING_NUM_IOVECS];
    long iobusy;
};
typedef struct hmll_fetcher_io_uring hmll_fetcher_io_uring_t;


enum hmll_error_code hmll_io_uring_init(struct hmll_context *, struct hmll_fetcher *, enum hmll_device);
size_t hmll_io_uring_fetch_range(struct hmll_context *, struct hmll_fetcher_io_uring *, struct hmll_range, const struct hmll_device_buffer *);
int hmll_io_uring_slot_available(long);

#endif // HMLL_FETCHER_IOURING_H
