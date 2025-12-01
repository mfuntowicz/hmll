#ifndef HMLL_FETCHER_UNIX_IO_URING_H
#define HMLL_FETCHER_UNIX_IO_URING_H

#include <liburing.h>

#include "hmll/status.h"
#include "hmll/types.h"

struct hmll_fetcher_io_uring {
    struct io_uring ring;
    struct io_uring_sq *sqe;
    struct io_uring_sq *cqe;
    struct iovec *io_vectors;
    size_t queue_depth;
};
typedef struct hmll_fetcher_io_uring hmll_fetcher_io_uring_t;

#endif // HMLL_FETCHER_UNIX_IO_URING_H
