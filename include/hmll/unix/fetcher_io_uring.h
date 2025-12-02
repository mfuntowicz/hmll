#ifndef HMLL_FETCHER_UNIX_IO_URING_H
#define HMLL_FETCHER_UNIX_IO_URING_H
#include <stdio.h>
#include <liburing.h>
#include <limits.h>

#include "hmll/status.h"
#include "hmll/types.h"

#define HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS 128
#define HMLL_IO_URING_DEFAULT_BUFFER_SIZE (4 * 1024)
#define HMLL_IO_URING_DEFAULT_READ_SIZE (4 * 1024)


typedef struct hmll_fetcher_io_uring_buffer hmll_fetcher_io_uring_buffer_t;

struct hmll_fetcher_io_uring {
    struct io_uring *ioring;
    struct iovec *iovs;
    bool *iobusy;
    size_t ioqd;
};
typedef struct hmll_fetcher_io_uring hmll_fetcher_io_uring_t;


hmll_status_t hmll_fetcher_io_uring_init(hmll_fetcher_io_uring_t **, size_t);
void hmll_fetcher_io_uring_free(hmll_fetcher_io_uring_t *);
hmll_status_t hmll_fetcher_io_uring_fetch(const hmll_context_t *, const hmll_fetcher_io_uring_t *, const char*);

#endif // HMLL_FETCHER_UNIX_IO_URING_H
