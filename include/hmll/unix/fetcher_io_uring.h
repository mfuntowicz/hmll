#ifndef HMLL_FETCHER_UNIX_IO_URING_H
#define HMLL_FETCHER_UNIX_IO_URING_H
#include <stdio.h>
#include <liburing.h>

#include "hmll/fetcher.h"
#include "hmll/status.h"
#include "hmll/types.h"

#define HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS 128U
#define HMLL_IO_URING_DEFAULT_BUFFER_SIZE (512U * 1024U)
#define HMLL_IO_URING_DEFAULT_READ_SIZE (512U * 1024U)

#define ROUND_UP_ALIGN(x) (((x) + ALIGNMENT - 1) & ~(ALIGNMENT - 1))

typedef struct hmll_fetcher_io_uring_buffer hmll_fetcher_io_uring_buffer_t;


struct hmll_fetcher_io_uring_payload
{
    void* ptr;
    size_t size;
    int32_t buffer;
};
typedef struct hmll_fetcher_io_uring_payload hmll_fetcher_io_uring_payload_t;


struct hmll_fetcher_io_uring {
    struct io_uring ioring;
    struct iovec iovs[HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS];
    struct hmll_fetcher_io_uring_payload iopylds[HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS];
    unsigned char iobusy[HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS];
};
typedef struct hmll_fetcher_io_uring hmll_fetcher_io_uring_t;


hmll_fetcher_io_uring_t hmll_fetcher_io_uring_init(void);
// void hmll_fetcher_io_uring_free(hmll_fetcher_io_uring_t *);
hmll_status_t hmll_fetcher_io_uring_fetch(
    const hmll_context_t *, hmll_fetcher_io_uring_t *, const char*, const hmll_device_buffer_t *);
hmll_status_t hmll_fetcher_io_uring_fetch_range(
    const hmll_context_t *, hmll_fetcher_io_uring_t *, hmll_fetch_range_t, const hmll_device_buffer_t *);

#endif // HMLL_FETCHER_UNIX_IO_URING_H
