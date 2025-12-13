#ifndef HMLL_FETCHER_UNIX_IO_URING_H
#define HMLL_FETCHER_UNIX_IO_URING_H
#include <stdio.h>
#include <liburing.h>

#include "hmll/fetcher.h"
#include "hmll/types.h"

#define HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS 128U
#define HMLL_IO_URING_DEFAULT_BUFFER_SIZE (256U * 1024U)

// Alignment for O_DIRECT reads (must match page size, typically 4096 bytes)
#define ALIGNMENT 4096U
#define PAGE_ALIGNED_UP(x) (((x) + ALIGNMENT - 1) & ~(ALIGNMENT - 1))
#define PAGE_ALIGNED_DOWN(x) ((x) & ~(ALIGNMENT - 1))

typedef struct hmll_fetcher_io_uring_buffer hmll_fetcher_io_uring_buffer_t;


struct hmll_fetcher_io_uring_payload
{
    void* ptr;
    size_t size;
    size_t discard;  // Number of bytes to skip at the beginning (for O_DIRECT alignment)
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


hmll_fetcher_io_uring_t hmll_fetcher_io_uring_init(struct hmll_context *);
// void hmll_fetcher_io_uring_free(hmll_fetcher_io_uring_t *);
enum hmll_error_code hmll_fetcher_io_uring_fetch(
    struct hmll_context *, struct hmll_fetcher_io_uring *, const char* name, const struct hmll_device_buffer *);
enum hmll_error_code hmll_fetcher_io_uring_fetch_range(
    struct hmll_context *, struct hmll_fetcher_io_uring *, struct hmll_fetch_range, const struct hmll_device_buffer *);

#endif // HMLL_FETCHER_UNIX_IO_URING_H
