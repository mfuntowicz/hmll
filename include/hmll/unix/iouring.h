#ifndef HMLL_FETCHER_IOURING_H
#define HMLL_FETCHER_IOURING_H

// The queue-depth matchs the number of slot (bits) allocable through iobusy var
#define HMLL_URING_QUEUE_DEPTH sizeof(long long) * 8
#define HMLL_URING_BUFFER_SIZE (128U * 1024)

#define ALIGNMENT 4096U
#define PAGE_ALIGNED_UP(x) (((x) + ALIGNMENT - 1) & ~(ALIGNMENT - 1))
#define PAGE_ALIGNED_DOWN(x) ((x) & ~(ALIGNMENT - 1))

#include <hmll/types.h>
#include <hmll/fetcher.h>
#include <liburing.h>

struct hmll_io_uring_user_payload
{
    // Positive is from the start, negative is from the end
    ssize_t discard;
    unsigned int slot;
};
typedef struct hmll_io_uring_user_payload hmll_io_uring_user_payload_t;

struct hmll_fetcher_io_uring {
    struct io_uring ioring;
    long long iobusy;
};
typedef struct hmll_fetcher_io_uring hmll_fetcher_io_uring_t;


enum hmll_error_code hmll_io_uring_init(struct hmll_context *, struct hmll_fetcher *, enum hmll_device);
struct hmll_fetch_range hmll_io_uring_fetch_range(struct hmll_context *, struct hmll_fetcher_io_uring *, struct hmll_range, struct hmll_device_buffer);
int hmll_io_uring_slot_available(long);

#endif // HMLL_FETCHER_IOURING_H
