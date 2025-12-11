#include "hmll/unix/fetcher_io_uring.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "hmll/hmll.h"

hmll_fetcher_io_uring_t hmll_fetcher_io_uring_init(struct hmll_context *ctx)
{
    hmll_fetcher_io_uring_t fetcher = {0};

    if (hmll_has_error(ctx))
        return fetcher;

    const size_t nblks = HMLL_IO_URING_DEFAULT_BUFFER_SIZE / HMLL_IO_URING_DEFAULT_READ_SIZE * HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS;

    // Making a queue with a depth which can hold all our block of memory-mapped buffers ... Is it correct?
    int status;
    if ((status = io_uring_queue_init(nblks, &fetcher.ioring, IORING_SETUP_SQPOLL)) < 0) {
        // return (hmll_status_t) {HMLL_ALLOCATION_FAILED, "Failed to initialize io_uring ring"};
    }

    void* arena = hmll_get_io_buffer(ctx, HMLL_DEVICE_CPU, HMLL_IO_URING_DEFAULT_BUFFER_SIZE * HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS);
    for (unsigned int i = 0; i < HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS; i++) {
        fetcher.iovs[i].iov_base = (char *)arena + HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS * i;
        fetcher.iovs[i].iov_len = HMLL_IO_URING_DEFAULT_BUFFER_SIZE;
    }

    status = io_uring_register_buffers(&fetcher.ioring, fetcher.iovs, HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS);
    return fetcher;
}

enum hmll_error_code hmll_fetcher_io_uring_fetch(
    struct hmll_context *ctx,
    struct hmll_fetcher_io_uring *fetcher,
    const char* name,
    const struct hmll_device_buffer *dst)
{
    if (hmll_has_error(ctx))
        return ctx->error;

#ifdef DEBUG
    printf("[DEBUG] fetching tensor %s\n", name);
#endif

    hmll_tensor_specs_t specs = hmll_get_tensor_specs(ctx, name);

    if (hmll_has_error(ctx))
        return ctx->error;

    const hmll_fetch_range_t range = (hmll_fetch_range_t){specs.start, specs.end};
    return hmll_fetcher_io_uring_fetch_range(ctx, fetcher, range, dst);
}

int32_t hmll_fetcher_io_uring_get_slot(const hmll_fetcher_io_uring_t *fetcher)
{
    for (size_t i = 0; i < HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS; ++i) {
        if (fetcher->iobusy[i] == 0) return (int32_t)i;
    }

    return -1;
}

void hmll_fetcher_io_uring_prepare_payload(hmll_fetcher_io_uring_t *fetcher, const int32_t slot, const size_t bytes_to_read, void *ptr)
{
    fetcher->iopylds[slot].buffer = slot;
    fetcher->iopylds[slot].size = bytes_to_read;
    fetcher->iopylds[slot].ptr = ptr; // buffer is 0-indexed while the file starts at range.start
    fetcher->iobusy[slot] = 1;
}

enum hmll_error_code hmll_fetcher_io_uring_fetch_range(
    hmll_context_t *ctx,
    hmll_fetcher_io_uring_t *fetcher,
    const hmll_fetch_range_t range,
    const hmll_device_buffer_t *dst
)
{
    const size_t size = range.end - range.start;
    const size_t nchunks = (size + HMLL_IO_URING_DEFAULT_BUFFER_SIZE - 1) / HMLL_IO_URING_DEFAULT_BUFFER_SIZE;

    if (dst->size < size) {
        ctx->error = HMLL_ERR_BUFFER_TOO_SMALL;
        return ctx->error;
    }

#ifdef DEBUG
    printf("[DEBUG] fetching [%lu -> %lu] size=%lu, chunks=%lu\n", range.start, range.end, size, nchunks);
#endif

    size_t chunks_submitted = 0;
    size_t chunks_completed = 0;
    size_t file_offset = range.start;

    // 1: Fill the pipeline -> submit up to queue depth operations
    while (chunks_submitted < nchunks && chunks_submitted < HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS) {
        // Find a free buffer
        const int32_t slot = hmll_fetcher_io_uring_get_slot(fetcher);
        if (slot == -1) break; // No free buffers

        // Calculate bytes to read (handle last chunk)
        size_t bytes_to_read = HMLL_IO_URING_DEFAULT_BUFFER_SIZE;
        if (file_offset + bytes_to_read > range.end) {
            bytes_to_read = range.end - file_offset;
        }

        // Write back in the buffer will happen in the `dst->ptr` which is 0-indexed whereas our read is not, account for it
        void* write_at = (char *)dst->ptr + (file_offset - range.start);
        hmll_fetcher_io_uring_prepare_payload(fetcher, slot, bytes_to_read, write_at);

        // Submit read operation
        struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
        io_uring_prep_read_fixed(sqe, ctx->source.fd, fetcher->iovs[slot].iov_base, bytes_to_read, file_offset, slot);
        io_uring_sqe_set_data(sqe, fetcher->iopylds + slot);

        file_offset += bytes_to_read;
        chunks_submitted++;

#ifdef DEBUG
        printf("[DEBUG] submitted chunk %lu: buffer=%d, offset=%lu, size=%lu\n",
               chunks_submitted, slot, file_offset - bytes_to_read, bytes_to_read);
#endif
    }

    // Submit all queued operations
    io_uring_submit(&fetcher->ioring);

    // 2. Process completions and submit new reads
    while (chunks_completed < nchunks) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&fetcher->ioring, &cqe) < 0){
            ctx->error = HMLL_ERR_IO_ERROR;
            return ctx->error;
        }

        // Check if read succeeded
        if (cqe->res < 0) {
#ifdef DEBUG
            printf("[ERROR] Read failed: %s\n", strerror(-cqe->res));
#endif
            io_uring_cqe_seen(&fetcher->ioring, cqe);

            ctx->error = HMLL_ERR_IO_ERROR;
            return ctx->error;
        }

        // Get the buffer that just completed
        const hmll_fetcher_io_uring_payload_t *payload = io_uring_cqe_get_data(cqe);

#ifdef DEBUG
        printf("[DEBUG] completed chunk %lu: buffer=%i, bytes=%d\n", chunks_completed + 1, payload->buffer, cqe->res);
#endif

        // TODO: Handle the memcpy somewhere else (i.e. another thread?)
        memcpy(payload->ptr, fetcher->iovs[payload->buffer].iov_base, payload->size);

        io_uring_cqe_seen(&fetcher->ioring, cqe);
        chunks_completed++;

        // Reuse this buffer for the next chunk if more data to read
        if (chunks_submitted < nchunks) {
            size_t bytes_to_read = HMLL_IO_URING_DEFAULT_BUFFER_SIZE;
            if (file_offset + bytes_to_read > range.end) {
                bytes_to_read = range.end - file_offset;
            }

            struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
            io_uring_prep_read_fixed(
                sqe, ctx->source.fd,
                fetcher->iovs[payload->buffer].iov_base,
                bytes_to_read,
                file_offset,
                payload->buffer);

            void* write_at = (char *)dst->ptr + (file_offset - range.start);
            hmll_fetcher_io_uring_prepare_payload(fetcher, payload->buffer, bytes_to_read, write_at);
            io_uring_sqe_set_data(sqe, fetcher->iopylds + payload->buffer);

            file_offset += bytes_to_read;
            chunks_submitted++;

            // bufferize the submission to avoid starving the kernel with syscalls
            io_uring_submit(&fetcher->ioring);

#ifdef DEBUG
            printf("[DEBUG] resubmitted buffer %d for chunk %lu\n", payload->buffer, chunks_submitted);
#endif
        } else {
            // No more chunks to read, mark buffer as free
            fetcher->iobusy[payload->buffer] = 0;
        }
    }

#ifdef DEBUG
    printf("[DEBUG] fetch complete: %lu chunks read\n", chunks_completed);
#endif

    return HMLL_ERR_SUCCESS;
}