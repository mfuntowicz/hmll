#include "hmll/unix/fetcher_io_uring.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "hmll/hmll.h"

hmll_fetcher_io_uring_t hmll_fetcher_io_uring_init(struct hmll_context *ctx)
{
    hmll_fetcher_io_uring_t fetcher = {0};

    if (hmll_has_error(hmll_get_error(ctx)))
        return fetcher;

    struct io_uring_params params = {0};
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 250;

    int iofiles[1];
    iofiles[0] = ctx->source.fd;

    io_uring_queue_init_params(HMLL_URING_NUM_IOVECS, &fetcher.ioring, &params);
    io_uring_register_files(&fetcher.ioring, iofiles, 1);

    // Allocate aligned buffer arena for O_DIRECT I/O (mmap returns page-aligned memory)
    void* arena = hmll_get_io_buffer(ctx, HMLL_DEVICE_CPU, HMLL_URING_BUFFER_SIZE * HMLL_URING_NUM_IOVECS);
    for (unsigned int i = 0; i < HMLL_URING_NUM_IOVECS; i++) {
        fetcher.iovs[i].iov_base = (char *)arena + HMLL_URING_BUFFER_SIZE * i;
        fetcher.iovs[i].iov_len = HMLL_URING_BUFFER_SIZE;
    }

    io_uring_register_buffers(&fetcher.ioring, fetcher.iovs, HMLL_URING_NUM_IOVECS);
    return fetcher;
}

enum hmll_error_code hmll_fetcher_io_uring_fetch(
    struct hmll_context *ctx,
    struct hmll_fetcher_io_uring *fetcher,
    const char* name,
    const struct hmll_device_buffer *dst)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return ctx->error;

#ifdef DEBUG
    printf("[DEBUG] fetching tensor %s\n", name);
#endif

    hmll_tensor_specs_t specs = hmll_get_tensor_specs(ctx, name);
    if (hmll_has_error(hmll_get_error(ctx)))
        return ctx->error;

    const hmll_fetch_range_t range = (hmll_fetch_range_t){specs.start, specs.end};
    return hmll_fetcher_io_uring_fetch_range(ctx, fetcher, range, dst);
}

int32_t hmll_fetcher_io_uring_get_slot(const hmll_fetcher_io_uring_t *fetcher)
{
    for (size_t i = 0; i < HMLL_URING_NUM_IOVECS; ++i) {
        if (fetcher->iobusy[i] == 0) return (int32_t)i;
    }
    return -1;
}

void hmll_fetcher_io_uring_prepare_payload(hmll_fetcher_io_uring_t *fetcher, const int32_t slot, const size_t bytes_to_read, const size_t discard, void *ptr)
{
    // TODO: discard - we need to account for the discardable byte on the end of the tensor-boundaries to be page-aligned
    fetcher->iopylds[slot].buffer = slot;
    fetcher->iopylds[slot].size = bytes_to_read;
    fetcher->iopylds[slot].discard = discard;
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

    if (dst->size < size) {
        ctx->error = HMLL_ERR_BUFFER_TOO_SMALL;
        return ctx->error;
    }

    size_t pages_requested = 0;
    size_t pages_completed = 0;
    size_t file_offset = range.start;

    // 1: Fill the pipeline -> submit up to queue depth operations
    while (file_offset < range.end && pages_requested < HMLL_URING_NUM_IOVECS) {
        // Find a free buffer
        const int32_t slot = hmll_fetcher_io_uring_get_slot(fetcher);
        if (slot == -1) break; // No free buffers

        // For O_DIRECT: calculate aligned offset and size for THIS chunk
        size_t chunk_aligned_offset = PAGE_ALIGNED_DOWN(file_offset);
        size_t chunk_discard = file_offset - chunk_aligned_offset;

        // Calculate bytes to read (account for discard to not overflow buffer)
        size_t bytes_to_read = HMLL_URING_BUFFER_SIZE - chunk_discard;
        if (file_offset + bytes_to_read > range.end) {
            bytes_to_read = range.end - file_offset;
        }

        // Calculate aligned size for this chunk
        size_t chunk_aligned_size = PAGE_ALIGNED_UP(bytes_to_read + chunk_discard);

        // Write back in the buffer will happen in the `dst->ptr` which is 0-indexed whereas our read is not, account for it
        void* write_at = (char *)dst->ptr + (file_offset - range.start);
        hmll_fetcher_io_uring_prepare_payload(fetcher, slot, bytes_to_read, chunk_discard, write_at);

        // Submit read operation with aligned offset and size
        struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
        sqe->flags = IOSQE_FIXED_FILE;
        io_uring_prep_read_fixed(sqe, 0, fetcher->iovs[slot].iov_base, chunk_aligned_size, chunk_aligned_offset, slot);
        io_uring_sqe_set_data(sqe, fetcher->iopylds + slot);

        file_offset += bytes_to_read;
        ++pages_requested;
#ifdef DEBUG
        printf("[DEBUG] submitted chunk %zu: buffer=%d, file_offset_before=%zu, disk_offset=%zu, disk_size=%zu, discard=%zu, useful=%zu, file_offset_after=%zu\n",
               pages_requested, slot, file_offset - bytes_to_read, chunk_aligned_offset, chunk_aligned_size, chunk_discard, bytes_to_read, file_offset);
#endif
    }

    // Submit all batched read operations
    io_uring_submit(&fetcher->ioring);

    // 2. Process completions and submit new reads
    unsigned int resubmitted = 0;  // Track how many requests we've queued for batch submission
    while (pages_completed < pages_requested) {
        struct io_uring_cqe *cqe;

        // Wait for at least one completion
        if (io_uring_wait_cqe(&fetcher->ioring, &cqe) < 0){
            ctx->error = HMLL_ERR_IO_ERROR;
            return ctx->error;
        }

        // Process this CQE and drain any other available CQEs (batch processing)
        do {
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

            // Verify we read enough data
            size_t bytes_read = (size_t)cqe->res;
            size_t bytes_needed = payload->discard + payload->size;

#ifdef DEBUG
            printf("[DEBUG] completed chunk %lu: buffer=%i, bytes_read=%zu, discard=%lu, useful=%lu, needed=%zu\n",
                   pages_completed + 1, payload->buffer, bytes_read, payload->discard, payload->size, bytes_needed);
#endif

            if (bytes_read < bytes_needed) {
#ifdef DEBUG
                printf("[ERROR] Short read: got %zu bytes, needed %zu (discard=%lu + useful=%lu)\n",
                       bytes_read, bytes_needed, payload->discard, payload->size);
#endif
                io_uring_cqe_seen(&fetcher->ioring, cqe);
                ctx->error = HMLL_ERR_IO_ERROR;
                return ctx->error;
            }

            // TODO: Handle the memcpy somewhere else (i.e. another thread?)
            // Skip the discard bytes at the beginning (for O_DIRECT alignment)
            // const char *src = (const char *)fetcher->iovs[payload->buffer].iov_base + payload->discard;
            // memcpy(payload->ptr, src, payload->size);

            io_uring_cqe_seen(&fetcher->ioring, cqe);
            ++pages_completed;

            // Reuse this buffer for the next chunk if more data to read
            if (file_offset < range.end) {
                // For O_DIRECT: calculate aligned offset and size for THIS chunk
                size_t chunk_aligned_offset = PAGE_ALIGNED_DOWN(file_offset);
                size_t chunk_discard = file_offset - chunk_aligned_offset;

                // Calculate bytes to read (account for discard to not overflow buffer)
                size_t bytes_to_read = HMLL_URING_BUFFER_SIZE - chunk_discard;
                if (file_offset + bytes_to_read > range.end) {
                    bytes_to_read = range.end - file_offset;
                }

                // Calculate aligned size for this chunk
                size_t chunk_aligned_size = PAGE_ALIGNED_UP(bytes_to_read + chunk_discard);

                void* write_at = (char *)dst->ptr + (file_offset - range.start);
                hmll_fetcher_io_uring_prepare_payload(fetcher, payload->buffer, bytes_to_read, chunk_discard, write_at);

                struct io_uring_sqe *sqe = io_uring_get_sqe(&fetcher->ioring);
                sqe->flags = IOSQE_FIXED_FILE;
                io_uring_prep_read_fixed(
                    sqe, 0,
                    fetcher->iovs[payload->buffer].iov_base,
                    chunk_aligned_size,
                    chunk_aligned_offset,
                    payload->buffer);
                io_uring_sqe_set_data(sqe, fetcher->iopylds + payload->buffer);

                file_offset += bytes_to_read;
                ++pages_requested;
                ++resubmitted;

#ifdef DEBUG
                printf("[DEBUG] resubmitted chunk %zu: buffer=%d, file_offset_before=%zu, disk_offset=%zu, disk_size=%zu, discard=%zu, useful=%zu, file_offset_after=%zu\n",
                       pages_requested, payload->buffer, file_offset - bytes_to_read, chunk_aligned_offset, chunk_aligned_size, chunk_discard, bytes_to_read, file_offset);
#endif
            } else {
                // No more chunks to read, mark buffer as free
                fetcher->iobusy[payload->buffer] = 0;
            }

            // Try to get another CQE without waiting (batch processing)
        } while (io_uring_peek_cqe(&fetcher->ioring, &cqe) == 0);

        // Submit all resubmitted requests in one batch
        if (resubmitted > 0) {
#ifdef DEBUG
            printf("[DEBUG] batch submitting %u requests\n", resubmitted);
#endif
            io_uring_submit(&fetcher->ioring);
            resubmitted = 0;
        }
    }

#ifdef DEBUG
    printf("[DEBUG] fetch complete: requested range=[%zu, %zu), size=%zu, chunks=%zu, final_file_offset=%zu\n",
           range.start, range.end, size, pages_completed, file_offset);
    if (file_offset < range.end) {
        printf("[WARNING] Did not read entire range! Missing bytes: %zu\n", range.end - file_offset);
    }
#endif

    return HMLL_ERR_SUCCESS;
}
