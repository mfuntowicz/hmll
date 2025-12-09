#include "hmll/unix/fetcher_io_uring.h"

#include <stdlib.h>
#include <string.h>

#include "hmll/hmll.h"

hmll_status_t hmll_fetcher_io_uring_init(hmll_fetcher_io_uring_t **fetcher, const size_t buffer_size)
{
    if ((*fetcher = malloc(sizeof(hmll_fetcher_io_uring_t))) == nullptr)
        return (hmll_status_t) {HMLL_ALLOCATION_FAILED, "Failed to allocate io_uring fetcher"};

    const size_t buffer_size_ = buffer_size == 0 ? HMLL_IO_URING_DEFAULT_BUFFER_SIZE : buffer_size;
    const size_t nblks = buffer_size_ / HMLL_IO_URING_DEFAULT_READ_SIZE * HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS;

    // Making a queue with a depth which can hold all our block of memory-mapped buffers ... Is it correct?
    int status;
    auto const ring = malloc(sizeof(struct io_uring));
    if ((status = io_uring_queue_init(nblks, ring, IORING_SETUP_SQPOLL)) < 0) {
        free(ring);
        return (hmll_status_t) {HMLL_ALLOCATION_FAILED, "Failed to initialize io_uring ring"};
    }

    // Allocate iovectors for fixed buffer copy - it avoids the kernel to map/unmap these buffers at every call
    // TODO: Check allocations
    (*fetcher)->iovs = malloc(HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS * sizeof(struct iovec));
    (*fetcher)->iopylds = malloc(HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS * sizeof(struct hmll_fetcher_io_uring_payload));
    (*fetcher)->iobusy = calloc(nblks, sizeof(bool));

    for (int i = 0; i < HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS; i++) {
        // Must be page-aligned for io_uring registered buffers
        auto const alloc_status = hmll_get_io_buffer(HMLL_DEVICE_CPU, &(*fetcher)->iovs[i].iov_base, buffer_size_);
        if (!hmll_success(alloc_status)) {
            free(ring);
            return alloc_status;
        }

        (*fetcher)->iovs[i].iov_len = buffer_size_;
    }

    status = io_uring_register_buffers(ring, (*fetcher)->iovs, HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS);

    (*fetcher)->ioqd = nblks;
    (*fetcher)->ioring = ring;

    return HMLL_SUCCEEDED;
}

void hmll_fetcher_io_uring_free(hmll_fetcher_io_uring_t *fetcher)
{
    if (fetcher) {
        io_uring_queue_exit(fetcher->ioring);

        if (fetcher->iovs) {
            for (size_t i = 0; i < HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS; ++i) {
                if (fetcher->iovs[i].iov_base) free(fetcher->iovs[i].iov_base);
                fetcher->iovs[i].iov_len = 0;
            }
            free(fetcher->iovs);
        }

        if (fetcher->iopylds) free(fetcher->iopylds);
        if (fetcher->iobusy) free(fetcher->iobusy);
        if (fetcher->ioring) free(fetcher->ioring);
        free(fetcher);
    }
}

hmll_status_t hmll_fetcher_io_uring_fetch(
    const hmll_context_t *ctx, const hmll_fetcher_io_uring_t *fetcher, const char* name, const hmll_device_buffer_t *dst)
{
#ifdef DEBUG
    printf("[DEBUG] fetching tensor %s\n", name);
#endif

    hmll_tensor_specs_t *specs;
    const hmll_status_t status = hmll_get_tensor_specs(ctx, name, &specs);
    if (!hmll_success(status)) {
        return status;
    }

    auto const range = (hmll_fetch_range_t){specs->start, specs->end};
    return hmll_fetcher_io_uring_fetch_range(ctx, fetcher, range, dst);
}

int32_t hmll_fetcher_io_uring_get_slot(const hmll_fetcher_io_uring_t *fetcher)
{
    for (size_t i = 0; i < fetcher->ioqd; ++i) {
        if (!fetcher->iobusy[i]) return (int32_t)i;
    }

    return -1;
}

void hmll_fetcher_io_uring_prepare_payload(const hmll_fetcher_io_uring_t *fetcher, const int32_t slot, const size_t bytes_to_read, void *ptr)
{
    fetcher->iopylds[slot].buffer = slot;
    fetcher->iopylds[slot].size = bytes_to_read;
    fetcher->iopylds[slot].ptr = ptr; // buffer is 0-indexed while the file starts at range.start
    fetcher->iobusy[slot] = true;
}

hmll_status_t hmll_fetcher_io_uring_fetch_range(
    const hmll_context_t *ctx,
    const hmll_fetcher_io_uring_t *fetcher,
    const hmll_fetch_range_t range,
    const hmll_device_buffer_t *dst
)
{
    auto status = HMLL_SUCCEEDED;
    auto const size = range.end - range.start;
    const size_t nchunks = (size + HMLL_IO_URING_DEFAULT_BUFFER_SIZE - 1) / HMLL_IO_URING_DEFAULT_BUFFER_SIZE;

    if (dst->size < size) {
        status.what = HMLL_BUFFER_TOO_SMALL;
        status.message = "Provided buffer is too small to hold requested range";
        return status;
    }

#ifdef DEBUG
    printf("[DEBUG] fetching [%lu -> %lu] size=%lu, chunks=%lu\n", range.start, range.end, size, nchunks);
#endif

    size_t chunks_submitted = 0;
    size_t chunks_completed = 0;
    size_t file_offset = range.start;

    // Phase 1: Fill the pipeline - submit up to queue depth operations
    while (chunks_submitted < nchunks && chunks_submitted < fetcher->ioqd) {
        // Find a free buffer
        auto const slot = hmll_fetcher_io_uring_get_slot(fetcher);
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
        struct io_uring_sqe *sqe = io_uring_get_sqe(fetcher->ioring);
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
    io_uring_submit(fetcher->ioring);

    // Phase 2: Producer/Consumer loop - process completions and submit new reads
    while (chunks_completed < nchunks) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(fetcher->ioring, &cqe) < 0){
            return (hmll_status_t){HMLL_IO_ERROR, "Failed to wait for io_uring completion"};
        }

        // Check if read succeeded
        if (cqe->res < 0) {
#ifdef DEBUG
            printf("[ERROR] Read failed: %s\n", strerror(-cqe->res));
#endif
            io_uring_cqe_seen(fetcher->ioring, cqe);
            return (hmll_status_t){HMLL_IO_ERROR, "Read operation failed"};
        }

        // Get the buffer that just completed
        auto const payload = (hmll_fetcher_io_uring_payload_t *)io_uring_cqe_get_data(cqe);

#ifdef DEBUG
        printf("[DEBUG] completed chunk %lu: buffer=%i, bytes=%d\n", chunks_completed + 1, payload->buffer, cqe->res);
#endif

        // TODO: Handle the memcpy somewhere else (i.e. another thread?)
        memcpy(payload->ptr, fetcher->iovs[payload->buffer].iov_base, payload->size);

        io_uring_cqe_seen(fetcher->ioring, cqe);
        chunks_completed++;

        // Reuse this buffer for the next chunk if more data to read
        if (chunks_submitted < nchunks) {
            size_t bytes_to_read = HMLL_IO_URING_DEFAULT_BUFFER_SIZE;
            if (file_offset + bytes_to_read > range.end) {
                bytes_to_read = range.end - file_offset;
            }

            struct io_uring_sqe *sqe = io_uring_get_sqe(fetcher->ioring);
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
            io_uring_submit(fetcher->ioring);

#ifdef DEBUG
            printf("[DEBUG] resubmitted buffer %d for chunk %lu\n", payload->buffer, chunks_submitted);
#endif
        } else {
            // No more chunks to read, mark buffer as free
            fetcher->iobusy[payload->buffer] = false;
        }
    }

#ifdef DEBUG
    printf("[DEBUG] fetch complete: %lu chunks read\n", chunks_completed);
#endif

    return status;
}