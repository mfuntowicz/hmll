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
    auto const ring = (struct io_uring*)malloc(sizeof(struct io_uring));
    if ((status = io_uring_queue_init(nblks, ring, 0))) {
        free(ring);
        return (hmll_status_t) {HMLL_ALLOCATION_FAILED, "Failed to initialize io_uring ring"};
    }

    // Allocate iovectors for fixed buffer copy - it avoids the kernel to map/unmap these buffers at every call
    // TODO: Check allocations
    (*fetcher)->iovs = malloc(HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS * sizeof(struct iovec));
    (*fetcher)->iobusy = calloc(nblks, sizeof(bool));

    for (int i = 0; i < HMLL_IO_URING_DEFAULT_NUM_IO_VECTORS; i++) {
        // Must be page-aligned for io_uring registered buffers
        if (((*fetcher)->iovs[i].iov_base = aligned_alloc(4096, buffer_size_)) == nullptr) {
            return (hmll_status_t) {HMLL_ALLOCATION_FAILED, "Failed to allocate page-aligned buffer"};
        }

        memset((*fetcher)->iovs[i].iov_base, 0, buffer_size_);
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

        if (fetcher->iobusy) free(fetcher->iobusy);
        if (fetcher->ioring) free(fetcher->ioring);
        free(fetcher);
    }
}

hmll_status_t hmll_fetcher_io_uring_fetch(
    const hmll_context_t *ctx, const hmll_fetcher_io_uring_t *fetcher, const char* name)
{
#ifdef DEBUG
    printf("[DEBUG] fetching tensor %s\n", name);
#endif

    hmll_tensor_specs_t *specs;
    const hmll_status_t status = hmll_get_tensor_specs(ctx, name, &specs);
    if (hmll_status_has_error(status)) {
        return status;
    }

    auto const size = specs->end - specs->start;
    size_t nchunks = (size + HMLL_IO_URING_DEFAULT_BUFFER_SIZE - 1) / HMLL_IO_URING_DEFAULT_BUFFER_SIZE;

#ifdef DEBUG
    printf("[DEBUG] tensor %s: [%lu -> %lu] size=%lu, chunks=%lu\n",
        name, specs->start, specs->end, size, nchunks);
#endif

    size_t chunks_submitted = 0;
    size_t chunks_completed = 0;
    size_t file_offset = specs->start;

    // Phase 1: Fill the pipeline - submit up to queue depth operations
    while (chunks_submitted < nchunks && chunks_submitted < fetcher->ioqd) {
        // Find a free buffer
        int32_t slot = -1;
        for (size_t i = 0; i < fetcher->ioqd; i++) {
            if (!fetcher->iobusy[i]) {
                slot = i;
                break;
            }
        }

        if (slot == -1) break; // No free buffers

        // Calculate bytes to read (handle last chunk)
        size_t bytes_to_read = HMLL_IO_URING_DEFAULT_BUFFER_SIZE;
        if (file_offset + bytes_to_read > specs->end) {
            bytes_to_read = specs->end - file_offset;
        }

        // Submit read operation
        struct io_uring_sqe *sqe = io_uring_get_sqe(fetcher->ioring);
        io_uring_prep_read_fixed(sqe, ctx->source.fd, fetcher->iovs[slot].iov_base, bytes_to_read, file_offset, slot);
        io_uring_sqe_set_data(sqe, (void*) (uintptr_t)slot);

        fetcher->iobusy[slot] = true;
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
    int ret = 0;
    while (chunks_completed < nchunks) {
        struct io_uring_cqe *cqe;
        if ((ret = io_uring_wait_cqe(fetcher->ioring, &cqe)) < 0){
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
        int bidx = (int)(uintptr_t)io_uring_cqe_get_data(cqe);

#ifdef DEBUG
        printf("[DEBUG] completed chunk %lu: buffer=%d, bytes=%d\n",
               chunks_completed + 1, bidx, cqe->res);
#endif

        // TODO: Process the data in fetcher->ioblks[buf_idx].buffer
        // The buffer contains cqe->res bytes of data

        io_uring_cqe_seen(fetcher->ioring, cqe);
        chunks_completed++;

        // Reuse this buffer for the next chunk if more data to read
        if (chunks_submitted < nchunks) {
            size_t bytes_to_read = HMLL_IO_URING_DEFAULT_BUFFER_SIZE;
            if (file_offset + bytes_to_read > specs->end) {
                bytes_to_read = specs->end - file_offset;
            }

            struct io_uring_sqe *sqe = io_uring_get_sqe(fetcher->ioring);
            io_uring_prep_read_fixed(
                sqe, ctx->source.fd, fetcher->iovs[bidx].iov_base, bytes_to_read, file_offset, bidx);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)bidx);

            file_offset += bytes_to_read;
            chunks_submitted++;
            io_uring_submit(fetcher->ioring);

#ifdef DEBUG
            printf("[DEBUG] resubmitted buffer %d for chunk %lu\n", bidx, chunks_submitted);
#endif
        } else {
            // No more chunks to read, mark buffer as free
            fetcher->iobusy[bidx] = false;
        }
    }

#ifdef DEBUG
    printf("[DEBUG] fetch complete: %lu chunks read\n", chunks_completed);
#endif

    return HMLL_SUCCEEDED;
}