//
// Pinned Memory Pool Implementation
// Uses cudaHostAlloc for page-locked memory that enables truly async H2D transfers.
//

#include "hmll/pinned_pool.h"
#include "hmll/hmll.h"

#include <stdlib.h>
#include <string.h>

#ifdef __HMLL_CUDA_ENABLED__

#include <cuda_runtime_api.h>

struct hmll_error hmll_pinned_pool_init(
    struct hmll_pinned_pool* pool,
    size_t num_buffers,
    size_t buffer_size
) {
    if (!pool) {
        return HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    }

    memset(pool, 0, sizeof(*pool));

    // Use defaults if not specified
    if (num_buffers == 0) num_buffers = 4;
    if (buffer_size == 0) buffer_size = HMLL_PINNED_BUFFER_SIZE;

    // Clamp to max
    if (num_buffers > HMLL_PINNED_POOL_MAX_BUFFERS) {
        num_buffers = HMLL_PINNED_POOL_MAX_BUFFERS;
    }

    pool->num_buffers = num_buffers;
    pool->buffer_size = buffer_size;

    pool->buffers = calloc(num_buffers, sizeof(struct hmll_pinned_buffer));
    if (!pool->buffers) {
        return HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    }

    // Allocate pinned buffers
    for (size_t i = 0; i < num_buffers; i++) {
        struct hmll_pinned_buffer* buf = &pool->buffers[i];

        // Use cudaHostAllocDefault for basic pinned memory
        // Could use cudaHostAllocWriteCombined for better H2D performance
        // but that makes reads from host slower
        cudaError_t err = cudaHostAlloc(&buf->ptr, buffer_size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            // Free already allocated buffers
            for (size_t j = 0; j < i; j++) {
                cudaFreeHost(pool->buffers[j].ptr);
            }
            free(pool->buffers);
            pool->buffers = NULL;
            return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
        }

        buf->size = buffer_size;
        buf->used = 0;
        buf->in_use = 0;
    }

    pool->initialized = 1;
    return HMLL_OK;
}

void* hmll_pinned_pool_acquire(
    struct hmll_pinned_pool* pool,
    size_t min_size
) {
    if (!pool || !pool->initialized || !pool->buffers) {
        return NULL;
    }

    // Find an available buffer that's large enough
    for (size_t i = 0; i < pool->num_buffers; i++) {
        struct hmll_pinned_buffer* buf = &pool->buffers[i];
        if (!buf->in_use && buf->size >= min_size) {
            buf->in_use = 1;
            buf->used = min_size;
            return buf->ptr;
        }
    }

    // No buffer available - could allocate a new one on demand
    // For now, return NULL and let caller handle
    return NULL;
}

void hmll_pinned_pool_release(
    struct hmll_pinned_pool* pool,
    void* ptr
) {
    if (!pool || !pool->initialized || !pool->buffers || !ptr) {
        return;
    }

    for (size_t i = 0; i < pool->num_buffers; i++) {
        struct hmll_pinned_buffer* buf = &pool->buffers[i];
        if (buf->ptr == ptr) {
            buf->in_use = 0;
            buf->used = 0;
            return;
        }
    }
}

size_t hmll_pinned_pool_buffer_size(
    struct hmll_pinned_pool* pool,
    void* ptr
) {
    if (!pool || !pool->initialized || !pool->buffers || !ptr) {
        return 0;
    }

    for (size_t i = 0; i < pool->num_buffers; i++) {
        struct hmll_pinned_buffer* buf = &pool->buffers[i];
        if (buf->ptr == ptr) {
            return buf->size;
        }
    }

    return 0;
}

void hmll_pinned_pool_destroy(struct hmll_pinned_pool* pool) {
    if (!pool || !pool->initialized || !pool->buffers) {
        return;
    }

    for (size_t i = 0; i < pool->num_buffers; i++) {
        struct hmll_pinned_buffer* buf = &pool->buffers[i];
        if (buf->ptr) {
            cudaFreeHost(buf->ptr);
            buf->ptr = NULL;
        }
    }

    free(pool->buffers);
    pool->buffers = NULL;
    pool->num_buffers = 0;
    pool->initialized = 0;
}

int hmll_pinned_pool_available(void) {
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

#endif // __HMLL_CUDA_ENABLED__
