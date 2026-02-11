//
// Pinned (Page-Locked) Memory Pool for Fast H2D Transfers
// Uses cudaHostAlloc for pinned memory that enables truly async cudaMemcpyAsync.
//

#ifndef HMLL_PINNED_POOL_H
#define HMLL_PINNED_POOL_H

#include "hmll/types.h"
#include <stddef.h>

#ifdef __HMLL_CUDA_ENABLED__

// Size of each pinned buffer in the pool
// 32MB is a good balance for large tensors vs memory usage
#ifndef HMLL_PINNED_BUFFER_SIZE
#define HMLL_PINNED_BUFFER_SIZE (32 * 1024 * 1024)
#endif

// Maximum number of pinned buffers in the pool
#ifndef HMLL_PINNED_POOL_MAX_BUFFERS
#define HMLL_PINNED_POOL_MAX_BUFFERS 8
#endif

// A pinned memory buffer
struct hmll_pinned_buffer {
    void* ptr;           // Pinned memory pointer (from cudaHostAlloc)
    size_t size;         // Allocated size
    size_t used;         // Currently used bytes (for sub-allocation)
    int in_use;          // Whether this buffer is currently in use
};

// Pinned memory pool
struct hmll_pinned_pool {
    struct hmll_pinned_buffer* buffers;
    size_t num_buffers;
    size_t buffer_size;   // Size of each buffer
    int initialized;
};

/// Initialize the pinned memory pool.
/// Allocates `num_buffers` buffers of `buffer_size` each using cudaHostAlloc.
/// @param pool Output pool structure
/// @param num_buffers Number of buffers (clamped to MAX_BUFFERS)
/// @param buffer_size Size of each buffer (use 0 for default)
/// @return HMLL_OK on success
struct hmll_error hmll_pinned_pool_init(
    struct hmll_pinned_pool* pool,
    size_t num_buffers,
    size_t buffer_size
);

/// Acquire a pinned buffer from the pool.
/// @param pool The pool
/// @param min_size Minimum required size
/// @return Pointer to pinned memory, or NULL if none available
void* hmll_pinned_pool_acquire(
    struct hmll_pinned_pool* pool,
    size_t min_size
);

/// Release a pinned buffer back to the pool.
/// @param pool The pool
/// @param ptr Pointer to release
void hmll_pinned_pool_release(
    struct hmll_pinned_pool* pool,
    void* ptr
);

/// Get the size of a pinned buffer.
/// @param pool The pool
/// @param ptr Pointer to query
/// @return Size of the buffer, or 0 if not found
size_t hmll_pinned_pool_buffer_size(
    struct hmll_pinned_pool* pool,
    void* ptr
);

/// Destroy the pinned memory pool and free all buffers.
/// @param pool Pool to destroy
void hmll_pinned_pool_destroy(struct hmll_pinned_pool* pool);

/// Check if pinned memory is available (CUDA enabled).
int hmll_pinned_pool_available(void);

#else // !__HMLL_CUDA_ENABLED__

// Stub structures for non-CUDA builds
struct hmll_pinned_buffer {
    void* ptr;
    size_t size;
    size_t used;
    int in_use;
};

struct hmll_pinned_pool {
    struct hmll_pinned_buffer* buffers;
    size_t num_buffers;
    size_t buffer_size;
    int initialized;
};

static inline struct hmll_error hmll_pinned_pool_init(
    struct hmll_pinned_pool* pool,
    size_t num_buffers,
    size_t buffer_size
) {
    HMLL_UNUSED(pool);
    HMLL_UNUSED(num_buffers);
    HMLL_UNUSED(buffer_size);
    struct hmll_error err = { HMLL_ERR_CUDA_NOT_ENABLED, 0 };
    return err;
}

static inline void* hmll_pinned_pool_acquire(
    struct hmll_pinned_pool* pool,
    size_t min_size
) {
    HMLL_UNUSED(pool);
    HMLL_UNUSED(min_size);
    return NULL;
}

static inline void hmll_pinned_pool_release(
    struct hmll_pinned_pool* pool,
    void* ptr
) {
    HMLL_UNUSED(pool);
    HMLL_UNUSED(ptr);
}

static inline size_t hmll_pinned_pool_buffer_size(
    struct hmll_pinned_pool* pool,
    void* ptr
) {
    HMLL_UNUSED(pool);
    HMLL_UNUSED(ptr);
    return 0;
}

static inline void hmll_pinned_pool_destroy(struct hmll_pinned_pool* pool) {
    HMLL_UNUSED(pool);
}

static inline int hmll_pinned_pool_available(void) { return 0; }

#endif // __HMLL_CUDA_ENABLED__

#endif // HMLL_PINNED_POOL_H
