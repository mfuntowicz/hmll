//
// CUDA Memory Pool Management
// Provides efficient GPU memory allocation using cudaMallocAsync with memory pools.
//

#ifndef HMLL_CUDA_POOL_H
#define HMLL_CUDA_POOL_H

#include "hmll/types.h"

#ifdef __HMLL_CUDA_ENABLED__
#include <cuda_runtime_api.h>

// Maximum number of CUDA devices we support
#define HMLL_MAX_CUDA_DEVICES 16

// Forward declaration
struct hmll_cuda_pool;

/// Initialize the CUDA pool for a specific device.
/// This is idempotent - calling multiple times is safe.
/// Returns HMLL_OK on success, or an error if initialization fails.
struct hmll_error hmll_cuda_pool_init(int device_id);

/// Get the memory pool handle for a device.
/// Pool must be initialized first via hmll_cuda_pool_init().
/// Returns NULL if device not initialized or invalid.
cudaMemPool_t hmll_cuda_pool_get(int device_id);

/// Allocate memory from the pool asynchronously.
/// The allocation is stream-ordered - it completes before any subsequent
/// operations on the same stream.
/// Returns NULL on failure.
void* hmll_cuda_pool_alloc_async(size_t size, cudaStream_t stream, int device_id);

/// Free memory back to the pool asynchronously.
/// The free is stream-ordered - it waits for prior operations on the stream.
void hmll_cuda_pool_free_async(void* ptr, cudaStream_t stream);

/// Trim the pool to release unused memory back to the system.
/// Useful when memory pressure is detected.
/// @param device_id The device whose pool to trim
/// @param min_bytes_to_keep Minimum bytes to retain in the pool
void hmll_cuda_pool_trim(int device_id, size_t min_bytes_to_keep);

/// Get current pool memory statistics.
/// @param device_id The device to query
/// @param used_bytes Output: bytes currently allocated from pool
/// @param reserved_bytes Output: bytes reserved by pool (including free blocks)
void hmll_cuda_pool_stats(int device_id, size_t* used_bytes, size_t* reserved_bytes);

/// Cleanup all CUDA pools. Call at program shutdown.
void hmll_cuda_pool_cleanup(void);

/// Check if CUDA pools are available (CUDA enabled and device supports pools).
int hmll_cuda_pool_available(void);

#else // !__HMLL_CUDA_ENABLED__

// Stub declarations for non-CUDA builds
// Note: We need hmll.h for HMLL_ERR macro, but can't include it due to circular deps
// Use direct struct initialization instead
static inline struct hmll_error hmll_cuda_pool_init(int device_id) {
    HMLL_UNUSED(device_id);
    struct hmll_error err = { HMLL_ERR_CUDA_NOT_ENABLED, 0 };
    return err;
}

static inline void* hmll_cuda_pool_alloc_async(size_t size, void* stream, int device_id) {
    HMLL_UNUSED(size);
    HMLL_UNUSED(stream);
    HMLL_UNUSED(device_id);
    return NULL;
}

static inline void hmll_cuda_pool_free_async(void* ptr, void* stream) {
    HMLL_UNUSED(ptr);
    HMLL_UNUSED(stream);
}

static inline void hmll_cuda_pool_trim(int device_id, size_t min_bytes_to_keep) {
    HMLL_UNUSED(device_id);
    HMLL_UNUSED(min_bytes_to_keep);
}

static inline void hmll_cuda_pool_stats(int device_id, size_t* used_bytes, size_t* reserved_bytes) {
    HMLL_UNUSED(device_id);
    if (used_bytes) *used_bytes = 0;
    if (reserved_bytes) *reserved_bytes = 0;
}

static inline void hmll_cuda_pool_cleanup(void) {}

static inline int hmll_cuda_pool_available(void) { return 0; }

#endif // __HMLL_CUDA_ENABLED__

#endif // HMLL_CUDA_POOL_H
