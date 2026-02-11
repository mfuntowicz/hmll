//
// CUDA Memory Pool Implementation
// Uses cudaMallocAsync with per-device memory pools for efficient GPU allocation.
//

#include "hmll/cuda_pool.h"
#include "hmll/hmll.h"

#ifdef __HMLL_CUDA_ENABLED__

#include <cuda_runtime_api.h>
#include <stdatomic.h>
#include <string.h>

// Per-device pool state
struct hmll_cuda_pool {
    cudaMemPool_t pool;
    atomic_int initialized;  // 0 = not init, 1 = initializing, 2 = ready
    int device_id;
};

// Global pool array (one per device)
static struct hmll_cuda_pool g_pools[HMLL_MAX_CUDA_DEVICES];

// Check if memory pools are supported (CUDA 11.2+)
static int check_pool_support(int device_id) {
    int supports_pools = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &supports_pools,
        cudaDevAttrMemoryPoolsSupported,
        device_id
    );
    return (err == cudaSuccess && supports_pools);
}

struct hmll_error hmll_cuda_pool_init(int device_id) {
    if (device_id < 0 || device_id >= HMLL_MAX_CUDA_DEVICES) {
        return HMLL_ERR(HMLL_ERR_INVALID_DEVICE);
    }

    struct hmll_cuda_pool* pool = &g_pools[device_id];

    // Fast path: already initialized
    int state = atomic_load(&pool->initialized);
    if (state == 2) {
        return HMLL_OK;
    }

    // Try to claim initialization
    int expected = 0;
    if (!atomic_compare_exchange_strong(&pool->initialized, &expected, 1)) {
        // Another thread is initializing, spin wait
        while (atomic_load(&pool->initialized) == 1) {
            // Busy wait (could use sched_yield on Linux)
        }
        return atomic_load(&pool->initialized) == 2 ? HMLL_OK : HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    // We claimed it, do initialization
    cudaError_t err;

    // Set device context
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        atomic_store(&pool->initialized, 0);
        return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    // Check if device supports memory pools
    if (!check_pool_support(device_id)) {
        // Fall back to default allocator (no pool)
        pool->pool = NULL;
        pool->device_id = device_id;
        atomic_store(&pool->initialized, 2);
        return HMLL_OK;
    }

    // Get the default memory pool for this device
    err = cudaDeviceGetDefaultMemPool(&pool->pool, device_id);
    if (err != cudaSuccess) {
        atomic_store(&pool->initialized, 0);
        return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    // Use default release threshold (0) which releases memory when not in use.
    // This allows memory to be reclaimed between benchmark iterations while still
    // benefiting from pool allocation during active use.

    pool->device_id = device_id;
    atomic_store(&pool->initialized, 2);

    return HMLL_OK;
}

cudaMemPool_t hmll_cuda_pool_get(int device_id) {
    if (device_id < 0 || device_id >= HMLL_MAX_CUDA_DEVICES) {
        return NULL;
    }

    struct hmll_cuda_pool* pool = &g_pools[device_id];
    if (atomic_load(&pool->initialized) != 2) {
        return NULL;
    }

    return pool->pool;
}

void* hmll_cuda_pool_alloc_async(size_t size, cudaStream_t stream, int device_id) {
    if (size == 0) return NULL;

    // Ensure pool is initialized
    struct hmll_error err = hmll_cuda_pool_init(device_id);
    if (hmll_check(err)) {
        return NULL;
    }

    struct hmll_cuda_pool* pool = &g_pools[device_id];
    void* ptr = NULL;
    cudaError_t cuda_err;

    if (pool->pool != NULL) {
        // Use pool-based async allocation
        cuda_err = cudaMallocFromPoolAsync(&ptr, size, pool->pool, stream);
    } else {
        // Fallback: device doesn't support pools, use regular async alloc
        cuda_err = cudaMallocAsync(&ptr, size, stream);
    }

    if (cuda_err != cudaSuccess) {
        // Allocation failed - could be OOM
        // Caller should handle by trimming pool or waiting for frees
        return NULL;
    }

    return ptr;
}

void hmll_cuda_pool_free_async(void* ptr, cudaStream_t stream) {
    if (ptr == NULL) return;

    cudaFreeAsync(ptr, stream);
}

void hmll_cuda_pool_trim(int device_id, size_t min_bytes_to_keep) {
    if (device_id < 0 || device_id >= HMLL_MAX_CUDA_DEVICES) {
        return;
    }

    struct hmll_cuda_pool* pool = &g_pools[device_id];
    if (atomic_load(&pool->initialized) != 2 || pool->pool == NULL) {
        return;
    }

    cudaMemPoolTrimTo(pool->pool, min_bytes_to_keep);
}

void hmll_cuda_pool_stats(int device_id, size_t* used_bytes, size_t* reserved_bytes) {
    if (used_bytes) *used_bytes = 0;
    if (reserved_bytes) *reserved_bytes = 0;

    if (device_id < 0 || device_id >= HMLL_MAX_CUDA_DEVICES) {
        return;
    }

    struct hmll_cuda_pool* pool = &g_pools[device_id];
    if (atomic_load(&pool->initialized) != 2 || pool->pool == NULL) {
        return;
    }

    if (used_bytes) {
        cudaMemPoolGetAttribute(
            pool->pool,
            cudaMemPoolAttrUsedMemCurrent,
            used_bytes
        );
    }

    if (reserved_bytes) {
        cudaMemPoolGetAttribute(
            pool->pool,
            cudaMemPoolAttrReservedMemCurrent,
            reserved_bytes
        );
    }
}

void hmll_cuda_pool_cleanup(void) {
    // Pools are device-default pools, no explicit destruction needed
    // Just reset our state
    for (int i = 0; i < HMLL_MAX_CUDA_DEVICES; i++) {
        // Trim pools to release memory
        if (atomic_load(&g_pools[i].initialized) == 2 && g_pools[i].pool != NULL) {
            cudaMemPoolTrimTo(g_pools[i].pool, 0);
        }
        atomic_store(&g_pools[i].initialized, 0);
        g_pools[i].pool = NULL;
    }
}

int hmll_cuda_pool_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }

    // Check if at least one device supports pools
    for (int i = 0; i < device_count && i < HMLL_MAX_CUDA_DEVICES; i++) {
        if (check_pool_support(i)) {
            return 1;
        }
    }

    return 0;
}

#endif // __HMLL_CUDA_ENABLED__
