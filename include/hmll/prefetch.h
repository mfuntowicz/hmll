//
// Prefetch Infrastructure for Pipelined Tensor Loading
// Manages concurrent load slots with async GPU allocation and transfers.
//
// Uses pinned memory staging for truly async cudaMemcpyAsync:
// - mmap provides source data (pageable)
// - memcpy to pinned staging buffer (sync but fast)
// - cudaMemcpyAsync from pinned to GPU (truly async)
//
// This enables pipeline parallelism:
// - Iteration N: GPU kernel processing tensor N-1
// - Iteration N: cudaMemcpyAsync copying tensor N from pinned to GPU
// - Iteration N: memcpy copying tensor N+1 from mmap to pinned
//

#ifndef HMLL_PREFETCH_H
#define HMLL_PREFETCH_H

#include "hmll/types.h"

// Default number of prefetch slots
#ifndef HMLL_PREFETCH_DEFAULT_SLOTS
#define HMLL_PREFETCH_DEFAULT_SLOTS 4
#endif

// Maximum prefetch slots
#ifndef HMLL_PREFETCH_MAX_SLOTS
#define HMLL_PREFETCH_MAX_SLOTS 16
#endif

// Default pinned staging buffer size (32 MB - good for large tensors)
#ifndef HMLL_PREFETCH_PINNED_SIZE
#define HMLL_PREFETCH_PINNED_SIZE (32 * 1024 * 1024)
#endif

// State of a prefetch slot
enum hmll_prefetch_state {
    HMLL_PREFETCH_IDLE = 0,      // Slot available for new load
    HMLL_PREFETCH_STAGING = 1,   // Copying to pinned buffer (sync phase)
    HMLL_PREFETCH_LOADING = 2,   // Async H2D in progress
    HMLL_PREFETCH_READY = 3,     // Load complete, awaiting consumption
    HMLL_PREFETCH_ERROR = 4,     // Load failed
};

// A single prefetch slot
struct hmll_prefetch_slot {
    void* stream;                // CUDA stream (cudaStream_t) for this slot, NULL for CPU
    void* done_event;            // CUDA event (cudaEvent_t), NULL for CPU
    void* pinned_buffer;         // Pinned staging buffer for this slot
    size_t pinned_size;          // Size of pinned buffer
    struct hmll_iobuf buffer;    // GPU buffer for this slot
    size_t tensor_index;         // Which tensor this slot is loading
    enum hmll_prefetch_state state;
};

// Prefetch context - manages multiple concurrent load slots
struct hmll_prefetch_ctx {
    struct hmll_prefetch_slot* slots;
    size_t num_slots;
    size_t next_slot;            // For round-robin slot selection
    int device_id;               // Target GPU device
    enum hmll_device device;     // Device type (CPU or CUDA)
    int use_pinned;              // Whether to use pinned memory staging
};

/// Initialize prefetch context with N slots.
/// For CUDA, creates streams and events for each slot.
/// @param ctx Output context to initialize
/// @param num_slots Number of concurrent load slots (clamped to MAX_SLOTS)
/// @param device Target device (CPU or CUDA)
/// @param device_id GPU device index (ignored for CPU)
/// @return HMLL_OK on success
struct hmll_error hmll_prefetch_init(
    struct hmll_prefetch_ctx* ctx,
    size_t num_slots,
    enum hmll_device device,
    int device_id
);

/// Enable or disable pinned memory staging.
/// Pinned memory allows truly async cudaMemcpyAsync but requires
/// a memcpy from source to pinned buffer first. For sequential loading
/// without compute overlap, this adds overhead. For workloads with
/// compute between loads, it enables better pipeline overlap.
/// @param ctx Prefetch context
/// @param enable 1 to enable pinned staging, 0 to disable
void hmll_prefetch_set_pinned(struct hmll_prefetch_ctx* ctx, int enable);

/// Start async load of tensor data into a slot.
/// Allocates GPU memory and initiates async copy from source.
/// @param ctx Prefetch context
/// @param src_ptr Source data pointer (from mmap)
/// @param size Tensor size in bytes
/// @param tensor_index Index to identify this tensor
/// @param out_slot Output: which slot was used
/// @return HMLL_OK on success
struct hmll_error hmll_prefetch_start_load(
    struct hmll_prefetch_ctx* ctx,
    const void* src_ptr,
    size_t size,
    size_t tensor_index,
    size_t* out_slot
);

/// Find a slot that is idle or has completed loading.
/// @param ctx Prefetch context
/// @return Slot index, or -1 if no slot available
int hmll_prefetch_find_available_slot(struct hmll_prefetch_ctx* ctx);

/// Check if a specific slot has completed loading (non-blocking).
/// @param ctx Prefetch context
/// @param slot_index Slot to check
/// @return 1 if ready, 0 if still loading
int hmll_prefetch_slot_ready(
    struct hmll_prefetch_ctx* ctx,
    size_t slot_index
);

/// Wait for a specific slot to complete loading (blocking).
/// @param ctx Prefetch context
/// @param slot_index Slot to wait for
/// @return HMLL_OK on success
struct hmll_error hmll_prefetch_wait_slot(
    struct hmll_prefetch_ctx* ctx,
    size_t slot_index
);

/// Find slot containing a specific tensor (by index).
/// @param ctx Prefetch context
/// @param tensor_index Tensor to find
/// @return Slot index, or -1 if not found
int hmll_prefetch_find_tensor(
    struct hmll_prefetch_ctx* ctx,
    size_t tensor_index
);

/// Get buffer from slot (transfers ownership to caller).
/// Slot state changes to IDLE after this call.
/// @param ctx Prefetch context
/// @param slot_index Slot to take buffer from
/// @param out_buffer Output: the buffer
/// @return HMLL_OK on success
struct hmll_error hmll_prefetch_take_buffer(
    struct hmll_prefetch_ctx* ctx,
    size_t slot_index,
    struct hmll_iobuf* out_buffer
);

/// Poll all slots for completion (updates slot states).
/// Call periodically to detect completed loads.
/// @param ctx Prefetch context
void hmll_prefetch_poll(struct hmll_prefetch_ctx* ctx);

/// Destroy prefetch context and free all resources.
/// @param ctx Context to destroy
void hmll_prefetch_destroy(struct hmll_prefetch_ctx* ctx);

#endif // HMLL_PREFETCH_H
