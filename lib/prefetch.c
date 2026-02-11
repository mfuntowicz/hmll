//
// Prefetch Implementation - CUDA Only
// Manages concurrent tensor loading with async GPU operations.
//
// Note: CPU prefetch is handled directly in Rust (mmap is lazy, no async needed).
//

#include "hmll/prefetch.h"
#include "hmll/hmll.h"

#include <stdlib.h>
#include <string.h>

#ifdef __HMLL_CUDA_ENABLED__

#include "hmll/cuda_pool.h"
#include <cuda_runtime_api.h>

struct hmll_error hmll_prefetch_init(
    struct hmll_prefetch_ctx* ctx,
    size_t num_slots,
    enum hmll_device device,
    int device_id
) {
    if (!ctx) {
        return HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    }

    // Only CUDA is supported
    if (device != HMLL_DEVICE_CUDA) {
        return HMLL_ERR(HMLL_ERR_INVALID_DEVICE);
    }

    memset(ctx, 0, sizeof(*ctx));

    // Clamp slot count
    if (num_slots == 0) num_slots = HMLL_PREFETCH_DEFAULT_SLOTS;
    if (num_slots > HMLL_PREFETCH_MAX_SLOTS) num_slots = HMLL_PREFETCH_MAX_SLOTS;

    ctx->device = device;
    ctx->device_id = device_id;
    ctx->num_slots = num_slots;
    ctx->next_slot = 0;

    ctx->slots = calloc(num_slots, sizeof(struct hmll_prefetch_slot));
    if (!ctx->slots) {
        return HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    }

    // Initialize each slot with its own stream and event
    for (size_t i = 0; i < num_slots; i++) {
        struct hmll_prefetch_slot* slot = &ctx->slots[i];
        slot->state = HMLL_PREFETCH_IDLE;
        slot->tensor_index = 0;
        slot->buffer.ptr = NULL;
        slot->buffer.size = 0;
        slot->buffer.device = HMLL_DEVICE_CUDA;

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            goto cleanup;
        }

        cudaStream_t stream;
        err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            goto cleanup;
        }
        slot->stream = stream;

        cudaEvent_t event;
        err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            cudaStreamDestroy(stream);
            slot->stream = NULL;
            goto cleanup;
        }
        slot->done_event = event;
        continue;

    cleanup:
        // Cleanup already-created resources
        for (size_t j = 0; j < i; j++) {
            if (ctx->slots[j].stream) {
                cudaStreamDestroy((cudaStream_t)ctx->slots[j].stream);
            }
            if (ctx->slots[j].done_event) {
                cudaEventDestroy((cudaEvent_t)ctx->slots[j].done_event);
            }
        }
        free(ctx->slots);
        ctx->slots = NULL;
        return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    // Initialize CUDA memory pool
    struct hmll_error pool_err = hmll_cuda_pool_init(device_id);
    if (hmll_check(pool_err)) {
        hmll_prefetch_destroy(ctx);
        return pool_err;
    }

    return HMLL_OK;
}

struct hmll_error hmll_prefetch_start_load(
    struct hmll_prefetch_ctx* ctx,
    const void* src_ptr,
    size_t size,
    size_t tensor_index,
    size_t* out_slot
) {
    if (!ctx || !ctx->slots) {
        return HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    }

    int slot_idx = hmll_prefetch_find_available_slot(ctx);
    if (slot_idx < 0) {
        return HMLL_ERR(HMLL_ERR_BUFFER_TOO_SMALL);
    }

    struct hmll_prefetch_slot* slot = &ctx->slots[slot_idx];

    // Handle zero-size tensors
    if (size == 0 || src_ptr == NULL) {
        slot->buffer.ptr = NULL;
        slot->buffer.size = 0;
        slot->tensor_index = tensor_index;
        slot->state = HMLL_PREFETCH_READY;
        if (out_slot) *out_slot = (size_t)slot_idx;
        return HMLL_OK;
    }

    slot->tensor_index = tensor_index;
    slot->state = HMLL_PREFETCH_LOADING;

    cudaStream_t stream = (cudaStream_t)slot->stream;
    cudaEvent_t event = (cudaEvent_t)slot->done_event;

    // Allocate from pool
    void* ptr = hmll_cuda_pool_alloc_async(size, stream, ctx->device_id);
    if (!ptr) {
        slot->state = HMLL_PREFETCH_ERROR;
        return HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    }

    slot->buffer.ptr = ptr;
    slot->buffer.size = size;
    slot->buffer.device = HMLL_DEVICE_CUDA;

    // Async copy
    cudaError_t err = cudaMemcpyAsync(ptr, src_ptr, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        hmll_cuda_pool_free_async(ptr, stream);
        slot->buffer.ptr = NULL;
        slot->state = HMLL_PREFETCH_ERROR;
        return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    // Record completion
    err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        hmll_cuda_pool_free_async(ptr, stream);
        slot->buffer.ptr = NULL;
        slot->state = HMLL_PREFETCH_ERROR;
        return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    if (out_slot) *out_slot = (size_t)slot_idx;
    return HMLL_OK;
}

int hmll_prefetch_find_available_slot(struct hmll_prefetch_ctx* ctx) {
    if (!ctx || !ctx->slots) return -1;

    hmll_prefetch_poll(ctx);

    for (size_t i = 0; i < ctx->num_slots; i++) {
        size_t idx = (ctx->next_slot + i) % ctx->num_slots;
        if (ctx->slots[idx].state == HMLL_PREFETCH_IDLE) {
            ctx->next_slot = (idx + 1) % ctx->num_slots;
            return (int)idx;
        }
    }

    return -1;
}

int hmll_prefetch_slot_ready(struct hmll_prefetch_ctx* ctx, size_t slot_index) {
    if (!ctx || !ctx->slots || slot_index >= ctx->num_slots) {
        return 0;
    }

    struct hmll_prefetch_slot* slot = &ctx->slots[slot_index];

    if (slot->state == HMLL_PREFETCH_READY || slot->state == HMLL_PREFETCH_ERROR) {
        return 1;
    }

    if (slot->state != HMLL_PREFETCH_LOADING) {
        return 0;
    }

    cudaError_t err = cudaEventQuery((cudaEvent_t)slot->done_event);
    if (err == cudaSuccess) {
        slot->state = HMLL_PREFETCH_READY;
        return 1;
    } else if (err == cudaErrorNotReady) {
        return 0;
    } else {
        slot->state = HMLL_PREFETCH_ERROR;
        return 1;
    }
}

struct hmll_error hmll_prefetch_wait_slot(struct hmll_prefetch_ctx* ctx, size_t slot_index) {
    if (!ctx || !ctx->slots || slot_index >= ctx->num_slots) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    struct hmll_prefetch_slot* slot = &ctx->slots[slot_index];

    if (slot->state == HMLL_PREFETCH_READY) {
        return HMLL_OK;
    }
    if (slot->state == HMLL_PREFETCH_ERROR) {
        return HMLL_ERR(HMLL_ERR_IO_ERROR);
    }
    if (slot->state != HMLL_PREFETCH_LOADING) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    cudaError_t err = cudaEventSynchronize((cudaEvent_t)slot->done_event);
    if (err != cudaSuccess) {
        slot->state = HMLL_PREFETCH_ERROR;
        return HMLL_ERR(HMLL_ERR_CUDA_ERROR);
    }

    slot->state = HMLL_PREFETCH_READY;
    return HMLL_OK;
}

int hmll_prefetch_find_tensor(struct hmll_prefetch_ctx* ctx, size_t tensor_index) {
    if (!ctx || !ctx->slots) return -1;

    for (size_t i = 0; i < ctx->num_slots; i++) {
        struct hmll_prefetch_slot* slot = &ctx->slots[i];
        if ((slot->state == HMLL_PREFETCH_LOADING || slot->state == HMLL_PREFETCH_READY) &&
            slot->tensor_index == tensor_index) {
            return (int)i;
        }
    }

    return -1;
}

struct hmll_error hmll_prefetch_take_buffer(
    struct hmll_prefetch_ctx* ctx,
    size_t slot_index,
    struct hmll_iobuf* out_buffer
) {
    if (!ctx || !ctx->slots || slot_index >= ctx->num_slots || !out_buffer) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    struct hmll_prefetch_slot* slot = &ctx->slots[slot_index];

    if (slot->state == HMLL_PREFETCH_LOADING) {
        struct hmll_error err = hmll_prefetch_wait_slot(ctx, slot_index);
        if (hmll_check(err)) return err;
    }

    if (slot->state == HMLL_PREFETCH_ERROR) {
        slot->state = HMLL_PREFETCH_IDLE;
        return HMLL_ERR(HMLL_ERR_IO_ERROR);
    }

    if (slot->state != HMLL_PREFETCH_READY) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    *out_buffer = slot->buffer;
    slot->buffer.ptr = NULL;
    slot->buffer.size = 0;
    slot->state = HMLL_PREFETCH_IDLE;

    return HMLL_OK;
}

void hmll_prefetch_poll(struct hmll_prefetch_ctx* ctx) {
    if (!ctx || !ctx->slots) return;

    for (size_t i = 0; i < ctx->num_slots; i++) {
        struct hmll_prefetch_slot* slot = &ctx->slots[i];
        if (slot->state == HMLL_PREFETCH_LOADING) {
            cudaError_t err = cudaEventQuery((cudaEvent_t)slot->done_event);
            if (err == cudaSuccess) {
                slot->state = HMLL_PREFETCH_READY;
            } else if (err != cudaErrorNotReady) {
                slot->state = HMLL_PREFETCH_ERROR;
            }
        }
    }
}

void hmll_prefetch_destroy(struct hmll_prefetch_ctx* ctx) {
    if (!ctx || !ctx->slots) return;

    for (size_t i = 0; i < ctx->num_slots; i++) {
        struct hmll_prefetch_slot* slot = &ctx->slots[i];
        cudaStream_t stream = (cudaStream_t)slot->stream;

        if (slot->buffer.ptr) {
            hmll_cuda_pool_free_async(slot->buffer.ptr, stream);
        }

        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }

        if (slot->done_event) {
            cudaEventDestroy((cudaEvent_t)slot->done_event);
        }
    }

    free(ctx->slots);
    ctx->slots = NULL;
    ctx->num_slots = 0;
}

#else // !__HMLL_CUDA_ENABLED__

// Stub implementations for non-CUDA builds
// These return errors since prefetch is CUDA-only

struct hmll_error hmll_prefetch_init(
    struct hmll_prefetch_ctx* ctx,
    size_t num_slots,
    enum hmll_device device,
    int device_id
) {
    HMLL_UNUSED(ctx);
    HMLL_UNUSED(num_slots);
    HMLL_UNUSED(device);
    HMLL_UNUSED(device_id);
    return HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
}

struct hmll_error hmll_prefetch_start_load(
    struct hmll_prefetch_ctx* ctx,
    const void* src_ptr,
    size_t size,
    size_t tensor_index,
    size_t* out_slot
) {
    HMLL_UNUSED(ctx);
    HMLL_UNUSED(src_ptr);
    HMLL_UNUSED(size);
    HMLL_UNUSED(tensor_index);
    HMLL_UNUSED(out_slot);
    return HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
}

int hmll_prefetch_find_available_slot(struct hmll_prefetch_ctx* ctx) {
    HMLL_UNUSED(ctx);
    return -1;
}

int hmll_prefetch_slot_ready(struct hmll_prefetch_ctx* ctx, size_t slot_index) {
    HMLL_UNUSED(ctx);
    HMLL_UNUSED(slot_index);
    return 0;
}

struct hmll_error hmll_prefetch_wait_slot(struct hmll_prefetch_ctx* ctx, size_t slot_index) {
    HMLL_UNUSED(ctx);
    HMLL_UNUSED(slot_index);
    return HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
}

int hmll_prefetch_find_tensor(struct hmll_prefetch_ctx* ctx, size_t tensor_index) {
    HMLL_UNUSED(ctx);
    HMLL_UNUSED(tensor_index);
    return -1;
}

struct hmll_error hmll_prefetch_take_buffer(
    struct hmll_prefetch_ctx* ctx,
    size_t slot_index,
    struct hmll_iobuf* out_buffer
) {
    HMLL_UNUSED(ctx);
    HMLL_UNUSED(slot_index);
    HMLL_UNUSED(out_buffer);
    return HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
}

void hmll_prefetch_poll(struct hmll_prefetch_ctx* ctx) {
    HMLL_UNUSED(ctx);
}

void hmll_prefetch_destroy(struct hmll_prefetch_ctx* ctx) {
    HMLL_UNUSED(ctx);
}

#endif // __HMLL_CUDA_ENABLED__
