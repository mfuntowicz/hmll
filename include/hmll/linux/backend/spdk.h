#ifndef HMLL_FETCHER_SPDK_H
#define HMLL_FETCHER_SPDK_H

#ifndef HMLL_SPDK_QUEUE_DEPTH
#define HMLL_SPDK_QUEUE_DEPTH 128U
#endif

#ifndef HMLL_SPDK_BUFFER_SIZE
#define HMLL_SPDK_BUFFER_SIZE (8U * 1024 * 1024)
#endif

// SPDK headers use zero-length arrays and flexible array extensions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-length-array"
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wflexible-array-extensions"
#endif
#include <spdk/nvme.h>
#include <spdk/env.h>
#pragma GCC diagnostic pop

#include "hmll/types.h"

// Per-namespace context
struct hmll_spdk_ns_entry {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    uint32_t sector_size;
    uint64_t num_sectors;
    char *path;  // Original file path for this namespace
};

// Forward declaration
struct hmll_spdk;

// I/O request tracking
struct hmll_spdk_io_request {
    void *dst_buffer;           // Final destination buffer
    void *staging_buffer;       // DMA-aligned staging buffer (if needed)
    size_t offset;              // Offset within the destination buffer
    size_t length;              // Length of the I/O request
    size_t file_offset;         // Offset in the file
    int ns_index;               // Which namespace this request belongs to
    int slot;                   // Slot index for tracking
    unsigned char completed;    // Completion flag
    int result;                 // Result code
    struct hmll_spdk *backend;  // Back-pointer to backend (for CUDA)
    struct hmll_spdk_io_request *next;  // For free list
};

// Slot management (similar to io_uring)
struct hmll_spdk_slots {
    long long msb;
    long long lsb;
};

#if defined(__HMLL_CUDA_ENABLED__)
#include <driver_types.h>

enum hmll_spdk_cuda_state {
    HMLL_CUDA_SPDK_IDLE = 0,
    HMLL_CUDA_SPDK_MEMCPY = 1,
};

struct hmll_spdk_cuda_context {
    cudaStream_t stream;
    cudaEvent_t done;
    size_t offset;
    int slot;
    enum hmll_spdk_cuda_state state;
};

static inline void hmll_spdk_cuda_stream_set_idle(enum hmll_spdk_cuda_state *state) {
    *state = HMLL_CUDA_SPDK_IDLE;
}

static inline void hmll_spdk_cuda_stream_set_memcpy(enum hmll_spdk_cuda_state *state) {
    *state = HMLL_CUDA_SPDK_MEMCPY;
}
#endif

// Main SPDK backend structure
struct hmll_spdk {
    // Namespace entries (one per file/device)
    struct hmll_spdk_ns_entry *ns_entries;
    size_t num_namespaces;

    // I/O request pool
    struct hmll_spdk_io_request *request_pool;
    struct hmll_spdk_io_request *free_requests;
    struct hmll_spdk_slots slots;

    // Slot-to-request mapping for fast lookup
    struct hmll_spdk_io_request *slot_requests[HMLL_SPDK_QUEUE_DEPTH];

    // DMA-aligned staging buffers for reads
    void **staging_buffers;

    // Statistics
    size_t outstanding_ios;
    size_t completed_ios;

    // Optional CUDA context
    void *device_ctx;
};

// Slot management functions (similar to io_uring)
static inline unsigned int hmll_spdk_slot_is_busy(const struct hmll_spdk_slots slots, const unsigned int slot) {
    if (slot < 64)
        return slots.lsb & (1LL << slot);
    return slots.msb & (1LL << (slot - 64));
}

static inline int hmll_spdk_slot_find_available(const struct hmll_spdk_slots slots) {
    const int pos_lsb = __builtin_ffsll(~slots.lsb);
    if (pos_lsb > 0)
        return pos_lsb - 1;

    const int pos_msb = __builtin_ffsll(~slots.msb);
    if (pos_msb > 0)
        return 64 + pos_msb - 1;

    return -1;
}

static inline void hmll_spdk_slot_set_busy(struct hmll_spdk_slots *slots, const unsigned int slot) {
    if (slot < 64) {
        slots->lsb |= 1LL << slot;
    } else {
        slots->msb |= 1LL << (slot - 64);
    }
}

static inline void hmll_spdk_slot_set_available(struct hmll_spdk_slots *slots, const unsigned int slot) {
    if (slot < 64) {
        slots->lsb &= ~(1LL << slot);
    } else {
        slots->msb &= ~(1LL << (slot - 64));
    }
}

// Function declarations
struct hmll_error hmll_spdk_init(struct hmll *ctx, enum hmll_device device);
void hmll_spdk_cleanup(struct hmll_spdk *backend);

#endif // HMLL_FETCHER_SPDK_H
