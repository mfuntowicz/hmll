#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "hmll/hmll.h"
#include "hmll/cuda.h"
#include "hmll/memory.h"
#include "hmll/linux/backend/spdk.h"

// SPDK headers use zero-length arrays and flexible array extensions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-length-array"
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wflexible-array-extensions"
#endif
#include <spdk/nvme.h>
#include <spdk/env.h>
#include <spdk/string.h>
#pragma GCC diagnostic pop

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif

/**
 * SPDK Backend Implementation for HMLL
 *
 * This backend uses SPDK's user-space NVMe driver to achieve maximum I/O performance
 * by bypassing the kernel. Key features:
 *
 * - Direct NVMe access via SPDK
 * - Zero-copy operations where possible
 * - Async I/O with completion polling
 * - DMA-aligned buffer management
 * - Optional CUDA support with staging buffers
 */

// Forward declarations
static void hmll_spdk_read_complete(void *arg, const struct spdk_nvme_cpl *completion);
static void hmll_spdk_attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                                 struct spdk_nvme_ctrlr *ctrlr,
                                 const struct spdk_nvme_ctrlr_opts *opts);

/**
 * Allocate a request from the free pool
 */
static struct hmll_spdk_io_request *hmll_spdk_alloc_request(struct hmll_spdk *backend) {
    if (backend->free_requests == NULL) {
        return NULL;
    }

    struct hmll_spdk_io_request *req = backend->free_requests;
    backend->free_requests = req->next;
    memset(req, 0, sizeof(*req));
    return req;
}

/**
 * Return a request to the free pool
 */
static void hmll_spdk_free_request(struct hmll_spdk *backend, struct hmll_spdk_io_request *req) {
    req->next = backend->free_requests;
    backend->free_requests = req;
}

/**
 * Completion callback for SPDK NVMe reads
 */
static void hmll_spdk_read_complete(void *arg, const struct spdk_nvme_cpl *completion) {
    struct hmll_spdk_io_request *req = arg;

    req->completed = 1;
    req->result = spdk_nvme_cpl_is_error(completion) ? -1 : 0;

    // For CUDA, trigger host-to-device copy
#ifdef __HMLL_CUDA_ENABLED__
    if (req->staging_buffer && req->result == 0) {
        struct hmll_spdk *backend = req->backend;
        struct hmll_spdk_cuda_context *dctx = backend->device_ctx;
        struct hmll_spdk_cuda_context *cd = &dctx[req->slot];

        cudaMemcpyAsync(req->dst_buffer, req->staging_buffer, req->length,
                       cudaMemcpyHostToDevice, cd->stream);
        cudaEventRecord(cd->done, cd->stream);
        hmll_spdk_cuda_stream_set_memcpy(&cd->state);
    }
#endif
}

/**
 * Reclaim completed CUDA transfers
 */
static inline size_t hmll_spdk_reclaim_cuda_slots(
    struct hmll_spdk *backend,
    const enum hmll_device device
) {
#ifdef __HMLL_CUDA_ENABLED__
    if (device != HMLL_DEVICE_CUDA) return 0;

    struct hmll_spdk_cuda_context *dctx = backend->device_ctx;
    size_t reclaimed = 0;

    for (size_t i = 0; i < HMLL_SPDK_QUEUE_DEPTH; ++i) {
        struct hmll_spdk_cuda_context *cd = dctx + i;
        if (hmll_spdk_slot_is_busy(backend->slots, i)) {
            if (cd->state == HMLL_CUDA_SPDK_MEMCPY && cudaEventQuery(cd->done) == cudaSuccess) {
                hmll_spdk_cuda_stream_set_idle(&cd->state);

                // Free the request and slot
                struct hmll_spdk_io_request *req = backend->slot_requests[cd->slot];
                if (req) {
                    reclaimed += req->length;
                    backend->slot_requests[cd->slot] = NULL;
                    hmll_spdk_free_request(backend, req);
                    backend->outstanding_ios--;
                }

                hmll_spdk_slot_set_available(&backend->slots, cd->slot);
            }
        }
    }
    return reclaimed;
#else
    HMLL_UNUSED(backend);
    HMLL_UNUSED(device);
    return 0;
#endif
}

/**
 * Convert file offset to LBA (Logical Block Address)
 */
static inline uint64_t hmll_spdk_offset_to_lba(size_t offset, uint32_t sector_size) {
    return offset / sector_size;
}

/**
 * Calculate number of sectors needed for a given length
 */
static inline uint32_t hmll_spdk_length_to_sectors(size_t length, uint32_t sector_size) {
    return (length + sector_size - 1) / sector_size;
}

/**
 * Submit a read request to SPDK
 */
static int hmll_spdk_submit_read(
    struct hmll_spdk *backend,
    struct hmll_spdk_io_request *req,
    const int ns_index,
    void *buffer,
    const size_t file_offset,
    const size_t length
) {
    struct hmll_spdk_ns_entry *ns_entry = &backend->ns_entries[ns_index];

    uint64_t lba = hmll_spdk_offset_to_lba(file_offset, ns_entry->sector_size);
    uint32_t lba_count = hmll_spdk_length_to_sectors(length, ns_entry->sector_size);

    // Ensure buffer is DMA-aligned (SPDK requirement)
    if ((uintptr_t)buffer % 4096 != 0) {
        return -1;
    }

    int rc = spdk_nvme_ns_cmd_read(
        ns_entry->ns,
        ns_entry->qpair,
        buffer,
        lba,
        lba_count,
        hmll_spdk_read_complete,
        req,
        0
    );

    if (rc != 0) {
        return rc;
    }

    backend->outstanding_ios++;
    return 0;
}

/**
 * Process completions from SPDK
 */
static inline int hmll_spdk_process_completions(struct hmll_spdk *backend, int ns_index) {
    struct hmll_spdk_ns_entry *ns_entry = &backend->ns_entries[ns_index];
    return spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
}

/**
 * Map source files to NVMe namespaces
 * For production use, this would parse the file path and determine which NVMe device to use
 * For now, we use a simple round-robin mapping
 */
static int hmll_spdk_map_file_to_namespace(
    struct hmll_spdk *backend,
    const int file_index
) {
    // Simple round-robin for now
    // In production, parse file path to determine actual NVMe device
    // e.g., /dev/nvme0n1 -> namespace 0, /dev/nvme1n1 -> namespace 1
    if (backend->num_namespaces == 0) return -1;
    return file_index % backend->num_namespaces;
}

/**
 * Fetch a single range from a file
 */
static ssize_t hmll_spdk_fetch_range_impl(
    struct hmll *ctx,
    struct hmll_spdk *backend,
    const struct hmll_iobuf *dst,
    const struct hmll_range range,
    const int iofile
) {
    if (hmll_check(ctx->error)) return -1;

    const size_t size = hmll_range_size(range);
    size_t bytes_submitted = 0;
    size_t bytes_completed = 0;

    // Map file index to namespace
    int ns_index = hmll_spdk_map_file_to_namespace(backend, iofile);
    if (ns_index < 0) {
        ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
        return -1;
    }

    while (bytes_completed < size) {
        // Reclaim completed CUDA transfers
        bytes_completed += hmll_spdk_reclaim_cuda_slots(backend, dst->device);

        // Submit requests
        while (bytes_submitted < size) {
            int slot = hmll_spdk_slot_find_available(backend->slots);
            if (slot < 0) break;

            struct hmll_spdk_io_request *req = hmll_spdk_alloc_request(backend);
            if (!req) break;

            hmll_spdk_slot_set_busy(&backend->slots, slot);

            const size_t remaining = size - bytes_submitted;
            const size_t chunk_size = (remaining < HMLL_SPDK_BUFFER_SIZE) ? remaining : HMLL_SPDK_BUFFER_SIZE;

            void *buffer;
            if (dst->device == HMLL_DEVICE_CPU) {
                buffer = (char *)dst->ptr + bytes_submitted;
            }
#if defined(__HMLL_CUDA_ENABLED__)
            else if (dst->device == HMLL_DEVICE_CUDA) {
                // Use staging buffer for CUDA
                buffer = backend->staging_buffers[slot];
                struct hmll_spdk_cuda_context *dctx = backend->device_ctx;
                dctx[slot].offset = bytes_submitted;
            }
#endif
            else {
                ctx->error = HMLL_ERR(HMLL_ERR_UNSUPPORTED_DEVICE);
                return -1;
            }

            req->dst_buffer = (char *)dst->ptr + bytes_submitted;
            req->staging_buffer = (dst->device == HMLL_DEVICE_CUDA) ? buffer : NULL;
            req->offset = bytes_submitted;
            req->length = chunk_size;
            req->file_offset = range.start + bytes_submitted;
            req->ns_index = ns_index;
            req->slot = slot;
            req->backend = backend;
            req->completed = 0;

            // Register request in slot mapping
            backend->slot_requests[slot] = req;

            if (hmll_spdk_submit_read(backend, req, ns_index, buffer, req->file_offset, chunk_size) < 0) {
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                backend->slot_requests[slot] = NULL;
                hmll_spdk_free_request(backend, req);
                hmll_spdk_slot_set_available(&backend->slots, slot);
                return -1;
            }

            bytes_submitted += chunk_size;
        }

        // Process completions
        int num_completions = hmll_spdk_process_completions(backend, ns_index);
        if (num_completions < 0) {
            ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
            return -1;
        }

        // Check for completed requests and handle slot reclamation
        for (int slot = 0; slot < (int)HMLL_SPDK_QUEUE_DEPTH; slot++) {
            if (!hmll_spdk_slot_is_busy(backend->slots, slot)) continue;

            struct hmll_spdk_io_request *req = backend->slot_requests[slot];
            if (!req || !req->completed) continue;

            // Check if request encountered an error
            if (req->result < 0) {
                ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
                backend->slot_requests[slot] = NULL;
                hmll_spdk_free_request(backend, req);
                hmll_spdk_slot_set_available(&backend->slots, slot);
                return -1;
            }

            // For CPU device, data is already in destination buffer
            // For CUDA device, async memcpy was triggered in completion callback
            // Just track completion and free the slot for CPU
            if (dst->device == HMLL_DEVICE_CPU) {
                bytes_completed += req->length;
                backend->slot_requests[slot] = NULL;
                hmll_spdk_free_request(backend, req);
                hmll_spdk_slot_set_available(&backend->slots, slot);
                backend->outstanding_ios--;
            }
            // For CUDA, slot will be freed by hmll_spdk_reclaim_cuda_slots when async copy completes
        }
    }

    return (ssize_t)bytes_completed;
}

/**
 * Fetch multiple ranges (scatter-gather)
 */
static ssize_t hmll_spdk_fetchv_range_impl(
    struct hmll *ctx,
    struct hmll_spdk *backend,
    const struct hmll_iobuf *dsts,
    const struct hmll_range *ranges,
    const int iofile,
    const size_t n
) {
    // Similar implementation to io_uring fetchv, but using SPDK APIs
    // This would handle multiple ranges in parallel across multiple namespaces

    if (hmll_check(ctx->error)) return -1;

    size_t total_bytes = 0;

    // For simplicity, iterate over each range
    // A production implementation would interleave these for better performance
    for (size_t i = 0; i < n; i++) {
        ssize_t result = hmll_spdk_fetch_range_impl(ctx, backend, &dsts[i], ranges[i], iofile);
        if (result < 0) {
            return result;
        }
        total_bytes += result;
    }

    return (ssize_t)total_bytes;
}

/**
 * Wrapper functions matching hmll_loader interface
 */
static ssize_t hmll_spdk_fetch_range(
    struct hmll *ctx,
    void *fetcher,
    const int iofile,
    const struct hmll_iobuf *dst,
    const struct hmll_range range
) {
    if (hmll_check(ctx->error))
        return -1;

    return hmll_spdk_fetch_range_impl(ctx, fetcher, dst, range, iofile);
}

static ssize_t hmll_spdk_fetchv_range(
    struct hmll *ctx,
    void *fetcher,
    const int iofile,
    const struct hmll_iobuf *dsts,
    const struct hmll_range *ranges,
    const size_t n
) {
    if (hmll_check(ctx->error))
        return -1;

    return hmll_spdk_fetchv_range_impl(ctx, fetcher, dsts, ranges, iofile, n);
}

/**
 * SPDK probe callback - called for each NVMe controller found
 */
static bool hmll_spdk_probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                               struct spdk_nvme_ctrlr_opts *opts)
{
    HMLL_UNUSED(cb_ctx);
    HMLL_UNUSED(opts);

    printf("  Found NVMe controller: %s\n", trid->traddr);

    // Accept all NVMe controllers
    return true;
}

/**
 * Context for attach callback
 */
struct hmll_spdk_probe_ctx {
    struct hmll_spdk *backend;
    size_t ns_index;
};

/**
 * SPDK attach callback - called when a controller is attached
 */
static void hmll_spdk_attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                                 struct spdk_nvme_ctrlr *ctrlr,
                                 const struct spdk_nvme_ctrlr_opts *opts)
{
    HMLL_UNUSED(opts);

    printf("  Attaching controller: %s\n", trid->traddr);

    struct hmll_spdk_probe_ctx *probe_ctx = cb_ctx;
    if (!probe_ctx || !probe_ctx->backend) return;

    struct hmll_spdk *backend = probe_ctx->backend;

    // Iterate through all active namespaces on this controller
    int num_ns = spdk_nvme_ctrlr_get_num_ns(ctrlr);
    printf("    Controller has %d namespaces\n", num_ns);
    for (int nsid = 1; nsid <= num_ns; nsid++) {
        struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (!ns || !spdk_nvme_ns_is_active(ns)) {
            continue;
        }

        if (probe_ctx->ns_index >= backend->num_namespaces) {
            break;
        }

        struct hmll_spdk_ns_entry *entry = &backend->ns_entries[probe_ctx->ns_index];

        // Store namespace information
        entry->ctrlr = ctrlr;
        entry->ns = ns;
        entry->sector_size = spdk_nvme_ns_get_sector_size(ns);
        entry->num_sectors = spdk_nvme_ns_get_num_sectors(ns);

        // Allocate I/O queue pair for this namespace
        struct spdk_nvme_io_qpair_opts qpair_opts;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ctrlr, &qpair_opts, sizeof(qpair_opts));
        qpair_opts.io_queue_size = HMLL_SPDK_QUEUE_DEPTH;

        entry->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, &qpair_opts, sizeof(qpair_opts));
        if (!entry->qpair) {
            continue;
        }

        // Store transport address for debugging
        char trid_str[128];
        snprintf(trid_str, sizeof(trid_str), "nvme%u:%s", nsid, trid->traddr);
        entry->path = strdup(trid_str);

        probe_ctx->ns_index++;
    }
}

/**
 * Initialize SPDK backend
 */
struct hmll_error hmll_spdk_init(struct hmll *ctx, const enum hmll_device device) {
    if (hmll_check(ctx->error))
        return ctx->error;

    printf("Initializing SPDK environment...\n");

    // Initialize SPDK environment
    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "hmll_spdk";
    opts.shm_id = -1;  // Let SPDK choose

    // Memory allocation settings - allow small hugepage-less operation
    opts.mem_size = 64;  // 64 MB minimum (SPDK default is much higher)
    opts.hugepage_single_segments = false;
    opts.unlink_hugepage = true;
    opts.no_pci = false;  // Enable PCI device scanning

    printf("Attempting SPDK env init (this requires hugepages)...\n");
    int rc = spdk_env_init(&opts);
    if (rc < 0) {
        // SPDK init failed - likely no hugepages or no NVMe hardware
        printf("SPDK env init failed (rc=%d). Possible causes:\n", rc);
        printf("  - Hugepages not configured (see: cat /proc/meminfo | grep Huge)\n");
        printf("  - Insufficient permissions (may need root or vfio setup)\n");
        printf("  - No compatible NVMe devices found\n");
        ctx->error = HMLL_ERR(HMLL_ERR_UNSUPPORTED_PLATFORM);
        return ctx->error;
    }

    printf("SPDK environment initialized successfully\n");

    struct hmll_spdk *backend = calloc(1, sizeof(struct hmll_spdk));
    if (!backend) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        return ctx->error;
    }

    // Allocate namespace entries (maximum of 128, same as queue depth)
    backend->num_namespaces = HMLL_SPDK_QUEUE_DEPTH;
    backend->ns_entries = calloc(backend->num_namespaces, sizeof(struct hmll_spdk_ns_entry));
    if (!backend->ns_entries) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        goto cleanup;
    }

    // Probe for NVMe devices with callback context
    printf("Probing for NVMe devices...\n");
    struct hmll_spdk_probe_ctx probe_ctx = {
        .backend = backend,
        .ns_index = 0
    };

    int probe_rc = spdk_nvme_probe(NULL, &probe_ctx, hmll_spdk_probe_cb, hmll_spdk_attach_cb, NULL);
    printf("Probe completed with rc=%d\n", probe_rc);

    if (probe_rc != 0) {
        printf("NVMe probe failed\n");
        ctx->error = HMLL_ERR(HMLL_ERR_IO_ERROR);
        goto cleanup;
    }

    // Verify we found at least one namespace
    printf("Found %zu NVMe namespaces\n", probe_ctx.ns_index);
    if (probe_ctx.ns_index == 0) {
        // No NVMe devices found - this is expected if no NVMe hardware is available
        // or if NVMe devices are bound to kernel driver instead of userspace driver
        printf("No NVMe devices found. Possible causes:\n");
        printf("  - No NVMe hardware present\n");
        printf("  - NVMe devices bound to kernel driver (need vfio-pci or uio_pci_generic)\n");
        printf("  To bind NVMe to userspace driver, run SPDK's setup.sh script:\n");
        printf("    sudo scripts/setup.sh\n");
        printf("  Or manually: sudo dpdk-devbind.py -b vfio-pci <pci-address>\n");
        printf("Falling back to other backend\n");

        // Note: We cannot call spdk_env_fini() here because:
        // 1. It's not exposed in the public API
        // 2. SPDK environment is process-wide and should remain initialized
        // 3. Multiple backends might try to use SPDK

        ctx->error = HMLL_ERR(HMLL_ERR_UNSUPPORTED_PLATFORM);
        goto cleanup;
    }

    // Update actual namespace count
    backend->num_namespaces = probe_ctx.ns_index;
    printf("SPDK backend initialized with %zu namespaces\n", backend->num_namespaces);

    // Initialize request pool
    backend->request_pool = calloc(HMLL_SPDK_QUEUE_DEPTH, sizeof(struct hmll_spdk_io_request));
    if (!backend->request_pool) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        goto cleanup;
    }

    // Build free list
    for (size_t i = 0; i < HMLL_SPDK_QUEUE_DEPTH - 1; i++) {
        backend->request_pool[i].next = &backend->request_pool[i + 1];
    }
    backend->request_pool[HMLL_SPDK_QUEUE_DEPTH - 1].next = NULL;
    backend->free_requests = &backend->request_pool[0];

    // Allocate staging buffers
    backend->staging_buffers = calloc(HMLL_SPDK_QUEUE_DEPTH, sizeof(void *));
    if (!backend->staging_buffers) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        goto cleanup;
    }

    for (size_t i = 0; i < HMLL_SPDK_QUEUE_DEPTH; i++) {
        backend->staging_buffers[i] = spdk_dma_zmalloc(HMLL_SPDK_BUFFER_SIZE, 4096, NULL);
        if (!backend->staging_buffers[i]) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            goto cleanup;
        }
    }

    // Initialize CUDA contexts if needed
    if (device == HMLL_DEVICE_CUDA) {
#if defined(__HMLL_CUDA_ENABLED__)
        struct hmll_spdk_cuda_context *cuda_ctx = calloc(HMLL_SPDK_QUEUE_DEPTH, sizeof(struct hmll_spdk_cuda_context));
        if (!cuda_ctx) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            goto cleanup;
        }
        backend->device_ctx = cuda_ctx;

        for (int i = 0; i < (int)HMLL_SPDK_QUEUE_DEPTH; ++i) {
            cuda_ctx[i].slot = i;
            CHECK_CUDA(cudaStreamCreateWithFlags(&cuda_ctx[i].stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(&cuda_ctx[i].done, cudaEventDisableTiming));
        }
#else
        ctx->error = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
        goto cleanup;
#endif
    }

    // Setup the loader
    if (ctx->fetcher == NULL) {
        ctx->fetcher = calloc(1, sizeof(struct hmll_loader));
        ctx->fetcher->device = device;
        ctx->fetcher->backend_impl_ = backend;
        ctx->fetcher->fetch_range_impl_ = hmll_spdk_fetch_range;
        ctx->fetcher->fetchv_range_impl_ = hmll_spdk_fetchv_range;
    }

    return HMLL_OK;

cleanup:
    hmll_spdk_cleanup(backend);
    free(backend);
    return ctx->error;
}

/**
 * Cleanup SPDK backend
 */
void hmll_spdk_cleanup(struct hmll_spdk *backend) {
    if (!backend) return;

    // Free staging buffers
    if (backend->staging_buffers) {
        for (size_t i = 0; i < HMLL_SPDK_QUEUE_DEPTH; i++) {
            if (backend->staging_buffers[i]) {
                spdk_dma_free(backend->staging_buffers[i]);
            }
        }
        free(backend->staging_buffers);
    }

    // Free request pool
    if (backend->request_pool) {
        free(backend->request_pool);
    }

    // Cleanup namespaces
    if (backend->ns_entries) {
        for (size_t i = 0; i < backend->num_namespaces; i++) {
            if (backend->ns_entries[i].qpair) {
                spdk_nvme_ctrlr_free_io_qpair(backend->ns_entries[i].qpair);
            }
            if (backend->ns_entries[i].path) {
                free(backend->ns_entries[i].path);
            }
        }
        free(backend->ns_entries);
    }

#if defined(__HMLL_CUDA_ENABLED__)
    if (backend->device_ctx) {
        struct hmll_spdk_cuda_context *cuda_ctx = backend->device_ctx;
        for (int i = 0; i < (int)HMLL_SPDK_QUEUE_DEPTH; ++i) {
            cudaStreamDestroy(cuda_ctx[i].stream);
            cudaEventDestroy(cuda_ctx[i].done);
        }
        free(backend->device_ctx);
    }
#endif
}
