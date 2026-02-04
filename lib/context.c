//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>
#include <string.h>
#include "hmll/hmll.h"

void hmll_destroy(struct hmll *ctx)
{
    if (ctx) {
        if (ctx->fetcher) {
            // Note: For mmap backend, we do NOT free backend_impl_ here.
            // The Rust wrapper manages mmap lifetime via Arc reference counting.
            // It will call hmll_mmap_free() when all references are dropped.
            // We only free the fetcher struct itself, not the backend data.

            // TODO(mfuntowicz): handle io_uring cleanup
            free(ctx->fetcher);
            ctx->fetcher = NULL;
        }
    }
}

struct hmll_error hmll_clone_context(struct hmll *dst, const struct hmll *src)
{
    if (!src || !dst) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    memcpy(dst, src, sizeof(struct hmll));

    // Reset error state for the new context
    dst->error = HMLL_OK;
    dst->fetcher = NULL;

    return HMLL_OK;
}

