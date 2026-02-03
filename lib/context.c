//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>
#include <string.h>
#include "hmll/hmll.h"

#if defined(_WIN32)
#include "hmll/win32/backend/mmap.h"
#elif defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/backend/mmap.h"
#endif

void hmll_destroy(struct hmll *ctx)
{
    if (ctx) {
        if (ctx->fetcher) {
#if defined(_WIN32)
            if (ctx->fetcher->kind == HMLL_FETCHER_MMAP && ctx->fetcher->backend_impl_) {
                // Release the loader's reference to the mmap.
                // If no views are outstanding, this will unmap and free.
                // Otherwise, cleanup happens when last view is freed.
                hmll_mmap_release(ctx->fetcher->backend_impl_);
            }
#elif defined(__unix__) || defined(__APPLE__)
            if (ctx->fetcher->kind == HMLL_FETCHER_MMAP && ctx->fetcher->backend_impl_) {
                // Release the loader's reference to the mmap.
                // If no views are outstanding, this will munmap and free.
                // Otherwise, cleanup happens when last view is freed.
                hmll_mmap_release(ctx->fetcher->backend_impl_);
            }
            // TODO(mfuntowicz): handle io_uring cleanup
#endif
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

