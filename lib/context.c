//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>
#include "hmll/hmll.h"

#if defined(_WIN32)
#include "hmll/win32/backend/mmap.h"
#elif defined(__unix__) || defined(__linux__) || defined(__APPLE__)
#include "hmll/unix/backend/mmap.h"
// TODO(mfuntowicz): include io_uring cleanup when implemented
#endif

void hmll_destroy(struct hmll *ctx)
{
    if (ctx) {
        if (ctx->fetcher) {
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
            if (ctx->fetcher->kind == HMLL_FETCHER_MMAP && ctx->fetcher->backend_impl_) {
                hmll_mmap_free(ctx->fetcher->backend_impl_);
            }
#endif
#if defined(__linux__)
            // TODO(mfuntowicz): handle io_uring cleanup when needed
#endif
            free(ctx->fetcher);
            ctx->fetcher = NULL;
        }
    }
}
