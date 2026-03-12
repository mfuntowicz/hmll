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
    if (!ctx) return;

    if (ctx->fetcher) {
        ctx->fetcher->backend_free(ctx->fetcher->backend_impl_);
        free(ctx->fetcher);
        ctx->fetcher = NULL;
    }
}
