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
            // TODO
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

