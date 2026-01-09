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

struct hmll_error hmll_clone_context(const struct hmll *src, struct hmll *dst)
{
    if (!src || !dst) {
        return HMLL_ERR(HMLL_ERR_INVALID_RANGE);
    }

    // Copy shared resources (fetcher, sources)
    dst->fetcher = src->fetcher;
    dst->sources = src->sources;
    dst->num_sources = src->num_sources;

    // Reset error state for the new context
    dst->error = HMLL_OK;

    return HMLL_OK;
}

