//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>

#include "hmll/status.h"
#include "hmll/types.h"

hmll_status_t hmll_context_free(const hmll_context_t *ctx)
{
    if (ctx->table.names) {
        const auto names = ctx->table.names;
        for (size_t i = 0; i < ctx->num_tensors; ++i)
            free(names + i);

        free(names);
    }

    if (ctx->table.tensors) {
        const auto tensors = ctx->table.tensors;
        for (size_t i = 0; i < ctx->num_tensors; ++i)
            free(tensors + i);

        free(tensors);
    }

    return HMLL_SUCCEEDED;
}
