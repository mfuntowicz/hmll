//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>
#include <string.h>

#include "hmll/hmll.h"
#include "hmll/status.h"
#include "hmll/types.h"


hmll_status_t hmll_get_tensor_specs(const hmll_context_t *ctx, const char *name, hmll_tensor_specs_t **specs)
{
    if (!ctx || ctx->num_tensors == 0) return (hmll_status_t){HMLL_EMPTY_TABLE, "Empty tensor table"};

    char **names = ctx->table.names;
    for (size_t i = 0; i < ctx->num_tensors; ++i) {
        if (strcmp(name, names[i]) == 0) {
            *specs = ctx->table.tensors + i;
            return HMLL_SUCCEEDED;
        }
    }

    return (hmll_status_t){HMLL_TENSOR_NOT_FOUND, "Tensor not found"};
}


void hmll_destroy(const hmll_context_t *ctx)
{
    for (size_t i = 0; i < ctx->num_tensors; ++i)
    {
        if (ctx->table.names && ctx->table.names[i]) free(ctx->table.names[i]);
        if (ctx->table.tensors) hmll_tensor_specs_free(ctx->table.tensors + i);
    }

    // if (ctx->table.names) free(ctx->table.names);
    // if (ctx->table.tensors) free(ctx->table.tensors);
}


void hmll_tensor_specs_free(hmll_tensor_specs_t *specs)
{
    if (specs) {
        if (specs->shape) free(specs->shape);
        specs->rank = specs->start = specs->end = 0;
    }
}
