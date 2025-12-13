//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>
#include <string.h>

#include "hmll/hmll.h"

unsigned int hmll_success(const enum hmll_error_code errno)
{
    return errno == HMLL_SUCCESS;
}

unsigned int hmll_has_error(const enum hmll_error_code errno)
{
    return errno != HMLL_SUCCESS;
}

enum hmll_error_code hmll_get_error(const struct hmll_context *ctx)
{
    return ctx->error;
}

struct hmll_tensor_specs hmll_get_tensor_specs(struct hmll_context *ctx, const char *name)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return (struct hmll_tensor_specs){0};

    if (ctx->num_tensors == 0) {
        ctx->error = HMLL_ERR_ALLOCATION_FAILED;
        return (struct hmll_tensor_specs){0};
    }

    char **names = ctx->table.names;
    for (size_t i = 0; i < ctx->num_tensors; ++i) {
        if (strcmp(name, names[i]) == 0) {
            return *(ctx->table.tensors + i);
        }
    }

    ctx->error = HMLL_ERR_TENSOR_NOT_FOUND;
    return (struct hmll_tensor_specs){0};
}


// void hmll_destroy(const hmll_context_t *ctx)
// {
//     for (size_t i = 0; i < ctx->num_tensors; ++i)
//     {
//         if (ctx->table.names && ctx->table.names[i]) free(ctx->table.names[i]);
//         if (ctx->table.tensors) hmll_tensor_specs_free(ctx->table.tensors + i);
//     }
//
//     // if (ctx->table.names) free(ctx->table.names);
//     // if (ctx->table.tensors) free(ctx->table.tensors);
// }


// void hmll_tensor_specs_free(hmll_tensor_specs_t *specs)
// {
//     if (specs) {
//         if (specs->shape) free(specs->shape);
//         specs->rank = specs->start = specs->end = 0;
//     }
// }
