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

int hmll_find_by_name(const struct hmll_context *ctx, const char *name)
{
    char **names = ctx->table.names;
    for (size_t i = 0; i < ctx->num_tensors; ++i) {
        if (strcmp(name, names[i]) == 0) {
            return (int)i;
        }
    }

    return -1;
}

int hmll_contains(const struct hmll_context *ctx, const char *name)
{
    return hmll_find_by_name(ctx, name) >= 0;
}

struct hmll_tensor_lookup_result hmll_get_tensor_specs(const struct hmll_context *ctx, const char *name)
{
    struct hmll_tensor_lookup_result result = {{0}, 0, HMLL_FALSE };
    if (!hmll_has_error(hmll_get_error(ctx))) {
        const int index = hmll_find_by_name(ctx, name);
        if (index >= 0) {
            result.found = 1;
            result.index = index;
            result.specs = ctx->table.tensors[index];
        }
    }

    return result;
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
