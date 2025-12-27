//
// Created by mfuntowicz on 12/1/25.
//

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
    TracyCZone(find_by_name, __HMLL_PROFILE_ENABLED__);
    char **names = ctx->table.names;
    for (size_t i = 0; i < ctx->num_tensors; ++i) {
        if (strcmp(name, names[i]) == 0) {
            TracyCZoneEnd(find_by_name);
            return (int)i;
        }
    }

    TracyCZoneEnd(find_by_name);
    return -1;
}

int hmll_contains(const struct hmll_context *ctx, const char *name)
{
    return hmll_find_by_name(ctx, name) >= 0;
}

struct hmll_tensor_lookup_result hmll_get_tensor_specs(const struct hmll_context *ctx, const char *name)
{
    TracyCZone(get_tensor_specs, __HMLL_PROFILE_ENABLED__);
    struct hmll_tensor_lookup_result result = {{0}, 0, HMLL_FALSE };
    if (!hmll_has_error(hmll_get_error(ctx))) {
        const int index = hmll_find_by_name(ctx, name);
        if (index >= 0) {
            result.found = 1;
            result.index = index;
            result.specs = ctx->table.tensors[index];
        }
    }
    TracyCZoneEnd(get_tensor_specs);
    return result;
}
