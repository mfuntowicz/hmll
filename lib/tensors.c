#include <stdlib.h>
#include <string.h>
#include "hmll/hmll.h"

void hmll_free_registry(struct hmll_registry *reg)
{
    if (!reg) return;

    // Free each tensor name string
    if (reg->names) {
        for (size_t i = 0; i < reg->num_tensors; ++i) {
            free(reg->names[i]);
        }
        free(reg->names);
        reg->names = NULL;
    }

    // Free tensors array
    if (reg->tensors) {
        free(reg->tensors);
        reg->tensors = NULL;
    }

    // Free indexes array
    if (reg->indexes) {
        free(reg->indexes);
        reg->indexes = NULL;
    }

    reg->num_tensors = 0;
}

int hmll_find_by_name(const struct hmll *ctx, const struct hmll_registry *reg, const char *name)
{
    if (hmll_check(ctx->error)) return -1;

    char **names = reg->names;
    for (size_t i = 0; i < reg->num_tensors; ++i) {
        if (strcmp(name, names[i]) == 0)
            return (int)i;
    }

    return -1;
}

unsigned char hmll_contains(const struct hmll *ctx, const struct hmll_registry *reg, const char *name)
{
    return hmll_find_by_name(ctx, reg, name) >= 0;
}

struct hmll_lookup_result hmll_lookup_tensor(const struct hmll *ctx, const struct hmll_registry *reg, const char *name)
{
    if (hmll_check(ctx->error)) return (struct hmll_lookup_result){0};

    struct hmll_lookup_result result = {0};
    const int index = hmll_find_by_name(ctx, reg, name);
    if (index >= 0) {
        result.index = (short)index;
        result.file = reg->indexes[index];
        result.specs = reg->tensors + index;
    }

    return result;
}

uint8_t hmll_nbits(const enum hmll_dtype dtype)
{
    switch (dtype)
    {
    case HMLL_DTYPE_FLOAT4:
    case HMLL_DTYPE_SIGNED_INT4:
    case HMLL_DTYPE_UNSIGNED_INT4:
        return 4;
    case HMLL_DTYPE_FLOAT6_E2M3:
    case HMLL_DTYPE_FLOAT6_E3M2:
        return 6;
    case HMLL_DTYPE_BOOL:
    case HMLL_DTYPE_FLOAT8_E4M3:
    case HMLL_DTYPE_FLOAT8_E5M2:
    case HMLL_DTYPE_FLOAT8_E8M0:
    case HMLL_DTYPE_SIGNED_INT8:
    case HMLL_DTYPE_UNSIGNED_INT8:
        return 8;
    case HMLL_DTYPE_BFLOAT16:
    case HMLL_DTYPE_FLOAT16:
    case HMLL_DTYPE_SIGNED_INT16:
    case HMLL_DTYPE_UNSIGNED_INT16:
        return 16;
    case HMLL_DTYPE_FLOAT32:
    case HMLL_DTYPE_SIGNED_INT32:
    case HMLL_DTYPE_UNSIGNED_INT32:
        return 32;
    case HMLL_DTYPE_COMPLEX:
    case HMLL_DTYPE_SIGNED_INT64:
    case HMLL_DTYPE_UNSIGNED_INT64:
        return 64;
    default:
        return 0;
    }
}

size_t hmll_nbytes(const struct hmll_tensor_specs *specs)
{
    if (!specs) return 0;

    const size_t numel = hmll_numel(specs);
    const size_t nbits = hmll_nbits(specs->dtype);
    return numel * (nbits / 8);
}

size_t hmll_numel(const struct hmll_tensor_specs *specs)
{
    if (!specs) return 0;
    if (specs->rank > HMLL_MAX_TENSOR_RANK) __builtin_unreachable();

    size_t numel = 1;
    for (size_t i = 0; i < specs->rank; ++i)
        numel *= specs->shape[i];

    return numel;
}