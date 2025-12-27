#include "stdint.h"
#include "hmll/types.h"
#include "hmll/tracing.h"

uint8_t hmll_sizeof(const hmll_tensor_data_type_t dtype)
{
    switch (dtype)
    {
    case HMLL_DTYPE_BFLOAT16:
    case HMLL_DTYPE_FLOAT16:
        return 2;
    default:
        return 4;
    }
}

size_t hmll_numel(const hmll_tensor_specs_t *specs)
{
    if (specs->rank > HMLL_MAX_TENSOR_RANK) __builtin_unreachable();

    HMLL_ZONE_START(hmll_numel)
    size_t numel = 1;
    for (size_t i = 0; i < specs->rank; ++i)
        numel *= specs->shape[i];

    HMLL_ZONE_END_SUCCESS(hmll_numel);
    return numel;
}
