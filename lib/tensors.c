#include "stdint.h"
#include "hmll/types.h"

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
    if (!specs || specs->rank == 0) return 0;

    size_t numel = 1;
    for (size_t i = 0; i < specs->rank; ++i)
        numel *= specs->shape[i];

    return numel;
}
