#include "stdint.h"
#include "hmll/types.h"

uint8_t hmll_get_dtype_nbytes(const hmll_tensor_data_type_t dtype)
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