#include "stdint.h"
#include "hmll/types.h"

// uint8_t hmll_nbits(const hmll_tensor_data_type_t dtype)
// {
//     switch (dtype)
//     {
//     case HMLL_DTYPE_FLOAT4:
//     case HMLL_DTYPE_SIGNED_INT4:
//     case HMLL_DTYPE_UNSIGNED_INT4:
//         return 4;
//     case HMLL_DTYPE_FLOAT6_E2M3:
//     case HMLL_DTYPE_FLOAT6_E3M2:
//         return 6;
//     case HMLL_DTYPE_BOOL:
//     case HMLL_DTYPE_FLOAT8_E4M3:
//     case HMLL_DTYPE_FLOAT8_E5M2:
//     case HMLL_DTYPE_FLOAT8_E8M0:
//     case HMLL_DTYPE_SIGNED_INT8:
//     case HMLL_DTYPE_UNSIGNED_INT8:
//         return 8;
//     case HMLL_DTYPE_BFLOAT16:
//     case HMLL_DTYPE_FLOAT16:
//     case HMLL_DTYPE_SIGNED_INT16:
//     case HMLL_DTYPE_UNSIGNED_INT16:
//         return 16;
//     case HMLL_DTYPE_FLOAT32:
//     case HMLL_DTYPE_SIGNED_INT32:
//     case HMLL_DTYPE_UNSIGNED_INT32:
//         return 32;
//     case HMLL_DTYPE_COMPLEX:
//     case HMLL_DTYPE_SIGNED_INT64:
//     case HMLL_DTYPE_UNSIGNED_INT64:
//         return 64;
//     default:
//         return 0;
//     }
// }
//
// size_t hmll_numel(const hmll_tensor_specs_t *specs)
// {
//     if (specs->rank > HMLL_MAX_TENSOR_RANK) __builtin_unreachable();
//
//     size_t numel = 1;
//     for (size_t i = 0; i < specs->rank; ++i)
//         numel *= specs->shape[i];
//
//     return numel;
// }
