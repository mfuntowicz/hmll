//
// Created by mfuntowicz on 12/11/25.
//
#include "hmll/types.h"

char *hmll_strerr(const enum hmll_error_code status)
{
    switch (status)
    {
    case HMLL_ERR_FILE_NOT_FOUND: return "File not found";
    case HMLL_ERR_ALLOCATION_FAILED: return "Failed to allocate memory";
    case HMLL_ERR_TABLE_EMPTY: return "No tensors found while reading the file";
    case HMLL_ERR_TENSOR_NOT_FOUND: return "Tensor not found in the known tensors table";
    case HMLL_ERR_CUDA_NOT_ENABLED: return "CUDA not enabled";
    case HMLL_ERR_CUDA_NO_DEVICE: return "No CUDA devices found";
    default: return "Unknown error happened. Please open an issues at https://github.com/mfuntowicz/hmll/issues.";
    }
}
