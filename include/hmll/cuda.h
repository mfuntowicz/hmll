//
// Created by mfuntowicz on 12/18/25.
//

#ifndef HMLL_CUDA_HPP
#define HMLL_CUDA_HPP

#include <cuda.h>

/// Taken from Morpheus
/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 */
#define CHECK_CUDA(call)                \
do {                                    \
    cudaError_t const status = (call);  \
    if (cudaSuccess != status)          \
        cudaGetLastError();             \
} while (0);

int hmll_cuda_device_count(void);
int hmll_cuda_allocation_flags(void);
#endif //HMLL_CUDA_HPP