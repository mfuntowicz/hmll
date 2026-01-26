//
// Created by mfuntowicz on 12/11/25.
//
#include <string.h>

#include "hmll/hmll.h"
#include "hmll/types.h"

#if defined(_WIN32)
// Thread-local buffer for strerror on Windows
static __declspec(thread) char strerror_buf[256];
#endif

const char *hmll_strerr(const struct hmll_error err)
{
    if (hmll_error_is_os_error(err)) {
#if defined(_WIN32)
        if (strerror_s(strerror_buf, sizeof(strerror_buf), err.sys_err) == 0)
            return strerror_buf;
        return "Unknown system error";
#else
        return strerror(err.sys_err);
#endif
    }

    if (hmll_error_is_lib_error(err))
    {
        switch (err.code)
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

    return "No error";
}

unsigned char hmll_error_is_os_error(const struct hmll_error err)
{
    return err.sys_err != HMLL_ERR_SUCCESS;
}

unsigned char hmll_error_is_lib_error(const struct hmll_error err)
{
    return err.code != HMLL_ERR_SUCCESS && err.code != HMLL_ERR_SYSTEM;
}
