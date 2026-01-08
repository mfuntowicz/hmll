#ifndef HMLL_TYPES_H
#define HMLL_TYPES_H

#include <stdio.h>
#include <stdint.h>

#ifndef HMLL_MAX_TENSOR_RANK
#define HMLL_MAX_TENSOR_RANK 5
#endif

#if defined(__linux) || defined(__unix)
#include "unix/file.h"
#endif

struct hmll;

enum hmll_status_code {
    HMLL_ERR_SUCCESS = 0,

    HMLL_ERR_UNSUPPORTED_PLATFORM,
    HMLL_ERR_UNSUPPORTED_FILE_FORMAT,
    HMLL_ERR_UNSUPPORTED_DEVICE,
    HMLL_ERR_ALLOCATION_FAILED,
    HMLL_ERR_TABLE_EMPTY,
    HMLL_ERR_TENSOR_NOT_FOUND,
    HMLL_ERR_INVALID_RANGE,

    HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED,
    HMLL_ERR_BUFFER_TOO_SMALL,

    HMLL_ERR_IO_ERROR,
    HMLL_ERR_FILE_NOT_FOUND,
    HMLL_ERR_FILE_EMPTY,
    HMLL_ERR_MMAP_FAILED,
    HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED,

    HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER,
    HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER,
    HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX,

    HMLL_ERR_CUDA_NOT_ENABLED,
    HMLL_ERR_CUDA_NO_DEVICE,

    HMLL_ERR_SYSTEM,
    HMLL_ERR_UNKNOWN_DTYPE,
};
typedef enum hmll_status_code hmll_error_code_t;


struct hmll_error {
    enum hmll_status_code code;
    int sys_err;
};
typedef struct hmll_error hmll_error_t;

enum hmll_dtype
{
    HMLL_DTYPE_BOOL,
    HMLL_DTYPE_BFLOAT16,
    HMLL_DTYPE_COMPLEX,
    HMLL_DTYPE_FLOAT4,
    HMLL_DTYPE_FLOAT6_E2M3,
    HMLL_DTYPE_FLOAT6_E3M2,
    HMLL_DTYPE_FLOAT8_E5M2,
    HMLL_DTYPE_FLOAT8_E4M3,
    HMLL_DTYPE_FLOAT8_E8M0,
    HMLL_DTYPE_FLOAT16,
    HMLL_DTYPE_FLOAT32,
    HMLL_DTYPE_SIGNED_INT4,
    HMLL_DTYPE_SIGNED_INT8,
    HMLL_DTYPE_SIGNED_INT16,
    HMLL_DTYPE_SIGNED_INT32,
    HMLL_DTYPE_SIGNED_INT64,
    HMLL_DTYPE_UNSIGNED_INT4,
    HMLL_DTYPE_UNSIGNED_INT8,
    HMLL_DTYPE_UNSIGNED_INT16,
    HMLL_DTYPE_UNSIGNED_INT32,
    HMLL_DTYPE_UNSIGNED_INT64,
    HMLL_DTYPE_UNKNOWN
};
typedef enum hmll_dtype hmll_dtype_t;

#ifdef __HMLL_TENSORS_ENABLED__
struct hmll_tensor_specs
{
    size_t shape[HMLL_MAX_TENSOR_RANK];
    size_t start;
    size_t end;
    uint8_t rank;
    enum hmll_dtype dtype;
};
typedef struct hmll_tensor_specs hmll_tensor_specs_t;

struct hmll_lookup_result
{
    unsigned short index;
    unsigned short file;
    const struct hmll_tensor_specs *specs;
};
typedef struct hmll_lookup_result hmll_lookup_result_t;

struct hmll_registry
{
    size_t num_tensors;
    struct hmll_tensor_specs *tensors;
    unsigned short *indexes;
    char **names;
};
typedef struct hmll_registry hmll_registry_t;
#endif

enum hmll_device
{
    HMLL_DEVICE_CPU,
    HMLL_DEVICE_CUDA
};
typedef enum hmll_device hmll_device_t;

struct hmll_iobuf
{
    size_t size;
    void *ptr;
    enum hmll_device device;
};
typedef struct hmll_iobuf hmll_iobuf_t;

struct hmll_range {
    size_t start;
    size_t end;
};
typedef struct hmll_range hmll_range_t;

struct hmll {
    struct hmll_loader *fetcher;
    const struct hmll_source *sources;
    size_t num_sources;
    struct hmll_error error;
};
typedef struct hmll hmll_t;
#endif // HMLL_TYPES_H
