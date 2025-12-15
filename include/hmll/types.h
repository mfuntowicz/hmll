#ifndef HMLL_TYPES_H
#define HMLL_TYPES_H

#include <stdint.h>
#include <stdio.h>

enum hmll_error_code
{
    HMLL_ERR_SUCCESS = 0,

    HMLL_ERR_FILE_NOT_FOUND = -1,
    HMLL_ERR_FILE_EMPTY = -2,
    HMLL_ERR_MMAP_FAILED = -3,
    HMLL_ERR_UNSUPPORTED_FILE_FORMAT = -4,
    HMLL_ERR_UNSUPPORTED_DEVICE = -5,
    HMLL_ERR_ALLOCATION_FAILED = -6,
    HMLL_ERR_TABLE_EMPTY = -7,
    HMLL_ERR_TENSOR_NOT_FOUND = -8,
    HMLL_ERR_BUFFER_TOO_SMALL = -9,

    HMLL_ERR_INVALID_RANGE = -10,
    HMLL_ERR_IO_ERROR = -11,
    HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER = -12,
    HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER = -13,

    HMLL_ERR_UNKNOWN_DTYPE = -20,
};
typedef enum hmll_error_code hmll_error_code_t;

enum hmll_source_kind
{
    HMLL_SOURCE_UNDEFINED,
    HMLL_SOURCE_FD,
    HMLL_SOURCE_MMAP
};
typedef enum hmll_source_kind hmll_source_kind_t;

struct hmll_source
{
#if defined(__linux) || defined(__unix) || defined(APPLE)
    int fd;
    char *content;
#endif
    size_t size;
    enum hmll_source_kind kind;
};
typedef struct hmll_source hmll_source_t;

enum hmll_flags
{
    HMLL_MMAP = 1 << 0,
    HMLL_SKIP_METADATA = 1 << 1
};
typedef enum hmll_flags hmll_flags_t;

enum hmll_file_kind
{
    HMLL_SAFETENSORS,
    HMLL_GGUF
};
typedef enum hmll_file_kind hmll_file_kind_t;

enum hmll_tensor_data_type
{
    HMLL_DTYPE_BFLOAT16,
    HMLL_DTYPE_FLOAT16,
    HMLL_DTYPE_FLOAT32,
    HMLL_DTYPE_UNKNOWN
};
typedef enum hmll_tensor_data_type hmll_tensor_data_type_t;

struct hmll_tensor_specs
{
    size_t start;
    size_t end;
    size_t *shape;
    uint8_t rank;
    enum hmll_tensor_data_type dtype;
};
typedef struct hmll_tensor_specs hmll_tensor_specs_t;

struct hmll_table
{
    struct hmll_tensor_specs *tensors;
    char **names;
};
typedef struct hmll_table hmll_table_t;

enum hmll_device
{
    HMLL_DEVICE_CPU,
};
typedef enum hmll_device hmll_device_t;


struct hmll_device_buffer
{
    void *ptr;
    size_t size;
    hmll_device_t device;
};
typedef struct hmll_device_buffer hmll_device_buffer_t;

struct hmll_context {
    struct hmll_source source;
    struct hmll_table table;
    size_t num_tensors;
    enum hmll_file_kind kind;
    enum hmll_error_code error;
};
typedef struct hmll_context hmll_context_t;

#endif // HMLL_TYPES_H
