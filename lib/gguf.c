//
// Created by mfuntowicz on 2/5/26.
//
#include <string.h>
#include <stdlib.h>

#include "hmll/hmll.h"

#define GGUF_HEADER 0x46554747
#define GGUF_DEFAULT_ALIGNMENT 32

// GGUF metadata value types
enum gguf_metadata_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// GGUF/GGML tensor types - mapping to hmll_dtype
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
};

// Skip a GGUF metadata value and return the new offset
static size_t skip_metadata_value(const unsigned char *content, size_t offset, uint32_t value_type)
{
    switch (value_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return offset + 1;

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return offset + 2;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return offset + 4;

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return offset + 8;

        case GGUF_TYPE_STRING: {
            uint64_t len = 0;
            memcpy(&len, content + offset, sizeof(len));
            return offset + sizeof(len) + len;
        }

        case GGUF_TYPE_ARRAY: {
            // Array: type (4 bytes) + count (8 bytes) + elements
            uint32_t arr_type = 0;
            memcpy(&arr_type, content + offset, sizeof(arr_type));
            offset += sizeof(arr_type);

            uint64_t count = 0;
            memcpy(&count, content + offset, sizeof(count));
            offset += sizeof(count);

            // Skip each element
            for (uint64_t i = 0; i < count; ++i) {
                offset = skip_metadata_value(content, offset, arr_type);
            }
            return offset;
        }

        default:
            return offset;
    }
}

// Map GGML type to hmll_dtype
static enum hmll_dtype ggml_type_to_hmll_dtype(uint32_t ggml_type)
{
    switch (ggml_type) {
        case GGML_TYPE_F32:     return HMLL_DTYPE_FLOAT32;
        case GGML_TYPE_F16:     return HMLL_DTYPE_FLOAT16;
        case GGML_TYPE_Q4_0:    return HMLL_DTYPE_Q4_0;
        case GGML_TYPE_Q4_1:    return HMLL_DTYPE_Q4_1;
        case GGML_TYPE_Q5_0:    return HMLL_DTYPE_Q5_0;
        case GGML_TYPE_Q5_1:    return HMLL_DTYPE_Q5_1;
        case GGML_TYPE_Q8_0:    return HMLL_DTYPE_Q8_0;
        case GGML_TYPE_Q8_1:    return HMLL_DTYPE_Q8_1;
        case GGML_TYPE_Q2_K:    return HMLL_DTYPE_Q2_K;
        case GGML_TYPE_Q3_K:    return HMLL_DTYPE_Q3_K;
        case GGML_TYPE_Q4_K:    return HMLL_DTYPE_Q4_K;
        case GGML_TYPE_Q5_K:    return HMLL_DTYPE_Q5_K;
        case GGML_TYPE_Q6_K:    return HMLL_DTYPE_Q6_K;
        case GGML_TYPE_Q8_K:    return HMLL_DTYPE_Q8_K;
        case GGML_TYPE_IQ2_XXS: return HMLL_DTYPE_IQ2_XXS;
        case GGML_TYPE_IQ2_XS:  return HMLL_DTYPE_IQ2_XS;
        case GGML_TYPE_IQ3_XXS: return HMLL_DTYPE_IQ3_XXS;
        case GGML_TYPE_IQ1_S:   return HMLL_DTYPE_IQ1_S;
        case GGML_TYPE_IQ4_NL:  return HMLL_DTYPE_IQ4_NL;
        case GGML_TYPE_IQ3_S:   return HMLL_DTYPE_IQ3_S;
        case GGML_TYPE_IQ2_S:   return HMLL_DTYPE_IQ2_S;
        case GGML_TYPE_IQ4_XS:  return HMLL_DTYPE_IQ4_XS;
        case GGML_TYPE_I8:      return HMLL_DTYPE_SIGNED_INT8;
        case GGML_TYPE_I16:     return HMLL_DTYPE_SIGNED_INT16;
        case GGML_TYPE_I32:     return HMLL_DTYPE_SIGNED_INT32;
        default:                return HMLL_DTYPE_UNKNOWN;
    }
}

size_t hmll_gguf_populate_registry(struct hmll *ctx, struct hmll_registry *reg, struct hmll_source source, size_t fid, size_t offset)
{
    HMLL_UNUSED(fid);
    HMLL_UNUSED(offset);
    if (hmll_check(ctx->error)) return 0;
    if (!source.content) return 0;

    size_t num_discovered_tensors = 0;
    const unsigned char *content = source.content;
    size_t current_offset = 0;

    // Read header (4 bytes)
    uint32_t header = 0;
    memcpy(&header, content + current_offset, sizeof(uint32_t));
    current_offset += sizeof(uint32_t);

    if (header != GGUF_HEADER) {
        ctx->error = HMLL_ERR(HMLL_ERR_GGUF_INVALID_HEADER);
        goto exit;
    }

    // Read version (4 bytes)
    uint32_t version = 0;
    memcpy(&version, content + current_offset, sizeof(uint32_t));
    current_offset += sizeof(uint32_t);

    if (version != 3) {
        ctx->error = HMLL_ERR(HMLL_ERR_GGUF_UNSUPPORTED_GGUF_VERSION);
        goto exit;
    }

    // Read tensor count (8 bytes)
    uint64_t num_tensors = 0;
    memcpy(&num_tensors, content + current_offset, sizeof(uint64_t));
    current_offset += sizeof(uint64_t);

    if (num_tensors <= 0) {
        ctx->error = HMLL_ERR(HMLL_ERR_TABLE_EMPTY);
        goto exit;
    }

    // Read metadata count (8 bytes)
    uint64_t num_metadata = 0;
    memcpy(&num_metadata, content + current_offset, sizeof(uint64_t));
    current_offset += sizeof(uint64_t);

    // Allocate registry arrays
    reg->num_tensors = num_tensors;
    reg->tensors = calloc(num_tensors, sizeof(struct hmll_tensor_specs));
    reg->names = calloc(num_tensors, sizeof(char*));
    reg->indexes = calloc(num_tensors, sizeof(unsigned short));

    if (!reg->tensors || !reg->names || !reg->indexes) {
        ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
        goto exit;
    }

    // Skip all metadata entries
    for (uint64_t i = 0; i < num_metadata; ++i) {
        uint64_t key_len = 0;
        memcpy(&key_len, content + current_offset, sizeof(key_len));

        current_offset += sizeof(key_len);
        current_offset += key_len;

        uint32_t value_type = 0;
        memcpy(&value_type, content + current_offset, sizeof(value_type));
        current_offset += sizeof(value_type);

        current_offset = skip_metadata_value(content, current_offset, value_type);
    }

    for (uint64_t i = 0; i < num_tensors; ++i) {
        uint64_t name_len = 0;
        memcpy(&name_len, content + current_offset, sizeof(name_len));
        current_offset += sizeof(name_len);

        reg->names[i] = malloc(name_len + 1);
        if (!reg->names[i]) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            goto exit;
        }
        memcpy(reg->names[i], content + current_offset, name_len);
        reg->names[i][name_len] = '\0';
        current_offset += name_len;

        uint32_t n_dims = 0;
        memcpy(&n_dims, content + current_offset, sizeof(n_dims));
        current_offset += sizeof(n_dims);

        if (n_dims > HMLL_MAX_TENSOR_RANK) {
            ctx->error = HMLL_ERR(HMLL_ERR_INVALID_RANGE);
            goto exit;
        }

        reg->tensors[i].rank = n_dims;
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = 0;
            memcpy(&dim, content + current_offset, sizeof(dim));
            reg->tensors[i].shape[d] = dim;
            current_offset += sizeof(dim);
        }

        uint32_t tensor_type = 0;
        memcpy(&tensor_type, content + current_offset, sizeof(tensor_type));
        current_offset += sizeof(tensor_type);

        reg->tensors[i].dtype = ggml_type_to_hmll_dtype(tensor_type);

        uint64_t tensor_offset = 0;
        memcpy(&tensor_offset, content + current_offset, sizeof(tensor_offset));
        current_offset += sizeof(tensor_offset);

        reg->tensors[i].start = tensor_offset;
        reg->indexes[i] = fid;
    }

    size_t t_start = current_offset;
    t_start = (t_start + GGUF_DEFAULT_ALIGNMENT - 1) & ~(GGUF_DEFAULT_ALIGNMENT - 1);

    for (uint64_t i = 0; i < num_tensors; ++i) {
        const size_t nbytes = hmll_numel(&reg->tensors[i]) * hmll_nbits(reg->tensors[i].dtype) / 8;
        reg->tensors[i].start += t_start;
        reg->tensors[i].end = reg->tensors[i].start + nbytes;
    }

    num_discovered_tensors = num_tensors;

exit:
    return num_discovered_tensors;
}