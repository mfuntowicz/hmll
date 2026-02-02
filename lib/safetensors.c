//
// Created by mfuntowicz on 12/1/25.
//
#include <fcntl.h>
#include <stdio.h>
#include "hmll/hmll.h"
#include "hmll/types.h"
#include <yyjson.h>

#ifdef _WIN32
#include "hmll/win32/file.h"

// strndup is not available on Windows
static char* strndup(const char* s, size_t n) {
    size_t len = strnlen(s, n);
    char* result = malloc(len + 1);
    if (result) {
        memcpy(result, s, len);
        result[len] = '\0';
    }
    return result;
}
#else
#include "hmll/unix/file.h"
#endif

char *hmll_safetensors_path_create(const char *path, const char *file) {
    if (!path || !file)
        return NULL;

    // TODO: handle windows separators
    const char *last_slash = strrchr(path, '/');

    size_t dir_len = 0;
    if (last_slash != NULL)
        dir_len = (last_slash - path) + 1;

    const size_t file_len = strlen(file);
    char *new_path = malloc(dir_len + file_len + 1);
    if (!new_path)
        return NULL;

    if (dir_len > 0)
        memcpy(new_path, path, dir_len);
    memcpy(new_path + dir_len, file, file_len + 1);

    return new_path;
}


static short hmll_safetensors_contains_key(char **files, const size_t num_files, const char *key, const size_t key_len)
{
    for (size_t i = 0; i < num_files; ++i) {
        if (strncmp(files[i], key, key_len) == 0) return (short)i;
    }
    return -1;
}

HMLL_STATIC enum hmll_dtype hmll_safetensors_dtype_from_str(const char *dtype, const size_t size)
{
    if (size == 0) return HMLL_DTYPE_UNKNOWN;
    if (strncmp(dtype, "BOOL", size) == 0) return HMLL_DTYPE_BOOL;
    if (strncmp(dtype, "BF16", size) == 0) return HMLL_DTYPE_BFLOAT16;
    if (strncmp(dtype, "C64", size) == 0) return HMLL_DTYPE_COMPLEX;
    if (strncmp(dtype, "F4", size) == 0) return HMLL_DTYPE_FLOAT4;
    if (strncmp(dtype, "F6_E2M3", size) == 0) return HMLL_DTYPE_FLOAT6_E2M3;
    if (strncmp(dtype, "F6_E3M2", size) == 0) return HMLL_DTYPE_FLOAT6_E3M2;
    if (strncmp(dtype, "F8_E8M0", size) == 0) return HMLL_DTYPE_FLOAT8_E8M0;
    if (strncmp(dtype, "F8_E4M3", size) == 0) return HMLL_DTYPE_FLOAT8_E4M3;
    if (strncmp(dtype, "F8_E5M2", size) == 0) return HMLL_DTYPE_FLOAT8_E5M2;
    if (strncmp(dtype, "F16", size) == 0) return HMLL_DTYPE_FLOAT16;
    if (strncmp(dtype, "F32", size) == 0) return HMLL_DTYPE_FLOAT32;
    if (strncmp(dtype, "F64", size) == 0) return HMLL_DTYPE_FLOAT64;
    if (strncmp(dtype, "I8", size) == 0) return HMLL_DTYPE_SIGNED_INT8;
    if (strncmp(dtype, "I16", size) == 0) return HMLL_DTYPE_SIGNED_INT16;
    if (strncmp(dtype, "I32", size) == 0) return HMLL_DTYPE_SIGNED_INT32;
    if (strncmp(dtype, "I64", size) == 0) return HMLL_DTYPE_SIGNED_INT64;
    if (strncmp(dtype, "U8", size) == 0) return HMLL_DTYPE_UNSIGNED_INT8;
    if (strncmp(dtype, "U16", size) == 0) return HMLL_DTYPE_UNSIGNED_INT16;
    if (strncmp(dtype, "U32", size) == 0) return HMLL_DTYPE_UNSIGNED_INT32;
    if (strncmp(dtype, "U64", size) == 0) return HMLL_DTYPE_UNSIGNED_INT64;

    return HMLL_DTYPE_UNKNOWN;
}

HMLL_STATIC struct hmll_error hmll_safetensors_header_parse_dtype(yyjson_val *dtype, struct hmll_tensor_specs *tensor)
{
    if (dtype && yyjson_is_str(dtype)) {
        const char* dtype_str = yyjson_get_str(dtype);
        const size_t dtype_len = yyjson_get_len(dtype);
        tensor->dtype = hmll_safetensors_dtype_from_str(dtype_str, dtype_len);
    } else {
        tensor->dtype = HMLL_DTYPE_UNKNOWN;
    }

    if (tensor->dtype == HMLL_DTYPE_UNKNOWN) {
        return HMLL_ERR(HMLL_ERR_UNKNOWN_DTYPE);
    }

    return HMLL_OK;
}

HMLL_STATIC struct hmll_error hmll_safetensors_header_parse_offsets(yyjson_val *offsets, struct hmll_tensor_specs *tensor)
{
    if (offsets && yyjson_is_arr(offsets)) {
        const size_t length = yyjson_arr_size(offsets);
        if (length >= 2) {
            yyjson_val* start_val = yyjson_arr_get(offsets, 0);
            yyjson_val* end_val = yyjson_arr_get(offsets, 1);

            if (yyjson_is_uint(start_val))
                tensor->start = yyjson_get_uint(start_val);

            if (yyjson_is_uint(end_val))
                tensor->end = yyjson_get_uint(end_val);
            return HMLL_ERR(HMLL_ERR_SUCCESS);
        }
        return HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);
    }
    return HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);
}

HMLL_STATIC struct hmll_error hmll_safetensors_header_parse_shape(yyjson_val *shape, struct hmll_tensor_specs *tensor)
{
    if (shape && yyjson_is_arr(shape)) {
        const size_t rank = yyjson_arr_size(shape);
        tensor->rank = (uint8_t)rank;

        if (rank > 0) {
            size_t shape_idx = 0, shape_max = 0;
            yyjson_val* dim_val;
            yyjson_arr_foreach(shape, shape_idx, shape_max, dim_val) {
                if (yyjson_is_uint(dim_val))
                    tensor->shape[shape_idx] = yyjson_get_uint(dim_val);
            }
            return HMLL_ERR(HMLL_ERR_SUCCESS);
        }
        return HMLL_ERR(HMLL_ERR_SUCCESS);
    }
    return HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);
}

HMLL_STATIC struct hmll_error hmll_safetensors_header_parse_tensor(yyjson_val *specs, hmll_tensor_specs_t *tensor)
{
    struct hmll_error error = HMLL_OK;

    // Parse dtype
    yyjson_val* dtype_val = yyjson_obj_get(specs, "dtype");
    error = hmll_safetensors_header_parse_dtype(dtype_val, tensor);

    if (hmll_check(error)) return error;

    // Parse shape
    yyjson_val* shape_val = yyjson_obj_get(specs, "shape");
    error = hmll_safetensors_header_parse_shape(shape_val, tensor);

    if (hmll_check(error)) return error;

    // Parse offsets
    yyjson_val* data_offsets_val = yyjson_obj_get(specs, "data_offsets");
    return hmll_safetensors_header_parse_offsets(data_offsets_val, tensor);
}

size_t hmll_safetensors_index(struct hmll *ctx, struct hmll_registry *reg, const struct hmll_source source)
{
    size_t num_files = 0;
    size_t num_allocated_files = 0;
    char **files = NULL;
    yyjson_doc *document = NULL;

    if (hmll_check(ctx->error))
        goto cleanup;

    yyjson_read_err error;
    if ((document = yyjson_read_opts((char *)source.content, source.size, YYJSON_READ_NOFLAG, NULL, &error)) == NULL) {
        ctx->error = HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX);
        goto cleanup;
    }

    yyjson_val *root = yyjson_doc_get_root(document);
    yyjson_val *map = yyjson_obj_get(root, "weight_map");
    if (map == NULL) {
        ctx->error = HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX);
        goto cleanup;
    }

    // indexes contain all the tensor -> file mapping, so we know how many tensors we have
    // note: the function only allocates memory to hold all the definition but does not fill it,
    // filling is handled by hmll_safetensors_populate_table
    reg->num_tensors = yyjson_obj_size(map);  // TODO (account for __metadata__ not in the index)
    reg->indexes = calloc(reg->num_tensors, sizeof(unsigned short*));
    reg->names = calloc(reg->num_tensors, sizeof(char*));
    reg->tensors = calloc(reg->num_tensors, sizeof(struct hmll_tensor_specs));

    size_t idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(map, idx, max, key, val) {
        const char  *s_file  = yyjson_get_str(val);
        const size_t l_file  = yyjson_get_len(val);

        // Handle duplicate filenames
        short fidx = hmll_safetensors_contains_key(files, num_files, s_file, l_file);
        if (fidx == -1) {
            ++num_files;
            if (num_allocated_files > 0) {
                num_allocated_files <<= 1;

                char **files_ = NULL;
                if ((files_ = realloc(files, num_allocated_files * sizeof(char *))) == NULL) {
                    ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
                    goto cleanup;
                }
                files = files_;
                files[num_files - 1] = NULL;
            } else {
                if ((files = calloc(num_files, sizeof(char *))) == NULL) {
                    ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
                    goto cleanup;
                }
                num_allocated_files = 1;
            }
            fidx = (short)(num_files - 1);
            files[fidx] = strndup(s_file, l_file);
        }
    }

    yyjson_doc_free(document);
    document = NULL;
    ctx->num_sources = num_files;
    ctx->sources = calloc(ctx->num_sources, sizeof(struct hmll_source));
    goto exit;

cleanup:
    if (files) {
        for (size_t i = 0; i < num_files; ++i) free(files[i]);
        free(files);
    }

    if (reg->indexes) { free(reg->indexes); reg->indexes = NULL; }
    if (reg->names) { free(reg->names); reg->names = NULL; }
    if (reg->tensors) { free(reg->tensors); reg->tensors = NULL; }

    if (document) yyjson_doc_free(document);
    num_files = 0;

exit:
    return num_files;
}

size_t hmll_safetensors_populate_registry(
    struct hmll *ctx,
    struct hmll_registry *reg,
    const struct hmll_source source,
    const size_t fid,
    const size_t offset
) {
    size_t tidx = 0, num_tensors = 0;
    if (hmll_check(ctx->error))
        goto exit;

    yyjson_doc *document = NULL;
    FILE *file = hmll_get_file_from_fd(source);
    if (!file) {
        ctx->error = HMLL_ERR(HMLL_ERR_FILE_OPEN_FAILED);
        goto freeup_and_exit;
    }

    uint64_t hsize;
    memcpy(&hsize, source.content, sizeof(uint64_t));

    // Parse JSON
    yyjson_read_err error;
    document = yyjson_read_opts((char *)source.content + sizeof(uint64_t), hsize, YYJSON_READ_NOFLAG, NULL, &error);
    if (!document) {
        ctx->error = HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER);
        goto freeup_and_exit;
    }

    yyjson_val *root = yyjson_doc_get_root(document);
    if (!yyjson_is_obj(root)) {
        ctx->error = HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER);
        goto freeup_and_exit;
    }

    // we don't allocate this the number of tensors is already set on the context
    // this means we are certainly reading from a chunked safetensors file and the context has an
    // omniscient view on the total number of tensors
    num_tensors = yyjson_obj_size(root);
    if (reg->num_tensors == 0) {
        if ((reg->names = calloc(num_tensors, sizeof(char*))) == NULL) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            goto freeup_and_exit;
        }

        if ((reg->tensors = calloc(num_tensors, sizeof(struct hmll_tensor_specs))) == NULL) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            goto freeup_and_exit;
        }

        if ((reg->indexes = calloc(num_tensors, sizeof(struct hmll_source))) == NULL) {
            ctx->error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
            goto freeup_and_exit;
        }
    }

    char **names = reg->names;
    unsigned short *indexes = reg->indexes;
    hmll_tensor_specs_t *tensors = reg->tensors;

    size_t idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(root, idx, max, key, val) {

        const char *keyval = yyjson_get_str(key);
        const unsigned is_metadata = strcmp(keyval, "__metadata__") == 0;

        // Skip __metadata__ if the flag is set
        if (is_metadata != HMLL_FALSE) continue;
        if (indexes != NULL) indexes[offset + tidx] = (unsigned short)fid; // can be NULL if not chunked safetensors

        const size_t name_len = yyjson_get_len(key);
        names[offset + tidx] = strndup(keyval, name_len);

        if (!yyjson_is_obj(val)) {
            ctx->error = HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);
            goto freeup_and_exit;
        }

        if (hmll_check(hmll_safetensors_header_parse_tensor(val, tensors + offset + tidx))) {
            ctx->error = HMLL_ERR(HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);
            goto freeup_and_exit;
        }

        // tensor offsets start at 0, we need to add header size + 8 to get the real position in the file
        tensors[offset + tidx].start += hsize + 8;
        tensors[offset + tidx].end += hsize + 8;

        ++tidx;
    }

freeup_and_exit:
    if (document) yyjson_doc_free(document);

exit:
    if (hmll_check(ctx->error)) return 0;
    if (reg->num_tensors == 0) reg->num_tensors = tidx;
    return tidx;
}
