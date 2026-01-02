//
// Created by mfuntowicz on 12/1/25.
//
#include "hmll/safetensors.h"

#include "hmll/hmll.h"
#include "hmll/types.h"
#include <yyjson.h>


char *hmll_safetensors_path_create(const char *path, const char *file) {
    if (path == NULL || file == NULL)
        return NULL;

    // TODO: handle windows separators
    const char *last_slash = strrchr(path, '/');

    size_t dir_len = 0;
    if (last_slash != NULL)
        dir_len = (last_slash - path) + 1;

    size_t file_len = strlen(file);
    char *new_path = malloc(dir_len + file_len + 1);
    if (new_path == NULL)
        return NULL;

    if (dir_len > 0)
        memcpy(new_path, path, dir_len);
    strcpy(new_path + dir_len, file);

    return new_path;
}


static short hmll_safetensors_contains_key(char **files, const size_t num_files, const char *key, const size_t key_len)
{
    for (size_t i = 0; i < num_files; ++i) {
        if (strncmp(files[i], key, key_len) == 0) return (short)i;
    }
    return -1;
}

static hmll_tensor_data_type_t hmll_safetensors_dtype_from_str(const char *dtype, const size_t size)
{
    if (strncmp(dtype, "BOOL", size) == 0) return HMLL_DTYPE_BOOL;
    if (strncmp(dtype, "BF16", size) == 0) return HMLL_DTYPE_BFLOAT16;
    if (strncmp(dtype, "C64", size) == 0) return HMLL_DTYPE_COMPLEX;
    if (strncmp(dtype, "FP4", size) == 0) return HMLL_DTYPE_FLOAT4;
    if (strncmp(dtype, "F6_E2M3", size) == 0) return HMLL_DTYPE_FLOAT6_E2M3;
    if (strncmp(dtype, "F6_E3M2", size) == 0) return HMLL_DTYPE_FLOAT6_E3M2;
    if (strncmp(dtype, "F8_E4M3", size) == 0) return HMLL_DTYPE_FLOAT8_E4M3;
    if (strncmp(dtype, "F8_E5M2", size) == 0) return HMLL_DTYPE_FLOAT8_E5M2;
    if (strncmp(dtype, "I8", size) == 0) return HMLL_DTYPE_SIGNED_INT8;
    if (strncmp(dtype, "I16", size) == 0) return HMLL_DTYPE_SIGNED_INT16;
    if (strncmp(dtype, "I32", size) == 0) return HMLL_DTYPE_SIGNED_INT32;
    if (strncmp(dtype, "I64", size) == 0) return HMLL_DTYPE_SIGNED_INT64;
    if (strncmp(dtype, "U8", size) == 0) return HMLL_DTYPE_UNSIGNED_INT8;
    if (strncmp(dtype, "U16", size) == 0) return HMLL_DTYPE_UNSIGNED_INT16;
    if (strncmp(dtype, "U32", size) == 0) return HMLL_DTYPE_UNSIGNED_INT32;
    if (strncmp(dtype, "U64", size) == 0) return HMLL_DTYPE_UNSIGNED_INT64;
    if (strncmp(dtype, "FP32", size) == 0) return HMLL_DTYPE_FLOAT32;
    if (strncmp(dtype, "FP16", size) == 0) return HMLL_DTYPE_FLOAT16;

    return HMLL_DTYPE_UNKNOWN;
}

static enum hmll_error_code hmll_safetensors_header_parse_dtype(yyjson_val *dtype, struct hmll_tensor_specs *tensor)
{
    if (dtype && yyjson_is_str(dtype)) {
        const char* dtype_str = yyjson_get_str(dtype);
        const size_t dtype_len = yyjson_get_len(dtype);
        tensor->dtype = hmll_safetensors_dtype_from_str(dtype_str, dtype_len);
    } else {
        tensor->dtype = HMLL_DTYPE_UNKNOWN;
    }

    if (tensor->dtype == HMLL_DTYPE_UNKNOWN) {
        return HMLL_ERR_UNKNOWN_DTYPE;
    }

    return HMLL_ERR_SUCCESS;
}

static enum hmll_error_code hmll_safetensors_header_parse_offsets(yyjson_val *offsets, struct hmll_tensor_specs *tensor)
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

            return HMLL_ERR_SUCCESS;
        }

        return HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER;
    }

    return HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER;
}

static enum hmll_error_code hmll_safetensors_header_parse_shape(yyjson_val *shape, struct hmll_tensor_specs *tensor)
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
            return HMLL_ERR_SUCCESS;
        }

        return HMLL_ERR_SUCCESS;
    }

    return HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER;
}

static enum hmll_error_code hmll_safetensors_header_parse_tensor(yyjson_val *specs, hmll_tensor_specs_t *tensor)
{
    enum hmll_error_code error = HMLL_ERR_SUCCESS;

    // Parse dtype
    yyjson_val* dtype_val = yyjson_obj_get(specs, "dtype");
    error = hmll_safetensors_header_parse_dtype(dtype_val, tensor);

    if (error != HMLL_ERR_SUCCESS) return error;

    // Parse shape
    yyjson_val* shape_val = yyjson_obj_get(specs, "shape");
    error = hmll_safetensors_header_parse_shape(shape_val, tensor);

    if (error != HMLL_ERR_SUCCESS) return error;

    // Parse offsets
    yyjson_val* data_offsets_val = yyjson_obj_get(specs, "data_offsets");
    return hmll_safetensors_header_parse_offsets(data_offsets_val, tensor);
}

struct hmll_safetensors_index
{
    const char **files;
    size_t n;
};

struct hmll_safetensors_index hmll_safetensors_read_index(struct hmll_context *ctx, const struct hmll_source source)
{
    size_t num_files = 0;
    size_t num_allocated_files = 0;
    char **files = NULL;

    if (hmll_has_error(hmll_get_error(ctx)))
        goto return_error;

    yyjson_read_err error;
    yyjson_doc *document = NULL;
    if ((document = yyjson_read_opts(source.content, source.size, YYJSON_READ_NOFLAG, NULL, &error)) == NULL) {
        ctx->error = HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX;
        goto return_error;
    }

    yyjson_val *root = yyjson_doc_get_root(document);
    yyjson_val *map = yyjson_obj_get(root, "weight_map");
    if (map == NULL) {
        ctx->error = HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX;
        goto cleanup;
    }

    // indexes contain all the tensor -> file mapping, so we know how many tensors we have
    // note: the function only allocates memory to hold all the definition but does not fill it,
    // filling is handled by hmll_safetensors_populate_table
    ctx->num_tensors = yyjson_obj_size(map) + 1;  // (account for __metadata__ not in the index)
    ctx->table.indexes = calloc(ctx->num_tensors, sizeof(unsigned short*));
    ctx->table.names = calloc(ctx->num_tensors, sizeof(char*));
    ctx->table.tensors = calloc(ctx->num_tensors, sizeof(struct hmll_tensor_specs));

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
                    ctx->error = HMLL_ERR_ALLOCATION_FAILED;
                    goto cleanup;
                }
                files = files_;
                files[num_files - 1] = NULL;
            } else {
                if ((files = calloc(num_files, sizeof(char *))) == NULL) {
                    ctx->error = HMLL_ERR_ALLOCATION_FAILED;
                    goto cleanup;
                }
                num_allocated_files = 1;
            }
            fidx = (short)(num_files - 1);
            files[fidx] = strndup(s_file, l_file);
        }
    }

    free(document);
    ctx->num_sources = num_files;
    ctx->sources = calloc(ctx->num_sources, sizeof(struct hmll_source));
    return (struct hmll_safetensors_index) {(const char **)files, num_files};

cleanup:
    if (files) {
        for (size_t i = 0; i < num_files; ++i) free(files[i]);
        free(files);
    }

    if (ctx->table.indexes) { free(ctx->table.indexes); ctx->table.indexes = NULL; }
    if (ctx->table.names) { free(ctx->table.names); ctx->table.names = NULL; }
    if (ctx->table.tensors) { free(ctx->table.tensors); ctx->table.tensors = NULL; }

    free(document);

return_error:
    return (struct hmll_safetensors_index) {0};
}

int hmll_safetensors_populate_table(
    struct hmll_context *ctx,
    const struct hmll_source source,
    const hmll_flags_t flags,
    const size_t fid,
    const size_t offset
)
{
    HMLL_UNUSED(flags);

    size_t num_tensors = 0;
    if (hmll_has_error(hmll_get_error(ctx)))
        goto exit;

    uint64_t hsize;
    memcpy(&hsize, source.content, sizeof(uint64_t));
    char *header = source.content + sizeof(uint64_t);

    // Parse JSON
    yyjson_read_err error;
    yyjson_doc *document = yyjson_read_opts(header, hsize, YYJSON_READ_NOFLAG, NULL, &error);
    if (!document) {
        ctx->error = HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER;
        goto freeup_and_exit;
    }

    yyjson_val *root = yyjson_doc_get_root(document);
    if (!yyjson_is_obj(root)) {
        ctx->error = HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER;
        goto freeup_and_exit;
    }

    // we don't allocate this the number of tensors is already set on the context
    // this means we are certainly reading from a chunked safetensors file and the context has an
    // omniscient view on the total number of tensors
    num_tensors = yyjson_obj_size(root);
    if (ctx->num_tensors == 0) {
        if ((ctx->table.names = calloc(num_tensors, sizeof(char*))) == NULL) {
            ctx->error = HMLL_ERR_ALLOCATION_FAILED;
            goto freeup_and_exit;
        }

        if ((ctx->table.tensors = calloc(num_tensors, sizeof(struct hmll_tensor_specs))) == NULL) {
            ctx->error = HMLL_ERR_ALLOCATION_FAILED;
            goto freeup_and_exit;
        }

        if ((ctx->table.indexes = calloc(num_tensors, sizeof(struct hmll_source))) == NULL) {
            ctx->error = HMLL_ERR_ALLOCATION_FAILED;
            goto freeup_and_exit;
        }
    }

    char **names = ctx->table.names;
    unsigned short *indexes = ctx->table.indexes;
    hmll_tensor_specs_t *tensors = ctx->table.tensors;

    size_t idx, max, tidx = 0;
    yyjson_val *key, *val;
    yyjson_obj_foreach(root, idx, max, key, val) {

        const char *keyval = yyjson_get_str(key);
        const unsigned is_metadata = strcmp(keyval, "__metadata__") == 0;

        // Skip __metadata__ if the flag is set
        if (is_metadata != HMLL_FALSE) continue;
            // if (!(flags & HMLL_SKIP_METADATA)) { /* TODO: Not implemented yet */ }

        if (indexes != NULL) indexes[offset + tidx] = fid; // can be NULL if not chunked safetensors

        const size_t name_len = yyjson_get_len(key);
        names[offset + tidx] = strndup(keyval, name_len);

        if (!yyjson_is_obj(val)) {
            ctx->error = HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER;
            goto freeup_and_exit;
        }

        if (hmll_safetensors_header_parse_tensor(val, tensors + offset + tidx) != HMLL_SUCCESS)
            goto freeup_and_exit;

        // tensor offsets start at 0, we need to add header size + 8 to get the real position in the file
        tensors[offset + tidx].start += hsize + 8;
        tensors[offset + tidx].end += hsize + 8;

        ++tidx;
    }

freeup_and_exit:
    yyjson_doc_free(document);

exit:
    if (hmll_has_error(hmll_get_error(ctx))) return ctx->error;

    if (ctx->num_tensors == 0) ctx->num_tensors = tidx;
    return tidx;
}

int hmll_safetensors_open(
    struct hmll_context *ctx, const char *path, const enum hmll_file_kind kind, const enum hmll_flags flags)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        return -1;

    int num_tensors = 0;
    const struct hmll_source source = hmll_open_mapped(ctx, path);
    if (kind == HMLL_SAFETENSORS_CHUNKED) {
        const struct hmll_safetensors_index index = hmll_safetensors_read_index(ctx, source);

        size_t offset = 0;
        for (size_t fid = 0; fid < index.n; ++fid) {
            char *p = hmll_safetensors_path_create(path, index.files[fid]);
            if (p == NULL) {
                ctx->error = HMLL_ERR_ALLOCATION_FAILED;
                return -1;
            }
            const struct hmll_source src = hmll_open_mapped(ctx, p);
            const int res = hmll_safetensors_populate_table(ctx, src, flags, fid, offset);
            if (res < 0) return ctx->error;

            ctx->sources[fid] = src;
            offset += res;
            free(p);
        }
        num_tensors = offset - (ctx->num_sources - 1);
    } else {
        ctx->num_sources = 1;
        ctx->sources = calloc(ctx->num_sources, sizeof(struct hmll_source));
        ctx->sources[0] = source;
        num_tensors = hmll_safetensors_populate_table(ctx, source, flags, 0, 0);
    }

    return num_tensors;
}