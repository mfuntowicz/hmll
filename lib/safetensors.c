//
// Created by mfuntowicz on 12/1/25.
//
#include "hmll/safetensors.h"

#include "hmll/hmll.h"
#include "hmll/types.h"
#include <yyjson.h>

enum hmll_error_code hmll_safetensors_header_parse_dtype(yyjson_val *dtype, struct hmll_tensor_specs *tensor)
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

enum hmll_error_code hmll_safetensors_header_parse_offsets(yyjson_val *offsets, struct hmll_tensor_specs *tensor)
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

enum hmll_error_code hmll_safetensors_header_parse_shape(yyjson_val *shape, struct hmll_tensor_specs *tensor)
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

enum hmll_error_code hmll_safetensors_header_parse_tensor(yyjson_val *specs, hmll_tensor_specs_t *tensor)
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

hmll_tensor_data_type_t hmll_safetensors_dtype_from_str(const char *dtype, const size_t size)
{
    if (strncmp(dtype, "BF16", size) == 0) return HMLL_DTYPE_BFLOAT16;
    if (strncmp(dtype, "FP32", size) == 0) return HMLL_DTYPE_FLOAT32;
    if (strncmp(dtype, "FP16", size) == 0) return HMLL_DTYPE_FLOAT16;

    return HMLL_DTYPE_UNKNOWN;
}

int hmll_safetensors_read_table(hmll_context_t *ctx, const hmll_flags_t flags)
{
    size_t num_tensors = 0;
    if (hmll_has_error(hmll_get_error(ctx)))
        goto exit;

    uint64_t hsize;
    memcpy(&hsize, ctx->source.content, sizeof(uint64_t));
    char *header = ctx->source.content + sizeof(uint64_t);

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

    num_tensors = yyjson_obj_size(root);
    if ((ctx->table.names = calloc(num_tensors, sizeof(char*))) == NULL)
        ctx->error = HMLL_ERR_ALLOCATION_FAILED;


    if ((ctx->table.tensors = calloc(num_tensors, sizeof(struct hmll_tensor_specs))) == NULL)
        ctx->error = HMLL_ERR_ALLOCATION_FAILED;

    char **names = ctx->table.names;
    hmll_tensor_specs_t *tensors = ctx->table.tensors;

    size_t idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(root, idx, max, key, val) {

        const char *keyval = yyjson_get_str(key);
        const int is_metadata = strcmp(keyval, "__metadata__") == 0;

        // Skip __metadata__ if the flag is set
        if (is_metadata)
            if (!(flags & HMLL_SKIP_METADATA)) { /* TODO: Not implemented yet */ }

        // Create the final tensor name with a null-terminator string to allow usage of str functions like strncmp
        const size_t name_len = yyjson_get_len(key);
        if ((names[idx] = calloc(name_len + 1, sizeof(char))) == NULL) {
            ctx->error = HMLL_ERR_ALLOCATION_FAILED;
            goto freeup_and_exit;
        }

        // TODO: Shall we copy the string here? Or just keep track of the pointer?
        strncpy(names[idx], keyval, name_len);

        // Parse tensor object
        if (!is_metadata) {
            if (!yyjson_is_obj(val)) {
                ctx->error = HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER;
                goto freeup_and_exit;
            }

            if (hmll_safetensors_header_parse_tensor(val, tensors + idx) != HMLL_SUCCESS)
                goto freeup_and_exit;

            // Tensor offsets start at 0, we need to add header size + 8 to get the real position in the file
            tensors[idx].start += hsize + 8;
            tensors[idx].end += hsize + 8;
        }

        // TODO: What if we allocated names but not tensors? num_tensors would be different ~~
        ++ctx->num_tensors;
    }

freeup_and_exit:
    yyjson_doc_free(document);

exit:
    if (hmll_has_error(hmll_get_error(ctx))) return ctx->error;
    return num_tensors;
}
