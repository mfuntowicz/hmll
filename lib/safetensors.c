//
// Created by mfuntowicz on 12/1/25.
//
#include "hmll/safetensors.h"

#include "hmll/hmll.h"
#include <yyjson.h>

hmll_status_code_t hmll_safetensors_header_parse_dtype(
    yyjson_val *dtype, hmll_tensor_specs_t *tensor, hmll_status_t *status)
{
    if (dtype && yyjson_is_str(dtype)) {
        const char* dtype_str = yyjson_get_str(dtype);
        const size_t dtype_len = yyjson_get_len(dtype);
        tensor->dtype = hmll_safetensors_dtype_from_str(dtype_str, dtype_len);
    }
    else {
        tensor->dtype = HMLL_DTYPE_UNKNOWN;
    }

    if (tensor->dtype == HMLL_DTYPE_UNKNOWN) {
        status->what = HMLL_UNKNOWN_DTYPE;
        status->message = "Unknown dtype";
    }

    return status->what;
}

hmll_status_code_t hmll_safetensors_header_parse_offsets(
    yyjson_val *offsets, hmll_tensor_specs_t *tensor, hmll_status_t *status)
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

            return HMLL_SUCCESS;
        }

        status->what = HMLL_SAFETENSORS_HEADER_INVALID;
        status->message = "Expected length 2 array (begin, end)";
    } else {
        status->what = HMLL_SAFETENSORS_HEADER_INVALID;
        status->message = "Expected offsets to be represented as an array";
    }
    return status->what;
}

hmll_status_code_t hmll_safetensors_header_parse_shape(
    yyjson_val *shape, hmll_tensor_specs_t *tensor, hmll_status_t *status)
{
    if (shape && yyjson_is_arr(shape)) {
        const size_t rank = yyjson_arr_size(shape);
        tensor->rank = (uint8_t)rank;

        if (rank > 0) {
            //TODO : Add malloc checking
            if ((tensor->shape = calloc(rank, sizeof(size_t))) == NULL) {
                status->what = HMLL_ALLOCATION_FAILED;
                status->message = "Failed to allocate memory to store tensor's shape";
            }

            size_t shape_idx = 0, shape_max = 0;
            yyjson_val* dim_val;
            yyjson_arr_foreach(shape, shape_idx, shape_max, dim_val) {
                if (yyjson_is_uint(dim_val))
                    tensor->shape[shape_idx] = yyjson_get_uint(dim_val);
            }
            return HMLL_SUCCESS;
        }

        tensor->shape = 0;
        return HMLL_SUCCESS;
    }

    status->what = HMLL_SAFETENSORS_HEADER_INVALID;
    status->message = "Expected shape to be an array of integers";
    return HMLL_SAFETENSORS_HEADER_INVALID;
}

hmll_status_code_t hmll_safetensors_header_parse_tensor(
    yyjson_val *specs, hmll_tensor_specs_t *tensor, hmll_status_t *status)
{
    // Parse dtype
    yyjson_val* dtype_val = yyjson_obj_get(specs, "dtype");
    if (hmll_safetensors_header_parse_dtype(dtype_val, tensor, status) != HMLL_SUCCESS) return status->what;

    // Parse shape
    yyjson_val* shape_val = yyjson_obj_get(specs, "shape");
    if (hmll_safetensors_header_parse_shape(shape_val, tensor, status) != HMLL_SUCCESS) return status->what;

    // Parse offsets
    yyjson_val* data_offsets_val = yyjson_obj_get(specs, "data_offsets");
    return hmll_safetensors_header_parse_offsets(data_offsets_val, tensor, status);
}


hmll_tensor_data_type_t hmll_safetensors_dtype_from_str(const char *dtype, const size_t size)
{
    if (strncmp(dtype, "BF16", size) == 0) return HMLL_DTYPE_BFLOAT16;
    if (strncmp(dtype, "FP32", size) == 0) return HMLL_DTYPE_FLOAT32;
    if (strncmp(dtype, "FP16", size) == 0) return HMLL_DTYPE_FLOAT16;

    return HMLL_DTYPE_UNKNOWN;
}


hmll_status_t hmll_safetensors_read_table(hmll_context_t *ctx, const hmll_flags_t flags)
{
    hmll_status_t status = HMLL_SUCCEEDED;

    // Get the size of the JSON header
    // TODO: Change that when fd is supported
    uint64_t hsize;
    memcpy(&hsize, ctx->source.content, sizeof(uint64_t));

    char *header = ctx->source.content + sizeof(uint64_t);

    // Parse JSON
    yyjson_read_err error;
    yyjson_doc *document = yyjson_read_opts(header, hsize, YYJSON_READ_NOFLAG, NULL, &error);
    if (!document) {
        status.what = HMLL_SAFETENSORS_HEADER_JSON_ERROR;
        status.message = error.msg;
        goto freeup_and_return;
    }

    yyjson_val *root = yyjson_doc_get_root(document);
    if (!yyjson_is_obj(root)) {
        status.what = HMLL_SAFETENSORS_HEADER_JSON_ERROR;
        status.message = error.msg;
        goto freeup_and_return;
    }

    const size_t num_tensors = yyjson_obj_size(root);
    if ((ctx->table.names = calloc(num_tensors, sizeof(char*))) == NULL) {
        status.what = HMLL_ALLOCATION_FAILED;
        status.message = "Failed to allocated memory to store tensor's names";
    }

    if ((ctx->table.tensors = calloc(num_tensors, sizeof(struct hmll_tensor_specs))) == NULL) {
        status.what = HMLL_ALLOCATION_FAILED;
        status.message = "Failed to allocated memory to store tensor's descriptor";
    }

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
            status.what = HMLL_ALLOCATION_FAILED;
            status.message = "Failed to allocated memory to store tensor's name";
            goto freeup_and_return;
        }

        // TODO: Shall we copy the string here? Or just keep track of the pointer?
        strncpy(names[idx], keyval, name_len);

        // Parse tensor object
        if (!is_metadata) {
            if (!yyjson_is_obj(val)) {
                status.what = HMLL_SAFETENSORS_HEADER_INVALID;
                status.message = "Expected JSON object";
                goto freeup_and_return;
            }

            if (hmll_safetensors_header_parse_tensor(val, tensors + idx, &status) != HMLL_SUCCESS)
                goto freeup_and_return;

            // Tensor offsets start at 0, we need to add header size + 8 to get the real position in the file
            tensors[idx].start += hsize + 8;
            tensors[idx].end += hsize + 8;
        }

        // TODO: What if we allocated names but not tensors? num_tensors would be different ~~
        ++ctx->num_tensors;
    }

freeup_and_return:
    if (document)
        yyjson_doc_free(document);

    // TODO: Free the table only - not the source
    return status;
}
