//
// Test-only header to expose internal safetensors functions
//

#ifndef HMLL_TESTS_SAFETENSORS_INTERNAL_H
#define HMLL_TESTS_SAFETENSORS_INTERNAL_H

#include "hmll/types.h"
#include <yyjson.h>

#ifdef __cplusplus
extern "C" {
#endif

// Expose internal function for testing
enum hmll_dtype hmll_safetensors_dtype_from_str(const char *dtype, size_t size);

// Expose path creation function
char *hmll_safetensors_path_create(const char *path, const char *file);

// Expose parsing functions for testing
struct hmll_error hmll_safetensors_header_parse_offsets(yyjson_val *offsets, struct hmll_tensor_specs *tensor);
struct hmll_error hmll_safetensors_header_parse_shape(yyjson_val *shape, struct hmll_tensor_specs *tensor);

#ifdef __cplusplus
}
#endif

#endif // HMLL_TESTS_SAFETENSORS_INTERNAL_H
