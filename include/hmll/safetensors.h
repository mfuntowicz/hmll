//
// Created by mfuntowicz on 12/1/25.
//

#ifndef HMLL_HMLL_SAFETENSORS_H
#define HMLL_HMLL_SAFETENSORS_H

#include "hmll/status.h"
#include "hmll/types.h"

hmll_tensor_data_type_t hmll_safetensors_dtype_from_str(const char *dtype, size_t size);
hmll_status_t hmll_safetensors_read_table(hmll_context_t *ctx, hmll_flags_t flags);

#endif //HMLL_HMLL_SAFETENSORS_H
