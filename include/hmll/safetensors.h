//
// Created by mfuntowicz on 12/1/25.
//

#ifndef HMLL_HMLL_SAFETENSORS_H
#define HMLL_HMLL_SAFETENSORS_H

#include "hmll/types.h"

int hmll_safetensors_populate_table(hmll_context_t *ctx, struct hmll_source source, hmll_flags_t flags, size_t fid, size_t offset);
struct hmll_safetensors_index hmll_safetensors_read_index(struct hmll *ctx, struct hmll_source source);
int hmll_safetensors_open(struct hmll *ctx, const char *path, enum hmll_file_kind kind, enum hmll_flags flags);

#endif //HMLL_HMLL_SAFETENSORS_H
