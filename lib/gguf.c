//
// Created by mfuntowicz on 2/5/26.
//
#include <string.h>

#include "hmll/hmll.h"

#define GGUF_HEADER 0x46554747

size_t hmll_gguf_populate_registry(struct hmll *ctx, struct hmll_registry *reg, struct hmll_source source, size_t fid, size_t offset)
{
    HMLL_UNUSED(reg);
    HMLL_UNUSED(fid);
    HMLL_UNUSED(offset);
    if (hmll_check(ctx->error)) return 0;
    if (!source.content) return 0;

    size_t num_discovered_tensors = 0;
    const unsigned char *content = source.content;

    uint32_t header = 0;
    memcpy(&header, content, sizeof(uint32_t));

    if (header != GGUF_HEADER) {
        ctx->error = HMLL_ERR(HMLL_ERR_GGUF_INVALID_HEADER);
        goto exit;
    }

    uint32_t version = 0;
    memcpy(&version, content + sizeof(uint32_t), sizeof(uint32_t));
    if (version != 3) {
        ctx->error = HMLL_ERR(HMLL_ERR_GGUF_UNSUPPORTED_GGUF_VERSION);
        goto exit;
    }

    uint64_t num_tensors = 0;
    memcpy(&num_tensors, content + sizeof(uint32_t) * 2, sizeof(uint64_t));
    if (num_tensors <= 0) {
        ctx->error = HMLL_ERR(HMLL_ERR_TABLE_EMPTY);
        goto exit;
    }

    uint64_t num_metadata = 0;
    memcpy(&num_metadata, content + sizeof(uint32_t) * 2, sizeof(uint64_t));

    printf("Discovered %zu tensors, %zu metadata", num_tensors, num_metadata);
exit:
    return num_discovered_tensors;
}