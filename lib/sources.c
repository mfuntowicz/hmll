#include "hmll/hmll.h"
#include "hmll/safetensors.h"

int hmll_open(hmll_context_t *ctx, const char *path, const hmll_file_kind_t kind, const hmll_flags_t flags)
{
    if (kind == HMLL_SAFETENSORS || kind == HMLL_SAFETENSORS_CHUNKED)
        return hmll_safetensors_open(ctx, path, kind, flags);

    ctx->error = HMLL_ERR_UNSUPPORTED_FILE_FORMAT;
    return 0;
}
