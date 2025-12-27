#include "hmll/hmll.h"
#include "hmll/safetensors.h"
#include "hmll/tracing.h"

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/mmap.h"
#endif


int hmll_open(const char *path, hmll_context_t *ctx, const hmll_file_kind_t kind, const hmll_flags_t flags)
{
    HMLL_ZONE_START(hmll_open)
    hmll_open_mmap(path, ctx);
    if (kind == HMLL_SAFETENSORS) {
        HMLL_ZONE_END_SUCCESS(hmll_open)
        return hmll_safetensors_read_table(ctx, flags);
    }

    HMLL_ZONE_END_ERROR(hmll_open)
    return HMLL_ERR_UNSUPPORTED_FILE_FORMAT;
}
