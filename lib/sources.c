#include "hmll/hmll.h"
#include "hmll/safetensors.h"

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/mmap.h"
#endif


hmll_status_t hmll_open(const char *path, hmll_context_t *ctx, const hmll_file_kind_t kind, const hmll_flags_t flags)
{
    if (flags & HMLL_MMAP) {
        const hmll_status_t status = hmll_open_mmap(path, ctx);
        if (!hmll_success(status))
            return status;

        if (kind == HMLL_SAFETENSORS)
            return hmll_safetensors_read_table(ctx, flags);

        return (hmll_status_t){HMLL_UNSUPPORTED_OP, "Only safetensors opening is supported for now"};
    }

    return (hmll_status_t){HMLL_UNSUPPORTED_OP, "Only mmap opening is supported for now"};
}
