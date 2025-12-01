#include <stdio.h>
#include <stdlib.h>
#include <hmll/hmll.h>

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        printf("No file specified.\nInvoke through hmll_safetensors_ex <path/to/safetensors/file>");
        return 1;
    }

    hmll_context_t ctx;

    // Get the tensors' table
    const hmll_status_t status = hmll_open(argv[1], &ctx, HMLL_SAFETENSORS, HMLL_MMAP | HMLL_SKIP_METADATA);
    if (hmll_success(status)) {
        printf("Successfully opened file %s (num tensors=%lu)", argv[1], ctx.num_tensors);
    } else {
        printf("Failed to open file %s", argv[1]);
        return 2;
    }

    return 0;
}
