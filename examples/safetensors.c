#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hmll/hmll.h>
#include <hmll/unix/fetcher_io_uring.h>

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        printf("No file specified.\nInvoke through hmll_safetensors_ex <path/to/safetensors/file>");
        return 1;
    }

    hmll_context_t ctx = {0};

    // Get the tensors' table
    hmll_status_t status = hmll_open(argv[1], &ctx, HMLL_SAFETENSORS, HMLL_MMAP | HMLL_SKIP_METADATA);
    if (hmll_success(status)) {
        printf("Successfully opened file %s (num tensors=%lu)\n", argv[1], ctx.num_tensors);

        // Allocate io_uring fetcher
        hmll_fetcher_io_uring_t fetcher = hmll_fetcher_io_uring_init();
        hmll_tensor_specs_t *specs;
        if (!hmll_success(status = hmll_get_tensor_specs(&ctx, "model.embed_tokens.weight", &specs))) {
            printf("%s", status.message);
            return status.what;;
        }

        const size_t numel = specs->shape[0] * specs->shape[1];
        void *ptr = calloc(numel * hmll_sizeof(specs->dtype), 1);

        const hmll_device_buffer_t buffer = {ptr, numel * hmll_sizeof(specs->dtype), HMLL_DEVICE_CPU};

        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (!hmll_success(status = hmll_fetcher_io_uring_fetch(&ctx, &fetcher, "model.embed_tokens.weight", &buffer))) {
            printf("io_uring fetch failed: %s\n", status.message);
        }

        // End timing and calculate elapsed time
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        double elapsed_ms = elapsed_ns / 1e6;
        double elapsed_s = elapsed_ns / 1e9;

        // Calculate throughput
        double size_mb = (double)(numel * hmll_sizeof(specs->dtype)) / (1024.0 * 1024.0);
        double throughput_mbps = size_mb / elapsed_s;

        printf("Fetch completed in %.3f ms (%.6f s)\n", elapsed_ms, elapsed_s);
        printf("Tensor size: %.2f MB\n", size_mb);
        printf("Throughput: %.2f MB/s\n", throughput_mbps);

        __bf16 *bf16_ptr = (__bf16 *)ptr;
        hmll_destroy(&ctx);
        if (ptr) free(ptr);
    } else {
        printf("Failed to open file %s\n", argv[1]);
        return 2;
    }

    return status.what;
}
