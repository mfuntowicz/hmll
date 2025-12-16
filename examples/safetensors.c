#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hmll/hmll.h>
#include <sys/mman.h>

#define ALIGNMENT 4096U
#define PAGE_ALIGNED_UP(x) (((x) + ALIGNMENT - 1) & ~(ALIGNMENT - 1))
#define PAGE_ALIGNED_DOWN(x) ((x) & ~(ALIGNMENT - 1))

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        printf("No file specified.\nInvoke through hmll_safetensors_ex <path/to/safetensors/file>");
        return 1;
    }

    // Get the tensors' table
    hmll_context_t ctx = {0};
    hmll_open(argv[1], &ctx, HMLL_SAFETENSORS, HMLL_MMAP | HMLL_SKIP_METADATA);
    hmll_fetcher_t fetcher = hmll_fetcher_init(&ctx, HMLL_DEVICE_CPU, HMLL_FETCHER_AUTO);
    hmll_tensor_specs_t specs = hmll_get_tensor_specs(&ctx, "model.embed_tokens.weight");

    if (hmll_success(hmll_get_error(&ctx)))
    {

        const size_t alstart = PAGE_ALIGNED_DOWN(specs.start);
        const size_t alend = PAGE_ALIGNED_UP(specs.end);
        const size_t alsize = alend - alstart;

        void *ptr = hmll_get_hugepage_buffer(&ctx, alsize);
        hmll_device_buffer_t buffer = {ptr, alsize, HMLL_DEVICE_CPU};

        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        struct hmll_fetch_range offsets = hmll_fetch_tensor(&ctx, fetcher, "model.embed_tokens.weight", buffer);

        // End timing and calculate elapsed time
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        double elapsed_ms = elapsed_ns / 1e6;
        double elapsed_s = elapsed_ns / 1e9;

        if (hmll_success(hmll_get_error(&ctx))) {
            // Calculate throughput
            double size_mb = (double)(alsize) / (1024.0 * 1024.0);
            double throughput_mbps = size_mb / elapsed_s;

            printf("Fetch completed in %.3f ms (%.6f s)\n", elapsed_ms, elapsed_s);
            printf("Tensor size: %.2f MB\n", size_mb);
            printf("Throughput: %.2f MB/s\n", throughput_mbps);

            __bf16 *bf16_ptr = ptr + offsets.start;
            float sum = 0;
            for (size_t i = 0; i < hmll_numel(&specs); ++i) sum += bf16_ptr[i];

            printf("Sum: %f\n", sum);
        } else {
            printf("Got an error while reading the safetensors: %s\n", hmll_strerr(ctx.error));
        }
    }

    return ctx.error;
}
