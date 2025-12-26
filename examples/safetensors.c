#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hmll/hmll.h>

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime.h>
#endif

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        printf("No file specified.\nInvoke through hmll_safetensors_ex <path/to/safetensors/file>");
        return 1;
    }

    // Get the tensors' table
    hmll_context_t ctx = {0};
    hmll_open(argv[1], &ctx, HMLL_SAFETENSORS, HMLL_MMAP | HMLL_SKIP_METADATA);
    hmll_fetcher_t fetcher = hmll_fetcher_init(&ctx, HMLL_DEVICE_CUDA, HMLL_FETCHER_AUTO);
    hmll_tensor_lookup_result_t lookup = hmll_get_tensor_specs(&ctx, "model.embed_tokens.weight");

    if (hmll_success(hmll_get_error(&ctx) && lookup.found))
    {
        hmll_device_buffer_t buffer = hmll_get_buffer_for_range(&ctx, HMLL_DEVICE_CUDA, (struct hmll_range){ lookup.specs.start, lookup.specs.end });
        if (hmll_success(hmll_get_error(&ctx))) {
            // Start timing
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            const hmll_fetch_range_t offsets = hmll_fetch_tensor(&ctx, fetcher, "model.embed_tokens.weight", buffer);

            // End timing and calculate elapsed time
            clock_gettime(CLOCK_MONOTONIC, &end);
            const double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
            const double elapsed_ms = elapsed_ns / 1e6;
            const double elapsed_s = elapsed_ns / 1e9;

            if (hmll_success(hmll_get_error(&ctx))) {
                // Calculate throughput
                const double size_mb = (double)(buffer.size) / (1024.0 * 1024.0);
                const double throughput_mbps = size_mb / elapsed_s;

                printf("Fetch completed in %.3f ms (%.6f s)\n", elapsed_ms, elapsed_s);
                printf("Tensor size: %.2f MB\n", size_mb);
                printf("Throughput: %.2f MB/s\n", throughput_mbps);

                __bf16 *bf16_ptr;
                if (fetcher.device == HMLL_DEVICE_CUDA) {
                    bf16_ptr = malloc(buffer.size);
                    cudaMemcpy(bf16_ptr, buffer.ptr + offsets.start, hmll_numel(&lookup.specs) * sizeof(__bf16), cudaMemcpyDeviceToHost);
                } else {
                    bf16_ptr = buffer.ptr + offsets.start;
                }

                float sum = 0;
                for (size_t i = 0; i < hmll_numel(&lookup.specs); ++i) sum += bf16_ptr[i];

                printf("Sum: %f\n", sum);
            } else {
                printf("Got an error while reading the safetensors: %s\n", hmll_strerr(ctx.error));
            }
        }
    }

    return ctx.error;
}
