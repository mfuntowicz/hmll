#include <stdio.h>
#include <stdlib.h>
#include <hmll/hmll.h>

#ifdef _WIN32
#include <windows.h>
typedef LARGE_INTEGER timespec_t;

static void tick(timespec_t *ts) {
    QueryPerformanceCounter(ts);
}

static double time_diff_ns(const timespec_t *start, const timespec_t *end) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return (double)(end->QuadPart - start->QuadPart) * 1e9 / frequency.QuadPart;
}
#else
#include <time.h>
typedef struct timespec timespec_t;

static void tick(timespec_t *ts) {
    clock_gettime(CLOCK_MONOTONIC, ts);
}

static double time_diff_ns(const timespec_t *start, const timespec_t *end) {
    return (end->tv_sec - start->tv_sec) * 1e9 + (end->tv_nsec - start->tv_nsec);
}
#endif

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime.h>
#endif

// #define TENSOR_NAME "language_model.model.layers.15.mlp.gate_proj.weight"
#define TENSOR_NAME "model.embed_tokens.weight"

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        printf("No file specified.\nInvoke through hmll_safetensors_ex <path/to/safetensors/file>");
        return 1;
    }

    hmll_t ctx = {0};
    hmll_source_t src = {0};
    if (hmll_check(hmll_source_open(argv[1], &src)))
        return 1;

    // Read safetensors table with all tensors mapping
    hmll_registry_t registry = {0};
    if (hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) == 0)
        return 2;

    if (hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CUDA, HMLL_FETCHER_IO_URING)))
        return 3;

    const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, TENSOR_NAME);
    if (hmll_success(ctx.error) && lookup.specs != NULL)
    {
        const hmll_range_t range = (struct hmll_range){ lookup.specs->start, lookup.specs->end };
        const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
        if (hmll_success(ctx.error)) {
            // Start timing
            timespec_t start, end;
            tick(&start);

            if (hmll_fetch(&ctx, lookup.file, &buffer, range.start) < range.end - range.start) {
                fprintf(stderr, "Failed to fetch data: %s", hmll_strerr(ctx.error));
                return 4;
            }

            // End timing and calculate elapsed time
            tick(&end);
            const double elapsed_ns = time_diff_ns(&start, &end);
            const double elapsed_ms = elapsed_ns / 1e6;
            const double elapsed_s = elapsed_ns / 1e9;

            if (hmll_success(ctx.error)) {
                // Calculate throughput
                const double size_mb = (double)(buffer.size) / (1024.0 * 1024.0);
                const double throughput_mbps = size_mb / elapsed_s;

                printf("Fetch completed in %.3f ms (%.6f s)\n", elapsed_ms, elapsed_s);
                printf("Tensor size: %.2f MB\n", size_mb);
                printf("Throughput: %.2f MB/s\n", throughput_mbps);

                unsigned short *bf16_ptr;
                if (ctx.fetcher->device == HMLL_DEVICE_CUDA) {
                    bf16_ptr = malloc(buffer.size);
                    cudaMemcpy(bf16_ptr, buffer.ptr, hmll_numel(lookup.specs) * sizeof(short), cudaMemcpyDeviceToHost);
                } else {
                    bf16_ptr = buffer.ptr;
                }

                unsigned long sum = 0;
                for (size_t i = 0; i < hmll_numel(lookup.specs); ++i) sum += bf16_ptr[i];

                printf("Sum: %lu\n", sum);
            } else {
                printf("Got an error while reading the safetensors: %s\n", hmll_strerr(ctx.error));
            }
        }
    }

    return 0;
}
