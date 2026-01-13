#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hmll/hmll.h>
#include <hmll/memory.h>

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "No file specified.\nInvoke through hmll_safetensors_ex <path/to/safetensors/file>");
        return 1;
    }

    int status = 0;

    struct hmll ctx = {0};
    struct hmll_error err;
    struct hmll_source *src;
    if ((src = calloc(argc - 1, sizeof(struct hmll_source))) == NULL) {
        fprintf(stderr, "Failed to allocated memory to store file sources");
        status = 2;
        goto exit;
    }

    printf("Opening %u files:\n", argc - 1);
    for (size_t i = 0; i < argc - 1; ++i) {
        if (hmll_check((err = hmll_source_open(argv[i + 1], src + i)))) {
            fprintf(stderr, "Failed to open source %s: %s\n", argv[i + 1], hmll_strerr(err));
            free(src);
            status = 3;
            goto exit;
        }
        printf("\t- %s -> %zu bytes (fd: %u)\n", argv[i + 1], src[i].size, src[i].fd);
    }

    err = hmll_loader_init(&ctx, src, argc - 1, HMLL_DEVICE_CUDA, HMLL_FETCHER_AUTO);
    if (hmll_check(err)) {
        fprintf(stderr, "Failed to initialize HMLL: %s\n", hmll_strerr(ctx.error));
        status = 4;
        goto clean;
    }

    printf("Successfully initialized HMLL (n_sources=%zu)\n", ctx.num_sources);

    const struct hmll_range rs[2] = {(struct hmll_range){39304, 604019080 / 2}, (struct hmll_range){604019080 / 2, 604019080}};
    const struct hmll_iobuf b0 = hmll_get_buffer_for_range(&ctx, HMLL_DEVICE_CPU, rs[0]);
    const struct hmll_iobuf b1 = hmll_get_buffer_for_range(&ctx, HMLL_DEVICE_CPU, rs[1]);
    const struct hmll_iobuf bs[2] = {b0, b1};

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    const ssize_t res = hmll_fetchv(&ctx, 0, bs, rs, 2);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    if (res < 0) {
        fprintf(stderr, "Failed to fetch tensor: %s\n", hmll_strerr(ctx.error));
        status = 4;
        goto clean;
    }

    const double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    const double total_bytes = (rs[0].end - rs[0].start) + (rs[1].end - rs[1].start);
    const double throughput_gbps = (total_bytes / elapsed) / 1e9;

    printf("Successfully fetched %zu ranges (%.2f MB) in %.3f seconds (%.2f GB/s)\n",
           (size_t)2, total_bytes / 1e6, elapsed, throughput_gbps);
clean:
    hmll_destroy(&ctx);

exit:
    return status;
}
