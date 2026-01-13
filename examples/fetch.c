#include <stdio.h>
#include <stdlib.h>
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

    struct hmll_iobuf buf = hmll_get_buffer_for_range(&ctx, HMLL_DEVICE_CUDA, (struct hmll_range){0, 1024});
    ssize_t res = hmll_fetch(&ctx, 0, &buf, (struct hmll_range){0, 1024});
    if (hmll_check(ctx.error)) {
        fprintf(stderr, "Failed to fetch tensor: %s\n", hmll_strerr(ctx.error));
        status = 4;
        goto clean;
    }

    printf("Successfully fetched tensor\n");
clean:
    hmll_destroy(&ctx);

exit:
    return status;
}
