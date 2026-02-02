/*
 * test_fetchv.c - Test hmll vectorized fetch functionality
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>
#include <hmll/memory.h>

#define TEST_FILE "test_fetchv_data.bin"
#define TEST_SIZE (128 * 1024)  /* 128KB */

static int create_test_file(const char *path, size_t size)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    char *data = malloc(size);
    if (!data) {
        fclose(f);
        return -1;
    }

    for (size_t i = 0; i < size; i++) {
        data[i] = (char)(i & 0xFF);
    }

    size_t written = fwrite(data, 1, size, f);
    free(data);
    fclose(f);

    return written == size ? 0 : -1;
}

int main(void)
{
    int status = 0;
    struct hmll ctx = {0};
    struct hmll_source src = {0};
    struct hmll_error err;

    printf("=== hmll Vectorized Fetch Tests ===\n\n");

    /* Setup */
    printf("[SETUP] Creating test file (%d KB)...\n", TEST_SIZE / 1024);
    if (create_test_file(TEST_FILE, TEST_SIZE) != 0) {
        printf("  FAIL: Could not create test file\n");
        return 1;
    }

    err = hmll_source_open(TEST_FILE, &src);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        remove(TEST_FILE);
        return 1;
    }

    err = hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        hmll_source_close(&src);
        remove(TEST_FILE);
        return 1;
    }
    printf("  Setup complete\n\n");

    /* Test 1: Fetch two ranges */
    printf("[TEST] Fetching 2 ranges...\n");
    struct hmll_range ranges2[2] = {
        {0, 4096},
        {8192, 12288}
    };
    struct hmll_iobuf bufs2[2] = {
        hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 4096, HMLL_MEM_DEVICE),
        hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 4096, HMLL_MEM_DEVICE)
    };

    if (bufs2[0].ptr == NULL || bufs2[1].ptr == NULL) {
        printf("  FAIL: Could not allocate buffers\n");
        status = 1;
        goto cleanup;
    }

    ssize_t fetched = hmll_fetchv(&ctx, 0, bufs2, ranges2, 2);
    if (hmll_check(ctx.error) || fetched < 0) {
        printf("  FAIL: %s\n", hmll_strerr(ctx.error));
        status = 1;
    } else {
        printf("  PASS: Fetched %zd bytes total\n", fetched);

        /* Verify first range */
        char *data = (char *)bufs2[0].ptr;
        int errors = 0;
        for (size_t i = 0; i < 4096; i++) {
            if (data[i] != (char)(i & 0xFF)) errors++;
        }

        /* Verify second range */
        data = (char *)bufs2[1].ptr;
        for (size_t i = 0; i < 4096; i++) {
            if (data[i] != (char)((8192 + i) & 0xFF)) errors++;
        }

        if (errors > 0) {
            printf("  FAIL: %d verification errors\n", errors);
            status = 1;
        } else {
            printf("  PASS: Both ranges verified\n\n");
        }
    }

    hmll_free_buffer(&bufs2[0]);
    hmll_free_buffer(&bufs2[1]);

    /* Test 2: Fetch multiple non-contiguous ranges */
    printf("[TEST] Fetching 4 non-contiguous ranges...\n");
    struct hmll_range ranges4[4] = {
        {0, 1024},
        {16384, 17408},
        {32768, 33792},
        {65536, 66560}
    };
    struct hmll_iobuf bufs4[4];
    for (int i = 0; i < 4; i++) {
        bufs4[i] = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 1024, HMLL_MEM_DEVICE);
        if (bufs4[i].ptr == NULL) {
            printf("  FAIL: Could not allocate buffer %d\n", i);
            status = 1;
            goto cleanup;
        }
    }

    fetched = hmll_fetchv(&ctx, 0, bufs4, ranges4, 4);
    if (hmll_check(ctx.error) || fetched < 0) {
        printf("  FAIL: %s\n", hmll_strerr(ctx.error));
        status = 1;
    } else {
        printf("  PASS: Fetched %zd bytes from 4 ranges\n", fetched);

        int errors = 0;
        for (int r = 0; r < 4; r++) {
            char *data = (char *)bufs4[r].ptr;
            size_t start = ranges4[r].start;
            for (size_t i = 0; i < 1024; i++) {
                if (data[i] != (char)((start + i) & 0xFF)) errors++;
            }
        }

        if (errors > 0) {
            printf("  FAIL: %d verification errors\n", errors);
            status = 1;
        } else {
            printf("  PASS: All 4 ranges verified\n\n");
        }
    }

    for (int i = 0; i < 4; i++) {
        hmll_free_buffer(&bufs4[i]);
    }

cleanup:
    printf("[CLEANUP] Cleaning up...\n");
    hmll_destroy(&ctx);
    hmll_source_close(&src);
    remove(TEST_FILE);

    printf("\n=== Tests Complete (status=%d) ===\n", status);
    return status;
}
