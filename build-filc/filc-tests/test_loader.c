/*
 * test_loader.c - Test hmll loader initialization and fetching
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>
#include <hmll/memory.h>

#define TEST_FILE "test_loader_data.bin"
#define TEST_SIZE (64 * 1024)  /* 64KB */

static int create_test_file(const char *path, size_t size)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    char *data = malloc(size);
    if (!data) {
        fclose(f);
        return -1;
    }

    /* Fill with recognizable pattern */
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

    printf("=== hmll Loader Tests ===\n\n");

    /* Create test file */
    printf("[SETUP] Creating test file (%d KB)...\n", TEST_SIZE / 1024);
    if (create_test_file(TEST_FILE, TEST_SIZE) != 0) {
        printf("  FAIL: Could not create test file\n");
        return 1;
    }
    printf("  Created %s\n\n", TEST_FILE);

    /* Open source */
    printf("[SETUP] Opening source...\n");
    err = hmll_source_open(TEST_FILE, &src);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup_file;
    }
    printf("  Opened: fd=%d, size=%zu\n\n", src.fd, src.size);

    /* Test 1: Initialize loader with mmap backend */
    printf("[TEST] Initializing loader (mmap backend, CPU)...\n");
    err = hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup_src;
    }
    printf("  PASS: num_sources=%zu\n\n", ctx.num_sources);

    /* Test 2: Fetch small range */
    printf("[TEST] Fetching small range (0-1024)...\n");
    struct hmll_iobuf buf1 = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 1024, HMLL_MEM_DEVICE);
    if (buf1.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer\n");
        status = 1;
        goto cleanup_ctx;
    }

    struct hmll_range range1 = {0, 1024};
    ssize_t fetched = hmll_fetch(&ctx, 0, &buf1, range1);
    if (hmll_check(ctx.error) || fetched < 0) {
        printf("  FAIL: %s\n", hmll_strerr(ctx.error));
        status = 1;
        hmll_free_buffer(&buf1);
        goto cleanup_ctx;
    }
    printf("  PASS: Fetched %zd bytes\n", fetched);

    /* Verify data */
    char *data = (char *)buf1.ptr;
    int errors = 0;
    for (size_t i = 0; i < 1024; i++) {
        if (data[i] != (char)(i & 0xFF)) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  FAIL: %d verification errors\n", errors);
        status = 1;
    } else {
        printf("  PASS: Data verified correctly\n\n");
    }
    hmll_free_buffer(&buf1);

    /* Test 3: Fetch from middle of file */
    printf("[TEST] Fetching from middle (32768-36864)...\n");
    struct hmll_iobuf buf2 = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 4096, HMLL_MEM_DEVICE);
    if (buf2.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer\n");
        status = 1;
        goto cleanup_ctx;
    }

    struct hmll_range range2 = {32768, 36864};
    fetched = hmll_fetch(&ctx, 0, &buf2, range2);
    if (hmll_check(ctx.error) || fetched < 0) {
        printf("  FAIL: %s\n", hmll_strerr(ctx.error));
        status = 1;
        hmll_free_buffer(&buf2);
        goto cleanup_ctx;
    }
    printf("  PASS: Fetched %zd bytes\n", fetched);

    /* Verify data from middle */
    data = (char *)buf2.ptr;
    errors = 0;
    for (size_t i = 0; i < 4096; i++) {
        char expected = (char)((32768 + i) & 0xFF);
        if (data[i] != expected) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  FAIL: %d verification errors\n", errors);
        status = 1;
    } else {
        printf("  PASS: Data verified correctly\n\n");
    }
    hmll_free_buffer(&buf2);

    /* Test 4: Fetch entire file */
    printf("[TEST] Fetching entire file (0-%d)...\n", TEST_SIZE);
    struct hmll_iobuf buf3 = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, TEST_SIZE, HMLL_MEM_DEVICE);
    if (buf3.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer\n");
        status = 1;
        goto cleanup_ctx;
    }

    struct hmll_range range3 = {0, TEST_SIZE};
    fetched = hmll_fetch(&ctx, 0, &buf3, range3);
    if (hmll_check(ctx.error) || fetched < 0) {
        printf("  FAIL: %s\n", hmll_strerr(ctx.error));
        status = 1;
        hmll_free_buffer(&buf3);
        goto cleanup_ctx;
    }
    printf("  PASS: Fetched %zd bytes\n", fetched);

    /* Verify entire file */
    data = (char *)buf3.ptr;
    errors = 0;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (data[i] != (char)(i & 0xFF)) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  FAIL: %d verification errors\n", errors);
        status = 1;
    } else {
        printf("  PASS: All %d bytes verified\n\n", TEST_SIZE);
    }
    hmll_free_buffer(&buf3);

cleanup_ctx:
    printf("[CLEANUP] Destroying context...\n");
    hmll_destroy(&ctx);

cleanup_src:
    printf("[CLEANUP] Closing source...\n");
    hmll_source_close(&src);

cleanup_file:
    remove(TEST_FILE);

    printf("\n=== Tests Complete (status=%d) ===\n", status);
    return status;
}
