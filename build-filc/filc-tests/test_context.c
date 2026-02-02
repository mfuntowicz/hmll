/*
 * test_context.c - Test hmll context operations
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>
#include <hmll/memory.h>

#define TEST_FILE "test_context_data.bin"
#define TEST_SIZE (32 * 1024)

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
    struct hmll ctx_clone = {0};
    struct hmll_source src = {0};
    struct hmll_error err;

    printf("=== hmll Context Tests ===\n\n");

    /* Setup */
    printf("[SETUP] Creating test file...\n");
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

    /* Test 1: Clone context */
    printf("[TEST] Cloning context...\n");
    err = hmll_clone_context(&ctx_clone, &ctx);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup;
    }
    printf("  PASS: Cloned context\n");
    printf("  Original: num_sources=%zu\n", ctx.num_sources);
    printf("  Clone: num_sources=%zu\n\n", ctx_clone.num_sources);

    /* Test 2: Verify clone has same properties */
    printf("[TEST] Verifying clone properties...\n");
    if (ctx_clone.num_sources != ctx.num_sources) {
        printf("  FAIL: num_sources mismatch\n");
        status = 1;
    } else if (ctx_clone.sources != ctx.sources) {
        printf("  FAIL: sources pointer mismatch\n");
        status = 1;
    } else {
        printf("  PASS: Clone properties match original\n\n");
    }

    /* Test 3: Use original context after clone */
    printf("[TEST] Using original context after clone...\n");
    struct hmll_iobuf buf1 = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 1024, HMLL_MEM_DEVICE);
    if (buf1.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer\n");
        status = 1;
        goto cleanup;
    }

    struct hmll_range range = {0, 1024};
    ssize_t fetched = hmll_fetch(&ctx, 0, &buf1, range);
    if (hmll_check(ctx.error) || fetched < 0) {
        printf("  FAIL: %s\n", hmll_strerr(ctx.error));
        status = 1;
    } else {
        printf("  PASS: Original context works (fetched %zd bytes)\n\n", fetched);
    }
    hmll_free_buffer(&buf1);

    /* Test 4: Clone with NULL pointers (error cases) */
    printf("[TEST] Clone with NULL dst...\n");
    err = hmll_clone_context(NULL, &ctx);
    if (!hmll_check(err)) {
        printf("  FAIL: Should have failed with NULL dst\n");
        status = 1;
    } else {
        printf("  PASS: Correctly returned error\n\n");
    }

    printf("[TEST] Clone with NULL src...\n");
    err = hmll_clone_context(&ctx_clone, NULL);
    if (!hmll_check(err)) {
        printf("  FAIL: Should have failed with NULL src\n");
        status = 1;
    } else {
        printf("  PASS: Correctly returned error\n\n");
    }

    /* Test 5: Destroy original, clone should still have valid source ref */
    printf("[TEST] Destroying original context...\n");
    hmll_destroy(&ctx);
    printf("  PASS: Original context destroyed\n");
    printf("  Clone still references: num_sources=%zu\n\n", ctx_clone.num_sources);

cleanup:
    hmll_destroy(&ctx);
    hmll_source_close(&src);
    remove(TEST_FILE);

    printf("=== Tests Complete (status=%d) ===\n", status);
    return status;
}
