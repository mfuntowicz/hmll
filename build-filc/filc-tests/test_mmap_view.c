/*
 * test_mmap_view.c - Test hmll zero-copy mmap view functionality
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>
#include <hmll/memory.h>

#define TEST_FILE "test_mmap_view_data.bin"
#define TEST_SIZE (64 * 1024)

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

    printf("=== hmll Mmap View Tests ===\n\n");

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

    /* Test 1: Get mmap view of file start */
    printf("[TEST] Getting mmap view (0-4096)...\n");
    struct hmll_iobuf view1 = {0};
    struct hmll_range range1 = {0, 4096};
    err = hmll_get_mmap_view(&ctx, 0, range1, &view1);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup;
    }
    printf("  PASS: Got view at %p, size=%zu\n", view1.ptr, view1.size);

    /* Verify data through view (read-only) */
    char *data = (char *)view1.ptr;
    int errors = 0;
    for (size_t i = 0; i < 4096; i++) {
        if (data[i] != (char)(i & 0xFF)) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  FAIL: %d verification errors\n", errors);
        status = 1;
    } else {
        printf("  PASS: Data verified through view\n\n");
    }

    /* Test 2: Get mmap view of middle section */
    printf("[TEST] Getting mmap view (16384-32768)...\n");
    struct hmll_iobuf view2 = {0};
    struct hmll_range range2 = {16384, 32768};
    err = hmll_get_mmap_view(&ctx, 0, range2, &view2);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup;
    }
    printf("  PASS: Got view at %p, size=%zu\n", view2.ptr, view2.size);

    data = (char *)view2.ptr;
    errors = 0;
    for (size_t i = 0; i < 16384; i++) {
        if (data[i] != (char)((16384 + i) & 0xFF)) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("  FAIL: %d verification errors\n", errors);
        status = 1;
    } else {
        printf("  PASS: Middle section verified\n\n");
    }

    /* Test 3: Get view of entire file */
    printf("[TEST] Getting mmap view of entire file...\n");
    struct hmll_iobuf view3 = {0};
    struct hmll_range range3 = {0, TEST_SIZE};
    err = hmll_get_mmap_view(&ctx, 0, range3, &view3);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup;
    }
    printf("  PASS: Got full view at %p, size=%zu\n", view3.ptr, view3.size);

    data = (char *)view3.ptr;
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
        printf("  PASS: Entire file verified through view\n\n");
    }

cleanup:
    printf("[CLEANUP] Cleaning up...\n");
    hmll_destroy(&ctx);
    hmll_source_close(&src);
    remove(TEST_FILE);

    printf("\n=== Tests Complete (status=%d) ===\n", status);
    return status;
}
