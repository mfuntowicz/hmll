/*
 * test_source.c - Test hmll source open/close functionality
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>

#define TEST_FILE "test_source_data.bin"
#define TEST_SIZE 8192

static int create_test_file(const char *path, size_t size)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    char *data = malloc(size);
    if (!data) {
        fclose(f);
        return -1;
    }

    /* Fill with pattern */
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
    struct hmll_source src = {0};
    struct hmll_error err;

    printf("=== hmll Source Tests ===\n\n");

    /* Create test file */
    printf("[TEST] Creating test file...\n");
    if (create_test_file(TEST_FILE, TEST_SIZE) != 0) {
        printf("  FAIL: Could not create test file\n");
        return 1;
    }
    printf("  Created %s (%d bytes)\n\n", TEST_FILE, TEST_SIZE);

    /* Test 1: Open valid file */
    printf("[TEST] Opening valid file...\n");
    err = hmll_source_open(TEST_FILE, &src);
    if (hmll_check(err)) {
        printf("  FAIL: %s\n", hmll_strerr(err));
        status = 1;
        goto cleanup;
    }
    printf("  PASS: fd=%d, size=%zu\n\n", src.fd, src.size);

    if (src.size != TEST_SIZE) {
        printf("  FAIL: Expected size %d, got %zu\n", TEST_SIZE, src.size);
        status = 1;
        goto cleanup;
    }

    /* Test 2: Close source */
    printf("[TEST] Closing source...\n");
    hmll_source_close(&src);
    printf("  PASS: Source closed\n\n");

    /* Test 3: Open non-existent file */
    printf("[TEST] Opening non-existent file...\n");
    err = hmll_source_open("/nonexistent/path/file.bin", &src);
    if (!hmll_check(err)) {
        printf("  FAIL: Should have failed for non-existent file\n");
        status = 1;
        goto cleanup;
    }
    printf("  PASS: Correctly returned error: %s\n\n", hmll_strerr(err));

    /* Test 4: Open multiple sources */
    printf("[TEST] Opening multiple sources...\n");
    struct hmll_source sources[3];
    for (int i = 0; i < 3; i++) {
        err = hmll_source_open(TEST_FILE, &sources[i]);
        if (hmll_check(err)) {
            printf("  FAIL: Could not open source %d: %s\n", i, hmll_strerr(err));
            status = 1;
            goto cleanup;
        }
    }
    printf("  PASS: Opened 3 sources\n");

    for (int i = 0; i < 3; i++) {
        hmll_source_close(&sources[i]);
    }
    printf("  PASS: Closed all sources\n\n");

cleanup:
    remove(TEST_FILE);
    printf("=== Tests Complete (status=%d) ===\n", status);
    return status;
}
