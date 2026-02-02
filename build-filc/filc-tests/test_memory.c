/*
 * test_memory.c - Test hmll memory allocation functionality
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>
#include <hmll/memory.h>

int main(void)
{
    int status = 0;
    struct hmll ctx = {0};

    printf("=== hmll Memory Tests ===\n\n");

    /* Test 1: Allocate small buffer */
    printf("[TEST] Allocating small buffer (1KB)...\n");
    struct hmll_iobuf buf1 = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 1024, HMLL_MEM_DEVICE);
    if (buf1.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer\n");
        return 1;
    }
    printf("  PASS: ptr=%p, size=%zu\n\n", buf1.ptr, buf1.size);

    /* Test 2: Write and read data */
    printf("[TEST] Writing pattern to buffer...\n");
    char *data = (char *)buf1.ptr;
    for (size_t i = 0; i < 1024; i++) {
        data[i] = (char)(i & 0xFF);
    }
    printf("  PASS: Wrote 1024 bytes\n\n");

    printf("[TEST] Verifying pattern...\n");
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
        printf("  PASS: All bytes verified\n\n");
    }

    /* Test 3: Free buffer */
    printf("[TEST] Freeing buffer...\n");
    hmll_free_buffer(&buf1);
    printf("  PASS: Buffer freed\n\n");

    /* Test 4: Allocate larger buffer */
    printf("[TEST] Allocating larger buffer (1MB)...\n");
    struct hmll_iobuf buf2 = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, 1024 * 1024, HMLL_MEM_DEVICE);
    if (buf2.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer\n");
        return 1;
    }
    printf("  PASS: ptr=%p, size=%zu\n\n", buf2.ptr, buf2.size);

    /* Test 5: Fill large buffer */
    printf("[TEST] Filling large buffer...\n");
    memset(buf2.ptr, 0xAB, buf2.size);
    printf("  PASS: Filled %zu bytes\n\n", buf2.size);

    hmll_free_buffer(&buf2);

    /* Test 6: Allocate buffer for range */
    printf("[TEST] Allocating buffer for range...\n");
    struct hmll_range range = {0, 4096};
    struct hmll_iobuf buf3 = hmll_get_buffer_for_range(&ctx, HMLL_DEVICE_CPU, range);
    if (buf3.ptr == NULL) {
        printf("  FAIL: Could not allocate buffer for range\n");
        return 1;
    }
    printf("  PASS: ptr=%p, size=%zu (range: %zu-%zu)\n\n",
           buf3.ptr, buf3.size, range.start, range.end);

    hmll_free_buffer(&buf3);

    /* Test 7: Multiple allocations */
    printf("[TEST] Multiple concurrent allocations...\n");
    struct hmll_iobuf bufs[10];
    for (int i = 0; i < 10; i++) {
        bufs[i] = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, (i + 1) * 1024, HMLL_MEM_DEVICE);
        if (bufs[i].ptr == NULL) {
            printf("  FAIL: Could not allocate buffer %d\n", i);
            status = 1;
            break;
        }
        /* Write to each buffer */
        memset(bufs[i].ptr, i, bufs[i].size);
    }
    printf("  PASS: Allocated 10 buffers\n");

    /* Free in reverse order */
    for (int i = 9; i >= 0; i--) {
        hmll_free_buffer(&bufs[i]);
    }
    printf("  PASS: Freed all buffers\n\n");

    printf("=== Tests Complete (status=%d) ===\n", status);
    return status;
}
