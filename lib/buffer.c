//
// Created by mfuntowicz on 12/2/25.
//
#include <stdlib.h>
#include <sys/mman.h>

#include "hmll/status.h"
#include "hmll/types.h"

#define BLOCK_ALIGNMENT 4096

hmll_status_t hmll_get_io_buffer(const hmll_device_t device, void **ptr, const size_t size)
{
    switch (device)
    {
    case HMLL_DEVICE_CPU:
        // Try to allocate with 2MB huge pages first
        *ptr = mmap(0, size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                    -1, 0);

        // If huge pages fail, fall back to regular anonymous mmap
        if (*ptr == MAP_FAILED) {
            *ptr = mmap(0, size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);

            if (*ptr == MAP_FAILED)
                return (hmll_status_t){HMLL_ALLOCATION_FAILED, "Failed to allocate device memory"};

        }

        return HMLL_SUCCEEDED;

    default:
        return (hmll_status_t){HMLL_ALLOCATION_FAILED, "Unsupported device type"};
    }
}