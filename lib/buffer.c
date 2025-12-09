//
// Created by mfuntowicz on 12/2/25.
//
#include <stdlib.h>

#include "hmll/status.h"
#include "hmll/types.h"

// TODO: Defer this to the system to retrieve the actual block size
#define BLOCK_ALIGNMENT 4096

hmll_status_t hmll_get_io_buffer(const hmll_device_t device, void **ptr, const size_t size)
{
    switch (device)
    {
    default:
        if ((*ptr = aligned_alloc(BLOCK_ALIGNMENT, size)) == nullptr)
            return (hmll_status_t){HMLL_ALLOCATION_FAILED, "Failed to allocate aligned device memory"};
        return HMLL_SUCCEEDED;
    }
}