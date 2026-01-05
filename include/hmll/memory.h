#ifndef HMLL_MEMORY_H
#define HMLL_MEMORY_H

#include <stdint.h>
#include "hmll/types.h"

#define ALIGN_PAGE 4096
#define ALIGN_UP(x, align) (((x) + align - 1) & ~(align - 1))
#define ALIGN_DOWN(x, align) ((x) & ~(align - 1))

static inline int hmll_is_aligned(const uintptr_t addr, const size_t align)
{
    return (addr & (align - 1)) == 0;
}

void *hmll_get_buffer(struct hmll *ctx, enum hmll_device device, size_t size);
void *hmll_get_io_buffer(struct hmll *ctx, enum hmll_device device, size_t size);
struct hmll_iobuf hmll_get_buffer_for_range(struct hmll *ctx, enum hmll_device device, struct hmll_range range);


#endif // HMLL_MEMORY_H
