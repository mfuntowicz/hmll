#ifndef HMLL_MEMORY_H
#define HMLL_MEMORY_H

#include <stdint.h>
#include "hmll/types.h"


#define HMLL_MEM_STAGING 0
#define HMLL_MEM_DEVICE  1

#define ALIGN_PAGE 4096
#define ALIGN_UP(x, align) (((x) + align - 1) & ~(align - 1))
#define ALIGN_DOWN(x, align) ((x) & ~(align - 1))

static inline int hmll_is_aligned(const uintptr_t addr, const size_t align)
{
    return (addr & (align - 1)) == 0;
}


#endif // HMLL_MEMORY_H
