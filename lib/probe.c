#include <stdlib.h>

#include "hmll/types.h"

struct hmll_probe hmll_get_probe(void)
{
    struct hmll_probe probe = {0};
#if defined(HMLL_CUDA_ENABLED)
    probe.featmask |= (1 << HMLL_FEATCODE_CUDA);
#endif

#if defined(__linux)
#include <linux/version.h>
    probe.featmask |= (1 << HMLL_FEATCODE_IO_URING);
#endif

    return probe;
}

unsigned char hmll_is_feature_supported(struct hmll_probe probe, enum hmll_featcode feat)
{
    return (probe.featmask >> feat) & 0x1;
}