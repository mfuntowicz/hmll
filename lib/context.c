//
// Created by mfuntowicz on 12/1/25.
//

#include <stdlib.h>
#include "hmll/hmll.h"

void hmll_destroy(struct hmll *ctx)
{
    if (ctx) {
        if (ctx->num_sources > 0) {
            for (size_t i = 0; i < ctx->num_sources; ++i)
                hmll_source_close(ctx->sources + i);

            free(ctx->sources);
            ctx->num_sources = 0;
        }

        if (ctx->fetcher) {
            // TODO
        }
    }
}

