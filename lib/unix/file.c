#include <stdlib.h>

#include "hmll/hmll.h"
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

enum hmll_error_code hmll_open(struct hmll_context *ctx, struct hmll_source **src, const char *path)
{
    HMLL_CHECK(ctx);

    int fd;
    if ((fd = open(path, O_RDONLY | O_DIRECT)) == -1) {
        ctx->error = HMLL_ERR_FILE_NOT_FOUND;
        goto failure;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        ctx->error = HMLL_ERR_FILE_NOT_FOUND;
        goto close_fd_then_failure;
    }

    if (sb.st_size == 0) {
        ctx->error = HMLL_ERR_FILE_EMPTY;
        goto close_fd_then_failure;
    }

    *src = calloc(1, sizeof(struct hmll_source));
    (*src)->fd = fd;
    (*src)->size = sb.st_size;
    return HMLL_ERR_SUCCESS;

close_fd_then_failure:
    close(fd);

failure:
    return ctx->error;
}


enum hmll_error_code hmll_close(struct hmll_context *ctx, struct hmll_source src)
{
    HMLL_CHECK(ctx);

    close(src.fd);
    return HMLL_ERR_SUCCESS;
}