#include "hmll/unix/mmap.h"

#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hmll/hmll.h"

enum hmll_error_code hmll_open_mmap(const char *path, hmll_context_t *ctx)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        goto return_error;

    const int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd == -1) {
        ctx->error = HMLL_ERR_FILE_NOT_FOUND;
        goto return_error;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        ctx->error = HMLL_ERR_FILE_NOT_FOUND;
        goto close_fd_and_return_error;
    }

    if (sb.st_size == 0) {
        ctx->error = HMLL_ERR_FILE_EMPTY;
        goto close_fd_and_return_error;
    }

    // 3. Map the file into memory
    // arguments: addr, length, prot, flags, fd, offset
    char *content = mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (content == MAP_FAILED) {
        ctx->error = HMLL_ERR_MMAP_FAILED;
        goto close_fd_and_return_error;
    }

    ctx->source.fd = fd;
    ctx->source.kind = HMLL_SOURCE_MMAP;
    ctx->source.content = content;
    ctx->source.size = sb.st_size;

    return HMLL_ERR_SUCCESS;

close_fd_and_return_error:
    close(fd);

return_error:
    return ctx->error;
}

void hmll_close_mmap(hmll_context_t *ctx)
{
    if (ctx && ctx->source.kind == HMLL_SOURCE_MMAP && ctx->source.size > 0) {
        munmap(ctx->source.content, ctx->source.size);
        ctx->source.kind = HMLL_SOURCE_UNDEFINED;
        ctx->source.size = 0;
    }
}

