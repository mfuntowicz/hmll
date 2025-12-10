#include "hmll/unix/mmap.h"
#include "hmll/status.h"

#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

hmll_status_t hmll_open_mmap(const char *path, hmll_context_t *ctx)
{
    hmll_status_t result = {0};
    const int fd = open(path, O_RDONLY);
    if (fd == -1) {
        result.what = HMLL_FILE_NOT_FOUND;
        result.message = path;
        return result;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        result.what = HMLL_FILE_NOT_FOUND;
        result.message = path;
        goto close_fd_and_return;
    }

    if (sb.st_size == 0) {
        result.what = HMLL_FILE_EMPTY;
        result.message = path;
        goto close_fd_and_return;
    }

    // 3. Map the file into memory
    // arguments: addr, length, prot, flags, fd, offset
    char *content = mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (content == MAP_FAILED) {
        result.what = HMLL_FILE_MMAP_FAILED;
        result.message = path;
        goto close_fd_and_return;
    }

    ctx->source.fd = fd;
    ctx->source.kind = HMLL_SOURCE_MMAP;
    ctx->source.content = content;
    ctx->source.size = sb.st_size;

    return HMLL_SUCCEEDED;

close_fd_and_return:
    close(fd);
    return result;
}

hmll_status_t hmll_close_mmap(hmll_context_t *ctx)
{
    if (ctx && ctx->source.kind == HMLL_SOURCE_MMAP && ctx->source.size > 0) {
        munmap(ctx->source.content, ctx->source.size);
        ctx->source.kind = HMLL_SOURCE_UNDEFINED;
        ctx->source.size = 0;
    }

    return HMLL_SUCCEEDED;
}

