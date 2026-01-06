#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hmll/hmll.h"

struct hmll_source hmll_open_mapped(hmll_context_t *ctx, const char *path)
{
    if (hmll_has_error(hmll_get_error(ctx)))
        goto return_null;

    int fd;
    if ((fd = open(path, O_RDONLY | O_DIRECT)) == -1) {
        ctx->error = HMLL_ERR_FILE_NOT_FOUND;
        goto return_null;
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

    return (struct hmll_source) {HMLL_SOURCE_MMAP, fd, content, sb.st_size};

close_fd_and_return_error:
    close(fd);

return_null:
    return (struct hmll_source) {0};
}

void hmll_close_mapped(struct hmll_source src)
{
    if (src.kind == HMLL_SOURCE_MMAP &&src.size > 0) {
        munmap(src.content, src.size);
        src.kind = HMLL_SOURCE_UNDEFINED;
        src.size = 0;
    }
}


