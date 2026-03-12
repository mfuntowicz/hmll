#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "hmll/hmll.h"

struct hmll_error hmll_source_open(const char *path, struct hmll_source *src)
{
    struct hmll_error error = HMLL_OK;

    int fd;
    if ((fd = open(path, O_RDONLY)) == -1) {
        error = HMLL_SYS_ERR(errno);
        goto exit;
    }

#if defined(__APPLE__)
    // Disable file caching on macOS (similar to O_DIRECT on Linux)
    fcntl(fd, F_NOCACHE, 1);
#endif

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        error = HMLL_SYS_ERR(errno);
        goto close_fd_then_exit;
    }

    if (sb.st_size == 0) {
        error = HMLL_ERR(HMLL_ERR_FILE_EMPTY);
        goto close_fd_then_exit;
    }

    const unsigned char *content = hmll_mmap_file(fd, sb.st_size);
    if (content == NULL) {
        error = HMLL_ERR(HMLL_ERR_MMAP_FAILED);
        goto close_fd_then_exit;
    }

    src->fd = fd;
    src->size = sb.st_size;
    src->content = content;

#if defined(__linux__)
    src->d_fd = open(path, O_RDONLY | O_DIRECT);
#else
    src->d_fd = -1;
#endif

    return HMLL_OK;

close_fd_then_exit:
    close(fd);

exit:
    return error;
}


void hmll_source_close(struct hmll_source *src)
{
    if (src && src->fd != -1) {
        close(src->fd);
        src->fd = -1;
    }
    if (src && src->d_fd != -1) {
        close(src->d_fd);
        src->d_fd = -1;
    }
}

void hmll_source_cleanup(struct hmll_source *src)
{
    if (src) {
        hmll_source_close(src);
        if (src->content && src->size > 0) {
            munmap((void *)src->content, src->size);
            src->size = 0;
        }
    }
}

void hmll_source_free(struct hmll_source *src)
{
    if (src) {
        hmll_source_cleanup(src);
        free(src);
    }
}

unsigned char *hmll_mmap_file(const int fd, const size_t size)
{
    unsigned char *buf;

#ifdef MAP_HUGETLB
    buf = mmap(0, size, PROT_READ, MAP_PRIVATE | MAP_HUGETLB, fd, 0);
    if (buf != MAP_FAILED) {
        return buf;
    }
#endif

    buf = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) {
        return NULL;
    }

#ifdef MADV_HUGEPAGE
    madvise(buf, size, MADV_HUGEPAGE);
#endif

    return buf;
}
