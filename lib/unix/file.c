#include <errno.h>
#include <stdlib.h>
#include <sys/file.h>
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

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        error = HMLL_SYS_ERR(errno);
        goto close_fd_then_exit;
    }

    if (sb.st_size == 0) {
        error = HMLL_ERR(HMLL_ERR_FILE_EMPTY);
        goto close_fd_then_exit;
    }

    src->fd = fd;
    src->size = sb.st_size;

    return HMLL_OK;

close_fd_then_exit:
    close(fd);

exit:
    return error;
}


void hmll_source_close(const struct hmll_source *src)
{
    if (src != NULL && src->fd > 0)
        close(src->fd);
}
