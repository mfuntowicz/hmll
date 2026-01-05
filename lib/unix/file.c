#include "hmll/error.h"
#include "hmll/hmll.h"
#include <errno.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

struct hmll_error hmll_source_open(const char *path, struct hmll_source *src)
{
    enum hmll_status_code error = HMLL_ERR_SUCCESS;

    int fd;
    if ((fd = open(path, O_RDONLY | O_DIRECT)) == -1) {
        error = hmll_error_from_os(errno);
        goto exit;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        error = hmll_error_from_os(errno);
        goto close_fd_then_exit;
    }

    if (sb.st_size == 0) {
        error = hmll_error_from_lib(HMLL_ERR_FILE_EMPTY);
        goto close_fd_then_exit;
    }

    src->fd = fd;
    src->size = sb.st_size;

    return HMLL_OK;

close_fd_then_exit:
    close(fd);

exit:
    return HMLL_SYS_ERR(error);
}


void hmll_source_close(struct hmll_source *src)
{
    if (src != NULL && src->fd > 0)
        close(src->fd);
}