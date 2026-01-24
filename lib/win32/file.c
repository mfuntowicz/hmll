#include "hmll/win32/file.h"
#include "hmll/hmll.h"

struct hmll_error hmll_source_open(const char *path, struct hmll_source *src)
{
    struct hmll_error error = HMLL_OK;

    HANDLE handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (!handle){
        error = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);
        goto close_fd_then_exit;
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle, &size)) {
        error = HMLL_ERR(HMLL_ERR_FILE_EMPTY);
        goto close_fd_then_exit;
    }

    src->handle = handle;
    src->size = size.QuadPart;

    return HMLL_OK;

close_fd_then_exit:
    CloseHandle(handle);
    return error;
}


void hmll_source_close(const struct hmll_source *src)
{
    if (src != NULL && src->handle)
        CloseHandle(src->handle);
}
