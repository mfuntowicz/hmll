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

    if (size.QuadPart == 0) {
        error = HMLL_ERR(HMLL_ERR_FILE_EMPTY);
        goto close_fd_then_exit;
    }

    unsigned char *content = hmll_mmap_file(handle, (size_t)size.QuadPart);
    if (content == NULL) {
        error = HMLL_ERR(HMLL_ERR_MMAP_FAILED);
        goto close_fd_then_exit;
    }

    src->handle = handle;
    src->size = (size_t)size.QuadPart;
    src->content = content;

    return HMLL_OK;

close_fd_then_exit:
    CloseHandle(handle);
    return error;
}


void hmll_source_close(struct hmll_source *src)
{
    if (src && src->handle) {
        CloseHandle(src->handle);
        src->handle = NULL;  // Mark as closed
    }
}

void hmll_source_cleanup(struct hmll_source *src)
{
    if (src) {
        hmll_source_close(src);
        if (src->content && src->size > 0) {
            UnmapViewOfFile(src->content);
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


unsigned char *hmll_mmap_file(HANDLE handle, size_t size)
{
    unsigned char *buf;
    HANDLE h_mapping;

    // Try to create file mapping with large pages first (requires SeManageVolumePrivilege)
    h_mapping = CreateFileMappingA(
        handle,
        NULL,
        PAGE_READONLY | SEC_LARGE_PAGES,
        (DWORD)((size >> 32) & 0xFFFFFFFF),
        (DWORD)(size & 0xFFFFFFFF),
        NULL
    );

    if (h_mapping != NULL) {
        buf = MapViewOfFile(h_mapping, FILE_MAP_READ, 0, 0, size);
        CloseHandle(h_mapping);
        if (buf != NULL) {
            return buf;
        }
    }

    // Fallback to regular pages
    h_mapping = CreateFileMappingA(
        handle,
        NULL,
        PAGE_READONLY,
        (DWORD)((size >> 32) & 0xFFFFFFFF),
        (DWORD)(size & 0xFFFFFFFF),
        NULL
    );

    if (h_mapping == NULL) {
        return NULL;
    }

    buf = MapViewOfFile(h_mapping, FILE_MAP_READ, 0, 0, size);
    CloseHandle(h_mapping);

    return buf;
}
