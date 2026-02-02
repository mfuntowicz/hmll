#ifndef HMLL_WIN_FILE_H
#define HMLL_WIN_FILE_H
#include <fcntl.h>
#include <stdio.h>
#include <io.h>
#include <windows.h>

struct hmll_source {
    HANDLE handle;
    size_t size;
    unsigned char *content;
};
typedef struct hmll_source hmll_source_t;

// Duplicate the HANDLE and return a FILE* that can be safely closed
// without affecting the original handle in hmll_source
static inline FILE *hmll_get_file_from_fd(hmll_source_t source) {
    HANDLE dup_handle;
    if (!DuplicateHandle(
        GetCurrentProcess(),
        source.handle,
        GetCurrentProcess(),
        &dup_handle,
        0,
        FALSE,
        DUPLICATE_SAME_ACCESS)) {
        return NULL;
    }

    int fd = _open_osfhandle((intptr_t)dup_handle, _O_RDONLY | _O_BINARY);
    if (fd == -1) {
        CloseHandle(dup_handle);
        return NULL;
    }

    FILE *fp = _fdopen(fd, "rb");
    if (!fp) {
        _close(fd);  // This will also close dup_handle
        return NULL;
    }

    return fp;
}

// Memory map a file handle, attempting to use large pages when possible
unsigned char *hmll_mmap_file(HANDLE handle, size_t size);

#endif // HMLL_WIN_FILE_H

