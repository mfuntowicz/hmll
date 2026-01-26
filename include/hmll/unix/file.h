#ifndef HMLL_UNIX_FILE_H
#define HMLL_UNIX_FILE_H

#include <stdio.h>
#include <unistd.h>

struct hmll_source {
    int fd;
    size_t size;
};
typedef struct hmll_source hmll_source_t;

// Duplicate the fd and return a FILE* that can be safely closed
// without affecting the original fd in hmll_source
static FILE *hmll_get_file_from_fd(const hmll_source_t source) {
    int dup_fd = dup(source.fd);
    if (dup_fd == -1) {
        return NULL;
    }

    FILE *fp = fdopen(dup_fd, "rb");
    if (!fp) {
        close(dup_fd);
        return NULL;
    }

    return fp;
}

#endif // HMLL_UNIX_FILE_H

