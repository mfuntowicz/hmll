#ifndef HMLL_UNIX_FILE_H
#define HMLL_UNIX_FILE_H

struct hmll_source {
    int fd;
    size_t size;
};
typedef struct hmll_source hmll_source_t;

#endif // HMLL_UNIX_FILE_H

