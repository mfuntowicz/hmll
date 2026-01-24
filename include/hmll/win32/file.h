#ifndef HMLL_WIN_FILE_H
#define HMLL_WIN_FILE_H
#include <windows.h>

struct hmll_source {
    HANDLE handle;
    LONGLONG size;
};
typedef struct hmll_source hmll_source_t;

#endif // HMLL_WIN_FILE_H

