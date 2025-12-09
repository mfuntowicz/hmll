#ifndef HMLL_FETCHER_H
#define HMLL_FETCHER_H


struct hmll_fetch_range {
    size_t start;
    size_t end;
};
typedef struct hmll_fetch_range hmll_fetch_range_t;

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/fetcher.h"
#endif

#endif // HMLL_FETCHER_H
