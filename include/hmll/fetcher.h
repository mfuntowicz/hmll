#ifndef HMLL_FETCHER_H
#define HMLL_FETCHER_H

enum hmll_fetcher_kind;
typedef enum hmll_fetcher_kind hmll_fetcher_kind_t;

#if defined(__linux) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/fetcher.h"
#endif

#endif // HMLL_FETCHER_H
