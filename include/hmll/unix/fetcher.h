#ifndef HMLL_UNIX_FETCHER_H
#define HMLL_UNIX_FETCHER_H

enum hmll_fetcher_kind
{
    HMLL_FETCHER_AUTO,
    HMLL_FETCHER_IO_URING
};
typedef enum hmll_fetcher_kind hmll_fetcher_kind_t;

#endif // HMLL_UNIX_FETCHER_H
