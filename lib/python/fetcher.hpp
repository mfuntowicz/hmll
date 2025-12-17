#ifndef PYHMLL_FETCHER_HPP
#define PYHMLL_FETCHER_HPP

#include "hmll/fetcher.h"

class HmllFetcher
{
    hmll_fetcher fetcher_;

public:
    explicit HmllFetcher(hmll_fetcher fetcher): fetcher_(fetcher) {}

    [[nodiscard]]
    hmll_device_t device() const;

    [[nodiscard]]
    hmll_fetcher_kind_t kind() const;
};

#endif // PYHMLL_FETCHER_HPP
