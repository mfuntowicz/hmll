#ifndef PYHMLL_FETCHER_HPP
#define PYHMLL_FETCHER_HPP

#include <memory>
#include <string>
#include <hmll/fetcher.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

class HmllContext;
class HmllFetcher
{
    hmll_fetcher_t fetcher_;
    nb::object ctx_obj_;  // Keep Python object alive
    hmll_context_t* ctx_ptr_;  // Raw pointer for C API calls

public:
    explicit HmllFetcher(hmll_fetcher fetcher, nb::object ctx_obj, hmll_context_t* ctx_ptr):
        fetcher_(fetcher), ctx_obj_(ctx_obj), ctx_ptr_(ctx_ptr) {}

    [[nodiscard]]
    hmll_device_t device() const;

    [[nodiscard]]
    hmll_fetcher_kind_t kind() const;

    [[nodiscard]]
    nb::ndarray<> fetch_contiguous(const std::string& name) const;
};

#endif // PYHMLL_FETCHER_HPP
