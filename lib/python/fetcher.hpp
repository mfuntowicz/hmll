#ifndef PYHMLL_FETCHER_HPP
#define PYHMLL_FETCHER_HPP

#include <memory>
#include <utility>
#include <vector>
#include <hmll/fetcher.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "hmll/hmll.h"

namespace nb = nanobind;

class HmllContext;
class Fetcher
{
    std::unique_ptr<hmll_t> ctx_;
    std::vector<hmll_source_t> srcs_;

public:
    static std::unique_ptr<Fetcher> from_paths(const std::vector<std::string>& paths, hmll_device_t device);

    Fetcher(Fetcher&&) = default;
    Fetcher& operator=(Fetcher&&) = default;
    Fetcher(const Fetcher&) = delete;
    Fetcher& operator=(const Fetcher&) = delete;
    explicit Fetcher(std::unique_ptr<hmll_t> ctx, std::vector<hmll_source_t>& srcs, hmll_device_t device);

    [[nodiscard]]
    hmll_device_t device() const;

    [[nodiscard]]
    hmll_fetcher_kind_t kind() const;

    [[nodiscard]]
    nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> fetch(size_t start, size_t end, int iofile) const;
};

#endif // PYHMLL_FETCHER_HPP
