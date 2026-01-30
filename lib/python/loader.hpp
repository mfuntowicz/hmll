#ifndef PYHMLL_FETCHER_HPP
#define PYHMLL_FETCHER_HPP

#include <memory>
#include <vector>
#include <hmll/loader.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "hmll/hmll.h"

namespace nb = nanobind;

class WeightLoader
{
    std::unique_ptr<hmll_t> ctx_;
    std::vector<hmll_source_t> srcs_;

public:
    static std::unique_ptr<WeightLoader> from_paths(const std::vector<std::string>& paths, hmll_device_t device);

    WeightLoader(WeightLoader&&) = default;
    WeightLoader& operator=(WeightLoader&&) = default;
    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;
    WeightLoader(std::vector<hmll_source_t> srcs, hmll_device_t device, std::unique_ptr<hmll_t> ctx, hmll_fetcher_kind_t backend = HMLL_FETCHER_AUTO);
    WeightLoader(std::vector<hmll_source_t> srcs, hmll_device_t device, hmll_fetcher_kind_t backend = HMLL_FETCHER_AUTO);

    [[nodiscard]]
    hmll_t* context() const;

    [[nodiscard]]
    hmll_device_t device() const;

    [[nodiscard]]
    hmll_fetcher_kind_t kind() const;

    [[nodiscard]] nb::ndarray<nb::c_contig>
    afetch(int iofile, size_t start, size_t end, hmll_dtype_t dtype, const size_t* shape, uint8_t rank) const;

    [[nodiscard]] size_t
    fetch(int iofile, size_t offset, uintptr_t dst, size_t size) const;

    [[nodiscard]] nb::ndarray<nb::ndim<1>, nb::c_contig>
    fetchv(int iofile, const std::vector<std::tuple<size_t, size_t>>& ranges, hmll_dtype_t dtype) const;
};

#endif // PYHMLL_FETCHER_HPP
