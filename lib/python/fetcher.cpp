#include "fetcher.hpp"
#include <format>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "formatters.hpp"

namespace nb = nanobind;


hmll_device_t HmllFetcher::device() const { return fetcher_.device; }
hmll_fetcher_kind_t HmllFetcher::kind() const { return fetcher_.kind; }


void init_fetcher(const nb::module_& m)
{
    nb::class_<HmllFetcher>(m, "HmllFetcher",
        R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc"
    )
    .def_prop_ro("device", &HmllFetcher::device)
    .def_prop_ro("kind", &HmllFetcher::kind)
    .def("__repr__", [](const HmllFetcher& self)
    {
        return std::format("HmllFetcher(kind={}, device={})", self.kind(), self.device());
    });
}