#include "fetcher.hpp"
#include <nanobind/nanobind.h>

namespace nb = nanobind;


void init_fetcher(const nb::module_& m)
{
    nb::class_<HmllFetcher>(m, "HmllFetcher",
        R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc"
    );
}