#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/xchar.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "hmll/hmll.h"

namespace nb = nanobind;

void init_specs(nb::module_& m)
{
    nb::class_<hmll_tensor_specs_t>(m, "TensorSpecs", nb::is_final())
        .def_ro("start", &hmll_tensor_specs_t::start)
        .def_ro("end", &hmll_tensor_specs_t::end)
        .def_ro("rank", &hmll_tensor_specs_t::rank)
        .def_prop_ro("dtype", [](const hmll_tensor_specs_t& specs) { return specs.dtype; })
        .def_prop_ro("offset",[](const hmll_tensor_specs_t& specs)
        {
            return std::make_pair(specs.start, specs.end);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("shape",[](const hmll_tensor_specs_t& specs)
        {
            return std::vector(specs.shape, specs.shape + specs.rank);
        }, nb::rv_policy::move)
        .def("__repr__", [](const hmll_tensor_specs_t& specs) {
            return fmt::format(
                FMT_COMPILE("TensorSpecs(start={}, end={}, shape=[{}], rank={})"),
                specs.start, specs.end, fmt::join(specs.shape, specs.shape + specs.rank, ", "), specs.rank
            );
        }, nb::rv_policy::reference_internal);
}