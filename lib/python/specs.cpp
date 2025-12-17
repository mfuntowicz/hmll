//
// Created by mfuntowicz on 12/17/25.
//

#include "specs.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

size_t HmllTensorSpecs::rank() const { return specs.rank; }

std::vector<size_t> HmllTensorSpecs::shape() const
{
    std::vector<size_t> shape(specs.rank);
    for (size_t i = 0; i < specs.rank; ++i) shape[i] = specs.shape[i];
    return shape;
}

void init_specs(const nb::module_& m)
{
    nb::class_<HmllTensorSpecs>(m, "HmllTensorSpecs",
        R"pbdoc(Contains all the information about a tensor)pbdoc")
    .def_prop_ro(
        "rank", &HmllTensorSpecs::rank,
        nb::sig("def rank(self) -> int"),
          R"pbdoc(
          Retrieve the tensor's rank.

          Returns:
            int: The rank of the tensor.
        )pbdoc")
    .def_prop_ro(
        "shape", [](const HmllTensorSpecs& specs) { return nb::cast(specs.shape()); },
        nb::sig("def shape(self) -> tuple[int, ...]"),
          R"pbdoc(
          The tensor shape as Python tuple.

          Returns:
            tuple(int): A tuple containing the sizes of each dimension.
        )pbdoc");
}
