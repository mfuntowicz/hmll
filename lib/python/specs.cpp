//
// Created by mfuntowicz on 12/17/25.
//

#include <memory>
#include <span>
#include <hmll/types.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;


class HmllTensorSpecs
{
    std::shared_ptr<hmll_tensor_specs_t> specs;

public:
    [[nodiscard]]
    size_t rank() const { return specs->rank; }

    [[nodiscard]]
    std::span<size_t> shape() const { return {specs->shape, specs->rank}; }

};

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
