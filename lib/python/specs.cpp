//
// Created by mfuntowicz on 12/17/25.
//

#include <hmll/hmll.h>
#include "specs.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

hmll_tensor_data_type_t HmllTensorSpecs::dtype() const { return specs.dtype; }

size_t HmllTensorSpecs::rank() const { return specs.rank; }

size_t HmllTensorSpecs::start() const { return specs.start; }
size_t HmllTensorSpecs::end() const { return specs.end; }
std::tuple<size_t, size_t> HmllTensorSpecs::offsets() const { return { specs.start, specs.end }; }

std::vector<size_t> HmllTensorSpecs::shape() const
{
    std::vector<size_t> shape(specs.rank);
    for (size_t i = 0; i < specs.rank; ++i) shape[i] = specs.shape[i];
    return shape;
}

void init_specs(const nb::module_& m)
{
    nb::enum_<hmll_tensor_data_type_t>(m, "HmllDataType",
        R"pbdoc(Describe the underlying element type for each value stored in a tensor)pbdoc")
        .value("BOOLEAN", HMLL_DTYPE_BOOL)
        .value("BFLOAT16", HMLL_DTYPE_BFLOAT16)
        .value("COMPLEX64", HMLL_DTYPE_COMPLEX)
        .value("FLOAT32", HMLL_DTYPE_FLOAT32)
        .value("FLOAT16", HMLL_DTYPE_FLOAT16)
        .value("FLOAT8_E4M3", HMLL_DTYPE_FLOAT8_E4M3)
        .value("FLOAT8_E5M2", HMLL_DTYPE_FLOAT8_E5M2)
        .value("FLOAT8_E8M0", HMLL_DTYPE_FLOAT8_E8M0)
        .value("FLOAT4", HMLL_DTYPE_FLOAT4)
        .value("FLOAT6_E2M3", HMLL_DTYPE_FLOAT6_E2M3)
        .value("FLOAT6_E3M2", HMLL_DTYPE_FLOAT6_E3M2)
        .value("SIGNED_INT4", HMLL_DTYPE_SIGNED_INT4)
        .value("SIGNED_INT8", HMLL_DTYPE_SIGNED_INT8)
        .value("SIGNED_INT16", HMLL_DTYPE_SIGNED_INT16)
        .value("SIGNED_INT32", HMLL_DTYPE_SIGNED_INT32)
        .value("SIGNED_INT64", HMLL_DTYPE_SIGNED_INT64)
        .value("UNSIGNED_INT4", HMLL_DTYPE_UNSIGNED_INT4)
        .value("UNSIGNED_INT8", HMLL_DTYPE_UNSIGNED_INT8)
        .value("UNSIGNED_INT16", HMLL_DTYPE_UNSIGNED_INT16)
        .value("UNSIGNED_INT32", HMLL_DTYPE_UNSIGNED_INT32)
        .value("UNSIGNED_INT64", HMLL_DTYPE_UNSIGNED_INT64)
        .value("UNKNOWN", HMLL_DTYPE_UNKNOWN)
    .def_prop_ro("nbits", [](const hmll_tensor_data_type dtype) -> size_t { return hmll_nbits(dtype); });

    nb::class_<HmllTensorSpecs>(m, "HmllTensorSpecs",
        R"pbdoc(Contains all the information about a tensor)pbdoc")
    .def_prop_ro("dtype", &HmllTensorSpecs::dtype,
        nb::sig("def dtype(self) -> HmllDataType"))
    .def_prop_ro(
        "rank", &HmllTensorSpecs::rank,
        nb::sig("def rank(self) -> int"),
          R"pbdoc(
          Retrieve the tensor's rank.

          Returns:
            int: The rank of the tensor.
        )pbdoc")
    .def_prop_ro(
    "start", &HmllTensorSpecs::start,
    nb::sig("def start(self) -> int"),
      R"pbdoc(
          Retrieve the tensor's start offset.

          Returns:
            int: The start offset of the tensor.
        )pbdoc")
    .def_prop_ro(
    "end", &HmllTensorSpecs::end,
    nb::sig("def end(self) -> int"),
      R"pbdoc(
          Retrieve the tensor's end offset.

          Returns:
            int: The end offset of the tensor.
        )pbdoc")
    .def_prop_ro(
    "offsets", &HmllTensorSpecs::offsets,
    nb::sig("def offsets(self) -> tuple(int, int)"),
      R"pbdoc(
          Retrieve the tensor' start and end offsets as a tuple.

          Returns:
            int: The start and end offsets of the tensor.
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
