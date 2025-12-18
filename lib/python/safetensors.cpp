//
// Created by mfuntowicz on 12/17/25.
//
#include <hmll/hmll.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "context.hpp"

namespace nb = nanobind;

void init_safetensors(nb::module_& m)
{
    m.def("safetensors", [](const std::string& path) -> HmllContext
    {
        return HmllContext::open(path, HMLL_SAFETENSORS, HMLL_MMAP | HMLL_SKIP_METADATA);
    });
}