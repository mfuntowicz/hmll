#include <iostream>
#include <nanobind/nanobind.h>

namespace nb = nanobind;


void init_context(const nb::module_&);
void init_specs(const nb::module_&);
void init_safetensors(nb::module_&);


NB_MODULE(_pyhmll_impl, m)
{
    m.doc() = "hmll: Hugging Face Model Loading Library - Efficient AI Model loading for modern AI workloads.";

    init_context(m);
    init_safetensors(m);
    init_specs(m);
}