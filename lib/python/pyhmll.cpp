#include <iostream>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_fetcher(const nb::module_&);

NB_MODULE(_pyhmll_impl, m)
{
    m.doc() = "hmll: Hugging Face Model Loading Library - Efficient AI Model loading for modern AI workloads.";

    init_fetcher(m);
}