#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include "hmll/hmll.h"

namespace nb = nanobind;
using namespace nb::literals;


void init_loader(nb::module_&);

#ifdef __HMLL_SAFETENSORS_ENABLED__
void init_safetensors(nb::module_&);
void init_specs(nb::module_&);
#endif

NB_MODULE(_pyhmll_impl, m)
{
    m.doc() = "hmll: High-Performance Model Loading Library - Efficient AI Model loading for modern AI workloads.";

    nb::class_<hmll_device_t>(m, "Device", R"pbdoc(Define all the targetable devices)pbdoc")
    .def_static("cpu", &hmll_device_cpu, "Create CPU device")
    .def_static("cuda", &hmll_device_cuda, "idx"_a.sig("int = 0"), "Create CUDA device with index")
    .def_prop_ro("kind", [](const hmll_device_t& d) { return d.kind; })
    .def_prop_ro("idx", [](const hmll_device_t& d) { return d.idx; })
    .def_prop_ro("is_cpu", [](const hmll_device_t& d) { return hmll_device_is_cpu(d); })
    .def_prop_ro("is_cuda", [](const hmll_device_t& d) { return hmll_device_is_cuda(d); })
    .def("__eq__", [](const hmll_device_t& a, const hmll_device_t& b) { return hmll_device_eq(a, b); })
    .def("__repr__", [](const hmll_device_t& d) { return hmll_device_is_cpu(d) ? "Device.cpu()" : fmt::format("Device.cuda({})", d.idx); });

    nb::enum_<hmll_device_kind_t>(m, "DeviceKind", R"pbdoc(Define all the targetable devices)pbdoc")
    .value("CPU", HMLL_DEVICE_CPU)
    .value("CUDA", HMLL_DEVICE_CUDA);

    nb::enum_<hmll_fetcher_kind_t>(m, "Backend", R"pbdoc(Define the I/O backend to use)pbdoc")
    .value("AUTO", HMLL_FETCHER_AUTO, "Automatically select backend (defaults to MMAP)")
#ifdef __HMLL_IO_URING_ENABLED__
    .value("IO_URING", HMLL_FETCHER_IO_URING, "Use io_uring for async I/O (Linux only)")
#endif
    .value("MMAP", HMLL_FETCHER_MMAP, "Use memory-mapped I/O");

    nb::enum_<hmll_dtype_t>(m, "dtype", R"pbdoc(Define all the targetable element type in a tensor)pbdoc")
    .value("BOOL", HMLL_DTYPE_BOOL)
    .value("BFLOAT16", HMLL_DTYPE_BFLOAT16)
    .value("COMPLEX", HMLL_DTYPE_COMPLEX)
    .value("FLOAT4", HMLL_DTYPE_FLOAT4)
    .value("FLOAT6_E2M3", HMLL_DTYPE_FLOAT6_E2M3)
    .value("FLOAT6_E3M2", HMLL_DTYPE_FLOAT6_E3M2)
    .value("FLOAT8_E5M2", HMLL_DTYPE_FLOAT8_E5M2)
    .value("FLOAT8_E4M3", HMLL_DTYPE_FLOAT8_E4M3)
    .value("FLOAT8_E8M0", HMLL_DTYPE_FLOAT8_E8M0)
    .value("FLOAT16", HMLL_DTYPE_FLOAT16)
    .value("FLOAT32", HMLL_DTYPE_FLOAT32)
    .value("FLOAT64", HMLL_DTYPE_FLOAT64)
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
    .value("UNKNOWN", HMLL_DTYPE_UNKNOWN);

    init_loader(m);

#ifdef __HMLL_SAFETENSORS_ENABLED__
    init_safetensors(m);
    init_specs(m);
#endif
}