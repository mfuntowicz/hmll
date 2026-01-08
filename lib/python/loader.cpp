#include "loader.hpp"
#include <format>
#include <sys/mman.h>
#include <hmll/hmll.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "formatters.hpp"
#include "ndarray.hpp"
#include "hmll/memory.h"

namespace nb = nanobind;
using namespace nb::literals;

hmll_device_t WeightLoader::device() const { return ctx_->fetcher->device; }
hmll_fetcher_kind_t WeightLoader::kind() const { return ctx_->fetcher->kind; }


std::unique_ptr<WeightLoader> WeightLoader::from_paths(const std::vector<std::string>& paths, const hmll_device_t device)
{
    auto ctx = std::make_unique<hmll_t>();
    std::vector<hmll_source> srcs(paths.size());

    for (size_t i = 0; i < paths.size(); ++i) {
        if (const auto res = hmll_source_open(paths[i].c_str(), &srcs[i]); hmll_check(res)) {
            for (size_t j = 0; j < i; ++j) {
                hmll_source_close(&srcs[j]);
            }
            throw std::runtime_error(paths[i] + ": " + hmll_strerr(res));
        }
    }

    return std::make_unique<WeightLoader>(std::move(ctx), srcs, device);
}

WeightLoader::WeightLoader(std::unique_ptr<hmll_t> ctx, std::vector<hmll_source_t>& srcs, const hmll_device_t device)
    : ctx_(std::move(ctx)), srcs_(std::move(srcs))
{
    hmll_loader_init(ctx_.get(), srcs_.data(), srcs_.size(), device, HMLL_FETCHER_AUTO);
}

nb::ndarray<nb::ndim<1>, nb::c_contig> WeightLoader::fetch(
    const size_t start, const size_t end, const hmll_dtype_t dtype, const int iofile) const
{
    auto buffer = std::make_unique<hmll_iobuf_t>();
    hmll_range_t offsets;

    {
        nb::gil_scoped_release release;
        const size_t nbytes = ALIGN_UP(end, 4096) - ALIGN_DOWN(start, 4096);

        // Allocate buffer for the tensor
        const auto dev = device();
        buffer->ptr = hmll_get_buffer(ctx_.get(), dev, nbytes);
        buffer->size = nbytes;
        buffer->device = dev;

        if (!buffer->ptr)
            throw std::runtime_error("Failed to allocate buffer");

        // Fetch the tensor data
        const auto range = hmll_range_t{start, end};
        offsets = hmll_fetch(ctx_.get(), buffer.get(), range, iofile);
        if (hmll_check(ctx_->error)) {
            munmap(buffer->ptr, buffer->size);
            throw std::runtime_error("Failed to read data");
        }
    }

    // Let's make sure we are not deleting the buffer before PyTorch releases it
    const hmll_iobuf_t* handle = buffer.release();
    const nb::capsule deleter(handle, [](void* p) noexcept {
        if (const auto* b = static_cast<hmll_iobuf_t*>(p)) {
            munmap(b->ptr, b->size);
            delete b;
        }
    });

    return hmll_to_ndarray({start, end}, *handle, offsets, dtype, deleter);
}

void init_fetcher(const nb::module_& m)
{
    nb::enum_<hmll_device_t>(m, "Device", R"pbdoc(Define all the targetable devices)pbdoc")
    .value("CPU", HMLL_DEVICE_CPU, "Target CPU device")
    .value("CUDA", HMLL_DEVICE_CUDA, "Target CUDA device");

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

    nb::class_<WeightLoader>(m, "WeightLoader", R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc")
    .def(nb::new_(&WeightLoader::from_paths), "paths"_a.sig("list[str]"), "device"_a.sig("Device"))
    .def_prop_ro("device", &WeightLoader::device)
    .def_prop_ro("kind", &WeightLoader::kind)
    .def("fetch", &WeightLoader::fetch, "start"_a.sig("int"), "end"_a.sig("int"), "dtype"_a.sig("dtype"), "iofile"_a.sig("int"))
    .def("__repr__", [](const WeightLoader& self)
    {
        return std::format("WeightLoader(kind={}, device={})", self.kind(), self.device());
    });
}