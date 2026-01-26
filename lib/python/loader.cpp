#include "loader.hpp"
#include <hmll/hmll.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "formatters.hpp"
#include "ndarray.hpp"
#include "hmll/memory.h"

namespace nb = nanobind;
using namespace nb::literals;

hmll_t* WeightLoader::context() const { return ctx_.get(); }
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

    return std::make_unique<WeightLoader>(srcs, device, std::move(ctx));
}

WeightLoader::WeightLoader(std::vector<hmll_source_t> srcs, const hmll_device_t device)
    : WeightLoader(std::move(srcs), device, std::make_unique<hmll_t>()) {}

WeightLoader::WeightLoader(std::vector<hmll_source_t> srcs, const hmll_device_t device, std::unique_ptr<hmll_t> ctx)
    : ctx_(std::move(ctx)), srcs_(std::move(srcs))
{
    if (hmll_check(hmll_loader_init(ctx_.get(), srcs_.data(), srcs_.size(), device, HMLL_FETCHER_AUTO))) {
        const std::string err = hmll_strerr(ctx_->error);
        throw std::runtime_error("Failed to initialize loader: " + err);
    }
}

nb::ndarray<nb::c_contig> WeightLoader::afetch(const int iofile, const size_t start, const size_t end, const hmll_dtype_t dtype, const size_t* shape, const uint8_t rank) const
{
    const auto ctx = ctx_.get();
    const auto dev = device();

    auto buf_guard = std::make_unique<hmll_iobuf_t>();
    {
        nb::gil_scoped_release release;

        const auto nbytes = end - start;
        *buf_guard = hmll_get_buffer(ctx, dev, nbytes, HMLL_MEM_DEVICE);

        const auto range = hmll_range_t{start, end};
        if (const auto res = hmll_fetch(ctx, iofile, buf_guard.get(), range); res <= 0) {
            hmll_free_buffer(buf_guard.get());
            const std::string err = hmll_strerr(ctx_->error);
            throw std::runtime_error("Failed to read data " + err);
        }
    }

    // Let's make sure we are not deleting the buffer before PyTorch releases it
    const auto buffer = buf_guard.release();
    nb::capsule deleter(buffer, [](void* p) noexcept {
        if (auto* b = static_cast<hmll_iobuf_t*>(p)) {
            hmll_free_buffer(b);
            delete b;
        }
    });

    return hmll_to_ndarray({start, end}, buffer, dtype, shape, rank, std::move(deleter));
}

size_t WeightLoader::fetch(const int iofile, const size_t offset, const uintptr_t dst, const size_t size) const
{
    nb::gil_scoped_release release;

    const auto ctx = ctx_.get();
    const auto dev = device();

    hmll_iobuf_t buf = {size, reinterpret_cast<void *>(dst), dev};
    const auto range = hmll_range_t{offset, offset + size};
    if (const auto res = hmll_fetch(ctx, iofile, &buf, range); res <= 0) {
        const std::string err = hmll_strerr(ctx_->error);
        throw std::runtime_error("Failed to read data " + err);
    } else {
        return static_cast<size_t>(res);
    }
}

void init_loader(nb::module_& m)
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
    .def("afetch", &WeightLoader::afetch)
    .def("fetch", &WeightLoader::fetch)
    .def("__repr__", [](const WeightLoader& self)
    {
        return fmt::format("WeightLoader(kind={}, device={})", self.kind(), self.device());
    });
}
