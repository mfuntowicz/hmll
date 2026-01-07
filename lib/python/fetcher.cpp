#include "fetcher.hpp"
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

hmll_device_t Fetcher::device() const { return ctx_->fetcher->device; }
hmll_fetcher_kind_t Fetcher::kind() const { return ctx_->fetcher->kind; }


std::unique_ptr<Fetcher> Fetcher::from_paths(const std::vector<std::string>& paths, const hmll_device_t device)
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

    return std::make_unique<Fetcher>(std::move(ctx), srcs, device);
}

Fetcher::Fetcher(std::unique_ptr<hmll_t> ctx, std::vector<hmll_source_t>& srcs, const hmll_device_t device)
    : ctx_(std::move(ctx)), srcs_(std::move(srcs))
{
    hmll_fetcher_init(ctx_.get(), srcs_.data(), srcs_.size(), device, HMLL_FETCHER_AUTO);
}

nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> Fetcher::fetch(const size_t start, const size_t end, const int iofile) const
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

    return hmll_to_ndarray({start, end}, *handle, offsets, deleter);
}

void init_fetcher(const nb::module_& m)
{
    nb::enum_<hmll_device_t>(m, "Device", R"pbdoc(Define all the targetable devices)pbdoc")
    .value("CPU", HMLL_DEVICE_CPU, "Target CPU device")
    .value("CUDA", HMLL_DEVICE_CUDA, "Target CUDA device");

    nb::class_<Fetcher>(m, "Fetcher", R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc")
    .def(nb::new_(&Fetcher::from_paths), "paths"_a.sig("list[str]"), "device"_a.sig("Device"))
    .def_prop_ro("device", &Fetcher::device)
    .def_prop_ro("kind", &Fetcher::kind)
    .def("fetch", &Fetcher::fetch, "start"_a.sig("int"), "end"_a.sig("int"), "iofile"_a.sig("int"))
    .def("__repr__", [](const Fetcher& self)
    {
        return std::format("Fetcher(kind={}, device={})", self.kind(), self.device());
    });
}