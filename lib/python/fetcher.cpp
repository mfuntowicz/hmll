#include "fetcher.hpp"
#include <format>
#include <sys/mman.h>
#include <hmll/hmll.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "context.hpp"
#include "formatters.hpp"
#include "ndarray.hpp"

namespace nb = nanobind;
using namespace nb::literals;

hmll_device_t HmllFetcher::device() const { return fetcher_.device; }
hmll_fetcher_kind_t HmllFetcher::kind() const { return fetcher_.kind; }

nb::ndarray<> HmllFetcher::fetch(const std::string& name) const
{
    auto buffer = std::make_unique<hmll_device_buffer_t>();
    hmll_tensor_specs_t specs;
    hmll_range_t offsets;

    {
        nb::gil_scoped_release release;

        // Get tensor specs
        const auto lookup = hmll_get_tensor_specs(ctx_ptr_, name.c_str());
        if (!lookup.found)
            throw std::runtime_error("Tensor not found: " + name);

        specs = lookup.specs;
        const size_t nbytes = ALIGN_UP(specs.end, 4096) - ALIGN_DOWN(specs.start, 4096);

        // Allocate buffer for the tensor
        buffer->ptr = hmll_get_buffer(ctx_ptr_, fetcher_.device, nbytes);
        buffer->size = nbytes;
        buffer->device = fetcher_.device;

        if (!buffer->ptr) {
            throw std::runtime_error("Failed to allocate buffer");
        }

        // Fetch the tensor data
        offsets = hmll_fetch_tensor(ctx_ptr_, fetcher_, name.c_str(), *buffer);
        if (hmll_has_error(hmll_get_error(ctx_ptr_))) {
            munmap(buffer->ptr, buffer->size);
            throw std::runtime_error("Failed to read data");
        }
    }

    // Let's make sure we are not deleting the buffer before PyTorch releases it
    const hmll_device_buffer_t* handle = buffer.release();
    const nb::capsule deleter(handle, [](void* p) noexcept {
        if (const auto* b = static_cast<hmll_device_buffer_t*>(p)) {
            munmap(b->ptr, b->size);
            delete b;
        }
    });

    return hmll_to_ndarray(specs, *handle, offsets, deleter);
}

void init_fetcher(const nb::module_& m)
{
    nb::class_<HmllFetcher>(m, "HmllFetcher",
        R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc"
    )
    .def_prop_ro("device", &HmllFetcher::device)
    .def_prop_ro("kind", &HmllFetcher::kind)
    .def("fetch", &HmllFetcher::fetch, "name"_a.sig("string"))
    .def("__getitem__", &HmllFetcher::fetch, "name"_a.sig("string"))
    .def("__repr__", [](const HmllFetcher& self)
    {
        return std::format("HmllFetcher(kind={}, device={})", self.kind(), self.device());
    });
}