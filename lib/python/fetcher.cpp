#include "fetcher.hpp"
#include <format>
#include <sys/mman.h>
#include <dlpack/dlpack.h>
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

nb::ndarray<> HmllFetcher::fetch_contiguous(const std::string& name) const
{
    // Get tensor specs
    auto lookup = hmll_get_tensor_specs(ctx_ptr_, name.c_str());
    if (!lookup.found) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const auto specs = lookup.specs;
    const size_t nbytes = PAGE_ALIGNED_UP(specs.end, 4096) - PAGE_ALIGNED_DOWN(specs.start, 4096);

    // Allocate buffer for the tensor
    auto* buffer = new hmll_device_buffer_t();
    buffer->ptr = hmll_get_hugepage_buffer(ctx_ptr_, nbytes);
    buffer->size = nbytes;
    buffer->device = fetcher_.device;

    if (!buffer->ptr) {
        delete buffer;
        throw std::runtime_error("Failed to allocate buffer");
    }

    // Fetch the tensor data
    const auto offsets = hmll_fetch_tensor(ctx_ptr_, fetcher_, name.c_str(), *buffer);
    if (hmll_has_error(hmll_get_error(ctx_ptr_))) {
        munmap(buffer->ptr, buffer->size);
        delete buffer;
        throw std::runtime_error("Failed to read data");
    }

    // Let's make sure we are not deleting the buffer before PyTorch releases it
    const nb::capsule deleter(buffer, [](void* p) noexcept {
        if (const auto* b = static_cast<hmll_device_buffer_t*>(p)) {
            munmap(b->ptr, b->size);
            delete b;
        }
    });

    return hmll_to_ndarray(specs, *buffer, offsets, deleter);
}

void init_fetcher(const nb::module_& m)
{
    nb::class_<HmllFetcher>(m, "HmllFetcher",
        R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc"
    )
    .def_prop_ro("device", &HmllFetcher::device)
    .def_prop_ro("kind", &HmllFetcher::kind)
    .def("__getitem__", &HmllFetcher::fetch_contiguous, "name"_a.sig("string"))
    .def("__repr__", [](const HmllFetcher& self)
    {
        return std::format("HmllFetcher(kind={}, device={})", self.kind(), self.device());
    });
}