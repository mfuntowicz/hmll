//
// Created by mfuntowicz on 1/8/26.
//
#include <filesystem>
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <hmll/hmll.h>

#include "loader.hpp"
#include "ndarray.hpp"

namespace nb = nanobind;
using namespace nb::literals;


class SafetensorsAccessor
{
    std::unique_ptr<WeightLoader> loader_;
    std::shared_ptr<hmll_registry_t> registry_;

public:
    explicit SafetensorsAccessor(const std::filesystem::path& path, const hmll_device_t device):
        registry_(std::make_shared<hmll_registry_t>())
    {
        hmll_source_t source;
        if (hmll_check(hmll_source_open(path.c_str(), &source)))
            throw std::runtime_error("Failed to open file: " + path.string());

        loader_ = std::make_unique<WeightLoader>(std::vector{source}, device);
        if (const auto ctx = loader_->context(); hmll_check(hmll_safetensors_populate_registry(ctx, registry_.get(), source, 0, 0)))
            throw std::runtime_error(
                "Failed to read tensor definition in file " + path.string() + ": " + hmll_strerr(ctx->error));
    }

    [[nodiscard]] hmll_device_t device() const { return loader_->device(); }
    [[nodiscard]] size_t size() const { return registry_->num_tensors; }
    
    [[nodiscard]] nb::ndarray<nb::ndim<1>, nb::c_contig> fetch(const std::string& name) const
    {;
        const auto registry = registry_.get();
        const auto index = hmll_find_by_name(loader_->context(), registry, name.c_str());
        
        if (index < 0 || index >= registry->num_tensors)
            throw nb::key_error(name.c_str());

        const auto specs = registry->tensors[index];
        const auto iofile = registry->indexes[index];

        // Delegate to WeightLoader for actual fetching
        return loader_->fetch(iofile, specs.start, specs.end, specs.dtype);
    }
};

void init_safetensors(nb::module_& m)
{
    nb::class_<SafetensorsAccessor>(m, "SafetensorsAccessor")
    .def("__len__", &SafetensorsAccessor::size)
    .def("__enter__", [](const nb::handle self) { return self; })
    .def("__exit__",
        [](SafetensorsAccessor&, nb::handle exc_type, nb::handle exc_value, nb::handle traceback) {
            return false;
        },
        nb::arg("exc_type").none(),
        nb::arg("exc_value").none(),
        nb::arg("traceback").none()
    )
    .def_prop_ro("device", &SafetensorsAccessor::device)
    .def("fetch", &SafetensorsAccessor::fetch);

    m.def("safetensors", [](const std::filesystem::path& path, const hmll_device_t device) {
        return new SafetensorsAccessor(path, device);
    }, nb::rv_policy::take_ownership);
}