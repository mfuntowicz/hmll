//
// Created by mfuntowicz on 1/8/26.
//
#include <filesystem>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <hmll/hmll.h>

#include "loader.hpp"

namespace nb = nanobind;
using namespace nb::literals;


class SafetensorsAccessor
{
    std::unique_ptr<WeightLoader> loader_;
    std::shared_ptr<hmll_registry_t> registry_;

public:
    SafetensorsAccessor(const std::filesystem::path& path, const hmll_device_t device, const bool is_sharded)
    {
        if (is_sharded) {
            hmll_source index{};
            const auto path_str = path.string();
            if (hmll_check(hmll_source_open(path_str.c_str(), &index)))
                throw std::runtime_error("Failed to open file: " + path_str);

            auto registry = std::make_shared<hmll_registry_t>();
            auto ctx = std::make_unique<hmll_t>();
            size_t num_files = 0, num_tensors = 0;
            if ((num_files = hmll_safetensors_index(ctx.get(), registry.get(), index)) == 0) {
                throw std::runtime_error(fmt::format("Failed to read tensor definition in file {}: {}", path, hmll_strerr(ctx->error)));
            }

            const auto parent = path.parent_path();
            auto sources = std::vector<hmll_source_t>(num_files);
            for (size_t i = 0; i < num_files; ++i) {
                const auto shard = parent / fmt::format(FMT_COMPILE("model-{:05}-of-{:05}.safetensors"), i + 1, num_files);
                const auto shard_str = shard.string();
                if (hmll_check(hmll_source_open(shard_str.c_str(), &sources[i]))) {
                    throw std::runtime_error(fmt::format(FMT_COMPILE("Failed to open file: {}"), shard_str));
                }

                if (const auto n_tensors = hmll_safetensors_populate_registry(ctx.get(), registry.get(), sources[i], i, num_tensors); n_tensors == 0) {
                    throw std::runtime_error(fmt::format(FMT_COMPILE("Failed to read tensor definition in file {}: {}"), shard.string(), hmll_strerr(ctx->error)));
                } else {
                    num_tensors += n_tensors;
                }
            }

            loader_ = std::make_unique<WeightLoader>(sources, device, std::move(ctx));
            registry_ = std::move(registry);
        } else {
            const auto path_str = path.string();
            hmll_source_t source;
            if (hmll_check(hmll_source_open(path_str.c_str(), &source)))
                throw std::runtime_error("Failed to open file: " + path_str);

            const auto registry = std::make_shared<hmll_registry_t>();
            auto ctx = std::make_unique<hmll_t>();

            if (const auto n_tensors = hmll_safetensors_populate_registry(ctx.get(), registry.get(), source, 0, 0); n_tensors == 0) {
                hmll_source_close(&source);
                throw std::runtime_error(fmt::format(FMT_COMPILE("Failed to read tensor definition in file {}: {}"), path, hmll_strerr(ctx->error)));
            }

            auto sources = std::vector<hmll_source_t>{source};
            loader_ = std::make_unique<WeightLoader>(std::move(sources), device, std::move(ctx));
            registry_ = std::move(registry);
        }
    }

    [[nodiscard]] hmll_device_t device() const { return loader_->device(); }
    [[nodiscard]] size_t size() const { return registry_->num_tensors; }

    [[nodiscard]] std::vector<std::string_view> names() const
    {
        auto names = std::vector<std::string_view>(size() );
        for (size_t i = 0; i < names.size(); ++i)
            names[i] = std::string_view(registry_->names[i]);

        return names;
    }

    [[nodiscard]] std::vector<hmll_tensor_specs_t*> specs() const
    {
        auto specs = std::vector<hmll_tensor_specs_t*>(size() );
        for (size_t i = 0; i < specs.size(); ++i)
            specs[i] = registry_->tensors + i;

        return specs;
    }

    [[nodiscard]] std::vector<std::pair<std::string_view, hmll_tensor_specs_t*>> named_specs() const
    {
        auto specs = std::vector<std::pair<std::string_view, hmll_tensor_specs_t*>>(size());
        for (size_t i = 0; i < specs.size(); ++i) {
            const auto name = std::string_view(registry_->names[i]);
            specs[i] = std::make_pair(name, registry_->tensors + i);
        }
        return specs;
    }

    [[nodiscard]] bool contains(const std::string_view name) const
    {
        return hmll_find_by_name(loader_->context(), registry_.get(), name.data()) >= 0;
    }

    const hmll_tensor_specs_t* operator[](const std::string_view name) const
    {
        if (const hmll_lookup_result lookup = hmll_lookup_tensor(loader_->context(), registry_.get(), name.data()); lookup.specs != nullptr)
            return lookup.specs;

        throw nb::key_error(name.data());
    }

    [[nodiscard]] nb::ndarray<nb::c_contig> afetch(const std::string& name) const
    {
        const auto registry = registry_.get();
        const auto index = hmll_find_by_name(loader_->context(), registry, name.c_str());
        
        if (index < 0 || index >= registry->num_tensors)
            throw nb::key_error(name.c_str());

        const auto [shape, start, end, rank, dtype] = registry->tensors[index];
        const auto iofile = registry->indexes[index];

        // Delegate to WeightLoader for actual fetching
        return loader_->afetch(iofile, start, end, dtype, shape, rank);
    }

    [[nodiscard]] size_t fetch(const std::string& name, const uintptr_t dst, const size_t size) const
    {
        const auto registry = registry_.get();
        const auto index = hmll_find_by_name(loader_->context(), registry, name.c_str());

        if (index < 0 || index >= registry->num_tensors)
            throw nb::key_error(name.c_str());

        const auto specs = registry->tensors[index];
        const auto iofile = registry->indexes[index];

        // Delegate to WeightLoader for actual fetching
        return loader_->fetch(iofile, specs.start, dst, size);
    }
};

void init_safetensors(nb::module_& m)
{
    nb::class_<SafetensorsAccessor>(m, "SafetensorsAccessor")
    .def("__len__", &SafetensorsAccessor::size)
    .def("__contains__", &SafetensorsAccessor::contains)
    .def("__getitem__", &SafetensorsAccessor::operator[])
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
    .def_prop_ro("names", &SafetensorsAccessor::names)
    .def_prop_ro("specs", &SafetensorsAccessor::specs, nb::rv_policy::reference_internal)
    .def_prop_ro("named_specs", &SafetensorsAccessor::named_specs, nb::rv_policy::reference_internal)
    .def("key", &SafetensorsAccessor::names)
    .def("values", &SafetensorsAccessor::specs, nb::rv_policy::reference_internal)
    .def("items", &SafetensorsAccessor::named_specs, nb::rv_policy::reference_internal)
    .def("afetch", &SafetensorsAccessor::afetch)
    .def("fetch", &SafetensorsAccessor::fetch);

    m.def("safetensors", [](const std::filesystem::path& path, const hmll_device_t device, const bool is_sharded) {
        return new SafetensorsAccessor(path, device, is_sharded);
    }, nb::rv_policy::take_ownership, "path"_a, "device"_a, "is_sharded"_a = false);
}
