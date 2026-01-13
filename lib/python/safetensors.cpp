//
// Created by mfuntowicz on 1/8/26.
//
#if defined(__linux__) || defined(__unix__)
#include <sys/mman.h>
#endif
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <hmll/hmll.h>
#include "ndarray.hpp"

namespace nb = nanobind;
using namespace nb::literals;


class SafetensorsAccessor
{
    hmll_device_t device_;
    std::shared_ptr<hmll_t> base_ctx_;
    std::shared_ptr<hmll_registry_t> registry_;
    std::shared_ptr<hmll_source_t> sources_;

    hmll_t* get_thread_context() const {
        // cache the thread accessing this instance, if single-threaded, will skip all map lookups and stuff
        thread_local const SafetensorsAccessor* t_last_owner_ = nullptr;
        thread_local hmll_t* t_last_ctx_ = nullptr;

        if (t_last_owner_ == this) [[likely]]
            return t_last_ctx_;

        // if another thread is accessing it, let's make sure to use the thread' specific instance (create it if needed)
        thread_local std::unordered_map<const SafetensorsAccessor*, std::unique_ptr<hmll_t>> t_local_map_;

        hmll_t* ptr = nullptr;
        if (const auto it = t_local_map_.find(this); it != t_local_map_.end()) {
            ptr = it->second.get();
        } else {
            auto ctx = std::make_unique<hmll_t>();
            if (!hmll_success(hmll_clone_context(base_ctx_.get(), ctx.get())))
                return nullptr;

            ptr = ctx.get();
            t_local_map_[this] = std::move(ctx);
        }

        t_last_owner_ = this;
        t_last_ctx_ = ptr;
        return ptr;
    }

public:
    explicit SafetensorsAccessor(const std::filesystem::path& path, const hmll_device_t device):
        device_(device),
        base_ctx_(std::make_shared<hmll_t>()),
        registry_(std::make_shared<hmll_registry_t>()),
        sources_(std::make_shared<hmll_source_t>())
    {
        if (!hmll_success(hmll_source_open(path.c_str(), sources_.get())))
            throw std::runtime_error("Failed to open file: " + path.string());


        base_ctx_->sources = sources_.get();
        base_ctx_->num_sources = 1;

        if (hmll_check(hmll_loader_init(base_ctx_.get(), sources_.get(), 1, device, HMLL_FETCHER_AUTO)))
            throw std::runtime_error("Failed to allocate loader");

        if (!hmll_success(hmll_safetensors_populate_registry(base_ctx_.get(), registry_.get(), *sources_, 0, 0)))
            throw std::runtime_error(
                "Failed to read tensor definition in file " + path.string() + ": " + hmll_strerr(base_ctx_->error));
    }

    [[nodiscard]] hmll_device_t device() const { return device_; }
    [[nodiscard]] size_t size() const { return registry_->num_tensors; }
    [[nodiscard]] nb::ndarray<nb::ndim<1>, nb::c_contig> fetch(const std::string& name) const
    {
        auto buffer = std::make_unique<hmll_iobuf_t>();
        hmll_tensor_specs_t specs;

        {
            nb::gil_scoped_release release;
            const auto ctx = get_thread_context();
            const auto registry = registry_.get();

            if (const auto index = hmll_find_by_name(ctx, registry, name.c_str()); index >= 0 && index < registry->num_tensors)
            {
                specs = registry->tensors[index];
                const auto iofile = registry->indexes[index];

                // Allocate buffer for the tensor
                const auto dev = device();
                const auto nbytes = specs.end - specs.start;
                buffer->ptr = hmll_get_buffer(ctx, dev, nbytes, HMLL_MEM_DEVICE);
                buffer->size = nbytes;
                buffer->device = dev;

                if (!buffer->ptr)
                    throw std::runtime_error("Failed to allocate buffer");

                // Fetch the tensor data
                const auto range = hmll_range_t{specs.start, specs.end};
                if (const auto res = hmll_fetch(ctx, buffer.get(), range, iofile); res < 0) {
                    munmap(buffer->ptr, buffer->size);
                    throw std::runtime_error("Failed to read data");
                }
            } else {
                throw nb::key_error(name.c_str());
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

        return hmll_to_ndarray({specs.start, specs.end}, *handle, specs.dtype, deleter);
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
    .def("fetch", &SafetensorsAccessor::fetch, "name"_a.sig("str"));

    m.def("safetensors", [](const std::filesystem::path& path, const hmll_device_t device) {
        return new SafetensorsAccessor(path, device);
    }, nb::rv_policy::take_ownership);
}