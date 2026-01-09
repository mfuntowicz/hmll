//
// Created by mfuntowicz on 1/8/26.
//
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <hmll/hmll.h>

namespace nb = nanobind;
using namespace nb::literals;


class SafetensorsAccessor
{
    std::shared_ptr<hmll_t> base_ctx_;
    std::shared_ptr<hmll_registry_t> registry_;
    std::shared_ptr<hmll_source_t> source_;

    // Thread-local contexts map: thread_id -> hmll_t
    mutable std::mutex ctx_map_mutex_;
    mutable std::unordered_map<std::thread::id, std::unique_ptr<hmll_t>> thread_contexts_;

    hmll_t* get_thread_context() const {
        const auto tid = std::this_thread::get_id();
        std::lock_guard lock(ctx_map_mutex_);

        auto it = thread_contexts_.find(tid);
        if (it == thread_contexts_.end()) {
            auto ctx = std::make_unique<hmll_t>();
            if (!hmll_success(hmll_clone_context(base_ctx_.get(), ctx.get()))) {
                return nullptr;
            }
            auto* ptr = ctx.get();
            thread_contexts_[tid] = std::move(ctx);
            return ptr;
        }
        return it->second.get();
    }

public:
    explicit SafetensorsAccessor(const std::filesystem::path& path)
        : base_ctx_(std::make_shared<hmll_t>()), registry_(std::make_shared<hmll_registry_t>()), source_(std::make_shared<hmll_source_t>())
    {
        if (!hmll_success(hmll_source_open(path.c_str(), source_.get()))) {
            throw std::runtime_error("Failed to open file: " + path.string());
        }

        base_ctx_->sources = source_.get();
        base_ctx_->num_sources = 1;
        base_ctx_->fetcher = nullptr;
        base_ctx_->error = HMLL_OK;

        if (!hmll_success(hmll_safetensors_populate_registry(base_ctx_.get(), registry_.get(), *source_, 0, 0))) {
            throw std::runtime_error("Failed to read tensor definition in file " + path.string() + ": " + hmll_strerr(base_ctx_->error));
        }
    }

    size_t size() const { return registry_->num_tensors; }
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
    );

    m.def("safetensors", [](const std::filesystem::path& path) {
        return new SafetensorsAccessor(path);
    }, nb::rv_policy::take_ownership);
}