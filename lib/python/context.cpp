#include "context.hpp"
#include <memory>
#include <optional>
#include <vector>
#include <hmll/hmll.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

Context::~Context() { if (ctx_) hmll_destroy(ctx_.get()); }
bool Context::has_error() const { return ctx_ && hmll_check(ctx_->error); }
bool Context::has_library_error() const { return ctx_ && hmll_error_is_lib_error(ctx_->error); }
bool Context::has_system_error() const { return ctx_ && hmll_error_is_os_error(ctx_->error); }

Context Context::open(const std::string& path)
{
    const std::span paths = {&path, 1};
    return open(paths);
}

Context Context::open(const std::span<const std::string> paths)
{
    auto ctx = std::make_unique<hmll_t>();
    std::vector<hmll_source> srcs(paths.size());

    for (size_t i = 0; i < paths.size(); ++i) {
        if (const auto res = hmll_source_open(paths[i].c_str(), &srcs[i]); hmll_check(res)) {
            for (size_t j = 0; j < i; ++j) {
                hmll_source_close(&srcs[j]);
            }
            throw std::runtime_error("Failed to open safetensors file " + paths[i] + ": " + hmll_strerr(res));
        }
    }

    return Context(std::move(ctx));
}

void init_context(const nb::module_& m)
{
    nb::enum_<hmll_device_t>(m, "HmllDevice",
        R"pbdoc(Define all the targetable devices)pbdoc"
    )
    .value("CPU", HMLL_DEVICE_CPU, "Target CPU device")
    .value("CUDA", HMLL_DEVICE_CUDA, "Target CUDA device");

    nb::enum_<hmll_fetcher_kind_t>(m, "HmllFetcherKind",
        R"pbdoc(Define all the available fetcher)pbdoc"
    ).value("AUTO", HMLL_FETCHER_AUTO, "Automatically choose the most appropriate fetcher");

    nb::class_<Context>(m, "Context",
        R"pbdoc(Hold all the information about the current state of the HMLL lib)pbdoc"
    )
    .def("__enter__", [](const Context& self) { return self; })
    .def("__exit__", [](
            const Context& self,
            const std::optional<nb::type_object>& exc_type,
            const std::optional<nb::object>& exc_value,
            const std::optional<nb::object>& traceback
        ) { },
        "exc_type"_a = nb::none(),
        "exc_value"_a = nb::none(),
        "traceback"_a = nb::none()
    );
}