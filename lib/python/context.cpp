#include "context.h"
#include <memory>
#include <optional>
#include <hmll/hmll.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;


bool HmllContext::has_error() const { return hmll_has_error(hmll_get_error(ctx_)); }
size_t HmllContext::num_tensors() const { return ctx_->num_tensors; }
bool HmllContext::contains(const std::string &name) const
{
    // TODO(mfuntowicz): Change that because it would put the ctx in error, while it's recoverable error
    hmll_get_tensor_specs(ctx_, name.c_str());
    return hmll_success(hmll_get_error(ctx_));
}


HmllContext HmllContext::open(const std::string& path, const hmll_file_kind kind, const int flags)
{
    const auto ctx = new hmll_context_t();
    if (const int result = hmll_open(path.c_str(), ctx, kind, static_cast<hmll_flags_t>(flags)); result < 0) {
        throw std::runtime_error(
            "Failed to open safetensors file " + path + ": " + hmll_strerr(hmll_get_error(ctx)));
    }

    return HmllContext(ctx);
}

void init_context(const nb::module_& m)
{
    nb::class_<HmllContext>(m, "HmllContext",
        R"pbdoc(Hold all the information about the current state of the HMLL lib)pbdoc"
    )
    .def_prop_ro("num_tensors", &HmllContext::num_tensors)
    .def("__contains__", &HmllContext::contains, "name"_a.sig("string"))
    .def("__enter__", [](const HmllContext& self) { return self; })
    .def("__exit__", [](
            const HmllContext& self,
            const std::optional<nb::type_object>& exc_type,
            const std::optional<nb::object>& exc_value,
            const std::optional<nb::object>& traceback
        ) { },
        "exc_type"_a = nb::none(),
        "exc_value"_a = nb::none(),
        "traceback"_a = nb::none()
    );
}