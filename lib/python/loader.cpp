#include "loader.hpp"
#include <hmll/hmll.h>
#include <fmt/format.h>
#include <fmt/compile.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


#include "hmll/memory.h"
#include "formatters.hpp"

#include "ndarray.hpp"

namespace nb = nanobind;
using namespace nb::literals;

static constexpr auto PYHMLL_ERR_FETCH = FMT_COMPILE("Failed to fetch data: {}");

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
            throw std::runtime_error(fmt::format(FMT_COMPILE("{}: {}"), paths[i], std::string_view { hmll_strerr(res) }));
        }
    }

    return std::make_unique<WeightLoader>(srcs, device, std::move(ctx));
}

WeightLoader::WeightLoader(std::vector<hmll_source_t> srcs, const hmll_device_t device, const hmll_fetcher_kind_t backend)
    : WeightLoader(std::move(srcs), device, std::make_unique<hmll_t>(), backend) {}

WeightLoader::WeightLoader(std::vector<hmll_source_t> srcs, const hmll_device_t device, std::unique_ptr<hmll_t> ctx, const hmll_fetcher_kind_t backend)
    : ctx_(std::move(ctx)), srcs_(std::move(srcs))
{
    if (hmll_check(hmll_loader_init(ctx_.get(), srcs_.data(), srcs_.size(), device, backend))) {
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
        *buf_guard = hmll_get_buffer(ctx, nbytes, HMLL_MEM_DEVICE);

        if (const auto res = hmll_fetch(ctx, iofile, buf_guard.get(), start); res <= 0) {
            hmll_free_buffer(buf_guard.get());
            const std::string err = hmll_strerr(ctx_->error);
            throw std::runtime_error(fmt::format(PYHMLL_ERR_FETCH, err));
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

    return hmll_to_ndarray(buffer, dtype, shape, rank, std::move(deleter));
}

size_t WeightLoader::fetch(const int iofile, const size_t offset, const uintptr_t dst, const size_t size) const
{
    nb::gil_scoped_release release;

    if (size == 0) return 0;

    const auto ctx = ctx_.get();
    const auto dev = device();

    const hmll_iobuf_t buf = {size, reinterpret_cast<void *const>(dst), dev};
    const auto [start, end] = hmll_range_t{offset, offset + size};
    if (const auto res = hmll_fetch(ctx, iofile, &buf, start); res <= 0) {
        const std::string err = hmll_strerr(ctx_->error);
        throw std::runtime_error(fmt::format(PYHMLL_ERR_FETCH, err));
    } else {
        return static_cast<size_t>(res);
    }
}

size_t WeightLoader::fetchv(const int iofile, const std::vector<std::tuple<size_t, size_t>>& ranges, const uintptr_t dst) const
{
    nb::gil_scoped_release release;

    const auto ctx = ctx_.get();
    const auto dev = device();
    const auto* fetcher = ctx->fetcher;

    if (ranges.empty())
        return 0;

    ssize_t res = 0;
    if (fetcher->fetchv_range_impl_) {
        std::vector<hmll_iobuf_t> dsts(ranges.size());
        std::vector<size_t> offsets(ranges.size());
        size_t dst_offset = 0;
        for (size_t i = 0; i < ranges.size(); ++i) {
            const auto [start, end] = ranges[i];
            const size_t nbytes = end - start;
            dsts[i] = {nbytes, reinterpret_cast<void *>(dst + dst_offset), dev};
            offsets[i] = start;
            dst_offset += nbytes;
        }
        res = hmll_fetchv(ctx, iofile, dsts.data(), offsets.data(), ranges.size());
        if (res < 0) {
            const std::string err = hmll_strerr(ctx_->error);
            throw std::runtime_error(fmt::format(PYHMLL_ERR_FETCH, err));
        }
        return static_cast<size_t>(res);
    }

    /* Fallback: sequential fetch when fetchv is not implemented (e.g. Win32 mmap) */
    size_t total = 0;
    size_t dst_offset = 0;
    for (const auto& [start, end] : ranges) {
        const size_t nbytes = end - start;
        const hmll_iobuf_t buf = {nbytes, reinterpret_cast<void*>(dst + dst_offset), dev};
        if (res = hmll_fetch(ctx, iofile, &buf, start); res <= 0) {
            const std::string err = hmll_strerr(ctx_->error);
            throw std::runtime_error(fmt::format(PYHMLL_ERR_FETCH, err));
        }
        total += static_cast<size_t>(res);
        dst_offset += nbytes;
    }
    return total;
}

void init_loader(nb::module_& m)
{
    nb::class_<WeightLoader>(m, "WeightLoader", R"pbdoc("Opaque type representing an allocated fetcher backend)pbdoc")
    .def(nb::new_(&WeightLoader::from_paths), "paths"_a.sig("list[str]"), "device"_a.sig("Device"))
    .def_prop_ro("device", &WeightLoader::device)
    .def_prop_ro("kind", &WeightLoader::kind)
    .def("afetch", &WeightLoader::afetch)
    .def("fetch", &WeightLoader::fetch)
    .def("fetchv", &WeightLoader::fetchv, "iofile"_a.sig("int"), "ranges"_a.sig("list[tuple[int, int]]"), "dst"_a.sig("int"))
    .def("__repr__", [](const WeightLoader& self)
    {
        return fmt::format(FMT_COMPILE("WeightLoader(kind={}, device={})"), self.kind(), self.device());
    });
}
