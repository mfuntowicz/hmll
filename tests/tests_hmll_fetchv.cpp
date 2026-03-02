//
// Test suite for hmll_fetchv public API (mmap, io_uring backends).
// Uses fixtures from create_fetchv_testing_safetensors.py; set
// HMLL_CI_FETCHV_SAFETENSORS_FPATH and HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH.
//

#include <catch2/catch_all.hpp>
#include "hmll/hmll.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#endif

#define HMLL_CI_FETCHV_SAFETENSORS_FPATH "HMLL_CI_FETCHV_SAFETENSORS_FPATH"
#define HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH "HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH"

namespace {

using BackendPair = std::pair<const char*, hmll_loader_kind>;
#if defined(__linux__)
constexpr BackendPair kBackends[] = {
    {"IO_URING", HMLL_FETCHER_IO_URING},
    {"MMAP", HMLL_FETCHER_MMAP},
};
#else
constexpr BackendPair kBackends[] = {{"MMAP", HMLL_FETCHER_MMAP}};
#endif

// Validate float32 buffer filled with deterministic pattern value[i] = i (as float)
void validate_float32_arange(const hmll_iobuf_t& buf, size_t numel, size_t start = 0) {
    REQUIRE(buf.ptr != nullptr);
    REQUIRE(buf.size >= numel * sizeof(float));
    const auto* p = static_cast<const float*>(buf.ptr);
    for (size_t i = 0; i < numel; ++i)
        REQUIRE(std::abs(p[i] - static_cast<float>(start + i)) < 1e-5f);
}

void validate_int32_deterministic(const hmll_iobuf_t& buf, size_t numel) {
    REQUIRE(buf.ptr != nullptr);
    REQUIRE(buf.size >= numel * sizeof(int32_t));
    const auto* p = static_cast<const int32_t*>(buf.ptr);
    for (size_t i = 0; i < numel; ++i) {
        const auto expected = (static_cast<int64_t>(i) % (1 << 15)) - (1 << 14);
        REQUIRE(p[i] == static_cast<int32_t>(expected));
    }
}

void validate_uint8_deterministic(const hmll_iobuf_t& buf, size_t numel) {
    REQUIRE(buf.ptr != nullptr);
    REQUIRE(buf.size >= numel);
    const auto* p = static_cast<const uint8_t*>(buf.ptr);
    for (size_t i = 0; i < numel; ++i)
        REQUIRE(p[i] == static_cast<uint8_t>(i % 256));
}

// Return total bytes in dsts[0..n-1]
size_t total_dst_size(const hmll_iobuf_t* dsts, size_t n) {
    size_t t = 0;
    for (size_t i = 0; i < n; ++i) t += dsts[i].size;
    return t;
}

} // namespace

TEST_CASE("fetchv - single-element fetchv (n=1)", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        const auto* tensor_name = "float32.vec16";
        hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name);
        REQUIRE_FALSE(hmll_check(ctx.error));
        REQUIRE(lookup.specs != nullptr);

        hmll_range_t range = {lookup.specs->start, lookup.specs->end};
        hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
        REQUIRE_FALSE(hmll_check(ctx.error));

        size_t offsets[1] = {range.start};
        hmll_iobuf_t dsts[1] = {buffer};
        ssize_t ret = hmll_fetchv(&ctx, lookup.file, dsts, offsets, 1);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == buffer.size);
        validate_float32_arange(buffer, 16);

        hmll_free_buffer(&buffer);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - multi-element same dtype", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        const char* names[] = {"float32.vec16", "float32.vec1024", "float32.vec8192", "float32.scalar"};
        hmll_iobuf_t dsts[4];
        size_t offsets[4];
        size_t total = 0;
        for (int i = 0; i < 4; ++i) {
            hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, names[i]);
            REQUIRE(lookup.specs != nullptr);
            hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            dsts[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE_FALSE(hmll_check(ctx.error));
            offsets[i] = range.start;
            total += dsts[i].size;
        }

        ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 4);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == total);
        validate_float32_arange(dsts[0], 16);
        validate_float32_arange(dsts[1], 1024);
        validate_float32_arange(dsts[2], 8192);
        validate_float32_arange(dsts[3], 1);

        for (auto & dst : dsts) hmll_free_buffer(&dst);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - multi-element mixed dtypes", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        hmll_lookup_result_t l_f32 = hmll_lookup_tensor(&ctx, &registry, "float32.vec16");
        hmll_lookup_result_t l_i32 = hmll_lookup_tensor(&ctx, &registry, "int32.vec16");
        hmll_lookup_result_t l_u8 = hmll_lookup_tensor(&ctx, &registry, "uint8.vec16");
        REQUIRE((l_f32.specs && l_i32.specs && l_u8.specs));

        hmll_iobuf_t dsts[3] = {
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l_f32.specs->start, l_f32.specs->end}),
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l_i32.specs->start, l_i32.specs->end}),
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l_u8.specs->start, l_u8.specs->end}),
        };
        size_t offsets[3] = {l_f32.specs->start, l_i32.specs->start, l_u8.specs->start};
        REQUIRE_FALSE(hmll_check(ctx.error));

        ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 3);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == dsts[0].size + dsts[1].size + dsts[2].size);
        validate_float32_arange(dsts[0], 16);
        validate_int32_deterministic(dsts[1], 16);
        validate_uint8_deterministic(dsts[2], 16);

        for (auto & dst : dsts) hmll_free_buffer(&dst);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - scattered reads within single tensor", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, "float32.vec8192");
        REQUIRE(lookup.specs != nullptr);
        const size_t elem_size = sizeof(float);
        const size_t base = lookup.specs->start;

        // Sub-ranges: [0..3], [100..103], [4000..4003]
        struct { size_t start_off; size_t numel; } ranges[] = {
            {0 * elem_size, 4},
            {100 * elem_size, 4},
            {4000 * elem_size, 4},
        };
        hmll_iobuf_t dsts[3];
        size_t offsets[3];
        for (int i = 0; i < 3; ++i) {
            size_t len = ranges[i].numel * elem_size;
            hmll_range_t r = {base + ranges[i].start_off, base + ranges[i].start_off + len};
            dsts[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r);
            offsets[i] = r.start;
        }

        ssize_t ret = hmll_fetchv(&ctx, lookup.file, dsts, offsets, 3);
        REQUIRE(ret >= 0);
        validate_float32_arange(dsts[0], 4, 0);
        validate_float32_arange(dsts[1], 4, 100);
        validate_float32_arange(dsts[2], 4, 4000);

        for (auto & dst : dsts) hmll_free_buffer(&dst);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - full tensor via fetchv matches hmll_fetch", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, "float32.vec1024");
        REQUIRE(lookup.specs != nullptr);
        hmll_range_t range = {lookup.specs->start, lookup.specs->end};
        hmll_iobuf_t buf_fetch = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
        hmll_iobuf_t buf_fetchv = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
        REQUIRE_FALSE(hmll_check(ctx.error));

        ssize_t r1 = hmll_fetch(&ctx, lookup.file, &buf_fetch, range.start);
        size_t offsets[1] = {range.start};
        hmll_iobuf_t dsts[1] = {buf_fetchv};
        ssize_t r2 = hmll_fetchv(&ctx, lookup.file, dsts, offsets, 1);
        REQUIRE(r1 > 0);
        REQUIRE(r2 > 0);
        REQUIRE(static_cast<size_t>(r1) == buf_fetch.size);
        REQUIRE(static_cast<size_t>(r2) == buf_fetchv.size);
        REQUIRE(std::memcmp(buf_fetch.ptr, buf_fetchv.ptr, buf_fetch.size) == 0);

        hmll_free_buffer(&buf_fetch);
        hmll_free_buffer(&buf_fetchv);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - n=0 returns 0", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, kBackends[0].second)));

    ssize_t ret = hmll_fetchv(&ctx, 0, nullptr, nullptr, 0);
    REQUIRE(ret == 0);
    REQUIRE_FALSE(hmll_check(ctx.error));

    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - scalar tensors", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        hmll_lookup_result_t l0 = hmll_lookup_tensor(&ctx, &registry, "float32.scalar");
        hmll_lookup_result_t l1 = hmll_lookup_tensor(&ctx, &registry, "int32.scalar");
        hmll_lookup_result_t l2 = hmll_lookup_tensor(&ctx, &registry, "uint8.scalar");
        REQUIRE((l0.specs && l1.specs && l2.specs));

        hmll_iobuf_t dsts[3] = {
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l0.specs->start, l0.specs->end}),
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l1.specs->start, l1.specs->end}),
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l2.specs->start, l2.specs->end}),
        };
        size_t offsets[3] = {l0.specs->start, l1.specs->start, l2.specs->start};

        ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 3);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == dsts[0].size + dsts[1].size + dsts[2].size);
        validate_float32_arange(dsts[0], 1);
        validate_int32_deterministic(dsts[1], 1);
        validate_uint8_deterministic(dsts[2], 1);

        for (auto & dst : dsts) hmll_free_buffer(&dst);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - large tensor exceeds io_uring buffer", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, "float32.large");
        REQUIRE(lookup.specs != nullptr);
        hmll_range_t range = {lookup.specs->start, lookup.specs->end};
        hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
        REQUIRE_FALSE(hmll_check(ctx.error));
        REQUIRE(buffer.size > 512 * 1024u); // > HMLL_URING_BUFFER_SIZE

        size_t offsets[1] = {range.start};
        hmll_iobuf_t dsts[1] = {buffer};
        ssize_t ret = hmll_fetchv(&ctx, lookup.file, dsts, offsets, 1);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == buffer.size);
        size_t numel = buffer.size / sizeof(float);
        validate_float32_arange(buffer, numel);

        hmll_free_buffer(&buffer);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - many concurrent ranges", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        const size_t N = 40;
        std::vector<hmll_lookup_result_t> lookups(N);
        std::vector<hmll_iobuf_t> dsts(N);
        std::vector<size_t> offsets(N);
        for (size_t i = 0; i < N; ++i) {
            lookups[i] = hmll_lookup_tensor(&ctx, &registry, "float32.vec16");
            REQUIRE(lookups[i].specs != nullptr);
            hmll_range_t range = {lookups[i].specs->start, lookups[i].specs->end};
            dsts[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            offsets[i] = range.start;
        }

        ssize_t ret = hmll_fetchv(&ctx, 0, dsts.data(), offsets.data(), N);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == total_dst_size(dsts.data(), N));
        for (size_t i = 0; i < N; ++i)
            validate_float32_arange(dsts[i], 16);

        for (size_t i = 0; i < N; ++i) hmll_free_buffer(&dsts[i]);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - return value equals sum of dst sizes", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, kBackends[0].second)));

    const char* names[] = {"float32.vec16", "int32.vec16", "float32.scalar"};
    hmll_iobuf_t dsts[3];
    size_t offsets[3];
    size_t expected_total = 0;
    for (int i = 0; i < 3; ++i) {
        hmll_lookup_result_t l = hmll_lookup_tensor(&ctx, &registry, names[i]);
        REQUIRE(l.specs != nullptr);
        hmll_range_t r = {l.specs->start, l.specs->end};
        dsts[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r);
        offsets[i] = r.start;
        expected_total += dsts[i].size;
    }
    ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 3);
    REQUIRE(ret >= 0);
    REQUIRE(static_cast<size_t>(ret) == expected_total);

    for (auto & dst : dsts) hmll_free_buffer(&dst);
    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - overlapping logical range same data", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, kBackends[0].second)));

    hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, "float32.vec16");
    REQUIRE(lookup.specs != nullptr);
    hmll_range_t range = {lookup.specs->start, lookup.specs->end};
    hmll_iobuf_t buf1 = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
    hmll_iobuf_t buf2 = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
    size_t offsets[2] = {range.start, range.start};
    hmll_iobuf_t dsts[2] = {buf1, buf2};
    ssize_t ret = hmll_fetchv(&ctx, lookup.file, dsts, offsets, 2);
    REQUIRE(ret >= 0);
    REQUIRE(std::memcmp(buf1.ptr, buf2.ptr, buf1.size) == 0);
    hmll_free_buffer(&buf1);
    hmll_free_buffer(&buf2);
    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
    hmll_source_close(&src);
}

// --- Sharded tests ---
TEST_CASE("fetchv - sharded index parsing", "[fetchv][safetensors][sharded]") {
    const char* index_path = std::getenv(HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH);
    if (!index_path) SKIP("HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t index_src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(index_path, &index_src)));
    hmll_registry_t registry = {};
    size_t num_files = hmll_safetensors_index(&ctx, &registry, index_src);
    REQUIRE(num_files > 0);
    REQUIRE(registry.num_tensors > 0);
    REQUIRE(registry.indexes != nullptr);
    REQUIRE(registry.names != nullptr);
    REQUIRE(registry.tensors != nullptr);
    hmll_source_close(&index_src);
    hmll_free_registry(&registry);
}

TEST_CASE("fetchv - sharded fetchv across files", "[fetchv][safetensors][sharded]") {
    const char* index_path = std::getenv(HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH);
    if (!index_path) SKIP("HMLL_CI_FETCHV_SHARDED_SAFETENSORS_FPATH not set");

    std::string dir(index_path);
    size_t slash = dir.find_last_of("/\\");
    if (slash != std::string::npos) dir.resize(slash + 1);
    else dir = "./";

    hmll_t ctx = {};
    hmll_source_t index_src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(index_path, &index_src)));
    hmll_registry_t registry = {};
    size_t num_files = hmll_safetensors_index(&ctx, &registry, index_src);
    REQUIRE(num_files > 0);
    hmll_source_close(&index_src);

    std::vector<hmll_source_t> sources(num_files);
    for (size_t i = 0; i < num_files; ++i) {
        char path[512];
        int n = std::snprintf(path, sizeof(path), "%smodel-%05zu-of-%05zu.safetensors",
                              dir.c_str(), i + 1, num_files);
        REQUIRE((n > 0 && static_cast<size_t>(n) < sizeof(path)));
        REQUIRE_FALSE(hmll_check(hmll_source_open(path, &sources[i])));
    }

    size_t offset = 0;
    for (size_t i = 0; i < num_files; ++i) {
        size_t n = hmll_safetensors_populate_registry(&ctx, &registry, sources[i], (unsigned short)i, offset);
        REQUIRE(n > 0);
        offset += n;
    }
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, sources.data(), num_files, HMLL_DEVICE_CPU, kBackends[0].second)));

    for (const auto& [name, backend] : kBackends) {
        hmll_destroy(&ctx);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, sources.data(), num_files, HMLL_DEVICE_CPU, backend)));

        hmll_lookup_result_t l0 = hmll_lookup_tensor(&ctx, &registry, "float32.shard0.vec16");
        hmll_lookup_result_t l1 = hmll_lookup_tensor(&ctx, &registry, "int32.shard1.vec16");
        hmll_lookup_result_t l2 = hmll_lookup_tensor(&ctx, &registry, "bfloat16.shard2.vec64");
        REQUIRE(l0.specs != nullptr);
        REQUIRE(l1.specs != nullptr);
        REQUIRE(l2.specs != nullptr);
        REQUIRE(l0.file == 0);
        REQUIRE(l1.file == 1);
        REQUIRE(l2.file == 2);

        hmll_iobuf_t dsts[3] = {
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l0.specs->start, l0.specs->end}),
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l1.specs->start, l1.specs->end}),
            hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, {l2.specs->start, l2.specs->end}),
        };
        size_t offsets[3] = {l0.specs->start, l1.specs->start, l2.specs->start};
        int iofiles[3] = {l0.file, l1.file, l2.file};

        ssize_t total = 0;
        for (int i = 0; i < 3; ++i) {
            hmll_iobuf_t d[1] = {dsts[i]};
            size_t o[1] = {offsets[i]};
            ssize_t r = hmll_fetchv(&ctx, iofiles[i], d, o, 1);
            REQUIRE(r >= 0);
            total += r;
        }
        validate_float32_arange(dsts[0], 16);
        validate_int32_deterministic(dsts[1], 16);
        REQUIRE(dsts[2].size == 64 * 2u); // bf16
        for (auto & dst : dsts) hmll_free_buffer(&dst);
    }

    for (size_t i = 0; i < num_files; ++i) hmll_source_close(&sources[i]);
    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
}

TEST_CASE("fetchv - fetchv matches fetch for several tensors", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        const char* names[] = {"float32.vec16", "int32.vec16", "uint8.vec16", "float32.scalar"};
        const size_t n_tensors = 4;
        std::vector<hmll_iobuf_t> buf_fetch(n_tensors);
        std::vector<hmll_iobuf_t> buf_fetchv(n_tensors);
        std::vector<size_t> offsets(n_tensors);
        size_t total = 0;
        for (size_t i = 0; i < n_tensors; ++i) {
            hmll_lookup_result_t l = hmll_lookup_tensor(&ctx, &registry, names[i]);
            REQUIRE(l.specs != nullptr);
            hmll_range_t r = {l.specs->start, l.specs->end};
            buf_fetch[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r);
            buf_fetchv[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r);
            offsets[i] = r.start;
            total += buf_fetch[i].size;
        }
        for (size_t i = 0; i < n_tensors; ++i) {
            hmll_lookup_result_t l = hmll_lookup_tensor(&ctx, &registry, names[i]);
            ssize_t r = hmll_fetch(&ctx, l.file, &buf_fetch[i], offsets[i]);
            REQUIRE(r > 0);
        }
        ssize_t rv = hmll_fetchv(&ctx, 0, buf_fetchv.data(), offsets.data(), n_tensors);
        REQUIRE(rv >= 0);
        REQUIRE(static_cast<size_t>(rv) == total);
        for (size_t i = 0; i < n_tensors; ++i)
            REQUIRE(std::memcmp(buf_fetch[i].ptr, buf_fetchv[i].ptr, buf_fetch[i].size) == 0);
        for (size_t i = 0; i < n_tensors; ++i) {
            hmll_free_buffer(&buf_fetch[i]);
            hmll_free_buffer(&buf_fetchv[i]);
        }
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - pre-existing error returns -1", "[fetchv][error]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, kBackends[0].second)));

    // Poison the context with an error
    ctx.error = HMLL_ERR(HMLL_ERR_IO_ERROR);

    hmll_iobuf_t dsts[1] = {};
    size_t offsets[1] = {0};
    ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 1);
    REQUIRE(ret == -1);

    ctx.error = HMLL_OK; // reset so cleanup works
    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - multiple large tensors interleaved", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);

    for (const auto& [name, backend] : kBackends) {
        INFO("Backend: " << name);
        REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, backend)));

        // Two large tensors, each > HMLL_URING_BUFFER_SIZE
        const char* names[] = {"float32.large", "int32.large"};
        hmll_iobuf_t dsts[2];
        size_t offsets[2];
        size_t total = 0;
        for (int i = 0; i < 2; ++i) {
            hmll_lookup_result_t l = hmll_lookup_tensor(&ctx, &registry, names[i]);
            REQUIRE(l.specs != nullptr);
            hmll_range_t r = {l.specs->start, l.specs->end};
            dsts[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r);
            offsets[i] = r.start;
            total += dsts[i].size;
            REQUIRE(dsts[i].size > 512 * 1024u);
        }

        ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 2);
        REQUIRE(ret >= 0);
        REQUIRE(static_cast<size_t>(ret) == total);
        validate_float32_arange(dsts[0], dsts[0].size / sizeof(float));
        validate_int32_deterministic(dsts[1], dsts[1].size / sizeof(int32_t));

        for (auto& dst : dsts) hmll_free_buffer(&dst);
        hmll_destroy(&ctx);
    }
    hmll_free_registry(&registry);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - heap scratch path (large N)", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, kBackends[0].second)));

    const size_t N = 300; // exceeds stack_scratch[8192]
    hmll_lookup_result_t l = hmll_lookup_tensor(&ctx, &registry, "float32.vec16");
    REQUIRE(l.specs != nullptr);

    std::vector<hmll_iobuf_t> dsts(N);
    std::vector<size_t> offsets(N);
    for (size_t i = 0; i < N; ++i) {
        hmll_range_t r = {l.specs->start, l.specs->end};
        dsts[i] = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r);
        offsets[i] = r.start;
    }

    ssize_t ret = hmll_fetchv(&ctx, 0, dsts.data(), offsets.data(), N);
    REQUIRE(ret >= 0);
    REQUIRE(static_cast<size_t>(ret) == total_dst_size(dsts.data(), N));
    for (size_t i = 0; i < N; ++i)
        validate_float32_arange(dsts[i], 16);

    for (size_t i = 0; i < N; ++i) hmll_free_buffer(&dsts[i]);
    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
    hmll_source_close(&src);
}

TEST_CASE("fetchv - zero-size buffer interspersed", "[fetchv][safetensors]") {
    const char* fpath = std::getenv(HMLL_CI_FETCHV_SAFETENSORS_FPATH);
    if (!fpath) SKIP("HMLL_CI_FETCHV_SAFETENSORS_FPATH not set");

    hmll_t ctx = {};
    hmll_source_t src = {};
    REQUIRE_FALSE(hmll_check(hmll_source_open(fpath, &src)));
    hmll_registry_t registry = {};
    REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
    REQUIRE_FALSE(hmll_check(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, kBackends[0].second)));

    hmll_lookup_result_t l = hmll_lookup_tensor(&ctx, &registry, "float32.vec16");
    REQUIRE(l.specs != nullptr);
    hmll_range_t r = {l.specs->start, l.specs->end};

    hmll_iobuf_t dsts[3] = {
        hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r),
        {.size = 0, .ptr = nullptr, .device = HMLL_DEVICE_CPU},  // zero-size
        hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, r),
    };
    size_t offsets[3] = {r.start, 0, r.start};

    ssize_t ret = hmll_fetchv(&ctx, 0, dsts, offsets, 3);
    REQUIRE(ret >= 0);
    REQUIRE(static_cast<size_t>(ret) == dsts[0].size + dsts[2].size);
    validate_float32_arange(dsts[0], 16);
    validate_float32_arange(dsts[2], 16);

    hmll_free_buffer(&dsts[0]);
    hmll_free_buffer(&dsts[2]);
    hmll_free_registry(&registry);
    hmll_destroy(&ctx);
    hmll_source_close(&src);
}