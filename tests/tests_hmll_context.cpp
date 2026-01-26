//
// Created by mfuntowicz on 1/9/26.
//

#include <catch2/catch_all.hpp>

#include "hmll/hmll.h"

TEST_CASE("hmll_clone_context handles null source", "[context]")
{
    hmll_t dst = {};
    auto err = hmll_clone_context(nullptr, &dst);
    REQUIRE_FALSE(hmll_success(err));
    REQUIRE(err.code == HMLL_ERR_INVALID_RANGE);
}

TEST_CASE("hmll_clone_context handles null destination", "[context]")
{
    hmll_t src = {};
    auto err = hmll_clone_context(&src, nullptr);
    REQUIRE_FALSE(hmll_success(err));
    REQUIRE(err.code == HMLL_ERR_INVALID_RANGE);
}

TEST_CASE("hmll_clone_context handles both null", "[context]")
{
    auto err = hmll_clone_context(nullptr, nullptr);
    REQUIRE_FALSE(hmll_success(err));
    REQUIRE(err.code == HMLL_ERR_INVALID_RANGE);
}

TEST_CASE("hmll_clone_context copies shared resources", "[context]")
{
    hmll_t src = {};
    hmll_loader_t fetcher = {};
    hmll_source_t source = {};

    src.fetcher = &fetcher;
    src.sources = &source;
    src.num_sources = 1;
    src.error = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);

    hmll_t dst = {};
    auto err = hmll_clone_context(&dst, &src);

    REQUIRE(hmll_success(err));
    REQUIRE(dst.sources == src.sources);
    REQUIRE(dst.num_sources == src.num_sources);
    // Note: fetcher is intentionally reset to NULL in clone
}

TEST_CASE("hmll_clone_context resets error state", "[context]")
{
    hmll_t src = {};
    src.error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);

    hmll_t dst = {};
    dst.error = HMLL_ERR(HMLL_ERR_TENSOR_NOT_FOUND);

    auto err = hmll_clone_context(&dst, &src);

    REQUIRE(hmll_success(err));
    REQUIRE(hmll_success(dst.error));
    REQUIRE(dst.error.code == HMLL_ERR_SUCCESS);
    REQUIRE(dst.error.sys_err == 0);
}

TEST_CASE("hmll_clone_context resets system error state", "[context]")
{
    hmll_t src = {};
    src.error = HMLL_SYS_ERR(42);

    hmll_t dst = {};
    auto err = hmll_clone_context(&dst, &src);

    REQUIRE(hmll_success(err));
    REQUIRE(hmll_success(dst.error));
    REQUIRE(dst.error.sys_err == 0);
}

TEST_CASE("hmll_clone_context copies multiple sources", "[context]")
{
    hmll_t src = {};
    hmll_source_t sources[3] = {};

    src.sources = sources;
    src.num_sources = 3;

    hmll_t dst = {};
    auto err = hmll_clone_context(&dst, &src);

    REQUIRE(hmll_success(err));
    REQUIRE(dst.sources == src.sources);
    REQUIRE(dst.num_sources == 3);
}

TEST_CASE("hmll_clone_context handles null fetcher", "[context]")
{
    hmll_t src = {};
    src.fetcher = nullptr;
    src.sources = nullptr;
    src.num_sources = 0;

    hmll_t dst = {};
    auto err = hmll_clone_context(&dst, &src);

    REQUIRE(hmll_success(err));
    REQUIRE(dst.fetcher == nullptr);
}

TEST_CASE("hmll_clone_context does not modify source", "[context]")
{
    hmll_t src = {};
    hmll_loader_t fetcher = {};
    hmll_source_t source = {};

    src.fetcher = &fetcher;
    src.sources = &source;
    src.num_sources = 1;
    src.error = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);

    // Take snapshot of source state
    auto original_fetcher = src.fetcher;
    auto original_sources = src.sources;
    auto original_num_sources = src.num_sources;
    auto original_error_code = src.error.code;

    hmll_t dst = {};
    hmll_clone_context(&dst, &src);

    // Verify source is unchanged
    REQUIRE(src.fetcher == original_fetcher);
    REQUIRE(src.sources == original_sources);
    REQUIRE(src.num_sources == original_num_sources);
    REQUIRE(src.error.code == original_error_code);
}

TEST_CASE("hmll_clone_context can be called multiple times", "[context]")
{
    hmll_t src = {};
    hmll_loader_t fetcher = {};
    hmll_source_t source = {};
    src.fetcher = &fetcher;
    src.sources = &source;
    src.num_sources = 1;

    hmll_t dst1 = {};
    hmll_t dst2 = {};
    hmll_t dst3 = {};

    REQUIRE(hmll_success(hmll_clone_context(&dst1, &src)));
    REQUIRE(hmll_success(hmll_clone_context(&dst2, &src)));
    REQUIRE(hmll_success(hmll_clone_context(&dst3, &src)));

    // Sources are shared, fetcher is reset to NULL
    REQUIRE(dst1.sources == src.sources);
    REQUIRE(dst2.sources == src.sources);
    REQUIRE(dst3.sources == src.sources);
}

TEST_CASE("hmll_clone_context allows independent error states", "[context]")
{
    hmll_t src = {};
    hmll_loader_t fetcher = {};
    src.fetcher = &fetcher;
    src.error = HMLL_OK;

    hmll_t dst1 = {};
    hmll_t dst2 = {};

    REQUIRE(hmll_success(hmll_clone_context(&dst1, &src)));
    REQUIRE(hmll_success(hmll_clone_context(&dst2, &src)));

    // Modify error states independently
    dst1.error = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);
    dst2.error = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);

    // Verify they don't affect each other or source
    REQUIRE(hmll_success(src.error));
    REQUIRE(dst1.error.code == HMLL_ERR_FILE_NOT_FOUND);
    REQUIRE(dst2.error.code == HMLL_ERR_ALLOCATION_FAILED);
}
