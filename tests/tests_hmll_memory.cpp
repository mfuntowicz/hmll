//
// Created by mfuntowicz on 12/19/25.
//

#include <catch2/catch_all.hpp>
#include <hmll/hmll.h>

TEST_CASE("page-aligned", "[memory][alignment]")
{
    REQUIRE(hmll_is_aligned(4096, 4096));
    REQUIRE(hmll_is_aligned(4096 * 0, 4096));
    REQUIRE(hmll_is_aligned(4096 * 128, 4096));
}

TEST_CASE("not page-aligned", "[memory][alignment]")
{
    REQUIRE_FALSE(hmll_is_aligned(4095, 4096));
    REQUIRE_FALSE(hmll_is_aligned(1, 4096));
    REQUIRE_FALSE(hmll_is_aligned(4096 * 128 - 1, 4096));
}

TEST_CASE("release small memory chunk", "[memory]")
{
    constexpr size_t nbytes = 1024;

    hmll_t ctx;
    hmll_iobuf buf = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, nbytes, HMLL_MEM_DEVICE);

    REQUIRE(buf.ptr != nullptr);
    REQUIRE(buf.size == nbytes);

    hmll_free_buffer(&buf);

    REQUIRE(buf.ptr == nullptr);
    REQUIRE(buf.size == 0);
}

TEST_CASE("release large memory chunk", "[memory]")
{
    constexpr size_t nbytes = 1024 * 1024 * 500;

    hmll_t ctx;
    hmll_iobuf buf = hmll_get_buffer(&ctx, HMLL_DEVICE_CPU, nbytes, HMLL_MEM_DEVICE);

    REQUIRE(buf.ptr != nullptr);
    REQUIRE(buf.size == nbytes);

    hmll_free_buffer(&buf);

    REQUIRE(buf.ptr == nullptr);
    REQUIRE(buf.size == 0);
}