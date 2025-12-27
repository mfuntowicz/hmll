//
// Created by mfuntowicz on 12/19/25.
//

#include <catch2/catch_all.hpp>
#include <hmll/memory.h>

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