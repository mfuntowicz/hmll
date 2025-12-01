//
// Created by mfuntowicz on 12/1/25.
//

#include <catch2/catch_all.hpp>

#include "hmll/hmll.h"

TEST_CASE("Status succeeded", "[status]")
{
    REQUIRE(hmll_success(HMLL_SUCCEEDED));
    REQUIRE(hmll_success({HMLL_SUCCESS, nullptr}));
    REQUIRE(hmll_success({HMLL_SUCCESS, "With a message"}));
}