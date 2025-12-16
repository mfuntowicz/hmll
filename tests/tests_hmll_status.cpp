//
// Created by mfuntowicz on 12/1/25.
//

#include <catch2/catch_all.hpp>

#include "hmll/hmll.h"

TEST_CASE("success", "[status]")
{
    REQUIRE(hmll_success(HMLL_ERR_SUCCESS));
    REQUIRE_FALSE(hmll_has_error(HMLL_ERR_SUCCESS));
}