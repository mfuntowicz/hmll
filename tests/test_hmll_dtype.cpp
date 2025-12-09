//
// Created by mfuntowicz on 12/8/25.
//

#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"

TEST_CASE("sizeof dtype", "[dtype]")
{
    REQUIRE(hmll_sizeof(HMLL_DTYPE_BFLOAT16) == 2);
    REQUIRE(hmll_sizeof(HMLL_DTYPE_FLOAT16)  == 2);
    REQUIRE(hmll_sizeof(HMLL_DTYPE_FLOAT32)  == 4);
}