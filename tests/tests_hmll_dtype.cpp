//
// Created by mfuntowicz on 12/8/25.
//

#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"

TEST_CASE("sizeof dtype", "[dtype]")
{
    REQUIRE(hmll_nbits(HMLL_DTYPE_BOOL)            == 8);
    REQUIRE(hmll_nbits(HMLL_DTYPE_BFLOAT16)        == 16);
    REQUIRE(hmll_nbits(HMLL_DTYPE_COMPLEX)         == 64);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT4)          == 4);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT6_E2M3)     == 6);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT6_E3M2)     == 6);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT8_E4M3)     == 8);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT8_E5M2)     == 8);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT8_E8M0)     == 8);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT16)         == 16);
    REQUIRE(hmll_nbits(HMLL_DTYPE_FLOAT32)         == 32);
    REQUIRE(hmll_nbits(HMLL_DTYPE_SIGNED_INT4)     == 4);
    REQUIRE(hmll_nbits(HMLL_DTYPE_SIGNED_INT8)     == 8);
    REQUIRE(hmll_nbits(HMLL_DTYPE_SIGNED_INT16)    == 16);
    REQUIRE(hmll_nbits(HMLL_DTYPE_SIGNED_INT32)    == 32);
    REQUIRE(hmll_nbits(HMLL_DTYPE_SIGNED_INT64)    == 64);
    REQUIRE(hmll_nbits(HMLL_DTYPE_UNSIGNED_INT4)   == 4);
    REQUIRE(hmll_nbits(HMLL_DTYPE_UNSIGNED_INT8)   == 8);
    REQUIRE(hmll_nbits(HMLL_DTYPE_UNSIGNED_INT16)  == 16);
    REQUIRE(hmll_nbits(HMLL_DTYPE_UNSIGNED_INT32)  == 32);
    REQUIRE(hmll_nbits(HMLL_DTYPE_UNSIGNED_INT64)  == 64);
    REQUIRE(hmll_nbits(HMLL_DTYPE_UNKNOWN)         == 0);
}
