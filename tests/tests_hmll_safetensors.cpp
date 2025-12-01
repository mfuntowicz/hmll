#include <catch2/catch_test_macros.hpp>

#include <hmll/hmll.h>

TEST_CASE("Safetensors dtype", "[dtype][safetensors]")
{
    REQUIRE(hmll_safetensors_dtype_from_str("BF16", 4) == HMLL_DTYPE_BFLOAT16);
    REQUIRE(hmll_safetensors_dtype_from_str("FP16", 4) == HMLL_DTYPE_FLOAT16);
    REQUIRE(hmll_safetensors_dtype_from_str("FP32", 4) == HMLL_DTYPE_FLOAT32);

    REQUIRE(hmll_safetensors_dtype_from_str("BF32", 4) == HMLL_DTYPE_UNKNOWN);
}