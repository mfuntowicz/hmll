//
// Created by mfuntowicz on 12/9/25.
//

#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"

TEST_CASE("Number of elements in a tensor", "[tensor]")
{
    hmll_tensor_specs_t specs;
    specs.dtype = HMLL_DTYPE_FLOAT32;
    specs.shape[0] = 4096;
    specs.shape[1] = 128;
    specs.rank = 2;
    REQUIRE(hmll_numel(&specs) == (128 * 4096));
}