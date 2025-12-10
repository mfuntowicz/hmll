//
// Created by mfuntowicz on 12/9/25.
//

#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"

TEST_CASE("Number of elements in a tensor", "[tensor]")
{
    std::vector<size_t> shape = {128, 4096};
    hmll_tensor_specs_t specs = {0, 10, shape.data(), 2, HMLL_DTYPE_BFLOAT16};

    REQUIRE(hmll_numel(&specs) == (128 * 4096));
}