#ifndef PYHMLL_SPECS_HPP
#define PYHMLL_SPECS_HPP

#include <vector>
#include <hmll/types.h>

class HmllTensorSpecs
{
    hmll_tensor_specs_t specs;

public:
    explicit HmllTensorSpecs(hmll_tensor_specs_t specs) : specs(specs) {}

    [[nodiscard]]
    size_t rank() const;

    [[nodiscard]]
    std::vector<size_t> shape() const;
};
#endif // PYHMLL_SPECS_HPP
