#ifndef PYHMLL_CONTEXT_H
#define PYHMLL_CONTEXT_H

#include <memory>
#include <optional>
#include <hmll/types.h>

#include "specs.hpp"

class HmllContext
{
    hmll_context_t* ctx_;

public:
    HmllContext() = default;
    explicit HmllContext(hmll_context_t* ctx): ctx_(ctx) {}

    static HmllContext open(const std::string& path, hmll_file_kind kind, int flags);

    /// Return a flag indicating if the underlying context is in error
    /// @return
    [[nodiscard]]
    bool has_error() const;

    /// Return the number of tensors known in the current context
    /// @return
    [[nodiscard]]
    size_t num_tensors() const;

    /// Check if the name of the tensor is present in the internal tensors table
    /// @param name Name of the tensor to look for
    /// @return True if the name is known in the internal tensors table, false if not.
    [[nodiscard]]
    bool contains(const std::string& name) const;

    /// Attempt to retrieve the target tensor specification if present in the tensors' table
    /// @param name Name of the tensor to look for
    /// @return
    [[nodiscard]]
    HmllTensorSpecs tensor(const std::string& name) const;

    void fetcher() const;
};
#endif // PYHMLL_CONTEXT_H
