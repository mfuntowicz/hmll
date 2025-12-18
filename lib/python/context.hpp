#ifndef PYHMLL_CONTEXT_H
#define PYHMLL_CONTEXT_H

#include <memory>
#include <hmll/fetcher.h>
#include <hmll/types.h>

#include "specs.hpp"

class HmllFetcher;
class HmllContext
{
    friend HmllFetcher;
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

    /// Create the specified fetcher to efficiently retrieve data from stored tensors
    /// @param device Target device to write data to
    /// @param kind Target fetcher type or AUTO to select the most efficient one depending on the system
    /// @return
    [[nodiscard]]
    HmllFetcher fetcher(hmll_device_t device, hmll_fetcher_kind_t kind) const;
};
#endif // PYHMLL_CONTEXT_H
