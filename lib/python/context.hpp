#ifndef PYHMLL_CONTEXT_H
#define PYHMLL_CONTEXT_H

#include <memory>
#include <span>
#include <hmll/types.h>

class HmllFetcher;
class Context
{
    friend HmllFetcher;
    std::unique_ptr<hmll_t> ctx_;

public:
    Context() = default;
    explicit Context(std::unique_ptr<hmll_t> ctx): ctx_(std::move(ctx)) {}

    // Destructor
    ~Context();

    // Copy operations for clonability
    Context(const Context& other);
    Context& operator=(const Context& other);

    // Move operations
    Context(Context&& other) noexcept = default;
    Context& operator=(Context&& other) noexcept = default;

    static Context open(const std::string& path);
    static Context open(std::span<const std::string> paths);

    /// Return a flag indicating if the underlying context is in error
    /// @return
    [[nodiscard]]
    bool has_error() const;

    bool has_library_error() const;
    bool has_system_error() const;
};
#endif // PYHMLL_CONTEXT_H
