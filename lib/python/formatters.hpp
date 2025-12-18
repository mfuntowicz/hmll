#ifndef PYHMLL_FORMATTERS_H
#define PYHMLL_FORMATTERS_H

#include <format>
#include "hmll/types.h"

template <>
struct std::formatter<hmll_device_t> : std::formatter<std::string> {
    auto format(const hmll_device_t& device, std::format_context& ctx) const {
        switch (device) {
        case HMLL_DEVICE_CPU:
            return std::formatter<std::string>::format("CPU", ctx);
        default:
            return std::formatter<std::string>::format("unknown", ctx);
        }
    }
};

template <>
struct std::formatter<hmll_fetcher_kind_t> : std::formatter<std::string> {
    auto format(const hmll_fetcher_kind_t& kind, std::format_context& ctx) const {
        return std::formatter<std::string>::format("opaque", ctx);
    }
};

#endif // PYHMLL_FORMATTERS_H