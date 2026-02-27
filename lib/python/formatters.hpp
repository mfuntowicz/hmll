#ifndef PYHMLL_FORMATTERS_H
#define PYHMLL_FORMATTERS_H

#include <fmt/format.h>
#include "hmll/types.h"

inline std::string format_as(const hmll_device_t d) {
  return hmll_device_is_cpu(d) ? "CPU" : fmt::format("CUDA:{}", d.idx);
}
constexpr auto format_as(const hmll_fetcher_kind_t f) { return fmt::underlying(f); }

#endif // PYHMLL_FORMATTERS_H
