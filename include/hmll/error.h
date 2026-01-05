//
// Created by mfuntowicz on 1/5/26.
//

#ifndef HMLL_ERROR_H
#define HMLL_ERROR_H

#include "types.h"

#define HMLL_OS_ERROR_OFFSET (-1000)

static inline int hmll_error_from_os(const int err) { return HMLL_OS_ERROR_OFFSET + err; }
static inline int hmll_error_from_lib(const enum hmll_status_code code) { return code; }

#endif //HMLL_ERROR_H