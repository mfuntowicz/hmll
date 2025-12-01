//
// Created by mfuntowicz on 12/1/25.
//

#include "hmll/status.h"

bool hmll_status_succeeded(const struct hmll_status status)
{
    return status.what == HMLL_STATUS_OK;
}

bool hmll_status_has_error(const struct hmll_status status)
{
    return status.what != HMLL_STATUS_OK;
}
