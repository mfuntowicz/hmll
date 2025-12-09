//
// Created by mfuntowicz on 12/1/25.
//

#include "hmll/status.h"

bool hmll_success(const struct hmll_status status)
{
    return status.what == HMLL_STATUS_OK;
}
