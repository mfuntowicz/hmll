#ifndef HMLL_TYPES_H
#define HMLL_TYPES_H

enum hmll_flags
{
    hmll_hmll = 1 << 0,
    hmll_gguf = 1 << 1,
    hmll_skip_metadata = 1 << 2
};
typedef hmll_flags hmll_flags_t;

struct hmll_context {
    
};
typedef hmll_context hmll_context_t;

#endif // HMLL_TYPES_H
