#ifndef HMLL_H
#define HMLL_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HMLL_EXTERN
#ifndef HMLL_STATIC
#ifdef _WIN32
#define HMLL_EXTERN __declspec(dllimport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define HMLL_EXTERN __attribute__((visibility("default")))
#else
#define HMLL_EXTERN
#endif
#else
#define HMLL_EXTERN
#endif
#endif

#include "status.h"
#include "types.h"
#include "safetensors.h"

HMLL_EXTERN hmll_status_t hmll_context_free(hmll_context_t *);

HMLL_EXTERN hmll_status_t hmll_open(const char *, hmll_context_t *, hmll_file_kind_t, hmll_flags_t);
HMLL_EXTERN hmll_status_t hmll_close(const char *, hmll_context_t *, hmll_flags_t);

HMLL_EXTERN hmll_status_t hmll_fetcher_init(hmll_fetcher_kind_t *);
HMLL_EXTERN hmll_status_t hmll_fetcher_free();

#ifdef __cplusplus
}
#endif
#endif // HMLL_H
