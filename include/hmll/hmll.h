#ifndef HMLL_H
#define HMLL_H
#ifdef __cplusplus
#define NO_EXCEPT noexcept
extern "C" {
#else
#define NO_EXCEPT
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

HMLL_EXTERN void hmll_context_free(const hmll_context_t *) NO_EXCEPT;
HMLL_EXTERN hmll_status_t hmll_open(const char *, hmll_context_t *, hmll_file_kind_t, hmll_flags_t) NO_EXCEPT;
HMLL_EXTERN hmll_status_t hmll_close(const char *, hmll_context_t *, hmll_flags_t) NO_EXCEPT;

HMLL_EXTERN void hmll_tensor_specs_free(hmll_tensor_specs_t *) NO_EXCEPT;
HMLL_EXTERN hmll_status_t hmll_get_tensor_specs(const hmll_context_t *, const char *, hmll_tensor_specs_t **) NO_EXCEPT;
HMLL_EXTERN uint8_t hmll_get_dtype_nbytes(hmll_tensor_data_type_t) NO_EXCEPT;

#ifdef __cplusplus
}
#endif
#endif // HMLL_H
