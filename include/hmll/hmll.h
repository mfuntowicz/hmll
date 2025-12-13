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

#define HMLL_SUCCESS 0U

#include "types.h"

HMLL_EXTERN unsigned int hmll_success(const struct hmll_context *);
HMLL_EXTERN unsigned int hmll_has_error(const struct hmll_context *);
HMLL_EXTERN char *hmll_strerr(enum hmll_error_code);

HMLL_EXTERN enum hmll_error_code hmll_open(const char *, hmll_context_t *, hmll_file_kind_t, hmll_flags_t) NO_EXCEPT;
HMLL_EXTERN void hmll_destroy(struct hmll_context *) NO_EXCEPT;
HMLL_EXTERN uint8_t hmll_sizeof(enum hmll_tensor_data_type) NO_EXCEPT;
HMLL_EXTERN size_t hmll_numel(struct hmll_tensor_specs *) NO_EXCEPT;

HMLL_EXTERN hmll_tensor_specs_t hmll_get_tensor_specs(struct hmll_context *, const char *) NO_EXCEPT;
void *hmll_get_buffer(struct hmll_context *, size_t) NO_EXCEPT;
void *hmll_get_io_buffer(struct hmll_context *, enum hmll_device, size_t) NO_EXCEPT;

#ifdef __cplusplus
}
#endif
#endif // HMLL_H
