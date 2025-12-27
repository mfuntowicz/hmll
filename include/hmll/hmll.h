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

#define HMLL_FALSE   0u
#define HMLL_SUCCESS 0U
#define HMLL_UNUSED(expr) (void)(expr);

#ifdef __HMLL_PROFILE_ENABLED__
#include <tracy/TracyC.h>
#define HMLL_MARK_ZONE_ENTER(name) TracyCZone(ctx_##name, 1); TracyCZoneName(ctx_##name, #name, strlen(#name))
#define HMLL_MARK_ZONE_EXIT(name) TracyCZoneEnd(ctx_##name)
#else
#define HMLL_MARK_ZONE_ENTER(name)
#define HMLL_MARK_ZONE_EXIT(name)
#endif

#include "fetcher.h"
#include "types.h"

HMLL_EXTERN unsigned int hmll_success(enum hmll_error_code errn);
HMLL_EXTERN unsigned int hmll_has_error(enum hmll_error_code errn);
HMLL_EXTERN enum hmll_error_code hmll_get_error(const struct hmll_context *);
HMLL_EXTERN char *hmll_strerr(enum hmll_error_code);

HMLL_EXTERN int hmll_open(const char *, hmll_context_t *, hmll_file_kind_t, hmll_flags_t) NO_EXCEPT;
HMLL_EXTERN void hmll_destroy(struct hmll_context *) NO_EXCEPT;
HMLL_EXTERN int hmll_find_by_name(const struct hmll_context *, const char *) NO_EXCEPT;
HMLL_EXTERN int hmll_contains(const struct hmll_context *, const char *) NO_EXCEPT;
HMLL_EXTERN uint8_t hmll_sizeof(enum hmll_tensor_data_type) NO_EXCEPT;
HMLL_EXTERN size_t hmll_numel(const struct hmll_tensor_specs *) NO_EXCEPT;
HMLL_EXTERN struct hmll_tensor_lookup_result hmll_get_tensor_specs(const struct hmll_context *, const char *) NO_EXCEPT;

void *hmll_get_buffer(struct hmll_context *, enum hmll_device, size_t) NO_EXCEPT;
struct hmll_device_buffer hmll_get_buffer_for_range(struct hmll_context *, enum hmll_device, struct hmll_range) NO_EXCEPT;
void *hmll_get_io_buffer(struct hmll_context *, enum hmll_device, size_t) NO_EXCEPT;

HMLL_EXTERN struct hmll_fetcher hmll_fetcher_init(struct hmll_context *, enum hmll_device, enum hmll_fetcher_kind kind);
HMLL_EXTERN struct hmll_range hmll_fetch_tensor(struct hmll_context *, struct hmll_fetcher, const char *, struct hmll_device_buffer);
HMLL_EXTERN struct hmll_range hmll_fetch_range(struct hmll_context *, struct hmll_fetcher, struct hmll_range, struct hmll_device_buffer);

#ifdef __cplusplus
}
#endif
#endif // HMLL_H
