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
#define HMLL_UNUSED(expr) (void)(expr)

#include "loader.h"
#include "memory.h"
#include "types.h"

#ifdef __linux__
#include "unix/file.h"
#include "unix/loader.h"
#endif

/** Error handling stubs **/
#define HMLL_RES(...) (struct hmll_error){ __VA_ARGS__ }
#define HMLL_OK  HMLL_RES(.code = HMLL_ERR_SUCCESS, .sys_err = 0)
#define HMLL_ERR(c) HMLL_RES(.code = (c), .sys_err = 0)
#define HMLL_SYS_ERR(e) HMLL_RES(.code = HMLL_ERR_SYSTEM, .sys_err = (e))

static inline unsigned char hmll_check(const struct hmll_error res)
{
    return res.code != HMLL_ERR_SUCCESS || res.sys_err != HMLL_ERR_SUCCESS;
}

static inline unsigned char hmll_success(const struct hmll_error error)
{
    return (int)error.code == HMLL_ERR_SUCCESS && error.sys_err == HMLL_ERR_SUCCESS;
}
HMLL_EXTERN unsigned char hmll_error_is_os_error(struct hmll_error err);
HMLL_EXTERN unsigned char hmll_error_is_lib_error(struct hmll_error err);
HMLL_EXTERN const char *hmll_strerr(struct hmll_error err);
HMLL_EXTERN void hmll_destroy(struct hmll *ctx) NO_EXCEPT;

/** Sources handling stubs **/
HMLL_EXTERN struct hmll_error hmll_source_open(const char *path, struct hmll_source *src) NO_EXCEPT;
HMLL_EXTERN void hmll_source_close(const struct hmll_source *src) NO_EXCEPT;

/** Memory handling stubs **/
void *hmll_get_buffer(struct hmll *ctx, enum hmll_device device, size_t size);
void *hmll_get_io_buffer(struct hmll *ctx, enum hmll_device device, size_t size);
struct hmll_iobuf hmll_get_buffer_for_range(struct hmll *ctx, enum hmll_device device, struct hmll_range range);

HMLL_EXTERN struct hmll_error hmll_loader_init(struct hmll *ctx, const struct hmll_source *srcs, size_t n, enum hmll_device device, enum hmll_loader_kind kind) NO_EXCEPT;
HMLL_EXTERN struct hmll_range hmll_fetch(struct hmll *ctx, struct hmll_iobuf *dst, struct hmll_range range,  size_t iofile) NO_EXCEPT;
#ifdef __cplusplus
}
#endif
#endif // HMLL_H
