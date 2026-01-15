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

#ifdef DEBUG
#define HMLL_STATIC
#else
#define HMLL_STATIC static
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

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

/** Error handling stubs **/
#define HMLL_RES(...) (struct hmll_error){ __VA_ARGS__ }
#define HMLL_OK  HMLL_RES(.code = HMLL_ERR_SUCCESS, .sys_err = 0)
#define HMLL_ERR(c) HMLL_RES(.code = (c), .sys_err = 0)
#define HMLL_SYS_ERR(e) HMLL_RES(.code = HMLL_ERR_SYSTEM, .sys_err = (e))

static inline unsigned char hmll_check(const struct hmll_error res) NO_EXCEPT
{
    return res.code != HMLL_ERR_SUCCESS || res.sys_err != HMLL_ERR_SUCCESS;
}

static inline unsigned char hmll_success(const struct hmll_error error) NO_EXCEPT
{
    return (int)error.code == HMLL_ERR_SUCCESS && error.sys_err == HMLL_ERR_SUCCESS;
}

HMLL_EXTERN unsigned char hmll_error_is_os_error(struct hmll_error err) NO_EXCEPT;
HMLL_EXTERN unsigned char hmll_error_is_lib_error(struct hmll_error err) NO_EXCEPT;
HMLL_EXTERN const char *hmll_strerr(struct hmll_error err) NO_EXCEPT;
HMLL_EXTERN void hmll_destroy(struct hmll *ctx) NO_EXCEPT;

/** Context cloning **/
HMLL_EXTERN struct hmll_error hmll_clone_context(struct hmll *dst, const struct hmll *src) NO_EXCEPT;

/** Sources handling stubs **/
HMLL_EXTERN struct hmll_error hmll_source_open(const char *path, struct hmll_source *src) NO_EXCEPT;
HMLL_EXTERN void hmll_source_close(const struct hmll_source *src) NO_EXCEPT;

/** Memory handling stubs **/
static inline size_t hmll_range_size(const struct hmll_range range) NO_EXCEPT { return range.end - range.start; }
HMLL_EXTERN void *hmll_alloc(size_t size, enum hmll_device device, int flags) NO_EXCEPT;
HMLL_EXTERN void hmll_free_buffer(struct hmll_iobuf *buffer) NO_EXCEPT;
HMLL_EXTERN struct hmll_iobuf hmll_get_buffer(struct hmll *ctx, enum hmll_device device, size_t size, int flags) NO_EXCEPT;
HMLL_EXTERN struct hmll_iobuf hmll_get_buffer_for_range(struct hmll *ctx, enum hmll_device device, struct hmll_range range) NO_EXCEPT;

/** Fetching stubs **/
HMLL_EXTERN struct hmll_error hmll_loader_init(
    struct hmll *ctx, const struct hmll_source *srcs, size_t n, enum hmll_device device, enum hmll_loader_kind kind) NO_EXCEPT;
HMLL_EXTERN ssize_t hmll_fetch(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, struct hmll_range range) NO_EXCEPT;
HMLL_EXTERN ssize_t hmll_fetchv(struct hmll *ctx, int iofile, const struct hmll_iobuf *dsts, const struct hmll_range *ranges, size_t n) NO_EXCEPT;

/** Tensors manipulation stubs - enabled if a higher-level tensor format is enabled **/
#ifdef __HMLL_TENSORS_ENABLED__
HMLL_EXTERN uint8_t hmll_nbits(enum hmll_dtype dtype) NO_EXCEPT;
HMLL_EXTERN size_t hmll_numel(const struct hmll_tensor_specs *specs) NO_EXCEPT;
HMLL_EXTERN unsigned char hmll_contains(const struct hmll *ctx, const struct hmll_registry *reg, const char *name) NO_EXCEPT;
HMLL_EXTERN int hmll_find_by_name(const struct hmll *ctx, const struct hmll_registry *reg, const char *name) NO_EXCEPT;
HMLL_EXTERN struct hmll_lookup_result hmll_lookup_tensor(const struct hmll *ctx, const struct hmll_registry *registry, const char *name) NO_EXCEPT;
HMLL_EXTERN ssize_t hmll_fetch_tensor(struct hmll *ctx, const struct hmll_registry *registry, struct hmll_iobuf *dst, const char *name) NO_EXCEPT;
#endif

/** Safetensors format support **/
#ifdef __HMLL_SAFETENSORS_ENABLED__
HMLL_EXTERN struct hmll_error hmll_safetensors_populate_registry(
    struct hmll *ctx, struct hmll_registry *reg, struct hmll_source source, size_t fid, size_t offset) NO_EXCEPT;
HMLL_EXTERN struct hmll_error hmll_safetensors_index(
    struct hmll *ctx, struct hmll_registry *reg, struct hmll_source source) NO_EXCEPT;
#endif
#ifdef __cplusplus
}
#endif
#endif // HMLL_H
