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
#ifdef __HMLL_EXPORTS__
#define HMLL_EXTERN __declspec(dllexport)
#else
#define HMLL_EXTERN __declspec(dllimport)
#endif
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

#if defined(_MSC_VER)
#define __builtin_unreachable() __assume(0)
#endif

#include "loader.h"
#include "memory.h"
#include "types.h"

// Platform-specific file handling - loader.h already handles platform detection
#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include "unix/file.h"
#elif defined(_WIN32)
#include "win32/file.h"
#endif

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

/** Error handling stubs **/
#ifdef __cplusplus
// C++ uses brace initialization (no parenthesized type)
#define HMLL_RES(...) hmll_error{ __VA_ARGS__ }
#else
// C uses compound literals
#define HMLL_RES(...) (struct hmll_error){ __VA_ARGS__ }
#endif
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

static inline struct hmll_device hmll_device_cpu(void) NO_EXCEPT
{
    struct hmll_device d = { HMLL_DEVICE_CPU, 0 };
    return d;
}

static inline struct hmll_device hmll_device_cuda(unsigned char idx) NO_EXCEPT
{
    struct hmll_device d = { HMLL_DEVICE_CUDA, idx };
    return d;
}

static inline unsigned char hmll_device_is_cpu(struct hmll_device device) NO_EXCEPT
{
    return device.kind == HMLL_DEVICE_CPU;
}

static inline unsigned char hmll_device_is_cuda(struct hmll_device device) NO_EXCEPT
{
    return device.kind == HMLL_DEVICE_CUDA;
}

static inline unsigned char hmll_device_eq(struct hmll_device a, struct hmll_device b) NO_EXCEPT
{
    return a.kind == b.kind && a.idx == b.idx;
}

HMLL_EXTERN unsigned char hmll_error_is_os_error(struct hmll_error err) NO_EXCEPT;
HMLL_EXTERN unsigned char hmll_error_is_lib_error(struct hmll_error err) NO_EXCEPT;
HMLL_EXTERN const char *hmll_strerr(struct hmll_error err) NO_EXCEPT;
HMLL_EXTERN void hmll_destroy(struct hmll *ctx) NO_EXCEPT;

/** Sources handling stubs **/
HMLL_EXTERN struct hmll_error hmll_source_open(const char *path, struct hmll_source *src) NO_EXCEPT;
HMLL_EXTERN void hmll_source_close(struct hmll_source *src) NO_EXCEPT;
HMLL_EXTERN void hmll_source_cleanup(struct hmll_source *src) NO_EXCEPT;
HMLL_EXTERN void hmll_source_free(struct hmll_source *src) NO_EXCEPT;

/** Memory handling stubs **/
static inline size_t hmll_range_size(const struct hmll_range range) NO_EXCEPT { return range.end - range.start; }
HMLL_EXTERN void *hmll_alloc(size_t size, struct hmll_device device, int flags) NO_EXCEPT;
HMLL_EXTERN void hmll_free_buffer(struct hmll_iobuf *buffer) NO_EXCEPT;
HMLL_EXTERN struct hmll_iobuf hmll_get_buffer(struct hmll *ctx, size_t size, int flags) NO_EXCEPT;
HMLL_EXTERN struct hmll_iobuf hmll_get_buffer_for_range(struct hmll *ctx, struct hmll_range range) NO_EXCEPT;
HMLL_EXTERN struct hmll_iobuf hmll_slice_buffer(const struct hmll_iobuf *src, struct hmll_range slice) NO_EXCEPT;

/** Fetching stubs **/
HMLL_EXTERN struct hmll_error hmll_loader_init(
    struct hmll *ctx, struct hmll_source *srcs, size_t n, struct hmll_device device, enum hmll_loader_kind kind) NO_EXCEPT;
HMLL_EXTERN ssize_t hmll_fetch(struct hmll *ctx, int iofile, const struct hmll_iobuf *dst, size_t offset) NO_EXCEPT;
HMLL_EXTERN ssize_t hmll_fetchv(struct hmll *ctx, int iofile, const struct hmll_iobuf *dsts, const size_t *offsets, size_t n) NO_EXCEPT;

/** Zero-copy mmap view (mmap backend, CPU device only) **/
HMLL_EXTERN struct hmll_error hmll_get_mmap_view(struct hmll *ctx, int iofile, struct hmll_range range, struct hmll_iobuf *out_view) NO_EXCEPT;

/** Tensors manipulation stubs - enabled if a higher-level tensor format is enabled **/
#ifdef __HMLL_TENSORS_ENABLED__
HMLL_EXTERN uint8_t hmll_nbits(enum hmll_dtype dtype) NO_EXCEPT;
HMLL_EXTERN size_t hmll_numel(const struct hmll_tensor_specs *specs) NO_EXCEPT;
HMLL_EXTERN void hmll_free_registry(struct hmll_registry *reg) NO_EXCEPT;
HMLL_EXTERN unsigned char hmll_contains(const struct hmll *ctx, const struct hmll_registry *reg, const char *name) NO_EXCEPT;
HMLL_EXTERN int hmll_find_by_name(const struct hmll *ctx, const struct hmll_registry *reg, const char *name) NO_EXCEPT;
HMLL_EXTERN struct hmll_lookup_result hmll_lookup_tensor(const struct hmll *ctx, const struct hmll_registry *registry, const char *name) NO_EXCEPT;
HMLL_EXTERN ssize_t hmll_fetch_tensor(struct hmll *ctx, const struct hmll_registry *registry, struct hmll_iobuf *dst, const char *name) NO_EXCEPT;
#endif

/** Safetensors format support **/
#ifdef __HMLL_SAFETENSORS_ENABLED__
HMLL_EXTERN size_t hmll_safetensors_populate_registry(struct hmll *ctx, struct hmll_registry *reg, struct hmll_source source, size_t fid, size_t offset) NO_EXCEPT;
HMLL_EXTERN size_t hmll_safetensors_index(struct hmll *ctx, struct hmll_registry *reg, struct hmll_source source) NO_EXCEPT;
#endif
#ifdef __cplusplus
}
#endif
#endif // HMLL_H
