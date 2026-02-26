//
// Created by mfuntowicz on 12/2/25.
//
#include "hmll/hmll.h"

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include "hmll/unix/backend/mmap.h"
#elif defined(_WIN32)
#include "hmll/win32/backend/mmap.h"
#endif

struct hmll_iobuf hmll_get_buffer_for_range(struct hmll *ctx,
                                            const struct hmll_range range) {
  if (hmll_check(ctx->error))
    return (struct hmll_iobuf){0};

  const size_t size = hmll_range_size(range);
  const struct hmll_iobuf buf =
      hmll_get_buffer(ctx, size, HMLL_MEM_DEVICE);
  if (hmll_check(ctx->error))
    return (struct hmll_iobuf){0};

  return buf;
}
struct hmll_iobuf hmll_slice_buffer(const struct hmll_iobuf *src,
                                    const struct hmll_range slice) {
  if (slice.end - slice.start < src->size) {
    void *ptr = (unsigned char *)src->ptr + slice.start;
    return (struct hmll_iobuf){slice.end - slice.start, ptr, src->device};
  }

  return (struct hmll_iobuf){0};
}

struct hmll_error hmll_get_mmap_view(struct hmll *ctx, const int iofile,
                                     const struct hmll_range range,
                                     struct hmll_iobuf *out_view) {
  return hmll_mmap_get_view(ctx, iofile, range, out_view);
}
