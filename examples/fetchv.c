#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hmll/hmll.h>

#ifdef _WIN32
#include <windows.h>
typedef LARGE_INTEGER timespec_t;

static void tick(timespec_t *ts) {
    QueryPerformanceCounter(ts);
}

static double time_diff_ns(const timespec_t *start, const timespec_t *end) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return (double)(end->QuadPart - start->QuadPart) * 1e9 / frequency.QuadPart;
}
#else
#include <time.h>
typedef struct timespec timespec_t;

static void tick(timespec_t *ts) {
    clock_gettime(CLOCK_MONOTONIC, ts);
}

static double time_diff_ns(const timespec_t *start, const timespec_t *end) {
    return (end->tv_sec - start->tv_sec) * 1e9 + (end->tv_nsec - start->tv_nsec);
}
#endif

#if defined(__HMLL_CUDA_ENABLED__)
#include <cuda_runtime.h>
#endif

static int path_ends_with(const char *path, const char *suffix) {
    const size_t plen = strlen(path);
    const size_t slen = strlen(suffix);
    return plen >= slen && strcmp(path + plen - slen, suffix) == 0;
}

static void get_dir_prefix(const char *path, char *dir, size_t dir_size) {
    strncpy(dir, path, dir_size - 1);
    dir[dir_size - 1] = '\0';
    char *last_slash = strrchr(dir, '/');
    if (last_slash) {
        *(last_slash + 1) = '\0';
    } else {
        dir[0] = '\0';
    }
}

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: hmll_examples_fetchv <path/to/file.safetensors|path/to/model.safetensors.index.json>\n");
        return 1;
    }

    const char *path = argv[1];
    const int is_sharded = path_ends_with(path, ".index.json");

    hmll_t ctx = {0};
    hmll_registry_t registry = {0};
    hmll_source_t *sources = NULL;
    size_t num_files = 1;

    if (is_sharded) {
        hmll_source_t index_src = {0};
        if (hmll_check(hmll_source_open(path, &index_src))) {
            fprintf(stderr, "Failed to open index file: %s\n", hmll_strerr(ctx.error));
            return 1;
        }

        num_files = hmll_safetensors_index(&ctx, &registry, index_src);
        hmll_source_close(&index_src);

        if (num_files == 0) {
            fprintf(stderr, "Failed to parse index: %s\n", hmll_strerr(ctx.error));
            return 2;
        }

        char dir[4096];
        get_dir_prefix(path, dir, sizeof(dir));

        sources = calloc(num_files, sizeof(hmll_source_t));
        if (!sources) {
            fprintf(stderr, "Allocation failed\n");
            hmll_free_registry(&registry);
            return 2;
        }

        for (size_t i = 0; i < num_files; ++i) {
            char shard_path[4096 + 64];
            snprintf(shard_path, sizeof(shard_path), "%smodel-%05zu-of-%05zu.safetensors",
                     dir, i + 1, num_files);
            if (hmll_check(hmll_source_open(shard_path, &sources[i]))) {
                fprintf(stderr, "Failed to open shard %zu (%s): %s\n",
                        i + 1, shard_path, hmll_strerr(ctx.error));
                for (size_t j = 0; j < i; ++j) hmll_source_close(&sources[j]);
                free(sources);
                hmll_free_registry(&registry);
                return 2;
            }
        }

        size_t offset = 0;
        for (size_t i = 0; i < num_files; ++i) {
            const size_t n = hmll_safetensors_populate_registry(&ctx, &registry, sources[i], i, offset);
            if (n == 0) {
                fprintf(stderr, "Failed to populate registry from shard %zu: %s\n",
                        i + 1, hmll_strerr(ctx.error));
                for (size_t j = 0; j < num_files; ++j) hmll_source_close(&sources[j]);
                free(sources);
                hmll_free_registry(&registry);
                return 2;
            }
            offset += n;
        }
    } else {
        sources = calloc(1, sizeof(hmll_source_t));
        if (!sources) {
            fprintf(stderr, "Allocation failed\n");
            return 2;
        }

        if (hmll_check(hmll_source_open(path, &sources[0]))) {
            fprintf(stderr, "Failed to open file: %s\n", hmll_strerr(ctx.error));
            free(sources);
            return 1;
        }

        if (hmll_safetensors_populate_registry(&ctx, &registry, sources[0], 0, 0) == 0) {
            fprintf(stderr, "Failed to populate registry: %s\n", hmll_strerr(ctx.error));
            hmll_source_close(&sources[0]);
            free(sources);
            return 2;
        }
    }

    printf("Registry: %zu tensor(s) across %zu file(s)\n", registry.num_tensors, num_files);

#if defined(__HMLL_CUDA_ENABLED__)
    const struct hmll_device device = hmll_device_cuda(0);
#else
    const struct hmll_device device = hmll_device_cpu();
#endif

    if (hmll_check(hmll_loader_init(&ctx, sources, num_files, device, HMLL_FETCHER_IO_URING))) {
        fprintf(stderr, "Failed to init loader: %s\n", hmll_strerr(ctx.error));
        for (size_t i = 0; i < num_files; ++i) hmll_source_close(&sources[i]);
        free(sources);
        hmll_free_registry(&registry);
        return 3;
    }

    timespec_t total_start, total_end;
    tick(&total_start);

    size_t total_bytes = 0;
    size_t total_errors = 0;

    /*
     * Core fetchv strategy: group all tensors that belong to the same shard and
     * submit them in a single hmll_fetchv call.  This lets the io_uring backend
     * pipeline all reads for a shard in one submission burst instead of issuing
     * individual SQEs one tensor at a time.
     */
    for (size_t fid = 0; fid < num_files; ++fid) {

        /* --- first pass: count tensors for this shard --- */
        size_t shard_n = 0;
        for (size_t t = 0; t < registry.num_tensors; ++t) {
            if (registry.indexes[t] == (unsigned short)fid)
                ++shard_n;
        }
        if (shard_n == 0) continue;

        /* --- allocate per-shard arrays --- */
        hmll_iobuf_t *dsts    = calloc(shard_n, sizeof(hmll_iobuf_t));
        size_t       *offsets = calloc(shard_n, sizeof(size_t));
        size_t       *tids    = calloc(shard_n, sizeof(size_t));  /* original tensor indices */

        if (!dsts || !offsets || !tids) {
            fprintf(stderr, "Shard %zu: allocation failed\n", fid);
            free(dsts); free(offsets); free(tids);
            total_errors += shard_n;
            continue;
        }

        /* --- second pass: build buffers and offsets arrays --- */
        size_t k = 0;
        size_t shard_bytes = 0;
        int alloc_ok = 1;

        for (size_t t = 0; t < registry.num_tensors && alloc_ok; ++t) {
            if (registry.indexes[t] != (unsigned short)fid) continue;

            const struct hmll_tensor_specs *specs = &registry.tensors[t];
            const hmll_range_t range = {specs->start, specs->end};

            dsts[k] = hmll_get_buffer_for_range(&ctx, range);
            if (!hmll_success(ctx.error)) {
                fprintf(stderr, "Shard %zu: buffer alloc failed for '%s': %s\n",
                        fid, registry.names[t], hmll_strerr(ctx.error));
                ctx.error = HMLL_OK;
                /* free already-allocated buffers in this shard */
                for (size_t j = 0; j < k; ++j) hmll_free_buffer(&dsts[j]);
                alloc_ok = 0;
                total_errors += shard_n;
                break;
            }

            offsets[k] = specs->start;
            tids[k]    = t;
            shard_bytes += specs->end - specs->start;
            ++k;
        }

        if (!alloc_ok) {
            free(dsts); free(offsets); free(tids);
            continue;
        }

        /* --- single fetchv call for the whole shard --- */
        printf("\n[Shard %zu/%zu] fetching %zu tensor(s) (%.2f MB) in one call\n",
               fid + 1, num_files, shard_n,
               (double)shard_bytes / (1024.0 * 1024.0));

        timespec_t shard_start, shard_end;
        tick(&shard_start);

        const ssize_t fetched = hmll_fetchv(&ctx, (int)fid, dsts, offsets, shard_n);

        tick(&shard_end);

        if (fetched < 0 || !hmll_success(ctx.error)) {
            fprintf(stderr, "Shard %zu: fetchv failed: %s\n",
                    fid, hmll_strerr(ctx.error));
            ctx.error = HMLL_OK;
            total_errors += shard_n;
        } else {
            const double shard_elapsed_s = time_diff_ns(&shard_start, &shard_end) / 1e9;
            const double shard_mb        = (double)shard_bytes / (1024.0 * 1024.0);

            total_bytes += (size_t)fetched;

            /* per-tensor report */
            for (size_t j = 0; j < shard_n; ++j) {
                const size_t t  = tids[j];
                const double mb = (double)dsts[j].size / (1024.0 * 1024.0);
                printf("  [%zu/%zu] %-60s  %8.2f MB\n",
                       t + 1, registry.num_tensors, registry.names[t], mb);
            }

            printf("Shard %zu throughput: %.2f MB/s  (%.3f s)\n",
                   fid + 1, shard_mb / shard_elapsed_s, shard_elapsed_s);
        }

        for (size_t j = 0; j < shard_n; ++j) hmll_free_buffer(&dsts[j]);
        free(dsts); free(offsets); free(tids);
    }

    tick(&total_end);

    const double total_elapsed_s = time_diff_ns(&total_start, &total_end) / 1e9;
    const double total_mb        = (double)total_bytes / (1024.0 * 1024.0);

    printf("\n=== Summary ===\n");
    printf("Tensors fetched : %zu (errors: %zu)\n",
           registry.num_tensors - total_errors, total_errors);
    printf("Total data      : %.2f MB\n", total_mb);
    printf("Total time      : %.3f s\n", total_elapsed_s);
    printf("Throughput      : %.2f MB/s\n", total_mb / total_elapsed_s);

    hmll_free_registry(&registry);
    for (size_t i = 0; i < num_files; ++i) hmll_source_close(&sources[i]);
    free(sources);

    return total_errors > 0 ? 5 : 0;
}
