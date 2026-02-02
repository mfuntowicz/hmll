# hmll Fil-C Test Suite

Functional tests for the hmll C library compiled with [Fil-C](https://fil-c.org/), a memory-safe C compiler.

## Quick Start

```bash
# Build and run all tests
make test

# Or build only
make
```

## Tests

| Test | Description |
|------|-------------|
| `test_source` | Source file open/close operations |
| `test_memory` | Memory allocation and buffer management |
| `test_loader` | Loader initialization and data fetching |
| `test_fetchv` | Vectorized fetch with multiple ranges |
| `test_context` | Context cloning and management |
| `test_mmap_view` | Zero-copy mmap view functionality |

## Prerequisites

1. **Fil-C compiler** at `/home/ubuntu/filc-0.677-linux-x86_64`
   ```bash
   # Download and install
   cd $HOME
   curl -LO https://github.com/pizlonator/fil-c/releases/download/v0.677/filc-0.677-linux-x86_64.tar.xz
   tar -xf filc-0.677-linux-x86_64.tar.xz
   cd filc-0.677-linux-x86_64
   ./setup.sh
   ```

2. **hmll built with Fil-C** at `/home/ubuntu/hmll/build-filc`
   ```bash
   cd /home/ubuntu/hmll
   mkdir -p build-filc && cd build-filc
   cmake .. \
       -DCMAKE_C_COMPILER=$HOME/filc-0.677-linux-x86_64/build/bin/clang \
       -DCMAKE_BUILD_TYPE=Release \
       -DHMLL_ENABLE_IO_URING=OFF \
       -DHMLL_ENABLE_SAFETENSORS=OFF
   make
   ```

## Make Targets

```bash
make              # Build all tests
make test         # Build and run all tests
make clean        # Remove built files
make help         # Show help

# Individual tests
make test-source  # Run source tests
make test-memory  # Run memory tests
make test-loader  # Run loader tests
make test-fetchv  # Run fetchv tests
make test-context # Run context tests
make test-mmap    # Run mmap view tests
```

## Custom Fil-C Path

```bash
FILC_ROOT=/custom/path/to/fil-c make test
```

## What Fil-C Provides

When running these tests with Fil-C compiled binaries, any memory safety violations in hmll will be caught at runtime:

- Use-after-free
- Buffer overflows (stack)
- Double-free
- Null pointer dereferences
- Type confusion

If a test crashes with a `filc panic` message, it indicates a memory safety bug was detected.

## References

- [Fil-C Website](https://fil-c.org/)
- [Fil-C GitHub](https://github.com/pizlonator/fil-c)
- [hmll Repository](https://github.com/your-org/hmll)
