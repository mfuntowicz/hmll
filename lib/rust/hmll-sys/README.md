# hmll-sys

Low-level FFI bindings to the [hmll](https://github.com/huggingface/hmll) library for high-performance ML model loading.

## Features

- `vendored` (default) - Build hmll from source
- `io_uring` (default) - Enable io_uring backend (Linux only)
- `safetensors` - Enable safetensors format support
- `cuda` - Enable CUDA support

## System Library

To use a system-installed libhmll instead of building from source:

```toml
[dependencies]
hmll-sys = { version = "0.1", default-features = false, features = ["io_uring"] }
```

This requires libhmll to be installed and discoverable via pkg-config.

## License

MIT OR Apache-2.0
