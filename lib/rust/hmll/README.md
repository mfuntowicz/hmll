# hmll

Safe, idiomatic Rust bindings to the [hmll](https://github.com/huggingface/hmll) library for high-performance ML model loading.

## Features

- `vendored` (default) - Build hmll from source
- `io_uring` (default) - Enable io_uring backend (Linux only)
- `safetensors` - Enable safetensors format support
- `cuda` - Enable CUDA support

## Usage

```toml
[dependencies]
hmll = "0.1"
```

## System Library

To use a system-installed libhmll instead of building from source:

```toml
[dependencies]
hmll = { version = "0.1", default-features = false, features = ["io_uring"] }
```

## License

MIT OR Apache-2.0
