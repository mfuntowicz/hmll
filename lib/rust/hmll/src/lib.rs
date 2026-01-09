//! Safe, idiomatic Rust bindings to the hmll library.
//!
//! This crate provides a safe, high-level interface to the hmll C library for
//! high-performance loading of machine learning model files.
//!
//! # Features
//!
//! - **`io_uring`** (default): High-performance I/O using io_uring on Linux
//! - **`safetensors`**: Native support for safetensors format
//! - **`cuda`**: CUDA memory support for GPU operations
//!
//! # Example
//!
//! ```no_run
//! use hmll::{Source, WeightLoader, Device, LoaderKind};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open a model file
//! let source = Source::open("model.safetensors")?;
//! let sources = [source];
//!
//! // Create a weight loader
//! let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)?;
//!
//! // Fetch a range of bytes
//! let data = loader.fetch(0..1024, 0)?;
//! println!("Fetched {} bytes", data.len());
//! # Ok(())
//! # }
//! ```

mod error;
mod source;
mod loader;
mod device;
mod buffer;

pub use error::{Error, Result};
pub use source::Source;
pub use loader::{WeightLoader, LoaderKind};
pub use device::Device;
pub use buffer::{Buffer, Range};

// Re-export common types
pub use hmll_sys::{
    HMLL_ERR_SUCCESS,
    HMLL_DEVICE_CPU,
    HMLL_DEVICE_CUDA,
};

#[cfg(feature = "safetensors")]
pub use hmll_sys::{
    hmll_dtype as DType,
    HMLL_DTYPE_FLOAT32,
    HMLL_DTYPE_FLOAT16,
    HMLL_DTYPE_BFLOAT16,
};
