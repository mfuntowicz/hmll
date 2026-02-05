//! High-performance, file-format agnostic byte loading for Rust.
//!
//! hmll provides fast I/O primitives for loading bytes from files. It is designed
//! to be embedded into file-format libraries (like safetensors) rather than used
//! directly by end users.
//!
//! # Features
//!
//! - **`io_uring`** (default, Linux only): High-performance async I/O
//! - **`cuda`**: Direct loading to GPU memory
//!
//! # Core API
//!
//! - [`Source`]: File handle with size information
//! - [`WeightLoader`]: Fast byte loading with multiple backend support
//! - [`Buffer`]: Loaded data (owned or zero-copy mmap view)
//!
//! # Multi-file Support
//!
//! hmll supports loading from multiple files simultaneously:
//!
//! ```no_run
//! # use hmll::{Source, WeightLoader, Device, LoaderKind};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sources = [
//!     Source::open("shard-00001.bin")?,
//!     Source::open("shard-00002.bin")?,
//! ];
//!
//! let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)?;
//!
//! // Fetch bytes from specific file by index
//! let data = loader.fetch(0..1024, 0)?;  // from shard 1
//! let data = loader.fetch(0..1024, 1)?;  // from shard 2
//! # Ok(())
//! # }
//! ```

mod buffer;
mod device;
mod error;
mod loader;
mod source;

pub use buffer::{Buffer, Range};
pub use device::Device;
pub use error::{Error, Result};
pub use loader::{LoaderKind, WeightLoader};
pub use source::Source;
