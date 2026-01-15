//! Weight loader implementation for efficient model loading.

use crate::{Buffer, Device, Error, Range, Result, Source};
use std::marker::PhantomData;
use std::ptr;

/// Loader backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoaderKind {
    /// Automatically select the best backend.
    Auto,
    /// Use io_uring backend (Linux only).
    #[cfg(target_os = "linux")]
    IoUring,
}

impl LoaderKind {
    /// Convert to the underlying C enum value.
    #[inline(always)]
    pub(crate) const fn to_raw(self) -> hmll_sys::hmll_loader_kind {
        match self {
            LoaderKind::Auto => hmll_sys::HMLL_FETCHER_AUTO,
            #[cfg(target_os = "linux")]
            LoaderKind::IoUring => hmll_sys::HMLL_FETCHER_IO_URING,
        }
    }
}

impl Default for LoaderKind {
    /// Default loader kind is Auto.
    ///
    /// Hot path - inline always for zero-cost default.
    #[inline(always)]
    fn default() -> Self {
        LoaderKind::Auto
    }
}

/// A high-performance weight loader for ML models.
///
/// `WeightLoader` encapsulates the hmll context, loader, and device configuration,
/// providing a safe interface for fetching weight data from model files.
///
/// # Example
///
/// ```no_run
/// use hmll::{Source, WeightLoader, Device, LoaderKind};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Open source files
/// let source1 = Source::open("model-00001-of-00003.safetensors")?;
/// let source2 = Source::open("model-00002-of-00003.safetensors")?;
/// let source3 = Source::open("model-00003-of-00003.safetensors")?;
/// let sources = [source1, source2, source3];
///
/// // Create a loader
/// let mut loader = WeightLoader::new(
///     &sources,
///     Device::Cpu,
///     LoaderKind::Auto
/// )?;
///
/// // Fetch data from the first file
/// let data = loader.fetch(0..1024, 0)?;
/// println!("Fetched {} bytes", data.len());
/// # Ok(())
/// # }
/// ```
pub struct WeightLoader<'a> {
    context: Box<hmll_sys::hmll>,
    sources: Vec<hmll_sys::hmll_source>,
    device: Device,
    _marker: PhantomData<&'a ()>,
}

impl<'a> WeightLoader<'a> {
    /// Create a new weight loader.
    ///
    /// # Arguments
    ///
    /// * `sources` - Slice of source files to load from
    /// * `device` - Target device (CPU or CUDA)
    /// * `kind` - Loader backend kind
    ///
    /// # Errors
    ///
    /// Returns an error if the loader initialization fails.
    pub fn new(sources: &'a [Source], device: Device, kind: LoaderKind) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::InvalidRange);
        }

        let sources_vec: Vec<hmll_sys::hmll_source> = sources
            .iter()
            .map(|s| *s.as_raw())
            .collect();

        let mut context = Box::new(hmll_sys::hmll {
            fetcher: ptr::null_mut(),
            sources: ptr::null(),
            num_sources: 0,
            error: hmll_sys::hmll_error {
                code: hmll_sys::HMLL_ERR_SUCCESS,
                sys_err: 0,
            },
        });

        unsafe {
            let err = hmll_sys::hmll_loader_init(
                context.as_mut(),
                sources_vec.as_ptr(),
                sources_vec.len(),
                device.to_raw(),
                kind.to_raw(),
            );
            Error::check_hmll_error(err)?;
        }

        Ok(Self {
            context,
            sources: sources_vec,
            device,
            _marker: PhantomData,
        })
    }

    /// Fetch a range of bytes from a specific source file.
    ///
    /// # Arguments
    ///
    /// * `range` - The byte range to fetch (start..end)
    /// * `file_index` - Index of the source file to fetch from
    ///
    /// # Returns
    ///
    /// A `Buffer` containing the fetched data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file index is out of bounds
    /// - The range is invalid
    /// - The fetch operation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hmll::{Source, WeightLoader, Device, LoaderKind};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let source = Source::open("model.safetensors")?;
    /// # let sources = [source];
    /// # let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)?;
    ///
    /// // Fetch first 1MB from the first file
    /// let data = loader.fetch(0..1024 * 1024, 0)?;
    /// println!("Fetched {} bytes", data.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn fetch<R: Into<Range>>(&mut self, range: R, file_index: i32) -> Result<Buffer> {
        let range = range.into();

        if file_index >= self.sources.len() as i32 {
            return Err(Error::InvalidRange);
        }

        if range.is_empty() {
            return Ok(unsafe { Buffer::from_raw_parts(ptr::null_mut(), 0, self.device, false) });
        }

        let mut iobuf = unsafe {
            hmll_sys::hmll_get_buffer_for_range(
                self.context.as_mut(),
                self.device.to_raw(),
                range.to_raw(),
            )
        };

        if iobuf.ptr.is_null() {
            return Err(Error::AllocationFailed);
        }

        let res = unsafe {
            hmll_sys::hmll_fetch(
                self.context.as_mut(),
                file_index,
                &mut iobuf,
                range.to_raw(),
            )
        };

        if res < 0 {
            let err = self.context.error;
            self.context.error = hmll_sys::hmll_error {
                code: hmll_sys::HMLL_ERR_SUCCESS,
                sys_err: 0,
            };
            return Err(Error::from_hmll_error(err));
        }

        Ok(unsafe {
            Buffer::from_raw_parts(
                iobuf.ptr as *mut u8,
                iobuf.size,
                self.device,
                false,
            )
        })
    }

    /// Get the device this loader is configured for.
    #[inline(always)]
    pub const fn device(&self) -> Device { self.device }

    /// Get the number of source files.
    #[inline(always)]
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Get information about a specific source file.
    #[inline]
    pub fn source_info(&self, index: usize) -> Option<SourceInfo> {
        if index < self.sources.len() {
            Some(SourceInfo {
                size: self.sources[index].size,
                #[cfg(target_family = "unix")]
                fd: self.sources[index].fd,
            })
        } else {
            None
        }
    }
}

impl<'a> Drop for WeightLoader<'a> {
    fn drop(&mut self) {
        unsafe {
            hmll_sys::hmll_destroy(self.context.as_mut());
        }
    }
}

// WeightLoader is Send but not Sync (mutable operations)
unsafe impl<'a> Send for WeightLoader<'a> {}

/// Information about a source file.
#[derive(Debug, Clone, Copy)]
pub struct SourceInfo {
    /// Size of the file in bytes
    pub size: usize,
    /// File descriptor (Unix only)
    #[cfg(target_family = "unix")]
    pub fd: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_sources() {
        let result = WeightLoader::new(&[], Device::Cpu, LoaderKind::Auto);
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_kind_default() {
        assert_eq!(LoaderKind::default(), LoaderKind::Auto);
    }

    #[test]
    fn test_device_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }
}
