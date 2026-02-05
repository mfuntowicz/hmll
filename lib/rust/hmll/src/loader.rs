//! Weight loader implementation for efficient model loading.

use crate::source::SourceHandle;
use crate::{Buffer, Device, Error, Range, Result, Source};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

/// Loader backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoaderKind {
    /// Automatically select the best backend.
    Auto,
    /// Use io_uring backend (Linux only, requires io_uring feature).
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    IoUring,
    /// Use mmap backend (cross-platform).
    Mmap,
}

impl LoaderKind {
    /// Convert to the underlying C enum value.
    #[inline(always)]
    pub(crate) const fn to_raw(self) -> hmll_sys::hmll_loader_kind {
        match self {
            LoaderKind::Auto => hmll_sys::HMLL_FETCHER_AUTO,
            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            LoaderKind::IoUring => hmll_sys::HMLL_FETCHER_IO_URING,
            LoaderKind::Mmap => hmll_sys::HMLL_FETCHER_MMAP,
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
    /// Raw sources passed to C layer
    sources: Vec<hmll_sys::hmll_source>,
    /// Arc handles for each source - keeps mmap alive while views exist
    source_handles: Vec<Arc<SourceHandle>>,
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

        // Clone Arc handles to keep sources alive while views exist
        let source_handles: Vec<Arc<SourceHandle>> = sources
            .iter()
            .map(|s| s.handle().clone())
            .collect();

        let mut sources_vec: Vec<hmll_sys::hmll_source> = sources.iter().map(|s| *s.as_raw()).collect();
        let mut context = Box::new(hmll_sys::hmll {
            fetcher: ptr::null_mut(),
            sources: ptr::null_mut(),
            num_sources: 0,
            error: hmll_sys::hmll_error {
                code: hmll_sys::HMLL_ERR_SUCCESS,
                sys_err: 0,
            },
        });

        unsafe {
            let err = hmll_sys::hmll_loader_init(
                context.as_mut(),
                sources_vec.as_mut_ptr(),
                sources_vec.len(),
                device.to_raw(),
                kind.to_raw(),
            );
            Error::check_hmll_error(err)?;

            // For mmap backend, close fds immediately - mmap is independent of fd.
            // NOTE: Could detect Auto resolving to mmap and close too, but left out for now.
            if kind == LoaderKind::Mmap {
                for handle in &source_handles {
                    let inner_ptr = &handle.inner as *const _ as *mut hmll_sys::hmll_source;
                    hmll_sys::hmll_source_close(inner_ptr);
                }
            }
        }

        Ok(Self {
            context,
            sources: sources_vec,
            source_handles,
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
    pub fn fetch<R: Into<Range>>(&mut self, range: R, file_index: usize) -> Result<Buffer> {
        let range = range.into();

        if file_index >= self.sources.len() {
            return Err(Error::InvalidRange);
        }

        if range.is_empty() {
            return Ok(Buffer::empty(self.device));
        }

        let iobuf = unsafe {
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
            hmll_sys::hmll_fetch(self.context.as_mut(), file_index as i32, &iobuf, range.start)
        };

        if res < 0 {
            let err = self.context.error;
            self.context.error = hmll_sys::hmll_error {
                code: hmll_sys::HMLL_ERR_SUCCESS,
                sys_err: 0,
            };
            return Err(Error::from_hmll_error(err));
        }

        Ok(unsafe { Buffer::from_raw_owned(iobuf) })
    }

    /// Fetch a zero-copy view of a range of bytes from a specific source file.
    ///
    /// This returns a `Buffer` that points directly into the mmap'd region
    /// without any memory allocation or copying. The buffer holds an Arc
    /// reference to the source, so it can safely **outlive** this `WeightLoader`.
    ///
    /// # Arguments
    ///
    /// * `range` - The byte range to get a view of (start..end)
    /// * `file_index` - Index of the source file
    ///
    /// # Returns
    ///
    /// A `Buffer` containing a view into the mmap'd data. The buffer does NOT
    /// own its memory and will NOT free it when dropped. Instead, it holds an
    /// Arc reference that keeps the source's mmap alive.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file index is out of bounds
    /// - The range is invalid
    /// - The device is not CPU (GPU requires copying)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hmll::{Source, WeightLoader, Device, LoaderKind};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let source = Source::open("model.safetensors")?;
    /// # let sources = [source];
    /// # let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)?;
    ///
    /// // Get a zero-copy view into the first 1MB
    /// let view = loader.fetch_view(0..1024 * 1024, 0)?;
    /// println!("Got view of {} bytes", view.len());
    /// // view does not own its memory - no allocation or free happens
    /// // view can outlive the loader due to Arc refcounting
    /// # Ok(())
    /// # }
    /// ```
    pub fn fetch_view<R: Into<Range>>(&mut self, range: R, file_index: usize) -> Result<Buffer> {
        let range = range.into();

        if file_index >= self.sources.len() {
            return Err(Error::InvalidRange);
        }

        if range.is_empty() {
            return Ok(Buffer::empty(self.device));
        }

        // Only CPU device supports views (GPU needs to copy to device memory)
        if self.device != Device::Cpu {
            return Err(Error::UnsupportedDevice);
        }

        // Clone the Arc to keep the source (and its mmap) alive
        let source_handle = self.source_handles[file_index].clone();

        // Get the mmap'd content pointer directly from the source
        let content_ptr = source_handle.inner.content;
        if content_ptr.is_null() {
            return Err(Error::MmapFailed);
        }

        // Calculate the view pointer
        let view_ptr = unsafe { (content_ptr as *mut u8).add(range.start) as *mut std::ffi::c_void };
        let view_size = range.len();

        // Create a view buffer - Arc keeps source (and mmap) alive
        Ok(unsafe { Buffer::from_source_view(view_ptr, view_size, self.device, source_handle) })
    }

    /// Get the device this loader is configured for.
    #[inline(always)]
    pub const fn device(&self) -> Device {
        self.device
    }

    /// Get the number of source files.
    #[inline(always)]
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Get information about a specific source file.
    #[inline]
    pub fn source_info(&self, index: usize) -> Option<SourceInfo> {
        self.sources.get(index).map(|s| SourceInfo { size: s.size })
    }
}

impl<'a> Drop for WeightLoader<'a> {
    fn drop(&mut self) {
        unsafe {
            hmll_sys::hmll_destroy(self.context.as_mut());
        }
    }
}

/// Information about a source file.
#[derive(Debug, Clone, Copy)]
pub struct SourceInfo {
    /// Size of the file in bytes
    pub size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content)
            .expect("Failed to write test content");
        file.flush().expect("Failed to flush");
        file
    }

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

    #[test]
    fn test_loader_creation() {
        let content = b"Test file content for loader creation test.";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        assert_eq!(loader.device(), Device::Cpu);
        assert_eq!(loader.num_sources(), 1);

        let info = loader.source_info(0).expect("Failed to get source info");
        assert_eq!(info.size, content.len());
    }

    #[test]
    fn test_fetch_full_file() {
        let content = b"This is the complete file content that we want to fetch entirely.";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let buffer = loader
            .fetch(0..content.len(), 0)
            .expect("Failed to fetch data");

        assert_eq!(buffer.len(), content.len());
        assert_eq!(buffer.device(), Device::Cpu);

        let slice = buffer.as_slice().expect("Failed to get slice");
        assert_eq!(slice, content);
    }

    #[test]
    fn test_fetch_partial_range() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let buffer = loader
            .fetch(10..20, 0)
            .expect("Failed to fetch partial data");

        assert_eq!(buffer.len(), 10);

        let slice = buffer.as_slice().expect("Failed to get slice");
        assert_eq!(slice, b"ABCDEFGHIJ");
    }

    #[test]
    fn test_fetch_empty_range() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let buffer = loader.fetch(5..5, 0).expect("Failed to fetch empty range");

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_fetch_invalid_file_index() {
        let content = b"Test content";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let result = loader.fetch(0..10, 99);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_sources() {
        let content1 = b"First file content here.";
        let content2 = b"Second file with different data.";
        let content3 = b"Third file completes the set.";

        let temp1 = create_test_file(content1);
        let temp2 = create_test_file(content2);
        let temp3 = create_test_file(content3);

        let source1 = Source::open(temp1.path()).expect("Failed to open source 1");
        let source2 = Source::open(temp2.path()).expect("Failed to open source 2");
        let source3 = Source::open(temp3.path()).expect("Failed to open source 3");

        let sources = [source1, source2, source3];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        assert_eq!(loader.num_sources(), 3);

        let buf1 = loader
            .fetch(0..content1.len(), 0)
            .expect("Failed to fetch file 1");
        let buf2 = loader
            .fetch(0..content2.len(), 1)
            .expect("Failed to fetch file 2");
        let buf3 = loader
            .fetch(0..content3.len(), 2)
            .expect("Failed to fetch file 3");

        assert_eq!(buf1.as_slice().unwrap(), content1);
        assert_eq!(buf2.as_slice().unwrap(), content2);
        assert_eq!(buf3.as_slice().unwrap(), content3);
    }

    #[test]
    fn test_large_file() {
        let size = 1024 * 1024; // 1 MiB
        let content: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let temp_file = create_test_file(&content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let buffer = loader
            .fetch(0..size, 0)
            .expect("Failed to fetch large file");

        assert_eq!(buffer.len(), size);

        let slice = buffer.as_slice().expect("Failed to get slice");
        assert_eq!(slice, content.as_slice());
    }

    #[test]
    fn test_source_info() {
        let content = b"Source info test content";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let info = loader.source_info(0);
        assert!(info.is_some());
        assert_eq!(info.unwrap().size, content.len());

        let info = loader.source_info(100);
        assert!(info.is_none());
    }

    #[test]
    fn test_buffer_to_vec() {
        let content = b"Convert me to a Vec!";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)
            .expect("Failed to create loader");

        let buffer = loader.fetch(0..content.len(), 0).expect("Failed to fetch");
        let vec = buffer.to_vec().expect("Failed to convert to vec");

        assert_eq!(vec, content);
    }

    #[test]
    fn test_mmap_loader_kind() {
        let content = b"Testing mmap loader backend explicitly.";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        let buffer = loader
            .fetch(0..content.len(), 0)
            .expect("Failed to fetch with mmap");

        assert_eq!(buffer.as_slice().unwrap(), content);
    }

    #[test]
    fn test_fetch_view_full_file() {
        let content = b"This is test content for fetch_view functionality.";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        let view = loader
            .fetch_view(0..content.len(), 0)
            .expect("Failed to get view");

        assert_eq!(view.len(), content.len());
        assert_eq!(view.device(), Device::Cpu);
        assert!(!view.is_owned()); // Views don't own their memory
        assert_eq!(view.as_slice().unwrap(), content);
    }

    #[test]
    fn test_fetch_view_partial_range() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        // Get a view of just the uppercase letters
        let view = loader
            .fetch_view(10..36, 0)
            .expect("Failed to get partial view");

        assert_eq!(view.len(), 26);
        assert!(!view.is_owned());
        assert_eq!(view.as_slice().unwrap(), b"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

    #[test]
    fn test_fetch_view_empty_range() {
        let content = b"Some content";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        let view = loader.fetch_view(5..5, 0).expect("Failed to get empty view");

        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
    }

    #[test]
    fn test_fetch_view_invalid_file_index() {
        let content = b"Test content";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        let result = loader.fetch_view(0..10, 99);
        assert!(result.is_err());
    }

    #[test]
    fn test_fetch_view_outlives_loader() {
        // Test that views can safely outlive the WeightLoader due to Arc refcounting
        let content = b"Data that should remain valid after loader is dropped.";
        let temp_file = create_test_file(content);

        let view = {
            let source = Source::open(temp_file.path()).expect("Failed to open source");
            let sources = [source];

            let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
                .expect("Failed to create mmap loader");

            loader.fetch_view(0..content.len(), 0).expect("Failed to get view")
            // loader AND sources are dropped here, but view holds Arc to source
        };

        // The view should still be valid after loader is dropped
        assert_eq!(view.len(), content.len());
        assert_eq!(view.as_slice().unwrap(), content);
    }

    #[test]
    fn test_fetch_view_multiple_views_same_source() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");
        let sources = [source];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        // Create multiple views from the same source
        let view1 = loader.fetch_view(0..10, 0).expect("Failed to get view 1");
        let view2 = loader.fetch_view(10..20, 0).expect("Failed to get view 2");
        let view3 = loader.fetch_view(20..30, 0).expect("Failed to get view 3");

        assert_eq!(view1.as_slice().unwrap(), b"0123456789");
        assert_eq!(view2.as_slice().unwrap(), b"ABCDEFGHIJ");
        assert_eq!(view3.as_slice().unwrap(), b"KLMNOPQRST");

        // Drop loader, views should still be valid
        drop(loader);

        assert_eq!(view1.as_slice().unwrap(), b"0123456789");
        assert_eq!(view2.as_slice().unwrap(), b"ABCDEFGHIJ");
        assert_eq!(view3.as_slice().unwrap(), b"KLMNOPQRST");
    }

    #[test]
    fn test_fetch_view_different_sources() {
        let content1 = b"First file content.";
        let content2 = b"Second file content.";

        let temp1 = create_test_file(content1);
        let temp2 = create_test_file(content2);

        let source1 = Source::open(temp1.path()).expect("Failed to open source 1");
        let source2 = Source::open(temp2.path()).expect("Failed to open source 2");

        let sources = [source1, source2];

        let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Mmap)
            .expect("Failed to create mmap loader");

        let view1 = loader.fetch_view(0..content1.len(), 0).expect("Failed to get view 1");
        let view2 = loader.fetch_view(0..content2.len(), 1).expect("Failed to get view 2");

        // Drop loader
        drop(loader);

        // Each view should keep its respective source alive
        assert_eq!(view1.as_slice().unwrap(), content1);
        assert_eq!(view2.as_slice().unwrap(), content2);
    }
}
