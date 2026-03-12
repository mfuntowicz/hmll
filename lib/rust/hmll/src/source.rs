//! Source file handling for hmll.

use crate::error::{Error, Result};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::Arc;

/// Internal handle to the C source, enabling Arc-based lifetime management.
///
/// When all Arc references to this handle are dropped, `hmll_source_cleanup()`
/// is called to close the fd and unmap the content. Rust manages the struct memory.
pub(crate) struct SourceHandle {
    pub(crate) inner: hmll_sys::hmll_source,
}

impl Drop for SourceHandle {
    fn drop(&mut self) {
        // hmll_source_cleanup closes fd and unmaps content
        unsafe {
            hmll_sys::hmll_source_cleanup(&mut self.inner);
        }
    }
}

// SourceHandle is Send and Sync - the underlying mmap is thread-safe for reads
unsafe impl Send for SourceHandle {}
unsafe impl Sync for SourceHandle {}

/// A source file for loading model weights.
///
/// This wraps mmap'd content, using Arc-based reference counting to ensure
/// the mmap'd memory stays alive as long as any views exist.
#[derive(Clone)]
pub struct Source {
    pub(crate) handle: Arc<SourceHandle>,
    path: PathBuf,
}

impl Source {
    /// Open a source file from a path.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hmll::Source;
    ///
    /// let source = Source::open("model.safetensors")?;
    /// println!("Opened file with size: {} bytes", source.size());
    /// # Ok::<(), hmll::Error>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let path_str = path_ref
            .to_str()
            .ok_or_else(|| Error::FileNotFound("Invalid UTF-8 in path".to_string()))?;

        let c_path = CString::new(path_str)
            .map_err(|_| Error::FileNotFound("Path contains null byte".to_string()))?;

        let mut source = hmll_sys::hmll_source {
            fd: -1,
            d_fd: -1,
            size: 0,
            content: null_mut(),
        };

        unsafe {
            let err = hmll_sys::hmll_source_open(c_path.as_ptr(), &mut source);
            Error::check_hmll_error(err)?;
        }

        Ok(Self {
            handle: Arc::new(SourceHandle { inner: source }),
            path: path_ref.to_path_buf(),
        })
    }

    /// Get the size of the source file in bytes.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.handle.inner.size
    }

    /// Get the path of the source file.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get a reference to the underlying hmll_source.
    ///
    /// This is useful for advanced use cases that require direct FFI access.
    #[inline(always)]
    pub fn as_raw(&self) -> &hmll_sys::hmll_source {
        &self.handle.inner
    }

    /// Get the Arc handle for this source.
    ///
    /// This is used internally for creating views that outlive the loader.
    #[inline(always)]
    pub(crate) fn handle(&self) -> Arc<SourceHandle> {
        self.handle.clone()
    }

    /// Close the file descriptor associated to the [`Source`].
    /// Useful for mmap when we don't need a dangling file descriptor.
    pub(crate) fn close_fd(&self) {
        unsafe {
            let inner_ptr = &self.handle.inner as *const _ as *mut hmll_sys::hmll_source;
            hmll_sys::hmll_source_close(inner_ptr);
        }
    }
}

// Source uses Arc internally, so Drop is automatic via Arc refcounting.
// When the last Arc<SourceHandle> is dropped, SourceHandle::drop() frees the source.

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("size", &self.size())
            .field("path", &self.path)
            .finish()
    }
}

// Source can be safely sent between threads (Arc<SourceHandle> is Send)
unsafe impl Send for Source {}
// Source can be safely shared between threads (Arc<SourceHandle> is Sync)
unsafe impl Sync for Source {}

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
    fn test_source_invalid_path() {
        let result = Source::open("/nonexistent/file.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_source_null_byte() {
        let result = Source::open("file\0name.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_source_open_and_size() {
        let content = b"Hello, HMLL! This is test data for the integration test.";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");

        assert_eq!(source.size(), content.len());
        assert!(source.path().exists());
    }

    #[test]
    fn test_source_clone_shares_handle() {
        let content = b"Test content for clone test.";
        let temp_file = create_test_file(content);

        let source1 = Source::open(temp_file.path()).expect("Failed to open source");
        let source2 = source1.clone();

        // Both sources should have the same size
        assert_eq!(source1.size(), source2.size());
        // They share the same underlying Arc handle
        assert!(Arc::ptr_eq(&source1.handle, &source2.handle));
    }

    #[test]
    fn test_source_has_mmap_content() {
        let content = b"Content to verify mmap.";
        let temp_file = create_test_file(content);

        let source = Source::open(temp_file.path()).expect("Failed to open source");

        // Content pointer should be non-null for mmap'd files
        assert!(!source.handle.inner.content.is_null());
    }
}
