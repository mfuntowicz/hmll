//! Source file handling for hmll.

use crate::error::{Error, Result};
use std::ffi::CString;
use std::path::Path;

/// A source file for loading model weights.
///
/// This wraps a file descriptor and ensures proper cleanup when dropped.
#[derive(Debug)]
pub struct Source {
    inner: hmll_sys::hmll_source,
    path: Option<String>,
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
            size: 0,
        };

        unsafe {
            let err = hmll_sys::hmll_source_open(c_path.as_ptr(), &mut source);
            Error::check_hmll_error(err)?;
        }

        Ok(Self {
            inner: source,
            path: Some(path_str.to_string()),
        })
    }

    /// Get the size of the source file in bytes.
    #[inline(always)]
    pub const fn size(&self) -> usize {
        self.inner.size
    }

    /// Get the file descriptor (platform-specific).
    #[cfg(target_family = "unix")]
    #[inline(always)]
    pub const fn fd(&self) -> i32 {
        self.inner.fd
    }

    /// Get the path of the source file if available.
    #[inline]
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Get a reference to the underlying hmll_source.
    #[inline(always)]
    pub(crate) const fn as_raw(&self) -> &hmll_sys::hmll_source {
        &self.inner
    }

    /// Consume self and return the raw hmll_source.
    ///
    /// # Safety
    ///
    /// The caller is responsible for calling hmll_source_close on the returned source.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) unsafe fn into_raw(mut self) -> hmll_sys::hmll_source {
        let source = self.inner;
        // Prevent Drop from running
        self.inner.fd = -1;
        source
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        // only close if we have a valid file descriptor
        if self.inner.fd >= 0 {
            unsafe {
                hmll_sys::hmll_source_close(&self.inner);
            }
        }
    }
}

// Source can be safely sent between threads
unsafe impl Send for Source {}
// Source can be safely shared between threads (read-only operations)
unsafe impl Sync for Source {}

#[cfg(test)]
mod tests {
    use super::*;

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
}
