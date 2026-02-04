//! Mmap handle for Arc-based lifetime management.
//!
//! This module provides a wrapper around the C mmap backend pointer,
//! enabling Rust's `Arc` to manage the mmap lifetime. When all references
//! to the mmap are dropped, the underlying memory is unmapped and freed.

use std::ffi::c_void;
use std::sync::Arc;

/// Opaque handle to the C mmap backend.
///
/// This struct wraps the raw pointer to the mmap backend and implements
/// `Drop` to call `hmll_mmap_free()` when the handle is dropped.
/// Typically used via `Arc<MmapHandle>` for shared ownership.
pub struct MmapHandle {
    ptr: *mut c_void,
}

impl MmapHandle {
    /// Create a new MmapHandle from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid mmap backend pointer obtained from
    /// `hmll_get_mmap_backend()`, or NULL.
    #[inline]
    pub(crate) unsafe fn new(ptr: *mut c_void) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Create an Arc-wrapped MmapHandle from a raw pointer.
    ///
    /// Returns None if the pointer is null.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid mmap backend pointer obtained from
    /// `hmll_get_mmap_backend()`, or NULL.
    #[inline]
    pub(crate) unsafe fn new_arc(ptr: *mut c_void) -> Option<Arc<Self>> {
        Self::new(ptr).map(Arc::new)
    }
}

impl Drop for MmapHandle {
    fn drop(&mut self) {
        // Safety: ptr was obtained from hmll_get_mmap_backend() and is valid
        // until we call hmll_mmap_free()
        unsafe {
            hmll_sys::hmll_mmap_free(self.ptr);
        }
    }
}

// MmapHandle is Send and Sync - the underlying mmap is thread-safe for reads
unsafe impl Send for MmapHandle {}
unsafe impl Sync for MmapHandle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_handle() {
        // Creating handle from null pointer should return None
        unsafe {
            assert!(MmapHandle::new(std::ptr::null_mut()).is_none());
            assert!(MmapHandle::new_arc(std::ptr::null_mut()).is_none());
        }
    }
}
