//! Buffer and range types for data operations.

use crate::Device;
use std::ops;

/// Represents a range of bytes to fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Range {
    pub start: usize,
    pub end: usize,
}

impl Range {
    /// Create a new range.
    ///
    /// This can be evaluated at compile time for constant ranges.
    #[inline(always)]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Get the length of the range.
    ///
    /// Hot path - inline always for zero-cost abstraction.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Check if the range is empty.
    ///
    /// Hot path - inline always for zero-cost abstraction.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Convert to the underlying C struct.
    ///
    /// Hot path - always inline for FFI conversion.
    #[inline(always)]
    pub(crate) fn to_raw(self) -> hmll_sys::hmll_range {
        hmll_sys::hmll_range {
            start: self.start,
            end: self.end,
        }
    }

    /// Convert from the underlying C struct.
    ///
    /// Hot path - always inline for FFI conversion.
    #[allow(unused)]
    #[inline(always)]
    pub(crate) const fn from_raw(range: hmll_sys::hmll_range) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl From<ops::Range<usize>> for Range {
    /// Convert from standard library Range.
    ///
    /// Hot path - inline always for zero-cost conversion.
    #[inline(always)]
    fn from(range: ops::Range<usize>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl From<Range> for ops::Range<usize> {
    /// Convert to standard library Range.
    ///
    /// Hot path - inline always for zero-cost conversion.
    #[inline(always)]
    fn from(range: Range) -> Self {
        range.start..range.end
    }
}

/// A buffer containing fetched data.
#[derive(Debug)]
pub struct Buffer {
    ptr: *mut u8,
    size: usize,
    device: Device,
    // We own this memory, so we need to track whether to free it
    #[allow(dead_code)]
    owned: bool,
}

impl Buffer {
    /// Create a new buffer from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` points to valid memory of at least `size` bytes.
    ///
    /// Hot path - inline always for construction.
    #[inline(always)]
    pub(crate) unsafe fn from_raw_parts(ptr: *mut u8, size: usize, device: Device, owned: bool) -> Self {
        Self {
            ptr,
            size,
            device,
            owned,
        }
    }

    /// Get the buffer as a byte slice (CPU only).
    ///
    /// Hot path - inline for efficient slice creation.
    #[inline]
    pub fn as_slice(&self) -> Option<&[u8]> {
        if self.device == Device::Cpu && !self.ptr.is_null() {
            unsafe { Some(std::slice::from_raw_parts(self.ptr, self.size)) }
        } else {
            None
        }
    }

    /// Get the size of the buffer in bytes.
    ///
    /// Hot path - inline always for zero-cost field access.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty.
    ///
    /// Hot path - inline always for zero-cost check.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the device where the buffer is located.
    ///
    /// Hot path - inline always for zero-cost field access.
    #[inline(always)]
    pub const fn device(&self) -> Device {
        self.device
    }

    /// Get a raw pointer to the buffer.
    ///
    /// Hot path - inline always for zero-cost pointer access.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Get a mutable raw pointer to the buffer.
    ///
    /// Hot path - inline always for zero-cost pointer access.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Convert to a Vec (copies data if on CPU, panics if on GPU).
    ///
    /// This is a less common operation, so we use regular inline.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice()
            .expect("Cannot convert GPU buffer to Vec")
            .to_vec()
    }
}

// Buffer is Send and Sync as long as the device supports it
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Note: In hmll, buffers are managed by the context
        // We don't manually free them here as they're part of the arena allocator
        // This is why we track `owned` - in the future we might need to handle this differently
    }
}
