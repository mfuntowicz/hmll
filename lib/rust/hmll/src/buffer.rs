//! Buffer and range types for data operations.

use crate::Device;
use hmll_sys::{hmll_free_buffer, hmll_iobuf};
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
///
/// Wraps the underlying `hmll_iobuf` C struct directly.
pub struct Buffer {
    buf: hmll_iobuf,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.buf.size)
            .field("ptr", &self.buf.ptr)
            .field("device", &self.device())
            .field("owned", &self.is_owned())
            .finish()
    }
}

impl Buffer {
    /// Create an empty buffer for the given device.
    ///
    /// This is useful when you need to represent a zero-length fetch result.
    #[inline(always)]
    pub fn empty(device: Device) -> Self {
        Self {
            buf: hmll_iobuf {
                size: 0,
                ptr: std::ptr::null_mut(),
                device: device.to_raw(),
                owned: 0,
                mmap_ref: std::ptr::null_mut(),
            },
        }
    }

    /// Create a new buffer from an `hmll_iobuf`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `buf.ptr` points to valid memory of at least `buf.size` bytes.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(buf: hmll_iobuf) -> Self {
        Self { buf }
    }

    /// Get the buffer as a byte slice (CPU only).
    #[inline]
    pub fn as_slice(&self) -> Option<&[u8]> {
        if self.device() == Device::Cpu {
            if self.buf.ptr.is_null() || self.buf.size == 0 {
                // Return empty slice for empty/null buffers
                Some(&[])
            } else {
                unsafe { Some(std::slice::from_raw_parts(self.buf.ptr as *const u8, self.buf.size)) }
            }
        } else {
            None
        }
    }

    /// Get the size of the buffer in bytes.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.buf.size
    }

    /// Check if the buffer is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.buf.size == 0
    }

    /// Get the device where the buffer is located.
    #[inline(always)]
    pub fn device(&self) -> Device {
        Device::from_raw(self.buf.device)
    }

    /// Get a raw pointer to the buffer.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.buf.ptr as *const u8
    }

    /// Convert to a Vec (copies data if on CPU, panics if on GPU).
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice()
            .expect("Cannot convert GPU buffer to Vec")
            .to_vec()
    }

    /// Check if this buffer owns its memory.
    ///
    /// Owned buffers are freed when dropped. Non-owned buffers (views) are
    /// not freed because they point to memory managed elsewhere (e.g., mmap'd region).
    #[inline(always)]
    pub const fn is_owned(&self) -> bool {
        self.buf.owned != 0
    }
}

// Buffer is Send and Sync as long as the device supports it
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.buf.ptr.is_null() {
            // hmll_free_buffer checks the owned flag and only frees if owned
            unsafe { hmll_free_buffer(&mut self.buf) };
        }
    }
}
