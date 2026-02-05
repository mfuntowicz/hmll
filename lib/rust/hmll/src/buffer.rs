//! Buffer and range types for data operations.

use crate::source::SourceHandle;
use crate::Device;
use hmll_sys::{hmll_free_buffer, hmll_iobuf};
use std::ops;
use std::sync::Arc;

/// Describes the ownership and lifetime semantics of a buffer.
enum BufferKind {
    /// Empty buffer - nothing to free or keep alive.
    Empty,
    /// Owned memory allocated via hmll (must be freed on drop).
    Owned,
    /// Zero-copy view into mmap'd source region.
    /// The Arc keeps the source (and its mmap) alive while this buffer exists.
    SourceView(#[allow(dead_code)] Arc<SourceHandle>),
}

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
/// Buffers come in three flavors:
/// - **Empty**: Zero-length buffer with no memory.
/// - **Owned**: Allocated memory that is freed when the buffer is dropped.
/// - **SourceView**: Zero-copy pointer into mmap'd memory, kept alive via Arc.
pub struct Buffer {
    buf: hmll_iobuf,
    kind: BufferKind,
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
            },
            kind: BufferKind::Empty,
        }
    }

    /// Create a new owned buffer from an `hmll_iobuf`.
    ///
    /// Owned buffers are freed when dropped.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `buf.ptr` points to valid memory of at least `buf.size` bytes,
    /// and that the memory was allocated via hmll allocation functions.
    #[inline(always)]
    pub(crate) unsafe fn from_raw_owned(buf: hmll_iobuf) -> Self {
        Self {
            buf,
            kind: BufferKind::Owned,
        }
    }

    /// Create a zero-copy view into mmap'd source memory.
    ///
    /// The Arc keeps the source (and its mmap) alive while this buffer exists.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` points to valid memory within the source's
    /// mmap region of at least `size` bytes.
    #[inline(always)]
    pub(crate) unsafe fn from_source_view(
        ptr: *mut std::ffi::c_void,
        size: usize,
        device: Device,
        source_handle: Arc<SourceHandle>,
    ) -> Self {
        Self {
            buf: hmll_iobuf {
                size,
                ptr,
                device: device.to_raw(),
            },
            kind: BufferKind::SourceView(source_handle),
        }
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

    /// Convert to a Vec (copies data if on CPU).
    ///
    /// Returns an error if the buffer is on a GPU device.
    #[inline]
    pub fn to_vec(&self) -> crate::Result<Vec<u8>> {
        self.as_slice()
            .map(|s| s.to_vec())
            .ok_or(crate::Error::UnsupportedDevice)
    }

    /// Check if this buffer owns its memory.
    ///
    /// Owned buffers are freed when dropped. Views point into mmap'd memory
    /// and are kept alive by an Arc reference to the source.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self.kind, BufferKind::Owned)
    }
}

// Buffer is Send and Sync as long as the device supports it
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let BufferKind::Owned = self.kind {
            if !self.buf.ptr.is_null() {
                unsafe { hmll_free_buffer(&mut self.buf) };
            }
        }
        // For SourceView: the Arc is dropped automatically, decrementing refcount.
        // When the last Arc is dropped, SourceHandle::drop() unmaps the memory.
    }
}
