//! Prefetch context for pipelined tensor loading.
//!
//! This module provides async prefetching of tensors, allowing I/O to overlap
//! with user processing (e.g., quantization).

use crate::{Buffer, Device, Error, Result};
use std::ptr;

/// State of a prefetch slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is idle and available.
    Idle,
    /// Copying to pinned staging buffer (sync phase).
    Staging,
    /// Async H2D load in progress.
    Loading,
    /// Load complete, buffer ready.
    Ready,
    /// Load failed.
    Error,
}

impl From<hmll_sys::hmll_prefetch_state> for SlotState {
    fn from(state: hmll_sys::hmll_prefetch_state) -> Self {
        match state {
            hmll_sys::HMLL_PREFETCH_IDLE => SlotState::Idle,
            hmll_sys::HMLL_PREFETCH_STAGING => SlotState::Staging,
            hmll_sys::HMLL_PREFETCH_LOADING => SlotState::Loading,
            hmll_sys::HMLL_PREFETCH_READY => SlotState::Ready,
            hmll_sys::HMLL_PREFETCH_ERROR => SlotState::Error,
            _ => SlotState::Error,
        }
    }
}

/// Prefetch context for managing concurrent tensor loading.
///
/// This context manages multiple "slots" that can load tensors concurrently.
/// For CUDA devices, each slot has its own stream and event for async operations.
///
/// # Example
///
/// ```no_run
/// use hmll::{Device, PrefetchContext};
///
/// # fn main() -> Result<(), hmll::Error> {
/// // Create prefetch context with 4 slots for CUDA device 0
/// let mut ctx = PrefetchContext::new(4, Device::Cuda, 0)?;
///
/// // Start loading tensor data asynchronously
/// let src_data: &[u8] = &[1, 2, 3, 4];
/// let slot = ctx.start_load(src_data.as_ptr() as *const _, src_data.len(), 0)?;
///
/// // ... do other work while loading ...
///
/// // Wait and get the buffer
/// let buffer = ctx.take_buffer(slot)?;
/// # Ok(())
/// # }
/// ```
pub struct PrefetchContext {
    inner: Box<hmll_sys::hmll_prefetch_ctx>,
    device: Device,
}

impl PrefetchContext {
    /// Create a new prefetch context.
    ///
    /// # Arguments
    ///
    /// * `num_slots` - Number of concurrent load slots (1-16)
    /// * `device` - Target device (CPU or CUDA)
    /// * `device_id` - GPU device index (ignored for CPU)
    pub fn new(num_slots: usize, device: Device, device_id: i32) -> Result<Self> {
        let mut inner = Box::new(hmll_sys::hmll_prefetch_ctx {
            slots: ptr::null_mut(),
            num_slots: 0,
            next_slot: 0,
            device_id: 0,
            device: device.to_raw(),
            use_pinned: 1, // Enable pinned memory by default
        });

        unsafe {
            let err = hmll_sys::hmll_prefetch_init(
                inner.as_mut(),
                num_slots,
                device.to_raw(),
                device_id,
            );
            Error::check_hmll_error(err)?;
        }

        Ok(Self { inner, device })
    }

    /// Start async load of tensor data into a slot.
    ///
    /// For CUDA, this allocates GPU memory and starts an async memcpy.
    /// For CPU, this just stores the pointer (zero-copy mmap view).
    ///
    /// # Arguments
    ///
    /// * `src_ptr` - Source data pointer (typically from mmap)
    /// * `size` - Size of tensor data in bytes
    /// * `tensor_index` - Index to identify this tensor
    ///
    /// # Returns
    ///
    /// The slot index that was used.
    pub fn start_load(
        &mut self,
        src_ptr: *const std::ffi::c_void,
        size: usize,
        tensor_index: usize,
    ) -> Result<usize> {
        let mut out_slot: usize = 0;

        unsafe {
            let err = hmll_sys::hmll_prefetch_start_load(
                self.inner.as_mut(),
                src_ptr,
                size,
                tensor_index,
                &mut out_slot,
            );
            Error::check_hmll_error(err)?;
        }

        Ok(out_slot)
    }

    /// Find a slot that is idle or has completed loading.
    ///
    /// Returns `None` if no slot is available.
    pub fn find_available_slot(&mut self) -> Option<usize> {
        let slot = unsafe { hmll_sys::hmll_prefetch_find_available_slot(self.inner.as_mut()) };
        if slot < 0 {
            None
        } else {
            Some(slot as usize)
        }
    }

    /// Check if a specific slot has completed loading (non-blocking).
    pub fn slot_ready(&mut self, slot_index: usize) -> bool {
        unsafe { hmll_sys::hmll_prefetch_slot_ready(self.inner.as_mut(), slot_index) != 0 }
    }

    /// Wait for a specific slot to complete loading (blocking).
    pub fn wait_slot(&mut self, slot_index: usize) -> Result<()> {
        unsafe {
            let err = hmll_sys::hmll_prefetch_wait_slot(self.inner.as_mut(), slot_index);
            Error::check_hmll_error(err)?;
        }
        Ok(())
    }

    /// Find slot containing a specific tensor (by index).
    ///
    /// Returns `None` if the tensor is not in any slot.
    pub fn find_tensor(&mut self, tensor_index: usize) -> Option<usize> {
        let slot = unsafe { hmll_sys::hmll_prefetch_find_tensor(self.inner.as_mut(), tensor_index) };
        if slot < 0 {
            None
        } else {
            Some(slot as usize)
        }
    }

    /// Get buffer from slot, transferring ownership to caller.
    ///
    /// The slot becomes idle after this call.
    /// For CUDA, this waits for the async copy to complete if needed.
    pub fn take_buffer(&mut self, slot_index: usize) -> Result<Buffer> {
        let mut iobuf = hmll_sys::hmll_iobuf {
            ptr: ptr::null_mut(),
            size: 0,
            device: self.device.to_raw(),
        };

        unsafe {
            let err =
                hmll_sys::hmll_prefetch_take_buffer(self.inner.as_mut(), slot_index, &mut iobuf);
            Error::check_hmll_error(err)?;
        }

        // PrefetchContext is CUDA-only; CPU prefetch is handled in Rust (safetensors)
        debug_assert_eq!(self.device, Device::Cuda, "PrefetchContext only supports CUDA");
        Ok(unsafe { Buffer::from_raw_owned(iobuf) })
    }

    /// Poll all slots for completion (updates slot states).
    ///
    /// Call periodically to detect completed loads without blocking.
    pub fn poll(&mut self) {
        unsafe {
            hmll_sys::hmll_prefetch_poll(self.inner.as_mut());
        }
    }

    /// Get the number of slots.
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.inner.num_slots
    }

    /// Get the device type.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get state of a specific slot.
    pub fn slot_state(&self, slot_index: usize) -> Option<SlotState> {
        if slot_index >= self.inner.num_slots {
            return None;
        }

        unsafe {
            let slot = &*self.inner.slots.add(slot_index);
            Some(SlotState::from(slot.state))
        }
    }
}

impl Drop for PrefetchContext {
    fn drop(&mut self) {
        unsafe {
            hmll_sys::hmll_prefetch_destroy(self.inner.as_mut());
        }
    }
}

// SAFETY: PrefetchContext can be moved between threads.
// - CUDA streams/events can be used from any thread (not concurrently)
// - All methods require `&mut self`, ensuring exclusive access
// - No `Sync` impl means &PrefetchContext cannot be shared across threads
// These constraints guarantee only one thread accesses the CUDA resources at a time.
unsafe impl Send for PrefetchContext {}

// Tests require CUDA since PrefetchContext is CUDA-only.
// CPU prefetch is handled directly in Rust (safetensors) without going through C.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_context_cuda() {
        // This test requires an actual CUDA device
        let ctx = PrefetchContext::new(4, Device::Cuda, 0);
        if ctx.is_err() {
            // No CUDA device available, skip test
            return;
        }

        let ctx = ctx.unwrap();
        assert_eq!(ctx.num_slots(), 4);
        assert_eq!(ctx.device(), Device::Cuda);
    }

    #[test]
    fn test_prefetch_cpu_returns_error() {
        // CPU is not supported, should return error
        let ctx = PrefetchContext::new(4, Device::Cpu, 0);
        assert!(ctx.is_err());
    }
}
