//! Device types for specifying where data should be loaded.

use hmll_sys::{hmll_device, hmll_device_kind};

/// Represents a device where data can be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU memory
    Cpu,
    /// CUDA GPU memory
    Cuda(u8),
}

impl Device {
    /// Convert to the underlying C enum value.
    #[inline(always)]
    pub(crate) const fn to_raw(self) -> hmll_device {
        match self {
            Device::Cpu => hmll_sys::hmll_device_cpu(),
            Device::Cuda(idx) => hmll_sys::hmll_device_cuda(idx),
        }
    }

    /// Convert from the underlying C enum value.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) const fn from_raw(device: hmll_device) -> Self {
        match device.kind {
            hmll_device_kind::HMLL_DEVICE_CPU => Device::Cpu,
            hmll_device_kind::HMLL_DEVICE_CUDA => Device::Cuda(device.idx),
        }
    }
}

impl Default for Device {
    /// Default device is CPU.
    #[inline(always)]
    fn default() -> Self {
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
        }
    }
}
