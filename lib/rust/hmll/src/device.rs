//! Device types for specifying where data should be loaded.

use hmll_sys::{hmll_device, HMLL_DEVICE_CPU, HMLL_DEVICE_CUDA};

/// Represents a device where data can be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU memory
    Cpu,
    /// CUDA GPU memory
    Cuda,
}

impl Device {
    /// Convert to the underlying C enum value.
    #[inline(always)]
    pub(crate) const fn to_raw(self) -> hmll_device {
        match self {
            Device::Cpu => HMLL_DEVICE_CPU,
            Device::Cuda => HMLL_DEVICE_CUDA,
        }
    }

    /// Convert from the underlying C enum value.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) const fn from_raw(device: hmll_device) -> Option<Self> {
        match device {
            HMLL_DEVICE_CPU => Some(Device::Cpu),
            HMLL_DEVICE_CUDA => Some(Device::Cuda),
            _ => None,
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
            Device::Cpu => write!(f, "CPU"),
            Device::Cuda => write!(f, "CUDA"),
        }
    }
}
