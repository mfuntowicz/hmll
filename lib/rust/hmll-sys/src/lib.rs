//! Low-level FFI bindings to the hmll library.
//!
//! This crate provides direct FFI bindings to the C library, generated using bindgen.
//! For a safe, idiomatic Rust API, use the `hmll` crate instead.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Re-export enum variants at module level for convenience
pub use hmll_device::*;
#[cfg(feature = "safetensors")]
pub use hmll_dtype::*;
pub use hmll_loader_kind::*;
pub use hmll_status_code::*;

// Optimized helper functions for zero-cost error checking

/// Check if an error represents success (hot path - inline always)
#[inline(always)]
pub const fn hmll_is_success(error: hmll_error) -> bool {
    matches!(error.code, hmll_status_code::HMLL_ERR_SUCCESS) && error.sys_err == 0
}

/// Check if an error represents failure (hot path - inline always)
#[inline(always)]
pub const fn hmll_check_error(res: hmll_error) -> bool {
    !matches!(res.code, hmll_status_code::HMLL_ERR_SUCCESS) || res.sys_err != 0
}

/// Get error code (hot path - inline always)
#[inline(always)]
pub const fn hmll_error_code(error: hmll_error) -> hmll_status_code {
    error.code
}

/// Get system error as i32 (hot path - inline always)
#[inline(always)]
pub const fn hmll_sys_error(error: hmll_error) -> i32 {
    error.sys_err
}

/// Check if device is CPU (hot path - inline always)
#[inline(always)]
pub const fn hmll_is_cpu(device: hmll_device) -> bool {
    matches!(device, hmll_device::HMLL_DEVICE_CPU)
}

/// Check if device is CUDA (hot path - inline always)
#[inline(always)]
pub const fn hmll_is_cuda(device: hmll_device) -> bool {
    matches!(device, hmll_device::HMLL_DEVICE_CUDA)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_success() {
        let success_err = hmll_error {
            code: hmll_status_code::HMLL_ERR_SUCCESS,
            sys_err: 0,
        };
        assert!(hmll_is_success(success_err));
        assert!(!hmll_check_error(success_err));
    }

    #[test]
    fn test_error_failure() {
        let fail_err = hmll_error {
            code: hmll_status_code::HMLL_ERR_IO_ERROR,
            sys_err: 0,
        };
        assert!(!hmll_is_success(fail_err));
        assert!(hmll_check_error(fail_err));
    }

    #[test]
    fn test_error_system() {
        let sys_err = hmll_error {
            code: hmll_status_code::HMLL_ERR_SUCCESS,
            sys_err: -1,
        };
        assert!(!hmll_is_success(sys_err));
        assert!(hmll_check_error(sys_err));
    }

    #[test]
    fn test_device_checks() {
        assert!(hmll_is_cpu(hmll_device::HMLL_DEVICE_CPU));
        assert!(!hmll_is_cuda(hmll_device::HMLL_DEVICE_CPU));
        assert!(hmll_is_cuda(hmll_device::HMLL_DEVICE_CUDA));
        assert!(!hmll_is_cpu(hmll_device::HMLL_DEVICE_CUDA));
    }

    #[test]
    fn test_error_getters() {
        let err = hmll_error {
            code: hmll_status_code::HMLL_ERR_IO_ERROR,
            sys_err: 42,
        };
        assert_eq!(hmll_error_code(err), hmll_status_code::HMLL_ERR_IO_ERROR);
        assert_eq!(hmll_sys_error(err), 42);
    }

    #[test]
    fn test_enum_values() {
        // Enums should have correct discriminant values
        assert_eq!(hmll_status_code::HMLL_ERR_SUCCESS as u32, 0);
        assert_eq!(hmll_device::HMLL_DEVICE_CPU as u32, 0);
        assert_eq!(hmll_device::HMLL_DEVICE_CUDA as u32, 1);
    }

    #[test]
    fn test_source_size() {
        let source = hmll_source { fd: -1, size: 1024 };
        assert_eq!(source.size, 1024);
        assert_eq!(source.fd, -1);
    }

    #[test]
    fn test_range_creation() {
        let range = hmll_range { start: 0, end: 100 };
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 100);
    }

    #[test]
    fn test_iobuf_creation() {
        let iobuf = hmll_iobuf {
            size: 4096,
            ptr: std::ptr::null_mut(),
            device: hmll_device::HMLL_DEVICE_CPU,
        };
        assert_eq!(iobuf.size, 4096);
        assert!(iobuf.ptr.is_null());
        assert_eq!(iobuf.device, hmll_device::HMLL_DEVICE_CPU);
    }
}
