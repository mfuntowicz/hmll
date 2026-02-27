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
pub use hmll_device_kind::*;
#[cfg(feature = "safetensors")]
pub use hmll_dtype::*;
pub use hmll_loader_kind::*;
pub use hmll_status_code::*;

// Optimized helper functions for zero-cost error checking

/// Check if an error represents success
#[inline(always)]
pub const fn hmll_is_success(error: hmll_error) -> bool {
    matches!(error.code, hmll_status_code::HMLL_ERR_SUCCESS) && error.sys_err == 0
}

/// Check if an error represents failure
#[inline(always)]
pub const fn hmll_check_error(res: hmll_error) -> bool {
    !matches!(res.code, hmll_status_code::HMLL_ERR_SUCCESS) || res.sys_err != 0
}

/// Take latest error from context and reset it to success
#[inline(always)]
pub const fn hmll_take_error(ctx: &mut hmll) -> hmll_error {
    let err = ctx.error;
    ctx.error = hmll_error {
        code: hmll_status_code::HMLL_ERR_SUCCESS,
        sys_err: 0,
    };
    err
}

/// Create CPU device
#[inline(always)]
pub const fn hmll_device_cpu() -> hmll_device {
    hmll_device {
        kind: hmll_device_kind::HMLL_DEVICE_CPU,
        idx: 0,
    }
}

/// Create CUDA device
#[inline(always)]
pub const fn hmll_device_cuda(idx: u8) -> hmll_device {
    hmll_device {
        kind: hmll_device_kind::HMLL_DEVICE_CUDA,
        idx,
    }
}

/// Get error code
#[inline(always)]
pub const fn hmll_error_code(error: hmll_error) -> hmll_status_code {
    error.code
}

/// Get system error as i32
#[inline(always)]
pub const fn hmll_sys_error(error: hmll_error) -> i32 {
    error.sys_err
}

/// Check if device is CPU
#[inline(always)]
pub const fn hmll_is_cpu(device: &hmll_device) -> bool {
    matches!(device.kind, hmll_device_kind::HMLL_DEVICE_CPU)
}

/// Check if device is CUDA
#[inline(always)]
pub const fn hmll_is_cuda(device: &hmll_device) -> bool {
    matches!(device.kind, hmll_device_kind::HMLL_DEVICE_CUDA)
}

/// Get device index
#[inline(always)]
pub const fn hmll_device_index(device: &hmll_device) -> u8 {
    device.idx
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
        let cpu = hmll_device {
            kind: hmll_device_kind::HMLL_DEVICE_CPU,
            idx: 0,
        };
        let cuda0 = hmll_device {
            kind: hmll_device_kind::HMLL_DEVICE_CUDA,
            idx: 0,
        };
        let cuda3 = hmll_device {
            kind: hmll_device_kind::HMLL_DEVICE_CUDA,
            idx: 3,
        };

        assert!(hmll_is_cpu(&cpu));
        assert!(!hmll_is_cuda(&cpu));
        assert!(hmll_is_cuda(&cuda0));
        assert!(!hmll_is_cpu(&cuda0));
        assert!(hmll_is_cuda(&cuda3));
        assert_eq!(hmll_device_index(&cuda3), 3);
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
        assert_eq!(hmll_device_kind::HMLL_DEVICE_CPU as u32, 0);
        assert_eq!(hmll_device_kind::HMLL_DEVICE_CUDA as u32, 1);
    }

    #[test]
    fn test_source_size() {
        let source = hmll_source {
            fd: -1,
            size: 1024,
            content: std::ptr::null(),
        };
        assert_eq!(source.size, 1024);
        assert_eq!(source.fd, -1);
        assert!(source.content.is_null());
    }

    #[test]
    fn test_range_creation() {
        let range = hmll_range { start: 0, end: 100 };
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 100);
    }

    #[test]
    fn test_take_error_returns_error_and_resets() {
        let mut ctx = hmll {
            fetcher: std::ptr::null_mut(),
            sources: std::ptr::null_mut(),
            num_sources: 0,
            error: hmll_error {
                code: hmll_status_code::HMLL_ERR_IO_ERROR,
                sys_err: 42,
            },
        };

        let err = hmll_take_error(&mut ctx);

        // Should return the original error
        assert_eq!(err.code, hmll_status_code::HMLL_ERR_IO_ERROR);
        assert_eq!(err.sys_err, 42);

        // Context should be reset to success
        assert!(hmll_is_success(ctx.error));
        assert_eq!(ctx.error.code, hmll_status_code::HMLL_ERR_SUCCESS);
        assert_eq!(ctx.error.sys_err, 0);
    }

    #[test]
    fn test_take_error_on_success_is_noop() {
        let mut ctx = hmll {
            fetcher: std::ptr::null_mut(),
            sources: std::ptr::null_mut(),
            num_sources: 0,
            error: hmll_error {
                code: hmll_status_code::HMLL_ERR_SUCCESS,
                sys_err: 0,
            },
        };

        let err = hmll_take_error(&mut ctx);

        assert!(hmll_is_success(err));
        assert!(hmll_is_success(ctx.error));
    }

    #[test]
    fn test_take_error_idempotent() {
        let mut ctx = hmll {
            fetcher: std::ptr::null_mut(),
            sources: std::ptr::null_mut(),
            num_sources: 0,
            error: hmll_error {
                code: hmll_status_code::HMLL_ERR_ALLOCATION_FAILED,
                sys_err: -1,
            },
        };

        let err1 = hmll_take_error(&mut ctx);
        let err2 = hmll_take_error(&mut ctx);

        // First call gets the error
        assert_eq!(err1.code, hmll_status_code::HMLL_ERR_ALLOCATION_FAILED);
        assert_eq!(err1.sys_err, -1);

        // Second call gets success (already reset)
        assert!(hmll_is_success(err2));
    }

    #[test]
    fn test_iobuf_creation() {
        let device = hmll_device {
            kind: HMLL_DEVICE_CPU,
            idx: 0,
        };
        let iobuf = hmll_iobuf {
            size: 4096,
            ptr: std::ptr::null_mut(),
            device,
        };
        assert_eq!(iobuf.size, 4096);
        assert!(iobuf.ptr.is_null());
        assert!(hmll_is_cpu(&iobuf.device));
    }
}
