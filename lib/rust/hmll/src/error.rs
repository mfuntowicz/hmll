//! Error types for hmll operations.

use std::ffi::CStr;
use thiserror::Error;

/// Result type alias for hmll operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using hmll.
#[derive(Debug, Error)]
pub enum Error {
    #[error("Unsupported platform")]
    UnsupportedPlatform,

    #[error("Unsupported file format")]
    UnsupportedFileFormat,

    #[error("Unsupported device")]
    UnsupportedDevice,

    #[error("Unsupported backend for this operation")]
    UnsupportedBackend,

    #[error("Memory allocation failed")]
    AllocationFailed,

    #[error("Table is empty")]
    TableEmpty,

    #[error("Tensor not found")]
    TensorNotFound,

    #[error("Invalid range")]
    InvalidRange,

    #[error("Buffer address not aligned")]
    BufferAddrNotAligned,

    #[error("Buffer too small")]
    BufferTooSmall,

    #[error("I/O error")]
    IoError,

    #[error("No source provided")]
    NoSourceProvided,

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("File is empty")]
    FileEmpty,

    #[error("Failed to open file")]
    FileOpenFailed,

    #[error("Failed to read file")]
    FileReadFailed,

    #[error("Failed to register file")]
    FileRegistrationFailed,

    #[error("Memory mapping failed")]
    MmapFailed,

    #[error("I/O buffer registration failed")]
    IoBufferRegistrationFailed,

    #[error("SafeTensors: Invalid JSON header")]
    SafeTensorsJsonInvalidHeader,

    #[error("SafeTensors: Malformed JSON header")]
    SafeTensorsJsonMalformedHeader,

    #[error("SafeTensors: Malformed JSON index")]
    SafeTensorsJsonMalformedIndex,

    #[error("CUDA not enabled")]
    CudaNotEnabled,

    #[error("No CUDA device available")]
    CudaNoDevice,

    #[error("System error: {0}")]
    SystemError(String),

    #[error("Unknown data type")]
    UnknownDType,

    #[error("Unknown error code")]
    Unknown,

    #[error("fetchv iobufs iterator is exhausted when expected to yield an additional element")]
    ExhaustedIterator,

    #[error("select file index is invalid: {0}")]
    InvalidFileIndex(usize),
}

impl Error {
    /// Convert a hmll_error to a Rust Error.
    #[cold]
    #[inline(never)]
    pub(crate) fn from_hmll_error(err: hmll_sys::hmll_error) -> Self {
        use hmll_sys::*;

        // Check if it's a system error
        if err.code == HMLL_ERR_SYSTEM {
            let msg = unsafe {
                let ptr = hmll_strerr(err);
                if ptr.is_null() {
                    format!("System error code: {}", err.sys_err)
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            return Error::SystemError(msg);
        }

        // Map hmll error codes to Rust errors
        match err.code {
            HMLL_ERR_SUCCESS => unreachable!("Success is not an error"),
            HMLL_ERR_UNSUPPORTED_PLATFORM => Error::UnsupportedPlatform,
            HMLL_ERR_UNSUPPORTED_FILE_FORMAT => Error::UnsupportedFileFormat,
            HMLL_ERR_UNSUPPORTED_DEVICE => Error::UnsupportedDevice,
            HMLL_ERR_ALLOCATION_FAILED => Error::AllocationFailed,
            HMLL_ERR_TABLE_EMPTY => Error::TableEmpty,
            HMLL_ERR_TENSOR_NOT_FOUND => Error::TensorNotFound,
            HMLL_ERR_INVALID_RANGE => Error::InvalidRange,
            HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED => Error::BufferAddrNotAligned,
            HMLL_ERR_BUFFER_TOO_SMALL => Error::BufferTooSmall,
            HMLL_ERR_IO_ERROR => Error::IoError,
            HMLL_ERR_NO_SOURCE_PROVIDED => Error::NoSourceProvided,
            HMLL_ERR_FILE_NOT_FOUND => Error::FileNotFound(String::new()),
            HMLL_ERR_FILE_EMPTY => Error::FileEmpty,
            HMLL_ERR_FILE_OPEN_FAILED => Error::FileOpenFailed,
            HMLL_ERR_FILE_READ_FAILED => Error::FileReadFailed,
            HMLL_ERR_FILE_REGISTRATION_FAILED => Error::FileRegistrationFailed,
            HMLL_ERR_MMAP_FAILED => Error::MmapFailed,
            HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED => Error::IoBufferRegistrationFailed,
            HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER => Error::SafeTensorsJsonInvalidHeader,
            HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER => Error::SafeTensorsJsonMalformedHeader,
            HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX => Error::SafeTensorsJsonMalformedIndex,
            HMLL_ERR_CUDA_NOT_ENABLED => Error::CudaNotEnabled,
            HMLL_ERR_CUDA_NO_DEVICE => Error::CudaNoDevice,
            HMLL_ERR_UNKNOWN_DTYPE => Error::UnknownDType,
            _ => Error::Unknown,
        }
    }

    /// Check if a hmll_error represents success.
    #[inline(always)]
    pub(crate) fn check_hmll_error(err: hmll_sys::hmll_error) -> Result<()> {
        if hmll_sys::hmll_is_success(err) {
            Ok(())
        } else {
            Err(Self::from_hmll_error(err))
        }
    }
}
