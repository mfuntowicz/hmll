//! Data types for tensor elements.

use hmll_sys::hmll_dtype;

/// Data type of tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// Boolean (1 bit logical, stored as 1 byte)
    Bool,
    /// Brain floating point (16-bit)
    BFloat16,
    /// Complex number (64-bit: 32-bit real + 32-bit imaginary)
    Complex64,
    /// 4-bit floating point
    Float4,
    /// 6-bit floating point (E2M3 format)
    Float6E2M3,
    /// 6-bit floating point (E3M2 format)
    Float6E3M2,
    /// 8-bit floating point (E4M3 format)
    Float8E4M3,
    /// 8-bit floating point (E5M2 format)
    Float8E5M2,
    /// 8-bit floating point (E8M0 format)
    Float8E8M0,
    /// Half precision floating point (16-bit)
    Float16,
    /// Single precision floating point (32-bit)
    Float32,
    /// Double precision floating point (64-bit)
    Float64,
    /// Signed 4-bit integer
    Int4,
    /// Signed 8-bit integer
    Int8,
    /// Signed 16-bit integer
    Int16,
    /// Signed 32-bit integer
    Int32,
    /// Signed 64-bit integer
    Int64,
    /// Unsigned 4-bit integer
    UInt4,
    /// Unsigned 8-bit integer
    UInt8,
    /// Unsigned 16-bit integer
    UInt16,
    /// Unsigned 32-bit integer
    UInt32,
    /// Unsigned 64-bit integer
    UInt64,
    /// Unknown or unsupported data type
    Unknown,
}

impl DType {
    /// Convert from the underlying C enum value.
    #[inline(always)]
    pub const fn from_raw(dtype: hmll_dtype) -> Self {
        match dtype {
            hmll_dtype::HMLL_DTYPE_BOOL => DType::Bool,
            hmll_dtype::HMLL_DTYPE_BFLOAT16 => DType::BFloat16,
            hmll_dtype::HMLL_DTYPE_COMPLEX => DType::Complex64,
            hmll_dtype::HMLL_DTYPE_FLOAT4 => DType::Float4,
            hmll_dtype::HMLL_DTYPE_FLOAT6_E2M3 => DType::Float6E2M3,
            hmll_dtype::HMLL_DTYPE_FLOAT6_E3M2 => DType::Float6E3M2,
            hmll_dtype::HMLL_DTYPE_FLOAT8_E4M3 => DType::Float8E4M3,
            hmll_dtype::HMLL_DTYPE_FLOAT8_E5M2 => DType::Float8E5M2,
            hmll_dtype::HMLL_DTYPE_FLOAT8_E8M0 => DType::Float8E8M0,
            hmll_dtype::HMLL_DTYPE_FLOAT16 => DType::Float16,
            hmll_dtype::HMLL_DTYPE_FLOAT32 => DType::Float32,
            hmll_dtype::HMLL_DTYPE_FLOAT64 => DType::Float64,
            hmll_dtype::HMLL_DTYPE_SIGNED_INT4 => DType::Int4,
            hmll_dtype::HMLL_DTYPE_SIGNED_INT8 => DType::Int8,
            hmll_dtype::HMLL_DTYPE_SIGNED_INT16 => DType::Int16,
            hmll_dtype::HMLL_DTYPE_SIGNED_INT32 => DType::Int32,
            hmll_dtype::HMLL_DTYPE_SIGNED_INT64 => DType::Int64,
            hmll_dtype::HMLL_DTYPE_UNSIGNED_INT4 => DType::UInt4,
            hmll_dtype::HMLL_DTYPE_UNSIGNED_INT8 => DType::UInt8,
            hmll_dtype::HMLL_DTYPE_UNSIGNED_INT16 => DType::UInt16,
            hmll_dtype::HMLL_DTYPE_UNSIGNED_INT32 => DType::UInt32,
            hmll_dtype::HMLL_DTYPE_UNSIGNED_INT64 => DType::UInt64,
            _ => DType::Unknown,
        }
    }

    /// Get the size in bits of this data type.
    ///
    /// Returns 0 for unknown types.
    #[inline]
    pub const fn bits(&self) -> u8 {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 8,
            DType::Float8E4M3 | DType::Float8E5M2 | DType::Float8E8M0 => 8,
            DType::Float4 | DType::Int4 | DType::UInt4 => 4,
            DType::Float6E2M3 | DType::Float6E3M2 => 6,
            DType::BFloat16 | DType::Float16 | DType::Int16 | DType::UInt16 => 16,
            DType::Float32 | DType::Int32 | DType::UInt32 => 32,
            DType::Float64 | DType::Complex64 | DType::Int64 | DType::UInt64 => 64,
            DType::Unknown => 0,
        }
    }

    /// Check if this is a floating point type.
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            DType::BFloat16
                | DType::Float4
                | DType::Float6E2M3
                | DType::Float6E3M2
                | DType::Float8E4M3
                | DType::Float8E5M2
                | DType::Float8E8M0
                | DType::Float16
                | DType::Float32
                | DType::Float64
        )
    }

    /// Check if this is a signed integer type.
    #[inline]
    pub const fn is_signed_int(&self) -> bool {
        matches!(
            self,
            DType::Int4 | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Check if this is an unsigned integer type.
    #[inline]
    pub const fn is_unsigned_int(&self) -> bool {
        matches!(
            self,
            DType::UInt4 | DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64
        )
    }

    /// Check if this is an integer type (signed or unsigned).
    #[inline]
    pub const fn is_int(&self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DType::Bool => "BOOL",
            DType::BFloat16 => "BF16",
            DType::Complex64 => "C64",
            DType::Float4 => "F4",
            DType::Float6E2M3 => "F6_E2M3",
            DType::Float6E3M2 => "F6_E3M2",
            DType::Float8E4M3 => "F8_E4M3",
            DType::Float8E5M2 => "F8_E5M2",
            DType::Float8E8M0 => "F8_E8M0",
            DType::Float16 => "F16",
            DType::Float32 => "F32",
            DType::Float64 => "F64",
            DType::Int4 => "I4",
            DType::Int8 => "I8",
            DType::Int16 => "I16",
            DType::Int32 => "I32",
            DType::Int64 => "I64",
            DType::UInt4 => "U4",
            DType::UInt8 => "U8",
            DType::UInt16 => "U16",
            DType::UInt32 => "U32",
            DType::UInt64 => "U64",
            DType::Unknown => "UNKNOWN",
        })
    }
}
