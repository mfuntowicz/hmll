//! Tests for safetensors parsing functionality.

#![cfg(feature = "safetensors")]

use hmll::{DType, Registry, Source};
use std::io::Write;
use tempfile::NamedTempFile;

/// Create a minimal safetensors file with a single tensor.
///
/// Format:
/// - 8 bytes: header size (little-endian u64)
/// - N bytes: JSON header
/// - Data bytes
fn create_test_safetensors(
    tensor_name: &str,
    dtype: &str,
    shape: &[usize],
    data: &[u8],
) -> NamedTempFile {
    let shape_json: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = shape_json.join(", ");

    let data_end = data.len();
    let header = format!(
        r#"{{"{}":{{"dtype":"{}","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        tensor_name, dtype, shape_str, data_end
    );

    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    file.write_all(&header_size.to_le_bytes())
        .expect("Failed to write header size");
    file.write_all(header_bytes)
        .expect("Failed to write header");
    file.write_all(data).expect("Failed to write data");
    file.flush().expect("Failed to flush");

    file
}

#[test]
fn test_parse_safetensors_f32_tensor() {
    // Create a 2x2 float32 tensor
    let data: Vec<u8> = vec![0u8; 16]; // 4 floats * 4 bytes
    let file = create_test_safetensors("test_tensor", "FP32", &[2, 2], &data);

    let source = Source::open(file.path()).expect("Failed to open source");
    let registry = Registry::from_safetensors(&source).expect("Failed to parse");

    assert_eq!(registry.len(), 1);

    let tensor = registry.get(0).expect("Should have tensor at index 0");
    assert_eq!(tensor.name, "test_tensor");
    assert_eq!(tensor.dtype, DType::Float32);
    assert_eq!(tensor.dtype.bits(), 32);
    assert!(tensor.dtype.is_float());
    assert_eq!(tensor.shape, &[2, 2]);
    assert_eq!(tensor.numel(), 4);
}

#[test]
fn test_parse_safetensors_bf16_tensor() {
    // Create a 4x4 bfloat16 tensor
    let data: Vec<u8> = vec![0u8; 32]; // 16 elements * 2 bytes
    let file = create_test_safetensors("weights", "BF16", &[4, 4], &data);

    let source = Source::open(file.path()).expect("Failed to open source");
    let registry = Registry::from_safetensors(&source).expect("Failed to parse");

    assert_eq!(registry.len(), 1);

    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.name, "weights");
    assert_eq!(tensor.dtype, DType::BFloat16);
    assert_eq!(tensor.dtype.bits(), 16);
    assert!(tensor.dtype.is_float());
    assert!(!tensor.dtype.is_int());
}

#[test]
fn test_registry_iteration() {
    let data: Vec<u8> = vec![0u8; 8];
    let file = create_test_safetensors("layer.weight", "FP16", &[2, 2], &data);

    let source = Source::open(file.path()).expect("Failed to open source");
    let registry = Registry::from_safetensors(&source).expect("Failed to parse");

    let tensors: Vec<_> = registry.iter().collect();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "layer.weight");
    assert_eq!(tensors[0].dtype, DType::Float16);
}

#[test]
fn test_dtype_display() {
    assert_eq!(format!("{}", DType::Float32), "F32");
    assert_eq!(format!("{}", DType::BFloat16), "BF16");
    assert_eq!(format!("{}", DType::Float16), "F16");
    assert_eq!(format!("{}", DType::Int8), "I8");
    assert_eq!(format!("{}", DType::UInt8), "U8");
}

#[test]
fn test_dtype_properties() {
    // Float types
    assert!(DType::Float32.is_float());
    assert!(DType::BFloat16.is_float());
    assert!(DType::Float16.is_float());
    assert!(!DType::Float32.is_int());

    // Integer types
    assert!(DType::Int8.is_signed_int());
    assert!(DType::Int32.is_signed_int());
    assert!(!DType::UInt8.is_signed_int());

    assert!(DType::UInt8.is_unsigned_int());
    assert!(DType::UInt32.is_unsigned_int());
    assert!(!DType::Int8.is_unsigned_int());

    assert!(DType::Int8.is_int());
    assert!(DType::UInt8.is_int());
    assert!(!DType::Float32.is_int());

    // Bit widths
    assert_eq!(DType::Float32.bits(), 32);
    assert_eq!(DType::Float16.bits(), 16);
    assert_eq!(DType::BFloat16.bits(), 16);
    assert_eq!(DType::Int8.bits(), 8);
    assert_eq!(DType::Int64.bits(), 64);
}
