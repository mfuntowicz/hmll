//! Tests for safetensors parsing functionality.

#![cfg(feature = "safetensors")]

use hmll::{DType, Registry, Source};
use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

/// Tensor definition for creating test safetensors files.
struct TensorDef<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [usize],
    data: &'a [u8],
}

/// Create a safetensors file with the given tensors.
///
/// Format:
/// - 8 bytes: header size (little-endian u64)
/// - N bytes: JSON header
/// - Data bytes
fn create_safetensors(tensors: &[TensorDef]) -> NamedTempFile {
    create_safetensors_with_metadata(tensors, None)
}

/// Create a safetensors file with the given tensors and optional metadata.
fn create_safetensors_with_metadata(
    tensors: &[TensorDef],
    metadata: Option<Value>,
) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    write_safetensors_to(&mut file, tensors, metadata);
    file
}

/// Write a safetensors file to an arbitrary writer.
fn write_safetensors_to<W: Write>(writer: &mut W, tensors: &[TensorDef], metadata: Option<Value>) {
    let mut header = Map::new();
    let mut all_data = Vec::new();
    let mut offset = 0usize;

    // Add metadata if provided
    if let Some(meta) = metadata {
        header.insert("__metadata__".to_string(), meta);
    }

    // Add tensor entries
    for tensor in tensors {
        let end_offset = offset + tensor.data.len();

        let tensor_info = json!({
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, end_offset]
        });

        header.insert(tensor.name.to_string(), tensor_info);
        all_data.extend_from_slice(tensor.data);
        offset = end_offset;
    }

    let header_json = serde_json::to_string(&header).expect("Failed to serialize header");
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    writer
        .write_all(&header_size.to_le_bytes())
        .expect("Failed to write header size");
    writer
        .write_all(header_bytes)
        .expect("Failed to write header");
    writer
        .write_all(&all_data)
        .expect("Failed to write data");
    writer.flush().expect("Failed to flush");
}

/// Write a safetensors file to a path.
fn write_safetensors_to_path(path: &Path, tensors: &[TensorDef]) {
    let mut file = std::fs::File::create(path).expect("Failed to create file");
    write_safetensors_to(&mut file, tensors, None);
}

#[test]
fn test_parse_safetensors_f32_tensor() {
    let data = vec![0u8; 16]; // 4 floats * 4 bytes
    let file = create_safetensors(&[TensorDef {
        name: "test_tensor",
        dtype: "FP32",
        shape: &[2, 2],
        data: &data,
    }]);

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
    let data = vec![0u8; 32]; // 16 elements * 2 bytes
    let file = create_safetensors(&[TensorDef {
        name: "weights",
        dtype: "BF16",
        shape: &[4, 4],
        data: &data,
    }]);

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
    let data = vec![0u8; 8];
    let file = create_safetensors(&[TensorDef {
        name: "layer.weight",
        dtype: "FP16",
        shape: &[2, 2],
        data: &data,
    }]);

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

#[test]
fn test_multiple_tensors() {
    let weight_data = vec![0u8; 64];
    let bias_data = vec![0u8; 16];
    let mean_data = vec![0u8; 16];

    let file = create_safetensors(&[
        TensorDef {
            name: "weight",
            dtype: "FP32",
            shape: &[4, 4],
            data: &weight_data,
        },
        TensorDef {
            name: "bias",
            dtype: "FP32",
            shape: &[4],
            data: &bias_data,
        },
        TensorDef {
            name: "running_mean",
            dtype: "FP32",
            shape: &[4],
            data: &mean_data,
        },
    ]);

    let source = Source::open(file.path()).expect("Failed to open source");
    let registry = Registry::from_safetensors(&source).expect("Failed to parse");

    assert_eq!(registry.len(), 3);
    assert!(!registry.is_empty());

    let names: Vec<_> = registry.iter().map(|t| t.name.to_string()).collect();
    assert!(names.contains(&"weight".to_string()));
    assert!(names.contains(&"bias".to_string()));
    assert!(names.contains(&"running_mean".to_string()));
}

#[test]
fn test_integer_dtypes() {
    let data = vec![0u8; 8];

    // Test signed integers
    for (dtype_str, expected_dtype, expected_bits) in [
        ("I8", DType::Int8, 8),
        ("I16", DType::Int16, 16),
        ("I32", DType::Int32, 32),
        ("I64", DType::Int64, 64),
    ] {
        let file = create_safetensors(&[TensorDef {
            name: "tensor",
            dtype: dtype_str,
            shape: &[8],
            data: &data,
        }]);
        let source = Source::open(file.path()).unwrap();
        let registry = Registry::from_safetensors(&source).unwrap();
        let tensor = registry.get(0).unwrap();
        assert_eq!(tensor.dtype, expected_dtype);
        assert_eq!(tensor.dtype.bits(), expected_bits);
        assert!(tensor.dtype.is_signed_int());
    }

    // Test unsigned integers
    for (dtype_str, expected_dtype) in [
        ("U8", DType::UInt8),
        ("U16", DType::UInt16),
        ("U32", DType::UInt32),
        ("U64", DType::UInt64),
    ] {
        let file = create_safetensors(&[TensorDef {
            name: "tensor",
            dtype: dtype_str,
            shape: &[8],
            data: &data,
        }]);
        let source = Source::open(file.path()).unwrap();
        let registry = Registry::from_safetensors(&source).unwrap();
        let tensor = registry.get(0).unwrap();
        assert_eq!(tensor.dtype, expected_dtype);
        assert!(tensor.dtype.is_unsigned_int());
    }
}

#[test]
fn test_tensor_shapes() {
    // 1D tensor
    let file = create_safetensors(&[TensorDef {
        name: "vec",
        dtype: "FP32",
        shape: &[16],
        data: &vec![0u8; 64],
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.shape, &[16]);
    assert_eq!(tensor.numel(), 16);

    // 3D tensor
    let file = create_safetensors(&[TensorDef {
        name: "cube",
        dtype: "FP32",
        shape: &[2, 3, 4],
        data: &vec![0u8; 96],
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.shape, &[2, 3, 4]);
    assert_eq!(tensor.numel(), 24);

    // 4D tensor (like conv weights)
    let file = create_safetensors(&[TensorDef {
        name: "conv",
        dtype: "FP16",
        shape: &[64, 32, 3, 3],
        data: &vec![0u8; 36864],
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.shape, &[64, 32, 3, 3]);
    assert_eq!(tensor.numel(), 18432);
}

#[test]
fn test_scalar_tensor() {
    let file = create_safetensors(&[TensorDef {
        name: "scalar",
        dtype: "FP32",
        shape: &[],
        data: &vec![0u8; 4],
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.shape, &[] as &[usize]);
    assert_eq!(tensor.numel(), 1); // Product of empty shape is 1
}

#[test]
fn test_tensor_size_bytes() {
    let file = create_safetensors(&[TensorDef {
        name: "tensor",
        dtype: "FP32",
        shape: &[2, 2],
        data: &vec![0u8; 16],
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.size_bytes(), 16);
}

#[test]
fn test_registry_get_out_of_bounds() {
    let file = create_safetensors(&[TensorDef {
        name: "tensor",
        dtype: "FP32",
        shape: &[2, 2],
        data: &vec![0u8; 16],
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();

    assert!(registry.get(0).is_some());
    assert!(registry.get(1).is_none());
    assert!(registry.get(100).is_none());
}

#[test]
fn test_safetensors_with_metadata() {
    let file = create_safetensors_with_metadata(
        &[TensorDef {
            name: "tensor",
            dtype: "FP32",
            shape: &[4],
            data: &vec![0u8; 16],
        }],
        Some(json!({"format": "pt"})),
    );

    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();

    // __metadata__ should be skipped, only the actual tensor should be present
    assert_eq!(registry.len(), 1);
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.name, "tensor");
}

#[test]
fn test_large_tensor_count() {
    let data = vec![0u8; 256];
    let tensors: Vec<_> = (0..50)
        .map(|i| TensorDef {
            name: Box::leak(format!("layer_{}", i).into_boxed_str()),
            dtype: "FP32",
            shape: &[8, 8],
            data: &data,
        })
        .collect();

    let file = create_safetensors(&tensors);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();

    assert_eq!(registry.len(), 50);
    assert_eq!(registry.iter().count(), 50);
}

#[test]
fn test_tensor_names_with_dots() {
    let data = vec![0u8; 32];
    let file = create_safetensors(&[
        TensorDef {
            name: "model.layers.0.self_attn.q_proj.weight",
            dtype: "BF16",
            shape: &[4096, 4096],
            data: &data,
        },
        TensorDef {
            name: "model.layers.0.self_attn.k_proj.weight",
            dtype: "BF16",
            shape: &[4096, 4096],
            data: &data,
        },
    ]);

    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();

    assert_eq!(registry.len(), 2);

    let names: Vec<_> = registry.iter().map(|t| t.name).collect();
    assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight"));
    assert!(names.contains(&"model.layers.0.self_attn.k_proj.weight"));
}

#[test]
fn test_float8_dtypes() {
    let data = vec![0u8; 8];

    let file = create_safetensors(&[TensorDef {
        name: "tensor_f8e4m3",
        dtype: "F8_E4M3",
        shape: &[8],
        data: &data,
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.dtype, DType::Float8E4M3);
    assert_eq!(tensor.dtype.bits(), 8);
    assert!(tensor.dtype.is_float());

    let file = create_safetensors(&[TensorDef {
        name: "tensor_f8e5m2",
        dtype: "F8_E5M2",
        shape: &[8],
        data: &data,
    }]);
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    assert_eq!(tensor.dtype, DType::Float8E5M2);
    assert_eq!(tensor.dtype.bits(), 8);
}

#[test]
fn test_roundtrip_tensor_data() {
    use hmll::{Device, LoaderKind, WeightLoader};

    // Create known f32 values: [1.0, 2.0, 3.0, 4.0]
    let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let data: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();

    let file = create_safetensors(&[TensorDef {
        name: "test",
        dtype: "FP32",
        shape: &[2, 2],
        data: &data,
    }]);

    // Parse registry to get tensor offsets
    let source = Source::open(file.path()).unwrap();
    let registry = Registry::from_safetensors(&source).unwrap();
    let tensor = registry.get(0).unwrap();
    let (start, end) = (tensor.start, tensor.end);

    // Open file again for the loader
    let source = Source::open(file.path()).unwrap();
    let sources = [source];
    let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto).unwrap();
    let buffer = loader.fetch(start..end, 0).unwrap();

    // Verify the raw bytes match
    assert_eq!(buffer.as_slice().unwrap(), &data);

    // Parse bytes back to f32 and verify values
    let read_values: Vec<f32> = buffer
        .as_slice()
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, values);
}

#[test]
fn test_sharded_safetensors() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    // Create shard 1
    let shard1_path = temp_dir.path().join("model-00001-of-00002.safetensors");
    let shard1_data = vec![0u8; 80];
    write_safetensors_to_path(
        &shard1_path,
        &[
            TensorDef {
                name: "layer1.weight",
                dtype: "FP32",
                shape: &[4, 4],
                data: &shard1_data[..64],
            },
            TensorDef {
                name: "layer1.bias",
                dtype: "FP32",
                shape: &[4],
                data: &shard1_data[64..],
            },
        ],
    );

    // Create shard 2
    let shard2_path = temp_dir.path().join("model-00002-of-00002.safetensors");
    let shard2_data = vec![0u8; 80];
    write_safetensors_to_path(
        &shard2_path,
        &[
            TensorDef {
                name: "layer2.weight",
                dtype: "FP32",
                shape: &[4, 4],
                data: &shard2_data[..64],
            },
            TensorDef {
                name: "layer2.bias",
                dtype: "FP32",
                shape: &[4],
                data: &shard2_data[64..],
            },
        ],
    );

    // Create index file
    let index_path = temp_dir.path().join("model.safetensors.index.json");
    let index_content = json!({
        "metadata": {"total_size": 160},
        "weight_map": {
            "layer1.weight": "model-00001-of-00002.safetensors",
            "layer1.bias": "model-00001-of-00002.safetensors",
            "layer2.weight": "model-00002-of-00002.safetensors",
            "layer2.bias": "model-00002-of-00002.safetensors"
        }
    });
    std::fs::write(&index_path, index_content.to_string()).unwrap();

    // Open all source files
    let index = Source::open(&index_path).expect("Failed to open index");
    let shard1 = Source::open(&shard1_path).expect("Failed to open shard 1");
    let shard2 = Source::open(&shard2_path).expect("Failed to open shard 2");

    // Parse using high-level API
    let registry =
        Registry::from_sharded_safetensors(&index, &[&shard1, &shard2]).expect("Failed to parse");

    assert_eq!(registry.len(), 4, "Should have 4 tensors total");

    // Verify all tensors are accessible
    let names: Vec<_> = registry.iter().map(|t| t.name.to_string()).collect();

    assert_eq!(names.len(), 4);
    assert!(names.contains(&"layer1.weight".to_string()));
    assert!(names.contains(&"layer1.bias".to_string()));
    assert!(names.contains(&"layer2.weight".to_string()));
    assert!(names.contains(&"layer2.bias".to_string()));

    // Verify source indexes
    for tensor in registry.iter() {
        if tensor.name.starts_with("layer1") {
            assert_eq!(
                tensor.source_index, 0,
                "layer1 tensors should be from file 0"
            );
        } else {
            assert_eq!(
                tensor.source_index, 1,
                "layer2 tensors should be from file 1"
            );
        }
    }
}
