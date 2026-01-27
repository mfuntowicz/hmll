//! Registry for tensor metadata from safetensors files.

use crate::error::{Error, Result};
use crate::{DType, Source};
use std::ffi::CStr;

/// A registry containing tensor metadata parsed from safetensors files.
///
/// This is a safe wrapper around the C `hmll_registry` struct that properly
/// frees allocated memory when dropped.
pub struct Registry {
    inner: hmll_sys::hmll_registry,
}

impl Registry {
    /// Parse a single safetensors file and return a registry of tensor metadata.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hmll::{Source, Registry};
    ///
    /// let source = Source::open("model.safetensors")?;
    /// let registry = Registry::from_safetensors(&source)?;
    ///
    /// for tensor in registry.iter() {
    ///     println!("{}: {}", tensor.name, tensor.dtype);
    /// }
    /// # Ok::<(), hmll::Error>(())
    /// ```
    pub fn from_safetensors(source: &Source) -> Result<Self> {
        let mut ctx: hmll_sys::hmll = unsafe { std::mem::zeroed() };
        let mut inner: hmll_sys::hmll_registry = unsafe { std::mem::zeroed() };

        unsafe {
            hmll_sys::hmll_safetensors_populate_registry(
                &mut ctx,
                &mut inner,
                *source.as_raw(),
                0,
                0,
            );
        }

        Error::check_hmll_error(ctx.error)?;

        if inner.num_tensors == 0 {
            return Err(Error::TableEmpty);
        }

        Ok(Self { inner })
    }

    /// Parse sharded safetensors from an index file and shard sources.
    ///
    /// # Arguments
    ///
    /// * `index` - The `model.safetensors.index.json` file
    /// * `shards` - The shard files in order (e.g., `model-00001-of-00002.safetensors`, ...)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hmll::{Source, Registry};
    ///
    /// let index = Source::open("model.safetensors.index.json")?;
    /// let shard1 = Source::open("model-00001-of-00002.safetensors")?;
    /// let shard2 = Source::open("model-00002-of-00002.safetensors")?;
    ///
    /// let registry = Registry::from_sharded_safetensors(&index, &[&shard1, &shard2])?;
    /// # Ok::<(), hmll::Error>(())
    /// ```
    pub fn from_sharded_safetensors(index: &Source, shards: &[&Source]) -> Result<Self> {
        let mut ctx: hmll_sys::hmll = unsafe { std::mem::zeroed() };
        let mut inner: hmll_sys::hmll_registry = unsafe { std::mem::zeroed() };

        // Parse the index file to get total tensor count and allocate registry
        let num_files = unsafe {
            hmll_sys::hmll_safetensors_index(&mut ctx, &mut inner, *index.as_raw())
        };

        Error::check_hmll_error(ctx.error)?;

        if num_files == 0 {
            return Err(Error::TableEmpty);
        }

        if shards.len() != num_files {
            // Clean up allocated registry before returning error
            unsafe { hmll_sys::hmll_free_registry(&mut inner) };
            return Err(Error::InvalidRange);
        }

        // Populate registry from each shard
        let mut offset = 0;
        for (fid, shard) in shards.iter().enumerate() {
            let count = unsafe {
                hmll_sys::hmll_safetensors_populate_registry(
                    &mut ctx,
                    &mut inner,
                    *shard.as_raw(),
                    fid,
                    offset,
                )
            };

            Error::check_hmll_error(ctx.error)?;
            offset += count;
        }

        if inner.num_tensors == 0 {
            return Err(Error::TableEmpty);
        }

        Ok(Self { inner })
    }

    /// Get the number of tensors in the registry.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.num_tensors
    }

    /// Check if the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.num_tensors == 0
    }

    /// Get tensor info by index.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<TensorInfo<'_>> {
        if index >= self.inner.num_tensors {
            return None;
        }

        unsafe {
            let specs = &*self.inner.tensors.add(index);
            let name_ptr = *self.inner.names.add(index);
            let name = if name_ptr.is_null() {
                ""
            } else {
                CStr::from_ptr(name_ptr).to_str().unwrap_or("")
            };

            // Get source index if available (for sharded models)
            let source_index = if self.inner.indexes.is_null() {
                0
            } else {
                *self.inner.indexes.add(index) as usize
            };

            Some(TensorInfo {
                name,
                dtype: DType::from_raw(specs.dtype),
                shape: &specs.shape[..specs.rank as usize],
                start: specs.start,
                end: specs.end,
                source_index,
            })
        }
    }

    /// Iterate over all tensors in the registry.
    pub fn iter(&self) -> impl Iterator<Item = TensorInfo<'_>> {
        (0..self.len()).filter_map(|i| self.get(i))
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        unsafe {
            hmll_sys::hmll_free_registry(&mut self.inner);
        }
    }
}

/// Information about a single tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo<'a> {
    /// Name of the tensor.
    pub name: &'a str,
    /// Data type of the tensor elements.
    pub dtype: DType,
    /// Shape of the tensor.
    pub shape: &'a [usize],
    /// Start offset in the file (after header).
    pub start: usize,
    /// End offset in the file (after header).
    pub end: usize,
    /// Index of the source file this tensor belongs to (for sharded models).
    pub source_index: usize,
}

impl TensorInfo<'_> {
    /// Get the size in bytes of this tensor's data.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Get the number of elements in this tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
