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
    /// Parse a safetensors file and return a registry of tensor metadata.
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
    ///     println!("{}: {:?}", tensor.name, tensor.dtype);
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

        // Check for errors
        Error::check_hmll_error(ctx.error)?;

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

            Some(TensorInfo {
                name,
                dtype: DType::from_raw(specs.dtype),
                shape: &specs.shape[..specs.rank as usize],
                start: specs.start,
                end: specs.end,
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
