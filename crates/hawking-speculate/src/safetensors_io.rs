//! Minimal safetensors reader.
//!
//! Used by the Eagle6 trained-head loader. Intentionally dependency-free
//! beyond `memmap2` + `serde_json` + `half` which are already in the
//! workspace — we don't pull the `safetensors` crate because we only
//! need a handful of dtypes (f32 + f16) and don't need write support.
//!
//! Format reminder (safetensors v0.x stable):
//!   bytes 0..8        : little-endian u64 header_len
//!   bytes 8..8+header : utf-8 JSON header (per-tensor entries + `__metadata__`)
//!   bytes 8+header..  : concatenated raw tensor bytes
//!
//! Each tensor entry in the header looks like:
//!   "name": { "dtype": "F32"|"F16"|..., "shape": [d0,d1,...], "data_offsets": [start,end] }
//! where the offsets are relative to the start of the tensor-data section
//! (i.e. relative to byte `8 + header_len` of the file).

use crate::{Error, Result};
use half::f16;
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// A loaded safetensors file. Holds an mmap of the file plus a parsed
/// index of tensor name → (dtype, shape, byte range in the data section).
pub struct SafeTensors {
    mmap: Mmap,
    data_start: usize,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    offset_start: usize, // relative to data_start
    offset_end: usize,
}

impl SafeTensors {
    /// Open and parse the header of a safetensors file. Mmap is held for
    /// the lifetime of the returned struct; tensor reads slice directly
    /// from the mapped pages.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(Error::Io)?;
        let mmap = unsafe { Mmap::map(&f).map_err(Error::Io)? };
        if mmap.len() < 8 {
            return Err(Error::Model(format!(
                "safetensors: file too short ({} bytes) at {}",
                mmap.len(),
                path.display()
            )));
        }
        let hlen = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if 8 + hlen > mmap.len() {
            return Err(Error::Model(format!(
                "safetensors: header_len {hlen} overruns file len {} at {}",
                mmap.len(),
                path.display()
            )));
        }
        let header_bytes = &mmap[8..8 + hlen];
        let header: Value = serde_json::from_slice(header_bytes).map_err(|e| {
            Error::Model(format!(
                "safetensors: header JSON parse failed at {}: {e}",
                path.display()
            ))
        })?;
        let header_obj = header.as_object().ok_or_else(|| {
            Error::Model(format!(
                "safetensors: header is not a JSON object at {}",
                path.display()
            ))
        })?;

        let mut metadata = HashMap::new();
        let mut entries = HashMap::new();
        for (k, v) in header_obj.iter() {
            if k == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (mk, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let obj = v.as_object().ok_or_else(|| {
                Error::Model(format!("safetensors: entry '{k}' is not an object"))
            })?;
            let dtype = obj
                .get("dtype")
                .and_then(|x| x.as_str())
                .ok_or_else(|| Error::Model(format!("safetensors: entry '{k}' missing dtype")))?
                .to_string();
            let shape = obj
                .get("shape")
                .and_then(|x| x.as_array())
                .ok_or_else(|| Error::Model(format!("safetensors: entry '{k}' missing shape")))?
                .iter()
                .map(|d| {
                    d.as_u64().map(|u| u as usize).ok_or_else(|| {
                        Error::Model(format!("safetensors: entry '{k}' shape dim not u64"))
                    })
                })
                .collect::<Result<Vec<usize>>>()?;
            let offsets = obj
                .get("data_offsets")
                .and_then(|x| x.as_array())
                .ok_or_else(|| {
                    Error::Model(format!("safetensors: entry '{k}' missing data_offsets"))
                })?;
            if offsets.len() != 2 {
                return Err(Error::Model(format!(
                    "safetensors: entry '{k}' data_offsets len != 2"
                )));
            }
            let a = offsets[0]
                .as_u64()
                .ok_or_else(|| Error::Model(format!("safetensors: '{k}' offset[0] not u64")))?
                as usize;
            let b = offsets[1]
                .as_u64()
                .ok_or_else(|| Error::Model(format!("safetensors: '{k}' offset[1] not u64")))?
                as usize;
            entries.insert(
                k.clone(),
                Entry {
                    dtype,
                    shape,
                    offset_start: a,
                    offset_end: b,
                },
            );
        }

        Ok(Self {
            mmap,
            data_start: 8 + hlen,
            entries,
            metadata,
        })
    }

    /// Metadata key/value pairs from `__metadata__`.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// True if a tensor with this name is present.
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Names of all tensors. Order is unstable (HashMap); sort if needed.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Shape of a named tensor. Returns Err if missing.
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.entries
            .get(name)
            .map(|e| e.shape.as_slice())
            .ok_or_else(|| Error::Model(format!("safetensors: missing tensor '{name}'")))
    }

    /// Read a tensor as f32, validating dtype is F32 and shape matches.
    pub fn read_f32(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let raw = self.slice(name, "F32", expected_shape, 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Read a tensor as f16, validating dtype is F16 and shape matches.
    pub fn read_f16(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f16>> {
        let raw = self.slice(name, "F16", expected_shape, 2)?;
        Ok(raw
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    fn slice(
        &self,
        name: &str,
        expected_dtype: &str,
        expected_shape: &[usize],
        bytes_per_elem: usize,
    ) -> Result<&[u8]> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Model(format!("safetensors: missing tensor '{name}'")))?;
        if e.dtype != expected_dtype {
            return Err(Error::Model(format!(
                "safetensors: '{name}' dtype is {} expected {expected_dtype}",
                e.dtype
            )));
        }
        if e.shape != expected_shape {
            return Err(Error::Model(format!(
                "safetensors: '{name}' shape is {:?} expected {:?}",
                e.shape, expected_shape
            )));
        }
        let n = expected_shape.iter().product::<usize>();
        let expected_bytes = n * bytes_per_elem;
        let span = e.offset_end - e.offset_start;
        if span != expected_bytes {
            return Err(Error::Model(format!(
                "safetensors: '{name}' byte span {span} != expected {expected_bytes}"
            )));
        }
        let abs_start = self.data_start + e.offset_start;
        let abs_end = self.data_start + e.offset_end;
        if abs_end > self.mmap.len() {
            return Err(Error::Model(format!(
                "safetensors: '{name}' overruns file (end={abs_end} len={})",
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[abs_start..abs_end])
    }
}
