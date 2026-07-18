//! Minimal safetensors reader (for the real GPT-OSS-120B source). Parses the 8-byte little-endian
//! header length, the JSON header (tensor name -> {dtype, shape, data_offsets}), and returns byte-slice
//! VIEWS into an mmap of one shard. No dense copy; bounded reads only. BF16 -> f32 primitive included.

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub begin: usize, // absolute file offset of data
    pub end: usize,
}

pub struct SafeTensors {
    map: Mmap,
    pub tensors: HashMap<String, TensorEntry>,
    pub data_start: usize,
    pub file_bytes: usize,
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl SafeTensors {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 8 {
            return Err(Error::Model("safetensors: too small".into()));
        }
        let hlen = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
        if 8 + hlen > map.len() {
            return Err(Error::Model("safetensors: header past eof".into()));
        }
        let hdr: serde_json::Value = serde_json::from_slice(&map[8..8 + hlen])
            .map_err(|e| Error::Model(format!("safetensors header json: {e}")))?;
        let data_start = 8 + hlen;
        let mut tensors = HashMap::new();
        if let Some(obj) = hdr.as_object() {
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = v["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
                let off = v["data_offsets"].as_array().ok_or_else(|| Error::Model("no data_offsets".into()))?;
                let begin = data_start + off[0].as_u64().unwrap_or(0) as usize;
                let end = data_start + off[1].as_u64().unwrap_or(0) as usize;
                tensors.insert(name.clone(), TensorEntry { dtype, shape, begin, end });
            }
        }
        let file_bytes = map.len();
        Ok(SafeTensors { map, tensors, data_start, file_bytes })
    }

    pub fn get(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors.get(name).ok_or_else(|| Error::Model(format!("missing tensor {name}")))
    }
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.get(name)?;
        Ok(&self.map[t.begin..t.end])
    }
    /// Read a BF16 tensor as f32.
    pub fn bf16_f32(&self, name: &str) -> Result<Vec<f32>> {
        let b = self.bytes(name)?;
        Ok(b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
    }
}
