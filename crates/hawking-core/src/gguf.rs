//! GGUF v3 reader. mmap-backed; tensor data is never copied — the
//! Metal device reads it via `MTLBuffer.newBufferWithBytesNoCopy:`.
//!
//! Format reference:
//!   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! Supports v2 and v3 (the wire format is identical for our purposes;
//! v3 just adds new metadata keys).

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

/// One entry in the tensor index. The actual bytes live in the mmap;
/// `data_offset` is the absolute file offset to the first byte of this
/// tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub data_offset: u64,
    pub byte_size: u64,
}

/// Subset of GGML tensor types we care about for v0.1. Numeric values
/// match the upstream `ggml_type` enum.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => return Err(Error::Gguf(format!("unknown ggml_type {other}"))),
        })
    }

    /// (block_size, bytes_per_block). For non-quantized types the
    /// block size is 1 element and the bytes are the element size.
    pub fn block_layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::BF16 => (1, 2),
            Self::F64 => (1, 8),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            // K-quants: 256-element super-blocks.
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }
}

/// A loaded metadata value. GGUF metadata is heterogenous; we keep
/// the raw decoded form so the model layer can pull out exactly what
/// it needs.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_str_array(&self) -> Option<Vec<&str>> {
        match self {
            Self::Array(a) => a.iter().map(|v| v.as_str()).collect(),
            _ => None,
        }
    }
    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            Self::Array(a) => a.iter().map(|v| v.as_u32()).collect(),
            _ => None,
        }
    }
    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            Self::Array(a) => a.iter().map(|v| v.as_f32()).collect(),
            _ => None,
        }
    }
}

/// The mmap-backed GGUF file. Cheap to clone the `Arc`-wrapped inner.
pub struct GgufFile {
    pub mmap: Mmap,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Order in which tensors appeared in the file; useful for
    /// debugging the loader.
    pub tensor_order: Vec<String>,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = File::open(path.as_ref())?;
        // Safety: we treat the mmap as read-only for the lifetime of
        // `GgufFile`. The OS may still invalidate it if the file is
        // truncated under us; that's caller-error.
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() >= 8 && &mmap[..8] == b"GRAVITY\0" {
            Self::from_gravity_mmap(mmap)
        } else {
            Self::from_mmap(mmap)
        }
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let mut p = Cursor::new(&mmap);
        let magic = p.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::Gguf(format!(
                "bad magic 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}"
            )));
        }
        let version = p.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported gguf version {version}")));
        }
        let tensor_count = p.u64()?;
        let kv_count = p.u64()?;

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = p.gguf_string()?;
            let val = p.gguf_value()?;
            metadata.insert(key, val);
        }

        // Tensor index.
        let mut infos: Vec<(String, Vec<u64>, GgmlType, u64)> =
            Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = p.gguf_string()?;
            let n_dims = p.u32()? as usize;
            if n_dims > 8 {
                return Err(Error::Gguf(format!("absurd n_dims {n_dims} for {name}")));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(p.u64()?);
            }
            let dtype = GgmlType::from_u32(p.u32()?)?;
            let local_offset = p.u64()?;
            infos.push((name, dims, dtype, local_offset));
        }

        // Align tensor data start to `general.alignment` (default 32).
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let header_end = p.pos as u64;
        let data_base = align_up(header_end, alignment);

        let mut tensors = HashMap::with_capacity(infos.len());
        let mut order = Vec::with_capacity(infos.len());
        for (name, dims, dtype, local_offset) in infos {
            let n_elems: u64 = dims.iter().product();
            let (block_size, bytes_per_block) = dtype.block_layout();
            if n_elems % block_size != 0 {
                return Err(Error::Gguf(format!(
                    "tensor {name}: {n_elems} elems not divisible by block {block_size}"
                )));
            }
            let byte_size = (n_elems / block_size) * bytes_per_block;
            let abs = data_base + local_offset;
            if abs + byte_size > mmap.len() as u64 {
                return Err(Error::Gguf(format!(
                    "tensor {name}: end {} past mmap len {}",
                    abs + byte_size,
                    mmap.len()
                )));
            }
            order.push(name.clone());
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    dtype,
                    data_offset: abs,
                    byte_size,
                },
            );
        }

        Ok(Self {
            mmap,
            version,
            tensor_count,
            metadata,
            tensors,
            tensor_order: order,
        })
    }

    /// Build the narrow GGUF view needed by the existing Mixtral adapter from
    /// a source-preserving Gravity shard.  Gravity descriptors carry runtime
    /// shapes and raw GGML codec names; reversing the two-dimensional shape
    /// here restores the GGUF convention without copying or dequantizing any
    /// payload. The returned mmap remains the Gravity file itself, so Metal
    /// reads the sealed artifact body directly.
    fn from_gravity_mmap(mmap: Mmap) -> Result<Self> {
        const MAGIC: &[u8; 8] = b"GRAVITY\0";
        if mmap.len() < 20 || &mmap[..8] != MAGIC {
            return Err(Error::Gguf("bad Gravity magic".into()));
        }
        let format_version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if format_version > 1 {
            return Err(Error::Gguf(format!(
                "unsupported Gravity format version {format_version}"
            )));
        }
        let header_len = u64::from_le_bytes(mmap[12..20].try_into().unwrap());
        let header_end = 20u64
            .checked_add(header_len)
            .ok_or_else(|| Error::Gguf("Gravity header length overflow".into()))?;
        if header_end > mmap.len() as u64 {
            return Err(Error::Gguf("Gravity header extends past file".into()));
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[20..header_end as usize])
            .map_err(|e| Error::Gguf(format!("Gravity header JSON: {e}")))?;
        let metadata_obj = header
            .get("gguf_metadata")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| Error::Gguf("Gravity shard has no gguf_metadata".into()))?;
        let mut metadata = HashMap::with_capacity(metadata_obj.len());
        for (key, value) in metadata_obj {
            metadata.insert(key.clone(), meta_value_from_json(value)?);
        }
        let descriptors = header
            .get("tensors")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| Error::Gguf("Gravity shard has no tensor descriptors".into()))?;
        let mut tensors = HashMap::with_capacity(descriptors.len());
        let mut tensor_order = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            let name = descriptor
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| Error::Gguf("Gravity tensor has no name".into()))?
                .to_string();
            let shape = descriptor
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::Gguf(format!("Gravity tensor {name} has no shape")))?;
            let runtime_shape: Vec<u64> = shape
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        Error::Gguf(format!("Gravity tensor {name} shape is not unsigned"))
                    })
                })
                .collect::<Result<_>>()?;
            if runtime_shape.is_empty() || runtime_shape.len() > 3 {
                return Err(Error::Gguf(format!(
                    "Gravity tensor {name} has unsupported rank {}",
                    runtime_shape.len()
                )));
            }
            let dims = if runtime_shape.len() <= 1 {
                runtime_shape.clone()
            } else {
                runtime_shape.iter().rev().copied().collect()
            };
            let codec = descriptor
                .get("codec")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| Error::Gguf(format!("Gravity tensor {name} has no codec")))?;
            let dtype = gravity_codec_to_ggml(codec).ok_or_else(|| {
                Error::Gguf(format!(
                    "Gravity tensor {name} has unsupported codec {codec:?}"
                ))
            })?;
            let bytes = descriptor
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| Error::Gguf(format!("Gravity tensor {name} has no byte length")))?;
            let offset = descriptor
                .get("offset")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| Error::Gguf(format!("Gravity tensor {name} has no offset")))?;
            let absolute = header_end
                .checked_add(offset)
                .ok_or_else(|| Error::Gguf(format!("Gravity tensor {name} offset overflow")))?;
            let end = absolute
                .checked_add(bytes)
                .ok_or_else(|| Error::Gguf(format!("Gravity tensor {name} size overflow")))?;
            if end > mmap.len() as u64 {
                return Err(Error::Gguf(format!(
                    "Gravity tensor {name} ends at {end}, past file {}",
                    mmap.len()
                )));
            }
            let elements = dims.iter().try_fold(1u64, |acc, dim| {
                acc.checked_mul(*dim).ok_or_else(|| {
                    Error::Gguf(format!("Gravity tensor {name} element count overflow"))
                })
            })?;
            let (block_size, bytes_per_block) = dtype.block_layout();
            if elements % block_size != 0 || elements / block_size * bytes_per_block != bytes {
                return Err(Error::Gguf(format!(
                    "Gravity tensor {name} geometry mismatch: {elements} elems, {bytes} bytes, codec {codec}"
                )));
            }
            if tensors
                .insert(
                    name.clone(),
                    TensorInfo {
                        name: name.clone(),
                        dims,
                        dtype,
                        data_offset: absolute,
                        byte_size: bytes,
                    },
                )
                .is_some()
            {
                return Err(Error::Gguf(format!("duplicate Gravity tensor {name}")));
            }
            tensor_order.push(name);
        }
        Ok(Self {
            mmap,
            version: 3,
            tensor_count: tensors.len() as u64,
            metadata,
            tensors,
            tensor_order,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    pub fn name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.get(name)?;
        let start = t.data_offset as usize;
        let end = start + t.byte_size as usize;
        Some(&self.mmap[start..end])
    }
}

fn gravity_codec_to_ggml(codec: &str) -> Option<GgmlType> {
    Some(match codec {
        "native.f32" => GgmlType::F32,
        "native.f16" => GgmlType::F16,
        "native.bf16" => GgmlType::BF16,
        "ggml.q3_k" => GgmlType::Q3_K,
        "ggml.q4_k" => GgmlType::Q4_K,
        "ggml.q5_k" => GgmlType::Q5_K,
        "ggml.q6_k" => GgmlType::Q6_K,
        "ggml.q8_0" => GgmlType::Q8_0,
        "ggml.q5_0" => GgmlType::Q5_0,
        _ => return None,
    })
}

fn meta_value_from_json(value: &serde_json::Value) -> Result<MetaValue> {
    Ok(match value {
        serde_json::Value::Bool(v) => MetaValue::Bool(*v),
        serde_json::Value::String(v) => MetaValue::String(v.clone()),
        serde_json::Value::Number(v) => {
            if let Some(v) = v.as_u64() {
                if v <= u32::MAX as u64 {
                    MetaValue::U32(v as u32)
                } else {
                    MetaValue::U64(v)
                }
            } else if let Some(v) = v.as_i64() {
                if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
                    MetaValue::I32(v as i32)
                } else {
                    MetaValue::I64(v)
                }
            } else if let Some(v) = v.as_f64() {
                MetaValue::F64(v)
            } else {
                return Err(Error::Gguf(
                    "unsupported JSON number in Gravity metadata".into(),
                ));
            }
        }
        serde_json::Value::Array(values) => MetaValue::Array(
            values
                .iter()
                .map(meta_value_from_json)
                .collect::<Result<_>>()?,
        ),
        serde_json::Value::Null => {
            return Err(Error::Gguf("null is not a GGUF metadata value".into()))
        }
        serde_json::Value::Object(_) => {
            return Err(Error::Gguf("object is not a GGUF metadata value".into()))
        }
    })
}

#[inline]
fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

// ---- Wire-format primitive types --------------------------------------

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum GgufKind {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufKind {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(Error::Gguf(format!("unknown gguf kind {other}"))),
        })
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(Error::Gguf("unexpected end of file".into()));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.read(1)?[0])
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.read(1)?[0] as i8)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    fn gguf_string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.read(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn gguf_value(&mut self) -> Result<MetaValue> {
        let kind = GgufKind::from_u32(self.u32()?)?;
        self.gguf_value_of(kind)
    }

    fn gguf_value_of(&mut self, kind: GgufKind) -> Result<MetaValue> {
        Ok(match kind {
            GgufKind::U8 => MetaValue::U8(self.u8()?),
            GgufKind::I8 => MetaValue::I8(self.i8()?),
            GgufKind::U16 => MetaValue::U16(self.u16()?),
            GgufKind::I16 => MetaValue::I16(self.i16()?),
            GgufKind::U32 => MetaValue::U32(self.u32()?),
            GgufKind::I32 => MetaValue::I32(self.i32()?),
            GgufKind::U64 => MetaValue::U64(self.u64()?),
            GgufKind::I64 => MetaValue::I64(self.i64()?),
            GgufKind::F32 => MetaValue::F32(self.f32()?),
            GgufKind::F64 => MetaValue::F64(self.f64()?),
            GgufKind::Bool => MetaValue::Bool(self.bool()?),
            GgufKind::String => MetaValue::String(self.gguf_string()?),
            GgufKind::Array => {
                let inner = GgufKind::from_u32(self.u32()?)?;
                let n = self.u64()? as usize;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    out.push(self.gguf_value_of(inner)?);
                }
                MetaValue::Array(out)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
    #[test]
    fn block_layout_q4k_is_144() {
        let (bs, bb) = GgmlType::Q4_K.block_layout();
        assert_eq!(bs, 256);
        assert_eq!(bb, 144);
    }
}
