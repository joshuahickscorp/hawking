//! A memory-mapped GGUF v2/v3 view. Unlike Candidate B (which reads the file into a Vec and then
//! dequantizes every weight to f32), Candidate C mmaps the file and exposes tensors as byte-slice
//! VIEWS into the mapping — the compressed representation is never copied or expanded. Little-endian.

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Other(u32),
}
impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
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
            o => Self::Other(o),
        }
    }
    /// (block_size in elements, bytes per block)
    pub fn block_layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::Other(_) => (1, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
    Other,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub data_offset: u64,
    pub byte_size: u64,
}

pub struct GgufFile {
    map: Mmap,
    pub mapped_bytes: usize,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, TensorInfo>,
}

#[inline]
fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(Error::Gguf("unexpected eof".into()));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.read(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }
    fn gstr(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.read(len)?).into_owned())
    }
    fn value(&mut self) -> Result<MetaValue> {
        let k = self.u32()?;
        self.value_of(k)
    }
    fn value_of(&mut self, kind: u32) -> Result<MetaValue> {
        Ok(match kind {
            0 => MetaValue::U32(self.u8()? as u32),
            1 => MetaValue::I32(self.read(1)?[0] as i8 as i32),
            2 => MetaValue::U32(u16::from_le_bytes(self.read(2)?.try_into().unwrap()) as u32),
            3 => MetaValue::I32(i16::from_le_bytes(self.read(2)?.try_into().unwrap()) as i32),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(i32::from_le_bytes(self.read(4)?.try_into().unwrap())),
            6 => MetaValue::F32(f32::from_le_bytes(self.read(4)?.try_into().unwrap())),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::String(self.gstr()?),
            9 => {
                let inner = self.u32()?;
                let n = self.u64()? as usize;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    out.push(self.value_of(inner)?);
                }
                MetaValue::Array(out)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => {
                self.read(8)?;
                MetaValue::Other
            }
            12 => {
                self.read(8)?;
                MetaValue::Other
            }
            o => return Err(Error::Gguf(format!("unknown gguf kind {o}"))),
        })
    }
}

impl GgufFile {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // Safety: the file is read-only for the lifetime of the mapping.
        let map = unsafe { Mmap::map(&file)? };
        let mapped_bytes = map.len();
        let (metadata, tensors) = Self::parse_header(&map)?;
        Ok(GgufFile { map, mapped_bytes, metadata, tensors })
    }

    fn parse_header(data: &[u8]) -> Result<(HashMap<String, MetaValue>, HashMap<String, TensorInfo>)> {
        let mut p = Cursor { buf: data, pos: 0 };
        if p.u32()? != GGUF_MAGIC {
            return Err(Error::Gguf("bad magic".into()));
        }
        let version = p.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported version {version}")));
        }
        let tensor_count = p.u64()?;
        let kv_count = p.u64()?;

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = p.gstr()?;
            let val = p.value()?;
            metadata.insert(key, val);
        }

        let mut infos: Vec<(String, Vec<u64>, GgmlType, u64)> = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = p.gstr()?;
            let n_dims = p.u32()? as usize;
            if n_dims > 8 {
                return Err(Error::Gguf("absurd n_dims".into()));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(p.u64()?);
            }
            let dtype = GgmlType::from_u32(p.u32()?);
            let local = p.u64()?;
            infos.push((name, dims, dtype, local));
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetaValue::U32(x)) => *x as u64,
            _ => 32,
        };
        let data_base = align_up(p.pos as u64, alignment);

        let mut tensors = HashMap::with_capacity(infos.len());
        for (name, dims, dtype, local) in infos {
            let n_elems: u64 = dims.iter().product();
            let (bs, bb) = dtype.block_layout();
            if bs == 0 || n_elems % bs != 0 {
                return Err(Error::Gguf(format!("{name}: {n_elems} not divisible by block {bs}")));
            }
            let byte_size = (n_elems / bs) * bb;
            let abs = data_base + local;
            if abs + byte_size > data.len() as u64 {
                return Err(Error::Gguf(format!("{name}: data past end")));
            }
            tensors.insert(name.clone(), TensorInfo { name, dims, dtype, data_offset: abs, byte_size });
        }
        Ok((metadata, tensors))
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::Gguf(format!("missing tensor {name}")))
    }
    /// A byte-slice VIEW of the tensor's compressed data, directly into the mmap (no copy).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor(name)?;
        Ok(&self.map[t.data_offset as usize..(t.data_offset + t.byte_size) as usize])
    }
    pub fn bytes_at(&self, off: u64, len: u64) -> &[u8] {
        &self.map[off as usize..(off + len) as usize]
    }

    pub fn meta_u32(&self, key: &str) -> Result<u32> {
        match self.metadata.get(key) {
            Some(MetaValue::U32(x)) => Ok(*x),
            Some(MetaValue::U64(x)) => Ok(*x as u32),
            _ => Err(Error::Gguf(format!("missing u32 metadata {key}"))),
        }
    }
    pub fn meta_f32(&self, key: &str) -> Result<f32> {
        match self.metadata.get(key) {
            Some(MetaValue::F32(x)) => Ok(*x),
            _ => Err(Error::Gguf(format!("missing f32 metadata {key}"))),
        }
    }
    pub fn meta_str(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(MetaValue::String(s)) => Ok(s.as_str()),
            _ => Err(Error::Gguf(format!("missing string metadata {key}"))),
        }
    }
    pub fn meta_str_array(&self, key: &str) -> Result<Vec<&str>> {
        match self.metadata.get(key) {
            Some(MetaValue::Array(a)) => Ok(a
                .iter()
                .map(|v| match v {
                    MetaValue::String(s) => s.as_str(),
                    _ => "",
                })
                .collect()),
            _ => Err(Error::Gguf(format!("missing array metadata {key}"))),
        }
    }
}
