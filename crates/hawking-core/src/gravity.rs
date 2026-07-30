//! Runtime + Metal PQ; container/PQ/AAP in artifact.
pub use crate::artifact::{
    activation_aware_sections, parse_activation_aware_header, parse_pq_header, pq_matvec,
    pq_matvec_f64_authority, pq_row, pq_sections, widen_native, ActivationAwareHeader,
    ActivationAwareSide, ActivationAwareTensor, GravityShard, PqHeader, PqTensor, TensorDescriptor,
};
use crate::{Error, Result};
use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
pub enum Tensor {
    Pq(PqTensor),
    ActivationAware(ActivationAwareTensor),
    Dense(Vec<f32>),
}
pub(super) fn matvec_dense(w: &[f32], x: &[f32], name: &str) -> Result<Vec<f32>> {
    if x.is_empty() || w.len() % x.len() != 0 {
        return Err(Error::Gravity(format!(
            "tensor {name:?}: {} values not {}-wide rows",
            w.len(),
            x.len()
        )));
    }
    Ok(w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect())
}
fn bf16_host_geometry(weight_le: &[u8], cols: usize, x: &[f32]) -> Result<()> {
    if x.len() != cols {
        return Err(Error::Gravity(format!("bf16 host x {} != {cols}", x.len())));
    }
    if cols == 0 || weight_le.len() % (cols * 2) != 0 {
        return Err(Error::Gravity(format!(
            "bf16 host: payload {} B not {cols}-wide bf16 rows",
            weight_le.len()
        )));
    }
    Ok(())
}
pub fn matvec_bf16_host(weight_le: &[u8], cols: usize, x: &[f32]) -> Result<Vec<f32>> {
    bf16_host_geometry(weight_le, cols, x)?;
    let w = widen_native("native.bf16", weight_le)?;
    matvec_dense(&w, x, "lm_head.weight")
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBf16Accumulation {
    Sequential,
    Neumaier,
    NeumaierCompensatedProduct,
}
impl NativeBf16Accumulation {
    pub const ALL: [Self; 3] = [
        Self::Sequential,
        Self::Neumaier,
        Self::NeumaierCompensatedProduct,
    ];
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Neumaier => "neumaier",
            Self::NeumaierCompensatedProduct => "neumaier_compensated_product",
        }
    }
    #[cfg(target_os = "macos")]
    pub const fn metal_kernel(self) -> &'static str {
        match self {
            Self::Sequential => "gemv_native_bf16_seq",
            Self::Neumaier => "gemv_native_bf16_neumaier",
            Self::NeumaierCompensatedProduct => "gemv_native_bf16_neumaier_compensated_product",
        }
    }
}
pub fn matvec_bf16_host_accumulation(
    weight_le: &[u8],
    cols: usize,
    x: &[f32],
    accumulation: NativeBf16Accumulation,
) -> Result<Vec<f32>> {
    if accumulation == NativeBf16Accumulation::Sequential {
        return matvec_bf16_host(weight_le, cols, x);
    }
    bf16_host_geometry(weight_le, cols, x)?;
    let weights = widen_native("native.bf16", weight_le)?;
    Ok(weights
        .chunks_exact(cols)
        .map(|row| {
            let mut sum = 0.0f32;
            let mut correction = 0.0f32;
            for (&weight, &activation) in row.iter().zip(x) {
                let product = weight * activation;
                let product_residual =
                    if accumulation == NativeBf16Accumulation::NeumaierCompensatedProduct {
                        weight.mul_add(activation, -product)
                    } else {
                        0.0
                    };
                let next = sum + product;
                let addition_residual = if sum.abs() >= product.abs() {
                    let delta = sum - next;
                    delta + product
                } else {
                    let delta = product - next;
                    delta + sum
                };
                correction += addition_residual;
                correction += product_residual;
                sum = next;
            }
            sum + correction
        })
        .collect())
}
fn row_dense(w: &[f32], index: usize, cols: usize, name: &str) -> Result<Vec<f32>> {
    let start = index * cols;
    if start + cols > w.len() {
        return Err(Error::Gravity(format!(
            "tensor {name:?}: row {index} out of range"
        )));
    }
    Ok(w[start..start + cols].to_vec())
}

// Activation-aware pack shard reader. Sole product consumer is LazyShard below;
// type and methods are module-private to gravity. Public AAP codec stays in artifact::aap.
const AAP_SCHEMA: &str = "hawking.glm52.activation_aware_pack.v1";
const AAP_MAX_INDEX_BYTES: u64 = 256 * 1024 * 1024;
const AAP_PASS_MAGIC: &[u8; 8] = b"GLM52PT0";
const AAP_BASIS_MAGIC: &[u8; 8] = b"GLM52BAS";
const AAP_HEADER_LEN: usize = 64;
// Local layout helpers for this private reader only. Same semantics as artifact
// macro_rules (sibling module cannot see those without #[macro_use] / widening).
// Macro form preserves official function count (fn helpers would +3).
macro_rules! aap_checked_end {
    ($offset:expr, $bytes:expr, $limit:expr, $label:expr) => {{
        let offset: u64 = $offset;
        let bytes: u64 = $bytes;
        let limit: u64 = $limit;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Gravity(format!("{}: ovf", $label)))?;
        if end > limit {
            return Err(Error::Gravity(format!("{}: past {limit}", $label)));
        }
        end
    }};
}
macro_rules! aap_section_end {
    ($base:expr, $count:expr, $unit:expr, $label:expr) => {{
        let base: usize = $base;
        let count: usize = $count;
        let unit: usize = $unit;
        let add = count
            .checked_mul(unit)
            .ok_or_else(|| Error::Gravity(format!("{}: size ovf", $label)))?;
        base.checked_add(add)
            .ok_or_else(|| Error::Gravity(format!("{}: end ovf", $label)))?
    }};
}
macro_rules! aap_require_magic {
    ($b:expr, $magic:expr, $label:expr) => {{
        let b: &[u8] = $b;
        let magic: &[u8] = $magic;
        if b.len() < magic.len() || &b[..magic.len()] != magic {
            Err(Error::Gravity(format!(
                "bad {} magic {:?}, expected {magic:?}",
                $label,
                b.get(..magic.len()).unwrap_or(&[])
            )))
        } else {
            Ok(())
        }
    }};
}
#[derive(Debug, Clone, Deserialize)]
struct AapBasisDescriptor {
    basis_layer: u16,
    rank: u32,
    offset: u64,
    bytes: u64,
}
#[derive(Debug, Clone, Deserialize)]
struct AapTensorDescriptor {
    name: String,
    disposition: String,
    #[serde(default)]
    dtype: Option<String>,
    offset: u64,
    bytes: u64,
    shape: Vec<u64>,
}
#[derive(Deserialize)]
struct AapIndex {
    schema: String,
    bases: Vec<AapBasisDescriptor>,
    tensors: Vec<AapTensorDescriptor>,
}
struct ActivationAwareShard {
    mmap: Mmap,
    body_offset: u64,
    tensors: HashMap<String, AapTensorDescriptor>,
    bases: HashMap<u16, AapBasisDescriptor>,
    source_dtypes: HashMap<String, String>,
}
impl ActivationAwareShard {
    fn open(
        path: &Path,
        source_dtypes: &HashMap<String, String>,
        expected_sha256: Option<&str>,
        verify_hash: bool,
    ) -> Result<Self> {
        let file = File::open(path)?;
        // Safety: same read-only/no-truncation contract as GravityShard.
        let mmap = unsafe { Mmap::map(&file)? };
        if verify_hash {
            let expected = expected_sha256
                .ok_or_else(|| Error::Gravity(format!("{}: aap no shard sha", path.display())))?;
            let observed = format!("{:x}", Sha256::digest(&mmap[..]));
            if observed != expected {
                return Err(Error::Gravity(format!(
                    "{}: aap sha want {expected}, got {observed}",
                    path.display()
                )));
            }
        }
        let source_dtypes = source_dtypes.clone();
        if mmap.len() < 8 {
            return Err(Error::Gravity("aap shard too short".into()));
        }
        let index_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if index_len == 0 || index_len > AAP_MAX_INDEX_BYTES {
            return Err(Error::Gravity(format!("aap invalid index_len {index_len}")));
        }
        let body_offset = aap_checked_end!(8, index_len, mmap.len() as u64, "aap-index");
        let index: AapIndex = serde_json::from_slice(&mmap[8..body_offset as usize])
            .map_err(|error| Error::Gravity(format!("aap index parse: {error}")))?;
        if index.schema != AAP_SCHEMA {
            return Err(Error::Gravity(format!(
                "aap schema {:?}, expected {AAP_SCHEMA:?}",
                index.schema
            )));
        }
        let body_bytes = mmap.len() as u64 - body_offset;
        let mut spans: Vec<(u64, u64)> = Vec::new();
        let mut bases = HashMap::with_capacity(index.bases.len());
        let mut tensors = HashMap::with_capacity(index.tensors.len());
        for basis in index.bases {
            let end = aap_checked_end!(basis.offset, basis.bytes, body_bytes, "aap-basis");
            if basis.bytes < AAP_HEADER_LEN as u64 {
                return Err(Error::Gravity(format!(
                    "basis span short {}",
                    basis.basis_layer
                )));
            }
            let off = basis.offset;
            if bases.insert(basis.basis_layer, basis).is_some() {
                return Err(Error::Gravity("dup basis".into()));
            }
            spans.push((off, end));
        }
        for tensor in index.tensors {
            let end = aap_checked_end!(tensor.offset, tensor.bytes, body_bytes, "aap-tensor");
            if tensor.bytes < AAP_HEADER_LEN as u64 {
                return Err(Error::Gravity(format!("aap span short {:?}", tensor.name)));
            }
            if tensor.shape.is_empty() || tensor.shape.len() > 2 {
                return Err(Error::Gravity(format!(
                    "aap shape {:?} {:?}",
                    tensor.name, tensor.shape
                )));
            }
            let off = tensor.offset;
            if tensors.insert(tensor.name.clone(), tensor).is_some() {
                return Err(Error::Gravity("dup aap tensor".into()));
            }
            spans.push((off, end));
        }
        spans.sort_by_key(|span| span.0);
        if spans.first().map(|s| s.0) != Some(0) {
            return Err(Error::Gravity("aap body not at 0".into()));
        }
        for pair in spans.windows(2) {
            if pair[0].1 != pair[1].0 {
                return Err(Error::Gravity(format!(
                    "aap payloads not contiguous ({} != {})",
                    pair[0].1, pair[1].0
                )));
            }
        }
        if spans.last().map(|s| s.1) != Some(body_bytes) {
            return Err(Error::Gravity(format!(
                "aap body ends {}, phys ends {body_bytes}",
                spans.last().map(|s| s.1).unwrap_or(0)
            )));
        }
        Ok(Self {
            mmap,
            body_offset,
            tensors,
            bases,
            source_dtypes,
        })
    }
    fn descriptor(&self, name: &str) -> Result<&AapTensorDescriptor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Gravity(format!("no aap tensor {name:?}")))
    }
    fn span(&self, offset: u64, bytes: u64, label: &str) -> Result<&[u8]> {
        let start = self
            .body_offset
            .checked_add(offset)
            .ok_or_else(|| Error::Gravity(format!("{label}: offset overflow")))?;
        let end = aap_checked_end!(start, bytes, self.mmap.len() as u64, label);
        Ok(&self.mmap[start as usize..end as usize])
    }
    fn codec_and_shape(&self, name: &str) -> Result<(String, Vec<u64>)> {
        let descriptor = self.descriptor(name)?;
        let codec = match descriptor.disposition.as_str() {
            "activation_aware" => "activation-aware.f16".to_string(),
            "pass_through" => {
                let source_dtype = descriptor
                    .dtype
                    .as_deref()
                    .or_else(|| self.source_dtypes.get(name).map(String::as_str))
                    .ok_or_else(|| Error::Gravity(format!("aap pass {name:?} no dtype")))?;
                match source_dtype {
                    "BF16" | "BFLOAT16" => "native.bf16".to_string(),
                    "F16" | "FLOAT16" => "native.f16".to_string(),
                    "F32" | "FLOAT32" => "native.f32".to_string(),
                    other => {
                        return Err(Error::Gravity(format!(
                            "activation-aware tensor {name:?} bad dtype {other:?}"
                        )))
                    }
                }
            }
            other => {
                return Err(Error::Gravity(format!(
                    "activation-aware tensor {name:?} bad disposition {other:?}"
                )))
            }
        };
        Ok((codec, descriptor.shape.clone()))
    }
    fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let descriptor = self.descriptor(name)?;
        let payload = self.span(descriptor.offset, descriptor.bytes, name)?;
        match descriptor.disposition.as_str() {
            "pass_through" => {
                aap_require_magic!(payload, AAP_PASS_MAGIC, "aap-pass")?;
                let ndim = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
                let rows = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as u64;
                let cols = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as u64;
                if ndim != descriptor.shape.len()
                    || descriptor.shape.first().copied().unwrap_or(0) != rows
                    || (ndim > 1 && descriptor.shape.get(1).copied().unwrap_or(0) != cols)
                {
                    return Err(Error::Gravity(format!(
                        "aap pass {name:?} shape mismatch {:?}",
                        descriptor.shape
                    )));
                }
                let (codec, _) = self.codec_and_shape(name)?;
                let unit = if codec == "native.f32" { 4 } else { 2 };
                let elements = descriptor.shape.iter().try_fold(1u64, |acc, &value| {
                    acc.checked_mul(value)
                        .ok_or_else(|| Error::Gravity(format!("{name}: element count overflow")))
                })?;
                let expected = elements
                    .checked_mul(unit)
                    .ok_or_else(|| Error::Gravity(format!("{name}: byte count overflow")))?;
                let raw = &payload[AAP_HEADER_LEN..];
                if raw.len() as u64 != expected {
                    return Err(Error::Gravity(format!(
                        "aap pass {name:?} raw bytes {} != {expected}",
                        raw.len()
                    )));
                }
                Ok(raw.to_vec())
            }
            "activation_aware" => {
                let header = parse_activation_aware_header(payload)?;
                if descriptor.shape != vec![header.rows as u64, header.cols as u64] {
                    return Err(Error::Gravity(format!(
                        "activation-aware tensor {name:?} shape [{},{}] != {:?}",
                        header.rows, header.cols, descriptor.shape
                    )));
                }
                if header.has_basis {
                    return Ok(payload.to_vec());
                }
                let coefficient_values = match header.side {
                    ActivationAwareSide::Input => {
                        (header.rows as usize).checked_mul(header.rank as usize)
                    }
                    ActivationAwareSide::Output => {
                        (header.rank as usize).checked_mul(header.cols as usize)
                    }
                }
                .ok_or_else(|| Error::Gravity(format!("{name}: coefficient size overflow")))?;
                let expected_coeff_bytes =
                    aap_section_end!(AAP_HEADER_LEN, coefficient_values, 2, name);
                if payload.len() != expected_coeff_bytes {
                    return Err(Error::Gravity(format!(
                        "activation-aware tensor {name:?} coeff bytes {} != {expected_coeff_bytes}",
                        payload.len()
                    )));
                }
                let basis_descriptor = self.bases.get(&header.basis_layer).ok_or_else(|| {
                    Error::Gravity(format!(
                        "activation-aware tensor {name:?} missing basis {}",
                        header.basis_layer
                    ))
                })?;
                let basis_payload = self.span(
                    basis_descriptor.offset,
                    basis_descriptor.bytes,
                    &format!("basis:{}", header.basis_layer),
                )?;
                aap_require_magic!(basis_payload, AAP_BASIS_MAGIC, "aap-basis")?;
                let hidden = u32::from_le_bytes(basis_payload[8..12].try_into().unwrap()) as usize;
                let basis_rank =
                    u32::from_le_bytes(basis_payload[12..16].try_into().unwrap()) as usize;
                let expected_hidden = match header.side {
                    ActivationAwareSide::Input => header.cols as usize,
                    ActivationAwareSide::Output => header.rows as usize,
                };
                if hidden != expected_hidden
                    || basis_rank != basis_descriptor.rank as usize
                    || basis_rank < header.rank as usize
                {
                    return Err(Error::Gravity(format!(
                    "activation-aware tensor {name:?} basis geom h={hidden}, rank={basis_rank}; want h={expected_hidden}, rank>={}", header.rank )));
                }
                let expected_basis_bytes =
                    aap_section_end!(AAP_HEADER_LEN, hidden * basis_rank, 2, name);
                if basis_payload.len() != expected_basis_bytes {
                    return Err(Error::Gravity(format!(
                    "activation-aware basis layer {} has {} bytes, expected {expected_basis_bytes}", header.basis_layer, basis_payload.len() )));
                }
                let rank = header.rank as usize;
                let mut runtime_payload = Vec::with_capacity(
                    payload.len() + hidden.saturating_mul(rank).saturating_mul(2),
                );
                runtime_payload.extend_from_slice(payload);
                runtime_payload[24] = 1;
                let basis_values = &basis_payload[AAP_HEADER_LEN..];
                if rank == basis_rank {
                    runtime_payload.extend_from_slice(basis_values);
                } else {
                    for row in 0..hidden {
                        let start = row * basis_rank * 2;
                        runtime_payload.extend_from_slice(&basis_values[start..start + rank * 2]);
                    }
                }
                Ok(runtime_payload)
            }
            other => Err(Error::Gravity(format!(
                "activation-aware tensor {name:?} bad disposition {other:?}"
            ))),
        }
    }
}
enum LazyShard {
    Gravity(GravityShard),
    ActivationAware(ActivationAwareShard),
}
impl LazyShard {
    fn codec_and_shape(&self, name: &str) -> Result<(String, Vec<u64>)> {
        match self {
            LazyShard::Gravity(shard) => {
                let descriptor = shard
                    .descriptor(name)
                    .ok_or_else(|| Error::Gravity(format!("no such tensor {name:?}")))?;
                Ok((descriptor.codec.clone(), descriptor.shape.clone()))
            }
            LazyShard::ActivationAware(shard) => shard.codec_and_shape(name),
        }
    }
    fn read_tensor(&self, name: &str, verify_hash: bool) -> Result<Vec<u8>> {
        match self {
            LazyShard::Gravity(shard) => shard.read_tensor(name, verify_hash),
            LazyShard::ActivationAware(shard) => shard.read_tensor(name),
        }
    }
    fn read_tensor_prefix_unverified(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        match self {
            LazyShard::Gravity(shard) => shard.read_tensor_for_prefix(name, bytes),
            LazyShard::ActivationAware(_) => Err(Error::Gravity(format!(
                "tensor {name}: gravity-pq prefix requested from activation-aware shard"
            ))),
        }
    }
}
pub const DEFAULT_NATIVE_DENSE_MEMO_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NativeDenseMemoStats {
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub high_water_bytes: u64,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub verifications: u64,
    pub verified_tensors: usize,
    pub evictions: u64,
}
struct MemoEntry {
    value: Vec<f32>,
    bytes: u64,
    last_tick: u64,
}
struct NativeDenseMemo {
    decoded: HashMap<String, MemoEntry>,
    verified: HashSet<String>,
    budget_bytes: u64,
    resident_bytes: u64,
    high_water_bytes: u64,
    clock: u64,
    hits: u64,
    misses: u64,
    verifications: u64,
    evictions: u64,
}
impl NativeDenseMemo {
    fn new(budget_bytes: u64) -> Self {
        Self {
            decoded: HashMap::new(),
            verified: HashSet::new(),
            budget_bytes,
            resident_bytes: 0,
            high_water_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            verifications: 0,
            evictions: 0,
        }
    }
    fn stats(&self) -> NativeDenseMemoStats {
        NativeDenseMemoStats {
            budget_bytes: self.budget_bytes,
            resident_bytes: self.resident_bytes,
            high_water_bytes: self.high_water_bytes,
            entries: self.decoded.len(),
            hits: self.hits,
            misses: self.misses,
            verifications: self.verifications,
            verified_tensors: self.verified.len(),
            evictions: self.evictions,
        }
    }
    fn is_verified(&self, name: &str) -> bool {
        self.verified.contains(name)
    }
    fn has_decoded(&self, name: &str) -> bool {
        self.decoded.contains_key(name)
    }
    fn take_decoded(&mut self, name: &str) -> Option<Vec<f32>> {
        if let Some(e) = self.decoded.get_mut(name) {
            self.clock = self.clock.saturating_add(1);
            e.last_tick = self.clock;
            self.hits = self.hits.saturating_add(1);
            Some(e.value.clone())
        } else {
            None
        }
    }
    fn note_miss(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }
    fn record_verification(&mut self, name: &str) {
        self.verified.insert(name.to_string());
        self.verifications = self.verifications.saturating_add(1);
    }
    fn mark_verified_without_hash(&mut self, name: &str) {
        self.verified.insert(name.to_string());
    }
    fn admit_decoded(&mut self, name: &str, value: Vec<f32>) {
        if self.decoded.contains_key(name) {
            self.clock = self.clock.saturating_add(1);
            if let Some(e) = self.decoded.get_mut(name) {
                e.last_tick = self.clock;
            }
            return;
        }
        let bytes = (value.len() as u64).saturating_mul(4);
        self.verified.insert(name.to_string());
        if bytes > self.budget_bytes {
            return;
        }
        while self.resident_bytes.saturating_add(bytes) > self.budget_bytes {
            if !self.evict_one() {
                return;
            }
        }
        self.clock = self.clock.saturating_add(1);
        self.decoded.insert(
            name.to_string(),
            MemoEntry {
                value,
                bytes,
                last_tick: self.clock,
            },
        );
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        if self.resident_bytes > self.high_water_bytes {
            self.high_water_bytes = self.resident_bytes;
        }
    }
    fn evict_one(&mut self) -> bool {
        let victim = self
            .decoded
            .iter()
            .min_by_key(|(_, e)| e.last_tick)
            .map(|(k, _)| k.clone());
        let Some(name) = victim else {
            return false;
        };
        if let Some(e) = self.decoded.remove(&name) {
            self.resident_bytes = self.resident_bytes.saturating_sub(e.bytes);
            self.evictions = self.evictions.saturating_add(1);
            true
        } else {
            false
        }
    }
}
enum Source {
    Eager(HashMap<String, Tensor>),
    Lazy {
        shard_dir: PathBuf,
        tensor_shard: HashMap<String, String>,
        open_shards: std::sync::Mutex<HashMap<String, LazyShard>>,
        verify_hash: bool,
        format: LazyFormat,
        source_dtypes: HashMap<String, String>,
        shard_sha256: HashMap<String, String>,
        dense_memo: std::sync::Mutex<NativeDenseMemo>,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LazyFormat {
    Gravity,
    ActivationAware,
}
fn json_str_map(v: &serde_json::Value, key: &str) -> HashMap<String, String> {
    v.get(key)
        .and_then(|value| value.as_object())
        .map(|object| {
            object
                .iter()
                .filter_map(|(n, v)| Some((n.clone(), v.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default()
}
fn lazy_source(
    shard_dir: PathBuf,
    tensor_shard: HashMap<String, String>,
    verify_hash: bool,
    format: LazyFormat,
    source_dtypes: HashMap<String, String>,
    shard_sha256: HashMap<String, String>,
) -> Source {
    Source::Lazy {
        shard_dir,
        tensor_shard,
        open_shards: std::sync::Mutex::new(HashMap::new()),
        verify_hash,
        format,
        source_dtypes,
        shard_sha256,
        dense_memo: std::sync::Mutex::new(NativeDenseMemo::new(
            DEFAULT_NATIVE_DENSE_MEMO_BUDGET_BYTES,
        )),
    }
}
fn decode_tensor(codec: &str, blob: &[u8], name: &str) -> Result<Tensor> {
    if codec == "gravity-pq" {
        Ok(Tensor::Pq(PqTensor::from_payload(blob)?))
    } else if codec == "activation-aware.f16" {
        Ok(Tensor::ActivationAware(
            ActivationAwareTensor::from_payload(blob)?,
        ))
    } else if codec.starts_with("native.") {
        Ok(Tensor::Dense(widen_native(codec, blob)?))
    } else {
        Err(Error::Gravity(format!(
            "tensor {name}: unsupported codec {codec:?}"
        )))
    }
}
pub struct GravityWeights {
    source: Source,
    pub header: serde_json::Value,
}
impl GravityWeights {
    pub fn open(path: &Path, verify_hash: bool) -> Result<GravityWeights> {
        let shard = GravityShard::open(path)?;
        let names: Vec<String> = shard.tensor_names().map(str::to_string).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in &names {
            let codec = shard
                .descriptor(name)
                .expect("name came from tensor_names")
                .codec
                .clone();
            let blob = shard.read_tensor(name, verify_hash)?;
            tensors.insert(name.clone(), decode_tensor(&codec, &blob, name)?);
        }
        Ok(GravityWeights {
            source: Source::Eager(tensors),
            header: shard.extra,
        })
    }
    pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<GravityWeights> {
        let g_idx = dir.join("model.gravity.index.json");
        let a_idx = dir.join("model.activation_aware.index.json");
        if g_idx.is_file() && a_idx.is_file() {
            return Err(Error::Gravity(format!(
                "{}: both gravity and activation-aware indexes present",
                dir.display()
            )));
        }
        let index_choice = if g_idx.is_file() {
            Some((g_idx, LazyFormat::Gravity))
        } else if a_idx.is_file() {
            Some((a_idx, LazyFormat::ActivationAware))
        } else {
            None
        };
        if let Some((index_path, format)) = index_choice {
            let bytes = std::fs::read(&index_path)
                .map_err(|e| Error::Gravity(format!("{}: {e}", index_path.display())))?;
            let manifest: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Gravity(format!("{}: {e}", index_path.display())))?;
            let tensor_shard = json_str_map(&manifest, "weight_map");
            if tensor_shard.is_empty() {
                return Err(Error::Gravity(format!(
                    "{}: weight_map is empty",
                    index_path.display()
                )));
            }
            let source_dtypes = json_str_map(&manifest, "tensor_dtypes");
            let shard_sha256 = json_str_map(&manifest, "shard_sha256");
            if format == LazyFormat::ActivationAware {
                let schema = manifest
                    .get("schema")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if schema != "hawking.activation_aware.model_index.v1" {
                    return Err(Error::Gravity(format!(
                        "{}: unsupported aap index schema {schema:?}",
                        index_path.display()
                    )));
                }
                if source_dtypes.len() != tensor_shard.len() {
                    return Err(Error::Gravity(format!(
                        "{}: tensor_dtypes {} != weights {}",
                        index_path.display(),
                        source_dtypes.len(),
                        tensor_shard.len()
                    )));
                }
                if verify_hash && shard_sha256.is_empty() {
                    return Err(Error::Gravity(format!(
                        "{}: verified aap open needs shard_sha256",
                        index_path.display()
                    )));
                }
            }
            let header = manifest
                .get("architecture")
                .map(|a| serde_json::json!({"architecture": a}))
                .ok_or_else(|| {
                    Error::Gravity(format!("{}: no architecture block", index_path.display()))
                })?;
            return Ok(GravityWeights {
                source: lazy_source(
                    dir.to_path_buf(),
                    tensor_shard,
                    verify_hash,
                    format,
                    source_dtypes,
                    shard_sha256,
                ),
                header,
            });
        }
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| Error::Gravity(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|n| {
                let s = n.to_string_lossy();
                s.starts_with("model-") && s.ends_with(".gravity")
            })
            .collect();
        names.sort();
        if names.is_empty() {
            return Err(Error::Gravity(format!(
                "{}: no gravity index or model-*.gravity shards",
                dir.display()
            )));
        }
        let mut tensor_shard = HashMap::new();
        let mut header = None;
        for name in &names {
            let filename = name.to_string_lossy().into_owned();
            let shard = GravityShard::open(&dir.join(name))?;
            header.get_or_insert_with(|| shard.extra.clone());
            for tname in shard.tensor_names() {
                tensor_shard.insert(tname.to_string(), filename.clone());
            }
        }
        Ok(GravityWeights {
            source: lazy_source(
                dir.to_path_buf(),
                tensor_shard,
                verify_hash,
                LazyFormat::Gravity,
                HashMap::new(),
                HashMap::new(),
            ),
            header: header.expect("names is non-empty"),
        })
    }
    pub fn contains(&self, name: &str) -> bool {
        match &self.source {
            Source::Eager(t) => t.contains_key(name),
            Source::Lazy { tensor_shard, .. } => tensor_shard.contains_key(name),
        }
    }
    pub fn tensor_names(&self) -> Vec<String> {
        match &self.source {
            Source::Eager(t) => t.keys().cloned().collect(),
            Source::Lazy { tensor_shard, .. } => tensor_shard.keys().cloned().collect(),
        }
    }
    pub fn dense_memo_stats(&self) -> NativeDenseMemoStats {
        match &self.source {
            Source::Eager(_) => NativeDenseMemoStats::default(),
            Source::Lazy { dense_memo, .. } => {
                dense_memo.lock().expect("gravity dense-memo mutex").stats()
            }
        }
    }
    pub fn raw_payload(&self, name: &str) -> Result<(String, Vec<u8>)> {
        let (codec, blob, _shape) = self.raw_payload_with_shape(name)?;
        Ok((codec, blob))
    }
    pub fn raw_payload_with_shape(&self, name: &str) -> Result<(String, Vec<u8>, Vec<u64>)> {
        match &self.source {
            Source::Eager(_) => Err(Error::Gravity(
                "raw_payload: not available in Eager mode (decoded at open, raw bytes discarded)"
                    .into(),
            )),
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
                format,
                source_dtypes,
                shard_sha256,
                ..
            } => Self::with_lazy_shard(
                shard_dir,
                tensor_shard,
                open_shards,
                *format,
                source_dtypes,
                shard_sha256,
                *verify_hash,
                name,
                |shard| {
                    let (codec, shape) = shard.codec_and_shape(name)?;
                    let blob = shard.read_tensor(name, *verify_hash)?;
                    Ok((codec, blob, shape))
                },
            ),
        }
    }
    pub fn pq_header_prefix_unverified_with_shape(
        &self,
        name: &str,
    ) -> Result<(PqHeader, Vec<u64>)> {
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Pq(tensor)) => Ok((
                    tensor.header,
                    vec![tensor.header.rows as u64, tensor.header.cols as u64],
                )),
                Some(Tensor::ActivationAware(_)) => Err(Error::Gravity(format!(
                    "tensor {name}: compact admission requires gravity-pq, found activation-aware tensor"
                ))),
                Some(Tensor::Dense(_)) => Err(Error::Gravity(format!(
                    "tensor {name}: compact admission requires gravity-pq, found native tensor"
                ))),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                format,
                source_dtypes,
                shard_sha256,
                verify_hash,
                ..
            } => Self::with_lazy_shard(
                shard_dir,
                tensor_shard,
                open_shards,
                *format,
                source_dtypes,
                shard_sha256,
                *verify_hash,
                name,
                |shard| {
                let (codec, shape) = shard.codec_and_shape(name)?;
                if codec != "gravity-pq" {
                    return Err(Error::Gravity(format!(
                        "tensor {name}: compact admission requires gravity-pq, found {:?}",
                        codec
                    )));
                }
                let prefix = shard.read_tensor_prefix_unverified(name, 64)?;
                Ok((parse_pq_header(&prefix)?, shape))
            }),
        }
    }
    fn with_lazy_shard<T>(
        shard_dir: &Path,
        tensor_shard: &HashMap<String, String>,
        open_shards: &std::sync::Mutex<HashMap<String, LazyShard>>,
        format: LazyFormat,
        source_dtypes: &HashMap<String, String>,
        shard_sha256: &HashMap<String, String>,
        verify_hash: bool,
        name: &str,
        f: impl FnOnce(&LazyShard) -> Result<T>,
    ) -> Result<T> {
        let filename = tensor_shard
            .get(name)
            .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?;
        if !open_shards
            .lock()
            .expect("gravity lazy-shard mutex")
            .contains_key(filename)
        {
            let path = shard_dir.join(filename);
            let shard = match format {
                LazyFormat::Gravity => LazyShard::Gravity(GravityShard::open(&path)?),
                LazyFormat::ActivationAware => {
                    LazyShard::ActivationAware(ActivationAwareShard::open(
                        &path,
                        source_dtypes,
                        shard_sha256.get(filename).map(String::as_str),
                        verify_hash,
                    )?)
                }
            };
            open_shards
                .lock()
                .expect("gravity lazy-shard mutex")
                .insert(filename.clone(), shard);
        }
        f(open_shards
            .lock()
            .expect("gravity lazy-shard mutex")
            .get(filename)
            .expect("just opened or already present"))
    }
    pub fn dense(&self, name: &str) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        cost_ledger::record_dense_call();
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Dense(v)) => {
                    cost_ledger::record_allocation((v.len() * 4) as u64);
                    Ok(v.clone())
                }
                Some(Tensor::Pq(_)) => Err(Error::Gravity(format!(
                    "tensor {name:?} is packed; expected a natively-carried dense tensor"
                ))),
                Some(Tensor::ActivationAware(_)) => Err(Error::Gravity(format!(
                    "tensor {name:?} is activation-aware; expected a natively-carried dense tensor"
                ))),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
                dense_memo,
                format,
                source_dtypes,
                shard_sha256,
            } => {
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if let Some(v) = memo.take_decoded(name) {
                        return Ok(v);
                    }
                    memo.note_miss();
                }
                let need_verify = {
                    let memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    *verify_hash && !memo.is_verified(name)
                };
                let (codec, blob) = Self::with_lazy_shard(
                    shard_dir,
                    tensor_shard,
                    open_shards,
                    *format,
                    source_dtypes,
                    shard_sha256,
                    *verify_hash,
                    name,
                    |shard| {
                        let (codec, _) = shard.codec_and_shape(name)?;
                        if !codec.starts_with("native.") {
                            return Err(Error::Gravity(format!(
                                "tensor {name:?} is packed; expected native dense"
                            )));
                        }
                        let blob = shard.read_tensor(name, need_verify)?;
                        Ok((codec, blob))
                    },
                )?;
                let v = {
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    widen_native(&codec, &blob)?
                };
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if need_verify {
                        memo.record_verification(name);
                    } else if !*verify_hash {
                        memo.mark_verified_without_hash(name);
                    }
                    if let Some(cached) = memo.take_decoded(name) {
                        return Ok(cached);
                    }
                    memo.admit_decoded(name, v.clone());
                }
                Ok(v)
            }
        }
    }
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        cost_ledger::record_matvec_call();
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Pq(t)) => t.matvec(x),
                Some(Tensor::ActivationAware(t)) => t.matvec(x),
                Some(Tensor::Dense(w)) => matvec_dense(w, x, name),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
                format,
                source_dtypes,
                shard_sha256,
                ..
            } => Self::with_lazy_shard(
                shard_dir,
                tensor_shard,
                open_shards,
                *format,
                source_dtypes,
                shard_sha256,
                *verify_hash,
                name,
                |shard| {
                    let (codec, _) = shard.codec_and_shape(name)?;
                    let blob = shard.read_tensor(name, *verify_hash)?;
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    match decode_tensor(&codec, &blob, name)? {
                        Tensor::Pq(t) => {
                            cost_ledger::record_active_bytes_for(name, blob.len() as u64);
                            t.matvec(x)
                        }
                        Tensor::ActivationAware(t) => {
                            cost_ledger::record_active_bytes_for(name, blob.len() as u64);
                            t.matvec(x)
                        }
                        Tensor::Dense(w) => {
                            cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
                            matvec_dense(&w, x, name)
                        }
                    }
                },
            ),
        }
    }
    pub fn row(&self, name: &str, index_: usize, cols: usize) -> Result<Vec<f32>> {
        use crate::cost_ledger::{self, Bucket};
        cost_ledger::record_row_call();
        match &self.source {
            Source::Eager(tensors) => match tensors.get(name) {
                Some(Tensor::Pq(t)) => t.row(index_),
                Some(Tensor::ActivationAware(t)) => t.row(index_),
                Some(Tensor::Dense(w)) => row_dense(w, index_, cols, name),
                None => Err(Error::Gravity(format!("artifact has no tensor {name:?}"))),
            },
            Source::Lazy {
                shard_dir,
                tensor_shard,
                open_shards,
                verify_hash,
                dense_memo,
                format,
                source_dtypes,
                shard_sha256,
            } => {
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if let Some(w) = memo.take_decoded(name) {
                        return row_dense(&w, index_, cols, name);
                    }
                }
                let need_verify = {
                    let memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    *verify_hash && !memo.is_verified(name)
                };
                let (codec, blob) = Self::with_lazy_shard(
                    shard_dir,
                    tensor_shard,
                    open_shards,
                    *format,
                    source_dtypes,
                    shard_sha256,
                    *verify_hash,
                    name,
                    |shard| {
                        let (codec, _) = shard.codec_and_shape(name)?;
                        let blob = shard.read_tensor(name, need_verify)?;
                        Ok((codec, blob))
                    },
                )?;
                {
                    let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                    if need_verify {
                        memo.record_verification(name);
                    } else if !*verify_hash {
                        memo.mark_verified_without_hash(name);
                    }
                }
                if codec == "gravity-pq" {
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    pq_row(&blob, index_)
                } else if codec == "activation-aware.f16" {
                    let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                    ActivationAwareTensor::from_payload(&blob)?.row(index_)
                } else if codec.starts_with("native.") {
                    let w = {
                        let _decode = cost_ledger::Scope::new(Bucket::PackedIndexDecode);
                        widen_native(&codec, &blob)?
                    };
                    let row = row_dense(&w, index_, cols, name)?;
                    {
                        let mut memo = dense_memo.lock().expect("gravity dense-memo mutex");
                        if !memo.has_decoded(name) {
                            memo.note_miss();
                            memo.admit_decoded(name, w);
                        }
                    }
                    Ok(row)
                } else {
                    Err(Error::Gravity(format!(
                        "tensor {name}: unsupported codec {codec:?}"
                    )))
                }
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PqMetalKernelVariant {
    Generic,
    Bits8Direct,
    Bits8Vec4,
    Bits8DoubleSingle,
    Bits8Vec4Split4,
    Bits8Vec4Split8,
}
impl PqMetalKernelVariant {
    pub const ALL: [Self; 6] = [
        Self::Generic,
        Self::Bits8Direct,
        Self::Bits8Vec4,
        Self::Bits8DoubleSingle,
        Self::Bits8Vec4Split4,
        Self::Bits8Vec4Split8,
    ];
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Bits8Direct => "bits8-direct",
            Self::Bits8Vec4 => "bits8-vec4",
            Self::Bits8DoubleSingle => "bits8-double-single",
            Self::Bits8Vec4Split4 => "bits8-2d-split4",
            Self::Bits8Vec4Split8 => "bits8-2d-split8",
        }
    }
    pub const fn kernel_name(self) -> &'static str {
        match self {
            Self::Generic => "gravity_pq_matvec",
            Self::Bits8Direct => "gravity_pq_matvec_bits8_direct",
            Self::Bits8Vec4 => "gravity_pq_matvec_bits8_vec4",
            Self::Bits8DoubleSingle => "gravity_pq_matvec_bits8_double_single",
            Self::Bits8Vec4Split4 | Self::Bits8Vec4Split8 => "gravity_pq_matvec_bits8_2d",
        }
    }
    pub const fn split_count(self) -> Option<u32> {
        match self {
            Self::Bits8Vec4Split4 => Some(4),
            Self::Bits8Vec4Split8 => Some(8),
            _ => None,
        }
    }
    pub const fn dispatches_per_matvec(self) -> usize {
        if self.split_count().is_some() {
            2
        } else {
            1
        }
    }
    pub fn supports(self, h: &PqHeader) -> bool {
        match self {
            Self::Generic => true,
            Self::Bits8Direct => h.bits == 8,
            Self::Bits8Vec4 | Self::Bits8DoubleSingle => h.bits == 8 && h.sub % 4 == 0,
            Self::Bits8Vec4Split4 | Self::Bits8Vec4Split8 => {
                h.bits == 8 && h.d % 4 == 0 && h.sub % 4 == 0
            }
        }
    }
    pub fn validate(self, h: &PqHeader) -> Result<()> {
        if self.supports(h) {
            Ok(())
        } else {
            Err(Error::Gravity(format!(
                "PQ kernel {} rejects D={} S={} sub={} card={} bits={}",
                self.as_str(),
                h.d,
                h.s,
                h.sub,
                h.card,
                h.bits
            )))
        }
    }
}

impl std::fmt::Display for PqMetalKernelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
impl std::str::FromStr for PqMetalKernelVariant {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let n = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|v| v.as_str() == n)
            .ok_or_else(|| {
                format!(
                    "unknown PQ kernel variant {s:?}; expected {:?}",
                    Self::ALL.map(|v| v.as_str())
                )
            })
    }
}
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, Default)]
pub struct PqMetalTimingSummary {
    pub min_us: f64,
    pub median_us: f64,
    pub p95_us: f64,
    pub mean_us: f64,
}
#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct PqMetalBenchmark {
    pub variant: PqMetalKernelVariant,
    pub warmup: usize,
    pub iterations: usize,
    pub wall: PqMetalTimingSummary,
    pub gpu: Option<PqMetalTimingSummary>,
}
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GravityPqParams {
    dim: u32,
    subspaces: u32,
    sub: u32,
    card: u32,
    rows: u32,
    cols: u32,
    nchunk: u32,
    bits: u32,
}
#[cfg(target_os = "macos")]
impl From<PqHeader> for GravityPqParams {
    fn from(h: PqHeader) -> Self {
        Self {
            dim: h.d as u32,
            subspaces: h.s as u32,
            sub: h.sub as u32,
            card: h.card as u32,
            rows: h.rows,
            cols: h.cols,
            nchunk: h.nchunk,
            bits: h.bits as u32,
        }
    }
}
#[cfg(target_os = "macos")]
pub struct PqMetalMatrix {
    header: PqHeader,
    params: GravityPqParams,
    codebooks: metal::Buffer,
    codes: metal::Buffer,
}
#[cfg(target_os = "macos")]
impl PqMetalMatrix {
    pub fn from_payload(ctx: &crate::metal::MetalContext, payload: &[u8]) -> Result<Self> {
        let header = parse_pq_header(payload)?;
        if header.rotate != 0 {
            return Err(Error::Gravity(
                "rotated gravity-pq artifacts (rotate=1) are not yet supported".into(),
            ));
        }
        let (cb, packed_codes) = pq_sections(payload)?;
        let mut codes_padded = Vec::with_capacity(packed_codes.len() + 4);
        codes_padded.extend_from_slice(packed_codes);
        codes_padded.extend_from_slice(&[0u8; 4]);
        Ok(Self {
            header,
            params: header.into(),
            codebooks: ctx.new_buffer_with_bytes_checked(cb)?,
            codes: ctx.new_buffer_with_bytes_checked(&codes_padded)?,
        })
    }
    pub const fn header(&self) -> PqHeader {
        self.header
    }
    fn prepare(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
    ) -> Result<()> {
        variant.validate(&self.header)?;
        let _ = ctx.pipeline(variant.kernel_name())?;
        if variant.split_count().is_some() {
            let _ = ctx.pipeline("gravity_pq_reduce_2d")?;
        }
        Ok(())
    }
    fn encode(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        variant: PqMetalKernelVariant,
        x: &metal::Buffer,
        y: &metal::Buffer,
        partials: Option<&metal::Buffer>,
    ) -> Result<()> {
        const ROW_TG: u32 = 256;
        let params = self.params;
        let set_params = |enc: &metal::ComputeCommandEncoderRef, idx: u64| {
            enc.set_bytes(
                idx,
                std::mem::size_of::<GravityPqParams>() as u64,
                &params as *const GravityPqParams as *const _,
            );
        };
        if let Some(splits) = variant.split_count() {
            let scratch = partials.ok_or_else(|| {
                Error::Gravity(format!("PQ kernel {variant} requires a partials buffer"))
            })?;
            tcb.dispatch_threads(
                variant.kernel_name(),
                (params.rows * 32, splits, 1),
                (32, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&self.codebooks), 0);
                    enc.set_buffer(1, Some(&self.codes), 0);
                    enc.set_buffer(2, Some(x), 0);
                    enc.set_buffer(3, Some(scratch), 0);
                    set_params(enc, 4);
                    enc.set_bytes(5, 4, &splits as *const u32 as *const _);
                },
            )?;
            tcb.dispatch_threads(
                "gravity_pq_reduce_2d",
                (params.rows.div_ceil(ROW_TG) * ROW_TG, 1, 1),
                (ROW_TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(scratch), 0);
                    enc.set_buffer(1, Some(y), 0);
                    set_params(enc, 2);
                    enc.set_bytes(3, 4, &splits as *const u32 as *const _);
                },
            )
        } else {
            let n_tg = params.rows.div_ceil(8);
            tcb.dispatch_threads(
                variant.kernel_name(),
                (n_tg * ROW_TG, 1, 1),
                (ROW_TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&self.codebooks), 0);
                    enc.set_buffer(1, Some(&self.codes), 0);
                    enc.set_buffer(2, Some(x), 0);
                    enc.set_buffer(3, Some(y), 0);
                    set_params(enc, 4);
                },
            )
        }
    }
    fn activation_buffers(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
        x: &[f32],
    ) -> Result<(metal::Buffer, metal::Buffer, Option<metal::Buffer>)> {
        if x.len() != self.header.cols as usize {
            return Err(Error::Gravity(format!(
                "PQ Metal matvec: x.len() {} != cols {}",
                x.len(),
                self.header.cols
            )));
        }
        self.prepare(ctx, variant)?;
        let x_buf = ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice::<f32, u8>(x))?;
        let y_buf =
            ctx.new_buffer_checked(self.header.rows as usize * std::mem::size_of::<f32>())?;
        let partials = variant
            .split_count()
            .map(|splits| {
                ctx.new_buffer_checked(
                    self.header.rows as usize * splits as usize * std::mem::size_of::<f32>(),
                )
            })
            .transpose()?;
        Ok((x_buf, y_buf, partials))
    }
    pub fn matvec(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let (x_buf, y_buf, partials) = self.activation_buffers(ctx, variant, x)?;
        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
        self.encode(&mut tcb, variant, &x_buf, &y_buf, partials.as_ref())?;
        tcb.commit_and_wait()?;
        let ptr = y_buf.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.header.rows as usize) }.to_vec())
    }
    pub fn benchmark(
        &self,
        ctx: &crate::metal::MetalContext,
        variant: PqMetalKernelVariant,
        x: &[f32],
        warmup: usize,
        iterations: usize,
    ) -> Result<PqMetalBenchmark> {
        use std::time::Instant;
        if iterations == 0 {
            return Err(Error::Gravity(
                "PQ Metal benchmark requires at least one iteration".into(),
            ));
        }
        let (x_buf, y_buf, partials) = self.activation_buffers(ctx, variant, x)?;
        let run_once = || -> Result<()> {
            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
            self.encode(&mut tcb, variant, &x_buf, &y_buf, partials.as_ref())?;
            tcb.commit_and_wait()
        };
        let _ = ctx.drain_trace();
        for _ in 0..warmup {
            run_once()?;
        }
        let _ = ctx.drain_trace();
        let mut wall_us = Vec::with_capacity(iterations);
        let mut gpu_us = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            run_once()?;
            wall_us.push(t0.elapsed().as_secs_f64() * 1e6);
            let samples = ctx.drain_trace();
            let gpu: Vec<u64> = samples.iter().filter_map(|s| s.gpu_us).collect();
            if gpu.len() == variant.dispatches_per_matvec() {
                gpu_us.push(gpu.into_iter().sum::<u64>() as f64);
            }
        }
        let summarize = |samples: &[f64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_by(f64::total_cmp);
            let percentile = |p: f64| {
                let i = ((sorted.len() - 1) as f64 * p).round() as usize;
                sorted[i.min(sorted.len() - 1)]
            };
            PqMetalTimingSummary {
                min_us: sorted[0],
                median_us: percentile(0.50),
                p95_us: percentile(0.95),
                mean_us: sorted.iter().sum::<f64>() / sorted.len() as f64,
            }
        };
        Ok(PqMetalBenchmark {
            variant,
            warmup,
            iterations,
            wall: summarize(&wall_us),
            gpu: (gpu_us.len() == iterations).then(|| summarize(&gpu_us)),
        })
    }
}
#[cfg(target_os = "macos")]
pub fn pq_matvec_metal_with_variant(
    ctx: &crate::metal::MetalContext,
    payload: &[u8],
    x: &[f32],
    variant: PqMetalKernelVariant,
) -> Result<Vec<f32>> {
    PqMetalMatrix::from_payload(ctx, payload)?.matvec(ctx, variant, x)
}
#[cfg(target_os = "macos")]
pub fn pq_matvec_metal(
    ctx: &crate::metal::MetalContext,
    payload: &[u8],
    x: &[f32],
) -> Result<Vec<f32>> {
    pq_matvec_metal_with_variant(ctx, payload, x, PqMetalKernelVariant::Generic)
}
#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    use crate::artifact::{parse_activation_aware_header, ActivationAwareTensor};
    use half::f16;
    use sha2::{Digest, Sha256};
    const PQ_MAGIC: &[u8; 8] = b"GLM52CPK";
    const AAP_SCHEMA: &str = "hawking.glm52.activation_aware_pack.v1";
    const AAP_PASS: &[u8; 8] = b"GLM52PT0";
    const AAP_BASIS: &[u8; 8] = b"GLM52BAS";
    const AAP_MAGIC: &[u8; 8] = b"GLM52AAP";
    const HDR: usize = 64;
    fn aap_payload(rows: u32, cols: u32, rank: u32, side: u16, coef: &[f32], basis: Option<&[f32]>) -> Vec<u8> {
        let mut p = Vec::from(AAP_MAGIC.as_slice());
        p.extend_from_slice(&rows.to_le_bytes());
        p.extend_from_slice(&cols.to_le_bytes());
        p.extend_from_slice(&rank.to_le_bytes());
        p.extend_from_slice(&7u16.to_le_bytes());
        p.extend_from_slice(&side.to_le_bytes());
        p.push(u8::from(basis.is_some()));
        p.resize(HDR, 0);
        p.extend(coef.iter().flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes()));
        if let Some(b) = basis {
            p.extend(b.iter().flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes()));
        }
        p
    }
    fn write_gravity(dir: &Path, model: &str, codec: &str, tensors: Vec<(String, Vec<u8>, serde_json::Value)>) -> PathBuf {
        use sha2::{Digest, Sha256};
        let mut body = Vec::new();
        let mut descs = Vec::new();
        for (name, blob, mut extra) in tensors {
            let hex = extra
                .get("sha256")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{:x}", Sha256::digest(&blob)));
            let o = extra.as_object_mut().unwrap();
            o.insert("name".into(), name.into());
            o.insert("codec".into(), codec.into());
            o.insert("offset".into(), (body.len() as u64).into());
            o.insert("bytes".into(), (blob.len() as u64).into());
            o.insert("sha256".into(), hex.into());
            descs.push(extra);
            body.extend_from_slice(&blob);
        }
        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1", "format_version": 1,
            "model": {"name": model}, "architecture": {}, "tokenizer": {},
            "compression": {"codec": codec}, "shard": {"index": 1, "count": 1},
            "integrity": {"tensor_count": descs.len()}, "tensors": descs,
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let path = dir.join("model-00001-of-00001.gravity");
        let mut out = Vec::with_capacity(20 + hb.len() + body.len());
        out.extend_from_slice(&[b'G', b'R', b'A', b'V', b'I', b'T', b'Y', 0]);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(&body);
        std::fs::write(&path, out).unwrap();
        path
    }
    fn write_native(dir: &Path, tensors: &[(&str, &[f32])]) -> PathBuf {
        let tensors = tensors.iter().map(|&(n, v)| {
            let blob = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            (n.to_string(), blob, serde_json::json!({"shape": [v.len() as u64], "elements": v.len() as u64}))
        }).collect();
        write_gravity(dir, "dense-memo-fixture", "native.f32", tensors)
    }
    #[test]
    fn activation_aware_factorized_input_and_output_match_dense_authority() {
        let basis = [1.0, 0.0, 0.0, 1.0, 0.5, -0.25];
        let input = ActivationAwareTensor::from_payload(&aap_payload(
            2, 3, 2, 1, &[2.0, -1.0, 0.5, 3.0], Some(&basis))).unwrap();
        assert_eq!(input.matvec(&[0.25, 2.0, -1.0]).unwrap(),
                   [2.0 * 0.25 + -2.0 + -1.25, 0.5 * 0.25 + 6.0 + 0.5]);
        assert_eq!(input.row(1).unwrap(), vec![0.5, 3.0, -0.5]);
        let output = ActivationAwareTensor::from_payload(&aap_payload(
            3, 2, 2, 2, &[1.0, 2.0, -0.5, 4.0], Some(&basis))).unwrap();
        assert_eq!(output.matvec(&[1.5, -0.75]).unwrap(), [0.0, -3.75, 0.9375]);
        assert_eq!(output.row(2).unwrap(), vec![0.625, 0.0]);
    }
    #[test]
    fn activation_aware_open_dir_attaches_shared_basis_and_decodes_pass_through() {
        let dir = tempfile::tempdir().unwrap();
        let (wname, nname, sname) = (
            "model.layers.0.input.weight", "model.layers.0.norm.weight", "model-00001-of-00001.aap");
        let basis = [1.0, 0.0, 0.0, 1.0, 0.5, -0.25];
        let mut bp = Vec::from(AAP_BASIS.as_slice());
        bp.extend_from_slice(&3u32.to_le_bytes());
        bp.extend_from_slice(&2u32.to_le_bytes());
        bp.resize(HDR, 0);
        bp.extend(basis.iter().flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes()));
        let wp = aap_payload(2, 3, 2, 1, &[2.0, -1.0, 0.5, 3.0], None);
        let mut np = Vec::from(AAP_PASS.as_slice());
        np.extend_from_slice(&1u32.to_le_bytes());
        np.extend_from_slice(&3u32.to_le_bytes());
        np.extend_from_slice(&0u32.to_le_bytes());
        np.resize(HDR, 0);
        for v in [1.5f32, -2.0, 0.25] {
            np.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let woff = bp.len() as u64;
        let noff = woff + wp.len() as u64;
        let index = serde_json::json!({
            "schema": AAP_SCHEMA, "shard": "model-00001-of-00001.safetensors", "shared_bases": true,
            "bases": [{"basis_layer": 7, "rank": 2, "offset": 0, "bytes": bp.len()}],
            "tensors": [
                {"name": wname, "disposition": "activation_aware", "offset": woff, "bytes": wp.len(), "shape": [2, 3]},
                {"name": nname, "disposition": "pass_through", "offset": noff, "bytes": np.len(), "shape": [3]},
            ],
        });
        let ib = serde_json::to_vec(&index).unwrap();
        let mut sb = Vec::new();
        sb.extend_from_slice(&(ib.len() as u64).to_le_bytes());
        sb.extend_from_slice(&ib);
        sb.extend_from_slice(&bp);
        sb.extend_from_slice(&wp);
        sb.extend_from_slice(&np);
        std::fs::write(dir.path().join(sname), &sb).unwrap();
        let sh = format!("{:x}", Sha256::digest(&sb));
        let manifest = serde_json::json!({
            "schema": "hawking.activation_aware.model_index.v1", "architecture": {"hidden_size": 3},
            "weight_map": {(wname): sname, (nname): sname},
            "tensor_dtypes": {(wname): "BF16", (nname): "BF16"},
            "shard_sha256": {(sname): sh},
        });
        std::fs::write(dir.path().join("model.activation_aware.index.json"),
                       serde_json::to_vec(&manifest).unwrap()).unwrap();
        let weights = GravityWeights::open_dir(dir.path(), true).unwrap();
        assert_eq!(weights.matvec(wname, &[0.25, 2.0, -1.0]).unwrap(), vec![-2.75, 6.625]);
        assert_eq!(weights.row(wname, 1, 3).unwrap(), vec![0.5, 3.0, -0.5]);
        assert_eq!(weights.dense(nname).unwrap(), vec![1.5, -2.0, 0.25]);
        let (codec, payload, shape) = weights.raw_payload_with_shape(wname).unwrap();
        assert_eq!((codec.as_str(), shape), ("activation-aware.f16", vec![2, 3]));
        assert!(parse_activation_aware_header(&payload).unwrap().has_basis);
    }
    #[test]
    fn pq_admission_header_prefix_does_not_claim_full_payload_verification() {
        let dir = tempfile::tempdir().unwrap();
        let name = "model.layers.0.self_attn.kv_b_proj.weight";
        let mut payload = vec![0u8; 64];
        payload[..8].copy_from_slice(PQ_MAGIC);
        payload[8..10].copy_from_slice(&32u16.to_le_bytes());
        payload[10..12].copy_from_slice(&1u16.to_le_bytes());
        payload[12..14].copy_from_slice(&32u16.to_le_bytes());
        payload[14..16].copy_from_slice(&256u16.to_le_bytes());
        payload[16..20].copy_from_slice(&2u32.to_le_bytes());
        payload[20..24].copy_from_slice(&32u32.to_le_bytes());
        payload[24..28].copy_from_slice(&1u32.to_le_bytes());
        payload[28..32].copy_from_slice(&7u32.to_le_bytes());
        payload[32..34].copy_from_slice(&8u16.to_le_bytes());
        payload[35] = 1;
        write_gravity(dir.path(), "pq-header-prefix-fixture", "gravity-pq", vec![(
            name.to_string(), payload,
            serde_json::json!({"shape": [2, 32], "elements": 64, "sha256": "00".repeat(32)}),
        )]);
        let weights = GravityWeights::open_dir(dir.path(), true).unwrap();
        let (header, shape) = weights.pq_header_prefix_unverified_with_shape(name).unwrap();
        assert_eq!(header, PqHeader {
            d: 32, s: 1, sub: 32, card: 256, rows: 2, cols: 32, nchunk: 1, seed: 7,
            bits: 8, rotate: 0, n_codebooks: 1,
        });
        assert_eq!(shape, vec![2, 32]);
        let err = weights.raw_payload_with_shape(name).expect_err("must enforce SHA-256");
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
    }
    #[test]
    fn dense_memo_verify_once_identity_row_and_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let vals: Vec<f32> = (0..64).map(|i| (i as f32) * 0.125 - 1.0).collect();
        write_native(dir.path(), &[("norm.weight", &vals)]);
        let weights = GravityWeights::open_dir(dir.path(), true).unwrap();
        assert_eq!(weights.dense("norm.weight").unwrap(), vals);
        let s = weights.dense_memo_stats();
        assert_eq!((s.verifications, s.misses, s.hits, s.entries), (1, 1, 0, 1));
        assert_eq!(s.resident_bytes, 64 * 4);
        assert!(s.budget_bytes >= s.resident_bytes);
        for _ in 0..50 { assert_eq!(weights.dense("norm.weight").unwrap(), vals); }
        let s = weights.dense_memo_stats();
        assert_eq!((s.verifications, s.misses, s.hits, s.entries, s.verified_tensors), (1, 1, 50, 1, 1));
        let dir = tempfile::tempdir().unwrap();
        let edge = vec![0.0f32, -0.0, 1.0, -1.0, f32::from_bits(1), f32::MIN, f32::MAX, std::f32::consts::PI];
        write_native(dir.path(), &[("a.weight", &edge), ("b.bias", &[0.5, -0.25, 2.0])]);
        let weights = GravityWeights::open_dir(dir.path(), true).unwrap();
        let a0 = weights.dense("a.weight").unwrap();
        let b0 = weights.dense("b.bias").unwrap();
        let a1 = weights.dense("a.weight").unwrap();
        let b1 = weights.dense("b.bias").unwrap();
        assert_eq!(&a0, &edge);
        assert_eq!(&a1, &a0);
        assert_eq!(b0, vec![0.5, -0.25, 2.0]);
        assert_eq!(&b1, &b0);
        for (i, (&x, &y)) in a0.iter().zip(a1.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "a.weight[{i}] bits");
        }
        let s = weights.dense_memo_stats();
        assert_eq!((s.verifications, s.hits, s.misses, s.entries), (2, 2, 2, 2));
        let dir = tempfile::tempdir().unwrap();
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        write_native(dir.path(), &[("embed.weight", &matrix)]);
        let weights = GravityWeights::open_dir(dir.path(), true).unwrap();
        assert_eq!(weights.row("embed.weight", 1, 4).unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(weights.dense_memo_stats().verifications, 1);
        assert_eq!(weights.dense("embed.weight").unwrap(), matrix);
        assert_eq!(weights.dense_memo_stats().hits, 1);
        let dir = tempfile::tempdir().unwrap();
        let big: Vec<f32> = (0..8).map(|i| i as f32).collect();
        write_native(dir.path(), &[("big.weight", &big)]);
        let mut memo = NativeDenseMemo::new(16);
        memo.record_verification("big.weight");
        memo.admit_decoded("big.weight", big.clone());
        assert!(!memo.has_decoded("big.weight") && memo.is_verified("big.weight"));
        assert_eq!((memo.stats().entries, memo.stats().verifications, memo.stats().verified_tensors), (0, 1, 1));
        let weights = GravityWeights::open_dir(dir.path(), true).unwrap();
        for _ in 0..5 { assert_eq!(weights.dense("big.weight").unwrap(), big); }
        let s = weights.dense_memo_stats();
        assert_eq!((s.verifications, s.hits, s.misses), (1, 4, 1));
    }
}
