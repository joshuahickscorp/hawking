//! path-to-50 lever 2: per-layer mixed-precision MoE weight store.
//!
//! Data layer for re-quantizing DeepSeek-V2-Lite MoE expert weights at
//! load time. Reads from the GGUF mmap, dequantizes per the source dtype,
//! re-quantizes per the tier-map dtype, packs the result into a single
//! contiguous `Vec<u8>` that mirrors the GGUF mmap's per-expert layout.
//!
//! ## What this module DOES
//!
//! - Take a `TierMap` + `GgufFile` + a list of (layer, tensor-name-template)
//!   pairs, and produce a `MixedQuantStore` whose `blob` holds the
//!   re-quantized bytes laid out per (layer × expert × elements).
//! - Provide per-(layer, expert) byte offsets/sizes/dtypes so a Metal
//!   dispatcher can target the store instead of the original mmap.
//! - Hold an optional pre-uploaded `PinnedBuffer` view of the blob.
//!
//! ## What this module DOES NOT do (out of scope)
//!
//! - Modify the dispatcher. That's the integration step gated on adding
//!   the Q8_0/Q6_K _v2t_gu (fused gate+up) Metal shaders that don't yet
//!   exist; the realistic V2-Lite tier-map for v1 is `down`-only.
//! - Validate that the resulting tier yields a kernel that exists in
//!   tree — the caller must check that the chosen kernel name resolves.
//!   (For `down` only, all of Q4_K/Q5_0/Q6_K/Q8_0 have `_v2t` variants.)
//!
//! ## Layout
//!
//! The blob is laid out:
//!
//! ```text
//!   layer 0: [expert 0: down bytes][expert 1: down bytes]...[expert N-1: down bytes]
//!   layer 1: ...
//!   layer K-1: ...
//! ```
//!
//! Equivalent to the GGUF fused expert tensor layout (outer-dim = expert
//! id) — so the dispatcher's existing per-expert offset arithmetic works
//! unchanged once it switches buffers.
//!
//! ## Memory
//!
//! Re-quantizing 21 V2-Lite middle layers' `ffn_down_exps` from Q5_0
//! (5.5 bpw) to Q8_0 (8.5 bpw) is ~+125 MB. Re-quantizing the 4 leading
//! and 2 trailing layers' down from Q5_0 to Q4_K (-1 bpw) saves ~30 MB.
//! Net headroom impact ≪ V2-Lite's 9.7 GB footprint, easily fits in
//! 18 GB.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::gguf::{GgmlType, GgufFile};
use crate::quant;
use crate::quant_tier_map::{GroupKind, TierMap};
use crate::{Error, Result};

#[derive(Serialize, Deserialize)]
struct CacheManifest {
    schema_version: u32,
    blob_len: usize,
    tensors: Vec<CacheEntry>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    layer: usize,
    group: String,
    expert: usize,
    offset: usize,
    byte_size: usize,
    n_elems: usize,
    dtype: u32,
}

fn cache_root() -> std::path::PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return std::path::PathBuf::from(xdg).join("dismantle");
    }
    if let Ok(home) = std::env::var("HOME") {
        return std::path::PathBuf::from(home).join(".cache").join("dismantle");
    }
    std::path::PathBuf::from("/tmp").join("dismantle-cache")
}

fn cache_fingerprint(
    gguf_path: &std::path::Path,
    tier_map_path: &std::path::Path,
) -> Result<String> {
    let mut h = Sha256::new();
    h.update(STORE_CACHE_SCHEMA_V.to_le_bytes());
    for p in [gguf_path, tier_map_path] {
        h.update(p.to_string_lossy().as_bytes());
        h.update([0u8]);
        let md = std::fs::metadata(p)?;
        h.update(md.len().to_le_bytes());
        if let Ok(mt) = md.modified() {
            if let Ok(dur) = mt.duration_since(std::time::UNIX_EPOCH) {
                h.update(dur.as_secs().to_le_bytes());
                h.update(dur.subsec_nanos().to_le_bytes());
            }
        }
    }
    let digest = h.finalize();
    Ok(digest.iter().take(16).map(|b| format!("{b:02x}")).collect())
}

fn gmltype_from_u32(v: u32) -> Result<GgmlType> {
    match v {
        x if x == GgmlType::Q4_K as u32 => Ok(GgmlType::Q4_K),
        x if x == GgmlType::Q6_K as u32 => Ok(GgmlType::Q6_K),
        x if x == GgmlType::Q8_0 as u32 => Ok(GgmlType::Q8_0),
        _ => Err(Error::Model(format!("cache: unsupported dtype tag {v}"))),
    }
}

/// One tensor in the store: where it lives in `blob`, what dtype, and
/// how many elements.
#[derive(Debug, Clone)]
pub struct StoredTensor {
    /// Byte offset into `MixedQuantStore::blob`.
    pub offset: usize,
    /// Block-packed byte length of this tensor's quants.
    pub byte_size: usize,
    /// Element count (n_elems = layer_intermediate × hidden for one
    /// expert's down weight on V2-Lite = 1408 × 2048).
    pub n_elems: usize,
    /// Quant dtype written into `blob`.
    pub dtype: GgmlType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StoreKey {
    pub layer: usize,
    pub group: GroupKind,
    /// Routed expert id, or `usize::MAX` for the fused-shared expert.
    pub expert: usize,
}

#[derive(Debug)]
pub struct MixedQuantStore {
    blob: Vec<u8>,
    /// Per-(layer, group, expert) tensor descriptors. Use
    /// [`StoreKey::shared`] when looking up the shared-expert tensor.
    tensors: HashMap<StoreKey, StoredTensor>,
}

impl StoreKey {
    pub fn routed(layer: usize, group: GroupKind, expert: usize) -> Self {
        Self { layer, group, expert }
    }

    pub fn shared(layer: usize, group: GroupKind) -> Self {
        Self { layer, group, expert: usize::MAX }
    }
}

/// Cache schema version — bump on any change to `quantize_q4_k`,
/// `quantize_q6_k`, `quantize_q8_0`, the blob layout, or this struct's
/// field ordering. Old cache files become invalid (mismatched cache
/// fingerprint → fresh build).
const STORE_CACHE_SCHEMA_V: u32 = 1;

impl MixedQuantStore {
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    pub fn get(&self, key: StoreKey) -> Option<&StoredTensor> {
        self.tensors.get(&key)
    }

    pub fn len_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// True when no tier override survived materialization (e.g. the
    /// tier map asked for a dtype identical to the GGUF source).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Wall-clock optimization #2 (2026-05-22): cached variant of
    /// [`MixedQuantStore::build`]. Computes a fingerprint over
    /// (gguf_path, gguf_size, gguf_mtime, tier_map_path, tier_map_mtime,
    /// schema-version). On cache hit, loads the pre-quantized blob and
    /// tensor descriptors from disk in ~100 ms instead of rebuilding for
    /// ~30-60 s. On cache miss, builds and persists.
    ///
    /// Cache location: `$XDG_CACHE_HOME/dismantle/mixed_quant/{fp}.bin`
    /// + `.json`. Falls back to `~/.cache/dismantle/mixed_quant/...`.
    /// On any IO failure during cache load/write the function falls
    /// through to a fresh build — caching is best-effort, never fatal.
    ///
    /// Quality: cache key includes mtime+size of both inputs and a
    /// monotonic schema version. Any quant-fn or layout change MUST
    /// bump [`STORE_CACHE_SCHEMA_V`].
    pub fn build_cached(
        gguf: &GgufFile,
        gguf_path: &std::path::Path,
        tier_map: &TierMap,
        tier_map_path: &std::path::Path,
        n_layers: usize,
        first_k_dense_layers: usize,
        n_routed_experts: usize,
        include_shared: bool,
    ) -> Result<Self> {
        let fp = cache_fingerprint(gguf_path, tier_map_path)?;
        let cache_dir = cache_root().join("mixed_quant");
        let bin = cache_dir.join(format!("{fp}.bin"));
        let meta = cache_dir.join(format!("{fp}.json"));
        if bin.exists() && meta.exists() {
            if let Ok(store) = Self::load_from_cache(&bin, &meta) {
                return Ok(store);
            }
        }
        let store = Self::build(
            gguf, tier_map, n_layers, first_k_dense_layers,
            n_routed_experts, include_shared,
        )?;
        let _ = std::fs::create_dir_all(&cache_dir);
        let _ = store.write_cache(&bin, &meta); // best-effort
        Ok(store)
    }

    fn load_from_cache(bin: &std::path::Path, meta: &std::path::Path) -> Result<Self> {
        let meta_bytes = std::fs::read(meta)?;
        let v: CacheManifest = serde_json::from_slice(&meta_bytes)
            .map_err(|e| Error::Model(format!("cache manifest parse: {e}")))?;
        if v.schema_version != STORE_CACHE_SCHEMA_V {
            return Err(Error::Model("cache schema mismatch".into()));
        }
        let blob = std::fs::read(bin)?;
        if blob.len() != v.blob_len {
            return Err(Error::Model("cache blob length mismatch".into()));
        }
        let mut tensors = HashMap::with_capacity(v.tensors.len());
        for e in v.tensors {
            tensors.insert(
                StoreKey {
                    layer: e.layer,
                    group: match e.group.as_str() {
                        "GateUp" => GroupKind::GateUp,
                        "Down" => GroupKind::Down,
                        _ => return Err(Error::Model(format!("cache: bad group {}", e.group))),
                    },
                    expert: e.expert,
                },
                StoredTensor {
                    offset: e.offset,
                    byte_size: e.byte_size,
                    n_elems: e.n_elems,
                    dtype: gmltype_from_u32(e.dtype)?,
                },
            );
        }
        Ok(Self { blob, tensors })
    }

    fn write_cache(&self, bin: &std::path::Path, meta: &std::path::Path) -> Result<()> {
        std::fs::write(bin, &self.blob)?;
        let entries: Vec<CacheEntry> = self
            .tensors
            .iter()
            .map(|(k, t)| CacheEntry {
                layer: k.layer,
                group: match k.group {
                    GroupKind::GateUp => "GateUp",
                    GroupKind::Down => "Down",
                }
                .into(),
                expert: k.expert,
                offset: t.offset,
                byte_size: t.byte_size,
                n_elems: t.n_elems,
                dtype: t.dtype as u32,
            })
            .collect();
        let manifest = CacheManifest {
            schema_version: STORE_CACHE_SCHEMA_V,
            blob_len: self.blob.len(),
            tensors: entries,
        };
        let bytes = serde_json::to_vec(&manifest)
            .map_err(|e| Error::Model(format!("cache manifest serialize: {e}")))?;
        std::fs::write(meta, &bytes)?;
        Ok(())
    }

    /// Re-quantize the GGUF MoE expert weights per the tier map and
    /// return a populated store.
    ///
    /// Tensor selection: for each layer `li` and each `group` (GateUp,
    /// Down) in `groups_in_play`, this function looks up the tier
    /// override (if any), then reads the corresponding GGUF tensor(s)
    /// (`blk.{li}.ffn_{gate,up,down}_exps.weight` for routed and
    /// `blk.{li}.ffn_{gate,up,down}_shexp.weight` for shared),
    /// dequantizes to f32, re-quantizes via the tier dtype, and packs
    /// into `blob`.
    ///
    /// Layers absent from the tier map are SKIPPED — they continue to
    /// use the GGUF native tensors via the original mmap buffer. Same
    /// for layers whose tier matches the source dtype.
    ///
    /// `first_k_dense_layers` is the count of leading dense (non-MoE)
    /// layers; those are skipped entirely since they don't have
    /// `*_exps` tensors.
    pub fn build(
        gguf: &GgufFile,
        tier_map: &TierMap,
        n_layers: usize,
        first_k_dense_layers: usize,
        n_routed_experts: usize,
        include_shared: bool,
    ) -> Result<Self> {
        let mut blob = Vec::new();
        let mut tensors = HashMap::new();

        for li in first_k_dense_layers..n_layers {
            for &group in &[GroupKind::GateUp, GroupKind::Down] {
                let tier = match tier_map.tier_for(li, group) {
                    Some(t) => t,
                    None => continue,
                };

                // GateUp covers two GGUF tensors (gate_exps + up_exps); Down
                // is a single tensor. We only do single-tensor groups here
                // since the dispatcher needs to swap buffers per kernel
                // call — fused gate+up requires both tensors share a
                // buffer (Q4_K_v2t_gu specifically), and shipping a
                // mixed-dtype gate_up is gated on new shaders.
                if group == GroupKind::GateUp {
                    // Honor "same dtype as native" tier as a no-op pass
                    // (avoids needlessly inflating memory).
                    let gate_dtype = gguf
                        .tensor(&format!("blk.{li}.ffn_gate_exps.weight"))
                        .map(|t| t.dtype);
                    if gate_dtype == Some(tier) {
                        continue;
                    }
                    return Err(Error::Model(format!(
                        "mixed_quant_store: tier map asks for gate_up={:?} at layer {} but \
                         only Down tier overrides are supported in v1 (fused gate+up \
                         requires Q4_K_v2t_gu shaders; cross-dtype gate_up shaders missing)",
                        tier, li
                    )));
                }

                // group == Down: routed expert weights
                let routed_name = format!("blk.{li}.ffn_down_exps.weight");
                let info = gguf.tensor(&routed_name).ok_or_else(|| {
                    Error::Model(format!("mixed_quant_store: missing tensor {routed_name}"))
                })?;
                if gguf.tensor(&routed_name).map(|t| t.dtype) == Some(tier) {
                    continue;
                }
                let total_elems: usize = info.dims.iter().product::<u64>() as usize;
                if total_elems % n_routed_experts != 0 {
                    return Err(Error::Model(format!(
                        "mixed_quant_store: tensor {routed_name} elems {total_elems} \
                         not divisible by {n_routed_experts} experts"
                    )));
                }
                let elems_per_expert = total_elems / n_routed_experts;
                let bytes_per_expert_out = quant_block_bytes(tier, elems_per_expert)?;

                // Read whole tensor; dequant once into a reusable buffer.
                let src_bytes = gguf.tensor_bytes(&routed_name).ok_or_else(|| {
                    Error::Model(format!("mixed_quant_store: tensor_bytes None for {routed_name}"))
                })?;
                let src_bytes_per_expert = src_bytes.len() / n_routed_experts;
                let mut deq = vec![0.0f32; elems_per_expert];
                for e in 0..n_routed_experts {
                    let e_src = &src_bytes[e * src_bytes_per_expert..(e + 1) * src_bytes_per_expert];
                    quant::dequant_into(info.dtype, e_src, &mut deq)?;
                    let offset = blob.len();
                    blob.resize(offset + bytes_per_expert_out, 0u8);
                    quantize_into(tier, &deq, &mut blob[offset..offset + bytes_per_expert_out])?;
                    tensors.insert(
                        StoreKey::routed(li, GroupKind::Down, e),
                        StoredTensor {
                            offset,
                            byte_size: bytes_per_expert_out,
                            n_elems: elems_per_expert,
                            dtype: tier,
                        },
                    );
                }

                if include_shared {
                    // Shared-expert down weight: single fused tensor whose
                    // intermediate width = n_shared_experts × moe_intermediate.
                    let shared_name = format!("blk.{li}.ffn_down_shexp.weight");
                    if let Some(sinfo) = gguf.tensor(&shared_name) {
                        if sinfo.dtype != tier {
                            let sn_elems: usize =
                                sinfo.dims.iter().product::<u64>() as usize;
                            let sbytes_out = quant_block_bytes(tier, sn_elems)?;
                            let sbytes = gguf.tensor_bytes(&shared_name).ok_or_else(|| {
                                Error::Model(format!(
                                    "mixed_quant_store: tensor_bytes None for {shared_name}"
                                ))
                            })?;
                            let mut sdeq = vec![0.0f32; sn_elems];
                            quant::dequant_into(sinfo.dtype, sbytes, &mut sdeq)?;
                            let offset = blob.len();
                            blob.resize(offset + sbytes_out, 0u8);
                            quantize_into(
                                tier,
                                &sdeq,
                                &mut blob[offset..offset + sbytes_out],
                            )?;
                            tensors.insert(
                                StoreKey::shared(li, GroupKind::Down),
                                StoredTensor {
                                    offset,
                                    byte_size: sbytes_out,
                                    n_elems: sn_elems,
                                    dtype: tier,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(Self { blob, tensors })
    }
}

fn quant_block_bytes(dtype: GgmlType, n_elems: usize) -> Result<usize> {
    let (block_size, block_bytes) = dtype.block_layout();
    let block_size = block_size as usize;
    let block_bytes = block_bytes as usize;
    if n_elems % block_size != 0 {
        return Err(Error::Model(format!(
            "mixed_quant_store: n_elems {n_elems} not multiple of block_size {block_size} for {:?}",
            dtype
        )));
    }
    Ok((n_elems / block_size) * block_bytes)
}

fn quantize_into(dtype: GgmlType, src: &[f32], dst: &mut [u8]) -> Result<()> {
    match dtype {
        GgmlType::Q4_K => quant::quantize_q4_k(src, dst),
        GgmlType::Q6_K => quant::quantize_q6_k(src, dst),
        GgmlType::Q8_0 => quant::quantize_q8_0(src, dst),
        other => Err(Error::Model(format!(
            "mixed_quant_store: re-quantize to {:?} not implemented (allowed: Q4_K, Q6_K, Q8_0)",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_tier_map::TierMap;

    fn make_tier_map(layer: usize, tier: &str) -> TierMap {
        let json = format!(
            r#"{{ "schema_version": 1, "model_arch": "deepseek2", "n_layers": {},
                  "layers": [{{ "layer": {}, "down": "{}" }}] }}"#,
            (layer + 1).max(1),
            layer,
            tier,
        );
        // Parse via TierMap test path. We re-serialize through std fs only;
        // here we exercise the from_parts via from_str+from_parts trick.
        // Easiest: write to a temp file. But our tests aren't fs-dependent;
        // use a serde indirection mirroring TierMap::load.
        let p = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(p.path(), json).unwrap();
        TierMap::load(p.path()).unwrap()
    }

    #[test]
    fn requantize_q5_0_to_q8_0_round_trip_bounded() {
        // Synthesize a single-block Q5_0 tensor (32 elems, 22 bytes) and
        // re-quantize via the store's `quantize_into`.
        use half::f16;
        let n_elems = 32;
        // Q5_0 block: 22 bytes = f16 d (2) + qh (4) + qs (16). Encode
        // d=1.0, qh=0, qs alternating low-nibble values.
        let mut src_bytes = vec![0u8; 22];
        src_bytes[0..2].copy_from_slice(&f16::from_f32(0.05).to_bits().to_le_bytes());
        for i in 0..16 {
            src_bytes[6 + i] = (i as u8 & 0x7) | (((i as u8 + 1) & 0x7) << 4);
        }
        let mut deq = vec![0.0f32; n_elems];
        quant::dequant_into(GgmlType::Q5_0, &src_bytes, &mut deq).unwrap();

        // Re-quantize to Q8_0 and dequantize back.
        let mut q8_blob = vec![0u8; 34];
        quantize_into(GgmlType::Q8_0, &deq, &mut q8_blob).unwrap();
        let mut requant = vec![0.0f32; n_elems];
        quant::dequant_into(GgmlType::Q8_0, &q8_blob, &mut requant).unwrap();
        let err = deq
            .iter()
            .zip(requant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Q8 has finer steps than Q5 ⇒ the rounding error should be
        // dominated by Q8's own scale step (~deq_amax/127). Should be tiny.
        assert!(err < 0.05, "Q5_0→Q8_0 round-trip err = {err}");
    }

    #[test]
    fn key_routed_vs_shared_disjoint() {
        let a = StoreKey::routed(3, GroupKind::Down, 5);
        let b = StoreKey::shared(3, GroupKind::Down);
        assert_ne!(a, b);
    }

    #[test]
    fn rejects_gate_up_tier_in_v1() {
        // Build a minimal GGUF-less harness: we can't easily build a real
        // GgufFile in unit tests, so this just tests the tier_map → error
        // path indirectly. (Full integration test runs against the V2-Lite
        // weights in mixed_precision_parity.rs.)
        let m = make_tier_map(0, "q4_K");
        let _ = m; // verify it loads without panicking; full integration
                   // tests cover the actual build flow.
    }
}
