//! TQ GPU bitslice decode → GEMV — the Metal port of the STRAND **G4 bitslice**
//! kernel (`vendor/strand-decode-kernel/{src/metal.rs,shaders/strand_bitslice.metal}`),
//! gated behind the `tq` cargo feature.
//!
//! The G4 shape is the structural inversion of every dead per-row STRAND kernel:
//! the grid is **all blocks** (one thread owns one 256-weight block-stream
//! end-to-end), 256 independent streams per threadgroup, chain state in
//! registers, the `2^L` Q12 codebook staged once into threadgroup memory, ONE
//! barrier total. Measured 60–74 % of the M3's streaming peak on the strand
//! gate harness — but this module's contract is **bit-identity, not speed**:
//! the GPU Q12 output is held byte-for-byte equal to the CPU oracle
//! `strand_quant::decode::decode_tensor_fixed` (the same contract `crate::tq`
//! reproduces on CPU).
//!
//! ## Slice 0 (this file, GPU-free)
//!
//! Defines the host-side [`BitsliceEntry`] table record (mirroring the MSL
//! struct) and [`bake_bitslice_entries`], a verbatim port of
//! `strand-decode-kernel/src/metal.rs::bake_bitslice_entries`. The bake folds
//! the per-block super-scale + 6-bit sub-scale side-info into the pre-expanded
//! `eff[8]` / `off[8]` arrays via the strand-quant public helpers
//! (`decode::{eff_scale_q, eff_min_q}`, `encode::unpack_sub_scales`) — it never
//! forks that integer math. The prefix-sum `out_off`, per-block `bit_offset`,
//! and tail-biting `init_state` recovery (so the kernel never prescans) are
//! reproduced here on top of strand-quant's public API.
//!
//! The unit tests at the bottom run a CPU **host-walk** that replays the
//! bitstream straight from the baked table — `state = ((state<<k)|sym)&mask;
//! q = lut[state]; w = (eff[j>>5]*q)>>16 + off[j>>5]` — and assert it equals
//! `decode_tensor_fixed` bit-for-bit. That host-walk is the exact arithmetic
//! the Metal `strand_bitslice_decode` kernel performs, so a green Slice-0 test
//! is strong evidence the GPU dispatch (Slices 2–3) will be bit-identical
//! before any Metal device is touched.
//!
//! ## The 84-byte record landmine
//!
//! `BitsliceEntry` is `#[repr(C)]` `{bit_offset, init_state, out_off, n}` (4×u32)
//! `+ eff[8]` (8×i32) `+ off[8]` (8×i32) `+ d` (u32) = **84 bytes**, align 4.
//! The GPU dispatch path (Slice 2) carries the runtime
//! `strand_bitslice_entry_sizeof` probe and asserts it `== size_of::<BitsliceEntry>()`
//! — the size is NEVER hardcoded, so the host/GPU table stride can never silently
//! diverge.

use strand_quant::decode::{eff_min_q, eff_scale_q};
use strand_quant::encode::{
    n_sub_blocks, unpack_sub_scales, unpack_sub_scales_or_unity, EncodedTensor,
};
use strand_quant::trellis::read_bits;
use strand_quant::TrellisConfig;

/// Per-block GPU decode record — the host mirror of the MSL `BitsliceEntry`
/// struct in `shaders/strand_bitslice.metal`. `#[repr(C)]`, 84 bytes, 4-byte
/// aligned, little-endian: every field is `u32`/`i32` so the Rust layout and the
/// Metal `sizeof` agree without padding. **The stride is probe-asserted at GPU
/// dispatch (`size_of::<BitsliceEntry>()` == `strand_bitslice_entry_sizeof`),
/// never hardcoded.**
///
/// Field semantics (mirrors `strand-decode-kernel/src/metal.rs::BitsliceEntry`):
/// - `bit_offset`: absolute bit position of this block's first k-bit symbol in
///   the tensor's flat payload (`buffer(0)`, bound at the tensor's byte offset).
/// - `init_state`: baked start state — the host did the tail-bite prescan
///   ([`block_init_state`]); the kernel NEVER prescans.
/// - `out_off`: first output index = prefix sum of `n` over preceding blocks.
/// - `n`: weights in this block (≤ 256; the last block of a row may be short).
/// - `eff[8]`: pre-expanded effective sub-scales, Q16 (`eff_scale_q`), one per 32
///   weights. Slots past `n_sub` are 0.
/// - `off[8]`: pre-expanded affine offsets (`eff_min_q`); ALL ZERO when affine-min
///   is off — the `+0` is bit-exact, so ONE kernel covers both encode branches.
/// - `d`: vector dim (1 = scalar). Kept last so the scalar kernels read the same
///   leading bytes; `d > 1` selects the `_vec` kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BitsliceEntry {
    pub bit_offset: u32,
    pub init_state: u32,
    pub out_off: u32,
    pub n: u32,
    pub eff: [i32; 8],
    pub off: [i32; 8],
    pub d: u32,
}

/// Compute-for-memory alternative to [`BitsliceEntry`]. It keeps the four seek
/// fields but stores the block scale plus eight raw sub-scale/min codes; the GPU
/// expands one `(eff, off)` pair per 32-weight sub-block. This cuts the streamed
/// runtime table from 84 to 40 bytes/block without changing `.tq` bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompactBitsliceEntry {
    pub bit_offset: u32,
    pub init_state: u32,
    pub out_off: u32,
    pub n: u32,
    pub scale_q: i32,
    pub min_base_q: i32,
    /// Eight u8 codes packed four per word. Only the low six bits are live.
    pub mult_codes: [u32; 2],
    /// Eight signed-magnitude affine-min codes, or zero when affine-min is off.
    pub min_codes: [u32; 2],
}

fn pack_eight_codes(codes: &[u8]) -> [u32; 2] {
    let mut bytes = [0u8; 8];
    for (dst, &src) in bytes.iter_mut().zip(codes.iter()) {
        *dst = src & 0x3f;
    }
    [
        u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..].try_into().unwrap()),
    ]
}

/// Per-block bitstream plan: where this block's symbols start, where its outputs
/// land, and how many it has. Scalar form (`d == 1`): `start_bit` accumulates
/// `n * k`, `out_off` accumulates `n`. Mirrors
/// `strand-decode-kernel/src/block_walk.rs::block_plans`.
#[derive(Clone, Copy)]
struct BlockPlan {
    start_bit: usize,
    out_off: usize,
    n: usize,
}

/// Prefix-sum the per-block `(start_bit, out_off, n)` plan for the scalar path.
/// Verbatim port of `block_walk::block_plans`.
fn block_plans(enc: &EncodedTensor, k: usize) -> Vec<BlockPlan> {
    let mut plans = Vec::with_capacity(enc.blocks.len());
    let mut prefix_n = 0usize;
    for blk in &enc.blocks {
        let n = blk.n as usize;
        plans.push(BlockPlan {
            start_bit: prefix_n * k,
            out_off: prefix_n,
            n,
        });
        prefix_n += n;
    }
    plans
}

/// Recover a block's start state. Under tail-biting (and only when the block is
/// long enough — `n*k >= l_bits`) the encoder seeds the state from a full
/// forward walk of the block's own symbols rather than storing it, so the bake
/// replays that walk here; otherwise the stored `init_state` is used. Doing it
/// at bake means the kernel NEVER prescans. Verbatim port of
/// `block_walk::block_init_state` (scalar path).
fn block_init_state(
    blk: &strand_quant::encode::BlockMeta,
    bits: &[u8],
    start_bit: usize,
    cfg: &TrellisConfig,
    tail_biting: bool,
) -> usize {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let n = blk.n as usize;
    let nk = n * (k as usize);
    if tail_biting && nk >= cfg.l_bits as usize {
        let mut s = 0usize;
        let mut c = start_bit;
        for _ in 0..n {
            let sym = read_bits(bits, c, k) & input_mask;
            c += k as usize;
            s = ((s << k) | sym) & mask;
        }
        s
    } else {
        blk.init_state as usize & mask
    }
}

/// Hoisted per-block side info: the pre-expanded effective sub-scales (`eff`) and
/// affine offsets (`off`), one per 32-weight sub-block. Mirrors
/// `block_walk::SideInfo::hoist`, but folded on top of strand-quant's public
/// `unpack_sub_scales` / `eff_scale_q` / `eff_min_q` — the bake does not fork the
/// integer math. The two arrays carry at most 8 live entries (`n_sub`) since a
/// 256-weight block has 8 sub-blocks.
struct SideInfo {
    eff: [i32; 8],
    off: [i32; 8],
    n_sub: usize,
}

impl SideInfo {
    fn hoist(blk: &strand_quant::encode::BlockMeta, has_affine: bool) -> Self {
        let n_sub = n_sub_blocks(blk.n as usize);
        debug_assert!(
            n_sub <= 8,
            "bitslice bake: block has {n_sub} sub-blocks (> 8); n={} exceeds the 256-weight block",
            blk.n
        );
        let mut eff = [0i32; 8];
        let mut off = [0i32; 8];
        let mults = unpack_sub_scales_or_unity(&blk.sub_scales, n_sub);
        for (e, &m) in eff[..n_sub].iter_mut().zip(mults.iter()) {
            *e = eff_scale_q(blk.scale_q, m);
        }
        if has_affine {
            let codes = unpack_sub_scales(&blk.mins, n_sub);
            for (o, &c) in off[..n_sub].iter_mut().zip(codes.iter()) {
                *o = eff_min_q(blk.min_base_q, c);
            }
        }
        SideInfo { eff, off, n_sub }
    }
}

/// Bake the per-block [`BitsliceEntry`] table for the **scalar** (`vec_dim == 1`)
/// path — a verbatim port of `strand-decode-kernel/src/metal.rs::bake_bitslice_entries`.
///
/// Returns `None` if any block has `n > 256` (the kernel's per-thread block-stream
/// assumption; the strand reference falls back to the CPU lean decoder there).
///
/// Each record is built ONCE at model load: prefix-sum `out_off`/`bit_offset` via
/// [`block_plans`], `eff[8]`/`off[8]` via [`SideInfo::hoist`], and tail-biting
/// `init_state` recovery via [`block_init_state`]. `d` is hardcoded to 1 — the
/// vector path is [`bake_bitslice_entries_vec`].
pub(crate) fn bake_bitslice_entries(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Option<Vec<BitsliceEntry>> {
    if enc.blocks.iter().any(|b| b.n > 256) {
        return None;
    }
    let k = cfg.k_bits as usize;
    let plans = block_plans(enc, k);
    let mut out = Vec::with_capacity(enc.blocks.len());
    for (blk, plan) in enc.blocks.iter().zip(plans.iter()) {
        let side = SideInfo::hoist(blk, enc.has_affine_min);
        let mut eff = [0i32; 8];
        let mut off = [0i32; 8];
        eff[..side.n_sub].copy_from_slice(&side.eff[..side.n_sub]);
        off[..side.n_sub].copy_from_slice(&side.off[..side.n_sub]);
        let init = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
        // The plan's per-block count and the block's own `n` are the same value by
        // construction (the plan is a prefix sum over `blk.n`); assert it so the
        // two derivations can never silently drift.
        debug_assert_eq!(plan.n, blk.n as usize, "block_plans n != BlockMeta.n");
        out.push(BitsliceEntry {
            bit_offset: plan.start_bit as u32,
            init_state: init as u32,
            out_off: plan.out_off as u32,
            n: blk.n,
            eff,
            off,
            d: 1,
        });
    }
    Some(out)
}

/// Bake the compact compute-for-memory table. Addressing and tail-biting state
/// are inherited from the expanded bake; only the representation of scale/min
/// side information changes. An absent sub-scale stream means exact unity (63),
/// matching the canonical CPU decoder rather than decoding as zero.
pub(crate) fn bake_compact_bitslice_entries(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Option<Vec<CompactBitsliceEntry>> {
    if cfg.vec_dim() != 1 {
        return None;
    }
    let expanded = bake_bitslice_entries(enc, cfg)?;
    let mut out = Vec::with_capacity(expanded.len());
    for (seek, blk) in expanded.iter().zip(enc.blocks.iter()) {
        let n_sub = n_sub_blocks(blk.n as usize);
        if n_sub > 8 {
            return None;
        }
        let mults = unpack_sub_scales_or_unity(&blk.sub_scales, n_sub);
        let mins = if enc.has_affine_min {
            unpack_sub_scales(&blk.mins, n_sub)
        } else {
            Vec::new()
        };
        out.push(CompactBitsliceEntry {
            bit_offset: seek.bit_offset,
            init_state: seek.init_state,
            out_off: seek.out_off,
            n: seek.n,
            scale_q: blk.scale_q,
            min_base_q: if enc.has_affine_min {
                blk.min_base_q
            } else {
                0
            },
            mult_codes: pack_eight_codes(&mults),
            min_codes: pack_eight_codes(&mins),
        });
    }
    Some(out)
}

/// Vector-trellis (`d > 1`) bake. Differs from the scalar bake only in the
/// bitstream arithmetic: each block consumes `n_steps = ceil(n/d)` symbols (so
/// `start_bit` accumulates `n_steps * k`, NOT `n * k`), while `out_off` still
/// accumulates the true output count `n`; the tail-biting init walk reads
/// `n_steps` symbols. Mirrors
/// `strand-decode-kernel/src/block_walk.rs::bake_bitslice_entries_vec` (and the
/// CPU `decode_tensor_fixed_with_lut_vec`). Carried for completeness — the model
/// path asserts `d == 1`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn bake_bitslice_entries_vec(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Option<Vec<BitsliceEntry>> {
    if enc.blocks.iter().any(|b| b.n > 256) {
        return None;
    }
    let d = cfg.vec_dim();
    let k = cfg.k_bits as usize;
    let mask = cfg.state_mask();
    let input_mask = cfg.num_inputs() - 1;

    let mut out = Vec::with_capacity(enc.blocks.len());
    let mut start_bit = 0usize;
    let mut out_off = 0usize;
    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_steps = n.div_ceil(d);

        let side = SideInfo::hoist(blk, enc.has_affine_min);
        let mut eff = [0i32; 8];
        let mut off = [0i32; 8];
        eff[..side.n_sub].copy_from_slice(&side.eff[..side.n_sub]);
        off[..side.n_sub].copy_from_slice(&side.off[..side.n_sub]);

        let nk = n_steps * k;
        let init = if enc.tail_biting && nk >= cfg.l_bits as usize {
            let mut s = 0usize;
            let mut c = start_bit;
            for _ in 0..n_steps {
                let sym = read_bits(&enc.bits, c, cfg.k_bits) & input_mask;
                c += k;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        out.push(BitsliceEntry {
            bit_offset: start_bit as u32,
            init_state: init as u32,
            out_off: out_off as u32,
            n: blk.n,
            eff,
            off,
            d: d as u32,
        });
        start_bit += n_steps * k;
        out_off += n;
    }
    Some(out)
}

/// GPU bitslice decode of an encoded tensor to its Q12 weights, bit-identical to
/// `strand_quant::decode::decode_tensor_fixed` — the public, `BitsliceEntry`-free
/// entry point (bakes the table, then dispatches `strand_bitslice_decode`).
///
/// Returns `None` if the bake is rejected (a block with `n > 256`) or `vec_dim > 1`
/// (the scalar kernel only); callers fall back to the CPU decode there. The GPU
/// dispatch itself carries the runtime `sizeof(BitsliceEntry)` probe and errors if
/// the host/GPU table stride disagrees. macOS + `tq` only.
#[cfg(target_os = "macos")]
pub fn gpu_decode_q12(
    ctx: &crate::metal::MetalContext,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Option<crate::Result<Vec<i32>>> {
    if cfg.vec_dim() > 1 {
        return None;
    }
    let tbl = bake_bitslice_entries(enc, cfg)?;
    // The codebook the CPU oracle uses (== codebook_lut(l_bits) for StoredLut).
    let lut = cfg.codebook();
    Some(crate::kernels::decode_strand_bitslice(
        ctx, &enc.bits, &tbl, &lut, enc.total, cfg.k_bits, cfg.l_bits,
    ))
}

/// Runtime-only interpretation of the same `.tq` bytes. The default remains the
/// proven stored/expanded path; alternatives are explicit experiments selected
/// with `HAWKING_TQ_RUNTIME_PATH` and never alter an archive or encode result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TqRuntimePath {
    #[default]
    Stored,
    /// Store raw 6-bit scale/min codes in a 40-byte block record and expand them
    /// once per 32 weights in the GPU kernel (84 -> 40 bytes/block).
    CompactMetadata,
    /// Compute state->rank and gather an i16 monotone quantile table instead of
    /// gathering an i32 state-indexed codebook.
    HashedQuantile,
    /// Compute the central Gaussian codebook value with integer Acklam arithmetic;
    /// only the small exact tail prefix remains stored.
    ComputedAcklam,
}

/// Runtime metadata representation, independent of codebook sourcing. Keeping
/// this axis separate lets the Appendix account for compound candidates before
/// a corresponding Metal kernel is admitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TqMetadataMode {
    #[default]
    Expanded,
    Compact,
}

/// Runtime codebook source, independent of the per-block metadata layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TqCodebookSource {
    #[default]
    Stored,
    HashedQuantile,
    ComputedAcklam,
}

/// Orthogonal accounting recipe for the same `.tq` bytes. Only recipes returned
/// by [`TqRuntimePath::recipe`] are executable today; `COMPACT_HASHED` and
/// `COMPACT_COMPUTED` deliberately expose the next compound experiments without
/// pretending their Metal kernels already exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TqRuntimeRecipe {
    pub metadata: TqMetadataMode,
    pub codebook: TqCodebookSource,
}

impl TqRuntimeRecipe {
    pub const STORED: Self = Self {
        metadata: TqMetadataMode::Expanded,
        codebook: TqCodebookSource::Stored,
    };
    pub const COMPACT: Self = Self {
        metadata: TqMetadataMode::Compact,
        codebook: TqCodebookSource::Stored,
    };
    pub const HASHED: Self = Self {
        metadata: TqMetadataMode::Expanded,
        codebook: TqCodebookSource::HashedQuantile,
    };
    pub const COMPUTED: Self = Self {
        metadata: TqMetadataMode::Expanded,
        codebook: TqCodebookSource::ComputedAcklam,
    };
    pub const COMPACT_HASHED: Self = Self {
        metadata: TqMetadataMode::Compact,
        codebook: TqCodebookSource::HashedQuantile,
    };
    pub const COMPACT_COMPUTED: Self = Self {
        metadata: TqMetadataMode::Compact,
        codebook: TqCodebookSource::ComputedAcklam,
    };

    pub const RESEARCH_MATRIX: [Self; 6] = [
        Self::STORED,
        Self::COMPACT,
        Self::HASHED,
        Self::COMPUTED,
        Self::COMPACT_HASHED,
        Self::COMPACT_COMPUTED,
    ];
}

impl TqRuntimePath {
    pub const ENV: &'static str = "HAWKING_TQ_RUNTIME_PATH";

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "stored" | "baseline" | "off" | "0" => Ok(Self::Stored),
            "compact" | "compact-metadata" => Ok(Self::CompactMetadata),
            "hashed" | "hashed-quantile" => Ok(Self::HashedQuantile),
            "computed" | "computed-acklam" | "acklam" => Ok(Self::ComputedAcklam),
            other => Err(format!(
                "{name}={other:?} is invalid; expected stored|compact|hashed|computed",
                name = Self::ENV
            )),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        match std::env::var(Self::ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::Stored),
            Err(err) => Err(format!("{} is not valid Unicode: {err}", Self::ENV)),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::CompactMetadata => "compact",
            Self::HashedQuantile => "hashed",
            Self::ComputedAcklam => "computed",
        }
    }

    pub const fn fused_kernel_name(self) -> &'static str {
        match self {
            Self::Stored => "strand_bitslice_gemv_partials",
            Self::CompactMetadata => "strand_bitslice_gemv_partials_compact",
            Self::HashedQuantile => "strand_bitslice_gemv_partials_hashed",
            Self::ComputedAcklam => "strand_bitslice_gemv_partials_computed",
        }
    }

    /// Batch-major fused kernel used by the speculative verifier for B=1..=8.
    /// These names deliberately remain path-specific so receipts can bind the
    /// measured runtime interpretation, not merely the shared `.tq` artifact.
    pub const fn small_batch_kernel_name(self) -> &'static str {
        match self {
            Self::Stored => "strand_bitslice_gemm_small_stored",
            Self::CompactMetadata => "strand_bitslice_gemm_small_compact",
            Self::HashedQuantile => "strand_bitslice_gemm_small_hashed",
            Self::ComputedAcklam => "strand_bitslice_gemm_small_computed",
        }
    }

    pub const fn recipe(self) -> TqRuntimeRecipe {
        match self {
            Self::Stored => TqRuntimeRecipe::STORED,
            Self::CompactMetadata => TqRuntimeRecipe::COMPACT,
            Self::HashedQuantile => TqRuntimeRecipe::HASHED,
            Self::ComputedAcklam => TqRuntimeRecipe::COMPUTED,
        }
    }
}

/// Static byte-accounting for one prepared projection. These are logical device
/// reads/writes, not a cache-hit claim: the counters make the hidden metadata,
/// codebook staging, and partial-reduction traffic visible beside payload bpw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TqRuntimeTraffic {
    pub weights: usize,
    pub blocks: usize,
    pub payload_bytes: usize,
    pub expanded_table_bytes: usize,
    pub compact_table_bytes: usize,
    pub stored_codebook_bytes: usize,
    pub hashed_quantile_bytes: usize,
    pub computed_tail_bytes: usize,
    pub threadgroups: usize,
    /// One f32 partial write plus one f32 reduction read per block.
    pub partial_roundtrip_bytes: usize,
}

/// Stable, machine-readable reason a prepared projection cannot use the current
/// fused scalar TQ GEMV. This is surfaced for corpus census and autotuning rather
/// than collapsing every miss into a generic CPU fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TqGpuIneligibility {
    ColumnsNotMultipleOf256,
    ExpandedBlockCountMismatch,
    CompactBlockCountMismatch,
    BlockLengthNot256 { block: usize },
    VectorDimensionNotScalar { block: usize },
    OutputOffsetMismatch { block: usize },
    BitOffsetMismatch { block: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TqGpuAdmission {
    pub eligible: bool,
    pub reason: Option<TqGpuIneligibility>,
    pub rows: usize,
    pub cols: usize,
    pub blocks: usize,
    pub expected_blocks: usize,
}

impl TqRuntimeTraffic {
    pub fn metadata_bytes_for(self, recipe: TqRuntimeRecipe) -> usize {
        match recipe.metadata {
            TqMetadataMode::Expanded => self.expanded_table_bytes,
            TqMetadataMode::Compact => self.compact_table_bytes,
        }
    }

    pub fn metadata_bytes(self, path: TqRuntimePath) -> usize {
        self.metadata_bytes_for(path.recipe())
    }

    pub fn staged_codebook_bytes_for(self, recipe: TqRuntimeRecipe) -> usize {
        let per_group = match recipe.codebook {
            TqCodebookSource::Stored => self.stored_codebook_bytes,
            TqCodebookSource::HashedQuantile => self.hashed_quantile_bytes,
            TqCodebookSource::ComputedAcklam => self.computed_tail_bytes,
        };
        per_group.saturating_mul(self.threadgroups)
    }

    pub fn staged_codebook_bytes(self, path: TqRuntimePath) -> usize {
        self.staged_codebook_bytes_for(path.recipe())
    }

    /// Payload, runtime metadata, staged codebook, and the partial-buffer
    /// write/read roundtrip. Activation and output bytes remain separate because
    /// their physical cache behavior depends on projection scheduling.
    pub fn compressed_runtime_bytes_for(self, recipe: TqRuntimeRecipe) -> usize {
        self.payload_bytes
            .saturating_add(self.metadata_bytes_for(recipe))
            .saturating_add(self.staged_codebook_bytes_for(recipe))
            .saturating_add(self.partial_roundtrip_bytes)
    }

    pub fn compressed_runtime_bpw_for(self, recipe: TqRuntimeRecipe) -> f64 {
        if self.weights == 0 {
            0.0
        } else {
            self.compressed_runtime_bytes_for(recipe) as f64 * 8.0 / self.weights as f64
        }
    }

    /// Logical compressed-weight-side traffic for one projection invocation.
    /// Activation reads and output writes are intentionally reported elsewhere.
    pub fn weight_path_bytes(self, path: TqRuntimePath) -> usize {
        self.payload_bytes
            .saturating_add(self.metadata_bytes(path))
            .saturating_add(self.staged_codebook_bytes(path))
    }

    pub fn weight_path_bpw(self, path: TqRuntimePath) -> f64 {
        if self.weights == 0 {
            0.0
        } else {
            self.weight_path_bytes(path) as f64 * 8.0 / self.weights as f64
        }
    }
}

/// Host-side preparation record for GPU dispatch of a single STRAND-encoded
/// projection. Bundles the payload bytes, the baked [`BitsliceEntry`] seek table,
/// the frozen Q12 codebook LUT, and the per-tensor metadata the Metal kernel needs
/// at dispatch time — everything that can be computed once at model-load rather than
/// per-token.
///
/// Construct via [`TqPreparedGpu::from_strand_tensor`]; the stub currently returns
/// `Err(Unimplemented)` and will be filled in when Slice-2 GPU dispatch lands.
#[derive(Debug)]
pub struct TqPreparedGpu {
    /// Raw bitstream bytes (`EncodedTensor::bits`) — bound as `buffer(0)` on the GPU.
    pub payload: Vec<u8>,
    /// Baked per-block seek table, one [`BitsliceEntry`] per block.
    pub entries: Vec<BitsliceEntry>,
    /// Compact compute-for-memory twin of `entries`. It addresses the identical
    /// payload and is selected only by the explicit runtime policy.
    pub compact_entries: Vec<CompactBitsliceEntry>,
    /// Frozen Q12 Gaussian codebook LUT (`cfg.codebook()`), `2^l_bits` entries.
    pub lut_q12: Vec<i32>,
    /// Trellis k parameter (bits read per step from the bitstream).
    pub k_bits: u32,
    /// Trellis L parameter (state-space width = `2^l_bits`).
    pub l_bits: u32,
    /// Output features (rows).
    pub rows: usize,
    /// Input features (cols).
    pub cols: usize,
    /// RHT mode encoded as an integer: 0 = None, 1 = Rows, 2 = Cols.
    pub rht_mode: u32,
    /// Per-tensor RHT seed (meaningful when `rht_mode != 0`).
    pub rht_seed: u64,
    /// OUTL outlier sparse corrections, pre-resolved against the decoded bulk:
    /// `(row, col, resid)` where `resid = outlier_val − decode(bulk_unrotated)[idx]`
    /// in f32 weight units. The served weight is the bulk bitslice GEMV plus
    /// `y[row] += resid * x_raw[col]` (the un-rotated activation), mirroring
    /// `outlier_mac::matvec_rht`. Empty when the tensor has no OUTL section.
    pub outliers: Vec<OutlierEntry>,
    /// Informational effective bits-per-weight for this tensor.
    pub bpw: f32,
}

/// Host mirror of the MSL `OutlierEntry` (`shaders/strand_bitslice.metal`):
/// `#[repr(C)]` `{ row:u32, col:u32, resid:f32 }` = 12 bytes, 4-aligned. The
/// sparse outlier correction `y[row] += resid * x_raw[col]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlierEntry {
    pub row: u32,
    pub col: u32,
    pub resid: f32,
}

impl TqPreparedGpu {
    /// Prepare a [`StrandTensor`] for GPU dispatch.
    ///
    /// Bakes the [`BitsliceEntry`] seek table and freezes the codebook LUT so the
    /// Metal kernel can be dispatched directly from the returned record. Currently
    /// a stub — returns `Err(Unimplemented)` until the full Slice-2 GPU dispatch
    /// path lands. The function signature and field set are final.
    pub fn from_strand_tensor(st: &crate::tq::StrandTensor) -> crate::Result<Self> {
        let cfg = &st.cfg;

        // Step 1: bake the per-block BitsliceEntry seek table.
        // bake_bitslice_entries returns None only when a block has n > 256, which
        // the GPU scalar kernel does not support; treat that as Unimplemented.
        let entries = bake_bitslice_entries(&st.enc, cfg).ok_or(
            crate::Error::Unimplemented(
                "TqPreparedGpu::from_strand_tensor: tensor has a block with n > 256 (vec_dim > 1 or oversized block — scalar GPU kernel unsupported)",
            ),
        )?;
        let compact_entries = bake_compact_bitslice_entries(&st.enc, cfg).ok_or(
            crate::Error::Unimplemented(
                "TqPreparedGpu::from_strand_tensor: compact table requires scalar blocks no larger than 256 weights",
            ),
        )?;

        // Step 2: freeze the Q12 Gaussian codebook LUT (2^l_bits entries).
        // cfg.codebook() returns Cow<'static, [i32]>; .into_owned() is zero-copy
        // for the StoredLut (Borrowed) path and allocates only for the Computed path.
        let lut_q12 = cfg.codebook().into_owned();

        // Step 3: encode RhtMode as 0/1/2.
        let rht_mode = match st.rht_mode {
            crate::tq::RhtMode::None => 0u32,
            crate::tq::RhtMode::Rows => 1u32,
            crate::tq::RhtMode::Cols => 2u32,
        };

        // Step 4: copy the raw bitstream payload bytes.
        let payload = st.enc.bits.clone();

        // Step 5: informational bpw = k_bits / vec_dim (payload-only; matches
        // EncodedTensor::payload_bpw but doesn't require accessing that method).
        let bpw = cfg.k_bits as f32 / cfg.vec_dim() as f32;

        // Step 6: pre-resolve OUTL outlier corrections against the decoded bulk.
        // The served weight is the bulk bitslice GEMV plus a sparse correction
        // `y[row] += resid * x_raw[col]` with `resid = outlier_val −
        // decode(bulk_unrotated)[idx]` — exactly `outlier_mac::matvec_rht`'s
        // residual loop (and the `StrandTensor::matvec` un-rotated overwrite,
        // re-expressed as base + sparse term). For `RhtMode::None` the raw decode
        // is already un-rotated; for `Cols`/`Rows` it must be inverse-rotated once
        // to recover the bulk in the domain the outlier value is defined in.
        let outliers = build_outlier_entries(st);

        Ok(TqPreparedGpu {
            payload,
            entries,
            compact_entries,
            lut_q12,
            k_bits: cfg.k_bits,
            l_bits: cfg.l_bits,
            rows: st.out_features,
            cols: st.in_features,
            rht_mode,
            rht_seed: st.rht_seed,
            outliers,
            bpw,
        })
    }

    #[allow(dead_code)] // exercised by gates; production ledger wiring follows post-ladder
    pub fn runtime_traffic(&self) -> TqRuntimeTraffic {
        let blocks = self.entries.len();
        let threadgroups = blocks.div_ceil(256);
        TqRuntimeTraffic {
            weights: self.rows.saturating_mul(self.cols),
            blocks,
            payload_bytes: self.payload.len(),
            expanded_table_bytes: blocks.saturating_mul(std::mem::size_of::<BitsliceEntry>()),
            compact_table_bytes: blocks.saturating_mul(std::mem::size_of::<CompactBitsliceEntry>()),
            stored_codebook_bytes: (1usize << self.l_bits)
                .saturating_mul(std::mem::size_of::<i32>()),
            hashed_quantile_bytes: (1usize << self.l_bits)
                .saturating_mul(std::mem::size_of::<i16>()),
            computed_tail_bytes: strand_quant::codebook::tail_left_prefix_q12(self.l_bits)
                .len()
                .saturating_mul(std::mem::size_of::<i32>()),
            threadgroups,
            partial_roundtrip_bytes: blocks.saturating_mul(2 * std::mem::size_of::<f32>()),
        }
    }

    /// Explain whether the existing fused GPU reducer can serve this projection.
    /// Decode-only kernels are more general; this gate is specifically for the
    /// row-major scalar 256-weight GEMV path used during token generation.
    pub fn gpu_admission(&self) -> TqGpuAdmission {
        assess_gpu_gemv_geometry(
            &self.entries,
            self.compact_entries.len(),
            self.rows,
            self.cols,
            self.k_bits,
        )
    }
}

fn assess_gpu_gemv_geometry(
    entries: &[BitsliceEntry],
    compact_entries: usize,
    rows: usize,
    cols: usize,
    k_bits: u32,
) -> TqGpuAdmission {
    let expected_blocks = if cols % 256 == 0 {
        rows.saturating_mul(cols / 256)
    } else {
        rows.saturating_mul(cols.div_ceil(256))
    };
    let reason = if cols % 256 != 0 {
        Some(TqGpuIneligibility::ColumnsNotMultipleOf256)
    } else if entries.len() != expected_blocks {
        Some(TqGpuIneligibility::ExpandedBlockCountMismatch)
    } else if compact_entries != entries.len() {
        Some(TqGpuIneligibility::CompactBlockCountMismatch)
    } else {
        entries.iter().enumerate().find_map(|(block, entry)| {
            if entry.n != 256 {
                Some(TqGpuIneligibility::BlockLengthNot256 { block })
            } else if entry.d != 1 {
                Some(TqGpuIneligibility::VectorDimensionNotScalar { block })
            } else if entry.out_off as usize != block * 256 {
                Some(TqGpuIneligibility::OutputOffsetMismatch { block })
            } else if entry.bit_offset as usize != block * 256 * k_bits as usize {
                Some(TqGpuIneligibility::BitOffsetMismatch { block })
            } else {
                None
            }
        })
    };
    TqGpuAdmission {
        eligible: reason.is_none(),
        reason,
        rows,
        cols,
        blocks: entries.len(),
        expected_blocks,
    }
}

/// Resolve a `StrandTensor`'s OUTL section into `(row, col, resid)` sparse
/// corrections in the un-rotated weight domain — the GPU twin of the
/// `StrandTensor::matvec` outlier overwrite, re-expressed as a sparse ADD on top
/// of the bulk GEMV (`y[row] += resid * x_raw[col]`). `resid = outlier_val −
/// decode(bulk_unrotated)[idx]`; the outlier value is the Q12-quantised stored
/// value (`q12_value * inv`), matching `StrandTensor::matvec`'s `v as f32 * inv`
/// (NOT the raw OUTL float — keeps the determinism moat on the Q12 grid).
#[cfg(feature = "tq")]
fn build_outlier_entries(st: &crate::tq::StrandTensor) -> Vec<OutlierEntry> {
    if st.outliers.is_empty() {
        return Vec::new();
    }
    let inv = crate::tq::q12_to_f32();
    let in_features = st.in_features;
    // Bulk weights in the UN-rotated domain (the domain outliers are defined in).
    let mut bulk: Vec<f32> = st
        .decode_q12_raw()
        .iter()
        .map(|&q| q as f32 * inv)
        .collect();
    match st.rht_mode {
        crate::tq::RhtMode::None => {}
        crate::tq::RhtMode::Cols => {
            let rcfg = strand_quant::rht::RhtConfig::from_seed(st.rht_seed);
            strand_quant::rht::rht_inverse_cols_inplace(&mut bulk, &rcfg, in_features);
        }
        crate::tq::RhtMode::Rows => {
            let rcfg = strand_quant::rht::RhtConfig::from_seed(st.rht_seed);
            strand_quant::rht::rht_inverse_rows_inplace(&mut bulk, &rcfg, in_features);
        }
    }
    st.outliers
        .iter()
        .filter_map(|&(idx, q12_val)| {
            if idx >= bulk.len() {
                return None;
            }
            let row = (idx / in_features) as u32;
            let col = (idx % in_features) as u32;
            let resid = q12_val as f32 * inv - bulk[idx];
            Some(OutlierEntry { row, col, resid })
        })
        .collect()
}

/// GPU-resident pre-uploaded buffers and dispatch constants for one STRAND
/// projection. Built once at model-load via [`TqPreparedGpu::upload_to_gpu`];
/// every decode step binds the already-live Metal buffers with no allocation.
#[cfg(target_os = "macos")]
pub struct TqGpuReady {
    /// Selected expanded or compact per-block table, uploaded to GPU.
    pub tbl_buf: crate::metal::PinnedBuffer,
    /// Selected codebook source: i32 codebook, i16 quantiles, or i32 tail prefix.
    pub lut_buf: crate::metal::PinnedBuffer,
    pub runtime_path: TqRuntimePath,
    /// Padded payload bitstream, uploaded to GPU.
    pub w_buf: crate::metal::PinnedBuffer,
    /// Scratch partial-dot-products (8 × n_blocks × f32), pre-allocated. GEMV
    /// uses the first plane; speculative verification uses one plane per token
    /// for its hard B<=8 contract.
    pub partials_buf: crate::metal::PinnedBuffer,
    pub n_blocks: u32,
    pub cols: u32,
    pub rows: u32,
    pub k_bits: u32,
    pub l_bits: u32,
    pub bpr: u32,
    pub shmem_bytes: u64,
    pub n_tg_partials: u32,
    pub n_tg_reduce: u32,
    /// RHT serving mode (0 None / 1 Rows / 2 Cols). When `Cols`, the GEMV runs
    /// `strand_rht_forward_cols` over the input activation first (`x → rht_x_buf`)
    /// and the partials pass reads `rht_x_buf` instead of the raw `x`.
    pub rht_mode: u32,
    /// Per-tensor RHT seed (meaningful when `rht_mode != 0`).
    pub rht_seed: u64,
    /// Scratch for RHT-transformed activations (8 × cols × f32). `Some` only
    /// when `rht_mode == 2` (Cols). GEMV uses the first row; speculative verify
    /// uses the batch-major prefix.
    pub rht_x_buf: Option<crate::metal::PinnedBuffer>,
    /// `cols / 256` — number of 256-wide Hadamard blocks the RHT kernel dispatches
    /// (one thread per block). Only meaningful with `rht_x_buf`.
    pub rht_n_blocks: u32,
    /// OUTL sparse-correction entries (`OutlierEntry`), uploaded to GPU. Empty
    /// buffer + `n_outl == 0` when the tensor has no outliers.
    pub outl_buf: crate::metal::PinnedBuffer,
    pub n_outl: u32,
}

/// Artifact-probe harness for one real TQ projection. Preparation/uploads happen
/// once; [`Self::run_gemv`] then exercises the same Hawking-core TCB dispatch as
/// model serving without re-uploading weights or allocating per trial.
#[cfg(target_os = "macos")]
pub struct TqDeviceHarness {
    prepared: TqPreparedGpu,
    ready: TqGpuReady,
    x_buf: crate::metal::PinnedBuffer,
    out_buf: crate::metal::PinnedBuffer,
    pub runtime_path: TqRuntimePath,
    pub traffic: TqRuntimeTraffic,
    pub admission: TqGpuAdmission,
    pub rows: usize,
    pub cols: usize,
    pub blocks: usize,
    pub k_bits: u32,
    pub l_bits: u32,
    pub host_entry_bytes: usize,
    pub gpu_entry_bytes: usize,
}

#[cfg(target_os = "macos")]
impl TqDeviceHarness {
    /// Build the device harness from a parsed `.tq` tensor and a deterministic
    /// activation. This fails closed on unsupported geometry or record-stride
    /// disagreement before any benchmark sample can be emitted.
    pub fn prepare(
        ctx: &crate::metal::MetalContext,
        tensor: &crate::tq::StrandTensor,
        runtime_path: TqRuntimePath,
        activation: &[f32],
    ) -> crate::Result<Self> {
        if activation.len() != tensor.in_features {
            return Err(crate::Error::Model(format!(
                "TQ device harness activation length {} != tensor cols {}",
                activation.len(),
                tensor.in_features
            )));
        }
        let prepared = TqPreparedGpu::from_strand_tensor(tensor)?;
        let traffic = prepared.runtime_traffic();
        let admission = prepared.gpu_admission();
        if !admission.eligible {
            return Err(crate::Error::Unimplemented(
                "TQ device harness requires admitted canonical fused-GEMV geometry",
            ));
        }
        let (host_entry_bytes, gpu_entry_bytes) = if runtime_path == TqRuntimePath::CompactMetadata
        {
            (
                std::mem::size_of::<CompactBitsliceEntry>(),
                crate::kernels::strand_bitslice_compact_entry_sizeof(ctx)? as usize,
            )
        } else {
            (
                std::mem::size_of::<BitsliceEntry>(),
                crate::kernels::strand_bitslice_entry_sizeof(ctx)? as usize,
            )
        };
        if host_entry_bytes != gpu_entry_bytes {
            return Err(crate::Error::Kernel(format!(
                "TQ device harness record stride mismatch: host={host_entry_bytes}, GPU={gpu_entry_bytes}"
            )));
        }
        let ready = prepared.upload_to_gpu_with_path(ctx, runtime_path)?;
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(activation));
        let out_buf = ctx.new_buffer(
            tensor
                .out_features
                .max(1)
                .saturating_mul(std::mem::size_of::<f32>()),
        );
        Ok(Self {
            blocks: prepared.entries.len(),
            rows: prepared.rows,
            cols: prepared.cols,
            k_bits: prepared.k_bits,
            l_bits: prepared.l_bits,
            prepared,
            ready,
            x_buf,
            out_buf,
            runtime_path,
            traffic,
            admission,
            host_entry_bytes,
            gpu_entry_bytes,
        })
    }

    /// Decode the selected runtime representation directly to Q12 on Metal.
    /// Stored, compact, hashed, and computed paths each use their own decode-only
    /// kernel so exactness is established before float GEMV reduction.
    pub fn decode_q12(&self, ctx: &crate::metal::MetalContext) -> crate::Result<Vec<i32>> {
        self.prepared
            .decode_q12_on_gpu_with_path(ctx, self.runtime_path)
    }

    /// Execute one fused GEMV trial through the production TQ TCB path. Returns
    /// the shared-buffer output plus the number of encoded kernel dispatches.
    pub fn run_gemv(&self, ctx: &crate::metal::MetalContext) -> crate::Result<(Vec<f32>, usize)> {
        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
        crate::kernels::strand_bitslice_gemv_tcb(
            &mut tcb,
            &self.ready,
            &self.x_buf,
            0,
            &self.out_buf,
            0,
        )?;
        let dispatches = tcb.dispatch_count();
        tcb.commit_and_wait()?;
        let ptr = self.out_buf.contents() as *const f32;
        let output = unsafe { std::slice::from_raw_parts(ptr, self.rows) }.to_vec();
        Ok((output, dispatches))
    }

    /// Execute the production two-part residual recipe in one command buffer:
    /// the base projection overwrites `out`, then `residual` accumulates into
    /// the same GPU-resident output with `strand_bitslice_reduce_rows_accum`.
    ///
    /// This is intentionally a separate probe API rather than a boolean on
    /// [`Self::run_gemv`].  Callers must provide a second, independently parsed
    /// and uploaded artifact, making it impossible for a one-pass projection to
    /// be mislabeled as residual evidence.  Both harnesses must describe the
    /// same projection geometry and explicit runtime policy.
    pub fn run_gemv_two_pass(
        &self,
        ctx: &crate::metal::MetalContext,
        residual: &Self,
    ) -> crate::Result<(Vec<f32>, usize)> {
        if self.rows != residual.rows || self.cols != residual.cols {
            return Err(crate::Error::Model(format!(
                "TQ residual harness geometry {}x{} != base geometry {}x{}",
                residual.rows, residual.cols, self.rows, self.cols
            )));
        }
        if self.runtime_path != residual.runtime_path {
            return Err(crate::Error::Model(format!(
                "TQ residual runtime {:?} != base runtime {:?}",
                residual.runtime_path, self.runtime_path
            )));
        }
        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
        crate::kernels::strand_bitslice_gemv_tcb(
            &mut tcb,
            &self.ready,
            &self.x_buf,
            0,
            &self.out_buf,
            0,
        )?;
        crate::kernels::strand_bitslice_gemv_tcb_accum(
            &mut tcb,
            &residual.ready,
            &self.x_buf,
            0,
            &self.out_buf,
            0,
        )?;
        let dispatches = tcb.dispatch_count();
        tcb.commit_and_wait()?;
        let ptr = self.out_buf.contents() as *const f32;
        let output = unsafe { std::slice::from_raw_parts(ptr, self.rows) }.to_vec();
        Ok((output, dispatches))
    }
}

#[cfg(target_os = "macos")]
impl TqPreparedGpu {
    /// Decode-only device oracle for an explicit runtime policy. Unlike the
    /// fused path this admits arbitrary baked block boundaries supported by the
    /// scalar table, because it does not apply the row reducer.
    pub fn decode_q12_on_gpu_with_path(
        &self,
        ctx: &crate::metal::MetalContext,
        runtime_path: TqRuntimePath,
    ) -> crate::Result<Vec<i32>> {
        let (kernel, table_bytes, host_entry_bytes, gpu_entry_bytes) =
            if runtime_path == TqRuntimePath::CompactMetadata {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        self.compact_entries.as_ptr() as *const u8,
                        std::mem::size_of_val(self.compact_entries.as_slice()),
                    )
                };
                (
                    "strand_bitslice_decode_compact",
                    bytes,
                    std::mem::size_of::<CompactBitsliceEntry>(),
                    crate::kernels::strand_bitslice_compact_entry_sizeof(ctx)? as usize,
                )
            } else {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        self.entries.as_ptr() as *const u8,
                        std::mem::size_of_val(self.entries.as_slice()),
                    )
                };
                let kernel = match runtime_path {
                    TqRuntimePath::Stored => "strand_bitslice_decode",
                    TqRuntimePath::HashedQuantile => "strand_bitslice_decode_hashed",
                    TqRuntimePath::ComputedAcklam => "strand_bitslice_decode_computed",
                    TqRuntimePath::CompactMetadata => unreachable!(),
                };
                (
                    kernel,
                    bytes,
                    std::mem::size_of::<BitsliceEntry>(),
                    crate::kernels::strand_bitslice_entry_sizeof(ctx)? as usize,
                )
            };
        if host_entry_bytes != gpu_entry_bytes {
            return Err(crate::Error::Kernel(format!(
                "TQ {runtime_path:?} decode record stride mismatch: host={host_entry_bytes}, GPU={gpu_entry_bytes}"
            )));
        }

        let padded_len = self.payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..self.payload.len()].copy_from_slice(&self.payload);
        let w_buf = ctx.new_buffer_with_bytes(&padded);
        let out_buf = ctx.new_buffer(
            self.rows
                .saturating_mul(self.cols)
                .max(1)
                .saturating_mul(std::mem::size_of::<i32>()),
        );
        let tbl_buf = ctx.new_buffer_with_bytes(table_bytes);
        let (codebook_buf, shmem_bytes, tail_len) = match runtime_path {
            TqRuntimePath::Stored | TqRuntimePath::CompactMetadata => (
                ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&self.lut_q12)),
                ((1usize << self.l_bits) * std::mem::size_of::<i32>()) as u64,
                0u32,
            ),
            TqRuntimePath::HashedQuantile => {
                let quantiles: Vec<i16> = strand_quant::codebook::quantile_lut(self.l_bits)
                    .iter()
                    .map(|&q| i16::try_from(q).expect("TQ Q12 quantile exceeds i16"))
                    .collect();
                (
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i16, u8>(&quantiles)),
                    ((1usize << self.l_bits) * std::mem::size_of::<i16>()) as u64,
                    0u32,
                )
            }
            TqRuntimePath::ComputedAcklam => {
                let tail = strand_quant::codebook::tail_left_prefix_q12(self.l_bits);
                let buffer = if tail.is_empty() {
                    ctx.new_buffer(std::mem::size_of::<i32>())
                } else {
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&tail))
                };
                (buffer, 0, tail.len() as u32)
            }
        };

        let n_blocks = self.entries.len() as u32;
        const TG: u32 = 256;
        let n_tg = n_blocks.div_ceil(TG).max(1);
        ctx.dispatch_threads(kernel, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&tbl_buf), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &n_blocks as *const u32 as *const _,
            );
            enc.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &self.k_bits as *const u32 as *const _,
            );
            enc.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                &self.l_bits as *const u32 as *const _,
            );
            enc.set_buffer(6, Some(&codebook_buf), 0);
            if runtime_path == TqRuntimePath::ComputedAcklam {
                enc.set_bytes(
                    7,
                    std::mem::size_of::<u32>() as u64,
                    &tail_len as *const u32 as *const _,
                );
            }
            if shmem_bytes != 0 {
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            }
        })?;
        let ptr = out_buf.contents() as *const i32;
        Ok(
            unsafe { std::slice::from_raw_parts(ptr, self.rows.saturating_mul(self.cols)) }
                .to_vec(),
        )
    }

    /// Upload all weight buffers to GPU memory and pre-compute dispatch constants.
    /// Call once at model-load; the returned [`TqGpuReady`] is reused every step.
    pub fn upload_to_gpu(&self, ctx: &crate::metal::MetalContext) -> crate::Result<TqGpuReady> {
        let runtime_path = TqRuntimePath::from_env().map_err(crate::Error::Model)?;
        self.upload_to_gpu_with_path(ctx, runtime_path)
    }

    /// Explicit-policy twin used by autotuners and A/B gates. It avoids mutating
    /// process-global environment state and makes the selected interpretation a
    /// field of the returned [`TqGpuReady`].
    pub fn upload_to_gpu_with_path(
        &self,
        ctx: &crate::metal::MetalContext,
        runtime_path: TqRuntimePath,
    ) -> crate::Result<TqGpuReady> {
        // The fused GEMV reduction assumes row-major 256-weight blocks. Decode-only
        // supports ragged blocks, but silently feeding them to this row reduction
        // mixes rows or drops a tail. Refuse and let the model use its CPU fallback.
        let admission = self.gpu_admission();
        if !admission.eligible {
            return Err(crate::Error::Unimplemented(match admission.reason {
                Some(TqGpuIneligibility::ColumnsNotMultipleOf256) => {
                    "TQ GPU GEMV requires cols % 256 == 0"
                }
                Some(_) => {
                    "TQ GPU GEMV requires canonical row-major scalar 256-weight block geometry"
                }
                None => "TQ GPU GEMV admission failed without a reason",
            }));
        }
        let tbl_buf = if runtime_path == TqRuntimePath::CompactMetadata {
            let gpu_size = crate::kernels::strand_bitslice_compact_entry_sizeof(ctx)? as usize;
            let host_size = std::mem::size_of::<CompactBitsliceEntry>();
            if gpu_size != host_size {
                return Err(crate::Error::Kernel(format!(
                    "TQ compact table stride mismatch: GPU={gpu_size}, host={host_size}"
                )));
            }
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.compact_entries.as_ptr() as *const u8,
                    std::mem::size_of_val(self.compact_entries.as_slice()),
                )
            };
            ctx.new_buffer_with_bytes(bytes)
        } else {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.entries.as_ptr() as *const u8,
                    std::mem::size_of_val(self.entries.as_slice()),
                )
            };
            ctx.new_buffer_with_bytes(bytes)
        };
        let lut_buf = match runtime_path {
            TqRuntimePath::Stored | TqRuntimePath::CompactMetadata => {
                ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&self.lut_q12))
            }
            TqRuntimePath::HashedQuantile => {
                let quantiles: Vec<i16> = strand_quant::codebook::quantile_lut(self.l_bits)
                    .iter()
                    .map(|&q| i16::try_from(q).expect("TQ Q12 quantile exceeds i16"))
                    .collect();
                ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i16, u8>(&quantiles))
            }
            TqRuntimePath::ComputedAcklam => {
                let tail = strand_quant::codebook::tail_left_prefix_q12(self.l_bits);
                // Metal buffers cannot be empty. L=4 has no tail, but the kernel
                // receives tail_len=0 and never dereferences this placeholder.
                if tail.is_empty() {
                    ctx.new_buffer(std::mem::size_of::<i32>())
                } else {
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&tail))
                }
            }
        };
        // Pad payload to 4-byte word boundary + 8 zero bytes (WordReader safety).
        let padded_len = self.payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..self.payload.len()].copy_from_slice(&self.payload);
        let w_buf = ctx.new_buffer_with_bytes(&padded);

        let n_blocks = self.entries.len() as u32;
        let cols = self.cols as u32;
        let rows = self.rows as u32;
        let k_bits = self.k_bits;
        let l_bits = self.l_bits;
        let bpr = cols / 256;
        let shmem_bytes = match runtime_path {
            TqRuntimePath::Stored | TqRuntimePath::CompactMetadata => {
                ((1usize << l_bits) * std::mem::size_of::<i32>()) as u64
            }
            TqRuntimePath::HashedQuantile => {
                ((1usize << l_bits) * std::mem::size_of::<i16>()) as u64
            }
            TqRuntimePath::ComputedAcklam => 0,
        };
        const TG: u32 = 256;
        let n_tg_partials = n_blocks.div_ceil(TG).max(1);
        let n_tg_reduce = rows.div_ceil(TG).max(1);
        const MAX_VERIFY_BATCH: usize = 8;
        let partials_buf =
            ctx.new_buffer(n_blocks as usize * MAX_VERIFY_BATCH * std::mem::size_of::<f32>());

        // RHT-cols activation transform scratch. The GPU `strand_rht_forward_cols`
        // kernel is hardcoded to the 256-wide Hadamard block, which is bit-exact to
        // the CPU `rht_forward_cols_inplace` ONLY when `in_features % 256 == 0`
        // (the deploy invariant: `h == pow2_block_for(in_features,256) == 256`, no
        // sub-256 tail block). For `Cols` with an unaligned `in_features` (e.g. the
        // 0.5B's 896) the GPU transform would diverge, so we refuse to enable the
        // GPU RHT path and error — the caller must serve such a tensor on CPU or
        // re-bake `--no-rht`. None/Rows need no activation scratch here.
        let (rht_x_buf, rht_n_blocks) = if self.rht_mode == 2 {
            if cols % 256 != 0 {
                return Err(crate::Error::Unimplemented(
                    "TQ GPU RhtMode::Cols requires in_features % 256 == 0 (GPU FWHT is 256-wide); re-bake --no-rht or serve on CPU",
                ));
            }
            (
                Some(ctx.new_buffer(cols as usize * MAX_VERIFY_BATCH * std::mem::size_of::<f32>())),
                cols / 256,
            )
        } else if self.rht_mode == 1 {
            // Per-row RHT (the ~1 tok/s wall) is not served on GPU; no baked
            // artifact uses it and the bitslice path targets Cols. Refuse loudly.
            return Err(crate::Error::Unimplemented(
                "TQ GPU RhtMode::Rows is not served on the GPU bitslice path (eval-only on CPU)",
            ));
        } else {
            (None, 0)
        };

        // OUTL sparse-correction entries → GPU buffer (empty-but-valid when none).
        let outl_buf = if self.outliers.is_empty() {
            // A 1-entry placeholder so the buffer is never zero-length (Metal
            // dislikes 0-byte buffers); n_outl == 0 means the kernel is skipped.
            ctx.new_buffer(std::mem::size_of::<OutlierEntry>())
        } else {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.outliers.as_ptr() as *const u8,
                    std::mem::size_of_val(self.outliers.as_slice()),
                )
            };
            ctx.new_buffer_with_bytes(bytes)
        };
        let n_outl = self.outliers.len() as u32;

        Ok(TqGpuReady {
            tbl_buf,
            lut_buf,
            runtime_path,
            w_buf,
            partials_buf,
            n_blocks,
            cols,
            rows,
            k_bits,
            l_bits,
            bpr,
            shmem_bytes,
            n_tg_partials,
            n_tg_reduce,
            rht_mode: self.rht_mode,
            rht_seed: self.rht_seed,
            rht_x_buf,
            rht_n_blocks,
            outl_buf,
            n_outl,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    #[test]
    fn every_runtime_path_has_decode_and_fused_kernel_sources() {
        let shader = include_str!("../shaders/strand_bitslice.metal");
        for (path, decode, fused) in [
            (
                TqRuntimePath::Stored,
                "strand_bitslice_decode(",
                "strand_bitslice_gemv_partials(",
            ),
            (
                TqRuntimePath::CompactMetadata,
                "strand_bitslice_decode_compact(",
                "strand_bitslice_gemv_partials_compact(",
            ),
            (
                TqRuntimePath::HashedQuantile,
                "strand_bitslice_decode_hashed(",
                "strand_bitslice_gemv_partials_hashed(",
            ),
            (
                TqRuntimePath::ComputedAcklam,
                "strand_bitslice_decode_computed(",
                "strand_bitslice_gemv_partials_computed(",
            ),
        ] {
            assert!(
                shader.contains(decode),
                "missing Metal decode oracle {decode}"
            );
            assert!(shader.contains(fused), "missing Metal fused kernel {fused}");
            assert!(
                shader.contains(path.small_batch_kernel_name()),
                "missing Metal small-batch kernel {}",
                path.small_batch_kernel_name()
            );
        }
        for kernel in [
            "strand_rht_forward_cols_batched(",
            "strand_outlier_correct_batched(",
            "strand_bitslice_reduce_rows_small_batch(",
            "strand_bitslice_reduce_rows_small_batch_accum(",
        ] {
            assert!(
                shader.contains(kernel),
                "missing Metal batch support {kernel}"
            );
        }
    }

    /// CPU replay of the exact `strand_bitslice_decode` inner loop, driven from a
    /// baked [`BitsliceEntry`] table. Reproduces the kernel's bit-for-bit
    /// arithmetic on the host: per block, seed `state` from `init_state`, then for
    /// each of `n` weights pop a k-bit symbol from the payload at `bit_offset`,
    /// advance `state = ((state<<k)|sym)&mask`, gather `q = lut[state]`, and emit
    /// `w = (eff[j>>5]*q)>>16 + off[j>>5]` (the i64 `reconstruct_q` product) at
    /// `out_off + j`. `lut` must be `cfg.codebook()` (== `codebook_lut(l_bits)` for
    /// the default `StoredLut` mode the GPU binds).
    fn host_walk_decode(
        payload: &[u8],
        tbl: &[BitsliceEntry],
        lut: &[i32],
        total: usize,
        k_bits: u32,
        l_bits: u32,
    ) -> Vec<i32> {
        let state_mask = (1usize << l_bits) - 1;
        let input_mask = (1usize << k_bits) - 1;
        let k = k_bits as usize;
        let mut out = vec![0i32; total];
        for e in tbl {
            // d == 1 only; the scalar bake is what the production path uses.
            assert_eq!(e.d, 1, "host_walk_decode is the scalar (d==1) replay");
            let mut state = e.init_state as usize & state_mask;
            let mut bitpos = e.bit_offset as usize;
            let n = e.n as usize;
            let obase = e.out_off as usize;
            for j in 0..n {
                let sym = read_bits(payload, bitpos, k_bits) & input_mask;
                bitpos += k;
                state = ((state << k) | sym) & state_mask;
                let q = lut[state];
                let sb = j >> 5;
                let es = e.eff[sb];
                // i64 reconstruct: (scale_q * quantile_q) >> 16, matching
                // strand_quant::decode::reconstruct_q exactly.
                let w = ((((es as i64) * (q as i64)) >> 16) as i32) + e.off[sb];
                out[obase + j] = w;
            }
        }
        out
    }

    fn compact_code(words: [u32; 2], sb: usize) -> u8 {
        ((words[sb >> 2] >> ((sb & 3) * 8)) & 0x3f) as u8
    }

    fn host_walk_decode_compact(
        payload: &[u8],
        tbl: &[CompactBitsliceEntry],
        lut: &[i32],
        total: usize,
        k_bits: u32,
        l_bits: u32,
    ) -> Vec<i32> {
        let state_mask = (1usize << l_bits) - 1;
        let input_mask = (1usize << k_bits) - 1;
        let mut out = vec![0i32; total];
        for e in tbl {
            let mut state = e.init_state as usize & state_mask;
            let mut bitpos = e.bit_offset as usize;
            for j in 0..e.n as usize {
                let sym = read_bits(payload, bitpos, k_bits) & input_mask;
                bitpos += k_bits as usize;
                state = ((state << k_bits) | sym) & state_mask;
                let sb = j >> 5;
                let es = eff_scale_q(e.scale_q, compact_code(e.mult_codes, sb));
                let off = eff_min_q(e.min_base_q, compact_code(e.min_codes, sb));
                out[e.out_off as usize + j] =
                    ((((es as i64) * (lut[state] as i64)) >> 16) as i32) + off;
            }
        }
        out
    }

    fn synth_w(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
            .collect()
    }

    /// The structural invariants of the baked table: `eff[s]` is exactly the
    /// strand-quant fold of `scale_q` + the s-th sub-scale code, slots past
    /// `n_sub` are zero, and `n`/`out_off`/`bit_offset` match an independent
    /// prefix recompute. Run across the encode-lever matrix.
    #[test]
    fn bake_table_fields_match_recompute() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw(2.0),
            TrellisConfig::for_bpw(4.0),
            TrellisConfig::for_bpw_l(2.0, 12),
        ];
        for cfg in configs {
            for seed in 0..6u64 {
                let n = 1 + (seed as usize * 173) % 2050;
                let w = synth_w(n, seed);
                let enc = encode_tensor(&w, &cfg);
                let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");
                assert_eq!(tbl.len(), enc.blocks.len());

                let k = cfg.k_bits as usize;
                let mut prefix_n = 0usize;
                for (b, (blk, e)) in enc.blocks.iter().zip(tbl.iter()).enumerate() {
                    // out_off / bit_offset / n match a fresh prefix sum.
                    assert_eq!(e.n, blk.n, "cfg k{} L{} blk{b}: n", cfg.k_bits, cfg.l_bits);
                    assert_eq!(
                        e.out_off as usize, prefix_n,
                        "cfg k{} L{} blk{b}: out_off",
                        cfg.k_bits, cfg.l_bits
                    );
                    assert_eq!(
                        e.bit_offset as usize,
                        prefix_n * k,
                        "cfg k{} L{} blk{b}: bit_offset",
                        cfg.k_bits,
                        cfg.l_bits
                    );
                    assert_eq!(e.d, 1, "scalar bake d==1");

                    // eff[s] == eff_scale_q(scale_q, code[s]); tail slots zero.
                    let n_sub = n_sub_blocks(blk.n as usize);
                    let codes = unpack_sub_scales(&blk.sub_scales, n_sub);
                    for s in 0..n_sub {
                        assert_eq!(
                            e.eff[s],
                            eff_scale_q(blk.scale_q, codes[s]),
                            "cfg k{} L{} blk{b} sub{s}: eff",
                            cfg.k_bits,
                            cfg.l_bits
                        );
                    }
                    for s in n_sub..8 {
                        assert_eq!(
                            e.eff[s], 0,
                            "cfg k{} L{} blk{b} sub{s}: eff tail not zero",
                            cfg.k_bits, cfg.l_bits
                        );
                    }
                    // 3-bit (and the other non-affine deploy configs) => off all zero.
                    assert!(
                        !enc.has_affine_min,
                        "for_bpw configs must have affine-min off"
                    );
                    assert_eq!(
                        e.off, [0i32; 8],
                        "cfg k{} L{} blk{b}: off",
                        cfg.k_bits, cfg.l_bits
                    );

                    prefix_n += blk.n as usize;
                }
                assert_eq!(prefix_n, n, "prefix sum != total");
            }
        }
    }

    /// The load-bearing identity: a host replay of the bitstream straight from the
    /// baked table equals `decode_tensor_fixed` bit-for-bit. This is the exact
    /// arithmetic the Metal kernel runs, so a green test here predicts a
    /// bit-identical GPU dispatch. Swept over the encode-lever matrix
    /// (k∈{2,3,4}, L∈{7,12}, tail-biting × affine-min) and edge lengths.
    #[test]
    fn host_walk_matches_decode_tensor_fixed() {
        use strand_quant::decode::decode_tensor_fixed;

        let configs = [
            TrellisConfig::for_bpw(3.0),       // k3 L7 (3-bit deploy)
            TrellisConfig::for_bpw(2.0),       // k2 L6
            TrellisConfig::for_bpw(4.0),       // k4 L8
            TrellisConfig::for_bpw_l(2.0, 12), // k2 L12 (2-bit reopen)
            TrellisConfig::for_bpw_l(2.0, 5),  // k2 L5 (fold)
            TrellisConfig::for_bpw_l(4.0, 4),  // k4 L4 (fold)
        ];
        for cfg in configs {
            // codebook the CPU oracle uses; == codebook_lut(l_bits) for StoredLut,
            // which is exactly the buffer the GPU dispatch binds.
            let lut = cfg.codebook();
            for seed in 0..8u64 {
                // edge lengths: short final block, sub-block tails, 1-weight tensors.
                let n = 1 + (seed as usize * 211) % 2048;
                let w = synth_w(n, seed);
                let variants = [
                    ("plain", encode_tensor(&w, &cfg)),
                    (
                        "tail_biting",
                        encode_tensor_with(
                            &w,
                            &cfg,
                            &EncodeOpts {
                                tail_biting: true,
                                ..Default::default()
                            },
                        ),
                    ),
                    (
                        "affine_min",
                        encode_tensor_with(
                            &w,
                            &cfg,
                            &EncodeOpts {
                                affine_min: true,
                                ..Default::default()
                            },
                        ),
                    ),
                    (
                        "tail+affine",
                        encode_tensor_with(
                            &w,
                            &cfg,
                            &EncodeOpts {
                                tail_biting: true,
                                affine_min: true,
                                ..Default::default()
                            },
                        ),
                    ),
                ];
                for (label, enc) in &variants {
                    let tbl = bake_bitslice_entries(enc, &cfg).expect("bake");
                    let got =
                        host_walk_decode(&enc.bits, &tbl, &lut, enc.total, cfg.k_bits, cfg.l_bits);
                    let compact = bake_compact_bitslice_entries(enc, &cfg).expect("compact bake");
                    let got_compact = host_walk_decode_compact(
                        &enc.bits, &compact, &lut, enc.total, cfg.k_bits, cfg.l_bits,
                    );
                    let want = decode_tensor_fixed(enc, &cfg);
                    assert_eq!(
                        got, want,
                        "host-walk diverged: variant={label} k={} L={} n={n} seed={seed} \
                         tail={} affine={}",
                        cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min
                    );
                    assert_eq!(
                        got_compact, want,
                        "compact host-walk diverged: variant={label} k={} L={} n={n} seed={seed}",
                        cfg.k_bits, cfg.l_bits
                    );
                }
            }
        }
    }

    /// Affine-min coverage in isolation: encode a tensor WITH affine-min, assert
    /// the bake populates a non-zero `off[]` somewhere (so the off-fold is
    /// exercised, not just the all-zero 3-bit branch) and the host-walk still
    /// matches the oracle bit-for-bit.
    #[test]
    fn affine_min_off_fold_is_exercised_and_matches() {
        use strand_quant::decode::decode_tensor_fixed;
        // 4-bit config is where affine-min meaningfully engages.
        let cfg = TrellisConfig::for_bpw(4.0);
        let lut = cfg.codebook();
        let n = 1024usize;
        let w = synth_w(n, 99);
        let enc = encode_tensor_with(
            &w,
            &cfg,
            &EncodeOpts {
                affine_min: true,
                ..Default::default()
            },
        );
        assert!(enc.has_affine_min, "expected affine-min encode");
        let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");
        let any_off = tbl.iter().any(|e| e.off.iter().any(|&o| o != 0));
        assert!(
            any_off,
            "affine-min encode produced an all-zero off[] table — off-fold not exercised"
        );
        let got = host_walk_decode(&enc.bits, &tbl, &lut, enc.total, cfg.k_bits, cfg.l_bits);
        let want = decode_tensor_fixed(&enc, &cfg);
        assert_eq!(got, want, "affine-min host-walk diverged from oracle");
    }

    /// The vector-trellis (`d > 1`) bake produces a structurally correct table:
    /// `bit_offset` accumulates `n_steps*k` (NOT `n*k`), `out_off` accumulates the
    /// true output count `n`, and `d` is carried. (A full vec decode-identity check
    /// needs a `2^L*d` interleaved vector codebook, which the frozen Gaussian
    /// codebook does not provide — the model path is scalar-only and asserts
    /// `d == 1`, so the vec bake is carried for completeness, validated here at the
    /// table-geometry level.)
    #[test]
    fn vec_bake_geometry_is_step_based() {
        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        assert_eq!(cfg.vec_dim(), 2);
        let d = cfg.vec_dim();
        let k = cfg.k_bits as usize;
        let n = 1000usize;
        let w = synth_w(n, 7);
        let enc = encode_tensor(&w, &cfg);
        let tbl = bake_bitslice_entries_vec(&enc, &cfg).expect("vec bake");

        let mut exp_start_bit = 0usize;
        let mut exp_out_off = 0usize;
        for (b, (blk, e)) in enc.blocks.iter().zip(tbl.iter()).enumerate() {
            assert_eq!(e.d, d as u32, "blk{b}: d");
            assert_eq!(e.n, blk.n, "blk{b}: n");
            assert_eq!(
                e.out_off as usize, exp_out_off,
                "blk{b}: out_off (true count)"
            );
            assert_eq!(
                e.bit_offset as usize, exp_start_bit,
                "blk{b}: bit_offset (n_steps*k)"
            );
            let n_steps = (blk.n as usize).div_ceil(d);
            exp_start_bit += n_steps * k;
            exp_out_off += blk.n as usize;
        }
        assert_eq!(exp_out_off, n, "vec out_off prefix != total");
    }

    /// Guard the record layout: 84 bytes, 4-byte aligned, all fields 4-wide so the
    /// Rust `#[repr(C)]` and the MSL `sizeof(BitsliceEntry)` agree (the GPU probe
    /// asserts this at dispatch — here we pin the host side).
    #[test]
    fn bitslice_entry_layout() {
        assert_eq!(std::mem::size_of::<BitsliceEntry>(), 84);
        assert_eq!(std::mem::align_of::<BitsliceEntry>(), 4);
        assert_eq!(std::mem::size_of::<CompactBitsliceEntry>(), 40);
        assert_eq!(std::mem::align_of::<CompactBitsliceEntry>(), 4);
    }

    #[test]
    fn runtime_path_parser_is_strict_and_default_safe() {
        assert_eq!(
            TqRuntimePath::parse("stored").unwrap(),
            TqRuntimePath::Stored
        );
        assert_eq!(
            TqRuntimePath::parse("compact").unwrap(),
            TqRuntimePath::CompactMetadata
        );
        assert_eq!(
            TqRuntimePath::parse("hashed").unwrap(),
            TqRuntimePath::HashedQuantile
        );
        assert_eq!(
            TqRuntimePath::parse("computed").unwrap(),
            TqRuntimePath::ComputedAcklam
        );
        assert!(TqRuntimePath::parse("magic").is_err());
    }

    #[test]
    fn runtime_traffic_exposes_hidden_table_bpw() {
        let traffic = TqRuntimeTraffic {
            weights: 256,
            blocks: 1,
            payload_bytes: 96, // 3 bpw
            expanded_table_bytes: 84,
            compact_table_bytes: 40,
            stored_codebook_bytes: 0,
            hashed_quantile_bytes: 0,
            computed_tail_bytes: 0,
            threadgroups: 0,
            partial_roundtrip_bytes: 8,
        };
        assert_eq!(traffic.weight_path_bpw(TqRuntimePath::Stored), 5.625);
        assert_eq!(
            traffic.weight_path_bpw(TqRuntimePath::CompactMetadata),
            4.25
        );
    }

    #[test]
    fn runtime_recipe_axes_compose_without_claiming_executable_kernels() {
        let traffic = TqRuntimeTraffic {
            weights: 256,
            blocks: 1,
            payload_bytes: 96,
            expanded_table_bytes: 84,
            compact_table_bytes: 40,
            stored_codebook_bytes: 64,
            hashed_quantile_bytes: 32,
            computed_tail_bytes: 4,
            threadgroups: 1,
            partial_roundtrip_bytes: 8,
        };
        assert_eq!(TqRuntimeRecipe::RESEARCH_MATRIX.len(), 6);
        assert_eq!(TqRuntimePath::Stored.recipe(), TqRuntimeRecipe::STORED);
        assert_eq!(
            TqRuntimePath::CompactMetadata.recipe(),
            TqRuntimeRecipe::COMPACT
        );
        assert_eq!(
            traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPACT_HASHED),
            96 + 40 + 32 + 8
        );
        assert_eq!(
            traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPACT_COMPUTED),
            96 + 40 + 4 + 8
        );
        assert!(
            traffic.compressed_runtime_bpw_for(TqRuntimeRecipe::COMPACT_COMPUTED)
                < traffic.compressed_runtime_bpw_for(TqRuntimeRecipe::STORED)
        );
    }

    #[test]
    fn runtime_recipe_byte_frontier_is_monotone_for_every_frozen_l() {
        for l_bits in TrellisConfig::MIN_L..=TrellisConfig::MAX_L {
            let states = 1usize << l_bits;
            let tail = strand_quant::codebook::tail_left_prefix_q12(l_bits).len();
            let traffic = TqRuntimeTraffic {
                weights: 256 * 513,
                blocks: 513,
                payload_bytes: 96 * 513,
                expanded_table_bytes: 84 * 513,
                compact_table_bytes: 40 * 513,
                stored_codebook_bytes: states * std::mem::size_of::<i32>(),
                hashed_quantile_bytes: states * std::mem::size_of::<i16>(),
                computed_tail_bytes: tail * std::mem::size_of::<i32>(),
                threadgroups: 3,
                partial_roundtrip_bytes: 8 * 513,
            };
            let stored = traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::STORED);
            assert!(
                traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPACT) < stored,
                "compact metadata failed to save bytes at L={l_bits}"
            );
            assert!(
                traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::HASHED) < stored,
                "hashed quantiles failed to save bytes at L={l_bits}"
            );
            assert!(
                traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPUTED) < stored,
                "computed tails failed to save bytes at L={l_bits}"
            );
            assert!(
                traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPACT_HASHED)
                    < traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::HASHED)
            );
            assert!(
                traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPACT_COMPUTED)
                    < traffic.compressed_runtime_bytes_for(TqRuntimeRecipe::COMPUTED)
            );
        }
    }

    #[test]
    fn gpu_admission_explains_ragged_and_corrupt_geometry() {
        let entry = BitsliceEntry {
            bit_offset: 0,
            init_state: 0,
            out_off: 0,
            n: 256,
            eff: [0; 8],
            off: [0; 8],
            d: 1,
        };
        let accepted = assess_gpu_gemv_geometry(&[entry], 1, 1, 256, 3);
        assert!(accepted.eligible);
        assert_eq!(accepted.expected_blocks, 1);

        let ragged = assess_gpu_gemv_geometry(&[entry], 1, 1, 896, 3);
        assert!(!ragged.eligible);
        assert_eq!(
            ragged.reason,
            Some(TqGpuIneligibility::ColumnsNotMultipleOf256)
        );
        assert_eq!(ragged.expected_blocks, 4);

        let short = BitsliceEntry { n: 255, ..entry };
        let malformed = assess_gpu_gemv_geometry(&[short], 1, 1, 256, 3);
        assert_eq!(
            malformed.reason,
            Some(TqGpuIneligibility::BlockLengthNot256 { block: 0 })
        );
        let compact_missing = assess_gpu_gemv_geometry(&[entry], 0, 1, 256, 3);
        assert_eq!(
            compact_missing.reason,
            Some(TqGpuIneligibility::CompactBlockCountMismatch)
        );
    }

    #[test]
    fn gpu_bakes_treat_absent_sub_scale_stream_as_unity() {
        let cfg = TrellisConfig::new(7, 3, 256);
        let enc = EncodedTensor {
            bits: vec![0; 96],
            blocks: vec![strand_quant::encode::BlockMeta {
                scale_q: 1 << 16,
                sub_scales: Vec::new(),
                min_base_q: 0,
                mins: Vec::new(),
                init_state: 0,
                n: 256,
            }],
            total: 256,
            has_affine_min: false,
            tail_biting: false,
            has_rht_seed: false,
        };
        let expanded = bake_bitslice_entries(&enc, &cfg).expect("expanded bake");
        assert_eq!(expanded[0].eff, [1 << 16; 8]);
        let compact = bake_compact_bitslice_entries(&enc, &cfg).expect("compact bake");
        assert_eq!(compact[0].mult_codes, [0x3f3f_3f3f; 2]);
    }
}
