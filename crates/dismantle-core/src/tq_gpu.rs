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
use strand_quant::encode::{n_sub_blocks, unpack_sub_scales, EncodedTensor};
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
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
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
    /// Informational effective bits-per-weight for this tensor.
    pub bpw: f32,
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

        Ok(TqPreparedGpu {
            payload,
            entries,
            lut_q12,
            k_bits: cfg.k_bits,
            l_bits: cfg.l_bits,
            rows: st.out_features,
            cols: st.in_features,
            rht_mode,
            rht_seed: st.rht_seed,
            bpw,
        })
    }
}

/// GPU-resident pre-uploaded buffers and dispatch constants for one STRAND
/// projection. Built once at model-load via [`TqPreparedGpu::upload_to_gpu`];
/// every decode step binds the already-live Metal buffers with no allocation.
#[cfg(target_os = "macos")]
pub struct TqGpuReady {
    /// Baked per-block BitsliceEntry seek table, uploaded to GPU.
    pub tbl_buf: crate::metal::PinnedBuffer,
    /// Q12 Gaussian codebook LUT (2^l_bits entries), uploaded to GPU.
    pub lut_buf: crate::metal::PinnedBuffer,
    /// Padded payload bitstream, uploaded to GPU.
    pub w_buf: crate::metal::PinnedBuffer,
    /// Scratch partial-dot-products (n_blocks × f32), pre-allocated.
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
}

#[cfg(target_os = "macos")]
impl TqPreparedGpu {
    /// Upload all weight buffers to GPU memory and pre-compute dispatch constants.
    /// Call once at model-load; the returned [`TqGpuReady`] is reused every step.
    pub fn upload_to_gpu(
        &self,
        ctx: &crate::metal::MetalContext,
    ) -> crate::Result<TqGpuReady> {
        let tbl_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.entries.as_ptr() as *const u8,
                std::mem::size_of_val(self.entries.as_slice()),
            )
        };
        let tbl_buf = ctx.new_buffer_with_bytes(tbl_bytes);
        let lut_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&self.lut_q12));
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
        let shmem_bytes = ((1usize << l_bits) * std::mem::size_of::<i32>()) as u64;
        const TG: u32 = 256;
        let n_tg_partials = n_blocks.div_ceil(TG).max(1);
        let n_tg_reduce = rows.div_ceil(TG).max(1);
        let partials_buf = ctx.new_buffer(n_blocks as usize * std::mem::size_of::<f32>());

        Ok(TqGpuReady {
            tbl_buf,
            lut_buf,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

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
                    let want = decode_tensor_fixed(enc, &cfg);
                    assert_eq!(
                        got, want,
                        "host-walk diverged: variant={label} k={} L={} n={n} seed={seed} \
                         tail={} affine={}",
                        cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min
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
    }
}
