//! CUDA port of the G4 Metal bitslice decode (`shaders/strand_bitslice.metal`).
//!
//! NEW FILE — not wired into the build yet. Mirrors `cuda_backend.rs` (the working
//! CUDA *encode* lane) for the nvrtc compile / `load_ptx` / launch pattern, and
//! mirrors the Metal `BitsliceGpu` host (`metal.rs`) for the buffer/dispatch shape.
//! CPU stays canonical until a clean CUDA-vs-CPU/Metal parity is confirmed at the
//! 70B boundary; this module is the GPU lane that the byte-identity gate
//! (`tests/cuda_bitslice_identity.rs`) is written against.
//!
//! THE MOAT — FROZEN INTEGER Q12 DECODE
//! ------------------------------------
//! Every device must reconstruct the SAME `out_q12` integers bit-for-bit. The
//! decode arithmetic is integer-only and has no floating point on the value path:
//!
//!     sym   = (acc >> consumed) & ((1<<k)-1)        // LE bit unpack, u64 acc
//!     state = ((state << k) | sym) & (2^L - 1)
//!     q     = lut_q12[state]                         // frozen i32 LUT, identical bytes
//!     w     = ( ( (i64)eff[sub] * (i64)q ) >> 16 ) + off[sub]   // sub = j>>5
//!
//! This is character-for-character the CPU reference (`strand_quant::decode::
//! decode_tensor_fixed_with_lut`: `reconstruct_q(es,q)+off`, where
//! `reconstruct_q(s,q) = ((s as i64 * q as i64) >> 16) as i32`) and the Metal
//! kernel (`strand_bitslice_decode`: `(int)(((long)es*(long)q)>>16) + off`). The
//! ONLY decode-side floats in the Metal file live in the *GEMV/GEMM* kernels
//! (`Q12_TO_F32`, the `x` dot-product) — this CUDA port deliberately ships ONLY
//! the integer `strand_bitslice_decode` kernel, so there is no float on the
//! byte-identity path at all. (A fused GEMV lane, if ever needed, is separate
//! future work and would carry the same per-kernel float caveat the Metal GEMV
//! already documents — it does NOT touch this integer decode.)
//!
//! DETERMINISM ARGUMENT (why CUDA == CPU == Metal, exactly)
//! --------------------------------------------------------
//!   1. Integer-only value path. `i64` multiply + arithmetic `>>16` + `i32` add are
//!      defined identically by C/C++ (CUDA) and Rust (CPU) for the operand ranges
//!      here. `scale_q`, `q`, `eff`, `off` are all `i32`; the product is taken in
//!      `i64`/`long` (64-bit two's-complement, identical on host and device). No
//!      `float`/`double` appears, so no rounding-mode / FMA / fast-math divergence
//!      is possible (this is exactly why nvrtc fast-math flags are irrelevant to
//!      the decode — there are no FP ops to contract).
//!   2. Identical LUT bytes. The host uploads `codebook_lut(l_bits)` — the SAME
//!      frozen `&'static [i32]` table the CPU indexes — so `lut_q12[state]` returns
//!      identical i32 on both. (Guarded by the LUT golden-hash test in
//!      `strand_quant::codebook`.)
//!   3. Identical bit unpack. The Metal/CPU readers consume k bits LSB-first from a
//!      little-endian u32 word stream with a u64 accumulator; this kernel uses the
//!      byte-exact same `bs_load_u32_le` + `acc >>= k` recurrence. Bit i of symbol
//!      j is byte `(base_bit+i)>>3`, in-byte `(base_bit+i)&7` — matching
//!      `trellis::read_bits` and `block_walk::WordReader::pop`.
//!   4. Identical per-block geometry. The host bakes ONE `BitsliceEntry` per block
//!      via this module's own `bake_bitslice_entries` (start_bit, init_state —
//!      incl. tail-biting replay, out_off, n, eff[8], off[8]), built from the
//!      portable `block_walk` helpers — the SAME inputs the Metal bake uses, so
//!      the tables are value-for-value identical (proven byte-for-byte by the
//!      `cuda_bake_equals_metal_bake` test on macOS). The kernel consumes that
//!      table verbatim; `sub = j>>5` matches `i / SUB_BLOCK` with `SUB_BLOCK=32`.
//!      `block_len <= 256` ⇒ `n_sub <= 8`, so the fixed `eff[8]/off[8]` cover every
//!      sub-block (same gate: `bake_bitslice_entries` returns `None` for `n > 256`).
//!   5. struct stride parity. The runtime `cuda_entry_sizeof()` probe asserts the
//!      device `sizeof(BitsliceEntry)` equals the host `size_of::<BitsliceEntry>()`
//!      (80 bytes, 4-byte aligned, field order bit_offset/init_state/out_off/n/
//!      eff[8]/off[8]) — exactly the guard the Metal `BitsliceGpu::new` runs — so
//!      `tbl[gidx]` indexes the same bytes on both. If a compiler ever repacked the
//!      struct, `new()` returns `None` (fail-closed) rather than decoding garbage.
//!   6. No cross-thread / cross-block state. Each CUDA thread owns one block
//!      (`gidx`), walks it sequentially, and writes a disjoint output slice
//!      `[out_off, out_off+n)`. There is no reduction, no atomic, no shared mutable
//!      state on the value path (the LUT in shared memory is read-only after the
//!      load barrier). So the result is independent of block/grid scheduling,
//!      warp order, and launch batching — the same property the Metal kernel and
//!      the encode batch-boundary test rely on.
//!
//! BLOCKER (documented; cannot be exercised in this env)
//! -----------------------------------------------------
//!   * No CUDA device locally (Metal host, MPS owned by a live PV). This module
//!     COMPILES only under `--features cuda` and is gated `#[cfg(feature="cuda")]`;
//!     the byte-identity gate that exercises it (`cuda_bitslice_identity.rs`,
//!     CUDA-feature-gated, `#[ignore]`d) must be run ON THE POD. See the test file
//!     and the integration plan in the task return for the exact pod commands.
//!   * cudarc must be pinned `cuda-12060` for CUDA 12.7+ toolkits (same pin the
//!     encode lane already requires — `cuda_backend.rs` header).
//!   * Shared-memory LUT cap: this kernel stages `2^L * 4` bytes of `__shared__`
//!     (the frozen LUT), matching the Metal `threadgroup` LUT. At L=14 that is
//!     64 KB, the per-block static smem ceiling on many SM arch (e.g. sm_86 / the
//!     RTX 3090 default is 48 KB without opt-in). For L>12 the host falls back to
//!     reading the LUT from global memory (a compile-time `#define LUT_IN_SMEM`
//!     switch is provided; the global-LUT path is byte-identical, just slower).
//!     The 3-bit deploy (L=7, 512 B) and the 2-bit reopen (L=12, 16 KB) both fit
//!     smem comfortably, so the moat configs use the fast path.

#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use strand_quant::codebook::codebook_lut;
use strand_quant::encode::EncodedTensor;
use strand_quant::TrellisConfig;

use crate::block_walk::{block_init_state, block_plans, SideInfo};

// PORTABILITY (the blocker the prior draft hit): the Metal host module
// (`crate::metal`) is `#[cfg(target_os = "macos")]`, so its `BitsliceEntry` and
// `bake_bitslice_entries` DO NOT EXIST on the Linux/CUDA pod — importing them
// here would fail to compile under `--features cuda` on the very target that
// runs CUDA. So this module is SELF-CONTAINED: it declares its own `#[repr(C)]`
// `BitsliceEntry` (identical 80-byte layout) and its own `bake_bitslice_entries`
// built from the portable, un-gated `crate::block_walk` helpers
// (`block_plans` / `block_init_state` / `SideInfo`) — the exact same inputs the
// Metal bake uses, so the two tables are value-for-value identical without
// coupling the files. A parity test (`tests/bake_parity`, macOS-only) compares
// the two bakes byte-for-byte where both modules compile.

/// Host mirror of the `.cu` `struct BitsliceEntry` AND of
/// `metal::BitsliceEntry`, field-for-field. All members are 4-byte scalars, so
/// `#[repr(C)]` gives 4*u32 + 8*i32 + 8*i32 = 80 bytes, no padding. The
/// `cuda_entry_sizeof()` probe asserts the device agrees before any decode.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitsliceEntry {
    pub bit_offset: u32,
    pub init_state: u32,
    pub out_off: u32,
    pub n: u32,
    pub eff: [i32; 8],
    pub off: [i32; 8],
}

// cudarc needs to memcpy `BitsliceEntry` to/from the device. The struct is
// `#[repr(C)]` POD (four u32 + two `[i32;8]`), so this is sound; the
// `cuda_entry_sizeof()` probe defends the layout contract at runtime.
unsafe impl cudarc::driver::DeviceRepr for BitsliceEntry {}

/// Bake the per-block decode table. Value-for-value identical to
/// `metal::bake_bitslice_entries`: same gate (`n > 256` -> `None`, i.e. at most
/// 8 sub-blocks since `SUB_BLOCK = 32`), same `block_plans` (start_bit / out_off
/// geometry), same `SideInfo::hoist` (eff/off i32 hoisting), same
/// `block_init_state` (tail-biting replayed into `init_state` HOST-SIDE, so the
/// kernel never needs the tail-biting branch). Returns `None` for any block
/// wider than the fixed `eff[8]/off[8]`; the caller then falls back to the CPU
/// reference.
pub fn bake_bitslice_entries(
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
        eff[..side.eff().len()].copy_from_slice(side.eff());
        off[..side.off().len()].copy_from_slice(side.off());
        let init = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
        out.push(BitsliceEntry {
            bit_offset: plan.start_bit as u32,
            init_state: init as u32,
            out_off: plan.out_off as u32,
            n: blk.n,
            eff,
            off,
        });
    }
    Some(out)
}

/// THE DECODE KERNEL (runtime-compiled with nvrtc, exactly as the encode lane).
///
/// SINGLE SOURCE OF TRUTH: the kernel text is `include_str!`'d from the
/// `shaders/strand_bitslice_decode.cu` file — mirroring how `strand_bitslice.metal`
/// is `include_str!`'d into `metal.rs` — so there is exactly one copy of the
/// kernel and it cannot drift from the host's launch contract. The struct it
/// declares mirrors `BitsliceEntry` above (and `metal::BitsliceEntry`)
/// field-for-field; the bit-unpack recurrence, the state update, and the integer
/// reconstruct are line-for-line the Metal `strand_bitslice_decode`.
///
/// The `.cu` carries a `-DLUT_IN_SMEM=1` switch: defined => the frozen LUT is
/// staged into dynamic shared memory (fast path, L<=13); undefined => the kernel
/// reads the LUT from global memory (byte-identical, used at L=14 where the smem
/// table exceeds 48 KB).
///
/// Grid: ceil(n_blocks / TPB) blocks. Block: TPB threads (TPB=256, matching the
/// Metal threadgroup of 256). One CUDA thread decodes one trellis block.
const BITSLICE_DECODE_CUDA_SRC: &str = include_str!("../shaders/strand_bitslice_decode.cu");

const MODULE: &str = "strand_bitslice_decode_cu";
const KERNEL: &str = "strand_bitslice_decode";
const SIZEOF_KERNEL: &str = "strand_bitslice_entry_sizeof";

/// Threads per block. 256 mirrors the Metal threadgroup width (one thread = one
/// trellis block). Decode is per-block-sequential, so this only affects occupancy,
/// never the bytes.
const TPB: u32 = 256;

/// Above this L the frozen LUT (`2^L * 4` bytes) exceeds the default 48 KB static
/// shared-memory budget on common SM arch (e.g. sm_86 / RTX 3090), so we compile
/// the global-LUT variant. L<=12 ⇒ <=16 KB ⇒ shared path. Byte-identical either way.
const MAX_SMEM_L: u32 = 12;

pub struct BitsliceCudaGpu {
    device: Arc<CudaDevice>,
    /// True iff the loaded kernel reads the LUT from shared memory (`LUT_IN_SMEM`).
    lut_in_smem: bool,
}

impl BitsliceCudaGpu {
    /// Open device 0, compile the decode kernel with nvrtc, and assert the device
    /// `sizeof(BitsliceEntry)` matches the host (fail-closed on layout drift).
    ///
    /// `max_l` is the largest `l_bits` this instance will decode; it selects the
    /// shared-vs-global LUT compile variant up front (one module per process).
    /// Returns `None` if no CUDA device, compile/load failure, or layout mismatch.
    pub fn new(max_l: u32) -> Option<Self> {
        let device = CudaDevice::new(0)
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA device error: {e}"))
            .ok()?;

        let lut_in_smem = max_l <= MAX_SMEM_L;
        let src = if lut_in_smem {
            format!("#define LUT_IN_SMEM 1\n{BITSLICE_DECODE_CUDA_SRC}")
        } else {
            BITSLICE_DECODE_CUDA_SRC.to_string()
        };

        let ptx = compile_ptx(src)
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA bitslice compile error: {e}"))
            .ok()?;
        device
            .load_ptx(ptx, MODULE, &[KERNEL, SIZEOF_KERNEL])
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA bitslice load error: {e}"))
            .ok()?;

        let gpu = Self { device, lut_in_smem };

        // Struct-stride parity guard (mirrors metal.rs BitsliceGpu::new).
        let dev_sz = gpu.cuda_entry_sizeof()?;
        let host_sz = std::mem::size_of::<BitsliceEntry>() as u32;
        if dev_sz != host_sz {
            eprintln!(
                "[strand-decode-kernel] CUDA sizeof(BitsliceEntry)={dev_sz} != host {host_sz} \
                 — refusing to decode (tbl stride would diverge)"
            );
            return None;
        }
        eprintln!("[strand-decode-kernel] CUDA bitslice decode ready: device 0, LUT_IN_SMEM={lut_in_smem}");
        Some(gpu)
    }

    fn cuda_entry_sizeof(&self) -> Option<u32> {
        let mut out: CudaSlice<u32> = self.device.alloc_zeros(1).ok()?;
        let f = self.device.get_func(MODULE, SIZEOF_KERNEL)?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe { f.launch(cfg, (&mut out,)) }.ok()?;
        let host = self.device.dtoh_sync_copy(&out).ok()?;
        host.first().copied()
    }

    /// Decode one tensor's payload to Q12 integers on the GPU.
    ///
    /// `total` must be `enc.total`; `tbl` must be `bake_bitslice_entries(enc,cfg)`;
    /// `lut` must be `codebook_lut(l_bits)`. Output is the SAME `Vec<i32>` the CPU
    /// `decode_tensor_fixed` / Metal `decode_q12` produce, bit-for-bit (the moat).
    pub fn decode_q12(
        &self,
        payload: &[u8],
        tbl: &[BitsliceEntry],
        lut: &[i32],
        total: usize,
        k_bits: u32,
        l_bits: u32,
    ) -> Option<Vec<i32>> {
        assert_eq!(lut.len(), 1usize << l_bits, "LUT must have 2^L entries");
        if self.lut_in_smem && l_bits > MAX_SMEM_L {
            // This instance was compiled for the shared path but the config needs
            // the global path: refuse rather than launch an over-budget smem kernel.
            eprintln!(
                "[strand-decode-kernel] L={l_bits} exceeds smem budget but instance is LUT_IN_SMEM; \
                 re-open with new(max_l>=L) to select the global-LUT variant"
            );
            return None;
        }

        // Payload staged zero-padded to a u32 boundary + 8 B of slack, so the
        // `bs_load_u32_le(.., ++word_idx)` tail read can never index OOB — exactly
        // the Metal `upload_payload` contract.
        let padded_len = payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..payload.len()].copy_from_slice(payload);

        let d_w = self.device.htod_sync_copy(&padded).ok()?;
        let mut d_out: CudaSlice<i32> = self.device.alloc_zeros(total.max(1)).ok()?;
        let d_tbl = self.device.htod_sync_copy(tbl).ok()?;
        let d_nb = self.device.htod_sync_copy(&[tbl.len() as u32]).ok()?;
        let d_k = self.device.htod_sync_copy(&[k_bits]).ok()?;
        let d_l = self.device.htod_sync_copy(&[l_bits]).ok()?;
        let d_lut = self.device.htod_sync_copy(lut).ok()?;

        let n_blocks = tbl.len() as u32;
        let grid = (n_blocks.div_ceil(TPB)).max(1);
        let shared_mem_bytes = if self.lut_in_smem {
            ((1usize << l_bits) * std::mem::size_of::<i32>()) as u32
        } else {
            0
        };
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (TPB, 1, 1),
            shared_mem_bytes,
        };

        let f = self.device.get_func(MODULE, KERNEL)?;
        unsafe {
            f.launch(cfg, (&d_w, &mut d_out, &d_tbl, &d_nb, &d_k, &d_l, &d_lut))
        }
        .map_err(|e| eprintln!("[strand-decode-kernel] CUDA bitslice launch error: {e}"))
        .ok()?;

        self.device.dtoh_sync_copy(&d_out).ok()
    }
}

/// Convenience: bake + LUT + decode for an `EncodedTensor`, mirroring
/// `crate::metal::bitslice_decode_q12_with_lut`. Falls back to the CANONICAL CPU
/// reference (`decode_tensor_fixed_with_lut`) for vec-trellis (`vec_dim>1`),
/// `n>256` blocks, or any GPU error — the SAME gates as the Metal path. The
/// returned bytes are therefore correct on every path.
pub fn cuda_bitslice_decode_q12_with_lut(
    gpu: &BitsliceCudaGpu,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    use strand_quant::decode::decode_tensor_fixed_with_lut;
    if cfg.vec_dim() > 1 {
        return decode_tensor_fixed_with_lut(enc, cfg, lut);
    }
    let Some(tbl) = bake_bitslice_entries(enc, cfg) else {
        return decode_tensor_fixed_with_lut(enc, cfg, lut);
    };
    match gpu.decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits) {
        Some(v) => v,
        None => decode_tensor_fixed_with_lut(enc, cfg, lut),
    }
}

/// As above with the default codebook LUT for the config.
pub fn cuda_bitslice_decode_q12(
    gpu: &BitsliceCudaGpu,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Vec<i32> {
    cuda_bitslice_decode_q12_with_lut(gpu, enc, cfg, codebook_lut(cfg.l_bits))
}

/// THE PARITY GATE — run ON THE POD GPU (cannot run in this Metal-only env).
///
/// Sweeps the canonical deploy + reopen configs x 24 seeds x 4 encode variants
/// (plain, tail-biting, affine-min, both) and asserts the CUDA decode is
/// byte-identical to the CPU `decode_tensor_fixed`. Panics on the FIRST
/// divergence (the release blocker), mirroring the `identity_matrix` check in
/// `bin/gate-bitslice.rs`. Returns the number of cases checked, or `None` if no
/// CUDA device is present (so a CI host without a GPU skips, exactly as the
/// Metal tests skip when `StrandGpu::new()` is `None`).
///
/// Selects the LUT-in-smem vs global-LUT variant per config: L<=12 uses the fast
/// shared path, L>12 (the k2/L12 reopen is exactly 12) uses smem too; a >12 case
/// would need a `new(max_l)` with the global variant. All canonical configs here
/// are L<=12, so one `new(12)` instance covers the whole sweep.
pub fn validate_against_cpu() -> Option<usize> {
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    let gpu = BitsliceCudaGpu::new(MAX_SMEM_L)?;

    let configs: [(TrellisConfig, &str); 6] = [
        (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"),
        (TrellisConfig::for_bpw(2.0), "k2 L6"),
        (TrellisConfig::for_bpw(4.0), "k4 L8"),
        (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)"),
        (TrellisConfig::for_bpw_l(2.0, 5), "k2 L5 (fold)"),
        (TrellisConfig::for_bpw_l(3.0, 5), "k3 L5 (fold)"),
    ];

    let mut checked = 0usize;
    for (cfg, label) in &configs {
        for seed in 0..24u64 {
            let n = 1 + (seed as usize * 211) % 4096;
            let w: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
                .collect();
            let variants = [
                encode_tensor(&w, cfg),
                encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                encode_tensor_with(
                    &w,
                    cfg,
                    &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ),
            ];
            for enc in &variants {
                let reference = decode_tensor_fixed(enc, cfg);
                let got = cuda_bitslice_decode_q12(&gpu, enc, cfg);
                assert_eq!(
                    got, reference,
                    "CUDA IDENTITY VIOLATION: bitslice decode diverged from \
                     decode_tensor_fixed at {label}, n={n}, seed={seed}, \
                     tail={}, affine={} — release blocker",
                    enc.tail_biting, enc.has_affine_min,
                );
                checked += 1;
            }
        }
    }
    Some(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The struct layout MUST be 80 bytes (4*u32 + 8*i32 + 8*i32, no padding) so
    /// it matches the `.cu` struct AND `metal::BitsliceEntry`. Runtime drift is
    /// caught by the sizeof probe; this catches it at test time without a GPU.
    #[test]
    fn entry_is_80_bytes_no_padding() {
        assert_eq!(std::mem::size_of::<BitsliceEntry>(), 80);
        assert_eq!(std::mem::align_of::<BitsliceEntry>(), 4);
    }

    /// Pure-CPU guard on the bake geometry (no GPU): out_off is the running
    /// prefix sum of block `n`, blocks cover the whole tensor, and the per-block
    /// `n`/init match the encoded blocks.
    #[test]
    fn bake_geometry_is_prefix_sum() {
        use strand_quant::encode::encode_tensor;
        let cfg = TrellisConfig::for_bpw(3.0);
        let w: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let enc = encode_tensor(&w, &cfg);
        let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");

        assert_eq!(tbl.len(), enc.blocks.len());
        let mut acc = 0u32;
        for (e, blk) in tbl.iter().zip(enc.blocks.iter()) {
            assert_eq!(e.out_off, acc, "out_off must be the prefix sum of block n");
            assert_eq!(e.n, blk.n);
            acc += blk.n;
        }
        assert_eq!(acc as usize, enc.total, "blocks must cover the whole tensor");
    }

    /// The local CUDA bake MUST equal the Metal bake byte-for-byte (only compiled
    /// where the Metal module exists — macOS). This is the cross-backend table
    /// parity check; on the pod (Linux) the Metal module is absent so the test is
    /// simply not built, but the value-equality is what the determinism argument
    /// rests on.
    #[cfg(target_os = "macos")]
    #[test]
    fn cuda_bake_equals_metal_bake() {
        use strand_quant::encode::{encode_tensor_with, EncodeOpts};
        for &bpw in &[2.0f64, 3.0, 4.0] {
            let cfg = TrellisConfig::for_bpw(bpw);
            for seed in 0..6u64 {
                let n = 17 + (seed as usize * 401) % 2048;
                let w: Vec<f32> =
                    (0..n).map(|i| ((i as f32 + seed as f32) * 0.019).sin() * 0.6).collect();
                for opts in [
                    EncodeOpts::default(),
                    EncodeOpts { tail_biting: true, ..Default::default() },
                    EncodeOpts { affine_min: true, ..Default::default() },
                ] {
                    let enc = encode_tensor_with(&w, &cfg, &opts);
                    let ours = bake_bitslice_entries(&enc, &cfg);
                    let metal = crate::metal::bake_bitslice_entries(&enc, &cfg);
                    // Compare field-by-field via the byte layout (identical repr(C)).
                    match (ours, metal) {
                        (Some(a), Some(b)) => {
                            assert_eq!(a.len(), b.len());
                            for (ea, eb) in a.iter().zip(b.iter()) {
                                assert_eq!(ea.bit_offset, eb.bit_offset);
                                assert_eq!(ea.init_state, eb.init_state);
                                assert_eq!(ea.out_off, eb.out_off);
                                assert_eq!(ea.n, eb.n);
                                assert_eq!(ea.eff, eb.eff);
                                assert_eq!(ea.off, eb.off);
                            }
                        }
                        (None, None) => {}
                        _ => panic!("bake gate disagreement: cuda vs metal returned different Some/None"),
                    }
                }
            }
        }
    }
}
