//! CUDA host DISPATCH for the bitslice decode (cudarc build/launch + memory-bounded
//! block-batched staging).
//!
//! SCOPE OF THIS FILE (the "host dispatch" PART)
//! ---------------------------------------------
//! NEW, self-contained, OS-agnostic module. It owns exactly the host side of the
//! CUDA decode lane:
//!   * the nvrtc build + `load_ptx` + `get_func` + `LaunchConfig` + `launch` pattern,
//!     mirroring the working CUDA ENCODE backend `strand-quant/src/cuda_backend.rs`;
//!   * grid/block sizing (one CUDA thread = one trellis block, 256 threads/block to
//!     mirror the Metal threadgroup width);
//!   * device buffer staging (payload zero-padding, LUT + table upload, output alloc);
//!   * and — the headline of this PART — MEMORY-BOUNDED BLOCK BATCHING that applies the
//!     encode-lane OOM lesson (`strand-quant/src/cuda_dispatch.rs`) to decode.
//!
//! It compiles the canonical kernel `shaders/strand_bitslice_decode.cu` (the parallel
//! "kernel" PART) via `include_str!`. It deliberately does NOT depend on
//! `crate::metal` (which is `#[cfg(target_os="macos")]` and absent on the Linux CUDA
//! pod): it carries its own `#[repr(C)]` `BitsliceEntry` (layout-identical, runtime-
//! asserted) and bakes the table through the OS-agnostic `crate::block_walk` helpers —
//! the SAME helpers `metal::bake_bitslice_entries` uses, so the baked values are
//! bit-for-bit identical to the Metal path.
//!
//! THE MOAT — BYTE-IDENTICAL DECODE (determinism argument)
//! -------------------------------------------------------
//! The value path is integer-only: `w = (i32)(((i64)eff[sb]*(i64)q) >> 16) + off[sb]`
//! with `q = lut[state]`, identical to the CPU `reconstruct_q(es,q)+off`
//! (`SCALE_SHIFT=16`) and the Metal `strand_bitslice_decode`. No float touches decode,
//! so there is no rounding-mode/FMA/fast-math freedom for nvrtc to exploit. This module
//! only STAGES and DISPATCHES; it does not alter the arithmetic. The two host-side
//! degrees of freedom it introduces — (a) how many blocks are staged per launch, and
//! (b) grid/block geometry — provably cannot change any output value:
//!   * Per-block independence: each block's `n` outputs depend only on its own
//!     `BitsliceEntry` (bit_offset, init_state, out_off, n, eff, off) + the global LUT;
//!     the kernel re-seeds `state` from `e.init_state` at the top of every block and
//!     has no cross-block/cross-thread mutable state after the read-only LUT load. So
//!     warp order, grid shape, and the batch partition are all value-invariant — the
//!     identical structural reason the encode batch-boundary test
//!     (`strand-quant/tests/cuda_batch_boundary_determinism.rs`) holds.
//!   * `out_off` is ABSOLUTE (index into the full tensor output), so a batch writes its
//!     blocks' disjoint absolute slices into one shared output buffer with no
//!     renumbering; concatenation of batches == the whole-tensor decode, exactly.
//!   * Struct-stride parity is asserted at `new()` via the `entry_sizeof` probe (80 B),
//!     mirroring the Metal `gpu_entry_sizeof` guard — fail-closed on layout drift.
//!
//! BLOCK-BATCHING: THE ENCODE-OOM LESSON, APPLIED TO DECODE
//! --------------------------------------------------------
//! The encode lane OOM'd at frontier scale because it staged the WHOLE tensor's GPU
//! back-buffer + per-block `levels_f32` at once; `cuda_dispatch.rs` fixed it by bounding
//! each launch to `cuda_batch_size(..)` blocks (and proving batch-boundary invariance).
//! Decode is far lighter — it has NO back-buffer (one i32 per weight, ~`num_states`x
//! smaller than encode's `block_len*num_states` back-pointers) — but the SAME footgun
//! exists: `decode_q12` would otherwise `alloc` a single `total*4`-byte device output
//! AND `dtoh` a `total*4`-byte host copy, and `archive-to-safetensors` decodes many
//! tensors (potentially concurrently). A 70B model's largest tensors are hundreds of
//! millions of weights; the host-staging blow-up is exactly the >GB-times-N-threads
//! pressure that killed the encode lane. So this module:
//!   1. exposes `cuda_decode_batch_size(max_block_len, n_blocks)` — a PURE, GPU-free,
//!      unit-testable function (mirrors `cuda_dispatch::cuda_batch_size`), with a floor
//!      of 1 (never 0 for a non-empty tensor; never an unbounded `.max(64)` footgun);
//!   2. stages at most `batch` blocks per launch, re-using the once-uploaded payload +
//!      LUT + table and slicing the table per launch (`try_slice`) — so transient
//!      DEVICE staging beyond the (unavoidable) output buffer is just the table slice;
//!   3. for the EXTREME case where even the full `total*4` output exceeds device memory,
//!      provides `decode_q12_chunked` which also bounds the OUTPUT buffer + the host
//!      copy per launch and returns the concatenation (block-disjoint => identical).

#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_lean_with_lut;
use strand_quant::encode::EncodedTensor;
use strand_quant::TrellisConfig;

use crate::block_walk::{block_init_state, block_plans, SideInfo};

/// The canonical decode kernel (the "kernel" PART). `include_str!` keeps the host and
/// device in one compilation unit, same discipline as `metal.rs`'s
/// `include_str!("../shaders/strand_bitslice.metal")`.
const BITSLICE_DECODE_CU: &str = include_str!("../shaders/strand_bitslice_decode.cu");

const MODULE: &str = "strand_bitslice_decode_dispatch";
const KERNEL_DECODE: &str = "strand_bitslice_decode";
const KERNEL_SIZEOF: &str = "strand_bitslice_entry_sizeof";

/// Threads per CUDA block == the Metal threadgroup width. One thread decodes one
/// trellis block; the LUT is staged cooperatively by all 256 threads. Geometry only —
/// never affects the decoded bytes.
const TPB: u32 = 256;

/// Per-launch OUTPUT staging budget (device `d_out` AND the host `dtoh` copy each cost
/// this in `decode_q12_chunked`). 512 MB mirrors the encode lane's `MAX_BACK_BYTES_CUDA`
/// and sits far under a 24 GB 3090 even with several concurrent worker threads.
const MAX_OUT_BYTES_CUDA: usize = 512 * 1024 * 1024;

/// Host mirror of the `.cu` `BitsliceEntry` (and the Metal one): `#[repr(C)]`, all
/// fields 4-byte scalars => 4*4 + 8*4 + 8*4 = 80 bytes, no padding. Field order is
/// load-bearing and asserted equal to the device `sizeof` at `new()`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BitsliceEntry {
    pub bit_offset: u32,
    pub init_state: u32,
    pub out_off: u32,
    pub n: u32,
    pub eff: [i32; 8],
    pub off: [i32; 8],
}

// Sound: `#[repr(C)]`, plain-old-data (u32/i32 + i32 arrays), no Drop, no padding holes
// (largest scalar 4 B, 80 B total) — exactly the `cuda_backend.rs::BlockParams`
// contract. The runtime `entry_sizeof` probe defends the layout.
unsafe impl DeviceRepr for BitsliceEntry {}

/// Bounded number of *blocks* per decode launch.
///
/// Pure function (no GPU) — unit-tested below, mirroring `cuda_dispatch::cuda_batch_size`.
/// Each block contributes at most `max_block_len * 4` output bytes; we choose the
/// largest batch whose output slice stays within `MAX_OUT_BYTES_CUDA`, clamped to
/// `[1, n_blocks]`. `max_block_len` is the largest STRAND block length in the tensor
/// (`<= 256` on the bitslice path); using the max is conservative for ragged tails.
///
/// NOTE: this bound matters for `decode_q12_chunked` (which also bounds the output
/// buffer). For the common `decode_q12` (single output buffer, batched table launches)
/// the same batch limits per-launch DEVICE pressure beyond the output buffer.
pub fn cuda_decode_batch_size(max_block_len: usize, n_blocks: usize) -> usize {
    if n_blocks == 0 {
        return 0;
    }
    let out_bytes_per_block = max_block_len.saturating_mul(std::mem::size_of::<i32>()).max(1);
    (MAX_OUT_BYTES_CUDA / out_bytes_per_block).max(1).min(n_blocks)
}

/// CUDA bitslice decoder. Holds the device + loaded module; cheap to keep alive across
/// tensors (like `BitsliceGpu` / `CudaViterbi`).
pub struct BitsliceCudaDispatch {
    device: Arc<CudaDevice>,
}

impl BitsliceCudaDispatch {
    /// Open device 0, compile + load the decode PTX, and assert device struct stride
    /// == host struct stride (80 B). Returns `None` (caller falls back to CPU) on any
    /// failure — same fail-closed contract as `BitsliceGpu::new` / `CudaViterbi::new`.
    ///
    /// The kernel stages the LUT in dynamic shared memory; at L<=12 that is <=16 KB,
    /// well under the 48 KB default per-block smem. (L>12 would need an opt-in smem
    /// raise or a global-LUT variant; the moat configs — L=7 3-bit, L=12 2-bit — fit.)
    pub fn new() -> Option<Self> {
        let device = CudaDevice::new(0)
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA device error: {e}"))
            .ok()?;

        let ptx = compile_ptx(BITSLICE_DECODE_CU)
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA decode compile error: {e}"))
            .ok()?;
        device
            .load_ptx(ptx, MODULE, &[KERNEL_DECODE, KERNEL_SIZEOF])
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA decode load error: {e}"))
            .ok()?;

        let gpu = Self { device };

        let dev_sz = gpu.gpu_entry_sizeof()?;
        let host_sz = std::mem::size_of::<BitsliceEntry>() as u32;
        if dev_sz != host_sz {
            eprintln!(
                "[strand-decode-kernel] CUDA sizeof(BitsliceEntry)={dev_sz} != host {host_sz} \
                 — refusing CUDA decode (tbl stride would diverge)"
            );
            return None;
        }
        eprintln!("[strand-decode-kernel] CUDA bitslice decode dispatch ready: device 0");
        Some(gpu)
    }

    /// Device `sizeof(BitsliceEntry)` via the 1-thread probe kernel.
    pub fn gpu_entry_sizeof(&self) -> Option<u32> {
        let mut d_out: CudaSlice<u32> = self.device.alloc_zeros(1).ok()?;
        let f = self.device.get_func(MODULE, KERNEL_SIZEOF)?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe { f.launch(cfg, (&mut d_out,)) }.ok()?;
        self.device.dtoh_sync_copy(&d_out).ok()?.first().copied()
    }

    /// Pad the payload to a u32 boundary + 8 slack bytes, zero-filled — so the kernel's
    /// `bs_load_u32_le(.., ++word_idx)` look-ahead is always in-bounds and reads zeros,
    /// matching the Metal `upload_payload` and the CPU reader's zero-fill past end.
    fn upload_payload(&self, payload: &[u8]) -> Option<CudaSlice<u8>> {
        let padded_len = payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..payload.len()].copy_from_slice(payload);
        self.device.htod_sync_copy(&padded).ok()
    }

    fn shared_bytes(l_bits: u32) -> u32 {
        ((1usize << l_bits) * std::mem::size_of::<i32>()) as u32
    }

    /// Decode a whole tensor to integer Q12, byte-identical to the CPU/Metal reference.
    ///
    /// One device output buffer (`total*4`), payload + LUT + table uploaded once, then
    /// the BLOCK TABLE is dispatched in bounded batches (`cuda_decode_batch_size`) so a
    /// single launch never carries an unbounded grid; each launch's `out_off` values are
    /// absolute, so they write disjoint slices of the shared output. Use this when the
    /// full output fits device memory (the common case, even at 70B per-tensor sizes on
    /// a 24 GB card). For tensors whose `total*4` itself overflows the device, see
    /// `decode_q12_chunked`.
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
        if tbl.is_empty() || total == 0 {
            return Some(Vec::new());
        }

        let d_w = self.upload_payload(payload)?;
        let d_lut = self.device.htod_sync_copy(lut).ok()?;
        let d_tbl = self.device.htod_sync_copy(tbl).ok()?;
        let mut d_out: CudaSlice<i32> = self.device.alloc_zeros(total).ok()?;

        let f = self.device.get_func(MODULE, KERNEL_DECODE)?;
        let shared = Self::shared_bytes(l_bits);

        let n_blocks = tbl.len();
        let max_n = tbl.iter().map(|e| e.n as usize).max().unwrap_or(1).max(1);
        let batch = cuda_decode_batch_size(max_n, n_blocks);

        let mut b0 = 0usize;
        while b0 < n_blocks {
            let b1 = (b0 + batch).min(n_blocks);
            let count = (b1 - b0) as u32;
            // Sub-slice the already-uploaded table; out_off is absolute so the launch
            // writes into the correct region of the shared d_out.
            let d_tbl_slice = d_tbl.try_slice(b0..b1).expect("block-table sub-slice in range");
            let cfg = LaunchConfig {
                grid_dim: (count.div_ceil(TPB), 1, 1),
                block_dim: (TPB, 1, 1),
                shared_mem_bytes: shared,
            };
            unsafe {
                f.clone().launch(
                    cfg,
                    (&d_w, &mut d_out, &d_tbl_slice, count, k_bits, l_bits, &d_lut),
                )
            }
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA decode launch error: {e}"))
            .ok()?;
            b0 = b1;
        }

        self.device.dtoh_sync_copy(&d_out).ok()
    }

    /// Extreme-scale variant: bound BOTH the per-launch grid AND the per-launch output
    /// buffer + host copy to `cuda_decode_batch_size` blocks, returning the
    /// concatenation. Identical bytes to `decode_q12` (blocks are output-disjoint and
    /// the LUT/payload are shared), but caps transient device + host staging so a tensor
    /// whose full `total*4` would not fit (or would, times N worker threads, blow the
    /// host cgroup) still decodes. This is the decode analogue of the encode lane's
    /// per-batch `back_flat` chunking.
    pub fn decode_q12_chunked(
        &self,
        payload: &[u8],
        tbl: &[BitsliceEntry],
        lut: &[i32],
        total: usize,
        k_bits: u32,
        l_bits: u32,
    ) -> Option<Vec<i32>> {
        assert_eq!(lut.len(), 1usize << l_bits, "LUT must have 2^L entries");
        if tbl.is_empty() || total == 0 {
            return Some(Vec::new());
        }

        // Payload + LUT uploaded once (shared across batches). The block TABLE is NOT
        // uploaded whole here — each batch uploads only its rebased slice, so device
        // residency is bounded to (payload + LUT + one batch's output + table slice).
        let d_w = self.upload_payload(payload)?;
        let d_lut = self.device.htod_sync_copy(lut).ok()?;

        let f = self.device.get_func(MODULE, KERNEL_DECODE)?;
        let shared = Self::shared_bytes(l_bits);

        let n_blocks = tbl.len();
        let max_n = tbl.iter().map(|e| e.n as usize).max().unwrap_or(1).max(1);
        let batch = cuda_decode_batch_size(max_n, n_blocks);

        let mut out = vec![0i32; total];
        let mut b0 = 0usize;
        while b0 < n_blocks {
            let b1 = (b0 + batch).min(n_blocks);
            let count = (b1 - b0) as u32;

            // Output extent for THIS batch: from the first block's out_off to the last
            // block's out_off + n. Blocks are emitted in output order by block_plans
            // (out_off strictly increasing by n), so the batch's outputs form one
            // contiguous [o0, o1) slice we can stage in isolation.
            let o0 = tbl[b0].out_off as usize;
            let o1 = tbl[b1 - 1].out_off as usize + tbl[b1 - 1].n as usize;
            debug_assert!(o1 <= total && o0 < o1, "batch output slice out of range");

            // A per-batch output buffer sized to this slice; the kernel writes absolute
            // indices, so we rebase by passing a table whose out_off is shifted by -o0.
            let mut rebased: Vec<BitsliceEntry> = tbl[b0..b1].to_vec();
            for e in rebased.iter_mut() {
                e.out_off -= o0 as u32;
            }
            let d_tbl_batch = self.device.htod_sync_copy(&rebased).ok()?;
            let mut d_out: CudaSlice<i32> = self.device.alloc_zeros(o1 - o0).ok()?;

            let cfg = LaunchConfig {
                grid_dim: (count.div_ceil(TPB), 1, 1),
                block_dim: (TPB, 1, 1),
                shared_mem_bytes: shared,
            };
            unsafe {
                f.clone().launch(
                    cfg,
                    (&d_w, &mut d_out, &d_tbl_batch, count, k_bits, l_bits, &d_lut),
                )
            }
            .map_err(|e| eprintln!("[strand-decode-kernel] CUDA decode (chunked) launch error: {e}"))
            .ok()?;

            let host_slice = self.device.dtoh_sync_copy(&d_out).ok()?;
            out[o0..o1].copy_from_slice(&host_slice);
            b0 = b1;
        }
        Some(out)
    }
}

/// Bake the per-block decode side-info into `BitsliceEntry` rows — IDENTICAL logic to
/// `metal::bake_bitslice_entries` (same `block_plans`, `SideInfo::hoist`,
/// `block_init_state`), but using THIS module's `BitsliceEntry` so it is OS-agnostic.
/// Returns `None` for vec-trellis (`vec_dim>1`) or any block with `n>256` — the SAME
/// gate as the Metal path; the caller then falls back to the CPU decode.
pub fn bake_bitslice_entries(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Option<Vec<BitsliceEntry>> {
    if cfg.vec_dim() > 1 {
        return None;
    }
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
        eff[..side.n_sub].copy_from_slice(side.eff());
        off[..side.n_sub].copy_from_slice(side.off());
        let init = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
        out.push(BitsliceEntry {
            bit_offset: plan.start_bit as u32,
            init_state: init as u32,
            out_off: plan.out_off as u32,
            n: blk.n as u32,
            eff,
            off,
        });
    }
    Some(out)
}

/// High-level convenience: bake + decode, falling back to the CPU reference
/// (`decode_lean_with_lut`, the SAME fallback the Metal path uses) when the GPU rejects
/// the config or the fast-path gate fails — so the result is byte-identical regardless
/// of which lane ran. Mirrors `metal::bitslice_decode_q12_with_lut`.
pub fn cuda_bitslice_decode_q12_with_lut(
    gpu: &BitsliceCudaDispatch,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    if cfg.vec_dim() > 1 {
        return decode_lean_with_lut(enc, cfg, lut);
    }
    let Some(tbl) = bake_bitslice_entries(enc, cfg) else {
        return decode_lean_with_lut(enc, cfg, lut);
    };
    match gpu.decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits) {
        Some(v) => v,
        None => decode_lean_with_lut(enc, cfg, lut),
    }
}

/// As above with the default codebook LUT.
pub fn cuda_bitslice_decode_q12(
    gpu: &BitsliceCudaDispatch,
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Vec<i32> {
    cuda_bitslice_decode_q12_with_lut(gpu, enc, cfg, codebook_lut(cfg.l_bits))
}

// ============================================================================
//  GPU-FREE unit tests for the batch-size math + the layout contract.
//  Mirror the in-source unit tests in strand-quant/src/cuda_dispatch.rs. Always
//  compiled under `--features cuda`. The actual on-device byte-identity gate
//  lives in a separate #[ignore]d pod test (see the integration plan).
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_entry_layout_is_80_bytes() {
        // Locks the wire layout shared with the .cu struct and the Metal struct.
        assert_eq!(std::mem::size_of::<BitsliceEntry>(), 80);
        assert_eq!(std::mem::align_of::<BitsliceEntry>(), 4);
    }

    #[test]
    fn decode_batch_never_exceeds_output_budget() {
        for &block_len in &[1usize, 32, 64, 128, 256] {
            let bs = cuda_decode_batch_size(block_len, 10_000_000);
            let out_bytes = bs * block_len * std::mem::size_of::<i32>();
            assert!(
                out_bytes <= MAX_OUT_BYTES_CUDA,
                "block_len={block_len}: batch={bs} -> out={out_bytes} > cap"
            );
            assert!(bs >= 1, "batch must be >= 1");
        }
    }

    #[test]
    fn decode_batch_clamps_and_handles_empty() {
        assert_eq!(cuda_decode_batch_size(256, 0), 0);
        assert_eq!(cuda_decode_batch_size(256, 1), 1);
        assert_eq!(cuda_decode_batch_size(256, 3), 3);
        // 512MB / (256*4) = 524288 blocks fit before clamping to n_blocks.
        assert_eq!(cuda_decode_batch_size(256, 524_288), 524_288);
        assert_eq!(
            cuda_decode_batch_size(256, 10_000_000),
            512 * 1024 * 1024 / (256 * 4)
        );
    }
}
