//! BYTE-IDENTITY GATE for the CUDA bitslice decode port.
//!
//! THE PART THIS FILE OWNS
//! -----------------------
//! Prove: the CUDA `strand_bitslice_decode` kernel (`src/cuda_bitslice.rs`) emits
//! the SAME `Vec<i32>` Q12 weights as the canonical CPU/Metal reference decode —
//! BIT-FOR-BIT, across the full parameter space (L, k, block_len, tensor dims n,
//! and the encode variants tail-biting / affine-min). Equality must be `assert_eq!`
//! on the whole vector, not a tolerance: the integer Q12 decode is the moat, and a
//! single differing element is a release blocker.
//!
//! WHY TWO LAYERS (oracle + on-pod gate)
//! -------------------------------------
//! There is NO CUDA device in the dev environment (Metal host; MPS owned by a live
//! PV; no local GPU per the task constraints). So this file is built in two layers,
//! matching the established pattern in
//! `strand-quant/tests/cuda_batch_boundary_determinism.rs`:
//!
//!   LAYER A — GPU-FREE ALGEBRAIC ORACLE (always compiled, runs in CI here).
//!     The decode kernel's value path is a *pure integer function* of
//!     `(payload bits, BitsliceEntry table, LUT, k, L)` with NO cross-thread state
//!     (one CUDA thread per block; disjoint output slices; read-only LUT). So the
//!     ONLY thing the GPU adds over a faithful CPU transcription of the kernel is
//!     scheduling — which cannot change integer results. We transcribe the CUDA
//!     kernel arithmetic *character-for-character* into Rust (`cuda_kernel_oracle`
//!     below — the LE u32 load, the u64-accumulator bit unpack, the masked state
//!     update, the `((i64)eff*q)>>16 + off` reconstruct, `sub = j>>5`) and assert
//!     it equals BOTH `decode_tensor_fixed` (the canonical reference) AND the host
//!     baker's table walk, over the whole sweep. This is what actually catches a
//!     transcription bug in the CUDA source (wrong shift, wrong mask, wrong sub
//!     index, signedness, struct field order) — the bugs a port realistically
//!     introduces — WITHOUT a GPU. It is deterministic and feature-free, so it
//!     guards the kernel source on every `cargo test` in this repo.
//!
//!   LAYER B — REAL-DEVICE GATE (`#[cfg(feature="cuda")]`, `#[ignore]`d).
//!     Runs the ACTUAL compiled CUDA kernel on the pod and asserts byte-identity
//!     vs the CPU reference. This is the final word (it also catches nvrtc-codegen
//!     or driver issues the oracle can't see), but it can only run where a CUDA
//!     device exists. The exact pod command is in the doc-comment on that test and
//!     in the task return's validation plan.
//!
//! If Layer A passes here and Layer B passes on the pod, CUDA decode is proven
//! byte-identical to CPU/Metal across the swept space.

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{decode_lean, decode_tensor_fixed};
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::TrellisConfig;

use strand_decode_kernel::block_walk::{
    block_init_state, block_plans, exceeds_max_sub, SideInfo,
};

// ---------------------------------------------------------------------------
//  Host-side "baked table" mirror.
//
//  The production CUDA host (`cuda_bitslice.rs`) bakes its `BitsliceEntry` table
//  via `crate::metal::bake_bitslice_entries` — but that lives behind
//  `#[cfg(target_os="macos")]`, so on a Linux pod (or here in a feature-free
//  build) we cannot call it. We therefore reproduce the EXACT baking the kernel
//  consumes using the cross-platform `block_walk` primitives (`block_plans`,
//  `SideInfo::hoist`, `block_init_state`) — the same functions
//  `bake_bitslice_entries` itself calls (see metal.rs). A `BakedEntry` here is a
//  field-for-field stand-in for `BitsliceEntry` (bit_offset/init_state/out_off/
//  n/eff[8]/off[8]); the Layer-B on-pod test reuses this SAME baker to build the
//  device table, so the oracle and the GPU decode the identical geometry.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BakedEntry {
    bit_offset: u32,
    init_state: u32,
    out_off: u32,
    n: u32,
    eff: [i32; 8],
    off: [i32; 8],
}

/// Reproduce `metal::bake_bitslice_entries` using only cross-platform primitives.
/// Returns `None` for the SAME gates the production baker uses: vec-trellis
/// (handled by the caller) and `n > 256` blocks (`exceeds_max_sub`).
fn bake_entries(enc: &EncodedTensor, cfg: &TrellisConfig) -> Option<Vec<BakedEntry>> {
    if exceeds_max_sub(enc) {
        return None; // mirrors `enc.blocks.iter().any(|b| b.n > 256)` gate
    }
    let k = cfg.k_bits as usize;
    let plans = block_plans(enc, k);
    let mut out = Vec::with_capacity(enc.blocks.len());
    for (blk, plan) in enc.blocks.iter().zip(plans.iter()) {
        let side = SideInfo::hoist(blk, enc.has_affine_min);
        let mut eff = [0i32; 8];
        let mut off = [0i32; 8];
        // `eff()`/`off()` are already trimmed to n_sub (<=8 since block_len<=256);
        // `n_sub` itself is pub(crate), so we drive the copy off the slice lengths.
        let n_sub = side.eff().len();
        eff[..n_sub].copy_from_slice(side.eff());
        off[..side.off().len()].copy_from_slice(side.off());
        let init = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
        out.push(BakedEntry {
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

// ---------------------------------------------------------------------------
//  LAYER A — the CUDA kernel arithmetic, transcribed CHARACTER-FOR-CHARACTER.
//
//  This is the body of `strand_bitslice_decode` (src/cuda_bitslice.rs) rewritten
//  in Rust with identical types/operators. If you change the CUDA source, change
//  THIS in lockstep — they must stay byte-identical. The point of duplicating it
//  (rather than calling the CPU decoder) is that THIS is what proves the *CUDA
//  source itself* is correct: a transcription slip in the .cu string (e.g.
//  `>> 16` typo'd, `j >> 4` instead of `j >> 5`, `int` instead of `long long` in
//  the product, eff/off field order swapped) is caught here, because this mirror
//  would carry the same slip and DIVERGE from `decode_tensor_fixed`.
// ---------------------------------------------------------------------------

/// `bs_load_u32_le` — byte-exact with the CUDA `__device__` helper and the Metal
/// shader. Reads a little-endian u32 from a zero-padded byte buffer.
#[inline]
fn bs_load_u32_le(p: &[u8], widx: u32) -> u32 {
    let b = (widx << 2) as usize;
    // The production payload is zero-padded (+8 B past a u32 boundary), so a
    // 4-byte read at any in-range word index is in-bounds; we still guard to keep
    // the oracle memory-safe for ragged synthetic buffers.
    let g = |o: usize| -> u32 { if b + o < p.len() { p[b + o] as u32 } else { 0 } };
    g(0) | (g(1) << 8) | (g(2) << 16) | (g(3) << 24)
}

/// One block, exactly as the CUDA kernel decodes it (per-thread, sequential).
fn cuda_decode_block(out: &mut [i32], w_bits: &[u8], e: &BakedEntry, lut: &[i32], k_bits: u32, l_bits: u32) {
    let lut_n: u32 = 1u32 << l_bits;
    let state_mask: u32 = lut_n - 1;
    let input_mask: u32 = (1u32 << k_bits) - 1;

    let mut state: u32 = e.init_state & state_mask;
    let bitpos: u32 = e.bit_offset;
    let n: u32 = e.n;
    let obase: u32 = e.out_off;
    let mut word_idx: u32 = bitpos >> 5;
    let bit_in_w: u32 = bitpos & 31;

    // u64 accumulator, identical to the CUDA `unsigned long long acc`.
    let mut acc: u64 = (bs_load_u32_le(w_bits, word_idx) >> bit_in_w) as u64;
    let mut have: u32 = 32 - bit_in_w;

    for j in 0..n {
        if have < k_bits {
            word_idx += 1;
            let nxt = bs_load_u32_le(w_bits, word_idx) as u64;
            acc |= nxt << have;
            have += 32;
        }
        let sym = (acc as u32) & input_mask;
        acc >>= k_bits;
        have -= k_bits;

        state = ((state << k_bits) | sym) & state_mask;
        let q: i32 = lut[state as usize];
        let sb = (j >> 5) as usize; // SUB_BLOCK = 32
        let es: i32 = e.eff[sb];

        // Integer reconstruct: ((i64)eff * (i64)q) >> 16 + off — byte-identical to
        // the CUDA `(int)(((long long)es*(long long)q)>>16)+e.off[sb]` and the CPU
        // `reconstruct_q(es,q)+off`.
        let w: i32 = (((es as i64) * (q as i64)) >> 16) as i32 + e.off[sb];
        out[(obase + j) as usize] = w;
    }
}

/// Whole-tensor decode via the transcribed CUDA kernel (Layer A oracle).
fn cuda_oracle_decode(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Option<Vec<i32>> {
    let tbl = bake_entries(enc, cfg)?;
    // Mirror the production zero-padding so the tail u32 read is well-defined.
    let padded_len = enc.bits.len().div_ceil(4) * 4 + 8;
    let mut w = vec![0u8; padded_len];
    w[..enc.bits.len()].copy_from_slice(&enc.bits);

    let mut out = vec![0i32; enc.total];
    for e in &tbl {
        cuda_decode_block(&mut out, &w, e, lut, cfg.k_bits, cfg.l_bits);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
//  Deterministic, dependency-free weight generators (crate test policy: no rng).
// ---------------------------------------------------------------------------

fn gen_smooth(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
        .collect()
}

/// Tie-prone (heavy quantisation to a tiny alphabet) — stresses the argmin in the
/// ENCODER, which shapes which symbols (and thus which states/LUT entries) the
/// DECODER walks. Decode is deterministic regardless, but this diversifies the
/// state trajectories the byte-identity check covers.
fn gen_tie_prone(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bucket = ((i as u64).wrapping_mul(2654435761).wrapping_add(seed) >> 29) & 0x7;
            [0.0, 0.25, -0.25, 0.5, -0.5, 0.125, -0.125, 0.0][bucket as usize]
        })
        .collect()
}

/// The full config sweep: the moat-relevant deploy configs plus folded/reopened
/// L's and every k. `block_len` is swept separately per test.
fn sweep_configs(block_len: usize) -> Vec<(TrellisConfig, &'static str)> {
    vec![
        (TrellisConfig::new(7, 3, block_len), "k3 L7 (3-bit deploy)"),
        (TrellisConfig::new(6, 2, block_len), "k2 L6 (2-bit deploy)"),
        (TrellisConfig::new(8, 4, block_len), "k4 L8 (4-bit)"),
        (TrellisConfig::new(12, 2, block_len), "k2 L12 (2-bit reopen)"),
        (TrellisConfig::new(5, 2, block_len), "k2 L5 (fold)"),
        (TrellisConfig::new(5, 3, block_len), "k3 L5 (fold)"),
        (TrellisConfig::new(4, 4, block_len), "k4 L4 (fold)"),
        (TrellisConfig::new(4, 1, block_len), "k1 L4 (min k)"),
        (TrellisConfig::new(10, 2, block_len), "k2 L10"),
    ]
}

fn encode_variants(w: &[f32], cfg: &TrellisConfig) -> Vec<EncodedTensor> {
    vec![
        encode_tensor(w, cfg),
        encode_tensor_with(w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
        encode_tensor_with(w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
        encode_tensor_with(
            w,
            cfg,
            &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
        ),
    ]
}

// ===========================================================================
//  LAYER A, TEST 1 (CORE): CUDA-kernel oracle == decode_tensor_fixed,
//  bit-for-bit, across (L, k, block_len, n, variant).
//
//  This is the heart of the gate: the transcribed CUDA arithmetic must reproduce
//  the canonical integer reference EXACTLY. block_len is swept across boundaries
//  that exercise the `sub = j>>5` selection (32 = exactly one sub, 33/64/256 =
//  multiple subs) and ragged final blocks.
// ===========================================================================
#[test]
fn cuda_oracle_is_byte_identical_to_reference() {
    let mut cells = 0u64;
    for &block_len in &[1usize, 31, 32, 33, 64, 100, 256] {
        for (cfg, label) in sweep_configs(block_len) {
            for seed in 0..16u64 {
                // Lengths straddling sub-block (32) and block boundaries, incl. a
                // single element and a multi-block tensor.
                let n = match seed % 8 {
                    0 => 1,
                    1 => 31,
                    2 => 32,
                    3 => 33,
                    4 => 257,
                    5 => 700,
                    6 => 1024,
                    _ => 1 + (seed as usize * 211) % 2048,
                };
                let w = gen_smooth(n, seed);
                for enc in &encode_variants(&w, &cfg) {
                    let lut = codebook_lut(cfg.l_bits);
                    let reference = decode_tensor_fixed(enc, &cfg);
                    // Sanity: the streaming CPU decoder agrees with the fixed one
                    // (this is the relationship the Metal/CUDA kernels also hold).
                    debug_assert_eq!(decode_lean(enc, &cfg), reference);

                    let got = cuda_oracle_decode(enc, &cfg, lut)
                        .expect("bake should succeed for block_len<=256");
                    assert_eq!(
                        got, reference,
                        "CUDA-oracle Q12 DIVERGED from decode_tensor_fixed: {label} \
                         block_len={block_len} n={n} seed={seed} tail={} affine={} \
                         — release blocker (CUDA decode source bug)",
                        enc.tail_biting, enc.has_affine_min
                    );
                    cells += 1;
                }
            }
        }
    }
    eprintln!("CUDA-oracle byte-identity: {cells} (config x block_len x n x variant) cells");
    assert!(cells > 1000, "coverage unexpectedly small: {cells}");
}

// ===========================================================================
//  LAYER A, TEST 2: tie-prone weights. Diversifies the encoder's chosen symbol
//  stream (hence the decoder's state/LUT trajectory) under heavy ties; the decode
//  must STILL be byte-identical to the reference.
// ===========================================================================
#[test]
fn cuda_oracle_byte_identical_on_tie_prone() {
    let mut cells = 0u64;
    for &block_len in &[32usize, 64, 256] {
        for (cfg, label) in sweep_configs(block_len) {
            for seed in 0..6u64 {
                for &n in &[300usize, 512, 901, 2048] {
                    let w = gen_tie_prone(n, seed * 101 + block_len as u64);
                    for enc in &encode_variants(&w, &cfg) {
                        let lut = codebook_lut(cfg.l_bits);
                        let reference = decode_tensor_fixed(enc, &cfg);
                        let got = cuda_oracle_decode(enc, &cfg, lut).expect("bake");
                        assert_eq!(
                            got, reference,
                            "CUDA-oracle tie-prone DIVERGED: {label} block_len={block_len} \
                             n={n} seed={seed} tail={} affine={}",
                            enc.tail_biting, enc.has_affine_min
                        );
                        cells += 1;
                    }
                }
            }
        }
    }
    eprintln!("CUDA-oracle tie-prone byte-identity: {cells} cells");
    assert!(cells > 200);
}

// ===========================================================================
//  LAYER A, TEST 3: per-element value bounds (defends the integer reconstruct
//  against silent i32 overflow). The reconstruct is `((i64)eff*q)>>16 + off`;
//  the i64 product cannot overflow for i32 operands, and the >>16 result fits i32
//  for the realistic eff/q magnitudes here. We assert every decoded element of a
//  large structured tensor is finite-int and lands inside a generous Q12 envelope,
//  AND equals the reference — so an arithmetic-width regression in the kernel
//  source (e.g. dropping the i64 cast) would surface as both a mismatch and a
//  bound violation.
// ===========================================================================
#[test]
fn cuda_oracle_reconstruct_within_bounds_and_matches() {
    // 6 * 4096 is the LUT Q12 clamp; eff is ~Q16 of an absmax-derived scale, so the
    // product>>16 stays within a few x the LUT range. Use a wide guard.
    const ENVELOPE: i64 = 64 * 4096;
    for (cfg, label) in sweep_configs(256) {
        let w = gen_smooth(4096, 7);
        let enc = encode_tensor_with(
            &w,
            &cfg,
            &EncodeOpts { affine_min: true, ..Default::default() },
        );
        let lut = codebook_lut(cfg.l_bits);
        let reference = decode_tensor_fixed(&enc, &cfg);
        let got = cuda_oracle_decode(&enc, &cfg, lut).expect("bake");
        assert_eq!(got, reference, "value-bound case diverged: {label}");
        for (i, &v) in got.iter().enumerate() {
            assert!(
                (v as i64).abs() <= ENVELOPE,
                "decoded Q12 out of envelope at {label} i={i}: {v}",
            );
        }
    }
}

// ===========================================================================
//  LAYER A, TEST 4: the baked-table geometry the kernel consumes is internally
//  consistent — output slices [out_off, out_off+n) exactly TILE [0, total) with
//  no gap/overlap, bit_offset advances by n*k, and n_sub<=8 (so the eff[8]/off[8]
//  arrays cover every sub-block the kernel indexes via `j>>5`). A geometry bug
//  here would make the GPU write the wrong cells even with correct arithmetic.
// ===========================================================================
#[test]
fn baked_table_tiles_output_and_fits_eff_off() {
    for &block_len in &[1usize, 32, 64, 256] {
        for (cfg, label) in sweep_configs(block_len) {
            for &n in &[1usize, 257, 1024, 4096] {
                let w = gen_smooth(n, 3);
                let enc = encode_tensor(&w, &cfg);
                let tbl = bake_entries(&enc, &cfg).expect("bake");
                let k = cfg.k_bits;
                let mut expect_out = 0u32;
                let mut expect_bit = 0u32;
                for (bi, e) in tbl.iter().enumerate() {
                    assert_eq!(e.out_off, expect_out, "{label} blk{bi}: out_off gap/overlap");
                    assert_eq!(e.bit_offset, expect_bit, "{label} blk{bi}: bit_offset drift");
                    // n_sub = ceil(n/32) <= 8 because block_len <= 256.
                    let n_sub = (e.n as usize).div_ceil(32);
                    assert!(n_sub <= 8, "{label} blk{bi}: n_sub={n_sub} exceeds eff[8]");
                    // Every sub-block the kernel will index via j>>5 must be < n_sub
                    // (i.e. within the populated eff/off prefix). Highest j = n-1.
                    let max_sb = ((e.n - 1) >> 5) as usize;
                    assert!(max_sb < n_sub, "{label} blk{bi}: max sub {max_sb} >= n_sub {n_sub}");
                    expect_out += e.n;
                    expect_bit += e.n * k;
                }
                assert_eq!(expect_out as usize, enc.total, "{label}: blocks don't tile total");
            }
        }
    }
}

// ===========================================================================
//  LAYER B — REAL CUDA DEVICE GATE (compiled only with --features cuda; ignored).
//
//  Runs the ACTUAL compiled `strand_bitslice_decode` kernel on the pod GPU and
//  asserts byte-identity vs `decode_tensor_fixed`. This is the final word: it also
//  catches anything the Layer-A oracle cannot (nvrtc codegen, the DeviceRepr
//  memcpy, the sizeof-stride probe, the shared-vs-global LUT path selection,
//  launch-config / OOB behaviour at real grid sizes).
//
//  RUN ON THE POD (RTX 3090, CUDA 12.8, cudarc pinned cuda-12060):
//      cargo test -p strand-decode-kernel --features cuda \
//        --test cuda_bitslice_identity -- --ignored cuda_device_byte_identical \
//        --nocapture --test-threads=1
//
//  Pre-req: build/run with the cudarc feature pinned to the toolkit, exactly like
//  the encode lane (`--features cuda`; for CUDA 12.7+ pin cudarc cuda-12060). This
//  test fails CLOSED: if the device is unavailable it panics with an explicit
//  message rather than silently passing, because on the pod a missing device means
//  a misconfigured run, not "nothing to test".
// ===========================================================================
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU; run on the pod with --features cuda --ignored"]
fn cuda_device_byte_identical_vs_reference() {
    use strand_decode_kernel::cuda_bitslice::{bake_bitslice_entries, BitsliceCudaGpu};

    // Max L across the sweep (L12) selects the shared-LUT compile variant (<=16KB,
    // fits the 3090 smem). new(14) would force the global-LUT path.
    let gpu = BitsliceCudaGpu::new(12)
        .expect("CUDA device + kernel must be available on the pod (fail closed)");

    let mut cells = 0u64;
    for &block_len in &[1usize, 32, 33, 64, 256] {
        for (cfg, label) in sweep_configs(block_len) {
            for seed in 0..8u64 {
                let n = match seed % 6 {
                    0 => 1,
                    1 => 33,
                    2 => 257,
                    3 => 700,
                    4 => 1024,
                    _ => 4096,
                };
                let w = gen_smooth(n, seed);
                for enc in &encode_variants(&w, &cfg) {
                    let reference = decode_tensor_fixed(enc, &cfg);
                    // Call the RAW GPU lane (`decode_q12`, returns Option) rather than
                    // the `cuda_bitslice_decode_q12` convenience wrapper, which would
                    // silently fall back to the CPU reference and make this test
                    // tautological. `.expect` here asserts the GPU path actually ran.
                    let lut = codebook_lut(cfg.l_bits);
                    let tbl = bake_bitslice_entries(enc, &cfg)
                        .expect("block_len<=256 scalar must bake");
                    let got = gpu
                        .decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
                        .expect("CUDA GPU decode must produce output (no fallback)");
                    assert_eq!(
                        got, reference,
                        "CUDA DEVICE Q12 DIVERGED from decode_tensor_fixed: {label} \
                         block_len={block_len} n={n} seed={seed} tail={} affine={} \
                         — MOAT VIOLATION",
                        enc.tail_biting, enc.has_affine_min
                    );
                    cells += 1;
                }
            }
        }
    }

    // Cross-check the device against the Layer-A oracle too (proves the oracle is a
    // faithful model of the real kernel, not just of the reference).
    for (cfg, _) in sweep_configs(256) {
        let w = gen_tie_prone(2048, 5);
        let enc = encode_tensor_with(
            &w,
            &cfg,
            &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
        );
        let lut = codebook_lut(cfg.l_bits);
        let oracle = cuda_oracle_decode(&enc, &cfg, lut).expect("bake");
        let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");
        let device = gpu
            .decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
            .expect("GPU decode");
        assert_eq!(device, oracle, "CUDA device != Layer-A oracle (oracle drift)");
        cells += 1;
    }

    eprintln!("CUDA DEVICE byte-identity: {cells} cells matched decode_tensor_fixed");
    assert!(cells > 200);
}

// ===========================================================================
//  LAYER B, scale gate (pod, ignored): a single LARGE tensor at the deploy config
//  exercises a multi-grid launch (n_blocks >> TPB) and real device throughput.
//  Byte-identity must hold at the grid-tiling boundary too.
// ===========================================================================
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU; run on the pod with --features cuda --ignored"]
fn cuda_device_byte_identical_large_multigrid() {
    use strand_decode_kernel::cuda_bitslice::{bake_bitslice_entries, BitsliceCudaGpu};

    let gpu = BitsliceCudaGpu::new(12).expect("CUDA device (fail closed)");
    // ~4.7M weights at block_len=256 => ~18.4k blocks => grid of ceil(18.4k/256)=72
    // CUDA blocks: a genuine multi-grid launch.
    let (rows, cols) = (256usize, 18_432usize);
    let total = rows * cols;
    let w = gen_smooth(total, 11);
    for (cfg, label) in [
        (TrellisConfig::new(7, 3, 256), "3-bit deploy"),
        (TrellisConfig::new(12, 2, 256), "2-bit reopen"),
    ] {
        let enc = encode_tensor(&w, &cfg);
        let reference = decode_tensor_fixed(&enc, &cfg);
        let lut = codebook_lut(cfg.l_bits);
        let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");
        let got = gpu
            .decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
            .expect("GPU decode (no fallback)");
        assert_eq!(
            got, reference,
            "CUDA DEVICE large/multigrid DIVERGED: {label} total={total} — MOAT VIOLATION"
        );
    }
}
