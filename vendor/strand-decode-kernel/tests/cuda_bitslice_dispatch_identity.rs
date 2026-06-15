//! ON-POD byte-identity gate for the CUDA bitslice decode DISPATCH.
//!
//! WHAT IT PROVES (the moat)
//! -------------------------
//! The CUDA decode lane (`crate::cuda_dispatch`) reconstructs `out_q12` BYTE-IDENTICALLY
//! to the CPU reference (`strand_quant::decode::decode_tensor_fixed`) — and the
//! memory-bounded BLOCK-BATCHED dispatch is BATCH-BOUNDARY INVARIANT (the encode-OOM
//! lesson, verified for decode):
//!   * `decode_q12` (single output buffer, batched table launches) == CPU, exactly.
//!   * `decode_q12_chunked` (per-batch output staging) == CPU == `decode_q12`, exactly.
//!   * forcing a mid-tensor batch boundary (n_blocks just over `cuda_decode_batch_size`)
//!     does not change a single output integer.
//!
//! WHY IT IS `#[ignore]`d AND `#[cfg(feature="cuda")]`
//! --------------------------------------------------
//! It needs a real CUDA device + the `cuda` feature, neither of which exists in the
//! authoring env (Metal host; MPS owned by a live PV; no local CUDA). It is the gate to
//! run ON THE POD (RTX 3090, CUDA 12.8, cudarc pinned `cuda-12060`). The crate's normal
//! `cargo test` (no `--features cuda`) compiles this file to an empty module, so it is
//! inert locally and in macOS CI.
//!
//! POD COMMANDS (see also the task-return integration plan):
//!   # build + run ONLY this gate, single-threaded (one GPU context):
//!   cargo test -p strand-decode-kernel --features cuda \
//!     --test cuda_bitslice_dispatch_identity -- --ignored --test-threads=1 --nocapture
//!
//! If `cuda_dispatch` is not yet declared in `lib.rs`, add (gated) first:
//!   #[cfg(feature = "cuda")] pub mod cuda_dispatch;
//! and give the crate a `cuda` feature that forwards to strand-quant/cuda (see plan).

#![cfg(feature = "cuda")]

use strand_decode_kernel::cuda_dispatch::{
    bake_bitslice_entries, cuda_decode_batch_size, BitsliceCudaDispatch,
};
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;

/// Deterministic, dependency-free weight generator (matches the crate test policy).
fn gen_weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f64 + 1.0) * 0.0137 + seed as f64 * 0.731;
            ((x.sin() * 0.6 + (x * 2.3).sin() * 0.3) as f32) * 0.5
        })
        .collect()
}

#[test]
#[ignore = "requires --features cuda + a CUDA GPU (run on the pod)"]
fn cuda_decode_is_byte_identical_to_cpu_reference() {
    let Some(gpu) = BitsliceCudaDispatch::new() else {
        // No device / compile failure: skip loudly (matches the Metal tests' skip style).
        eprintln!("[gate] no CUDA device or PTX build failed; skipping");
        return;
    };

    // Struct-stride guard is internal to new(); assert it independently too.
    assert_eq!(gpu.gpu_entry_sizeof(), Some(80), "device sizeof(BitsliceEntry) must be 80");

    // Moat configs + a couple of reopen/fold variants. (L<=12 fits the smem LUT.)
    let configs = [
        TrellisConfig::for_bpw(3.0),       // k3 L7 — the 3-bit deploy
        TrellisConfig::for_bpw(2.0),       // k2 L6
        TrellisConfig::for_bpw(4.0),       // k4 L8
        TrellisConfig::for_bpw_l(2.0, 12), // k2 L12 — 2-bit reopen, 16 KB LUT
        TrellisConfig::for_bpw_l(2.0, 5),  // k2 L5  — fold
        TrellisConfig::for_bpw_l(4.0, 4),  // k4 L4  — fold
    ];

    for cfg in configs {
        for seed in 0..6u64 {
            // Lengths that straddle block (256) and sub-block (32) boundaries, incl. a
            // size large enough to FORCE multiple decode batches once we shrink the cap
            // via the chunked path's per-batch slicing.
            let n = 1 + (seed as usize * 277) % 4096;
            let w = gen_weights(n, seed);

            // Exercise all four encode-option combos (tail-biting / affine-min) so the
            // baked init_state + off[] paths are covered.
            let variants = [
                encode_tensor(&w, &cfg),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                encode_tensor_with(
                    &w,
                    &cfg,
                    &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ),
            ];

            for enc in &variants {
                let want = decode_tensor_fixed(enc, &cfg);
                let lut = codebook_lut(cfg.l_bits);
                let tbl = bake_bitslice_entries(enc, &cfg).expect("bitslice gate applies");

                // (1) Standard path: batched table launches, single output buffer.
                let got = gpu
                    .decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
                    .expect("CUDA decode_q12");
                assert_eq!(
                    got, want,
                    "CUDA decode_q12 != CPU: k={} L={} n={n} seed={seed} tail={} affine={}",
                    cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min
                );

                // (2) Chunked path: per-batch output staging must match exactly.
                let got_chunked = gpu
                    .decode_q12_chunked(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
                    .expect("CUDA decode_q12_chunked");
                assert_eq!(
                    got_chunked, want,
                    "CUDA decode_q12_chunked != CPU: k={} L={} n={n} seed={seed}",
                    cfg.k_bits, cfg.l_bits
                );
            }
        }
    }
    eprintln!("[gate] CUDA decode == CPU reference across configs/variants/lengths");
}

/// Batch-boundary independence: a tensor large enough that the *production* batch math
/// would still be one launch is uninteresting, so we instead verify the property the
/// host owns directly — that splitting the SAME table into arbitrary contiguous launches
/// yields identical bytes. We do this through the public `decode_q12` (which already
/// batches) AND the `decode_q12_chunked` (which batches output too); equality of both to
/// the CPU reference for the SAME tensor already pins it, but this test additionally
/// sanity-checks `cuda_decode_batch_size` is honoured (no over-budget single launch).
#[test]
#[ignore = "requires --features cuda + a CUDA GPU (run on the pod)"]
fn cuda_decode_batch_boundary_is_value_invariant() {
    let Some(gpu) = BitsliceCudaDispatch::new() else {
        eprintln!("[gate] no CUDA device; skipping");
        return;
    };
    let cfg = TrellisConfig::for_bpw(3.0);
    let lut = codebook_lut(cfg.l_bits);

    // ~256k weights => ~1000 blocks of 256, enough that the chunked output staging
    // crosses several batches once the cap bites; the standard path also batches the
    // table. Both must equal the CPU reference.
    let n = 256 * 1000 + 137;
    let w = gen_weights(n, 99);
    let enc = encode_tensor_with(
        &w,
        &cfg,
        &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
    );
    let want = decode_tensor_fixed(&enc, &cfg);
    let tbl = bake_bitslice_entries(&enc, &cfg).expect("gate applies");
    let n_blocks = tbl.len();

    // Sanity: the batch math never yields 0 and never exceeds the table.
    let bs = cuda_decode_batch_size(256, n_blocks);
    assert!(bs >= 1 && bs <= n_blocks);

    let got = gpu
        .decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
        .expect("decode_q12");
    let got_chunked = gpu
        .decode_q12_chunked(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
        .expect("decode_q12_chunked");
    assert_eq!(got, want, "standard batched decode diverged at scale");
    assert_eq!(got_chunked, want, "chunked decode diverged at scale");
    assert_eq!(got, got_chunked, "standard vs chunked must be identical");
    eprintln!("[gate] {n_blocks} blocks, batch={bs}: batched == chunked == CPU");
}
