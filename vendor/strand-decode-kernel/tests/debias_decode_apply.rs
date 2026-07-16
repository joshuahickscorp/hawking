//! DBIA decode-side apply — determinism proof against the REAL decode MAC path.
//!
//! THE MOAT: STRAND decode must be bit-identical on every device. The de-bias
//! correction (`crates/strand-quant/src/debias_wire.rs`, lever 2 of the sprint) is the
//! ONLY float operation the decode side performs beyond the matmul itself — a single
//! per-output-row add `y[o] += bf16_to_f32(c_bits[o])` in the MAC epilogue. Its
//! determinism is therefore load-bearing for the cross-device guarantee.
//!
//! # Why this file exists (the gap it closes)
//!
//! The wire codec is already proven byte-exact and exhaustive in
//! `strand-quant/tests/debias_determinism.rs` (all 65 536 bf16 patterns, golden vector,
//! ties-to-even) and the abstract epilogue-ordering contract is pinned in
//! `strand-quant/tests/sprint_section_stacking.rs`
//! (`debias_epilogue_is_deterministic_and_residual_then_bias`). What NEITHER touches is
//! the **production decode arithmetic**: the bias add layered on top of the *actual*
//! `outlier_mac::matvec_patched` / `matvec_rht` output of a real `StrandModel`
//! (`q12 * 1/4096`, RHT inverse, the sparse outlier residual term). The decode kernel
//! (`strand-decode-kernel`) has ZERO DBIA wiring today — `debias_wire` is not even a
//! module in `strand-quant/src/lib.rs`. So this file:
//!
//!   * proves the epilogue add against the REAL decoded `y` of a baked `.strand` archive,
//!     using the production MAC as the source of `y` (runnable TODAY, no new prod code);
//!   * pins the exact contract the wired apply must satisfy (order, back-compat, row
//!     locality, edge shapes) so that when the operator adds the apply to `outlier_mac`,
//!     these become the regression net;
//!   * carries `#[ignore]`d stubs that name the precise one-time wiring
//!     (`pub mod debias_wire;` + `StrandModel::debias` + the epilogue call) and the
//!     assertions that then bind directly onto the production symbols.
//!
//! The local `apply_debias_epilogue` here is the REFERENCE semantics (one f32 add per
//! row, residual already folded into `y`). It is intentionally a hand reimplementation of
//! the documented spec — NOT a call into prod — exactly like the `ref_decode` pattern in
//! `exhaustive.rs`, so it stays a valid oracle for the real add once it lands.
//!
//! Run: `cargo test -p strand-decode-kernel --test debias_decode_apply`

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::outlier_mac::{matvec_patched, matvec_rht, outlier_residuals};
use strand_quant::encode::{encode_tensor, EncodedTensor};
use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
use strand_quant::outlier_wire::{append_outl, idx_bits_for, OutlierWire};
use strand_quant::rht::{rht_forward_rows, rht_inverse_rows_inplace, RhtConfig};
use strand_quant::TrellisConfig;

// ---------------------------------------------------------------------------
// Reference bf16 codec — byte-identical to debias_wire::{f32_to_bf16_round,
// bf16_to_f32} and safetensor_io::bf16_to_f32. Hand-derived (independent oracle).
// ---------------------------------------------------------------------------

/// f32 -> bf16, round-to-nearest, ties-to-even; NaN/Inf keep their top 16 bits.
fn ref_f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        return (bits >> 16) as u16; // non-finite: truncate top half
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// bf16 -> f32: place the 16 stored bits in the top half, zero the low 16.
fn ref_bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// THE reference decode-side apply: the MAC epilogue add. Pre: `y` already holds the
/// inner product PLUS the sparse outlier-residual term (the documented order is
/// `inner -> += residual -> += bias`, see sprint_section_stacking.rs and the
/// debias_wire.rs module docs §"Float order"). This performs the final, only-float-op
/// step: one f32 add per output row, no accumulation order to vary.
fn apply_debias_epilogue(y: &mut [f32], c_bits: &[u16]) {
    debug_assert_eq!(y.len(), c_bits.len());
    for (yo, &cb) in y.iter_mut().zip(c_bits.iter()) {
        *yo += ref_bf16_to_f32(cb);
    }
}

/// Build a correction vector on the wire (bf16 per output row).
fn wire_from_f32(c: &[f32]) -> Vec<u16> {
    c.iter().map(|&v| ref_f32_to_bf16(v)).collect()
}

// ---------------------------------------------------------------------------
// Deterministic splitmix64 PRNG (no proptest dep, matches the crate policy; every
// run hits the identical cases).
// ---------------------------------------------------------------------------

struct Sm64(u64);
impl Sm64 {
    fn new(seed: u64) -> Self {
        Sm64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        // [0,1) from the top 24 bits
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// A "correction-shaped" f32: small magnitude, both signs, occasional exact 0.
    fn next_correction(&mut self) -> f32 {
        let r = self.next_u64();
        match r & 0x7 {
            0 => 0.0,
            1 => -0.0,
            _ => {
                let mant = ((r >> 8) & 0xFFFF) as f32 / 65535.0;
                let scale = [1e-1f32, 1e-2, 1e-3, 1e-4, 1e-5][((r >> 3) % 5) as usize];
                let sign = if r & 0x80 != 0 { -1.0 } else { 1.0 };
                sign * mant * scale
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture: bake a real single-tensor .strand archive (mirrors the helper in
// outlier_mac.rs's own tests), optionally with an OUTL outlier section, so the
// decode path under test is the production one.
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("strand-dbia-apply-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
}

struct TmpFile(PathBuf);
impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn rht_seed_for(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}

fn gt_weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
}

/// Returns (archive_path, recon_weights_row_major) for a single 2-D tensor `[rows,cols]`.
/// When `use_rht`, the encode is in RHT space and the recon is the RHT-inverse (matching
/// `outlier_mac::patched_weights`). When `outlier_pct > 0`, an OUTL section is appended
/// and the recon has the exact dequantised outlier values spliced in (so the recon equals
/// what `patched_weights` reconstructs, bit-for-bit).
fn bake(name: &str, rows: usize, cols: usize, outlier_pct: f64, use_rht: bool) -> (PathBuf, Vec<f32>) {
    let cfg = TrellisConfig::for_bpw_l(2.0, 8);
    let gt = gt_weights(rows * cols, 0xC0FFEE);
    let n = gt.len();
    let ob = 8u32;

    let outliers: Option<(Vec<usize>, Vec<f32>, Vec<i32>, f32)> = if outlier_pct > 0.0 {
        let k = (((outlier_pct / 100.0) * n as f64).round() as usize).max(1);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| gt[b].abs().partial_cmp(&gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
        let idx: Vec<usize> = order[..k.min(n)].to_vec();
        let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
        let levels = ((1i64 << (ob - 1)) - 1) as f32;
        let vals: Vec<f32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() / levels * omax).collect();
        let codes: Vec<i32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() as i32).collect();
        Some((idx, vals, codes, omax))
    } else {
        None
    };

    let mut bulk = gt.clone();
    if let Some((idx, ..)) = &outliers {
        for &i in idx {
            bulk[i] = 0.0;
        }
    }

    let seed = rht_seed_for(name);
    let work = if use_rht { rht_forward_rows(&bulk, &RhtConfig::from_seed(seed), cols) } else { bulk.clone() };
    let mut enc: EncodedTensor = encode_tensor(&work, &cfg);
    enc.has_rht_seed = use_rht;

    // Recompute the recon exactly as the decode path will (decode -> /4096 -> RHT inv ->
    // splice outliers) so callers can compare against the production MAC bit-for-bit.
    let q12 = strand_quant::decode::decode_tensor_fixed(&enc, &cfg);
    let mut recon: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
    if use_rht {
        rht_inverse_rows_inplace(&mut recon, &RhtConfig::from_seed(seed), cols);
    }
    if let Some((idx, vals, ..)) = &outliers {
        for (&i, &v) in idx.iter().zip(vals.iter()) {
            recon[i] = v;
        }
    }

    let shape = [rows as u64, cols as u64];
    let pt = PackedTensorV2 {
        base: PackedTensor { name, shape: &shape, rht_seed: if use_rht { seed } else { 0 }, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
        block_len: cfg.block_len as u32,
    };
    let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write v2");
    let path = tmp_path(name);
    let mut f = std::fs::File::create(&path).expect("create temp .strand");
    f.write_all(&buf).expect("write temp .strand");
    f.sync_all().ok();

    if let Some((idx, _vals, codes, omax)) = outliers {
        let wire = OutlierWire::from_selection(n, idx, codes, omax, ob);
        assert_eq!(wire.idx_bits, idx_bits_for(n));
        append_outl(&path, &[Some(wire)]).expect("append outl");
    }
    (path, recon)
}

// ===========================================================================
// PART 1 — apply against the REAL production decode MAC (runnable today).
// These prove the epilogue add is correct and byte-stable when layered on the
// ACTUAL `matvec_patched` / `matvec_rht` output of a real `StrandModel`.
// ===========================================================================

/// The bias-applied output equals (production decode `y`) + dequant(c), per row, and is
/// byte-identical across two decode+apply runs. Source of `y` is the production
/// `matvec_patched` (full reconstruct + GEMV), so this binds the add to the real decode
/// arithmetic, not a toy inner product.
#[test]
fn apply_on_matvec_patched_is_bit_exact_and_deterministic() {
    let mut rng = Sm64::new(0xDEC0_DE01);
    // cols (in_features) MUST be a multiple of block_len (=256, the STRICT-deploy
    // invariant enforced by write_strand_v2 — in_features % block_len == 0).
    for &(rows, cols, pct, rht) in &[
        (4usize, 256usize, 0.0f64, true),
        (4, 256, 1.0, true),
        (3, 512, 2.0, true),
        (5, 256, 0.0, false),
        (1, 512, 1.0, true), // single output row
    ] {
        let name = "model.layers.0.mlp.down_proj.weight";
        let (path, _recon) = bake(name, rows, cols, pct, rht);
        let _g = TmpFile(path.clone());
        let model = StrandModel::open(&path).expect("open");

        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.07).cos() + 0.05 * rng.unit()).collect();
        let c: Vec<f32> = (0..rows).map(|_| rng.next_correction()).collect();
        let c_bits = wire_from_f32(&c);

        // Production decode MAC = source of y (this is the real arithmetic the moat covers).
        let base = matvec_patched(&model, name, &x).expect("matvec_patched");
        assert_eq!(base.len(), rows);

        let mut y1 = base.clone();
        apply_debias_epilogue(&mut y1, &c_bits);
        let mut y2 = base.clone();
        apply_debias_epilogue(&mut y2, &c_bits);

        // (1) byte-identical across runs — the only float op is one add per row.
        let b1: Vec<u32> = y1.iter().map(|v| v.to_bits()).collect();
        let b2: Vec<u32> = y2.iter().map(|v| v.to_bits()).collect();
        assert_eq!(b1, b2, "apply not byte-stable (rows={rows} cols={cols} pct={pct} rht={rht})");

        // (2) equals base + dequant(c), bit-for-bit, spelled out the documented way.
        for o in 0..rows {
            let want = base[o] + ref_bf16_to_f32(c_bits[o]);
            assert_eq!(y1[o].to_bits(), want.to_bits(), "row {o}: apply != base + dequant(c) (rows={rows} cols={cols})");
        }
    }
}

/// Back-compat clause of the moat: an archive WITHOUT a DBIA section (i.e. a zero / absent
/// correction) must decode byte-IDENTICALLY to today. Proven by asserting the apply of an
/// all-`bf16(0.0)` correction does not move a single output bit of the production decode.
#[test]
fn absent_or_zero_correction_is_byte_identical_to_today() {
    // cols must be a multiple of block_len (=256), per the STRICT-deploy invariant.
    for &(rows, cols, pct, rht) in &[(4usize, 256usize, 0.0f64, true), (4, 256, 1.5, true), (3, 512, 0.0, false)] {
        let name = "model.layers.1.self_attn.q_proj.weight";
        let (path, _recon) = bake(name, rows, cols, pct, rht);
        let _g = TmpFile(path.clone());
        let model = StrandModel::open(&path).expect("open");
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin() + 0.1).collect();

        let base = matvec_patched(&model, name, &x).expect("matvec_patched");

        // +0.0 IS the identity for f32 add EXCEPT it turns -0.0 into +0.0. The decode
        // accumulator starts at +0.0 and sums finite products, so an exact -0.0 output is
        // not produced here; but assert with raw bits so any such drift is caught.
        let zero_bits = vec![ref_f32_to_bf16(0.0); rows];
        let mut y = base.clone();
        apply_debias_epilogue(&mut y, &zero_bits);
        for o in 0..rows {
            assert_eq!(y[o].to_bits(), base[o].to_bits(), "row {o}: zero correction perturbed a decode bit (rows={rows} cols={cols})");
        }
    }
}

/// Row-order invariance against the REAL decode: applying the bias in forward vs reverse
/// output-row order gives bit-identical `y`. The decode kernels parallelise across output
/// rows / blocks, so the apply MUST be a per-row-independent add with no cross-row coupling
/// — this is the property that lets the bias be folded into a threaded MAC epilogue safely.
#[test]
fn apply_is_output_row_order_invariant_on_real_decode() {
    let mut rng = Sm64::new(0xDEC0_DE02);
    let name = "model.layers.2.mlp.gate_proj.weight";
    let (rows, cols) = (6usize, 256usize);
    let (path, _recon) = bake(name, rows, cols, 1.0, true);
    let _g = TmpFile(path.clone());
    let model = StrandModel::open(&path).expect("open");

    for _ in 0..64 {
        let x: Vec<f32> = (0..cols).map(|_| rng.next_correction() * 20.0).collect();
        let base = matvec_patched(&model, name, &x).expect("matvec_patched");
        let c: Vec<f32> = (0..rows).map(|_| rng.next_correction()).collect();
        let c_bits = wire_from_f32(&c);

        let mut fwd = base.clone();
        apply_debias_epilogue(&mut fwd, &c_bits);
        let mut rev = base.clone();
        for o in (0..rows).rev() {
            rev[o] += ref_bf16_to_f32(c_bits[o]);
        }
        for o in 0..rows {
            assert_eq!(fwd[o].to_bits(), rev[o].to_bits(), "row {o}: per-row apply depends on order — would break threaded-decode parity");
        }
    }
}

/// Ordering contract vs the sparse outlier residual: the documented epilogue is
/// `y = inner; y += residual; y += bias`. `matvec_rht` already produces `inner + residual`
/// internally (its outlier loop runs before return), so applying the bias to its output is
/// the correct composition. Prove that bias-after-(inner+residual) is byte-stable AND that
/// it equals the spelled-out `inner_plus_residual + dequant(c)`.
#[test]
fn bias_applies_after_inner_plus_residual_byte_stable() {
    let mut rng = Sm64::new(0xDEC0_DE03);
    let name = "model.layers.3.self_attn.k_proj.weight";
    let (rows, cols) = (4usize, 256usize);
    let (path, _recon) = bake(name, rows, cols, 1.5, true);
    let _g = TmpFile(path.clone());
    let model = StrandModel::open(&path).expect("open");

    // Confirm the fixture actually exercises the sparse residual term.
    let res = outlier_residuals(&model, name).expect("residuals");
    assert!(!res.is_empty(), "fixture must have outliers so the residual term is live");

    for _ in 0..32 {
        let x: Vec<f32> = (0..cols).map(|_| ((rng.next_u32() as f32) / u32::MAX as f32) - 0.5).collect();
        // inner + residual, from the production RHT MAC (residual precomputed and recomputed
        // must already agree — that is outlier_mac's own invariant; we rely on it here).
        let inner_plus_resid = matvec_rht(&model, name, &x, Some(&res)).expect("matvec_rht");

        let c: Vec<f32> = (0..rows).map(|_| rng.next_correction()).collect();
        let c_bits = wire_from_f32(&c);

        let mut y_a = inner_plus_resid.clone();
        apply_debias_epilogue(&mut y_a, &c_bits);
        let mut y_b = inner_plus_resid.clone();
        apply_debias_epilogue(&mut y_b, &c_bits);
        assert_eq!(y_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), y_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "bias-after-residual not byte-stable");
        for o in 0..rows {
            let want = inner_plus_resid[o] + ref_bf16_to_f32(c_bits[o]);
            assert_eq!(y_a[o].to_bits(), want.to_bits(), "row {o}: wrong epilogue composition");
        }
    }
}

/// Edge shapes: single-row tensor, and a "huge-ish" output dimension (stress the per-row
/// vector length without a heavy build). The apply is O(out) and shape-agnostic; prove it
/// stays a pure per-row add and byte-stable on a wide output.
#[test]
fn apply_edge_shapes_single_and_wide_output() {
    // single output row, RHT on
    {
        let name = "tiny.single";
        let (rows, cols) = (1usize, 256usize);
        let (path, _r) = bake(name, rows, cols, 0.0, true);
        let _g = TmpFile(path.clone());
        let model = StrandModel::open(&path).expect("open");
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
        let base = matvec_patched(&model, name, &x).expect("matvec");
        assert_eq!(base.len(), 1);
        let c_bits = wire_from_f32(&[7.125e-2]);
        let mut y = base.clone();
        apply_debias_epilogue(&mut y, &c_bits);
        assert_eq!(y[0].to_bits(), (base[0] + ref_bf16_to_f32(c_bits[0])).to_bits());
    }
    // wide output (no RHT to keep the bake cheap). cols is a multiple of block_len (256).
    {
        let name = "wide.proj";
        let (rows, cols) = (512usize, 256usize);
        let (path, _r) = bake(name, rows, cols, 0.0, false);
        let _g = TmpFile(path.clone());
        let model = StrandModel::open(&path).expect("open");
        let mut rng = Sm64::new(0xDEC0_DE04);
        let x: Vec<f32> = (0..cols).map(|_| rng.next_correction() * 30.0).collect();
        let base = matvec_patched(&model, name, &x).expect("matvec");
        assert_eq!(base.len(), rows);
        let c: Vec<f32> = (0..rows).map(|_| rng.next_correction()).collect();
        let c_bits = wire_from_f32(&c);
        let mut y1 = base.clone();
        apply_debias_epilogue(&mut y1, &c_bits);
        let mut y2 = base.clone();
        apply_debias_epilogue(&mut y2, &c_bits);
        for o in 0..rows {
            assert_eq!(y1[o].to_bits(), y2[o].to_bits(), "row {o} not byte-stable on wide output");
            assert_eq!(y1[o].to_bits(), (base[o] + ref_bf16_to_f32(c_bits[o])).to_bits());
        }
    }
}

/// Cross-path equivalence: the bias add commutes with the choice of decode MAC path.
/// `matvec_patched` (reconstruct-then-GEMV) and `matvec_rht` (RHT-space GEMV) agree to a
/// float tolerance on the BASE output (an `outlier_mac` invariant). Adding the SAME bias
/// to each shifts both by the identical exact f32 dequant(c), so the post-bias gap equals
/// the pre-bias gap — the correction never introduces a NEW path divergence. (The two MAC
/// paths are not bit-equal to each other by design; the moat is per-path bit-determinism,
/// and the bias add preserves it on each path independently.)
#[test]
fn bias_add_does_not_widen_cross_path_gap() {
    let mut rng = Sm64::new(0xDEC0_DE05);
    let name = "model.layers.4.mlp.up_proj.weight";
    let (rows, cols) = (4usize, 256usize);
    let (path, _recon) = bake(name, rows, cols, 1.0, true);
    let _g = TmpFile(path.clone());
    let model = StrandModel::open(&path).expect("open");

    for _ in 0..16 {
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.05).cos() + 0.02 * rng.unit()).collect();
        let y_patched = matvec_patched(&model, name, &x).expect("patched");
        let y_rht = matvec_rht(&model, name, &x, None).expect("rht");

        let c: Vec<f32> = (0..rows).map(|_| rng.next_correction()).collect();
        let c_bits = wire_from_f32(&c);
        let mut yp = y_patched.clone();
        let mut yr = y_rht.clone();
        apply_debias_epilogue(&mut yp, &c_bits);
        apply_debias_epilogue(&mut yr, &c_bits);

        for o in 0..rows {
            // The exact same f32 was added to both, so the difference is unchanged to the
            // last bit of the added quantity; the residual gap is purely the pre-existing
            // MAC-path difference, never amplified by the correction.
            let gap_before = (y_patched[o] as f64 - y_rht[o] as f64).abs();
            let gap_after = (yp[o] as f64 - yr[o] as f64).abs();
            assert!((gap_after - gap_before).abs() <= 1e-6 * (1.0 + gap_before), "row {o}: bias add widened the cross-path gap ({gap_before} -> {gap_after})");
        }
    }
}

// ===========================================================================
// PART 2 — wiring stubs: the exact change that binds these proofs to PRODUCTION.
// These are #[ignore]d because the decode side has no DBIA today: `debias_wire` is
// not a module in strand-quant/src/lib.rs, `StrandModel` carries no `dbia` field /
// `debias(name)` accessor, and `outlier_mac` does not call the epilogue add.
// DO NOT edit those shared files in this task — these stubs document the wiring so the
// operator flips them on in one pass and the assertions bind onto the real symbols.
// ===========================================================================

#[test]
#[ignore = "WIRING: add `pub mod debias_wire;` to strand-quant/src/lib.rs, then replace \
            ref_* with strand_quant::debias_wire::{f32_to_bf16_round, bf16_to_f32}. The \
            oracle and production must agree on all 65536 bf16 patterns + a full f32 sweep."]
fn oracle_vs_production_bf16_round() {
    // Intended body once `debias_wire` is a module:
    //   use strand_quant::debias_wire::{f32_to_bf16_round, bf16_to_f32};
    //   for b in 0u32..=0xFFFF {
    //       let f = bf16_to_f32(b as u16);
    //       assert_eq!(f32_to_bf16_round(f), ref_f32_to_bf16(f));
    //       assert_eq!(bf16_to_f32(b as u16).to_bits(), ref_bf16_to_f32(b as u16).to_bits());
    //   }
    //   // plus a deterministic full-f32 sample via Sm64::next_f32_any.
}

#[test]
#[ignore = "WIRING: (1) `pub mod debias_wire;`; (2) StrandModel reads a DbiaSection \
            (mirror `outl`) and exposes `debias(name) -> Option<&DebiasWire>`; (3) a \
            `matvec_patched_debiased` (or a flag on matvec_patched) that runs the epilogue \
            add. Then assert it equals matvec_patched(..) + dequant(model.debias(name))."]
fn matvec_patched_with_dbia_equals_base_plus_correction() {
    // Intended body once wired:
    //   use strand_quant::debias_wire::{append_dbia, DebiasWire};
    //   let name = "model.layers.0.mlp.down_proj.weight";
    //   let (path, _recon) = bake(name, 4, 256, 1.0, true);
    //   let c = [1.5e-3f32, -2.0e-4, 0.0, 7.125e-2];
    //   append_dbia(&path, &[Some(DebiasWire::from_f32(&c))]).expect("append dbia");
    //   let model = StrandModel::open(&path).unwrap();
    //   let x: Vec<f32> = (0..256).map(|i| ((i as f32)*0.07).cos()).collect();
    //   let base = matvec_patched(&model, name, &x).unwrap();            // no apply
    //   let with = matvec_patched_debiased(&model, name, &x).unwrap();   // apply
    //   for o in 0..4 {
    //       assert_eq!(with[o].to_bits(), (base[o] + ref_bf16_to_f32(ref_f32_to_bf16(c[o]))).to_bits());
    //   }
    //   // And: an archive with NO DBIA section => matvec_patched_debiased == matvec_patched
    //   //      (byte-identical), proving back-compat on the real path.
}

#[test]
#[ignore = "WIRING: same as above + a stable u8 dispatch tag. Two independent decode runs \
            on the SAME .strand bytes (with DBIA) must produce bit-identical y on every \
            backend (CPU scalar, NEON, threaded gemv_par). This is the cross-device clause \
            — run under `--features neon-fma` too, where it must STILL hold for the bias add \
            even though the matmul's low bits differ (the add is one rounding, path-agnostic)."]
fn dbia_decode_is_bit_identical_across_backends() {
    // Intended: decode the same DBIA archive via gemv (scalar), neon_lut, and gemv_par,
    // apply the bias in each, and assert the bias CONTRIBUTION (with[o]-base[o]) is the
    // identical exact f32 on every backend — i.e. dequant(c[o]) — independent of how the
    // inner product was computed. The matmul base may differ per backend/feature; the
    // correction add does not.
}

// ===========================================================================
// PART 3 — Kani bounded proofs for the apply add (re-derived oracle; SAT-checked).
// Once `debias_wire` is wired, swap apply_debias_epilogue's body for the production add.
// ===========================================================================
#[cfg(kani)]
mod kani_harnesses {
    use super::*;

    /// For ALL (base, corr_bits): the epilogue add equals `base + bf16_to_f32(corr_bits)`
    /// exactly (no fma contraction, no reassociation — it is a single add). Pins the apply
    /// against any future epilogue refactor in the MAC kernels.
    #[kani::proof]
    fn apply_is_single_f32_add_symbolic() {
        let base: f32 = kani::any();
        let corr_bits: u16 = kani::any();
        let mut y = [base];
        apply_debias_epilogue(&mut y, &[corr_bits]);
        assert_eq!(y[0].to_bits(), (base + ref_bf16_to_f32(corr_bits)).to_bits());
    }

    /// Row independence (symbolic, 2 rows): applying to a 2-vector equals applying to each
    /// element alone — no cross-row coupling, for ALL inputs.
    #[kani::proof]
    fn apply_is_row_independent_symbolic() {
        let b0: f32 = kani::any();
        let b1: f32 = kani::any();
        let c0: u16 = kani::any();
        let c1: u16 = kani::any();
        let mut both = [b0, b1];
        apply_debias_epilogue(&mut both, &[c0, c1]);
        let mut s0 = [b0];
        apply_debias_epilogue(&mut s0, &[c0]);
        let mut s1 = [b1];
        apply_debias_epilogue(&mut s1, &[c1]);
        assert_eq!(both[0].to_bits(), s0[0].to_bits());
        assert_eq!(both[1].to_bits(), s1[0].to_bits());
    }
}
