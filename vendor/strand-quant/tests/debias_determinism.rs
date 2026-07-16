//! Determinism hardening for the de-bias DBIA surface (de-bias per-row bias add +
//! serialize/parse round-trip). The MOAT: STRAND decode is bit-identical on every
//! device. The de-bias correction is the *only* float op the decode side performs —
//! a single per-output-row `y[o] += bf16_to_f32(c_bits[o])` add — so its determinism
//! is load-bearing for the cross-device guarantee.
//!
//! This file is the sibling of `exhaustive.rs`: it re-derives the spec independently
//! and proves the surface against it (the `ref_decode` pattern), exhaustively over a
//! bounded domain, with a Kani bounded proof. It is dependency-free (no proptest) to
//! match the crate's `[dev-dependencies] = none` policy; property-style coverage uses
//! a deterministic splitmix64 PRNG so every run hits the identical cases.
//!
//! # Two surfaces, two reachabilities (honest scope)
//!
//! 1. **Per-row bias compute** lives in `strand_quant::debias` (WIRED: `pub mod debias`
//!    in lib.rs). `debias_tensor` produces the per-row additive correction
//!    `c_i = -mu_bar * S_i`. This is REAL reachable code — the tests below that touch
//!    `debias_tensor` / `matvec` / `output_error` exercise the production functions and
//!    prove the strongest determinism property of the per-row add: **row-independence /
//!    order-invariance** (row `i`'s correction depends ONLY on `w[i]`, never on the row
//!    count, neighbouring rows, or row order), plus bit-exact reproducibility, plus the
//!    apply-add being a single order-free f32 add.
//!
//! 2. **bf16 quantize + wire serialize/parse** lives in `debias_wire.rs`, which (as of
//!    this writing) is NOT declared as a module in lib.rs — the surface is "(planned)".
//!    An integration test is a separate crate and `debias_wire.rs` does `use crate::
//!    format::{..}`, so it cannot be `#[path]`-included here without resolving against a
//!    `format` module the test crate does not have. So component B re-derives the bf16
//!    round / dequant / wire codec from the documented spec (debias_wire.rs module docs)
//!    as an independent oracle and proves the format's byte-stability EXHAUSTIVELY over
//!    all 65,536 bf16 patterns and a structured f32 sweep. This is a golden-vector /
//!    spec proof: it pins the exact bytes the production code must emit. The GAP it does
//!    not close — until `debias_wire` is wired into lib.rs — is binding the oracle to the
//!    production `f32_to_bf16_round` / `append_dbia` / `read_dbia_bytes` symbols. The
//!    `#[ignore]`d `oracle_vs_production_*` stubs document exactly the one-line change
//!    (`pub mod debias_wire;`) that flips this proof onto the real code.
//!
//! Run: `cargo test -p strand-quant --test debias_determinism`
//! Kani: `cargo kani -p strand-quant --harness debias_determinism::...` (after wiring)

use strand_quant::debias::{debias_tensor, output_error};

// ---------------------------------------------------------------------------
// Independent reference oracle for the DBIA wire (mirrors exhaustive.rs::ref_decode:
// re-implement the spec by hand, then assert the surface equals it).
// Spec source: crates/strand-quant/src/debias_wire.rs module docs §"Float order".
// ---------------------------------------------------------------------------

/// f32 -> bf16, round-to-nearest, ties-to-even. NaN/Inf keep their top 16 bits.
/// Hand-derived from IEEE-754 (NOT a copy of the production one-liner): the dropped
/// 16 mantissa bits decide the rounding; a tie (dropped bits == 0x8000) breaks toward
/// an even kept-mantissa LSB.
fn ref_f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp_all_ones = (bits & 0x7f80_0000) == 0x7f80_0000;
    if exp_all_ones {
        // Inf or NaN: truncate the top half (matches production f32_to_bf16_round). The
        // result class is decided purely by the KEPT 7 mantissa bits: nonzero -> NaN,
        // zero -> Inf. A NaN whose payload lives only in the dropped low 16 bits thus
        // collapses to Inf — deterministic and device-uniform; Rust's quiet f32::NAN
        // (0x7fc0_0000, kept bit set) always stays NaN. `non_finite_preserves_top_half`
        // pins both branches.
        return (bits >> 16) as u16;
    }
    let kept = bits >> 16; // sign(1) | exp(8) | top-7-mantissa
    let dropped = bits & 0xffff; // the 16 bits we are rounding away
    let half = 0x8000u32;
    let round_up = dropped > half || (dropped == half && (kept & 1) == 1);
    (kept + round_up as u32) as u16
}

/// bf16 -> f32: place the 16 stored bits in the top half, zero the low 16.
fn ref_bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// One tensor's correction on the wire: bf16 per output row, row 0 first.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RefWire {
    c_bits: Vec<u16>,
}
impl RefWire {
    fn from_f32(c: &[f32]) -> Self {
        RefWire { c_bits: c.iter().map(|&v| ref_f32_to_bf16(v)).collect() }
    }
}

const PAGE: usize = 4096;
const DBIA_MAGIC: &[u8; 4] = b"DBIA";
const DBIA_VERSION: u32 = 1;
const DBIA_HEADER_BYTES: usize = 32;
const DBIA_RECORD_FIXED_BYTES: usize = 8;

/// Serialise a DBIA section body (header + records), independent of the file appender.
/// `None` => zero-length record (absent). Returns the section bytes (no trailer/pad).
fn ref_section_bytes(wires: &[Option<RefWire>], out_features: &[usize]) -> Vec<u8> {
    assert_eq!(wires.len(), out_features.len());
    let mut o = Vec::new();
    o.extend_from_slice(DBIA_MAGIC);
    o.extend_from_slice(&DBIA_VERSION.to_le_bytes());
    o.extend_from_slice(&(wires.len() as u32).to_le_bytes());
    o.extend_from_slice(&0u32.to_le_bytes()); // flags
    o.extend_from_slice(&[0u8; 16]); // reserved
    assert_eq!(o.len(), DBIA_HEADER_BYTES);
    for (w, &out) in wires.iter().zip(out_features.iter()) {
        match w {
            None => {
                o.extend_from_slice(&0u32.to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());
            }
            Some(w) => {
                assert_eq!(w.c_bits.len(), out, "record/out_features mismatch");
                assert!(!w.c_bits.is_empty(), "Some(empty) is ambiguous; use None");
                o.extend_from_slice(&(w.c_bits.len() as u32).to_le_bytes());
                o.extend_from_slice(&0u32.to_le_bytes());
                for &b in &w.c_bits {
                    o.extend_from_slice(&b.to_le_bytes());
                }
            }
        }
    }
    o
}

/// Parse a DBIA section body back into per-tensor wires. Validates every field the
/// production parser validates (magic, version, n_tensors, flags, reserved, lengths,
/// no trailing bytes). Returns Err on any inconsistency.
fn ref_parse_section(s: &[u8], out_features: &[usize]) -> Result<Vec<Option<RefWire>>, String> {
    if s.len() < DBIA_HEADER_BYTES {
        return Err("section shorter than header".into());
    }
    if &s[0..4] != &DBIA_MAGIC[..] {
        return Err("bad magic".into());
    }
    if u32::from_le_bytes(s[4..8].try_into().unwrap()) != DBIA_VERSION {
        return Err("bad version".into());
    }
    let n = u32::from_le_bytes(s[8..12].try_into().unwrap()) as usize;
    if n != out_features.len() {
        return Err("n_tensors mismatch".into());
    }
    if u32::from_le_bytes(s[12..16].try_into().unwrap()) != 0 {
        return Err("flags set".into());
    }
    if s[16..32].iter().any(|&b| b != 0) {
        return Err("reserved not zero".into());
    }
    let mut p = DBIA_HEADER_BYTES;
    let mut take = |n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).filter(|&e| e <= s.len()).ok_or("truncated")?;
        let sl = &s[p..end];
        p = end;
        Ok(sl)
    };
    let mut out = Vec::with_capacity(n);
    for &of in out_features {
        let len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        if u32::from_le_bytes(take(4)?.try_into().unwrap()) != 0 {
            return Err("record reserved not zero".into());
        }
        if len == 0 {
            out.push(None);
            continue;
        }
        if len != of {
            return Err("len != out_features".into());
        }
        let raw = take(len * 2)?;
        let c_bits = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        out.push(Some(RefWire { c_bits }));
    }
    if p != s.len() {
        return Err("trailing bytes".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64) — property-style coverage without a proptest dep.
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
    /// An f32 drawn from the FULL bit space (so subnormals, NaN, Inf, -0 all appear).
    fn next_f32_any(&mut self) -> f32 {
        f32::from_bits(self.next_u32())
    }
    /// A "correction-shaped" f32: small magnitude, both signs, plus the occasional 0.
    fn next_correction(&mut self) -> f32 {
        let r = self.next_u64();
        match r & 0x7 {
            0 => 0.0,
            1 => -0.0,
            _ => {
                let mant = ((r >> 8) & 0xFFFF) as f32 / 65535.0; // [0,1]
                let scale = [1e-1f32, 1e-2, 1e-3, 1e-4, 1e-5][((r >> 3) % 5) as usize];
                let sign = if r & 0x80 != 0 { -1.0 } else { 1.0 };
                sign * mant * scale
            }
        }
    }
}

// ===========================================================================
// COMPONENT A — per-row bias-add determinism (REAL `strand_quant::debias` code)
// ===========================================================================

/// THE core determinism property for the per-row add: the correction for output row
/// `i` is a pure function of that row's weights alone. Computing a tensor of `R` rows
/// gives, in slot `i`, the bit-identical f32 you get by debiasing row `i` as a
/// 1-row tensor. Therefore row order, row count, and neighbouring rows can never
/// change any single channel's correction — the add is safe to apply per-channel on
/// any device, in any order, with no cross-channel coupling.
#[test]
fn per_row_correction_is_row_local_bit_exact() {
    let mut rng = Sm64::new(0xDB1A_0001);
    let mut checked = 0u64;
    for &in_f in &[1usize, 2, 7, 8, 31, 32, 33, 64, 257] {
        for trial in 0..40 {
            let rows = 1 + (rng.next_u32() as usize % 9); // 1..=9 rows
                                                          // Build a multi-row weight/recon pair.
            let mut w = vec![0.0f32; rows * in_f];
            let mut recon = vec![0.0f32; rows * in_f];
            for k in 0..rows * in_f {
                w[k] = rng.next_f32_any();
                // recon = a *different* finite-ish value so the rowsum delta is real
                recon[k] = rng.next_f32_any();
            }
            let mu_bar = if trial % 3 == 0 { 0.0 } else { rng.next_correction() * 100.0 };

            let full = debias_tensor(&w, &recon, in_f, mu_bar, 16);
            assert_eq!(full.bias_correction.len(), rows);
            assert_eq!(full.rowsum_bias.len(), rows);

            for i in 0..rows {
                let row_w = &w[i * in_f..(i + 1) * in_f];
                let row_r = &recon[i * in_f..(i + 1) * in_f];
                let solo = debias_tensor(row_w, row_r, in_f, mu_bar, 16);
                // bit-exact equality of the f32 correction and rowsum bias.
                // NaN-safe comparison via raw bits (NaN != NaN under ==, but the bit
                // pattern must still be reproduced identically for determinism).
                assert_eq!(full.bias_correction[i].to_bits(), solo.bias_correction[0].to_bits(), "row {i} correction not row-local (in_f={in_f} rows={rows})");
                assert_eq!(full.rowsum_bias[i].to_bits(), solo.rowsum_bias[0].to_bits(), "row {i} rowsum not row-local (in_f={in_f} rows={rows})");
            }
            checked += 1;
        }
    }
    eprintln!("per-row row-locality: {checked} multi-row tensors decomposed");
}

/// Permuting the rows of the input permutes the output identically and bit-exactly —
/// the function commutes with any row permutation (corollary of row-locality, but
/// proven directly because the apply side reorders channels freely across devices).
#[test]
fn correction_commutes_with_row_permutation() {
    let mut rng = Sm64::new(0xDB1A_0002);
    let in_f = 17usize;
    for _ in 0..200 {
        let rows = 2 + (rng.next_u32() as usize % 8);
        let mut w = vec![0.0f32; rows * in_f];
        let mut recon = vec![0.0f32; rows * in_f];
        for k in 0..rows * in_f {
            w[k] = rng.next_correction();
            recon[k] = rng.next_correction();
        }
        let mu_bar = rng.next_correction() * 50.0;
        let base = debias_tensor(&w, &recon, in_f, mu_bar, 16);

        // build a permutation via Fisher-Yates with the same PRNG stream
        let mut perm: Vec<usize> = (0..rows).collect();
        for i in (1..rows).rev() {
            let j = rng.next_u32() as usize % (i + 1);
            perm.swap(i, j);
        }
        let mut wp = vec![0.0f32; rows * in_f];
        let mut rp = vec![0.0f32; rows * in_f];
        for (newi, &oldi) in perm.iter().enumerate() {
            wp[newi * in_f..(newi + 1) * in_f].copy_from_slice(&w[oldi * in_f..(oldi + 1) * in_f]);
            rp[newi * in_f..(newi + 1) * in_f].copy_from_slice(&recon[oldi * in_f..(oldi + 1) * in_f]);
        }
        let permed = debias_tensor(&wp, &rp, in_f, mu_bar, 16);
        for (newi, &oldi) in perm.iter().enumerate() {
            assert_eq!(permed.bias_correction[newi].to_bits(), base.bias_correction[oldi].to_bits(), "permutation changed a correction bit pattern");
        }
    }
}

/// Bit-exact reproducibility: the same (w, recon, mu_bar) yields the identical f32
/// bit patterns on every call. (Guards against any hidden nondeterminism — threading,
/// hashing, iteration order — sneaking into the encode-side compute.)
#[test]
fn correction_is_run_to_run_bit_stable() {
    let mut rng = Sm64::new(0xDB1A_0003);
    let in_f = 53usize;
    for _ in 0..150 {
        let rows = 1 + (rng.next_u32() as usize % 6);
        let mut w = vec![0.0f32; rows * in_f];
        let mut recon = vec![0.0f32; rows * in_f];
        for k in 0..rows * in_f {
            w[k] = rng.next_f32_any();
            recon[k] = rng.next_f32_any();
        }
        let mu = rng.next_correction() * 10.0;
        let a = debias_tensor(&w, &recon, in_f, mu, 16);
        let b = debias_tensor(&w, &recon, in_f, mu, 16);
        let abits: Vec<u32> = a.bias_correction.iter().map(|v| v.to_bits()).collect();
        let bbits: Vec<u32> = b.bias_correction.iter().map(|v| v.to_bits()).collect();
        assert_eq!(abits, bbits);
        assert_eq!(a.bpw_cost.to_bits(), b.bpw_cost.to_bits());
    }
}

/// The decode-side apply IS a single per-row f32 add: `y[o] += c[o]`. Prove it is
/// order-free (computing rows in forward vs reverse order gives identical bits) and
/// reproducible. This is the reference for the MAC epilogue add (debias_wire.rs docs).
fn apply_epilogue(y: &mut [f32], c: &[f32]) {
    for (yo, &co) in y.iter_mut().zip(c.iter()) {
        *yo += co;
    }
}

#[test]
fn apply_add_is_order_free_and_bit_stable() {
    let mut rng = Sm64::new(0xDB1A_0004);
    for _ in 0..3000 {
        let n = 1 + (rng.next_u32() as usize % 16);
        let y0: Vec<f32> = (0..n).map(|_| rng.next_f32_any()).collect();
        let c: Vec<f32> = (0..n).map(|_| rng.next_correction()).collect();

        // forward
        let mut yf = y0.clone();
        apply_epilogue(&mut yf, &c);
        // reverse application order — each add is independent, so result must match bit-for-bit
        let mut yr = y0.clone();
        for o in (0..n).rev() {
            yr[o] += c[o];
        }
        for o in 0..n {
            assert_eq!(yf[o].to_bits(), yr[o].to_bits(), "per-row add depends on iteration order — would break cross-device parity");
            // and equals the documented single add
            assert_eq!(yf[o].to_bits(), (y0[o] + c[o]).to_bits());
        }
        // rerun reproducibility
        let mut yf2 = y0.clone();
        apply_epilogue(&mut yf2, &c);
        for o in 0..n {
            assert_eq!(yf[o].to_bits(), yf2[o].to_bits());
        }
    }
}

/// Exhaustive apply-add over a *structured corner domain* of f32 bit patterns
/// (signed zeros, subnormals, the smallest/largest normals, 1.0±ulp, infinities,
/// a quiet NaN). For every (base, correction) pair the single add must be
/// reproducible and equal to `f32 + f32` — no fma contraction, no double rounding.
#[test]
fn apply_add_corner_domain_is_pure_f32_add() {
    let corners: [u32; 16] = [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x0000_0001, // smallest +subnormal
        0x8000_0001, // smallest -subnormal
        0x007f_ffff, // largest subnormal
        0x0080_0000, // smallest +normal
        0x3f80_0000, // 1.0
        0x3f80_0001, // 1.0 + 1ulp
        0x3f7f_ffff, // 1.0 - 1ulp
        0xbf80_0000, // -1.0
        0x7f7f_ffff, // f32::MAX
        0xff7f_ffff, // -f32::MAX
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0000, // quiet NaN
        0x4048_f5c3, // ~3.14
    ];
    let mut checked = 0u64;
    for &a in &corners {
        for &b in &corners {
            let base = f32::from_bits(a);
            let corr = f32::from_bits(b);
            let r1 = base + corr;
            let r2 = {
                let mut y = [base];
                apply_epilogue(&mut y, &[corr]);
                y[0]
            };
            // Use raw bits so NaN compares identically (and to detect any -0/+0 drift).
            assert_eq!(r1.to_bits(), r2.to_bits(), "apply != f32 add at ({a:#x},{b:#x})");
            // determinism across a second evaluation
            assert_eq!((base + corr).to_bits(), r1.to_bits());
            checked += 1;
        }
    }
    assert_eq!(checked, 16 * 16);
    eprintln!("apply-add corner domain: {checked} (base,corr) pairs are pure f32 add");
}

/// End-to-end on REAL `debias_tensor` + `matvec`: on a constant activation `x = mu*1`
/// the corrected output error is the per-row identity `y_rec - y_ref + c_i ≈ 0`, and
/// is reproducible. This pins the *meaning* of the correction (eq. 2/3) while also
/// exercising the real `output_error` path used by gate-debias.
#[test]
fn constant_activation_identity_on_real_code() {
    let mut rng = Sm64::new(0xDB1A_0005);
    let in_f = 24usize;
    let out = 5usize;
    for _ in 0..100 {
        let w: Vec<f32> = (0..out * in_f).map(|_| rng.next_correction() * 4.0).collect();
        // a coarse "quantized" recon so the rowsum bias is genuinely nonzero
        let recon: Vec<f32> = w.iter().map(|&v| (v * 6.0).round() / 6.0).collect();
        let mu_bar = 0.25f32 + rng.next_correction();
        let r = debias_tensor(&w, &recon, in_f, mu_bar, 16);
        let x = vec![mu_bar; in_f];
        let (_, rms_uncorr) = output_error(&w, &recon, in_f, std::slice::from_ref(&x), None);
        let (mean_corr, rms_corr) = output_error(&w, &recon, in_f, &[x], Some(&r.bias_correction));
        // corrected error vanishes (float-tolerance — this is the encode-side f64 math,
        // not the bit-exact decode add).
        assert!(rms_corr <= rms_uncorr + 1e-6, "correction increased error");
        assert!(rms_corr < 1e-3, "corrected rms should be ~0 on constant x: {rms_corr}");
        assert!(mean_corr.abs() < 1e-3);
    }
}

// ===========================================================================
// COMPONENT B — bf16 round + wire serialize/parse (independent oracle; pins bytes)
// ===========================================================================

/// EXHAUSTIVE over all 65,536 bf16 bit patterns: bf16 is a fixed point of the round
/// (idempotence), i.e. rounding an already-bf16 value never moves it. This is the
/// property the production `f32_bf16_round_trip_is_top16_and_ties_even` test only
/// checks on 7 values; here it is total.
#[test]
fn bf16_is_total_fixed_point_of_round() {
    for b in 0u32..=0xFFFF {
        let b = b as u16;
        let f = ref_bf16_to_f32(b);
        let r = ref_f32_to_bf16(f);
        if (b & 0x7f80) == 0x7f80 && (b & 0x007f) != 0 {
            // NaN class: stays a NaN of the SAME bit pattern (top half preserved verbatim).
            assert_eq!(r, b, "bf16 NaN pattern {b:#06x} not preserved");
            assert!(ref_bf16_to_f32(r).is_nan());
        } else {
            assert_eq!(r, b, "bf16 {b:#06x} is not a fixed point (dequant->round drifted)");
        }
    }
    eprintln!("bf16 round fixed-point: all 65536 patterns idempotent");
}

/// bf16_to_f32 is an exact left inverse of (round ∘ already-bf16): the f32 you get
/// back has its low 16 mantissa bits zero and its top half equals the stored u16.
#[test]
fn dequant_places_bits_in_top_half_exhaustive() {
    for b in 0u32..=0xFFFF {
        let b = b as u16;
        let f = ref_bf16_to_f32(b);
        assert_eq!(f.to_bits() & 0xFFFF, 0, "dequant left low 16 bits set for {b:#06x}");
        assert_eq!((f.to_bits() >> 16) as u16, b, "dequant top half != stored bits {b:#06x}");
    }
}

/// Round-to-nearest-EVEN tie discipline, proven on the exact tie inputs across the
/// full bf16 exponent/mantissa grid: for every bf16 value `v`, the f32 exactly halfway
/// to the next bf16 (`v.bits<<16 | 0x8000`) rounds to whichever of {v, v+1ulp} has an
/// even low mantissa bit. This is the byte-stability crux (a wrong tie rule = wrong
/// bytes = cross-device divergence).
#[test]
fn ties_round_to_even_across_bf16_grid() {
    let mut checked = 0u64;
    for b in 0u32..=0xFFFE {
        let b16 = b as u16;
        // skip non-finite (exp all ones): the round truncates there, no tie semantics.
        if (b16 & 0x7f80) == 0x7f80 {
            continue;
        }
        let mid = ((b as u32) << 16) | 0x8000; // exactly half a bf16 ulp above `b`
        let got = ref_f32_to_bf16(f32::from_bits(mid));
        let expect = if (b16 & 1) == 0 { b16 } else { b16.wrapping_add(1) };
        assert_eq!(got, expect, "tie at bf16 {b16:#06x} did not round to even");
        checked += 1;
    }
    eprintln!("ties-to-even: {checked} exact-midpoint inputs verified");
}

/// Just-below / just-above the midpoint always round down / up respectively, over the
/// whole finite bf16 grid (completes the rounding-direction proof around every tie).
#[test]
fn near_tie_rounds_toward_nearer_neighbor() {
    for b in 0u32..=0xFFFE {
        let b16 = b as u16;
        if (b16 & 0x7f80) == 0x7f80 {
            continue;
        }
        let base = (b as u32) << 16;
        // midpoint - 1 ulp(f32) rounds down to b
        assert_eq!(ref_f32_to_bf16(f32::from_bits(base | 0x7fff)), b16);
        // midpoint + 1 ulp(f32) rounds up to b+1
        assert_eq!(ref_f32_to_bf16(f32::from_bits(base | 0x8001)), b16.wrapping_add(1));
    }
}

/// NaN/Inf class: every non-finite f32 (exp all ones) keeps its top 16 bits under the
/// round, and a NaN stays a NaN. Swept over a structured set of exponent-all-ones
/// patterns (every sign, the two infinities, and mantissa patterns that fall in both
/// the kept-7 and dropped-16 fields).
#[test]
fn non_finite_preserves_top_half() {
    let mut count = 0u64;
    for sign in [0u32, 0x8000_0000] {
        // mantissa patterns: probe the kept-7 (bits 16..23) and dropped-16 (bits 0..16)
        for m in [0u32, 1, 0x7f, 0x80, 0xffff, 0x1_0000, 0x40_0000, 0x7f_ffff, 0x12_3456] {
            let bits = sign | 0x7f80_0000 | m;
            let f = f32::from_bits(bits);
            let got = ref_f32_to_bf16(f);
            // Top half is ALWAYS preserved verbatim (deterministic truncation).
            assert_eq!(got, (bits >> 16) as u16, "non-finite top half changed");
            // The result class depends ONLY on the kept 7 mantissa bits (m's top half):
            //   kept-7 nonzero  -> stays NaN
            //   kept-7 zero     -> the truncation deterministically yields Inf, EVEN if the
            //                      original f32 was a NaN whose payload lived entirely in the
            //                      dropped low 16 bits. This is a documented, device-uniform
            //                      collapse (production f32_to_bf16_round does the same `>>16`);
            //                      Rust's quiet f32::NAN (0x7fc0_0000) has a set kept bit so it
            //                      always stays NaN — only synthetic low-payload NaNs collapse.
            let kept_mantissa = (bits >> 16) & 0x007f;
            if kept_mantissa != 0 {
                assert!(f.is_nan() && ref_bf16_to_f32(got).is_nan(), "NaN with kept payload must stay NaN");
            } else if m != 0 {
                // a NaN whose payload is only in the dropped bits -> deterministically Inf
                assert!(f.is_nan(), "input should be NaN here");
                assert!(ref_bf16_to_f32(got).is_infinite(), "low-payload NaN must collapse to Inf");
            } else {
                assert!(f.is_infinite() && ref_bf16_to_f32(got).is_infinite());
            }
            count += 1;
        }
    }
    eprintln!("non-finite preservation: {count} exp-all-ones patterns");
}

/// Full wire serialize -> parse round-trip, property-style over random tensor counts,
/// shapes, present/absent records, and correction magnitudes. The parsed wires must
/// equal the input wires EXACTLY (bf16 payload byte-for-byte), and re-serialising the
/// parse must reproduce the identical section bytes (encode/decode are mutual inverses
/// on the canonical form).
#[test]
fn wire_round_trip_is_byte_exact_property() {
    let mut rng = Sm64::new(0xDB1A_0006);
    let mut checked = 0u64;
    for _ in 0..2000 {
        let n_tensors = 1 + (rng.next_u32() as usize % 6);
        let mut out_features = Vec::with_capacity(n_tensors);
        let mut wires: Vec<Option<RefWire>> = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let out = 1 + (rng.next_u32() as usize % 64);
            out_features.push(out);
            // ~1/4 of records absent
            if rng.next_u32() % 4 == 0 {
                wires.push(None);
            } else {
                let c: Vec<f32> = (0..out).map(|_| rng.next_correction()).collect();
                wires.push(Some(RefWire::from_f32(&c)));
            }
        }

        let bytes = ref_section_bytes(&wires, &out_features);
        // determinism of serialization itself
        assert_eq!(bytes, ref_section_bytes(&wires, &out_features));

        let parsed = ref_parse_section(&bytes, &out_features).expect("valid section parses");
        assert_eq!(parsed, wires, "wire round-trip changed the corrections");

        // re-serialise the parse: byte-identical (canonical fixed point)
        let bytes2 = ref_section_bytes(&parsed, &out_features);
        assert_eq!(bytes, bytes2, "serialize∘parse is not the identity on bytes");

        // header invariants the production parser also enforces
        assert_eq!(&bytes[0..4], &DBIA_MAGIC[..]);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, n_tensors);
        // payload size accounting matches the documented record layout
        let expect_len: usize = DBIA_HEADER_BYTES + wires.iter().map(|w| DBIA_RECORD_FIXED_BYTES + w.as_ref().map_or(0, |x| x.c_bits.len() * 2)).sum::<usize>();
        assert_eq!(bytes.len(), expect_len, "section length drifted from layout spec");
        checked += 1;
    }
    eprintln!("wire round-trip: {checked} random sections byte-exact");
}

/// The parser rejects every corruption the spec names: bad magic, wrong version,
/// wrong n_tensors, set flags, nonzero reserved (header & record), wrong record
/// length, truncation, and trailing bytes. (Rejection is itself a determinism
/// property — a malformed section must fail closed identically everywhere, never
/// silently decode to device-dependent garbage.)
#[test]
fn parser_rejects_all_corruptions() {
    let out_features = vec![4usize, 3];
    let wires = vec![Some(RefWire::from_f32(&[1.5e-3, -2.0e-4, 0.0, 7.125e-2])), None];
    let good = ref_section_bytes(&wires, &out_features);
    assert!(ref_parse_section(&good, &out_features).is_ok());

    let mutate = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = good.clone();
        f(&mut b);
        b
    };

    // bad magic
    assert!(ref_parse_section(&mutate(&|b| b[0] ^= 0xFF), &out_features).is_err());
    // wrong version
    assert!(ref_parse_section(&mutate(&|b| b[4] ^= 0xFF), &out_features).is_err());
    // wrong n_tensors field
    assert!(ref_parse_section(&mutate(&|b| b[8] = 9), &out_features).is_err());
    // flags set
    assert!(ref_parse_section(&mutate(&|b| b[12] = 1), &out_features).is_err());
    // header reserved nonzero
    assert!(ref_parse_section(&mutate(&|b| b[16] = 1), &out_features).is_err());
    // record reserved nonzero (record 0 starts at header end +4)
    assert!(ref_parse_section(&mutate(&|b| b[DBIA_HEADER_BYTES + 4] = 1), &out_features).is_err());
    // record length mismatch (claim 5 rows for an out=4 tensor)
    assert!(ref_parse_section(&mutate(&|b| b[DBIA_HEADER_BYTES] = 5), &out_features).is_err());
    // truncated payload (drop the last byte)
    assert!(ref_parse_section(&good[..good.len() - 1], &out_features).is_err());
    // trailing byte
    let mut extra = good.clone();
    extra.push(0);
    assert!(ref_parse_section(&extra, &out_features).is_err());
    // n_tensors arg disagreement (parser told a different out_features count)
    assert!(ref_parse_section(&good, &[4, 3, 1]).is_err());
}

/// GOLDEN VECTOR — a frozen byte image of a known section. If the wire layout ever
/// shifts (field order, endianness, header size, bf16 rounding of these exact
/// constants), this exact-bytes assertion fails. The constants are the same ones the
/// production `dbia_round_trip_and_v2_reader_compat` test uses, so this is a
/// cross-test consistency anchor.
#[test]
fn golden_section_bytes_are_frozen() {
    let out_features = vec![4usize, 3];
    let wires = vec![Some(RefWire::from_f32(&[1.5e-3, -2.0e-4, 0.0, 7.125e-2])), None];
    let bytes = ref_section_bytes(&wires, &out_features);

    // Header: magic, version=1, n_tensors=2, flags=0, 16 zero reserved.
    let mut want = Vec::new();
    want.extend_from_slice(b"DBIA");
    want.extend_from_slice(&1u32.to_le_bytes());
    want.extend_from_slice(&2u32.to_le_bytes());
    want.extend_from_slice(&0u32.to_le_bytes());
    want.extend_from_slice(&[0u8; 16]);
    // Record 0: len=4, reserved=0, then the four bf16 (round of the constants).
    want.extend_from_slice(&4u32.to_le_bytes());
    want.extend_from_slice(&0u32.to_le_bytes());
    for &v in &[1.5e-3f32, -2.0e-4, 0.0, 7.125e-2] {
        want.extend_from_slice(&ref_f32_to_bf16(v).to_le_bytes());
    }
    // Record 1: absent (len=0, reserved=0).
    want.extend_from_slice(&0u32.to_le_bytes());
    want.extend_from_slice(&0u32.to_le_bytes());

    assert_eq!(bytes, want, "DBIA wire layout changed — update the golden or the code");

    // Pin the actual bf16 codes so a silent change in the round rule is caught even if
    // someone "fixes" both sides in lockstep.
    assert_eq!(ref_f32_to_bf16(1.5e-3), 0x3ac5);
    assert_eq!(ref_f32_to_bf16(-2.0e-4), 0xb952);
    assert_eq!(ref_f32_to_bf16(0.0), 0x0000);
    assert_eq!(ref_f32_to_bf16(7.125e-2), 0x3d92);
}

/// Page-alignment arithmetic the appender relies on (the trailer must land so the
/// whole file stays a multiple of PAGE — see debias_wire.rs `append_dbia`). Pure
/// integer, exhaustively sound over a representative sweep.
#[test]
fn page_align_arithmetic_is_sound() {
    let page_align = |x: usize| (x + PAGE - 1) & !(PAGE - 1);
    for base in 0..(3 * PAGE) {
        let a = page_align(base);
        assert!(a >= base && a < base + PAGE);
        assert_eq!(a % PAGE, 0);
        if base % PAGE == 0 {
            assert_eq!(a, base);
        }
    }
}

// ===========================================================================
// Oracle <-> production binding stubs (the documented GAP).
// These compile but are #[ignore]d because `debias_wire` is not yet a module.
// Wiring it in (`pub mod debias_wire;` in lib.rs) + un-ignoring + swapping the
// oracle calls for the real `strand_quant::debias_wire::*` flips every Component-B
// proof above onto the production symbols.
// ===========================================================================

#[test]
#[ignore = "debias_wire not wired into lib.rs yet; see module GAP note. Un-ignore after \
            adding `pub mod debias_wire;` and binding to the real f32_to_bf16_round."]
fn oracle_vs_production_bf16_round() {
    // Intended body once reachable:
    //   for b in 0u32..=0xFFFF { let f = ref_bf16_to_f32(b as u16);
    //       assert_eq!(strand_quant::debias_wire::f32_to_bf16_round(f), ref_f32_to_bf16(f)); }
    // plus a full-f32 sweep sample. Left as a stub to avoid editing shared lib.rs here.
}

#[test]
#[ignore = "debias_wire not wired into lib.rs yet; see module GAP note."]
fn oracle_vs_production_wire_round_trip() {
    // Intended: build a v2 archive, append_dbia(real), read_dbia_bytes(real),
    // assert the parsed c_bits equal RefWire::from_f32(..).c_bits for the same inputs,
    // and that two independent appends are byte-identical (production
    // `append_is_byte_deterministic` already shows the latter for one fixed input).
}

// ---------------------------------------------------------------------------
// Kani bounded proofs (mirror proofs.rs::kani_harnesses). Re-derived oracle only,
// since the production debias_wire symbols are not reachable; once wired, swap to them.
// ---------------------------------------------------------------------------
#[cfg(kani)]
mod kani_harnesses {
    use super::*;

    /// For ALL u16, bf16 is a fixed point of the round (symbolic, unbounded over the
    /// 16-bit domain). Stronger than the 65,536-case enumeration: SAT-checked.
    #[kani::proof]
    fn bf16_round_is_total_fixed_point_symbolic() {
        let b: u16 = kani::any();
        let f = ref_bf16_to_f32(b);
        let r = ref_f32_to_bf16(f);
        assert_eq!(r, b);
    }

    /// For ALL f32, the round result's low 16 bits are zero when re-expanded, i.e.
    /// dequant∘round lands on a bf16 grid point (closure of the encode map).
    #[kani::proof]
    fn round_then_dequant_is_on_bf16_grid_symbolic() {
        let x: f32 = kani::any();
        let b = ref_f32_to_bf16(x);
        let f = ref_bf16_to_f32(b);
        assert_eq!(f.to_bits() & 0xFFFF, 0);
        assert_eq!((f.to_bits() >> 16) as u16, b);
    }

    /// The apply add is a single deterministic f32 add for ALL (base, corr): proven
    /// equal to `base + corr` symbolically (no contraction/reassociation possible on
    /// one add, but this pins it against any future epilogue refactor).
    #[kani::proof]
    fn apply_add_equals_f32_add_symbolic() {
        let base: f32 = kani::any();
        let corr: f32 = kani::any();
        let mut y = [base];
        apply_epilogue(&mut y, &[corr]);
        assert_eq!(y[0].to_bits(), (base + corr).to_bits());
    }
}
