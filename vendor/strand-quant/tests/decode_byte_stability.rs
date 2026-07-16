//! Cross-platform byte-stability proofs for the frozen-Q12 LUT decode arithmetic.
//!
//! THE MOAT: STRAND decode is bit-identical on *every* device. The load-bearing
//! integer step is the reconstruction
//!
//! ```text
//!     recon = (eff_scale_q(scale_q, sub_code) * lut[state]) >> 16
//! ```
//!
//! i.e. `(scale_q * q) >> 16` after the `>> 6` sub-scale fold. `crates/strand-quant/
//! src/proofs.rs` already proves, exhaustively and via Kani, that each stage *equals
//! its i128 oracle* (`reconstruct_q == (s*q)>>16`, `eff_scale_q == (s*mult)>>6`, …).
//!
//! What `proofs.rs` does NOT yet pin — and what this file adds — are the four
//! properties that actually make the *bytes* identical on a different CPU/OS/endian
//! than the one that froze the tables:
//!
//!   1. `chain_golden_*` — a FROZEN GOLDEN BYTE VECTOR. We FNV-1a-hash the
//!      little-endian bytes of the full `recon(eff_scale_q(scale,code), q)` chain
//!      over a fixed scale grid × all 64 sub-scale codes × every entry of the
//!      frozen quantile LUTs. The hash is a hardcoded constant computed off-box.
//!      Any platform that computes the integer math differently — or any future
//!      edit that perturbs it — changes the hash and fails. This is the literal
//!      "byte-stability across platforms" contract for this surface.
//!
//!   2. `recon_is_floor_div_*` — the deepest portability hazard of `>> 16` is its
//!      meaning on a NEGATIVE product. Rust *defines* `>>` on signed integers as an
//!      arithmetic (sign-replicating) shift, but the decode's device-independence
//!      rests on that shift being exactly floor-division by 65536. We prove
//!      `(s*q) >> 16 == (s*q).div_euclid(65536)` for negatives (Kani over all i32 s,
//!      plus an enumerated boundary sweep). A platform whose `>>` rounded toward zero
//!      would diverge here; this nails it shut.
//!
//!   3. `chain_stays_in_i32_*` — the composed per-element pipeline
//!      `recon(eff_scale_q(scale, code), q)` never overflows i64 and never leaves
//!      i32, for the real frozen-LUT |q| ≤ 6·4096 range. `proofs.rs` proves the
//!      stages in isolation; this proves they COMPOSE without surprise (Kani +
//!      enumeration), so the byte width that gets serialized is always 4.
//!
//!   4. `decode_path_is_float_free` — a structural guard: the decode hot path
//!      (`decode.rs`) must contain no `f32`/`f64`. A stray float op is exactly how
//!      a "deterministic" integer decode silently becomes platform-dependent
//!      (x87 80-bit vs SSE, FMA contraction, fast-math). This keeps the no-float
//!      invariant from regressing.
//!
//! Run (fast, no heavy build):
//!   cargo test -p strand-quant --test decode_byte_stability
//! Full L=4..14 golden sweep (~52M cases, release):
//!   cargo test -p strand-quant --release --test decode_byte_stability -- --ignored
//!
//! Bounded proofs (the `#[cfg(kani)]` module at the bottom): NOTE that `cargo kani`
//! only scans the *lib* target — it does NOT discover harnesses living in an
//! integration-test file (verified: `cargo kani list` enumerates only the four
//! `src/proofs.rs` harnesses). The two harnesses below are therefore provided as a
//! drop-in patch: paste them into `src/proofs.rs`'s `kani_harnesses` module (which is
//! already `#[cfg(any(test, kani))]`-gated) to run them with
//!   cargo kani -p strand-quant --harness recon_is_floor_div_symbolic
//! They compile and lower to CBMC cleanly as-is; the enumerated tests in THIS file
//! (`recon_is_floor_div_boundary_sweep`, `chain_stays_in_i32_boundary_sweep`) cover
//! the same properties over the full bounded domain that matters and run today.

use strand_quant::codebook::quantile_lut;
use strand_quant::decode::{eff_scale_q, reconstruct_q, SCALE_SHIFT};
use strand_quant::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};

/// The Q12 quantile clamp: every frozen LUT entry satisfies |q| <= 6 * 2^12.
const Q_CLAMP: i64 = 6 * (1 << 12);

/// FNV-1a over the little-endian bytes of i32 values. Endianness of the *host* is
/// irrelevant: `to_le_bytes` is defined to emit little-endian regardless of the CPU,
/// so this hash is itself a cross-platform quantity. If the decode arithmetic ever
/// produced a different i32 on some target, the hash would move.
struct Fnv(u64);
impl Fnv {
    #[inline]
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    #[inline]
    fn eat_i32(&mut self, v: i32) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// The fixed cross-platform scale grid. Deterministic, explicit, and identical in the
/// off-box generator that produced the golden constants below. Mixes small magnitudes,
/// powers of two ± the >>16 / >>6 rounding boundaries, the i32 extremes, and a couple of
/// alternating-bit patterns (0x5555.., 0x3333..) that stress sign extension.
const SCALES: [i32; 25] = [
    0,
    1,
    -1,
    2,
    -2,
    4096,
    -4096,
    1 << 16,
    -(1 << 16),
    32768,
    -32768,
    65537,
    -65537,
    1 << 20,
    -(1 << 20),
    1 << 30,
    -(1 << 30),
    i32::MAX,
    i32::MIN,
    i32::MAX - 1,
    -(i32::MAX),
    1_431_655_765, // 0x55555555
    -1_431_655_765,
    858_993_459, // 0x33333333
    -858_993_459,
];

/// Hash the full `recon(eff_scale_q(scale, code), q)` chain over the frozen quantile
/// LUTs for `l_bits in l_lo..=l_hi`, all 64 sub-scale codes, and every `SCALES` entry.
/// Returns `(hash, case_count)`. Uses the crate's REAL `eff_scale_q` / `reconstruct_q`
/// and the REAL frozen `quantile_lut`, so the golden constant pins production code.
fn chain_hash(l_lo: u32, l_hi: u32) -> (u64, u64) {
    let mut h = Fnv::new();
    let mut count = 0u64;
    for l in l_lo..=l_hi {
        let qlut = quantile_lut(l);
        for code in 0u8..64 {
            for &scale in &SCALES {
                let es = eff_scale_q(scale, code);
                for &q in qlut {
                    h.eat_i32(reconstruct_q(es, q));
                    count += 1;
                }
            }
        }
    }
    (h.0, count)
}

#[test]
fn chain_golden_fast_l4_to_l7() {
    // Small Ls (quantile lens 16/32/64/128) — runs in well under a second.
    let (hash, count) = chain_hash(4, 7);
    assert_eq!(count, 384_000, "case-count drift (grid changed)");
    assert_eq!(
        hash, 0x4b72_f341_92bc_3e11,
        "frozen-LUT decode arithmetic produced different bytes than the golden vector \
         (L=4..7). Either the integer math changed, the frozen quantile LUTs drifted, \
         or this platform computes (scale*q)>>16 differently — the determinism MOAT is broken."
    );
}

#[test]
#[ignore = "heavy: ~52M chain evaluations across all 11 frozen Ls; run with --release -- --ignored"]
fn chain_golden_full_l4_to_l14() {
    let (hash, count) = chain_hash(FROZEN_MIN_L, FROZEN_MAX_L);
    assert_eq!(count, 52_403_200, "case-count drift (grid or frozen-L range changed)");
    assert_eq!(hash, 0xc3ad_262e_5604_336e, "frozen-LUT decode arithmetic drift over the full L=4..14 sweep");
}

/// Six hand-computed `(scale, code, q) -> recon` triples. These are human-auditable
/// anchors for the golden hash: if `chain_golden_*` fails, these localize whether the
/// break is in `eff_scale_q` (the `>>6` fold) or `reconstruct_q` (the `>>16` step), and
/// they include both the i32 extremes and a negative-rounding case (recon = +24, not 0).
#[test]
fn chain_spot_values_are_exact() {
    let qlut4 = quantile_lut(4); // len 16, symmetric: [-7630, …, +7630]
    let q_lo = qlut4[0];
    let q_hi = qlut4[15];

    // (scale, code, q, expected eff_scale_q, expected recon)
    let cases: [(i32, u8, i32, i32, i32); 6] = [
        (1 << 16, 0, q_lo, 1024, -120),
        (1 << 16, 63, q_hi, 1 << 16, 7630),
        (i32::MIN, 0, q_lo, -33_554_432, 3_906_560),
        (i32::MIN, 63, q_hi, i32::MIN, -250_019_840),
        (i32::MAX, 63, qlut4[8], i32::MAX, 10_518_527),
        (-4096, 7, qlut4[3], -512, 24), // floor(-512 * -3180 / 65536) — positive
    ];
    for (scale, code, q, want_es, want_recon) in cases {
        let es = eff_scale_q(scale, code);
        assert_eq!(es, want_es, "eff_scale_q({scale},{code})");
        assert_eq!(reconstruct_q(es, q), want_recon, "reconstruct_q({es},{q})");
    }
}

// ---------------------------------------------------------------------------
// Property: `(s*q) >> 16` is exactly floor-division by 65536 (the device-independent
// definition of the shift), for both signs. This is the property that distinguishes a
// portable arithmetic shift from an implementation-defined / round-toward-zero one.
// ---------------------------------------------------------------------------

/// Enumerated boundary sweep of the floor-div identity. Quantiles span every frozen LUT
/// entry; scales span the explicit grid plus the immediate neighbourhood of every 65536
/// multiple boundary (where >> vs truncation would disagree for negatives).
#[test]
fn recon_is_floor_div_boundary_sweep() {
    // quantiles: every distinct frozen LUT value, plus the ± clamp extremes.
    let mut qs: Vec<i32> = Vec::new();
    for l in FROZEN_MIN_L..=FROZEN_MAX_L {
        qs.extend_from_slice(quantile_lut(l));
    }
    qs.push(Q_CLAMP as i32);
    qs.push(-(Q_CLAMP as i32));
    qs.sort_unstable();
    qs.dedup();

    // scales: the grid, plus integers straddling 65536-multiple boundaries so that the
    // signed product lands just above / on / just below an exact multiple of 2^16.
    let mut scales: Vec<i32> = SCALES.to_vec();
    for m in -4i64..=4 {
        let base = m * 65536;
        for d in -2i64..=2 {
            let x = base + d;
            if (i32::MIN as i64..=i32::MAX as i64).contains(&x) {
                scales.push(x as i32);
            }
        }
    }
    scales.sort_unstable();
    scales.dedup();

    let mut checked = 0u64;
    for &s in &scales {
        for &q in &qs {
            let prod = s as i64 * q as i64; // proven non-overflowing below + in proofs.rs
            let shifted = reconstruct_q(s, q) as i64;
            // The contract: arithmetic >> 16 == floor(prod / 65536). div_euclid on a
            // positive divisor IS the floor. This is the only place the sign of the
            // shift matters, so assert it directly rather than restating `>>`.
            assert_eq!(shifted, prod.div_euclid(1 << SCALE_SHIFT), "shift is not floor-division at s={s} q={q} (prod={prod})");
            // And confirm it genuinely floors *downward* for negatives, i.e. it is NOT
            // truncation toward zero whenever there is a nonzero remainder.
            if prod < 0 && prod % (1 << SCALE_SHIFT) != 0 {
                let trunc = prod / (1 << SCALE_SHIFT); // Rust `/` truncates toward zero
                assert_eq!(shifted, trunc - 1, "negative shift did not round down at s={s} q={q}");
            }
            checked += 1;
        }
    }
    eprintln!("floor-div sweep: {} scales x {} quantiles = {checked} pairs", scales.len(), qs.len());
}

// ---------------------------------------------------------------------------
// Property: the composed per-element decode pipeline stays inside i64 (no product
// overflow) and lands inside i32 (so exactly 4 bytes are serialized), for the real
// frozen-LUT |q| range. Enumerated here; Kani-proven over symbolic inputs below.
// ---------------------------------------------------------------------------

#[test]
fn chain_stays_in_i32_boundary_sweep() {
    let mut qs: Vec<i32> = Vec::new();
    for l in FROZEN_MIN_L..=FROZEN_MAX_L {
        qs.extend_from_slice(quantile_lut(l));
    }
    qs.sort_unstable();
    qs.dedup();

    let mut checked = 0u64;
    for &scale in &SCALES {
        for code in 0u8..64 {
            let es = eff_scale_q(scale, code);
            for &q in &qs {
                // i64 product never overflows: |es| <= 2^31, |q| <= Q_CLAMP, product
                // magnitude <= 2^31 * 24576 < 2^46 << 2^63.
                let prod = es as i64 * q as i64;
                assert!(prod.unsigned_abs() < (1u64 << 63), "chain product overflows i64 at scale={scale} code={code} q={q}");
                // recon fits i32: |prod>>16| <= 2^31 * 24576 / 65536 < 2^31.
                let recon = prod >> SCALE_SHIFT;
                assert!((i32::MIN as i64..=i32::MAX as i64).contains(&recon), "chain recon leaves i32 at scale={scale} code={code} q={q}: {recon}");
                // and the impl agrees with that i64 oracle.
                assert_eq!(reconstruct_q(es, q) as i64, recon);
                checked += 1;
            }
        }
    }
    eprintln!("chain-range sweep: {checked} (scale,code,quantile) triples");
}

// ---------------------------------------------------------------------------
// Structural guard: the decode hot path is float-free. A single f32/f64 op is the
// classic way a "deterministic" integer decoder silently becomes platform-dependent.
// ---------------------------------------------------------------------------

#[test]
fn decode_path_is_float_free() {
    // Resolve decode.rs relative to this test file (CARGO_MANIFEST_DIR = the crate root).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/decode.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    // The decode module is permitted ONE bounded float region: the `decode_tensor`
    // f32 *wrapper* + its `Q12_TO_F32` constant, which exist solely to hand integer
    // Q12 output to float-expecting callers. That wrapper is exact (proven by
    // `f32_wrapper_is_exact_q12` in exhaustive.rs) and downstream of the bit-identical
    // integer decode. Everything ABOVE it — the actual decode arithmetic — must be
    // float-free. We scan only that region.
    let cutoff = src.find("const Q12_TO_F32").expect("decode.rs no longer has the Q12_TO_F32 float-wrapper marker — re-audit the float boundary");
    let integer_region = &src[..cutoff];

    // Strip line comments so doc/comment prose mentioning floats doesn't trip us.
    let mut offenders: Vec<(usize, String)> = Vec::new();
    for (i, raw) in integer_region.lines().enumerate() {
        let code = match raw.find("//") {
            Some(p) => &raw[..p],
            None => raw,
        };
        // Token-ish scan for float types and float literals in the integer decode region.
        // `f32`/`f64` type mentions, and decimal-point numeric literals (e.g. `1.0`).
        let has_float_ty = code.contains("f32") || code.contains("f64");
        let has_float_lit = contains_float_literal(code);
        if has_float_ty || has_float_lit {
            offenders.push((i + 1, raw.trim().to_string()));
        }
    }
    assert!(
        offenders.is_empty(),
        "float appears in the integer decode region of decode.rs (above Q12_TO_F32) — \
         this can break bit-identical decode across platforms:\n{}",
        offenders.iter().map(|(ln, s)| format!("  line {ln}: {s}")).collect::<Vec<_>>().join("\n")
    );
}

/// True if `s` contains a Rust float literal of the form `<digit>.<digit>` (a decimal
/// point flanked by ASCII digits). Deliberately conservative: it ignores `..` ranges,
/// struct field access (`x.0` is matched, but field-`.0` on integers does not occur in
/// the integer decode region; if it ever did we would *want* to look).
fn contains_float_literal(s: &str) -> bool {
    let b = s.as_bytes();
    for i in 1..b.len().saturating_sub(1) {
        if b[i] == b'.' && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Bounded proofs (Kani). See the module header: `cargo kani` does not scan
// integration-test files, so these run only after being pasted into
// `src/proofs.rs`'s `kani_harnesses` module. They are kept here, next to their
// enumerated counterparts, as a reviewed drop-in and as living documentation of the
// symbolic property each enumeration approximates. Each closes a portability gap over
// ALL i32 inputs rather than a finite grid.
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_harnesses {
    use super::*;

    /// `(s*q) >> 16` is floor-division by 65536 for ALL i32 `s` and every clamped `q`.
    /// This is strictly stronger than `proofs::reconstruct_q_total_symbolic` (which
    /// restates `>>`): it proves the shift IS the floor, i.e. the device-independent
    /// rounding, distinguishing it from a round-toward-zero shift.
    #[kani::proof]
    fn recon_is_floor_div_symbolic() {
        let s: i32 = kani::any();
        let q: i32 = kani::any();
        kani::assume((-(Q_CLAMP as i32)..=Q_CLAMP as i32).contains(&q));
        let prod = s as i64 * q as i64;
        assert_eq!(reconstruct_q(s, q) as i64, prod.div_euclid(1 << SCALE_SHIFT));
    }

    /// The composed pipeline `recon(eff_scale_q(scale, code), q)` for any i32 `scale`,
    /// any sub-scale `code`, and any clamped frozen-LUT `q`:
    ///   * the inner `eff_scale_q` product fits i64,
    ///   * the outer `recon` product fits i64,
    ///   * the result lands in i32 (so exactly 4 bytes serialize),
    ///   * the whole thing equals its i128 oracle.
    /// Closes the "stages compose without overflow" gap left by the isolated proofs.
    #[kani::proof]
    fn chain_composes_in_i32_symbolic() {
        let scale: i32 = kani::any();
        let code: u8 = kani::any();
        let q: i32 = kani::any();
        kani::assume((-(Q_CLAMP as i32)..=Q_CLAMP as i32).contains(&q));

        let es = eff_scale_q(scale, code);
        // inner product fits i64 (|scale| <= 2^31, mult <= 64):
        let inner = scale as i128 * ((code as i128 & 0x3F) + 1);
        assert!(inner.unsigned_abs() < 1u128 << 63);
        assert_eq!(es as i128, inner >> 6);

        // outer product fits i64 (|es| <= 2^31, |q| <= Q_CLAMP):
        let outer = es as i128 * q as i128;
        assert!(outer.unsigned_abs() < 1u128 << 63);

        let recon = reconstruct_q(es, q);
        // lands in i32 and equals the i128 oracle:
        let oracle = outer >> SCALE_SHIFT;
        assert!(oracle >= i32::MIN as i128 && oracle <= i32::MAX as i128);
        assert_eq!(recon as i128, oracle);
    }
}
