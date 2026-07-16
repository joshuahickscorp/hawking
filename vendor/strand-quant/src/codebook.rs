pub const QUANTILE_SHIFT: u32 = 12;

const QUANTILE_CLAMP_Q12: i32 = 6 * (1 << QUANTILE_SHIFT);

/// Acklam coefficients — the **single source of truth** shared by the f64
/// generator ([`inv_norm_cdf`]) and the integer-exact computed codebook
/// ([`acklam_central_q12`]). Lifted to module scope so both paths quote the
/// identical numbers (the computed path must reproduce the frozen table the f64
/// path bakes, byte-for-byte).
const ACKLAM_A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.38357751867269e+02, -3.066479806614716e+01, 2.506628277459239e+00];
const ACKLAM_B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
const ACKLAM_C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
const ACKLAM_D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
/// Lower break-point of Acklam's central region (`p in [P_LOW, 1-P_LOW]`).
const ACKLAM_P_LOW: f64 = 0.02425;

fn inv_norm_cdf(p: f64) -> f64 {
    // Coefficients for Acklam's algorithm (shared module consts).
    const A: [f64; 6] = ACKLAM_A;
    const B: [f64; 5] = ACKLAM_B;
    const C: [f64; 6] = ACKLAM_C;
    const D: [f64; 4] = ACKLAM_D;
    // Break-points of the central region.
    const P_LOW: f64 = ACKLAM_P_LOW;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

pub fn build_quantile_lut_f64(l_bits: u32) -> Vec<i32> {
    let n = 1usize << l_bits;
    let mut lut = Vec::with_capacity(n);
    let denom = n as f64;
    for s in 0..n {
        let p = (s as f64 + 0.5) / denom;
        let q = inv_norm_cdf(p);

        let scaled = (q * (1u32 << QUANTILE_SHIFT) as f64).round();
        let v = if scaled.is_finite() {
            scaled as i64
        } else if scaled > 0.0 {
            QUANTILE_CLAMP_Q12 as i64
        } else {
            -(QUANTILE_CLAMP_Q12 as i64)
        };
        let v = v.clamp(-(QUANTILE_CLAMP_Q12 as i64), QUANTILE_CLAMP_Q12 as i64);
        lut.push(v as i32);
    }
    lut
}

#[inline]
pub fn hash_state(s: usize, l_bits: u32) -> usize {
    let mask = (1usize << l_bits) - 1;
    let r = (l_bits / 2).max(1) as usize;
    let mut h = s & mask;
    h = (h ^ (h >> r)) & mask;
    h = h.wrapping_mul(0x2545_F491_4F6C_DD1D) & mask;
    h = (h ^ (h >> r)) & mask;
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask;
    h & mask
}

/// State-indexed Q12 codebook value obtained by computing the deterministic
/// state-to-rank permutation and gathering from the monotone quantile table.
/// This is byte-identical to `codebook_lut(l_bits)[state]`, but exposes the
/// useful middle point between a full state-indexed LUT and full Acklam compute.
#[inline]
pub fn qcb_hashed(state: usize, l_bits: u32) -> i32 {
    quantile_lut(l_bits)[hash_state(state, l_bits)]
}

/// Materialise the codebook through [`qcb_hashed`]. Encoder and vector paths use
/// this compatibility form; scalar decoders call `qcb_hashed` directly.
pub fn codebook_lut_hashed(l_bits: u32) -> Vec<i32> {
    use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
    let l = l_bits.clamp(FROZEN_MIN_L, FROZEN_MAX_L);
    (0..1usize << l).map(|s| qcb_hashed(s, l)).collect()
}

pub fn build_codebook_lut_f64(l_bits: u32) -> Vec<i32> {
    let ranks = build_quantile_lut_f64(l_bits);
    let n = 1usize << l_bits;
    (0..n).map(|s| ranks[hash_state(s, l_bits)]).collect()
}

// ─── Computed codebook (Variant A: bit-exact integer Acklam, zero re-encode) ──
//
// The frozen tables above are produced offline by the f64 generators. The
// *computed* path below reproduces the very same integers in **pure integer
// arithmetic** — no float anywhere — so a decoder can synthesise any codebook
// entry from `(state, L)` with a few ALU ops instead of a `2^L`-entry gather.
// Bit-exactness vs the frozen table is asserted as a hard contract test
// (`computed_codebook_matches_frozen` + `computed_codebook_golden_hash`).
//
// Strategy (measured exact, see those tests):
//   * Central region (`p in [P_LOW, 1-P_LOW]`, ~97.6% of ranks): evaluate
//     Acklam's central rational in i128 fixed point at `FB` fractional bits.
//     At FB = 40 this reproduces the f64 `round(.*4096)` result for **every**
//     central rank, all L in 4..=14 (0 mismatches over 31_164 entries).
//   * Tails (`p < P_LOW` or `p > P_HIGH`, the extreme ~2.4%): the f64 tail
//     branch uses `ln`/`sqrt`, which are the hard transcendentals to reproduce
//     bit-exactly. We instead read the tail values straight from the already
//     golden-hashed frozen quantile table (`quantile_lut`). The tails are
//     antisymmetric (`q[r] == -q[n-1-r]`, asserted), so only the left prefix is
//     needed and the right suffix is its negation. This is the "hybrid" the
//     design calls for: the central 97.6% is gather-free pure compute, the
//     tail 2.4% is a tiny constant-bounded lookup.

/// Fixed-point fractional bits for the integer Acklam central evaluation.
///
/// 40 bits leaves comfortable headroom: the largest intermediate is a degree-5
/// Horner product `coef(~2^8 in Q40) * r2(<2^40)` which stays well inside i128,
/// and 40 fractional bits resolve the rational to far below one Q12 ULP — enough
/// to match the f64 rounding at every central rank (proven by test).
const ACKLAM_FB: u32 = 40;

/// Acklam central numerator coefficients pre-scaled to Q(`ACKLAM_FB`) as exact
/// integers: `ACKLAM_A_Q[i] == round(ACKLAM_A[i] * 2^ACKLAM_FB)`. Baked so the
/// **runtime central path touches no float at all** (the determinism contract).
/// The equality with the f64 source is asserted by `baked_acklam_coeffs_exact`.
const ACKLAM_A_Q: [i128; 6] = [-43647126486026, 242932804329501, -303386605671354, 152125956971009, -33716302037132, 2756066937579];
/// Acklam central denominator coefficients pre-scaled to Q(`ACKLAM_FB`); see
/// [`ACKLAM_A_Q`]. The implicit trailing `B[5] = 1.0` is `1 << ACKLAM_FB`.
const ACKLAM_B_Q: [i128; 5] = [-59897104064522, 177665506509332, -171192838788807, 73448819171239, -14602263792188];

/// Per-`L` count of **left-tail** ranks (`p < P_LOW`), `L = 4..=14` at index
/// `L - 4`. Captured offline from the f64 `p < P_LOW` decision (the tail region
/// is a contiguous prefix; by antisymmetry the right tail has the same count).
/// Ranks `[0, t)` and `[n-t, n)` use the frozen tail lookup; the rest use the
/// integer central rational. The bit-exact contract test catches any drift.
const TAIL_LEFT_LEN: [usize; 11] = [0, 1, 2, 3, 6, 12, 25, 50, 99, 199, 397];

/// Rounded i128 division, half away from zero — mirrors f64 `.round()` on the
/// final rational so the integer path lands on the same Q12 integer.
#[inline]
fn div_round_i128(n: i128, d: i128) -> i128 {
    debug_assert!(d != 0);
    if (n >= 0) == (d > 0) {
        (n + d / 2) / d
    } else {
        (n - d / 2) / d
    }
}

/// Evaluate Acklam's **central** rational for rank `r` of an `L`-bit table in
/// i128 fixed point and return the Q12 quantile (clamped to `±6 sigma`).
///
/// `p = (r + 0.5) / 2^L = (2r + 1) / 2^(L+1)`. Caller must only pass *central*
/// ranks (`TAIL_LEFT_LEN[L-4] <= r < 2^L - TAIL_LEFT_LEN[L-4]`); tail ranks are
/// handled by [`tail_quantile`].
///
/// **Zero floating point**: every operand is an integer (the coefficients are
/// the baked `ACKLAM_A_Q`/`ACKLAM_B_Q` Q40 consts), so the result is byte-
/// identical on every CPU/GPU/WASM target — the STRAND decode contract.
fn acklam_central_q12(r: usize, l_bits: u32) -> i32 {
    // q = p - 1/2 = ((2r+1) - 2^L) / 2^(L+1), exact in Q(FB) because FB >= L+1.
    let l_plus_1 = l_bits + 1;
    let q_num: i128 = (2 * r as i128 + 1) - (1i128 << l_bits); // (2r+1) - 2^L
    let q_q: i128 = (q_num << ACKLAM_FB) >> l_plus_1; // q in Q(FB), exact (low bits zero)
    let r2_q: i128 = (q_q * q_q) >> ACKLAM_FB; // q^2 in Q(FB)

    let one_q: i128 = 1i128 << ACKLAM_FB;

    // num = ((((A0*r2 + A1)*r2 + A2)*r2 + A3)*r2 + A4)*r2 + A5, then *q.
    let mut num = ACKLAM_A_Q[0];
    for &a in &ACKLAM_A_Q[1..] {
        num = ((num * r2_q) >> ACKLAM_FB) + a;
    }
    num = (num * q_q) >> ACKLAM_FB;

    // den = ((((B0*r2 + B1)*r2 + B2)*r2 + B3)*r2 + B4)*r2 + 1.0.
    let mut den = ACKLAM_B_Q[0];
    for &b in &ACKLAM_B_Q[1..] {
        den = ((den * r2_q) >> ACKLAM_FB) + b;
    }
    den = (den * r2_q) >> ACKLAM_FB;
    den += one_q; // implicit B[5] = 1.0

    // result = num/den (the Q(FB) scales cancel); want round(result * 2^12).
    let q12 = div_round_i128(num << QUANTILE_SHIFT, den);
    (q12 as i32).clamp(-QUANTILE_CLAMP_Q12, QUANTILE_CLAMP_Q12)
}

/// Tail-region quantile (Q12) for rank `r` of an `L`-bit table, read from the
/// golden-hashed frozen quantile table. Right-tail ranks reuse the left prefix
/// by antisymmetry (`q[r] == -q[n-1-r]`). Caller guarantees `r` is a tail rank.
/// **Integer only** (a constant-bounded lookup, ≤ `TAIL_LEFT_LEN` per side).
#[inline]
fn tail_quantile(r: usize, l_bits: u32) -> i32 {
    let n = 1usize << l_bits;
    let t = TAIL_LEFT_LEN[(l_bits - crate::lut_tables::FROZEN_MIN_L) as usize];
    let q = quantile_lut(l_bits);
    if r < t {
        q[r] // left tail, verbatim
    } else {
        // right tail: q[r] = -q[n-1-r], and n-1-r < t here.
        -q[n - 1 - r]
    }
}

/// Quantile (Q12) at **rank** `r` for an `L`-bit table, computed bit-exactly:
/// integer Acklam central rational for the central 97.6%, frozen tail lookup for
/// the extreme 2.4%. Reproduces `build_quantile_lut_f64(L)[r]` exactly (proven
/// by the contract test). **Integer only.**
///
/// `l_bits` is clamped into the frozen range `[FROZEN_MIN_L, FROZEN_MAX_L]` to
/// match [`quantile_lut`].
#[inline]
pub fn quantile_q12_computed(r: usize, l_bits: u32) -> i32 {
    use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
    let l = l_bits.clamp(FROZEN_MIN_L, FROZEN_MAX_L);
    let n = 1usize << l;
    let t = TAIL_LEFT_LEN[(l - FROZEN_MIN_L) as usize];
    if r < t || r >= n - t {
        tail_quantile(r, l)
    } else {
        acklam_central_q12(r, l)
    }
}

/// State-indexed codebook value (Q12) for trellis state `state` at width `L`,
/// computed bit-exactly with **no LUT gather**: `quantile_q12_computed(hash_state(state, L), L)`.
///
/// This is the drop-in replacement for `codebook_lut(L)[state]`: Variant A
/// reproduces the frozen `CODEBOOK_LUTS` entry exactly (contract-tested), so
/// switching a decode/encode site from the gather to this compute is byte-for-
/// byte identical. **Integer only** — safe on the determinism-pinned path.
///
/// `l_bits` is clamped into the frozen range to match [`codebook_lut`].
#[inline]
pub fn qcb(state: usize, l_bits: u32) -> i32 {
    quantile_q12_computed(hash_state(state, l_bits), l_bits)
}

/// Materialise the full state-indexed codebook for width `L` by computing every
/// entry via [`qcb`] (no frozen-table borrow). Byte-identical to `codebook_lut(L)`
/// under Variant A. This is the array the encoder consumes when configured for
/// the computed codebook, keeping all encode sites on the existing `&[i32]`
/// interface while the decode hot loop can instead call [`qcb`] inline.
pub fn codebook_lut_computed(l_bits: u32) -> Vec<i32> {
    use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
    let l = l_bits.clamp(FROZEN_MIN_L, FROZEN_MAX_L);
    let n = 1usize << l;
    (0..n).map(|s| qcb(s, l)).collect()
}

/// Left-tail prefix (Q12) for width `L`: the `TAIL_LEFT_LEN[L-4]` quantile values
/// at ranks `[0, t)`. These are the **only** stored values a no-LUT decoder needs
/// — the central 97.6% are computed via the integer Acklam path ([`qcb`]) and the
/// right tail is the negation of this prefix by antisymmetry
/// (`q[r] == -q[n-1-r]`, contract-tested). Exposed for the GPU computed-codebook
/// kernel, which carries this tiny constant-bounded table instead of the full
/// `2^L`-entry LUT (≤ 25 entries at L12 vs 4096). `l_bits` is clamped to the
/// frozen range to match [`quantile_lut`].
pub fn tail_left_prefix_q12(l_bits: u32) -> Vec<i32> {
    use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
    let l = l_bits.clamp(FROZEN_MIN_L, FROZEN_MAX_L);
    let t = TAIL_LEFT_LEN[(l - FROZEN_MIN_L) as usize];
    quantile_lut(l)[..t].to_vec()
}

pub fn quantile_lut(l_bits: u32) -> &'static [i32] {
    use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L, QUANTILE_LUTS};
    let l = l_bits.clamp(FROZEN_MIN_L, FROZEN_MAX_L);
    QUANTILE_LUTS[(l - FROZEN_MIN_L) as usize]
}

pub fn codebook_lut(l_bits: u32) -> &'static [i32] {
    use crate::lut_tables::{CODEBOOK_LUTS, FROZEN_MAX_L, FROZEN_MIN_L};
    let l = l_bits.clamp(FROZEN_MIN_L, FROZEN_MAX_L);
    CODEBOOK_LUTS[(l - FROZEN_MIN_L) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_is_monotone_and_symmetric() {
        let lut = build_quantile_lut_f64(10);

        for w in lut.windows(2) {
            assert!(w[1] >= w[0], "quantile LUT not monotone: {} then {}", w[0], w[1]);
        }

        let n = lut.len();
        for s in 0..n {
            let a = lut[s];
            let b = lut[n - 1 - s];
            assert!((a + b).abs() <= 2, "not antisymmetric at s={s}: {a} vs {b}");
        }

        assert!(lut[n / 2].abs() <= (1 << QUANTILE_SHIFT) / 256);
    }

    #[test]
    fn lut_generation_is_deterministic() {
        assert_eq!(build_quantile_lut_f64(8), build_quantile_lut_f64(8));
        assert_eq!(build_quantile_lut_f64(12), build_quantile_lut_f64(12));
        assert_eq!(build_codebook_lut_f64(9), build_codebook_lut_f64(9));
    }

    #[test]
    fn frozen_tables_match_f64_generators() {
        use crate::lut_tables::{CODEBOOK_LUTS, FROZEN_MAX_L, FROZEN_MIN_L, QUANTILE_LUTS};
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            let i = (l - FROZEN_MIN_L) as usize;
            assert_eq!(QUANTILE_LUTS[i], build_quantile_lut_f64(l).as_slice(), "quantile L={l}");
            assert_eq!(CODEBOOK_LUTS[i], build_codebook_lut_f64(l).as_slice(), "codebook L={l}");
        }

        assert_eq!(quantile_lut(10), build_quantile_lut_f64(10).as_slice());
        assert_eq!(codebook_lut(10), build_codebook_lut_f64(10).as_slice());
    }

    #[test]
    fn frozen_lut_golden_hash() {
        use crate::lut_tables::{CODEBOOK_LUTS, LUT_GOLDEN_HASH, QUANTILE_LUTS};
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in QUANTILE_LUTS.iter().chain(CODEBOOK_LUTS.iter()) {
            for &v in *t {
                for b in v.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        assert_eq!(h, LUT_GOLDEN_HASH, "frozen LUT golden hash mismatch (table drift)");
    }

    #[test]
    fn hash_is_a_bijection() {
        for l in [4u32, 8, 10] {
            let n = 1usize << l;
            let mut seen = vec![false; n];
            for s in 0..n {
                let h = hash_state(s, l);
                assert!(h < n, "hash out of range");
                assert!(!seen[h], "hash collision at L={l}: state {s} -> {h}");
                seen[h] = true;
            }
            assert!(seen.iter().all(|&b| b), "hash did not cover all ranks at L={l}");
        }
    }

    #[test]
    fn codebook_successors_spread_across_distribution() {
        let l = 10u32;
        let k = 3u32;
        let cb = codebook_lut(l);
        let mask = (1usize << l) - 1;
        let full_range = (cb.iter().copied().max().unwrap() - cb.iter().copied().min().unwrap()) as f64;
        let mut avg_spread = 0.0f64;
        let trials = 64usize;
        for s in 0..trials {
            let succ: Vec<i32> = (0..(1usize << k)).map(|i| cb[((s << k) | i) & mask]).collect();
            let spread = (succ.iter().copied().max().unwrap() - succ.iter().copied().min().unwrap()) as f64;
            avg_spread += spread;
        }
        avg_spread /= trials as f64;

        assert!(avg_spread > 0.25 * full_range, "successor spread {avg_spread:.0} too small vs full range {full_range:.0}",);
    }

    #[test]
    fn inv_norm_cdf_known_points() {
        assert!(inv_norm_cdf(0.5).abs() < 1e-9);
        assert!((inv_norm_cdf(0.975) - 1.959963985).abs() < 1e-4);
        assert!((inv_norm_cdf(0.8413447) - 1.0).abs() < 1e-3);
    }

    // ─── Computed codebook (Variant A) — HARD CONTRACT GATES ─────────────────

    #[test]
    fn baked_acklam_coeffs_exact() {
        // The baked Q40 integer coefficients must equal round(f64_coef * 2^FB)
        // exactly — proves the consts (which make the runtime path float-free)
        // were transcribed correctly from the f64 source of truth.
        let scale = (1u128 << ACKLAM_FB) as f64;
        for (i, &a) in ACKLAM_A.iter().enumerate() {
            assert_eq!(ACKLAM_A_Q[i], (a * scale).round() as i128, "A[{i}] baked coeff wrong");
        }
        for (i, &b) in ACKLAM_B.iter().enumerate() {
            assert_eq!(ACKLAM_B_Q[i], (b * scale).round() as i128, "B[{i}] baked coeff wrong");
        }
    }

    #[test]
    fn tail_is_antisymmetric_in_frozen_table() {
        // The computed path relies on `q[r] == -q[n-1-r]` to derive the right
        // tail from the stored left prefix. Prove it holds across every frozen
        // table for every tail rank.
        use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            let n = 1usize << l;
            let t = TAIL_LEFT_LEN[(l - FROZEN_MIN_L) as usize];
            let q = quantile_lut(l);
            for r in 0..t {
                assert_eq!(q[r], -q[n - 1 - r], "tail not antisymmetric L={l} r={r}");
            }
        }
    }

    #[test]
    fn tail_left_len_matches_p_low_boundary() {
        // The frozen TAIL_LEFT_LEN cutoffs must equal the f64 `p < P_LOW`
        // decision exactly (so the integer cutoff selects the identical central
        // set the f64 generator did). Re-derive from the f64 boundary and compare.
        use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            let n = 1usize << l;
            let mut left = 0usize;
            let mut right = 0usize;
            for r in 0..n {
                let p = (r as f64 + 0.5) / n as f64;
                if p < ACKLAM_P_LOW {
                    left += 1;
                }
                if p > 1.0 - ACKLAM_P_LOW {
                    right += 1;
                }
            }
            let t = TAIL_LEFT_LEN[(l - FROZEN_MIN_L) as usize];
            assert_eq!(t, left, "TAIL_LEFT_LEN wrong (left) at L={l}");
            assert_eq!(t, right, "tail asymmetric count at L={l}: left={left} right={right}");
        }
    }

    #[test]
    fn computed_quantile_matches_frozen_quantile() {
        // quantile_q12_computed(r,L) must equal build_quantile_lut_f64(L)[r] for
        // every rank, every frozen L. (The central via integer Acklam, the tail
        // via the frozen lookup.)
        use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L};
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            let frozen = quantile_lut(l);
            for (r, &want) in frozen.iter().enumerate() {
                assert_eq!(quantile_q12_computed(r, l), want, "computed quantile mismatch L={l} r={r}");
            }
        }
    }

    #[test]
    fn computed_codebook_matches_frozen() {
        // THE contract gate: qcb(s,L) == CODEBOOK_LUTS[L-4][s] for ALL s, all L
        // in [FROZEN_MIN_L, FROZEN_MAX_L]. Variant A reproduces the frozen
        // state-indexed codebook byte-for-byte — nothing downstream lands unless
        // this is green.
        use crate::lut_tables::{CODEBOOK_LUTS, FROZEN_MAX_L, FROZEN_MIN_L};
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            let frozen = CODEBOOK_LUTS[(l - FROZEN_MIN_L) as usize];
            for (s, &want) in frozen.iter().enumerate() {
                assert_eq!(qcb(s, l), want, "qcb mismatch L={l} state={s}");
            }
            // And the materialised array equals the frozen accessor slice.
            assert_eq!(codebook_lut_computed(l).as_slice(), codebook_lut(l));
        }
    }

    #[test]
    fn computed_codebook_golden_hash() {
        // FNV-1a over the *computed* quantile+codebook tables (same order the
        // freezer hashes) must equal the frozen LUT_GOLDEN_HASH. This proves the
        // pure-integer path regenerates the entire golden dataset bit-for-bit.
        use crate::lut_tables::{FROZEN_MAX_L, FROZEN_MIN_L, LUT_GOLDEN_HASH};
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |v: i32| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        // Quantile tables (computed), L low→high.
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            let n = 1usize << l;
            for r in 0..n {
                feed(quantile_q12_computed(r, l));
            }
        }
        // Codebook tables (computed), L low→high.
        for l in FROZEN_MIN_L..=FROZEN_MAX_L {
            for &v in &codebook_lut_computed(l) {
                feed(v);
            }
        }
        assert_eq!(h, LUT_GOLDEN_HASH, "computed-codebook golden hash mismatch");
    }

    #[test]
    fn computed_codebook_is_deterministic() {
        // Same (state, L) ⇒ identical value on repeated calls (no hidden state,
        // float, or ordering). The property the decode determinism rests on.
        for l in [4u32, 7, 10, 12, 14] {
            let n = 1usize << l;
            for s in (0..n).step_by((n / 64).max(1)) {
                assert_eq!(qcb(s, l), qcb(s, l));
            }
        }
    }
}
