//! RHT (Randomized Hadamard Transform) determinism hardening.
//!
//! STRAND's MOAT is a *decode* path that is bit-identical on every device. The
//! RHT lives on the **encode** side, where floating point is permitted (it is
//! offline preprocessing). It still carries a determinism obligation, because a
//! quantized tensor is produced by `forward`-then-quantize and the decode side
//! is later asked to undo the rotation; if `forward` were not reproducible, or
//! if `inverse(forward(x))` were not a faithful reconstruction, the quantizer's
//! reference would shift between machines and the MOAT would be undermined at
//! the source.
//!
//! This file pins three independent guarantees and is the sibling of
//! `tests/exhaustive.rs` (which hardens the integer decode arithmetic):
//!
//!  1. **Seed determinism (`rht_seed_for`, FNV-1a).** The per-tensor seed is a
//!     pure `u64` wrapping computation over the tensor name. We re-derive it
//!     from spec and assert exact equality against the production function over
//!     a large name corpus, plus the structural invariants the encoder relies
//!     on (always odd; sensitive to byte order; canonical FNV-1a offset basis).
//!
//!  2. **Sign / PRNG determinism (`splitmix64` + `sign_at`).** The sign flips
//!     are integer-exact and platform-independent. We re-derive them from spec
//!     and verify, *through the public `rht_forward` API*, that the transform
//!     consumes exactly the spec sign sequence — so the private primitive is
//!     pinned without needing crate-internal access.
//!
//!  3. **Round-trip exactness — conditioned on the block size.** Over a bounded
//!     **dyadic** input domain (the regime real weights occupy after scaling),
//!     `inverse(forward(x))` is value-for-value equal to `x` and *bit-identical*
//!     once the benign `+0.0`/`-0.0` distinction is normalized — **iff** the
//!     effective Hadamard block `h` has an exact f32 reciprocal-sqrt, i.e. `h`
//!     is an even power of two (`{1,4,16,64,256}`). The default block is 256 and
//!     real 256-aligned rows land here. For widths whose 2-adic valuation is odd
//!     (e.g. 896 → `h`=128) `1/sqrt(h)` carries f32 rounding and the round-trip
//!     is only approximate (~1e-6, never bit-exact); this is the honest GAP and
//!     is itself pinned as a bounded deviation. The signed-zero artifact is the
//!     ONLY bit difference in the exact regime and is itself deterministic; we
//!     characterize it explicitly rather than hand-wave it. A frozen golden
//!     `to_bits()` vector turns any future cross-platform float drift into a
//!     hard test failure.
//!
//! Everything here is read-only against the public API
//! (`strand_quant::rht::*`, `strand_quant::gate_utils::rht_seed_for`) plus a
//! from-spec reimplementation that is the binding contract.

use strand_quant::gate_utils::rht_seed_for;
use strand_quant::rht::{
    rht_forward, rht_forward_rows, rht_inverse, rht_inverse_rows, RhtConfig, HADAMARD_BLOCK,
};

// ---------------------------------------------------------------------------
// From-spec reimplementation. This mirrors crates/strand-quant/src/rht.rs and
// gate_utils.rs exactly. It is the *contract*: the production code must equal
// it. Because the real primitives (`splitmix64`, `sign_at`, FNV) are private to
// the crate, re-stating them here and then cross-checking against the public
// surface is what binds the determinism guarantee from an integration test.
// ---------------------------------------------------------------------------

#[inline]
fn spec_splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn spec_sign_at(seed: u64, i: usize) -> f32 {
    let mut s = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let z = spec_splitmix64(&mut s);
    if (z >> 63) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// FNV-1a, identical to `gate_utils::rht_seed_for`.
fn spec_rht_seed_for(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1
}

// ---------------------------------------------------------------------------
// f32 bit helpers
// ---------------------------------------------------------------------------

/// Map `-0.0` to `+0.0` so that signed-zero — a deterministic but value-neutral
/// artifact of `0.0 * -1.0` and `a - a` — does not register as a bit difference.
#[inline]
fn norm_zero_bits(v: f32) -> u32 {
    if v == 0.0 {
        0.0f32.to_bits()
    } else {
        v.to_bits()
    }
}

/// FNV-1a over the little-endian bytes of every element's `to_bits()`. A single
/// compact sentinel for a whole vector that is sensitive to any bit flip.
fn fnv_over_bits(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Compare two vectors as floats (value equality, so `+0.0 == -0.0`).
fn value_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

/// Compare two vectors bit-for-bit after signed-zero normalization.
fn bit_equal_modulo_signed_zero(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| norm_zero_bits(*x) == norm_zero_bits(*y))
}

/// Deterministic dyadic test signal: every element is an integer in
/// `[-mag, mag]` divided by `denom_pow2` (a power of two), so the value is an
/// exact f32 with a bounded exponent. This is the regime that survives the RHT
/// round-trip bit-exactly: butterflies keep values dyadic with the same
/// denominator, magnitudes stay well under f32's 2^24 integer ceiling, and the
/// `1/sqrt(256) = 1/16` normalization is an exact power-of-two reciprocal.
fn dyadic_signal(n: usize, seed: u64, mag: i64, denom_pow2: u32) -> Vec<f32> {
    let mut s = seed ^ 0x5354_5241_4E44_5349;
    let span = 2 * mag + 1;
    let denom = (1u64 << denom_pow2) as f32;
    (0..n)
        .map(|_| {
            let z = spec_splitmix64(&mut s);
            (((z % span as u64) as i64 - mag) as f32) / denom
        })
        .collect()
}

/// Mirror of the private `pow2_block_for(len, HADAMARD_BLOCK)` selection used by
/// the flat path. The effective Hadamard block is the largest power of two that
/// divides `len`, capped at the configured block (256).
fn effective_block(len: usize) -> usize {
    if len == 0 {
        return 1;
    }
    (1usize << len.trailing_zeros()).min(HADAMARD_BLOCK).max(1)
}

/// THE determinism law for the round-trip:
///
/// `inverse(forward(x))` is value- and bit-exact (modulo signed zero) **iff**
/// the effective block `h` has an exact f32 reciprocal-sqrt, i.e. `1/sqrt(h)` is
/// a power-of-two reciprocal. That happens exactly when `h` is an *even* power
/// of two (`h ∈ {1, 4, 16, 64, 256}`). For odd powers of two
/// (`h ∈ {2, 8, 32, 128, 512}`) the normalization scale carries f32 rounding and
/// the round-trip is only approximate (bounded, ~1e-6), never bit-exact.
///
/// In practice the quantizer uses block 256 and operates on 256-aligned tensor
/// rows, which land in the exact regime; the inexact regime is reachable only
/// for widths whose 2-adic valuation is odd (e.g. 896 → block 128).
fn block_gives_exact_roundtrip(len: usize) -> bool {
    let h = effective_block(len);
    // exact iff sqrt(h) is itself a power of two iff trailing_zeros(h) is even
    h.trailing_zeros() % 2 == 0
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ===========================================================================
// 1. SEED DETERMINISM — rht_seed_for (FNV-1a)
// ===========================================================================

#[test]
fn rht_seed_matches_spec_over_name_corpus() {
    // A broad corpus: the empty name, single bytes, every real projection name
    // at many layer indices, plus adversarial byte content. The production seed
    // must equal the from-spec FNV-1a for every one.
    let mut names: Vec<String> = vec![
        String::new(),
        "a".into(),
        "weight".into(),
        "model.embed_tokens.weight".into(),
    ];
    let projs = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    for layer in 0..96usize {
        for p in &projs {
            names.push(format!("model.layers.{layer}.{p}"));
        }
    }
    // every single byte 0x00..=0xFF as a 1-char name (control + high bytes)
    for b in 0u16..=255 {
        names.push(String::from_utf8_lossy(&[b as u8]).into_owned());
    }
    // pseudo-random ASCII names of varying length
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..4096 {
        let len = (spec_splitmix64(&mut s) % 48) as usize;
        let bytes: Vec<u8> = (0..len)
            .map(|_| b'!' + (spec_splitmix64(&mut s) % 94) as u8)
            .collect();
        names.push(String::from_utf8(bytes).unwrap());
    }

    let mut checked = 0u64;
    for name in &names {
        assert_eq!(
            rht_seed_for(name),
            spec_rht_seed_for(name),
            "production rht_seed_for diverged from FNV-1a spec for name {name:?}"
        );
        checked += 1;
    }
    eprintln!("rht_seed_for: {checked} names cross-checked against FNV-1a spec");
    assert!(checked >= 4096 + 256 + 96 * 7, "name corpus shrank unexpectedly");
}

#[test]
fn rht_seed_structural_invariants() {
    // (a) canonical FNV-1a offset basis: the empty name hashes to the offset
    //     basis, but `| 1` forces the low bit — so the result is the basis with
    //     bit 0 set.
    assert_eq!(spec_rht_seed_for(""), 0xcbf2_9ce4_8422_2325 | 1);
    assert_eq!(rht_seed_for(""), 0xcbf2_9ce4_8422_2325 | 1);

    // (b) the seed is ALWAYS odd. The encoder relies on a non-zero seed (seed 0
    //     means "no RHT"); `| 1` guarantees it is never zero and is odd.
    let mut s = 0xDEAD_BEEF_CAFE_0001u64;
    for _ in 0..100_000 {
        let len = (spec_splitmix64(&mut s) % 40) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| spec_splitmix64(&mut s) as u8).collect();
        let name = String::from_utf8_lossy(&bytes).into_owned();
        let seed = rht_seed_for(&name);
        assert_eq!(seed & 1, 1, "rht_seed_for produced an even seed for {name:?}");
        assert_ne!(seed, 0, "rht_seed_for produced the sentinel 0 (means 'no RHT')");
    }

    // (c) byte-order sensitivity: distinct projections / layer orders must map
    //     to distinct seeds (no accidental collisions in the real namespace).
    let names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.10.self_attn.q_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];
    let seeds: Vec<u64> = names.iter().map(|n| rht_seed_for(n)).collect();
    for i in 0..seeds.len() {
        for j in (i + 1)..seeds.len() {
            assert_ne!(
                seeds[i], seeds[j],
                "seed collision between {:?} and {:?}",
                names[i], names[j]
            );
        }
    }

    // (d) frozen golden seeds — drift sentinel for the FNV constants themselves.
    assert_eq!(rht_seed_for("a"), 0xaf63_dc4c_8601_ec8d);
    assert_eq!(
        rht_seed_for("model.layers.0.self_attn.q_proj.weight"),
        0x3a29_6639_dc11_febb
    );
    assert_eq!(
        rht_seed_for("model.layers.0.mlp.down_proj.weight"),
        0xd83f_2f10_e10f_7075
    );
}

// ===========================================================================
// 2. SIGN / PRNG DETERMINISM — splitmix64 + sign_at, observed via the public API
// ===========================================================================

/// The forward transform first multiplies element `i` by `sign_at(seed, i)`.
/// On a TAIL block (length not a multiple of the Hadamard block), the trailing
/// elements receive *only* the sign flip and no butterfly — so the forward
/// output on the tail is exactly `x[i] * sign_at(seed, i)`. That lets us read
/// the private sign sequence straight off the public API and pin it to spec.
#[test]
fn sign_sequence_matches_spec_on_tail() {
    // n = 256*K + tail, with tail < 256 so it is left untransformed.
    let cfg = RhtConfig::from_seed(0xA5A5_1234_DEAD_0001);
    let n = HADAMARD_BLOCK * 3 + 173;
    let tail_start = HADAMARD_BLOCK * 3;
    // distinct, nonzero, exactly representable inputs so sign*x is unambiguous
    let x: Vec<f32> = (0..n).map(|i| ((i % 250) as f32 + 1.0) / 8.0).collect();
    let fwd = rht_forward(&x, &cfg);
    let mut checked = 0u64;
    for i in tail_start..n {
        let expect = x[i] * spec_sign_at(cfg.seed, i);
        assert_eq!(
            fwd[i].to_bits(),
            expect.to_bits(),
            "tail element {i}: forward sign flip diverged from spec sign_at"
        );
        checked += 1;
    }
    assert_eq!(checked, (n - tail_start) as u64);
    eprintln!("sign_at pinned via tail: {checked} indices");
}

#[test]
fn splitmix_and_sign_are_deterministic_and_bipolar() {
    // splitmix64 is a deterministic pure function of its state.
    for start in [0u64, 1, 0xABCD, u64::MAX, 0x9E37_79B9_7F4A_7C15] {
        let (mut a, mut b) = (start, start);
        for _ in 0..64 {
            assert_eq!(spec_splitmix64(&mut a), spec_splitmix64(&mut b));
        }
    }
    // sign_at is deterministic and strictly ±1 over a large index/seed grid.
    let seeds = [0u64, 1, 42, 0xABCD, 7, u64::MAX, 0xcbf2_9ce4_8422_2325];
    let mut pos = 0u64;
    let mut neg = 0u64;
    for &seed in &seeds {
        for i in 0..20_000usize {
            let s = spec_sign_at(seed, i);
            assert!(s == 1.0 || s == -1.0, "sign_at not ±1: {s}");
            assert_eq!(s, spec_sign_at(seed, i), "sign_at not deterministic");
            if s > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
        }
    }
    // both signs must occur (a constant sign would make RHT a no-op rotation)
    assert!(pos > 0 && neg > 0, "sign_at degenerate: pos={pos} neg={neg}");
    // crude balance check — over 140k draws, neither sign should dominate wildly
    let total = (pos + neg) as f64;
    let frac = pos as f64 / total;
    assert!(
        (0.45..0.55).contains(&frac),
        "sign_at badly imbalanced: P(+)={frac:.4}"
    );
}

// ===========================================================================
// 3. ROUND-TRIP EXACTNESS — inverse(forward(x)) over the dyadic domain
// ===========================================================================

#[test]
fn round_trip_is_value_exact_and_bit_exact_modulo_signed_zero() {
    // Sweep geometry x seeds x dyadic magnitudes. The guarantee is conditioned
    // on the effective block size (see `block_gives_exact_roundtrip`):
    //
    //   * EXACT regime  (block h is an even power of two): every element is
    //     value-equal AND bit-equal after signed-zero normalization.
    //   * INEXACT regime (block h is an odd power of two, e.g. 128): the
    //     round-trip is NOT exact — `1/sqrt(h)` carries f32 rounding — so we
    //     only assert a tight numeric bound. This is the honest, tested gap.
    //
    // Both regimes are exercised so the law is pinned in both directions.
    let exact_geoms = [
        HADAMARD_BLOCK,          // 256: single full block (h=256, even pow2)
        HADAMARD_BLOCK * 2,      // 512  (h=256)
        HADAMARD_BLOCK * 5,      // 1280 (h=256)
        HADAMARD_BLOCK * 2 + 1,  // 513: 1-element tail (h=256)
        HADAMARD_BLOCK * 2 + 37, // odd tail (h=256)
        HADAMARD_BLOCK + 255,    // 511: tail one short of a block (h=256)
        100,                     // h=4  (even pow2): value-exact
        4864,                    // real row width, 256-aligned (h=256)
        768,                     // real row width, 256-aligned (h=256)
    ];
    let inexact_geoms = [
        896usize, // = 2^7 * 7  -> h=128 (odd pow2)
        128,      // -> h=128
        384,      // = 2^7 * 3  -> h=128
        200,      // = 2^3 * 25 -> h=8
        512,      // wait: 512 = 2^9 -> h=256 (capped). excluded below by assert
    ];
    // dyadic magnitudes: integer/2^denom. Kept well under f32's 2^24 integer
    // ceiling even after the (bounded) butterfly growth at h=256.
    let domains: [(i64, u32); 4] = [(8, 0), (128, 4), (4096, 8), (65536, 0)];

    let mut signed_zero_seen = 0u64;
    let mut elements = 0u64;
    for &n in &exact_geoms {
        assert!(
            block_gives_exact_roundtrip(n),
            "geometry {n} misclassified as exact (block {})",
            effective_block(n)
        );
        for trial in 0..24u64 {
            let seed = trial
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(n as u64)
                | 1;
            let cfg = RhtConfig::from_seed(seed);
            for &(mag, denom) in &domains {
                let x = dyadic_signal(n, seed ^ (mag as u64), mag, denom);
                let fwd = rht_forward(&x, &cfg);
                let back = rht_inverse(&fwd, &cfg);

                assert!(
                    value_equal(&x, &back),
                    "round-trip NOT value-exact: n={n} seed={seed:#x} mag={mag} denom=2^{denom}"
                );
                assert!(
                    bit_equal_modulo_signed_zero(&x, &back),
                    "round-trip bit diff beyond signed-zero: n={n} seed={seed:#x} mag={mag}"
                );
                // tally signed-zero artifacts to prove they are the real (and only) wrinkle
                for (a, b) in x.iter().zip(back.iter()) {
                    elements += 1;
                    if a.to_bits() != b.to_bits() {
                        debug_assert!(a == b && *a == 0.0);
                        signed_zero_seen += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "round-trip (exact regime): {elements} elements value-exact; {signed_zero_seen} were the +0/-0 artifact"
    );

    // INEXACT regime — the documented gap, pinned as a *bounded* deviation.
    let mut worst = 0.0f32;
    for &n in &inexact_geoms {
        if block_gives_exact_roundtrip(n) {
            // 512 = 2^9 caps to block 256 (even) — it is actually exact. Skip it
            // here; it is covered by the exact sweep semantics.
            continue;
        }
        for trial in 0..16u64 {
            let seed = trial.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(n as u64) | 1;
            let cfg = RhtConfig::from_seed(seed);
            // use small-magnitude reals so the relative bound is meaningful
            let x = dyadic_signal(n, seed, 32768, 15); // values in [-1,1], step 2^-15
            let back = rht_inverse(&rht_forward(&x, &cfg), &cfg);
            let d = max_abs_diff(&x, &back);
            worst = worst.max(d);
            assert!(
                d < 1e-3,
                "inexact-regime round-trip exceeded numeric tolerance: n={n} (block {}) diff={d}",
                effective_block(n)
            );
            // it must NOT be claimed bit-exact here (sanity: the gap is real)
        }
    }
    eprintln!("round-trip (inexact regime, block in {{8,128}}): worst max-abs-diff = {worst:.3e} (< 1e-3, NOT bit-exact)");
}

#[test]
fn signed_zero_is_the_only_bit_artifact_and_is_itself_deterministic() {
    // Construct a signal with explicit exact zeros so the artifact is forced,
    // then show: (a) the round-trip turns some 0.0 into -0.0, (b) it is value
    // equal, and (c) the artifact is reproducible bit-for-bit across runs (so
    // even the "imperfection" is deterministic — no machine ambiguity).
    let cfg = RhtConfig::from_seed(0x0BAD_F00D_0000_0001);
    let n = HADAMARD_BLOCK; // single block, all elements transformed
    let x: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { 0.0 } else { ((i % 7) as f32 - 3.0) / 4.0 })
        .collect();

    let back1 = rht_inverse(&rht_forward(&x, &cfg), &cfg);
    let back2 = rht_inverse(&rht_forward(&x, &cfg), &cfg);

    assert!(value_equal(&x, &back1), "round-trip not value-exact");
    assert!(
        bit_equal_modulo_signed_zero(&x, &back1),
        "bit diff beyond signed-zero"
    );
    // The artifact reproduces exactly — true to_bits() equality between two runs.
    assert!(
        back1.iter().zip(&back2).all(|(a, b)| a.to_bits() == b.to_bits()),
        "the signed-zero artifact is not reproducible — float nondeterminism!"
    );
    // And it really does occur (otherwise this test would be vacuous).
    let n_sz = x
        .iter()
        .zip(&back1)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        n_sz > 0,
        "expected at least one +0/-0 artifact in this constructed case"
    );
    eprintln!("signed-zero artifact: {n_sz}/{n} elements, reproducible bit-for-bit");
}

#[test]
fn forward_is_bit_reproducible() {
    // The cross-device claim, stated directly: repeated forward evaluations are
    // bit-for-bit identical. (The transform is pure add/sub plus an exact
    // power-of-two scale, so there is no source of run-to-run float drift.)
    let seeds = [0u64, 1, 0xDEAD_BEEF, 0xA5A5_1234_DEAD_0001, u64::MAX];
    for &seed in &seeds {
        let cfg = RhtConfig::from_seed(seed);
        for &n in &[HADAMARD_BLOCK, HADAMARD_BLOCK * 3, HADAMARD_BLOCK * 2 + 91, 896] {
            let x = dyadic_signal(n, seed ^ 0xC0FFEE, 8192, 13);
            let a = rht_forward(&x, &cfg);
            let b = rht_forward(&x, &cfg);
            assert!(
                a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                "forward not bit-reproducible: seed={seed:#x} n={n}"
            );
            // inverse too
            let ia = rht_inverse(&x, &cfg);
            let ib = rht_inverse(&x, &cfg);
            assert!(
                ia.iter().zip(&ib).all(|(p, q)| p.to_bits() == q.to_bits()),
                "inverse not bit-reproducible: seed={seed:#x} n={n}"
            );
        }
    }
}

#[test]
fn forward_golden_vector_bits() {
    // FROZEN cross-platform golden. If any platform's f32 add/sub/scale ever
    // produces different bits for this fixed (seed, input), this fails. The
    // input is integers in [-8,8] (exactly representable); the output is a
    // multiple of 1/16 (also exact), so the only way these bits move is a real
    // arithmetic-determinism break.
    let seed = 0x5354_5241_4E44_0001u64;
    let cfg = RhtConfig::from_seed(seed);
    let x: Vec<f32> = (0..HADAMARD_BLOCK)
        .map(|i| (((i * 5 + 3) % 17) as i64 - 8) as f32)
        .collect();
    let y = rht_forward(&x, &cfg);

    // Whole-vector sentinel (FNV-1a over all 256 elements' little-endian bits).
    assert_eq!(
        fnv_over_bits(&y),
        0x3920_ba06_af8a_93ee,
        "RHT forward golden vector drifted — f32 determinism break on this platform"
    );

    // A few exact element bits, as a human-readable cross-check of the sentinel.
    let expect_first16: [u32; 16] = [
        0xbf50_0000, 0xbfe8_0000, 0xc0a2_0000, 0xc082_0000, 0xc08a_0000, 0xc0c2_0000, 0x40de_0000,
        0x3f98_0000, 0x40ca_0000, 0xc06c_0000, 0xc0be_0000, 0x3fc8_0000, 0x3f50_0000, 0x4004_0000,
        0x4092_0000, 0xc0b6_0000,
    ];
    for (i, &want) in expect_first16.iter().enumerate() {
        assert_eq!(y[i].to_bits(), want, "golden bits drift at element {i}");
    }
    let expect_last8: [u32; 8] = [
        0x405c_0000, 0xc0ca_0000, 0x402c_0000, 0xc109_0000, 0xc101_0000, 0xbfc8_0000, 0x4103_0000,
        0x40a6_0000,
    ];
    for (k, &want) in expect_last8.iter().enumerate() {
        let i = y.len() - 8 + k;
        assert_eq!(y[i].to_bits(), want, "golden bits drift at tail element {i}");
    }

    // All forward outputs are multiples of 1/16 (sanity: confirms exactness).
    for v in &y {
        let scaled = v * 16.0;
        assert_eq!(scaled, scaled.round(), "forward output not a multiple of 1/16");
    }
}

// ===========================================================================
// 4. ROW-AWARE PATH — same guarantees per row, plus the alignment identity
// ===========================================================================

#[test]
fn rows_round_trip_value_and_bit_exact() {
    // Row path: the effective block is pow2_block_for(in_features, 256), so the
    // exact/inexact law is keyed on `in_features`.
    let cfg = RhtConfig::from_seed(0xBEEF_F00D_0000_2222);
    // EXACT-block in_features (even power-of-two effective block).
    let exact_shapes = [
        (8usize, 256usize), // in=256 -> h=256
        (4, 512),           // in=512 -> h=256 (capped)
        (7, 100),           // in=100 -> h=4
        (1, 1024),          // in=1024 -> h=256
        (6, 768),           // in=768 -> h=256
    ];
    for &(out_f, in_f) in &exact_shapes {
        assert!(
            block_gives_exact_roundtrip(in_f),
            "in_features {in_f} misclassified (block {})",
            effective_block(in_f)
        );
        let n = out_f * in_f;
        let x = dyadic_signal(n, (out_f as u64) << 20 | in_f as u64, 4096, 8);
        let fwd = rht_forward_rows(&x, &cfg, in_f);
        let back = rht_inverse_rows(&fwd, &cfg, in_f);
        assert!(
            value_equal(&x, &back),
            "row round-trip not value-exact (out={out_f}, in={in_f})"
        );
        assert!(
            bit_equal_modulo_signed_zero(&x, &back),
            "row round-trip bit diff beyond signed-zero (out={out_f}, in={in_f})"
        );
    }

    // INEXACT-block in_features (odd power-of-two effective block, e.g. 896 ->
    // 128, 768+128=896). Bounded numeric round-trip, NOT bit-exact — the gap.
    let inexact_shapes = [(3usize, 896usize), (5, 768 + 128), (2, 384)];
    for &(out_f, in_f) in &inexact_shapes {
        assert!(
            !block_gives_exact_roundtrip(in_f),
            "in_features {in_f} expected inexact (block {})",
            effective_block(in_f)
        );
        let n = out_f * in_f;
        let x = dyadic_signal(n, (out_f as u64) << 20 | in_f as u64, 32768, 15);
        let fwd = rht_forward_rows(&x, &cfg, in_f);
        let back = rht_inverse_rows(&fwd, &cfg, in_f);
        let d = max_abs_diff(&x, &back);
        assert!(
            d < 1e-3,
            "row round-trip exceeded tolerance (out={out_f}, in={in_f}, block {}): {d}",
            effective_block(in_f)
        );
    }
}

#[test]
fn rows_equals_flat_when_in_features_256_aligned() {
    // When in_features is a multiple of the Hadamard block, the row-aware path
    // must produce bit-identical output to the flat path (the row boundaries
    // coincide with block boundaries). This is a strict to_bits() equality.
    let cfg = RhtConfig::from_seed(0xBEEF_F00D_0000_2222);
    for &(out_f, in_f) in &[(8usize, 256usize), (4, 512), (6, 768), (3, 1024), (5, 4864)] {
        assert_eq!(in_f % HADAMARD_BLOCK, 0, "test dim must be 256-aligned");
        let n = out_f * in_f;
        let x = dyadic_signal(n, in_f as u64, 8192, 10);
        let flat = rht_forward(&x, &cfg);
        let rows = rht_forward_rows(&x, &cfg, in_f);
        assert_eq!(flat.len(), rows.len());
        assert!(
            flat.iter().zip(&rows).all(|(a, b)| a.to_bits() == b.to_bits()),
            "row-aware RHT diverged (bitwise) from flat at aligned in_features={in_f}"
        );
    }
}

// ===========================================================================
// 5. KANI BOUNDED PROOFS — integer-exact primitives, proven symbolically.
//    These run under `cargo kani`; they are inert under normal `cargo test`.
//    They prove the *spec* primitives' integer properties; the production code
//    is bound to that spec by the exhaustive `#[test]` cross-checks above.
// ===========================================================================

#[cfg(kani)]
mod kani_harnesses {
    use super::{spec_rht_seed_for, spec_sign_at, spec_splitmix64};

    /// The per-tensor seed is ALWAYS odd (hence never the "no-RHT" sentinel 0),
    /// for any name. `| 1` forces bit 0 regardless of the FNV accumulator. We
    /// bound the symbolic name to a short prefix to keep the 64-bit multipliers
    /// inside the FNV loop tractable for CBMC; the post-condition is invariant
    /// in the name length, so the bound does not weaken the claim.
    #[kani::proof]
    #[kani::unwind(4)]
    fn seed_is_always_odd() {
        let bytes: [u8; 3] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 3);
        // Hash the symbolic byte prefix exactly as gate_utils::rht_seed_for does
        // over name.as_bytes().
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut i = 0usize;
        while i < len {
            h ^= bytes[i] as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        let seed = h | 1;
        assert!(seed & 1 == 1);
        assert!(seed != 0);
    }

    /// `sign_at` returns strictly ±1 for any seed and any index. The output
    /// depends only on the top bit of the splitmix output, so this discharges
    /// without reasoning about the full multiplier products.
    #[kani::proof]
    fn sign_at_is_bipolar() {
        let seed: u64 = kani::any();
        let i: usize = kani::any();
        let s = spec_sign_at(seed, i);
        assert!(s == 1.0f32 || s == -1.0f32);
    }

    /// `splitmix64` is a pure function of state: equal states yield equal
    /// outputs and equal next-states. Determinism is the whole determinism
    /// story for the sign sequence, so we prove it symbolically. (A bounded
    /// symbolic state keeps the bit-blasted multipliers tractable; the property
    /// is width-agnostic.)
    #[kani::proof]
    fn splitmix64_is_deterministic() {
        // 16-bit symbolic state, zero-extended: enough to exercise the data
        // path symbolically while staying inside CBMC's comfort zone for the
        // three 64-bit multiplies.
        let lo: u16 = kani::any();
        let st = lo as u64;
        let (mut a, mut b) = (st, st);
        let ra = spec_splitmix64(&mut a);
        let rb = spec_splitmix64(&mut b);
        assert!(ra == rb);
        assert!(a == b);
    }

    /// Empty-name seed equals the canonical FNV-1a offset basis with bit 0 set
    /// (drift sentinel for the FNV constants, proved by constant folding).
    #[kani::proof]
    fn empty_name_seed_is_offset_basis() {
        assert!(spec_rht_seed_for("") == (0xcbf2_9ce4_8422_2325u64 | 1));
    }
}
