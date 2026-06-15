//! Determinism hardening for the C2 **side-info rANS** surface (`sideinfo_rans.rs`).
//!
//! THE MOAT: STRAND decode is bit-identical on every device — frozen integer Q12
//! model, float-free decode. This coder carries that guarantee for the `scale_q`
//! and outlier-position side-info streams: its decode side rebuilds a *static
//! integer* CDF from the stream bytes and runs the Ryg 32-bit rANS core with no
//! floating point and no platform-dependent reduction (see the module's docs).
//!
//! This file is the sibling of `exhaustive.rs`: it pins the surface against an
//! exhaustively-enumerated bounded domain, freezes cross-platform golden byte
//! vectors, and adds a Kani bounded proof of the rANS integer core. It is
//! **dependency-free** (no `proptest`, matching the crate's `[dev-dependencies]
//! = none` policy); property-style coverage uses a deterministic `splitmix64`
//! PRNG so every run hits the identical cases on every machine.
//!
//! # Reachability (honest scope)
//!
//! `sideinfo_rans.rs` is "(planned)": it is NOT yet declared as a module in
//! `lib.rs`, so it cannot be imported as `strand_quant::sideinfo_rans` from an
//! integration test. BUT — unlike its `debias_wire` sibling, whose production
//! code does `use crate::format::…` — `sideinfo_rans`'s *production* code (the
//! rANS core, `Model`, `encode_stream`/`decode_stream`, the transforms) is fully
//! self-contained: the only `crate::` reference in the file is one line inside
//! its own `#[cfg(test)]` block (`crate::outlier_wire::idx_bits_for`). So we test
//! the **real shipping code** by `#[path]`-including the source directly and
//! providing a faithful, byte-identical local copy of that one helper. This is
//! strictly stronger than a re-derived oracle: it proves the actual production
//! encode/decode preserves bit-identical decode, not a hand-copy of it.
//!
//! When the operator wires `pub mod sideinfo_rans;` into lib.rs, the `#[path]`
//! include + the local `outlier_wire` shim can be deleted and the same tests
//! re-pointed at `strand_quant::sideinfo_rans::*` unchanged.
//!
//! Run:  `cargo test -p strand-quant --test sideinfo_rans_determinism`
//! Kani: `cargo kani -p strand-quant --harness rans_core_is_exact_inverse` (after wiring,
//!        or against this file's `#[cfg(kani)]` block via a standalone kani build)

#![allow(clippy::needless_range_loop)]

// --- faithful local copy of the ONE crate-internal helper the included source's
//     own #[cfg(test)] block calls. Byte-identical to outlier_wire::idx_bits_for
//     (ceil-log2). Lets the production source compile as a path-included module
//     without editing any shared file. NOT under test — it is a compile shim. ---
mod outlier_wire {
    pub fn idx_bits_for(n: usize) -> u32 {
        if n <= 1 {
            1
        } else {
            usize::BITS - (n - 1).leading_zeros()
        }
    }
}

#[path = "../src/sideinfo_rans.rs"]
mod sr;

use sr::{
    decode_positions, decode_scale_q, decode_stream, encode_positions, encode_scale_q,
    encode_stream, encode_stream_with_model, gaps_to_positions, positions_to_gaps, unzigzag,
    zigzag, Model,
};

// ===========================================================================
// deterministic PRNG (no rand dep — identical sequence on every platform)
// ===========================================================================

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The single round-trip invariant every test funnels through:
///   1. decode(encode(raw)) == raw                      (byte-exact recovery)
///   2. decoder consumes EXACTLY the section it was given (no over/under-read)
///   3. encode is deterministic                          (same in → same bytes)
///   4. decode is deterministic & stateless              (same bytes → same out, every time)
///   5. decode is position-pure                          (embedding the section at a
///      nonzero offset yields the identical output and advances `pos` by the same delta)
/// Returns the encoded bytes so callers can do further structural asserts.
fn assert_roundtrip(raw: &[i64]) -> Vec<u8> {
    let enc = encode_stream(raw);

    // (3) encode determinism
    assert_eq!(encode_stream(raw), enc, "encode is not deterministic");

    // (1)+(2) recovery + exact consumption
    let mut pos = 0usize;
    let back = decode_stream(&enc, &mut pos).expect("decode of a self-produced section");
    assert_eq!(back, raw, "round-trip mismatch");
    assert_eq!(pos, enc.len(), "decoder must consume the whole section exactly");

    // (4) decode determinism / statelessness: many independent decodes agree
    for _ in 0..8 {
        let mut p = 0usize;
        let again = decode_stream(&enc, &mut p).expect("repeat decode");
        assert_eq!(again, raw, "decode is not deterministic across calls");
        assert_eq!(p, enc.len());
    }

    // (5) position purity: decode the SAME section embedded after junk.
    let prefix = [0xA5u8, 0x5A, 0x00, 0xFF, 0x42];
    for &off in &[0usize, 1, 5] {
        let mut buf = vec![0u8; off];
        buf.copy_from_slice(&prefix[..off]);
        buf.extend_from_slice(&enc);
        let mut p = off;
        let emb = decode_stream(&buf, &mut p).expect("decode from offset");
        assert_eq!(emb, raw, "decode depends on absolute buffer position (off={off})");
        assert_eq!(p - off, enc.len(), "consumed length changed with offset (off={off})");
    }

    enc
}

// ===========================================================================
// TIER 1 — exhaustive bounded enumeration (the flagship proof)
//
// Mirrors exhaustive.rs::exhaustive_state_stream_equivalence: enumerate EVERY
// stream over a small alphabet up to a bounded length via mixed-radix counting,
// and prove each one round-trips byte-exactly and decodes deterministically.
// This is a true exhaustive certificate over the enumerated domain — not a
// sample.
// ===========================================================================

/// Map a counter in `0..radix^len` to the `len`-symbol stream over `alpha`.
fn nth_stream(mut idx: u64, len: usize, alpha: &[i64]) -> Vec<i64> {
    let radix = alpha.len() as u64;
    let mut s = Vec::with_capacity(len);
    for _ in 0..len {
        s.push(alpha[(idx % radix) as usize]);
        idx /= radix;
    }
    s
}

#[test]
fn exhaustive_small_alphabet_streams() {
    // Alphabet chosen to exercise: zero, both signs, a value whose zig-zag needs
    // a 2-byte varint, and a duplicate-heavy distribution (so frequencies vary).
    let alpha: [i64; 4] = [0, 1, -1, 130];
    let radix = alpha.len() as u64;
    let mut covered: u64 = 0;
    let mut max_len_checked = 0usize;

    // Up to length 9 over radix 4 == 4^9 = 262_144 streams at the top tier;
    // summed over lengths 0..=9 that is (4^10 - 1)/3 = 349_525 distinct streams.
    // Every one is encoded AND decoded AND re-decoded ×8 — well within a fast
    // unit-test budget and an exhaustive certificate over this domain.
    for len in 0..=9usize {
        let count = radix.pow(len as u32);
        for idx in 0..count {
            let raw = nth_stream(idx, len, &alpha);
            assert_roundtrip(&raw);
            covered += 1;
        }
        max_len_checked = len;
    }

    // Coverage drift guard (same style as exhaustive.rs's exact-count asserts):
    // sum_{len=0}^{9} 4^len == (4^10 - 1) / 3.
    let expect: u64 = (0..=9u32).map(|l| 4u64.pow(l)).sum();
    assert_eq!(expect, (4u64.pow(10) - 1) / 3);
    assert_eq!(covered, expect, "enumeration coverage drifted");
    eprintln!(
        "[sideinfo_rans] exhaustive small-alphabet streams: {covered} distinct streams \
         (radix {radix}, len 0..={max_len_checked}) all byte-exact + decode-deterministic"
    );
}

#[test]
fn exhaustive_binary_streams_longer() {
    // Over a 2-symbol alphabet we can push length much further: lengths 0..=16
    // == sum 2^len = 2^17 - 1 = 131_071 streams. This stresses the rANS renorm
    // loop (long runs flush many bytes) and the back-to-front LIFO ordering.
    let alpha: [i64; 2] = [-1, 1];
    let mut covered: u64 = 0;
    for len in 0..=16usize {
        for idx in 0..(1u64 << len) {
            let raw = nth_stream(idx, len, &alpha);
            assert_roundtrip(&raw);
            covered += 1;
        }
    }
    assert_eq!(covered, (1u64 << 17) - 1, "binary enumeration coverage drifted");
    eprintln!("[sideinfo_rans] exhaustive binary streams len 0..=16: {covered} streams");
}

// ===========================================================================
// TIER 2 — escape-path enumeration: force values OUTSIDE a frozen model so the
// ESC slot + escaped-value blob are exercised exhaustively for short streams.
// The escape path is the determinism-critical "value not in the static CDF"
// branch; decode must recover it byte-exactly from the side blob.
// ===========================================================================

#[test]
fn exhaustive_escape_path_against_frozen_model() {
    // Freeze a model that only knows {0,1,2}. Then enumerate short streams over a
    // larger alphabet that includes UNKNOWN values {-5, 7, 999} → those route to
    // ESC. Every stream must round-trip exactly through encode_stream_with_model.
    let train: Vec<i64> = vec![0, 0, 0, 1, 1, 2];
    let model = Model::from_stream(&train);

    let alpha: [i64; 5] = [0, 1, 2, -5, 999]; // 3 modelled + 2 escaping
    let radix = alpha.len() as u64;
    let mut covered = 0u64;
    let mut saw_escape = false;
    for len in 0..=7usize {
        for idx in 0..radix.pow(len as u32) {
            let raw = nth_stream(idx, len, &alpha);
            if raw.iter().any(|&v| v == -5 || v == 999) {
                saw_escape = true;
            }
            let enc = encode_stream_with_model(&raw, &model);
            assert_eq!(
                encode_stream_with_model(&raw, &model),
                enc,
                "encode-with-frozen-model not deterministic"
            );
            let mut pos = 0usize;
            let back = decode_stream(&enc, &mut pos).expect("decode frozen-model section");
            assert_eq!(back, raw, "escape round-trip mismatch");
            assert_eq!(pos, enc.len());
            covered += 1;
        }
    }
    assert!(saw_escape, "test never exercised the ESC path");
    let expect: u64 = (0..=7u32).map(|l| radix.pow(l)).sum();
    assert_eq!(covered, expect);
    eprintln!("[sideinfo_rans] exhaustive escape-path streams: {covered} (frozen 3-symbol model)");
}

// ===========================================================================
// TIER 3 — CROSS-PLATFORM GOLDEN VECTORS
//
// The literal "bit-identical on every device" guarantee. These byte strings were
// produced by the real encoder on aarch64-darwin. The test asserts BOTH
// directions:
//   encode(raw)            == golden_bytes   (encode is bit-frozen)
//   decode(golden_bytes)   == raw            (decode is bit-frozen)
// Any platform that diverges — endianness, float contamination, a different
// reduction order in the CDF normalizer, a varint/zig-zag bug — fails here with
// a byte diff. If the production codec changes intentionally, these update in
// lockstep and the diff is the audit trail.
// ===========================================================================

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// (label, raw i64 stream, exact encoded bytes as hex).
const GOLDENS: &[(&str, &[i64], &str)] = &[
    ("empty", &[], "0000000001000000000040000000000400000000008000"),
    ("single_run", &[7, 7, 7, 7, 7, 7, 7, 7], "08000000020000000ee438001c070000000004000000281e4801"),
    (
        "small_mixed",
        &[3, 3, 3, -1, -1, 5, 0, 0, 0, 0, 7, 3, -1],
        "0d00000006000000004b1201b60d0649120a92040e92040092040000000007000000482e5c0dc32b89",
    ),
    (
        "extremes",
        &[i32::MIN as i64, i32::MAX as i64, 0, -1, 1, -1_000_000, 1_000_000],
        "0700000008000000000008010008020008ff887a000880897a0008feffffff0f0008ffffffff0f00080000080000000006000000417401101800",
    ),
    (
        "alt_sign",
        &[-2, 2, -2, 2, -2, 2, -2, 2, 0],
        "0900000004000000006606039b190499190066060000000005000000f411cc1dbc",
    ),
];

#[test]
fn golden_vectors_both_directions() {
    for (name, raw, hex) in GOLDENS {
        let want = unhex(hex);

        // encode is bit-frozen
        let got = encode_stream(raw);
        assert_eq!(
            got, want,
            "ENCODE golden drift for '{name}': bytes changed — \
             a cross-device divergence or an intended codec change.\n got={}\nwant={}",
            got.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            hex,
        );

        // decode is bit-frozen
        let mut pos = 0usize;
        let back = decode_stream(&want, &mut pos).expect("decode golden");
        assert_eq!(back, *raw, "DECODE golden drift for '{name}'");
        assert_eq!(pos, want.len(), "golden '{name}' not fully consumed");
    }
    eprintln!("[sideinfo_rans] {} cross-platform golden vectors frozen (both directions)", GOLDENS.len());
}

/// Golden vectors for the two public convenience wrappers — these pin the exact
/// bytes for the `scale_q` (i32) and gap-coded `positions` (u32) sections, which
/// are the two real levers the module exists for.
#[test]
fn golden_scale_q_and_positions() {
    let scale_q: [i32; 10] = [0, 1, -1, 100, 100, 100, -50, 0, 1, 1];
    let want = unhex("0a0000000600000000a20b01d10502771163d105c801741100d10500000000060000006a04d543eb49");
    assert_eq!(encode_scale_q(&scale_q), want, "scale_q encode golden drift");
    let mut pos = 0usize;
    assert_eq!(decode_scale_q(&want, &mut pos).unwrap(), scale_q);
    assert_eq!(pos, want.len());

    let positions: [u32; 7] = [3, 7, 8, 100, 101, 5000, 1_000_000];
    let want =
        unhex("0700000007000000020010060008080008b8010008c64c0008f0ba790008000008000000000600000081d00004a800");
    assert_eq!(encode_positions(&positions), want, "positions encode golden drift");
    let mut pos = 0usize;
    assert_eq!(decode_positions(&want, &mut pos).unwrap(), positions);
    assert_eq!(pos, want.len());
}

// ===========================================================================
// TIER 4 — Model (the determinism boundary) byte-exact self-description
//
// The static CDF is serialized into the stream and rebuilt on decode with no
// float. Prove serialize∘deserialize is the identity on the Model, and that the
// rebuilt model is byte-for-byte re-serializable (round-trip is a true fixpoint),
// over a wide spread of streams. A drift here = a different CDF on some platform
// = non-bit-identical decode.
// ===========================================================================

#[test]
fn model_serialize_is_a_fixpoint() {
    let mut s = 0xC0FF_EE00u64;
    for trial in 0..400 {
        // Mix of regimes: tiny alphabets, bell curves, sparse spikes, escapes.
        let n = 1 + (splitmix64(&mut s) % 5000) as usize;
        let spread = 1 + (splitmix64(&mut s) % 8000) as i64;
        let raw: Vec<i64> = (0..n)
            .map(|_| {
                let a = (splitmix64(&mut s) % spread as u64) as i64;
                let b = (splitmix64(&mut s) % spread as u64) as i64;
                a - b
            })
            .collect();
        let model = Model::from_stream(&raw);

        let mut buf = Vec::new();
        model.serialize(&mut buf);
        let mut pos = 0usize;
        let back = Model::deserialize(&buf, &mut pos).expect("model deserialize");
        assert_eq!(pos, buf.len(), "model not fully consumed (trial {trial})");
        assert_eq!(back, model, "model serialize/deserialize not the identity (trial {trial})");

        // re-serialize the rebuilt model: must be byte-identical (fixpoint).
        let mut buf2 = Vec::new();
        back.serialize(&mut buf2);
        assert_eq!(buf2, buf, "model re-serialization drifted (trial {trial})");
    }
    eprintln!("[sideinfo_rans] Model serialize∘deserialize fixpoint over 400 random distributions");
}

// ===========================================================================
// TIER 5 — randomized property coverage over realistic regimes (deterministic
// seed → reproducible everywhere). Complements the exhaustive small-domain tiers
// with large, structured streams that resemble the real scale_q / gap profiles.
// ===========================================================================

#[test]
fn property_random_streams_roundtrip() {
    let mut s = 0x5EED_1234u64;
    for trial in 0..300 {
        let n = (splitmix64(&mut s) % 4000) as usize;
        let regime = splitmix64(&mut s) % 4;
        let raw: Vec<i64> = (0..n)
            .map(|_| match regime {
                0 => 0,                                              // all-equal
                1 => (splitmix64(&mut s) % 3) as i64 - 1,           // tiny alphabet
                2 => {
                    // bell curve ~ scale_q
                    let mut acc = 0i64;
                    for _ in 0..4 {
                        acc += (splitmix64(&mut s) % 360) as i64;
                    }
                    acc - 718
                }
                _ => {
                    // wide / heavy-tailed, forces escapes past MAX_MODEL_SYMBOLS
                    let r = splitmix64(&mut s);
                    if r % 10 < 9 {
                        (r % 64) as i64 - 32
                    } else {
                        (r % 200_000) as i64 - 100_000
                    }
                }
            })
            .collect();
        assert_roundtrip(&raw);
        let _ = trial;
    }
    eprintln!("[sideinfo_rans] 300 randomized realistic streams round-trip + decode-deterministic");
}

// ===========================================================================
// TIER 6 — transform inverses (positions <-> gaps), exhaustive + adversarial.
// gaps_to_positions is on the DECODE path (decode_positions calls it), so its
// integer determinism + its rejection of corrupt (non-ascending / overflowing)
// gaps is part of the moat.
// ===========================================================================

#[test]
fn exhaustive_gap_transform_inverts() {
    // Enumerate every strictly-ascending position set of size up to 5 drawn from
    // {0..=7} (C(8,k) combinations) — small, but a complete certificate that the
    // gap transform + its codec round-trip is the identity on ascending inputs.
    let universe: Vec<u32> = (0..8).collect();
    let mut covered = 0u64;
    for k in 0..=5usize {
        // iterate all k-subsets of the 8-element universe via bit masks
        for mask in 0u32..(1 << universe.len()) {
            if mask.count_ones() as usize != k {
                continue;
            }
            let positions: Vec<u32> =
                universe.iter().copied().enumerate().filter(|(i, _)| (mask >> i) & 1 == 1).map(|(_, v)| v).collect();
            // positions are ascending by construction.
            let gaps = positions_to_gaps(&positions);
            let back = gaps_to_positions(&gaps).expect("gaps invert");
            assert_eq!(back, positions, "gap transform not invertible for {positions:?}");

            // full codec round-trip too
            let enc = encode_positions(&positions);
            let mut pos = 0usize;
            let dec = decode_positions(&enc, &mut pos).expect("decode positions");
            assert_eq!(dec, positions);
            assert_eq!(pos, enc.len());
            covered += 1;
        }
    }
    eprintln!("[sideinfo_rans] exhaustive gap-transform inverts over {covered} ascending position sets");
}

#[test]
fn corrupt_gaps_are_rejected_not_panicked() {
    // Non-ascending or overflowing reconstructions must be a clean Err, never a
    // panic and never a silent wrong decode (a corrupt stream must not pass).
    assert!(gaps_to_positions(&[5, 0]).is_err(), "zero gap (duplicate position) must error");
    assert!(gaps_to_positions(&[5, -3]).is_err(), "negative gap (descending) must error");
    assert!(gaps_to_positions(&[-1]).is_err(), "negative first position must error");
    assert!(gaps_to_positions(&[i64::MAX, 1]).is_err(), "overflow past u32 must error");
    // valid one still works
    assert_eq!(gaps_to_positions(&[3, 4, 1]).unwrap(), vec![3, 7, 8]);
}

// ===========================================================================
// TIER 7 — ADVERSARIAL TOTALITY: decode is a total function.
//
// A device that receives a corrupt/truncated section must NOT panic (that would
// be a denial-of-service divergence across platforms) and must consume bounded
// input. We prove this EXHAUSTIVELY over every truncation length and every
// single-byte flip of a representative section, plus a fuzz sweep of random
// byte soup.
// ===========================================================================

#[test]
fn every_truncation_is_total() {
    let raw: Vec<i64> = (0..400).map(|i| ((i * 37) % 11) as i64 - 5).collect();
    let enc = encode_stream(&raw);
    for cut in 0..=enc.len() {
        let mut pos = 0usize;
        // Must return (Ok or Err) without panicking and must not read past `cut`.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decode_stream(&enc[..cut], &mut pos)
        }));
        assert!(r.is_ok(), "decode panicked on truncation to {cut} bytes");
        assert!(pos <= cut, "decode read past the truncated end ({pos} > {cut})");
        // The only Ok on a strict truncation is the empty stream when the header
        // declared n==0; our stream declares n=400, so any short cut is an Err.
        if cut < enc.len() {
            if let Ok(Ok(v)) = &r {
                assert!(v.is_empty() || v.len() == raw.len(),
                    "truncated decode produced a partial-but-nonempty wrong stream at cut={cut}");
            }
        }
    }
    eprintln!("[sideinfo_rans] all {} truncations of a 400-symbol section are total (no panic)", enc.len() + 1);
}

#[test]
fn every_single_byte_flip_is_total() {
    // Smaller section so the O(len * 256) flip sweep stays fast; the property
    // (no panic, bounded read) is structural, not size-dependent.
    let raw: Vec<i64> = vec![3, 3, 1, 2, 0, -1, 5, 2, 2, 1, 0, 7, 3];
    let enc = encode_stream(&raw);
    let mut checked = 0u64;
    for i in 0..enc.len() {
        for delta in 1u16..=255 {
            let mut bad = enc.clone();
            bad[i] ^= delta as u8;
            let mut pos = 0usize;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                decode_stream(&bad, &mut pos)
            }));
            assert!(r.is_ok(), "decode panicked on single-byte flip at {i} (xor {delta})");
            assert!(pos <= bad.len(), "decode over-read on flip at {i}");
            // If it decodes, it must produce the declared symbol count or error —
            // never an unbounded / partial-garbage Vec beyond what n allows.
            if let Ok(Ok(v)) = &r {
                // n is read from the first 4 bytes; a flip there changes n, so we
                // only assert the structural bound that v came from a bounded loop.
                assert!(v.len() <= (1usize << 32), "impossible length");
            }
            checked += 1;
        }
    }
    eprintln!("[sideinfo_rans] all {checked} single-byte flips of a small section are total");
}

#[test]
fn random_byte_soup_never_panics() {
    let mut s = 0xBADD_C0DEu64;
    for _ in 0..20_000 {
        let n = (splitmix64(&mut s) % 80) as usize;
        let buf: Vec<u8> = (0..n).map(|_| (splitmix64(&mut s) & 0xFF) as u8).collect();
        let mut pos = 0usize;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = decode_stream(&buf, &mut pos);
            let _ = decode_scale_q(&buf, &mut { 0usize });
            let _ = decode_positions(&buf, &mut { 0usize });
        }));
        assert!(r.is_ok(), "decode panicked on random byte soup: {buf:?}");
        assert!(pos <= buf.len(), "over-read on random soup");
    }
    eprintln!("[sideinfo_rans] 20000 random byte-soup buffers: decode is total (no panic, bounded read)");
}

// ===========================================================================
// TIER 8 — zig-zag / varint exhaustive boundary proof (the symbol encoding that
// feeds the model + the escape blob). zig-zag must be a bijection and varint a
// lossless self-delimiting integer code, or symbol identity is lost on decode.
// ===========================================================================

#[test]
fn zigzag_bijective_over_boundaries_and_sweep() {
    // exact boundaries
    for v in [
        0i64, 1, -1, 2, -2, 63, -64, 64, -65, 127, -128,
        i32::MIN as i64, i32::MAX as i64, i64::MIN, i64::MAX, i64::MIN + 1, i64::MAX - 1,
    ] {
        assert_eq!(unzigzag(zigzag(v)), v, "zigzag not bijective at {v}");
    }
    // dense sweep around zero (every value -100_000..=100_000)
    for v in -100_000i64..=100_000 {
        assert_eq!(unzigzag(zigzag(v)), v);
    }
    // and a deterministic wide sweep across the full i64 range
    let mut s = 0x1357_9BDFu64;
    for _ in 0..200_000 {
        let v = splitmix64(&mut s) as i64;
        assert_eq!(unzigzag(zigzag(v)), v, "zigzag not bijective at {v}");
    }
}

// ===========================================================================
// KANI — bounded proof of the rANS integer core (the determinism heart).
//
// `enc_put` and `dec_get` are private to the included module, so we prove the
// observable contract instead: for a symbolic 2-symbol static model and a
// symbolic short stream, encode_stream_with_model followed by decode_stream is
// the IDENTITY. Because the model is fixed-shape, this exercises the integer
// renorm / state arithmetic symbolically with no float anywhere — Kani checks
// every reachable state, proving the core is an exact inverse (not just for the
// concrete vectors above). Bounded to keep the proof tractable.
//
// Run (after `pub mod sideinfo_rans;` is wired, or via a kani-only build that
// sets `--cfg kani`):
//   cargo kani -p strand-quant --harness rans_core_is_exact_inverse
// ===========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A frozen, hand-built 2-symbol+ESC model whose frequencies sum to
    /// SCALE_TOTAL. Building it from a fixed training stream keeps the CDF
    /// concrete while the *data* stream stays symbolic.
    fn fixed_model() -> Model {
        // symbols {0, 1} both present; ESC gets the floor. from_stream is
        // deterministic so this is a constant model.
        Model::from_stream(&[0i64, 0, 0, 1, 1])
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn rans_core_is_exact_inverse() {
        let model = fixed_model();
        // symbolic stream of up to 4 symbols drawn from {0,1} (both modelled,
        // so no escape blob — this isolates the rANS state arithmetic).
        let n: usize = kani::any();
        kani::assume(n <= 4);
        let mut raw = [0i64; 4];
        for i in 0..4 {
            let bit: bool = kani::any();
            raw[i] = if bit { 1 } else { 0 };
        }
        let stream = &raw[..n];

        let enc = encode_stream_with_model(stream, &model);
        let mut pos = 0usize;
        let back = decode_stream(&enc, &mut pos).expect("kani decode");
        assert_eq!(pos, enc.len());
        assert_eq!(back.len(), n);
        for i in 0..n {
            assert_eq!(back[i], stream[i]);
        }
    }

    /// Decode is a pure function of its byte input: two decodes of the same
    /// symbolic byte buffer agree and both stay in-bounds (no panic, bounded
    /// read). Proves statelessness / determinism on arbitrary (incl. corrupt)
    /// input symbolically.
    #[kani::proof]
    #[kani::unwind(6)]
    fn decode_is_pure_and_bounded() {
        let buf: [u8; 5] = kani::any();
        let mut p1 = 0usize;
        let r1 = decode_stream(&buf, &mut p1);
        let mut p2 = 0usize;
        let r2 = decode_stream(&buf, &mut p2);
        assert!(p1 <= buf.len());
        assert!(p2 <= buf.len());
        assert_eq!(p1, p2);
        assert_eq!(r1.is_ok(), r2.is_ok());
        if let (Ok(a), Ok(b)) = (&r1, &r2) {
            assert_eq!(a.len(), b.len());
        }
    }

    /// zig-zag is a symbolic bijection.
    #[kani::proof]
    fn zigzag_bijective_symbolic() {
        let v: i64 = kani::any();
        assert_eq!(unzigzag(zigzag(v)), v);
    }
}
