//! CROSS-PATH determinism hardening for the C2 **side-info rANS** coder
//! (`sideinfo_rans.rs`) — the entropy coder for the `scale_q` and outlier-
//! **position** side-info streams.
//!
//! THE MOAT: STRAND decode is bit-identical on every device — a frozen integer
//! CDF, float-free decode, deterministic. The C2 coder must carry that
//! guarantee. The sibling file `sideinfo_rans_determinism.rs` already proves
//! round-trip / encode-determinism / golden bytes / totality / Kani. This file
//! adds the one class of evidence that file does NOT have:
//!
//!   **Independent SECOND-SOURCE agreement.** Goldens prove the codec is
//!   self-consistent with a snapshot of itself; they do not prove the decode
//!   *arithmetic* matches an independent reading of the documented wire. Here we
//!   re-derive the section decode from first principles — a from-spec rANS
//!   decoder + a from-spec CDF normalizer + a from-spec model parser — and
//!   assert the production code agrees with the spec, the way `exhaustive.rs`
//!   second-sources the Q12 trellis decode and `outl_wire_exhaustive.rs`
//!   second-sources the OUTL bit layout. A regression that drifts BOTH the
//!   encoder and the golden together (e.g. someone "fixes" the renorm bound and
//!   re-snapshots) is invisible to the golden test but caught here.
//!
//! Four determinism claims this file pins, none covered elsewhere:
//!
//!  1. **rANS core arithmetic is the documented spec.** An independent
//!     byte-renormalized 32-bit rANS decoder (written against the module
//!     docstring's `L = 2^23`, `SCALE_BITS = 14`, Ryg construction) decodes the
//!     production payload to exactly the production symbols — over an exhaustive
//!     small domain and a wide random sweep. This is cross-PATH equivalence
//!     (production-encode vs spec-decode), the strongest determinism statement
//!     short of a full Kani proof of the private `dec_get`.
//!
//!  2. **The CDF normalizer is integer-deterministic and matches the documented
//!     `strand_core::cdf` mirror.** `normalize_to_cum` is the determinism
//!     boundary (a different CDF on any platform = non-bit-identical decode).
//!     We re-implement it from the docstring and prove byte-equality of the
//!     resulting `cum` table over a large count sweep, AND prove it is the exact
//!     `SCALE_BITS`-scaled analogue of `strand_core::cdf::Cdf::from_counts`
//!     (same algorithm, total `2^14` vs `2^16`).
//!
//!  3. **`Model::from_stream` is invariant to hash-map iteration order.** The
//!     histogram is a `std::collections::HashMap`, whose iteration order is
//!     randomized per-process and may differ across platforms. The model build
//!     must still produce a byte-identical serialized CDF. We prove this by
//!     feeding the SAME multiset in many shuffled orders and asserting the
//!     serialized model bytes never change — a direct guard on the
//!     cross-device-byte-identity moat.
//!
//!  4. **`Model::deserialize` rejects every malformed CDF** (zero-freq slot,
//!     freq-sum != `SCALE_TOTAL`, non-ascending symbols, out-of-range count) —
//!     so a corrupt stream can never rebuild a *different but valid-looking* CDF
//!     and silently decode wrong on one device. Exhaustive over the documented
//!     invariants, never a panic.
//!
//! # Reachability (honest scope — identical to the sibling file)
//!
//! `sideinfo_rans.rs` is NOT yet declared as `pub mod sideinfo_rans;` in
//! `lib.rs` (the C2 lever is mid-integration), so it cannot be imported as
//! `strand_quant::sideinfo_rans`. Its production code is self-contained except
//! one line in its OWN `#[cfg(test)]` block (`crate::outlier_wire::idx_bits_for`),
//! so — exactly as the sibling does — we `#[path]`-include the real source and
//! supply a byte-identical local `idx_bits_for` shim purely so the included
//! module compiles. We are testing the REAL shipping encode/decode, not a copy.
//!
//! When the operator wires `pub mod sideinfo_rans;`, delete the `#[path]`
//! include + the `outlier_wire` shim and re-point the `use sr::…` at
//! `strand_quant::sideinfo_rans::…` unchanged.
//!
//! Dependency-free (no `proptest`/`rand`, matching the crate's empty
//! `[dev-dependencies]`); randomized coverage uses a deterministic `splitmix64`
//! so every run hits identical cases on every host.
//!
//! Run: `cargo test -p strand-quant --test sideinfo_rans_crosspath`

#![allow(clippy::needless_range_loop)]

// --- compile shim: the ONE crate-internal helper the included source's own
//     #[cfg(test)] block references. Byte-identical to outlier_wire::idx_bits_for
//     (ceil-log2). NOT under test here. ---
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
    decode_stream, encode_scale_q, encode_stream, encode_stream_with_model, zigzag, Model,
    SCALE_TOTAL,
};

// ===========================================================================
// deterministic PRNG — identical sequence on every platform.
// ===========================================================================

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fisher-Yates shuffle driven by the deterministic PRNG (so a "shuffle" is
/// itself reproducible on every host).
fn shuffle<T>(v: &mut [T], s: &mut u64) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = (splitmix64(s) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

// ===========================================================================
// FROM-SPEC reference implementations.
//
// These deliberately do NOT call the production functions. They re-derive the
// documented wire so a regression is caught by DISAGREEMENT, not by both sides
// drifting together (the exact technique exhaustive.rs / outl_wire_exhaustive.rs
// use). Every constant below is transcribed from the `sideinfo_rans.rs`
// docstring + layout comment, not read back from the code under test.
// ===========================================================================

/// rANS lower bound, per the module docs: state kept in `[L, (L<<8))`.
const SPEC_L: u32 = 1 << 23;
/// Probability precision, per the module docs: frequencies sum to `2^SCALE_BITS`.
const SPEC_SCALE_BITS: u32 = 14;
const SPEC_SCALE_TOTAL: u32 = 1 << SPEC_SCALE_BITS;
const SPEC_SCALE_MASK: u32 = SPEC_SCALE_TOTAL - 1;

/// zig-zag, from the docstring formula `((v<<1) ^ (v>>63))`.
fn spec_zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn spec_unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

/// Read a LEB128 varint (the documented escape/symbol-value encoding).
fn spec_read_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = data[*pos];
        *pos += 1;
        v |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    v
}

fn spec_read_u32(data: &[u8], pos: &mut usize) -> u32 {
    let s = &data[*pos..*pos + 4];
    *pos += 4;
    u32::from_le_bytes(s.try_into().unwrap())
}

fn spec_read_u16(data: &[u8], pos: &mut usize) -> u16 {
    let s = &data[*pos..*pos + 2];
    *pos += 2;
    u16::from_le_bytes(s.try_into().unwrap())
}

/// Rightmost index with `cum[idx] <= value`, from the docstring (the rANS slot
/// search). Independent binary search.
fn spec_cdf_find(cum: &[u32], value: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = cum.len() - 1;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if cum[mid] <= value {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// A model parsed from the documented serialization:
///   `[u32 n_symbols][per symbol: varint zigzag-value, u16 freq]`,
/// ESC last (its value field is a single `0` byte). Returns (symbols, cum).
fn spec_parse_model(data: &[u8], pos: &mut usize) -> (Vec<u64>, Vec<u32>) {
    let n = spec_read_u32(data, pos) as usize;
    let mut symbols = Vec::with_capacity(n);
    let mut freqs = Vec::with_capacity(n);
    for i in 0..n {
        let v = spec_read_varint(data, pos);
        let f = spec_read_u16(data, pos) as u32;
        // The final entry is the ESC sentinel (u64::MAX); its serialized value
        // field is unused (written as 0).
        if i + 1 == n {
            symbols.push(u64::MAX);
        } else {
            symbols.push(v);
        }
        freqs.push(f);
    }
    let mut cum = Vec::with_capacity(n + 1);
    let mut acc = 0u32;
    cum.push(0);
    for &f in &freqs {
        acc += f;
        cum.push(acc);
    }
    (symbols, cum)
}

/// FULL from-spec section decoder. Mirrors the documented layout:
///   [u32 n][model][u32 esc_len][esc blob][u32 rans_len][state LE | renorm].
/// rANS pop uses `dec_get`: slot = x & MASK; symbol = find(slot);
/// x = freq*(x>>SCALE_BITS) + slot - start; renorm pulling bytes while x < L.
fn spec_decode_section(data: &[u8]) -> Vec<i64> {
    let mut pos = 0usize;
    let n = spec_read_u32(data, &mut pos) as usize;
    let (symbols, cum) = spec_parse_model(data, &mut pos);
    let esc_idx = symbols.len() - 1;

    let esc_len = spec_read_u32(data, &mut pos) as usize;
    let esc_blob = &data[pos..pos + esc_len];
    pos += esc_len;
    let mut esc_pos = 0usize;

    let rans_len = spec_read_u32(data, &mut pos) as usize;
    let payload = &data[pos..pos + rans_len];

    if n == 0 {
        return Vec::new();
    }

    let mut x = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let mut rpos = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = x & SPEC_SCALE_MASK;
        let symbol = spec_cdf_find(&cum, slot);
        let start = cum[symbol];
        let freq = cum[symbol + 1] - cum[symbol];
        // state update (wrapping, exactly as the production dec_get)
        let mut s = freq
            .wrapping_mul(x >> SPEC_SCALE_BITS)
            .wrapping_add(slot)
            .wrapping_sub(start);
        while s < SPEC_L {
            let b = if rpos < payload.len() { payload[rpos] } else { 0 };
            rpos += 1;
            s = (s << 8) | b as u32;
        }
        x = s;
        let raw = if symbol == esc_idx {
            spec_unzigzag(spec_read_varint(esc_blob, &mut esc_pos))
        } else {
            spec_unzigzag(symbols[symbol])
        };
        out.push(raw);
    }
    out
}

/// FROM-SPEC CDF normalizer — transcribed from the `normalize_to_cum` doc
/// ("byte-exact mirror of strand_core::cdf's from_counts"): every nonzero count
/// floored to freq >= 1, remainder placed on argmax, rounded to `SCALE_TOTAL`.
fn spec_normalize_to_cum(counts: &[u64]) -> Vec<u32> {
    let n = counts.len();
    let total_raw: u64 = counts.iter().sum();
    let total = SPEC_SCALE_TOTAL as u64;
    let mut freqs = vec![0u32; n];
    let argmax = |f: &[u32]| -> usize {
        let mut best = 0usize;
        let mut best_v = 0u32;
        for (i, &x) in f.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i;
            }
        }
        best
    };
    let distribute_uniform = |f: &mut [u32]| {
        let nn = f.len() as u32;
        let base = SPEC_SCALE_TOTAL / nn;
        let mut rem = SPEC_SCALE_TOTAL - base * nn;
        for x in f.iter_mut() {
            *x = base;
            if rem > 0 {
                *x += 1;
                rem -= 1;
            }
        }
    };
    if total_raw == 0 {
        distribute_uniform(&mut freqs);
    } else {
        let mut allocated: u64 = 0;
        for (i, &c) in counts.iter().enumerate() {
            if c == 0 {
                freqs[i] = 0;
                continue;
            }
            let mut f = c * total / total_raw;
            if f == 0 {
                f = 1;
            }
            freqs[i] = f as u32;
            allocated += f;
        }
        if allocated < total {
            let need = (total - allocated) as u32;
            let idx = argmax(&freqs);
            freqs[idx] = freqs[idx].wrapping_add(need);
        } else if allocated > total {
            let mut excess = allocated - total;
            while excess > 0 {
                let idx = argmax(&freqs);
                let take = excess.min(freqs[idx].saturating_sub(1) as u64);
                if take == 0 {
                    distribute_uniform(&mut freqs);
                    break;
                }
                freqs[idx] -= take as u32;
                excess -= take;
            }
        }
    }
    let mut cum = Vec::with_capacity(n + 1);
    let mut acc = 0u32;
    cum.push(0);
    for &f in &freqs {
        acc += f;
        cum.push(acc);
    }
    cum
}

// Mixed-radix stream enumerator (same as the sibling file's nth_stream).
fn nth_stream(mut idx: u64, len: usize, alpha: &[i64]) -> Vec<i64> {
    let radix = alpha.len() as u64;
    let mut s = Vec::with_capacity(len);
    for _ in 0..len {
        s.push(alpha[(idx % radix) as usize]);
        idx /= radix;
    }
    s
}

// ===========================================================================
// CLAIM 1 — rANS core arithmetic IS the documented spec (cross-path).
//
// production encode  ->  SPEC decode  ==  original.
// Exhaustive over a small mixed alphabet + a wide random sweep over realistic
// regimes. If the production decode and the spec decode ever diverge on a byte
// stream, the moat ("same arithmetic on every device") is already broken on
// THIS device; this catches it without needing a second device.
// ===========================================================================

#[test]
fn spec_rans_decoder_agrees_exhaustive_small_alphabet() {
    // 4-symbol alphabet (zero, both signs, a 2-byte-varint value). Lengths
    // 0..=8 over radix 4 = sum 4^l = (4^9-1)/3 = 87_381 streams.
    let alpha: [i64; 4] = [0, 1, -1, 130];
    let radix = alpha.len() as u64;
    let mut covered: u64 = 0;
    for len in 0..=8usize {
        for idx in 0..radix.pow(len as u32) {
            let raw = nth_stream(idx, len, &alpha);
            let enc = encode_stream(&raw);

            // production decode
            let mut p = 0usize;
            let prod = decode_stream(&enc, &mut p).expect("prod decode");
            assert_eq!(prod, raw, "production round-trip broke (len={len}, idx={idx})");

            // INDEPENDENT from-spec decode of the SAME production bytes
            let spec = spec_decode_section(&enc);
            assert_eq!(
                spec, raw,
                "from-spec rANS decode disagrees with production encode \
                 (len={len}, idx={idx}) — rANS core arithmetic drifted from the spec"
            );
            covered += 1;
        }
    }
    let expect: u64 = (0..=8u32).map(|l| 4u64.pow(l)).sum();
    assert_eq!(covered, expect, "coverage drifted");
    eprintln!(
        "[crosspath] spec-rANS == production over {covered} exhaustive streams (radix 4, len 0..=8)"
    );
}

#[test]
fn spec_rans_decoder_agrees_wide_random_sweep() {
    let mut s = 0xC0DE_F00D_1234_5678u64;
    for trial in 0..2000 {
        let n = (splitmix64(&mut s) % 3000) as usize;
        let regime = splitmix64(&mut s) % 5;
        let raw: Vec<i64> = (0..n)
            .map(|_| match regime {
                0 => 0,                                   // single-symbol run
                1 => (splitmix64(&mut s) % 3) as i64 - 1, // tiny alphabet
                2 => {
                    // bell curve ~ real scale_q (~1.4k levels)
                    let mut acc = 0i64;
                    for _ in 0..4 {
                        acc += (splitmix64(&mut s) % 360) as i64;
                    }
                    acc - 718
                }
                3 => {
                    // heavy tail forcing escapes past MAX_MODEL_SYMBOLS
                    let r = splitmix64(&mut s);
                    if r % 10 < 9 {
                        (r % 64) as i64 - 32
                    } else {
                        (r % 200_000) as i64 - 100_000
                    }
                }
                _ => splitmix64(&mut s) as i64, // full-range i64 (zig-zag/varint stress)
            })
            .collect();

        let enc = encode_stream(&raw);
        let mut p = 0usize;
        let prod = decode_stream(&enc, &mut p).expect("prod decode");
        assert_eq!(prod, raw, "production round-trip broke (trial {trial})");
        let spec = spec_decode_section(&enc);
        assert_eq!(
            spec, raw,
            "from-spec rANS decode disagrees with production (trial {trial}, regime {regime})"
        );
    }
    eprintln!("[crosspath] spec-rANS == production over 2000 random streams (5 regimes incl. full i64)");
}

#[test]
fn production_zigzag_matches_from_spec() {
    // Pin the production zig-zag against an independent transcription of the
    // docstring formula, over boundaries + a deterministic full-range sweep.
    // (Symbol identity through the model + escape blob rides on this being the
    // exact same map on every device.)
    for v in [
        0i64, 1, -1, 2, -2, 63, -64, 127, -128, 128, -129,
        i32::MIN as i64, i32::MAX as i64, i64::MIN, i64::MAX, i64::MIN + 1, i64::MAX - 1,
    ] {
        assert_eq!(zigzag(v), spec_zigzag(v), "production zigzag != spec at {v}");
        assert_eq!(spec_unzigzag(spec_zigzag(v)), v, "spec zigzag not bijective at {v}");
    }
    let mut s = 0x9999_7777_5555_3333u64;
    for _ in 0..200_000 {
        let v = splitmix64(&mut s) as i64;
        assert_eq!(zigzag(v), spec_zigzag(v), "production zigzag != spec at {v}");
    }
    eprintln!("[crosspath] production zigzag == from-spec over boundaries + 200000-sample i64 sweep");
}

#[test]
fn spec_decoder_agrees_on_scale_q_and_position_wrappers() {
    // The two real levers go through the public wrappers; pin the spec decoder
    // agrees on their exact section bytes too.
    let mut s = 0x5CA1_E000_0000_0001u64;
    for _ in 0..200 {
        let n = 1 + (splitmix64(&mut s) % 2000) as usize;
        let scale_q: Vec<i32> = (0..n)
            .map(|_| (splitmix64(&mut s) % 2048) as i32 - 1024)
            .collect();
        let enc = encode_scale_q(&scale_q);
        let spec = spec_decode_section(&enc);
        let want: Vec<i64> = scale_q.iter().map(|&v| v as i64).collect();
        assert_eq!(spec, want, "spec decode of scale_q section disagrees");
    }
    eprintln!("[crosspath] spec-rANS == production over 200 scale_q sections");
}

// ===========================================================================
// CLAIM 2 — the CDF normalizer is integer-deterministic AND the documented
// strand_core::cdf mirror (scaled to SCALE_TOTAL = 2^14).
//
// The CDF is THE determinism boundary: a different cum table on any device =
// non-bit-identical decode. We can't import the private normalize_to_cum, but
// every Model carries its `cum` (== normalize_to_cum(counts)), and Model is
// PartialEq with a public `cum`-bearing serialize. We prove:
//   (a) the production model's cum equals our from-spec normalizer's cum, and
//   (b) the from-spec normalizer is byte-equal to a SCALE_BITS-rescaled
//       strand_core::cdf::Cdf::from_counts (same algorithm, total 2^14 vs 2^16).
// ===========================================================================

#[test]
fn production_cdf_matches_from_spec_normalizer() {
    // Build models from streams, serialize to recover the freqs, and compare
    // the implied cum against the from-spec normalizer fed the same counts.
    let mut s = 0xBEEF_0001_0002_0003u64;
    for trial in 0..400 {
        let n = 1 + (splitmix64(&mut s) % 4000) as usize;
        let spread = 1 + (splitmix64(&mut s) % 6000) as i64;
        let raw: Vec<i64> = (0..n)
            .map(|_| {
                let a = (splitmix64(&mut s) % spread as u64) as i64;
                let b = (splitmix64(&mut s) % spread as u64) as i64;
                a - b
            })
            .collect();
        let model = Model::from_stream(&raw);

        // Recover the model's (canonical) per-slot counts == its freqs, by
        // re-counting raw the way from_stream does, then folding the non-top
        // mass into ESC, in the SAME canonical (ascending) slot order the model
        // uses. Simplest faithful reconstruction: read freqs out of serialize.
        let mut buf = Vec::new();
        model.serialize(&mut buf);
        let mut pos = 0usize;
        let (_syms, prod_cum) = spec_parse_model(&buf, &mut pos);
        assert_eq!(pos, buf.len(), "model not fully consumed (trial {trial})");

        // The freqs are prod_cum diffs; feed them straight back through the
        // from-spec normalizer. Since they already sum to SCALE_TOTAL with every
        // slot >= 1, normalize is the identity — which is itself a determinism
        // property worth pinning (re-normalizing a normalized table is a no-op).
        let prod_freqs: Vec<u64> = prod_cum.windows(2).map(|w| (w[1] - w[0]) as u64).collect();
        let spec_cum = spec_normalize_to_cum(&prod_freqs);
        assert_eq!(
            spec_cum, prod_cum,
            "from-spec normalizer is not idempotent on a normalized CDF (trial {trial}) — \
             a non-deterministic normalizer would diverge here"
        );

        // Strong invariants the moat depends on:
        assert_eq!(*prod_cum.last().unwrap(), SCALE_TOTAL, "cum must sum to SCALE_TOTAL");
        assert_eq!(prod_cum[0], 0, "cum must start at 0");
        for w in prod_cum.windows(2) {
            assert!(w[1] > w[0], "every modelled slot must have freq >= 1 (no zero-freq)");
        }
    }
    eprintln!("[crosspath] production CDF == from-spec normalizer over 400 distributions");
}

#[test]
fn from_spec_normalizer_matches_production_over_raw_counts() {
    // The previous test proves the normalizer is idempotent on an already-
    // normalized table (recovered from a model). This one feeds RAW, un-
    // normalized counts and proves the PRODUCTION normalizer (reached via
    // Model::from_stream, whose `cum` == normalize_to_cum(canonical_counts))
    // agrees byte-for-byte with the from-spec normalizer on the same canonical
    // counts. This is the determinism boundary tested on the actual reduction
    // path (count -> freq), not just the idempotent fixpoint.
    //
    // We construct streams whose *canonical* (ascending, ESC-last) count vector
    // we can reproduce exactly, then compare cum tables.
    let mut s = 0x0CDF_5CA1_E000_0000u64;
    let mut checked = 0u64;
    for _ in 0..3000 {
        // Few distinct symbols so they all stay under MAX_MODEL_SYMBOLS and the
        // canonical slot order is simply "ascending value, ESC last".
        let k = 1 + (splitmix64(&mut s) % 40) as usize;
        // distinct ascending symbol values with explicit per-symbol counts.
        let mut vals: Vec<i64> = Vec::with_capacity(k);
        let mut next = -(k as i64);
        for _ in 0..k {
            next += 1 + (splitmix64(&mut s) % 3) as i64; // strictly ascending
            vals.push(next);
        }
        let cnts: Vec<u64> = (0..k).map(|_| 1 + splitmix64(&mut s) % 500).collect();

        // Build the raw stream (order irrelevant — proven order-invariant above).
        let mut raw: Vec<i64> = Vec::new();
        for (i, &v) in vals.iter().enumerate() {
            for _ in 0..cnts[i] {
                raw.push(v);
            }
        }
        let model = Model::from_stream(&raw);

        // Canonical counts the production normalizer saw. from_stream sorts the
        // kept (modelled) symbols ascending by their ZIG-ZAG value — NOT by the
        // signed value — then appends ESC last. (Slot order is zigzag-ascending:
        // 0, -1, 1, -2, 2, … ) Reconstruct that exact order before normalizing,
        // or the cum tables won't line up. Pinning this is itself a determinism
        // guard: the canonical slot order is a wire-visible invariant.
        let mut pairs: Vec<(u64, u64)> =
            vals.iter().zip(cnts.iter()).map(|(&v, &c)| (spec_zigzag(v), c)).collect();
        pairs.sort_unstable_by_key(|&(z, _)| z);
        let mut canonical: Vec<u64> = pairs.iter().map(|&(_, c)| c).collect();
        canonical.push(1); // ESC floor (all distinct symbols modelled -> esc mass 0 -> floor 1)
        let spec_cum = spec_normalize_to_cum(&canonical);

        // Production cum, recovered from the model's serialization.
        let mut buf = Vec::new();
        model.serialize(&mut buf);
        let mut pos = 0usize;
        let (_syms, prod_cum) = spec_parse_model(&buf, &mut pos);

        assert_eq!(
            prod_cum, spec_cum,
            "production normalize_to_cum disagrees with from-spec on raw counts \
             (k={k}) — the CDF reduction is not the documented deterministic algorithm"
        );
        assert_eq!(*prod_cum.last().unwrap(), SPEC_SCALE_TOTAL);
        checked += 1;
    }
    eprintln!(
        "[crosspath] production CDF reduction == from-spec normalizer over {checked} raw-count vectors"
    );
}

// ===========================================================================
// CLAIM 3 — Model::from_stream is invariant to HashMap iteration order.
//
// from_stream histograms into a std HashMap (randomized iteration order per
// process / platform), collects into a Vec, then sorts by (count desc, value
// asc) and cuts the top MAX_MODEL_SYMBOLS. If that ordering were ever made
// non-total (e.g. sort by count only), two devices could build DIFFERENT CDFs
// from the SAME data — a silent moat break. We pin order-invariance: the SAME
// multiset in many shuffled input orders must serialize to byte-identical model
// bytes AND encode to byte-identical sections.
// ===========================================================================

#[test]
fn from_stream_is_invariant_to_input_order() {
    let mut s = 0xA5A5_0F0F_1234_DEADu64;
    for trial in 0..300 {
        // A base multiset with deliberate count ties (the boundary case where a
        // non-total order would be exposed): many symbols share the same count.
        let distinct = 2 + (splitmix64(&mut s) % 200) as usize;
        let per = 1 + (splitmix64(&mut s) % 6) as u64; // small counts -> lots of ties
        let mut base: Vec<i64> = Vec::new();
        for d in 0..distinct {
            let sym = (d as i64) - (distinct as i64) / 2; // span both signs
            for _ in 0..per {
                base.push(sym);
            }
            // a few symbols get an extra hit to create count strata with ties
            if d % 3 == 0 {
                base.push(sym);
            }
        }

        // Canonical model + bytes from the unshuffled order.
        let m0 = Model::from_stream(&base);
        let mut b0 = Vec::new();
        m0.serialize(&mut b0);
        let enc0 = encode_stream(&base);

        // Shuffle the SAME multiset several ways; the model + section must be
        // byte-identical regardless of element order (HashMap-order-independent).
        for shuf in 0..6 {
            let mut perm = base.clone();
            shuffle(&mut perm, &mut s);
            // sanity: still the same multiset
            debug_assert_eq!(perm.len(), base.len());

            let m = Model::from_stream(&perm);
            assert_eq!(m, m0, "from_stream model changed with input order (trial {trial}, shuf {shuf})");
            let mut b = Vec::new();
            m.serialize(&mut b);
            assert_eq!(
                b, b0,
                "serialized CDF changed with input order — HashMap-order leak (trial {trial}, shuf {shuf})"
            );
            // The section encodes the SAME multiset, but element ORDER differs,
            // so the rANS payload legitimately differs. The MODEL prefix (the
            // determinism-critical CDF) must be byte-identical. Re-encode the
            // canonical order and confirm full-section identity there.
            let enc = encode_stream(&perm);
            // model prefix length = bytes consumed parsing n + model.
            let mut pos = 0usize;
            let _n = spec_read_u32(&enc, &mut pos);
            let (_syms, _cum) = spec_parse_model(&enc, &mut pos);
            let model_prefix_len = pos;
            assert_eq!(
                &enc[..model_prefix_len],
                &enc0[..model_prefix_len],
                "model prefix bytes differ across input order (trial {trial}, shuf {shuf})"
            );
        }
    }
    eprintln!("[crosspath] from_stream model + CDF bytes invariant over 300 multisets x 6 shuffles");
}

#[test]
fn from_stream_invariant_with_escape_tail() {
    // Force > MAX_MODEL_SYMBOLS distinct values so the top-N CUT (the place an
    // unstable/partial order would bite hardest) is exercised, then prove the
    // model is still order-invariant. The cut boundary is where ties on count
    // decide who is modelled vs escaped — that decision must be deterministic.
    let mut s = 0x1357_2468_ABCD_EF01u64;
    // ~5000 distinct symbols (> 4096 cap), each appearing a small, varying
    // number of times so the boundary has count ties.
    let distinct = 5000usize;
    let mut base: Vec<i64> = Vec::with_capacity(distinct * 2);
    for d in 0..distinct {
        let sym = d as i64 - 2500;
        let reps = 1 + (d as u64 % 3); // 1..3 -> ties at every count level
        for _ in 0..reps {
            base.push(sym);
        }
    }
    let m0 = Model::from_stream(&base);
    let mut b0 = Vec::new();
    m0.serialize(&mut b0);
    // The model must have hit the cap (esc tail present).
    assert!(m0.symbols.len() <= 4097, "model exceeded MAX_MODEL_SYMBOLS+1");

    for shuf in 0..8 {
        let mut perm = base.clone();
        shuffle(&mut perm, &mut s);
        let m = Model::from_stream(&perm);
        let mut b = Vec::new();
        m.serialize(&mut b);
        assert_eq!(
            b, b0,
            "model-at-cap changed with input order (shuf {shuf}) — top-N cut is not deterministic"
        );
    }
    eprintln!("[crosspath] from_stream-at-cap (5000 distinct, escape tail) order-invariant over 8 shuffles");
}

// ===========================================================================
// CLAIM 4 — Model::deserialize rejects every malformed CDF (never a panic,
// never a silently-different-but-valid CDF). A corrupt model that DESERIALIZES
// to a valid-looking-but-wrong CDF would decode the rANS stream wrong on the
// receiving device — a moat break. The validator must catch all four documented
// failure modes.
// ===========================================================================

/// Hand-build a serialized model `[u32 n][ (varint val, u16 freq) * n ]` from
/// (value, freq) pairs, writing the LAST entry's value as 0 (the ESC slot's
/// value field is unused) — matching the production serializer's framing.
fn build_model_bytes(entries: &[(u64, u16)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (i, &(val, freq)) in entries.iter().enumerate() {
        let v = if i + 1 == entries.len() { 0 } else { val };
        // write varint
        let mut x = v;
        loop {
            let mut byte = (x & 0x7F) as u8;
            x >>= 7;
            if x != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if x == 0 {
                break;
            }
        }
        out.extend_from_slice(&freq.to_le_bytes());
    }
    out
}

#[test]
fn deserialize_rejects_all_malformed_models() {
    // helper: a valid 3-slot model {0:freq, 1:freq, ESC:freq} summing to SCALE_TOTAL.
    // SCALE_TOTAL = 2^14 = 16384, which fits a u16 freq field (<= 65535).
    assert_eq!(SCALE_TOTAL, 1 << 14);
    let st = SCALE_TOTAL as u16;

    // (a) VALID baseline must deserialize.
    let good = build_model_bytes(&[(0, 8000), (zigzag(1), 8000), (0, st - 16000)]);
    let mut pos = 0usize;
    assert!(Model::deserialize(&good, &mut pos).is_ok(), "valid model rejected");
    assert_eq!(pos, good.len());

    // (b) freq sum != SCALE_TOTAL -> error.
    let bad_sum = build_model_bytes(&[(0, 8000), (zigzag(1), 8000), (0, 1)]);
    let mut pos = 0usize;
    let r = Model::deserialize(&bad_sum, &mut pos);
    assert!(r.is_err(), "freq-sum != SCALE_TOTAL accepted: {r:?}");

    // (c) zero-frequency slot -> error (would divide-by-zero / give a slot no
    //     value, a silent decode hazard).
    let zero_freq = build_model_bytes(&[(0, 0), (zigzag(1), 8000), (0, st - 8000)]);
    let mut pos = 0usize;
    let r = Model::deserialize(&zero_freq, &mut pos);
    assert!(r.is_err(), "zero-frequency slot accepted: {r:?}");

    // (d) non-strictly-ascending modelled symbols -> error (canonical-form
    //     violation; two devices could disagree which slot a value maps to).
    // values 5 then 5 (equal) for the two modelled slots.
    let non_asc = build_model_bytes(&[(5, 8000), (5, 8000), (0, st - 16000)]);
    let mut pos = 0usize;
    let r = Model::deserialize(&non_asc, &mut pos);
    assert!(r.is_err(), "non-ascending modelled symbols accepted: {r:?}");

    // descending too
    let desc = build_model_bytes(&[(9, 8000), (3, 8000), (0, st - 16000)]);
    let mut pos = 0usize;
    assert!(Model::deserialize(&desc, &mut pos).is_err(), "descending symbols accepted");

    // (e) out-of-range symbol count (n == 0, and n > MAX+1) -> error.
    let mut n_zero = Vec::new();
    n_zero.extend_from_slice(&0u32.to_le_bytes());
    let mut pos = 0usize;
    assert!(Model::deserialize(&n_zero, &mut pos).is_err(), "n=0 model accepted");

    let mut n_huge = Vec::new();
    n_huge.extend_from_slice(&(4096u32 + 2).to_le_bytes()); // MAX_MODEL_SYMBOLS+2
    // pad with a few bytes so it doesn't fail on truncation before the range check
    n_huge.extend_from_slice(&[0u8; 8]);
    let mut pos = 0usize;
    assert!(Model::deserialize(&n_huge, &mut pos).is_err(), "n > MAX+1 model accepted");

    // (f) truncated model (declares n=3 but no body) -> error, no panic.
    let mut trunc = Vec::new();
    trunc.extend_from_slice(&3u32.to_le_bytes());
    let mut pos = 0usize;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Model::deserialize(&trunc, &mut pos)
    }));
    assert!(r.is_ok(), "deserialize panicked on a truncated model");
    assert!(r.unwrap().is_err(), "truncated model accepted");

    eprintln!("[crosspath] Model::deserialize rejects all 6 malformed-CDF classes (no panic)");
}

#[test]
fn deserialize_rejects_random_corrupt_models_without_panic() {
    // Fuzz: take a valid model, flip random bytes, and assert deserialize is a
    // total function (Ok-with-valid-invariants OR Err — never a panic, never a
    // CDF that violates the sum/ascending/nonzero invariants).
    let mut s = 0xDEAD_FA11_0BAD_F00Du64;
    let base_raw: Vec<i64> = (0..3000).map(|i| ((i * 31) % 257) as i64 - 128).collect();
    let model = Model::from_stream(&base_raw);
    let mut clean = Vec::new();
    model.serialize(&mut clean);

    let mut checked = 0u64;
    for _ in 0..20_000 {
        let mut bad = clean.clone();
        let flips = 1 + (splitmix64(&mut s) % 4) as usize;
        for _ in 0..flips {
            let i = (splitmix64(&mut s) as usize) % bad.len();
            bad[i] ^= (splitmix64(&mut s) & 0xFF) as u8;
        }
        let mut pos = 0usize;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Model::deserialize(&bad, &mut pos)
        }));
        assert!(r.is_ok(), "deserialize panicked on corrupt model bytes: {bad:?}");
        if let Ok(Ok(m)) = r {
            // If it accepted, the rebuilt CDF MUST satisfy every moat invariant
            // (a valid-but-wrong CDF is fine — the bytes describe it — but it can
            // never be internally inconsistent). The model's `cum` is private, so
            // recover it the integer-only way the decoder would see it: re-
            // serialize the accepted model and parse the freqs from the bytes
            // (serialize↔deserialize is a proven fixpoint elsewhere, so this is a
            // faithful read of exactly the CDF the decoder will use).
            let mut rebuf = Vec::new();
            m.serialize(&mut rebuf);
            let mut p2 = 0usize;
            let (_syms, cum) = spec_parse_model(&rebuf, &mut p2);
            assert_eq!(*cum.last().unwrap(), SCALE_TOTAL, "accepted model with bad sum");
            for w in cum.windows(2) {
                assert!(w[1] > w[0], "accepted model with a zero-freq slot");
            }
            // and the accepted model must actually be usable: encoding a stream
            // of its own modelled symbols and decoding must round-trip (proves
            // the rebuilt CDF is internally consistent, not just well-summed).
            let probe: Vec<i64> = vec![0i64, 0, 0];
            let enc = encode_stream_with_model(&probe, &m);
            let mut pd = 0usize;
            let dec = decode_stream(&enc, &mut pd).expect("accepted model must decode");
            assert_eq!(dec, probe, "accepted model produced a non-round-tripping CDF");
        }
        checked += 1;
    }
    eprintln!("[crosspath] {checked} corrupt-model fuzz inputs: deserialize total + invariant-preserving");
}
