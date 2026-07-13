//! Encode -> decode equivalence: HARDENING the determinism MOAT for the
//! trellis Viterbi *encode* side.
//!
//! The existing `exhaustive.rs` proves the three *decoders*
//! (`decode_tensor_fixed`, `decode_lean`, vec) agree on *hand-crafted*
//! bitstreams. `tests.rs::encode_decode_path_consistency` proves decode
//! matches the encoder's implied path — but only for ONE config (L=8,k=4,
//! scalar, no tail-biting, no affine).
//!
//! This file closes the gap. It proves, across a broad sweep of
//! (L, k, block_len, vec_dim) and every lever combination
//! (adaptive sub-scales / tail-biting / affine-min), that:
//!
//!   decode_tensor_fixed( encode(weights) )
//!     == an INDEPENDENT integer-only replay of the encoded bits + side info
//!
//! and that decode_lean and the vector decoder agree with it too.
//!
//! The replay below is a SECOND, deliberately-redundant integer implementation
//! of the wire contract (read symbols, fold into state, look up the frozen Q12
//! LUT, apply Q12 scale + affine offset). It does NOT call any production
//! decoder, so agreement means the encoder wrote bits + side-info that decode
//! to a bit-identical, float-free reconstruction on the canonical CPU path —
//! the property every device's decode must reproduce.
//!
//! Self-contained: no new dev-dependencies (matches the crate's "tests are
//! self-contained" policy). Deterministic enumeration, no RNG crate.

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{
    decode_lean, decode_lean_with_lut, decode_tensor, decode_tensor_fixed,
    decode_tensor_fixed_with_lut,
};
use strand_quant::encode::{
    encode_tensor, encode_tensor_with, encode_tensor_with_lut, n_sub_blocks, unpack_sub_scales,
    vector_lut_from_scalar, EncodeOpts, EncodedTensor, SUB_BLOCK,
};
use strand_quant::QUANTILE_SHIFT;
use strand_quant::TrellisConfig;

const SCALE_SHIFT: u32 = 16;
const SUB_SCALE_SHIFT: u32 = 6;

// ---- independent integer primitives (a second implementation of the wire) ----

fn ind_read_bits(bytes: &[u8], start_bit: usize, nbits: u32) -> usize {
    let mut v = 0usize;
    for i in 0..nbits as usize {
        let bit = start_bit + i;
        let byte = bit / 8;
        if byte < bytes.len() && (bytes[byte] >> (bit % 8)) & 1 == 1 {
            v |= 1 << i;
        }
    }
    v
}

fn ind_eff_scale_q(scale_q: i32, code: u8) -> i32 {
    // The 6-bit sub-scale code is a Q6 multiplier (mult = code+1, then >>6),
    // NOT the Q16 base scale.
    let mult = (code as i64 & 0x3F) + 1;
    (((scale_q as i64) * mult) >> SUB_SCALE_SHIFT) as i32
}

fn ind_eff_min_q(min_base_q: i32, code: u8) -> i32 {
    let mag = (code & 0x1F) as i64;
    if mag == 0 {
        return 0;
    }
    let base = (min_base_q.unsigned_abs()) as i64;
    let signed = if code & 0x20 != 0 { base * mag } else { -(base * mag) };
    (signed / 31) as i32
}

fn ind_reconstruct_q(scale_q: i32, q: i32) -> i32 {
    // NB: the product is shifted by SCALE_SHIFT (the scale's fixed-point
    // shift), NOT by QUANTILE_SHIFT. The Q12 lives in `q`; the scale is Q16.
    (((scale_q as i64) * (q as i64)) >> SCALE_SHIFT) as i32
}

/// Independent integer replay of the encoded tensor against a (possibly
/// vector) LUT. Mirrors the documented wire contract WITHOUT touching any
/// production decoder. `d` is the vector dimension (1 for scalar).
fn replay(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32], d: usize) -> Vec<i32> {
    let l = cfg.l_bits;
    let k = cfg.k_bits;
    let mask = (1usize << l) - 1;
    let imask = (1usize << k) - 1;

    let mut out = Vec::with_capacity(enc.total);
    let mut bit_cursor = 0usize;

    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_sub = n_sub_blocks(n);
        // An omitted scale stream is the canonical non-adaptive wire form.
        // Replay that contract independently: Q6 code 63 is exact unity.
        let scodes = if blk.sub_scales.is_empty() {
            vec![63; n_sub]
        } else {
            unpack_sub_scales(&blk.sub_scales, n_sub)
        };
        let eff: Vec<i32> = scodes.iter().map(|&c| ind_eff_scale_q(blk.scale_q, c)).collect();
        let offs: Vec<i32> = if enc.has_affine_min {
            let mcodes = unpack_sub_scales(&blk.mins, n_sub);
            mcodes.iter().map(|&c| ind_eff_min_q(blk.min_base_q, c)).collect()
        } else {
            Vec::new()
        };

        let n_steps = n.div_ceil(d);
        let nk = n_steps * k as usize;

        // Start state: either replay-from-bits (tail-biting) or the stored seed.
        let mut state = if enc.tail_biting && nk >= l as usize {
            let mut s = 0usize;
            let mut c = bit_cursor;
            for _ in 0..n_steps {
                let sym = ind_read_bits(&enc.bits, c, k) & imask;
                c += k as usize;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        let mut produced = 0usize;
        for _ in 0..n_steps {
            let sym = ind_read_bits(&enc.bits, bit_cursor, k) & imask;
            bit_cursor += k as usize;
            state = ((state << k) | sym) & mask;

            // Vector decode emits min(remaining, d) lanes from lut[state*d + j].
            let remaining = n - produced;
            let emit = remaining.min(d);
            let base = state * d;
            for j in 0..emit {
                let i = produced + j;
                let q = lut[base + j];
                let es = eff[i / SUB_BLOCK];
                let off = offs.get(i / SUB_BLOCK).copied().unwrap_or(0);
                out.push(ind_reconstruct_q(es, q) + off);
            }
            produced += emit;
        }
    }
    out
}

// ---- deterministic weight generators (no rng crate) ----

/// Pseudo-Gaussian-ish deterministic weights. Mixes a few sinusoids so the
/// Viterbi actually has structure to track; seeded for reproducibility.
fn gen_weights(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f64 + 1.0) * 0.0137 + seed as f64 * 0.731;
            let v = x.sin() * 0.6 + (x * 2.3 + 0.4).sin() * 0.3 + (x * 0.31).cos() * 0.1;
            (v as f32) * amp
        })
        .collect()
}

// ---- the equivalence core, run for one concrete (cfg, opts, weights) ----

fn assert_scalar_equivalence(weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts, ctx: &str) {
    let lut = codebook_lut(cfg.l_bits);

    let enc = encode_tensor_with(weights, cfg, opts);
    // basic structural invariants of the bitstream
    assert_eq!(enc.total, weights.len(), "total mismatch [{ctx}]");
    assert_eq!(enc.tail_biting, opts.tail_biting, "tail flag [{ctx}]");
    assert_eq!(enc.has_affine_min, opts.affine_min, "affine flag [{ctx}]");

    // 1) The independent integer replay is the ground truth recon.
    let expected = replay(&enc, cfg, lut, 1);
    assert_eq!(expected.len(), weights.len(), "replay length [{ctx}]");

    // 2) All production decoders must equal it, bit-for-bit.
    let fixed = decode_tensor_fixed(&enc, cfg);
    let lean = decode_lean(&enc, cfg);
    assert_eq!(fixed, expected, "decode_tensor_fixed != independent replay [{ctx}]");
    assert_eq!(lean, expected, "decode_lean != independent replay [{ctx}]");

    // 3) Explicit-LUT decoders agree (same path, different entry point).
    assert_eq!(
        decode_tensor_fixed_with_lut(&enc, cfg, lut),
        expected,
        "decode_tensor_fixed_with_lut drift [{ctx}]"
    );
    assert_eq!(
        decode_lean_with_lut(&enc, cfg, lut),
        expected,
        "decode_lean_with_lut drift [{ctx}]"
    );

    // 4) Encode is reproducible (same input -> same bits & side info).
    let enc2 = encode_tensor_with(weights, cfg, opts);
    assert_eq!(enc, enc2, "re-encode produced different bits [{ctx}]");

    // 5) Decode is reproducible.
    assert_eq!(decode_tensor_fixed(&enc, cfg), fixed, "re-decode drift [{ctx}]");

    // 6) f32 public wrapper is EXACTLY q * 2^-12 of the integer recon.
    let f = decode_tensor(&enc, cfg);
    let q12_to_f32 = 1.0f32 / (1u32 << QUANTILE_SHIFT) as f32;
    for (idx, (&q, &x)) in fixed.iter().zip(f.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            ((q as f32) * q12_to_f32).to_bits(),
            "f32 wrapper not exact at i={idx} [{ctx}]"
        );
    }
}

// =====================================================================
//  TEST 1: broad (L, k, block_len) sweep x all lever combinations,
//          scalar path, with the awkward-length boundary baked in.
// =====================================================================
#[test]
fn encode_decode_scalar_equivalence_sweep() {
    let mut cases = 0u64;
    // L sweep includes the frozen-table extremes (4) and a large state count (10),
    // plus 5/6/8 in between. k sweeps the full legal range 1..=4.
    for l in [4u32, 5, 6, 8, 10] {
        for k in 1u32..=4 {
            if k > l {
                continue;
            }
            // block_len includes 1 (degenerate), small, the canonical 256, and
            // values that are NOT multiples of SUB_BLOCK(32) to exercise the
            // sub-scale tail. The length set per block_len is bounded so tiny
            // block_lens (which fragment a long tensor into hundreds of
            // Viterbi'd blocks) don't blow up runtime — large `n` is paired
            // only with the larger block_lens, where it still crosses block,
            // sub-block(32), and short-final-block boundaries.
            for &(block_len, lengths) in &[
                (1usize, &[1usize, 2, 5, 33][..]),
                (7usize, &[1usize, 7, 8, 33][..]),
                (32usize, &[1usize, 31, 32, 33, 65][..]),
                (33usize, &[1usize, 32, 33, 34, 67][..]),
                (64usize, &[1usize, 63, 64, 65, 200][..]),
                (100usize, &[1usize, 99, 100, 257][..]),
                (256usize, &[1usize, 255, 256, 257, 600][..]),
            ] {
                let cfg = TrellisConfig::new(l, k, block_len);
                // Confirm the config realised what we asked (clamping caveats).
                assert_eq!(cfg.k_bits, k.clamp(1, TrellisConfig::MAX_K), "k clamp [{l},{k}]");
                assert!(cfg.l_bits >= cfg.k_bits, "L>=k invariant [{l},{k}]");
                assert_eq!(cfg.block_len, block_len.max(1), "block_len [{l},{k}]");

                // Lengths chosen to straddle block boundaries, sub-block (32)
                // boundaries, and produce short final blocks.
                for &n in lengths {
                    let weights = gen_weights(n, (l * 131 + k * 17 + n as u32) as u64, 0.45);
                    for &adaptive in &[true, false] {
                        for &tail_biting in &[false, true] {
                            for &affine_min in &[false, true] {
                                let opts = EncodeOpts {
                                    adaptive,
                                    tail_biting,
                                    affine_min,
                                    ..Default::default()
                                };
                                let ctx = format!(
                                    "L={l} k={k} bl={block_len} n={n} adapt={adaptive} tail={tail_biting} affine={affine_min}"
                                );
                                assert_scalar_equivalence(&weights, &cfg, &opts, &ctx);
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("encode->decode scalar equivalence: {cases} (cfg x lever x length) cases");
    assert!(cases > 2_500, "coverage unexpectedly small: {cases}");
}

// =====================================================================
//  TEST 2: vector (vec_dim) path. The decoder emits min(remaining, d)
//          lanes per step from lut[state*d + j]; the independent replay
//          must reproduce that, including the n % d != 0 short tail.
//          Uses the scalar codebook broadcast to d lanes so the encode
//          and decode share a frozen, float-free LUT.
// =====================================================================
#[test]
fn encode_decode_vector_equivalence_sweep() {
    let mut cases = 0u64;
    for l in [4u32, 6, 8] {
        for k in 1u32..=4 {
            if k > l {
                continue;
            }
            let scalar_lut = codebook_lut(l);
            for d in 1u32..=TrellisConfig::MAX_VEC_DIM {
                let cfg = TrellisConfig::new(l, k, 128).with_vec_dim(d);
                assert_eq!(cfg.vec_dim(), d as usize, "vec_dim clamp [{l},{k},{d}]");
                let vlut = vector_lut_from_scalar(scalar_lut, d as usize);
                assert_eq!(vlut.len(), cfg.num_states() * d as usize, "vlut shape");

                // Lengths chosen so n % d hits 0, 1, .., d-1 and crosses
                // block + sub-block boundaries.
                for &n in &[1usize, 3, 7, 32, 33, 100, 129, 257] {
                    let weights = gen_weights(n, (l * 977 + k * 41 + d * 7 + n as u32) as u64, 0.5);
                    for &adaptive in &[true, false] {
                        for &tail_biting in &[false, true] {
                            for &affine_min in &[false, true] {
                                let opts = EncodeOpts {
                                    adaptive,
                                    tail_biting,
                                    affine_min,
                                    ..Default::default()
                                };
                                let enc =
                                    encode_tensor_with_lut(&weights, &cfg, &opts, &vlut);
                                assert_eq!(enc.total, n, "vec total");

                                let expected = replay(&enc, &cfg, &vlut, d as usize);
                                assert_eq!(expected.len(), n, "vec replay length");

                                // The vec decode path is reached via the
                                // *_with_lut entry points when vec_dim > 1.
                                let fixed =
                                    decode_tensor_fixed_with_lut(&enc, &cfg, &vlut);
                                let lean = decode_lean_with_lut(&enc, &cfg, &vlut);
                                let ctx = format!(
                                    "VEC L={l} k={k} d={d} n={n} adapt={adaptive} tail={tail_biting} affine={affine_min}"
                                );
                                assert_eq!(
                                    fixed, expected,
                                    "vec decode_tensor_fixed_with_lut != replay [{ctx}]"
                                );
                                assert_eq!(
                                    lean, expected,
                                    "vec decode_lean_with_lut != replay [{ctx}]"
                                );

                                // Reproducible encode + decode.
                                let enc2 =
                                    encode_tensor_with_lut(&weights, &cfg, &opts, &vlut);
                                assert_eq!(enc, enc2, "vec re-encode drift [{ctx}]");
                                assert_eq!(
                                    decode_tensor_fixed_with_lut(&enc, &cfg, &vlut),
                                    fixed,
                                    "vec re-decode drift [{ctx}]"
                                );

                                // At d=1 the vector LUT must agree with the
                                // pure-scalar pipeline byte-for-byte.
                                if d == 1 {
                                    let enc_s = encode_tensor_with(&weights, &cfg, &opts);
                                    assert_eq!(
                                        enc, enc_s,
                                        "d=1 vec encode != scalar encode [{ctx}]"
                                    );
                                    assert_eq!(
                                        fixed,
                                        decode_tensor_fixed(&enc_s, &cfg),
                                        "d=1 vec decode != scalar decode [{ctx}]"
                                    );
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("encode->decode vector equivalence: {cases} cases");
    assert!(cases > 1_000, "vec coverage unexpectedly small: {cases}");
}

// =====================================================================
//  TEST 3: tail-biting init-state INDEPENDENCE.
//
//  For a tail-bitten block the decoder DERIVES the start state from the
//  payload bits and ignores BlockMeta.init_state. So mutating the stored
//  init_state of every block must NOT change the decode. This guards the
//  encode->decode contract against a class of bugs where decode silently
//  falls back to the stored seed.
// =====================================================================
#[test]
fn tail_biting_decode_ignores_stored_init_state() {
    let mut checked = 0u64;
    for l in [4u32, 5, 8] {
        for k in 1u32..=4 {
            if k > l {
                continue;
            }
            // block_len * k >= L guarantees a *full* block is tail-bitten;
            // choosing n as an exact multiple of block_len means EVERY block
            // is full (no short final block to fall back to its stored seed).
            let block_len = 64usize;
            assert!(block_len * (k as usize) >= l as usize);
            for &nblk in &[1usize, 2, 4] {
                let n = nblk * block_len;
                let cfg = TrellisConfig::new(l, k, block_len);
                let weights = gen_weights(n, (l * 17 + k * 3 + n as u32) as u64, 0.4);
                let opts = EncodeOpts { tail_biting: true, ..Default::default() };
                let enc = encode_tensor_with(&weights, &cfg, &opts);
                let base = decode_tensor_fixed(&enc, &cfg);

                // Verify the *premise*: every block here is genuinely tail-bitten
                // (n_steps*k >= L), otherwise this test would be vacuous.
                for blk in &enc.blocks {
                    let n_steps = cfg.num_steps(blk.n as usize);
                    assert!(
                        n_steps * (k as usize) >= l as usize,
                        "block not tail-bitten; test would be vacuous (L={l},k={k},n={n})"
                    );
                }

                // Corrupt every stored init_state and confirm decode is unchanged.
                let mut tampered = enc.clone();
                for blk in &mut tampered.blocks {
                    blk.init_state = blk.init_state.wrapping_add(0x5A5A_5A5A);
                }
                assert_eq!(
                    decode_tensor_fixed(&tampered, &cfg),
                    base,
                    "tail-bitten decode depends on stored init_state (L={l},k={k},n={n})"
                );
                assert_eq!(
                    decode_lean(&tampered, &cfg),
                    base,
                    "tail-bitten decode_lean depends on stored init_state (L={l},k={k},n={n})"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    eprintln!("tail-biting init-state independence: {checked} configs");
}

// =====================================================================
//  TEST 4: the f32-metric ENCODE path (STRAND_F32_METRIC=1) produces bits
//          that decode identically through the integer-deterministic
//          decoder. The metric used during *search* is allowed to be f32,
//          but the *bits it emits* must still decode float-free and match
//          the independent integer replay. This run is single-threaded and
//          sets/clears the env var around itself.
//
//  NOTE: this test mutates a process-global env var. It is `#[ignore]` by
//  default so it never races the other tests under the default multi-thread
//  runner. Run explicitly with:
//      STRAND_NO_GPU=1 cargo test -p strand-quant --test encode_decode_equivalence \
//          -- --ignored f32_metric_encode --test-threads=1
// =====================================================================
#[test]
#[ignore = "mutates global env (STRAND_F32_METRIC); run single-threaded explicitly"]
fn f32_metric_encode_decodes_float_free() {
    std::env::set_var("STRAND_NO_GPU", "1");
    std::env::set_var("STRAND_F32_METRIC", "1");

    let mut cases = 0u64;
    for l in [4u32, 6, 8] {
        for k in 1u32..=4 {
            if k > l {
                continue;
            }
            let cfg = TrellisConfig::new(l, k, 128);
            for &n in &[1usize, 33, 200, 257] {
                let weights = gen_weights(n, (l * 53 + k * 11 + n as u32) as u64, 0.5);
                for &tail_biting in &[false, true] {
                    let opts = EncodeOpts { tail_biting, ..Default::default() };
                    let enc = encode_tensor_with(&weights, &cfg, &opts);
                    let lut = codebook_lut(l);
                    let expected = replay(&enc, &cfg, lut, 1);
                    assert_eq!(
                        decode_tensor_fixed(&enc, &cfg),
                        expected,
                        "f32-metric encode did not decode float-free (L={l},k={k},n={n},tail={tail_biting})"
                    );
                    assert_eq!(
                        decode_lean(&enc, &cfg),
                        expected,
                        "f32-metric decode_lean drift (L={l},k={k},n={n},tail={tail_biting})"
                    );
                    cases += 1;
                }
            }
        }
    }

    std::env::remove_var("STRAND_F32_METRIC");
    std::env::remove_var("STRAND_NO_GPU");
    assert!(cases > 0);
    eprintln!("f32-metric encode->decode: {cases} cases");
}

// =====================================================================
//  TEST 5: empty + all-zero degenerate inputs round-trip cleanly.
// =====================================================================
#[test]
fn degenerate_inputs_round_trip() {
    for l in [4u32, 8] {
        for k in 1u32..=4 {
            if k > l {
                continue;
            }
            let cfg = TrellisConfig::new(l, k, 64);

            // Empty.
            let enc = encode_tensor(&[], &cfg);
            assert_eq!(enc.total, 0);
            assert!(decode_tensor_fixed(&enc, &cfg).is_empty());
            assert!(decode_lean(&enc, &cfg).is_empty());

            // All zeros (exercises the absmax==0 scale_q==0 path -> all recon 0).
            for &n in &[1usize, 33, 200] {
                let zeros = vec![0.0f32; n];
                for &affine in &[false, true] {
                    let opts = EncodeOpts { affine_min: affine, ..Default::default() };
                    let enc = encode_tensor_with(&zeros, &cfg, &opts);
                    let expected = replay(&enc, &cfg, codebook_lut(l), 1);
                    let got = decode_tensor_fixed(&enc, &cfg);
                    assert_eq!(got, expected, "zeros replay mismatch L={l} k={k} n={n}");
                    assert!(
                        got.iter().all(|&q| q == 0),
                        "all-zero input did not decode to all-zero (L={l},k={k},n={n})"
                    );
                }
            }
        }
    }
}
