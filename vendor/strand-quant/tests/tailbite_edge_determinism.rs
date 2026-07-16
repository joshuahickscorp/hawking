//! Determinism hardening for the tail-biting + init-state decode surface.
//!
//! THE MOAT: STRAND decode is bit-identical on every device — a frozen integer
//! Q12 LUT, float-free reconstruction. This file pins the *edge cases* of that
//! guarantee for the tail-biting / init-state machinery, which `exhaustive.rs`
//! does not reach:
//!
//!   * EMPTY blocks (n = 0) — `exhaustive.rs` starts at n = 1. An empty block
//!     appends no output and consumes no payload bits; the two production
//!     decoders track the bit cursor by *different* means
//!     (`decode_tensor_fixed` carries a running `bit_cursor`; `decode_lean`
//!     recomputes it as `out.len() * k`), so an empty block — especially one
//!     interleaved between real blocks — is precisely where those two
//!     bookkeeping schemes can drift apart.
//!   * The exact tail-biting <-> init-state SWITCH boundary: the decoders use
//!     `tail_biting && n*k >= l_bits` to choose between deriving the start state
//!     from the payload (tail-biting) and trusting the stored `init_state`. We
//!     enumerate n*k = l-1, l, l+1 so the seam itself is covered on both sides.
//!   * NON-32-aligned block widths (the sub-scale block is 32): 31/33/63/65/895/
//!     897 give a partial trailing sub-block, plus a 896-wide block (the dim
//!     Q4_K cannot hit uniformly — a STRAND niche) and genuinely HUGE blocks.
//!   * ALL-ZERO and ALL-OUTLIER symbol streams against extreme scales
//!     (i32::MIN/MAX, 0), where reconstruction hits its i32 clamp corners.
//!   * The init-state DETERMINISM property itself: when a block is tail-bitten,
//!     the decode MUST NOT depend on the stored `init_state`. Encoders on
//!     different devices may park different junk there; the decode is identical.
//!
//! Every test asserts a THREE-WAY equality — `decode_lean` == `decode_tensor_fixed`
//! == an independent spec reference re-implemented here from the wire format —
//! exactly as `exhaustive.rs` does, so a regression in any single path is caught
//! against two others.
//!
//! Run: `cargo test -p strand-quant --test tailbite_edge_determinism`

// Test ergonomics: boxed-closure tables and `&[x.clone()]` literals keep the
// edge-case generators readable; these lints are not load-bearing here.
#![allow(clippy::type_complexity, clippy::redundant_closure)]

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{decode_lean, decode_tensor, decode_tensor_fixed};
use strand_quant::encode::{pack_sub_scales, BlockMeta, EncodedTensor};
use strand_quant::TrellisConfig;

// ---------------------------------------------------------------------------
// Independent spec reference (re-derived from the wire layout, not shared with
// the production decoders). Mirrors exhaustive.rs::ref_decode so this file is a
// third, self-contained oracle.
// ---------------------------------------------------------------------------

fn ref_read_bits(bytes: &[u8], start_bit: usize, nbits: u32) -> usize {
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

fn ref_unpack6(bytes: &[u8], n: usize) -> Vec<u8> {
    (0..n).map(|i| ref_read_bits(bytes, i * 6, 6) as u8).collect()
}

fn ref_decode(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    let l = cfg.l_bits;
    let k = cfg.k_bits;
    let mask = (1usize << l) - 1;
    let imask = (1usize << k) - 1;
    let mut out = Vec::with_capacity(enc.total);
    let mut cursor = 0usize;

    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_sub = n.div_ceil(32);
        let scodes = ref_unpack6(&blk.sub_scales, n_sub);
        let mcodes: Vec<u8> = if enc.has_affine_min { ref_unpack6(&blk.mins, n_sub) } else { Vec::new() };

        // Tail-biting start state: only when n*k >= l_bits, else trust init_state.
        let mut state = if enc.tail_biting && n * k as usize >= l as usize {
            let mut s = 0usize;
            for i in 0..n {
                s = ((s << k) | (ref_read_bits(&enc.bits, cursor + i * k as usize, k) & imask)) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        for i in 0..n {
            let sym = ref_read_bits(&enc.bits, cursor, k) & imask;
            cursor += k as usize;
            state = ((state << k) | sym) & mask;
            let q = lut[state] as i64;
            let scode = (scodes[i / 32] & 0x3F) as i64;
            let es = (blk.scale_q as i64 * (scode + 1)) >> 6;
            let recon = ((es * q) >> 16) as i32;
            let off = if enc.has_affine_min {
                let c = mcodes[i / 32];
                let mag = (c & 0x1F) as i64;
                if mag == 0 {
                    0i32
                } else {
                    let base = (blk.min_base_q.unsigned_abs()) as i64;
                    let s = if c & 0x20 != 0 { base * mag } else { -(base * mag) };
                    (s / 31) as i32
                }
            } else {
                0
            };
            out.push(recon + off);
        }
    }
    out
}

fn pack_symbols(syms: &[usize], k: u32) -> Vec<u8> {
    let total_bits = syms.len() * k as usize;
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut cursor = 0usize;
    for &s in syms {
        for b in 0..k as usize {
            if (s >> b) & 1 == 1 {
                bytes[cursor / 8] |= 1 << (cursor % 8);
            }
            cursor += 1;
        }
    }
    bytes
}

/// Build an `EncodedTensor` from explicit per-block symbol streams. Each block's
/// init_state / scale / sub-scale codes / affine min codes are caller-supplied so
/// adversarial edge values can be injected directly onto the wire.
#[allow(clippy::too_many_arguments)]
fn make_tensor(block_syms: &[Vec<usize>], k: u32, init_states: &[u32], scale_qs: &[i32], sub_codes: &[Vec<u8>], tail_biting: bool, affine: Option<(&[i32], &[Vec<u8>])>) -> EncodedTensor {
    let all_syms: Vec<usize> = block_syms.iter().flatten().copied().collect();
    let bits = pack_symbols(&all_syms, k);
    let mut blocks = Vec::new();
    let mut total = 0usize;
    for (b, syms) in block_syms.iter().enumerate() {
        let n = syms.len();
        total += n;
        let (min_base_q, mins) = match affine {
            Some((bases, codes)) => (bases[b], pack_sub_scales(&codes[b])),
            None => (0, Vec::new()),
        };
        blocks.push(BlockMeta { scale_q: scale_qs[b], sub_scales: pack_sub_scales(&sub_codes[b]), min_base_q, mins, init_state: init_states[b], n: n as u32 });
    }
    EncodedTensor { bits, blocks, total, has_rht_seed: false, tail_biting, has_affine_min: affine.is_some() }
}

/// The MOAT assertion: the three decoders agree bit-for-bit, AND the f32 wrapper
/// is exactly the Q12 fixed-point divided by 4096 (no float drift in decode).
fn assert_three_way(enc: &EncodedTensor, cfg: &TrellisConfig, ctx: &str) {
    let lut = codebook_lut(cfg.l_bits);
    let reference = ref_decode(enc, cfg, lut);
    let lean = decode_lean(enc, cfg);
    let fixed = decode_tensor_fixed(enc, cfg);
    assert_eq!(lean, reference, "decode_lean != spec reference [{ctx}]");
    assert_eq!(fixed, reference, "decode_tensor_fixed != spec reference [{ctx}]");
    // f32 wrapper must be the exact Q12 -> f32 cast (the float-free promise).
    let f = decode_tensor(enc, cfg);
    assert_eq!(f.len(), fixed.len(), "f32 length mismatch [{ctx}]");
    for (a, b) in fixed.iter().zip(f.iter()) {
        assert_eq!(*b, (*a as f32) * (1.0 / 4096.0), "f32 wrapper drift [{ctx}]");
    }
}

/// Helper to build a per-block sub-scale code vector of the right length.
fn unity_subs(n: usize) -> Vec<u8> {
    vec![63u8; n.div_ceil(32).max(1)]
}

/// All four production (L,k) corners that exhaustive.rs uses, plus the bpw
/// configs the real decoder ships with.
const LK: [(u32, u32); 4] = [(4, 2), (4, 3), (5, 2), (5, 3)];

/// Distinguished scales: identity, fractional, negatives, zero, i32 corners.
const SCALES: [i32; 8] = [1 << 16, 4096, -(1 << 16), 1, -1, 0, i32::MAX, i32::MIN];

// ===========================================================================
// EDGE 1 — empty blocks (n = 0), alone and INTERLEAVED with real blocks.
// This is the cursor-bookkeeping divergence trap: decode_tensor_fixed advances a
// running bit_cursor; decode_lean recomputes the start-state cursor from
// out.len()*k and reads symbols through an independent WordBitReader. An empty
// block must leave both schemes in lockstep, and must not consume payload.
// ===========================================================================

#[test]
fn empty_blocks_alone_and_interleaved() {
    let mut covered = 0u64;
    for (l, k) in LK {
        let cfg = TrellisConfig::new(l, k, 256);
        let imask = (1usize << k) - 1;
        for &tail in &[false, true] {
            for &affine in &[false, true] {
                // A small zoo of block-length layouts, each with empty blocks in
                // leading / middle / trailing / adjacent positions.
                let layouts: &[&[usize]] = &[&[0], &[0, 0], &[0, 0, 0], &[0, 5], &[5, 0], &[5, 0, 7], &[0, 5, 0], &[0, 0, 9, 0, 0], &[33, 0, 1, 0, 64], &[0, 256, 0, 256, 0]];
                for layout in layouts {
                    let mut block_syms = Vec::new();
                    let mut inits = Vec::new();
                    let mut scales = Vec::new();
                    let mut subs: Vec<Vec<u8>> = Vec::new();
                    let mut bases = Vec::new();
                    let mut minc: Vec<Vec<u8>> = Vec::new();
                    for (b, &n) in layout.iter().enumerate() {
                        let syms: Vec<usize> = (0..n).map(|i| ((i + b * 131).wrapping_mul(2654435761) >> 11) & imask).collect();
                        block_syms.push(syms);
                        // adversarial init_state: large, irrelevant if tail-bitten
                        inits.push(((b as u32).wrapping_mul(0x9E37_79B9)) | 0x8000_0001);
                        scales.push(SCALES[(b + n) % SCALES.len()]);
                        subs.push(unity_subs(n));
                        bases.push([0i32, 4096, 1 << 18][b % 3]);
                        minc.push(vec![((b * 7 + 17) % 64) as u8; n.div_ceil(32).max(1)]);
                    }
                    let aff = if affine { Some((&bases[..], &minc[..])) } else { None };
                    let enc = make_tensor(&block_syms, k, &inits, &scales, &subs, tail, aff);
                    // total must equal the sum of block lengths (no phantom output)
                    assert_eq!(enc.total, layout.iter().sum::<usize>());
                    assert_three_way(&enc, &cfg, &format!("empty L={l} k={k} tail={tail} affine={affine} layout={layout:?}"));
                    covered += 1;
                }
            }
        }
    }
    eprintln!("empty-block layouts: {covered} tensors");
}

// ===========================================================================
// EDGE 2 — the tail-biting <-> init-state SWITCH boundary, EXHAUSTIVE over the
// symbol stream right at n*k in {l-1, l, l+1}. Below the threshold a tail_biting
// tensor must fall back to init_state (so output DOES depend on init_state); at
// or above it the output must be independent of init_state. We assert both.
// ===========================================================================

#[test]
fn switch_boundary_exhaustive_both_sides() {
    let mut covered = 0u64;
    for (l, k) in LK {
        let cfg = TrellisConfig::new(l, k, 256);
        let n_states = 1usize << l;
        let imask = (1usize << k) - 1;
        // n values whose n*k brackets l_bits from just-below to just-above.
        // (n*k can't always equal l exactly when k does not divide l; we take the
        //  n's that straddle the threshold so both branches are exercised.)
        let lk = l as usize;
        let n_lo = lk.saturating_sub(1) / k as usize; // largest n with n*k <  l (mostly)
        let n_hi = lk.div_ceil(k as usize) + 1; // smallest n with n*k >= l, plus one
        for n in n_lo..=n_hi {
            if n == 0 {
                continue;
            }
            let nk = n * k as usize;
            let below = nk < lk; // init-state branch even when tail_biting
            let n_streams = 1usize << nk;
            // Keep the enumeration bounded: cap at 2^14 streams.
            if n_streams > (1 << 14) {
                continue;
            }
            for stream in 0..n_streams {
                let syms: Vec<usize> = (0..n).map(|i| (stream >> (i * k as usize)) & imask).collect();
                let scale = SCALES[(stream + n) % SCALES.len()];

                if below {
                    // tail_biting=true but below threshold => must behave exactly
                    // like a non-tail-biting block keyed on init_state. For EVERY
                    // init: three-way holds, AND a plain (tail_biting=false) tensor
                    // with the same init decodes byte-identically (the two branches
                    // provably coincide below the threshold).
                    for init in 0..n_states {
                        let enc = make_tensor(std::slice::from_ref(&syms), k, &[init as u32], &[scale], &[unity_subs(n)], true, None);
                        assert_three_way(&enc, &cfg, &format!("switch-below L={l} k={k} n={n} init={init} stream={stream}"));
                        // Cross-check: a non-tail-biting tensor with the same init
                        // must decode identically (below threshold the branches
                        // coincide).
                        let enc_plain = make_tensor(std::slice::from_ref(&syms), k, &[init as u32], &[scale], &[unity_subs(n)], false, None);
                        assert_eq!(
                            decode_lean(&enc, &cfg),
                            decode_lean(&enc_plain, &cfg),
                            "below-threshold tail_biting must equal plain init_state \
                             (L={l} k={k} n={n} init={init} stream={stream})"
                        );
                        covered += 1;
                    }
                } else {
                    // At/above threshold: output is INDEPENDENT of stored init_state.
                    // Drive init_state with the full range plus garbage and require a
                    // single fixed output, three-way verified.
                    let inits = [0u32, (n_states - 1) as u32, (n_states / 2) as u32, 0xFFFF_FFFF, 0x8000_0000, 0xDEAD_BEEF];
                    let mut first: Option<Vec<i32>> = None;
                    for &init in &inits {
                        let enc = make_tensor(std::slice::from_ref(&syms), k, &[init], &[scale], &[unity_subs(n)], true, None);
                        assert_three_way(&enc, &cfg, &format!("switch-above L={l} k={k} n={n} init={init} stream={stream}"));
                        let out = decode_lean(&enc, &cfg);
                        match &first {
                            None => first = Some(out),
                            Some(f) => assert_eq!(
                                &out, f,
                                "tail-bitten output depends on stored init_state — MOAT \
                                 BREACH (L={l} k={k} n={n} stream={stream} init={init})"
                            ),
                        }
                        covered += 1;
                    }
                }
            }
        }
    }
    eprintln!("switch-boundary tensors: {covered}");
}

// ===========================================================================
// EDGE 3 — non-32-aligned block widths (partial trailing sub-scale block), the
// 896-wide STRAND niche dim, and HUGE single blocks. Real symbol streams driven
// pseudo-randomly; full three-way + tail/affine cross-product.
// ===========================================================================

#[test]
fn unaligned_and_huge_block_widths() {
    let mut covered = 0u64;
    // 31/33/63/65 straddle one and two sub-blocks; 895/896/897 are the niche dim
    // and its neighbours; 1024/4096 are "huge"; 257/8191 are prime-ish odd sizes.
    let widths = [1usize, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 257, 511, 895, 896, 897, 1024, 4096, 8191];
    for (l, k) in LK {
        let cfg = TrellisConfig::new(l, k, 256);
        let imask = (1usize << k) - 1;
        for &tail in &[false, true] {
            for &affine in &[false, true] {
                for &n in &widths {
                    let syms: Vec<usize> = (0..n).map(|i| (i.wrapping_mul(2654435761) >> 9) & imask).collect();
                    let n_sub = n.div_ceil(32);
                    // distinct sub-scale code per sub-block (partial last one included)
                    let subc: Vec<u8> = (0..n_sub).map(|s| ((s * 13 + 1) % 64) as u8).collect();
                    let minc: Vec<u8> = (0..n_sub).map(|s| ((s * 29 + 5) % 64) as u8).collect();
                    let scale = SCALES[(n + l as usize) % SCALES.len()];
                    let bases_i32 = [1i32 << 18];
                    let aff = if affine { Some((&bases_i32[..], std::slice::from_ref(&minc))) } else { None };
                    let enc = make_tensor(std::slice::from_ref(&syms), k, &[0x1234_5678], &[scale], std::slice::from_ref(&subc), tail, aff);
                    assert_three_way(&enc, &cfg, &format!("width L={l} k={k} tail={tail} affine={affine} n={n}"));
                    covered += 1;
                }
            }
        }
    }
    eprintln!("unaligned/huge widths: {covered} tensors");
}

// ===========================================================================
// EDGE 4 — all-zero and all-outlier symbol streams against extreme scales. These
// drive the reconstruction to its i32 clamp corners (i32::MIN/MAX * Q_CLAMP) and
// pin that the integer arithmetic does not wrap differently across the three
// decoders. "All-zero" = every symbol 0 (state walks toward state 0); "all-
// outlier" = every symbol the max input (state walks toward the top of the LUT).
// ===========================================================================

#[test]
fn all_zero_and_all_outlier_streams() {
    let mut covered = 0u64;
    for (l, k) in LK {
        let cfg = TrellisConfig::new(l, k, 256);
        let max_sym = (1usize << k) - 1;
        // n chosen to exceed the tail-biting threshold for every (l,k) here.
        let widths = [1usize, 8, 32, 33, 256, 896, 1024];
        let streams: [(&str, Box<dyn Fn(usize) -> usize>); 4] = [
            ("all-zero", Box::new(|_| 0usize)),
            ("all-max", Box::new(move |_| max_sym)),
            ("alt-0-max", Box::new(move |i| if i % 2 == 0 { 0 } else { max_sym })),
            ("ramp", Box::new(move |i| i & max_sym)),
        ];
        for (sname, sf) in &streams {
            for &n in &widths {
                let syms: Vec<usize> = (0..n).map(|i| sf(i)).collect();
                let n_sub = n.div_ceil(32);
                for &scale in &SCALES {
                    for &tail in &[false, true] {
                        // sub-scale codes: include 0 (eff_scale_q with code 0 => x1),
                        // 63 (unity x1 too at SUB_SCALE_SHIFT=6 -> (x*64)>>6=x), and a
                        // mid code; we cycle them across sub-blocks.
                        let subc: Vec<u8> = (0..n_sub).map(|s| [0u8, 63, 1, 31][s % 4]).collect();
                        let enc = make_tensor(std::slice::from_ref(&syms), k, &[0xABCD_1234], &[scale], std::slice::from_ref(&subc), tail, None);
                        assert_three_way(&enc, &cfg, &format!("extreme L={l} k={k} stream={sname} n={n} scale={scale} tail={tail}"));
                        covered += 1;
                    }
                }
            }
        }
    }
    eprintln!("all-zero/all-outlier extreme tensors: {covered}");
}

// ===========================================================================
// EDGE 5 — affine-min extremes with the largest safe base, on tail-bitten blocks.
// recon + offset must stay within i32 (the encoder's safe-base ledger is proved
// in proofs.rs; here we confirm the *decoders agree* at that boundary on the
// edge-case widths, including a partial last sub-block carrying a max min-code).
// ===========================================================================

#[test]
fn affine_min_extremes_on_edge_widths() {
    let cfg = TrellisConfig::new(5, 3, 256);
    let k = 3u32;
    let imask = (1usize << k) - 1;
    // max safe base from proofs.rs recon_plus_min_offset_add_bound.
    let max_safe_base: i32 = 1_342_177_279;
    let bases = [0i32, 1, 31, 4096, 1 << 20, max_safe_base];
    let widths = [1usize, 31, 32, 33, 65, 896, 897];
    let mut covered = 0u64;
    for &base in &bases {
        for &n in &widths {
            let syms: Vec<usize> = (0..n).map(|i| (i.wrapping_mul(40503) >> 3) & imask).collect();
            let n_sub = n.div_ceil(32);
            // min-codes hit both signs (0..31 negative side, 32..63 positive) and
            // the magnitude extremes 0 and 31.
            let minc: Vec<u8> = (0..n_sub).map(|s| [0u8, 0x1F, 0x20, 0x3F, 17, 48][s % 6]).collect();
            let subc: Vec<u8> = (0..n_sub).map(|s| ((s * 7 + 9) % 64) as u8).collect();
            for &tail in &[false, true] {
                for &scale in &[1i32 << 16, i32::MAX, 0, -(1 << 16)] {
                    let enc =
                        make_tensor(std::slice::from_ref(&syms), k, &[0xFEED_0001], &[scale], std::slice::from_ref(&subc), tail, Some((std::slice::from_ref(&base), std::slice::from_ref(&minc))));
                    assert_three_way(&enc, &cfg, &format!("affine-edge base={base} n={n} tail={tail} scale={scale}"));
                    covered += 1;
                }
            }
        }
    }
    eprintln!("affine-min edge tensors: {covered}");
}

// ===========================================================================
// EDGE 6 — init_state determinism stress: a SINGLE tail-bitten payload, decoded
// under EVERY representable init_state (full 2^l sweep for small l) plus 32-bit
// garbage, must yield ONE fixed output. This is the device-portability guarantee
// stated bluntly: the stored init_state field is decode-irrelevant once a block
// tail-bites, so two encoders that disagree on it still decode identically.
// ===========================================================================

#[test]
fn tail_biting_ignores_init_state_full_sweep() {
    let mut covered = 0u64;
    for (l, k) in LK {
        let cfg = TrellisConfig::new(l, k, 256);
        let n_states = 1usize << l;
        let imask = (1usize << k) - 1;
        // n large enough to tail-bite for all (l,k) here, plus a couple of widths.
        for &n in &[8usize, 33, 256] {
            // a few representative payloads
            for seed in [1u64, 7, 1234567, 0xFFFF_FFFF] {
                let syms: Vec<usize> = (0..n).map(|i| (((i as u64).wrapping_add(seed)).wrapping_mul(2654435761) >> 13) as usize & imask).collect();
                let scale = 1i32 << 16;
                let mut canonical: Option<Vec<i32>> = None;
                // full init sweep + 32-bit garbage values
                let garbage = [0xFFFF_FFFFu32, 0x8000_0000, 0xDEAD_BEEF, 0x0BAD_F00D];
                let all_inits = (0..n_states as u32).chain(garbage);
                for init in all_inits {
                    let enc = make_tensor(std::slice::from_ref(&syms), k, &[init], &[scale], &[unity_subs(n)], true, None);
                    // three-way each time too
                    let lut = codebook_lut(cfg.l_bits);
                    let reference = ref_decode(&enc, &cfg, lut);
                    let lean = decode_lean(&enc, &cfg);
                    let fixed = decode_tensor_fixed(&enc, &cfg);
                    assert_eq!(lean, reference, "lean!=ref L={l} k={k} n={n} init={init}");
                    assert_eq!(fixed, reference, "fixed!=ref L={l} k={k} n={n} init={init}");
                    match &canonical {
                        None => canonical = Some(lean),
                        Some(c) => assert_eq!(
                            &lean, c,
                            "init_state {init} changed a tail-bitten decode — MOAT BREACH \
                             (L={l} k={k} n={n} seed={seed})"
                        ),
                    }
                    covered += 1;
                }
            }
        }
    }
    eprintln!("init-state determinism sweep: {covered} (decode, all collapsed per group)");
}

// ===========================================================================
// CONTROL — proves the init-state invariants above are NON-VACUOUS: below the
// tail-biting threshold the stored init_state genuinely changes the decode (so
// EDGE 2's below-branch is meaningful), while at/above it the decode is fixed.
// If a refactor ever made init_state always-irrelevant (or always-relevant),
// one of these two assertions fails — catching a silently weakened guarantee.
// ===========================================================================

#[test]
fn init_state_control_below_changes_above_fixed() {
    // L=4,k=2 => threshold is n*k >= 4, i.e. n >= 2.
    let cfg = TrellisConfig::new(4, 2, 256);

    // n=1 (nk=2 < 4): init-state branch. Two inits must be able to differ.
    let below = vec![1usize];
    let br = std::slice::from_ref(&below);
    let a = decode_lean(&make_tensor(br, 2, &[0], &[1 << 16], &[unity_subs(1)], true, None), &cfg);
    let b = decode_lean(&make_tensor(br, 2, &[15], &[1 << 16], &[unity_subs(1)], true, None), &cfg);
    assert_ne!(
        a, b,
        "below-threshold init_state had no effect — the init-state branch is dead, \
         so EDGE 2's below-branch and the whole switch would be vacuous"
    );

    // n=4 (nk=8 >= 4): tail-biting branch. init0, init15, and 32-bit garbage must
    // all collapse to one output.
    let above = vec![1usize, 2, 3, 0];
    let ar = std::slice::from_ref(&above);
    let c = decode_lean(&make_tensor(ar, 2, &[0], &[1 << 16], &[unity_subs(4)], true, None), &cfg);
    let d = decode_lean(&make_tensor(ar, 2, &[15], &[1 << 16], &[unity_subs(4)], true, None), &cfg);
    let e = decode_lean(&make_tensor(ar, 2, &[0xDEAD_BEEF], &[1 << 16], &[unity_subs(4)], true, None), &cfg);
    assert_eq!(c, d, "above-threshold tail-bite not init-independent");
    assert_eq!(d, e, "above-threshold tail-bite not garbage-init-independent");
}

// ===========================================================================
// EDGE 7 — many-empty-and-real blocks in one tensor with adaptive sub-scales,
// covering the cursor accounting across a long chain where empty blocks sit at
// every junction. This is the "huge tensor with holes" composite case.
// ===========================================================================

#[test]
fn long_chain_with_holes_and_adaptive_subscales() {
    let mut covered = 0u64;
    for (l, k) in LK {
        let cfg = TrellisConfig::new(l, k, 256);
        let imask = (1usize << k) - 1;
        for &tail in &[false, true] {
            // Block lengths: alternate real/empty, with assorted non-aligned sizes.
            let lens: Vec<usize> = vec![0, 1, 0, 32, 0, 33, 0, 256, 0, 895, 0, 896, 0, 257, 0, 0, 64, 0];
            let mut block_syms = Vec::new();
            let mut inits = Vec::new();
            let mut scales = Vec::new();
            let mut subs: Vec<Vec<u8>> = Vec::new();
            for (b, &n) in lens.iter().enumerate() {
                let syms: Vec<usize> = (0..n).map(|i| ((i + b * 977).wrapping_mul(2654435761) >> 7) & imask).collect();
                block_syms.push(syms);
                inits.push(((b as u32).wrapping_mul(0x85EB_CA77)) | 1);
                scales.push(SCALES[(b * 3 + l as usize) % SCALES.len()]);
                let n_sub = n.div_ceil(32).max(1);
                subs.push((0..n_sub).map(|s| ((s * 5 + b * 3) % 64) as u8).collect());
            }
            let enc = make_tensor(&block_syms, k, &inits, &scales, &subs, tail, None);
            assert_eq!(enc.total, lens.iter().sum::<usize>());
            assert_three_way(&enc, &cfg, &format!("longchain L={l} k={k} tail={tail}"));
            covered += 1;
        }
    }
    eprintln!("long-chain-with-holes tensors: {covered}");
}
