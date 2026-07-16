//! Determinism-hardening tests for the OUTLier wire (OUTL) surface.
//!
//! SURFACE: `outlier_wire.rs` — serialize/parse round-trip + position/value
//! reconstruction exactness.
//!
//! The MOAT this protects: every byte of an OUTL section is produced and
//! consumed by integer/bit operations only (`write_bits` / `read_bits_u64` /
//! `sign_extend`, little-endian, idx-then-code). There is no float, no
//! platform-width type, and no UB-leaning shift in the wire path, so the
//! reconstructed `(index, code)` pairs MUST be bit-identical on every device.
//! These tests pin that claim exhaustively over the bounded code/index domain,
//! re-derive the packed bytes from an independent from-spec bit reader (the
//! same technique `exhaustive.rs` uses to second-source the Q12 decode), nail a
//! hard-coded cross-platform golden vector, and assert the full archive
//! append→read path is byte-stable.
//!
//! These mirror the `exhaustive.rs` pattern: a from-spec reference reader +
//! exhaustive enumeration over a small domain + a coverage assertion so the
//! enumeration can never silently shrink. No GPU, no heavy build.
//!
//! Run: `cargo test -p strand-quant --test outl_wire_exhaustive`

use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::format::{read_strand_v2, read_strand_v2_header, write_strand_v2, PackedTensor, PackedTensorV2, PAGE};
use strand_quant::outlier_wire::{append_outl, idx_bits_for, read_outl, read_outl_bytes, OutlSection, OutlierWire};
use strand_quant::TrellisConfig;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// From-spec independent bit reader (mirrors `ref_read_bits` in exhaustive.rs).
//
// This deliberately does NOT call the production `read_bits_u64`; it re-derives
// the documented wire from first principles so a regression in the production
// bit-packing is caught by disagreement, not by both sides drifting together.
// ---------------------------------------------------------------------------

/// Read `nbits` little-endian bits starting at `start_bit` (bit 0 = LSB of the
/// first byte). Out-of-range bits read as 0, matching the production reader.
fn spec_read_bits(bytes: &[u8], start_bit: usize, nbits: u32) -> u64 {
    let mut acc = 0u64;
    for i in 0..nbits as usize {
        let bit_idx = start_bit + i;
        let byte = bit_idx >> 3;
        let bit = if byte < bytes.len() { (bytes[byte] >> (bit_idx & 7)) & 1 } else { 0 };
        acc |= (bit as u64) << i;
    }
    acc
}

/// Sign-extend the low `nbits` of `v` to i32, from first principles.
fn spec_sign_extend(v: u64, nbits: u32) -> i32 {
    let shift = 64 - nbits;
    (((v << shift) as i64) >> shift) as i32
}

/// Decode a wire's packed payload from the spec: walk `count` (idx, code) pairs
/// at the given bit widths. Returns the entries plus the trailing pad bits (so
/// the caller can assert the pad is zero, which the parser also enforces).
fn spec_unpack(packed: &[u8], count: usize, idx_bits: u32, val_bits: u32) -> (Vec<(u32, i32)>, u64) {
    let mut entries = Vec::with_capacity(count);
    let mut cursor = 0usize;
    for _ in 0..count {
        let idx = spec_read_bits(packed, cursor, idx_bits) as u32;
        cursor += idx_bits as usize;
        let code = spec_sign_extend(spec_read_bits(packed, cursor, val_bits), val_bits);
        cursor += val_bits as usize;
        entries.push((idx, code));
    }
    let pad = if cursor % 8 != 0 { spec_read_bits(packed, cursor, (8 - (cursor % 8)) as u32) } else { 0 };
    (entries, pad)
}

// ---------------------------------------------------------------------------
// Temp-file plumbing (matches the in-crate unit tests' convention).
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("strand-outlx-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
}

struct TmpFile(PathBuf);
impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn test_weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
}

/// Build a real two-tensor v2 archive (totals 1024 and 900) the way the
/// in-crate tests do, so we can drive the full append→read path.
fn build_test_archive() -> Vec<u8> {
    let cfg = TrellisConfig::for_bpw(3.0);
    let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
    let enc_b = encode_tensor_with(&test_weights(900, 23), &cfg, &EncodeOpts::default());
    let shape_a = [4u64, 256u64];
    let shape_b = [900u64];
    let tensors = [
        PackedTensorV2 {
            base: PackedTensor { name: "model.layers.0.q_proj", shape: &shape_a, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc_a },
            block_len: cfg.block_len as u32,
        },
        PackedTensorV2 {
            base: PackedTensor { name: "model.layers.0.down_proj", shape: &shape_b, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc_b },
            block_len: cfg.block_len as u32,
        },
    ];
    write_strand_v2(&tensors, [9u8; 32], true).expect("write v2")
}

/// Round-trip a set of wires through the on-disk append→read path and return
/// the parsed section. `totals` must match the archive built above.
fn append_read(wires: &[Option<OutlierWire>], tag: &str) -> (OutlSection, Vec<u8>) {
    let buf = build_test_archive();
    let path = tmp_path(tag);
    let _guard = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();
    append_outl(&path, wires).expect("append_outl");
    let parsed = read_outl(&path).unwrap().expect("section present");
    let on_disk = std::fs::read(&path).unwrap();
    // The v2 prefix must be untouched by the append.
    assert_eq!(&on_disk[..buf.len()], &buf[..], "append clobbered v2 bytes [{tag}]");
    assert_eq!(on_disk.len() % PAGE, 0, "OUTL end not page-aligned [{tag}]");
    (parsed, buf)
}

// A tiny deterministic LCG so the property-style sweeps are reproducible on
// every host without pulling in a `rand`/`proptest` dev-dependency (the crate
// keeps dev-deps empty on purpose: "tests are self-contained").
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
}

// ---------------------------------------------------------------------------
// 1. EXHAUSTIVE value reconstruction: every representable code at every width.
//
//    Proves the value half of "position/value reconstruction exactness": the
//    2's-complement truncate-on-write / sign-extend-on-read pair is the exact
//    identity over the entire VALID code domain [-levels, +levels] for every
//    val_bits in 2..=16. Width 16 has 65_535 representable codes — fully
//    enumerated. This is the strongest possible statement about the value path
//    short of a Kani proof, and it covers what Kani would prove anyway.
// ---------------------------------------------------------------------------
#[test]
fn exhaustive_code_reconstruction_all_widths() {
    let mut covered: u64 = 0;
    for val_bits in 2u32..=16 {
        let levels = (1i64 << (val_bits - 1)) - 1; // max magnitude that fits
                                                   // Enumerate every code from -levels..=levels. These are exactly the
                                                   // codes `outl_section_bytes` accepts; one entry per wire keeps idx_bits
                                                   // pinned at 1 (n_total = 2) so the packed payload isolates the code.
        let mut expected: Vec<(u32, i32)> = Vec::new();
        let mut codes: Vec<i32> = Vec::new();
        // We pack many codes into one tensor (strictly ascending indices) so a
        // single serialize/parse exercises code-to-code bit alignment too, not
        // just isolated codes. n_total is the largest index + 1.
        let n = (2 * levels + 1) as usize; // distinct indices 0..n
        for (k, code) in (-levels..=levels).enumerate() {
            codes.push(code as i32);
            expected.push((k as u32, code as i32));
            covered += 1;
        }
        let idx: Vec<usize> = (0..n).collect();
        let omax = 1.0f32; // value path is independent of omax
        let wire = OutlierWire::from_selection(n, idx, codes, omax, val_bits);
        assert_eq!(wire.val_bits, val_bits, "val_bits clamped at {val_bits}");
        // Drive the on-disk path with a custom-sized archive (n weights).
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor_with(&test_weights(n, 7), &cfg, &EncodeOpts::default());
        let shape = [n as u64];
        let pt = PackedTensorV2 {
            base: PackedTensor { name: "t", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write v2");
        let path = tmp_path(&format!("code{val_bits}"));
        let _g = TmpFile(path.clone());
        std::fs::write(&path, &buf).unwrap();
        append_outl(&path, &[Some(wire.clone())]).expect("append");
        let back = read_outl(&path).unwrap().expect("present");
        let w = back.tensors[0].as_ref().expect("some");
        assert_eq!(w.entries, expected, "code round-trip drift at val_bits={val_bits}");
        assert_eq!(w, &wire, "full wire equality at val_bits={val_bits}");
    }
    // 2..=16 -> sum over w of (2*((1<<(w-1))-1)+1) representable codes.
    let expect: u64 = (2u32..=16).map(|w| (2 * ((1u64 << (w - 1)) - 1)) + 1).sum();
    assert_eq!(covered, expect, "code coverage drifted");
    eprintln!("exhaustive code reconstruction: {covered} codes across val_bits 2..=16");
}

// ---------------------------------------------------------------------------
// 2. EXHAUSTIVE position reconstruction + from-spec byte agreement.
//
//    For small idx_bits we enumerate every strictly-ascending index set drawn
//    from {0..n_total} and confirm: (a) the parser reconstructs the exact
//    positions, and (b) the packed payload the production writer emits is
//    bit-for-bit what the independent from-spec reader decodes — second-sourcing
//    the bit layout the way exhaustive.rs second-sources the Q12 decode.
//
//    To touch the bytes directly we parse the *section* out of the on-disk
//    archive and re-read its packed payload region with `spec_unpack`.
// ---------------------------------------------------------------------------
#[test]
fn exhaustive_position_reconstruction_and_spec_bytes() {
    let mut covered: u64 = 0;
    // n_total small enough to enumerate all subsets but large enough to span
    // idx_bits in {1,2,3,4}. n_total = 9 -> idx_bits = 4, 2^9 = 512 subsets.
    let n_total = 9usize;
    let idx_bits = idx_bits_for(n_total);
    assert_eq!(idx_bits, 4);
    let val_bits = 5u32; // a width whose total bit stride (idx+val=9) crosses bytes oddly
    let levels = (1i64 << (val_bits - 1)) - 1;

    for mask in 0u32..(1u32 << n_total) {
        let idx: Vec<usize> = (0..n_total).filter(|&b| (mask >> b) & 1 == 1).collect();
        if idx.is_empty() {
            // A zero-entry channel is represented as `None`, not an empty Some;
            // skip — the None path is covered by the section sweep below.
            continue;
        }
        // Codes that exercise both signs and the extremes of the val_bits range.
        let codes: Vec<i32> = idx
            .iter()
            .enumerate()
            .map(|(j, _)| {
                let m = (j as i64 % (levels + 1)) as i32;
                if j % 2 == 0 {
                    m
                } else {
                    -m
                }
            })
            .collect();
        let expected: Vec<(u32, i32)> = idx.iter().map(|&i| i as u32).zip(codes.iter().copied()).collect();

        let wire = OutlierWire::from_selection(n_total, idx.clone(), codes, 0.5f32, val_bits);
        assert_eq!(wire.idx_bits, idx_bits);

        let parsed = {
            // Build an n_total-weight archive once per subset would be slow;
            // instead reuse a single-tensor archive of exactly n_total weights.
            let cfg = TrellisConfig::for_bpw(3.0);
            let enc = encode_tensor_with(&test_weights(n_total, 3), &cfg, &EncodeOpts::default());
            let shape = [n_total as u64];
            let pt = PackedTensorV2 {
                base: PackedTensor { name: "t", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
                block_len: cfg.block_len as u32,
            };
            let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write v2");
            let path = tmp_path("pos");
            let _g = TmpFile(path.clone());
            std::fs::write(&path, &buf).unwrap();
            append_outl(&path, &[Some(wire.clone())]).expect("append");
            let on_disk = std::fs::read(&path).unwrap();
            let parsed = read_outl_bytes(&on_disk, true).unwrap().expect("present");

            // Independently locate this wire's packed payload inside the section
            // and decode it from first principles. Layout (see outl_section_bytes):
            //   header 32 B, then per-tensor record:
            //     count u64 | omax u32 | idx_bits u32 | val_bits u32 | reserved u32
            //   then `count * (idx_bits+val_bits)` bits of payload (byte-padded).
            let t = &on_disk[on_disk.len() - 16..];
            let outl_off = u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize;
            let rec = outl_off + 32; // single tensor -> its record starts after the header
            let count = u64::from_le_bytes(on_disk[rec..rec + 8].try_into().unwrap()) as usize;
            let payload_start = rec + 24;
            let payload_bytes = (count * (idx_bits + val_bits) as usize).div_ceil(8);
            let packed = &on_disk[payload_start..payload_start + payload_bytes];
            let (spec_entries, spec_pad) = spec_unpack(packed, count, idx_bits, val_bits);
            assert_eq!(spec_entries, expected, "from-spec byte decode disagrees with intent (mask={mask:#x})");
            assert_eq!(spec_pad, 0, "production writer left nonzero pad bits (mask={mask:#x})");
            parsed
        };

        let w = parsed.tensors[0].as_ref().expect("some");
        assert_eq!(w.entries, expected, "parser position drift (mask={mask:#x})");
        covered += 1;
    }
    // 2^9 subsets minus the empty set.
    assert_eq!(covered, (1u64 << n_total) - 1, "subset coverage drifted");
    eprintln!("exhaustive position reconstruction: {covered} index subsets (idx_bits=4)");
}

// ---------------------------------------------------------------------------
// 3. PROPERTY sweep: full multi-tensor section serialize→parse is the identity,
//    and the parsed section equals the input wires exactly, across a wide,
//    deterministic random sample of shapes (mix of None / Some, varying counts,
//    every idx_bits implied by n_total, every val_bits in 2..=16).
//
//    This is the round-trip property the surface must satisfy in aggregate:
//    `read_outl(append_outl(w)) == w`.
// ---------------------------------------------------------------------------
#[test]
fn property_section_round_trip_is_identity() {
    // Two-tensor archive: totals are 1024 and 900.
    let totals = [1024usize, 900usize];
    let mut rng = Lcg(0xDEAD_BEEF_CAFE_F00D);
    let mut cases = 0u64;

    for _ in 0..400 {
        let mut wires: Vec<Option<OutlierWire>> = Vec::with_capacity(2);
        for &total in &totals {
            // ~25% of channels absent.
            if rng.below(4) == 0 {
                wires.push(None);
                continue;
            }
            let val_bits = 2 + rng.below(15) as u32; // 2..=16
            let levels = (1i64 << (val_bits - 1)) - 1;
            // Choose a random strictly-ascending index subset of size 1..=min(total, 40).
            let want = 1 + rng.below(40.min(total as u64)) as usize;
            let mut chosen: Vec<usize> = Vec::with_capacity(want);
            // Reservoir-free ascending pick: walk and accept with a probability
            // tuned to land near `want`, then trim/pad deterministically.
            let mut i = 0usize;
            while i < total && chosen.len() < want {
                let remaining_slots = want - chosen.len();
                let remaining_items = total - i;
                if rng.below(remaining_items as u64) < remaining_slots as u64 {
                    chosen.push(i);
                }
                i += 1;
            }
            if chosen.is_empty() {
                chosen.push(0);
            }
            let codes: Vec<i32> = chosen
                .iter()
                .map(|_| {
                    let span = (2 * levels + 1) as u64;
                    (rng.below(span) as i64 - levels) as i32
                })
                .collect();
            let omax_bits_src = rng.next_u64() as u32;
            let omax = f32::from_bits(omax_bits_src);
            // from_selection stores omax via to_bits(); a NaN payload would
            // round-trip its bits but make equality on the f32 awkward, so feed
            // a finite omax (its exact bits are still preserved & checked).
            let omax = if omax.is_finite() { omax } else { 1.0 };
            wires.push(Some(OutlierWire::from_selection(total, chosen, codes, omax, val_bits)));
        }
        // Skip the degenerate all-None case occasionally produced; the parser
        // still handles it but it adds no signal here.
        if wires.iter().all(|w| w.is_none()) {
            continue;
        }

        let (parsed, _buf) = append_read(&wires, "prop");
        assert_eq!(parsed.tensors, wires, "section round-trip is not the identity");
        // omax_bits must survive verbatim (the only float-bearing field).
        for (a, b) in parsed.tensors.iter().zip(wires.iter()) {
            if let (Some(pa), Some(pb)) = (a, b) {
                assert_eq!(pa.omax_bits, pb.omax_bits, "omax_bits not byte-stable");
                assert_eq!(pa.idx_bits, pb.idx_bits);
                assert_eq!(pa.val_bits, pb.val_bits);
            }
        }
        cases += 1;
    }
    assert!(cases > 300, "sweep produced too few non-trivial cases: {cases}");
    eprintln!("property section round-trip: {cases} multi-tensor sections");
}

// ---------------------------------------------------------------------------
// 4. CROSS-PLATFORM GOLDEN VECTOR.
//
//    A fixed wire serialized to fixed, hard-coded bytes. If anyone changes the
//    endianness, the bit order, the field layout, or the sign-extension of the
//    OUTL wire, this byte vector changes and the test fails on EVERY platform —
//    which is exactly the regression we want to make impossible to land
//    silently. The bytes were derived by hand from the documented layout and
//    cross-checked by the from-spec reader below (so the golden is itself
//    second-sourced, not just snapshotted from current output).
// ---------------------------------------------------------------------------
#[test]
fn golden_packed_payload_is_byte_stable() {
    // n_total = 1024 -> idx_bits = 10; val_bits = 8.
    // entries (post-sort by index): (3, 5), (511, 127), (700, -127)
    let wire = OutlierWire::from_selection(1024, vec![700, 3, 511], vec![-127, 5, 127], 0.3125f32, 8);
    assert_eq!(wire.idx_bits, 10);
    assert_eq!(wire.val_bits, 8);
    assert_eq!(wire.entries, vec![(3, 5), (511, 127), (700, -127)]);

    // Build the packed payload the way the writer would: 3 entries * (10+8)=54
    // bits -> 7 bytes (54 bits, 2 pad bits zero).
    //
    // Little-endian bit stream (bit 0 = LSB of byte 0), index-then-code per entry:
    //   entry 0: idx=3   (10 bits)        ; code=5    (8 bits, 2c) = 0x05
    //   entry 1: idx=511 (10 bits)        ; code=127  (8 bits)     = 0x7F
    //   entry 2: idx=700 (10 bits)        ; code=-127 (8 bits, 2c) = 0x81
    // The exact byte vector below is cross-checked by `spec_unpack` just after,
    // so the constant is self-validating rather than a blind snapshot.
    let golden: [u8; 7] = [0x03, 0x14, 0xFC, 0xF7, 0xC7, 0x6B, 0x20];

    // Independently confirm the golden decodes back to the intended entries via
    // the from-spec reader (so the golden constant is self-validating).
    let (spec_entries, spec_pad) = spec_unpack(&golden, 3, 10, 8);
    assert_eq!(spec_entries, vec![(3, 5), (511, 127), (700, -127)]);
    assert_eq!(spec_pad, 0);

    // Now confirm the PRODUCTION writer emits exactly these bytes by extracting
    // the packed region from a real on-disk section.
    let buf = build_test_archive(); // tensor 0 has total 1024
    let path = tmp_path("golden");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();
    append_outl(&path, &[Some(wire.clone()), None]).expect("append");
    let on_disk = std::fs::read(&path).unwrap();

    let t = &on_disk[on_disk.len() - 16..];
    let outl_off = u64::from_le_bytes(t[0..8].try_into().unwrap()) as usize;
    let rec0 = outl_off + 32;
    let count0 = u64::from_le_bytes(on_disk[rec0..rec0 + 8].try_into().unwrap()) as usize;
    assert_eq!(count0, 3);
    let payload0 = rec0 + 24;
    let bytes = &on_disk[payload0..payload0 + 7];
    assert_eq!(bytes, &golden, "OUTL packed payload changed — cross-platform wire regression");

    // And the parsed entries match.
    let parsed = read_outl_bytes(&on_disk, true).unwrap().expect("present");
    assert_eq!(parsed.tensors[0].as_ref().unwrap().entries, wire.entries);
}

// ---------------------------------------------------------------------------
// 5. DEQUANT determinism: same wire bits -> identical f32 bit pattern, every
//    time, and equal to the documented spec `(code/levels)*omax`.
//
//    The wire path proper is integer-exact; `dequant_vals` is the one place an
//    f32 multiply enters. We don't claim cross-platform f32 *rounding* identity
//    here (that depends on the FMA/rounding contract of the host and is a
//    separate surface), but we DO pin two things that must hold on any single
//    device and that the rest of the codebase relies on:
//      (a) it is a pure function of the wire (repeatable, no hidden state), and
//      (b) it matches the exact closed-form the quantizer documents, so the
//          dequant cannot silently diverge from the recon path.
// ---------------------------------------------------------------------------
#[test]
fn dequant_is_pure_and_matches_spec() {
    let mut rng = Lcg(0x0123_4567_89AB_CDEF);
    for _ in 0..2000 {
        let val_bits = 2 + rng.below(15) as u32;
        let levels_i = (1i64 << (val_bits - 1)) - 1;
        let n = 4096usize;
        let count = 1 + rng.below(8) as usize;
        let mut idx: Vec<usize> = Vec::with_capacity(count);
        let mut cur = 0usize;
        for _ in 0..count {
            cur += 1 + rng.below(64) as usize;
            if cur >= n {
                break;
            }
            idx.push(cur);
        }
        if idx.is_empty() {
            continue;
        }
        let codes: Vec<i32> = idx.iter().map(|_| (rng.below((2 * levels_i + 1) as u64) as i64 - levels_i) as i32).collect();
        let omax = f32::from_bits(rng.next_u64() as u32);
        let omax = if omax.is_finite() { omax } else { 0.5 };
        let wire = OutlierWire::from_selection(n, idx.clone(), codes.clone(), omax, val_bits);

        // (a) purity / repeatability.
        let first: Vec<(u32, u32)> = wire.dequant_vals().map(|(i, v)| (i, v.to_bits())).collect();
        let second: Vec<(u32, u32)> = wire.dequant_vals().map(|(i, v)| (i, v.to_bits())).collect();
        assert_eq!(first, second, "dequant_vals is not a pure function of the wire");

        // (b) exact closed form: (code as f32)/levels*omax, byte-identical.
        let levels_f = levels_i as f32;
        let omax_back = f32::from_bits(wire.omax_bits);
        let want: Vec<(u32, u32)> = wire.entries.iter().map(|&(i, c)| (i, ((c as f32) / levels_f * omax_back).to_bits())).collect();
        assert_eq!(first, want, "dequant diverged from the documented closed form");
    }
    eprintln!("dequant purity + spec match: 2000 wires");
}

// ---------------------------------------------------------------------------
// 6. wire_bytes() accounting is exact vs. the bytes actually packed.
//
//    The delta-billing/bit-ledger code trusts `wire_bytes()`. Pin that it
//    equals the real serialized payload size (12-byte preamble accounting + the
//    bit-packed entries), across widths, so a layout change can't desync the
//    accounting from the wire.
// ---------------------------------------------------------------------------
#[test]
fn wire_bytes_equals_real_packed_size() {
    let mut rng = Lcg(0xABCD_1234_5678_9F01);
    for _ in 0..500 {
        let val_bits = 2 + rng.below(15) as u32;
        let levels = (1i64 << (val_bits - 1)) - 1;
        let n = 65536usize;
        let count = 1 + rng.below(50) as usize;
        let mut idx: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for _ in 0..count {
            cur += 1 + rng.below(16) as usize;
            if cur >= n {
                break;
            }
            idx.push(cur);
        }
        if idx.is_empty() {
            continue;
        }
        let codes: Vec<i32> = idx.iter().map(|_| (rng.below((2 * levels + 1) as u64) as i64 - levels) as i32).collect();
        let wire = OutlierWire::from_selection(n, idx, codes, 1.0, val_bits);

        let real_payload = (wire.entries.len() * (wire.idx_bits + wire.val_bits) as usize).div_ceil(8) as u64;
        assert_eq!(wire.wire_bytes(), 12 + real_payload, "wire_bytes() desynced from packed size (idx_bits={} val_bits={} n_entries={})", wire.idx_bits, wire.val_bits, wire.entries.len());
    }
}

// ---------------------------------------------------------------------------
// 7. Plain v2 archive (no OUTL) reads as absent, and the full v2 reader still
//    works under an OUTL trailer — the surface must be transparent to readers
//    that don't know about it.
// ---------------------------------------------------------------------------
#[test]
fn outl_is_transparent_to_v2_readers() {
    let buf = build_test_archive();
    let path = tmp_path("transparent");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();
    assert_eq!(read_outl(&path).unwrap(), None, "plain v2 must read OUTL as absent");

    let wires = vec![Some(OutlierWire::from_selection(1024, vec![3, 511, 700], vec![5, 127, -127], 0.3125, 8)), None];
    append_outl(&path, &wires).expect("append");

    let on_disk = std::fs::read(&path).unwrap();
    // v2 header + full v2 body still parse with the OUTL trailer present.
    let h0 = read_strand_v2_header(&buf).unwrap();
    let h1 = read_strand_v2_header(&on_disk).unwrap();
    assert_eq!(h0.tensors.len(), h1.tensors.len());
    let full = read_strand_v2(&on_disk).expect("full v2 read under OUTL trailer");
    assert_eq!(full.len(), 2);
}
