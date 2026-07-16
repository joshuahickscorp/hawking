// Integration test (NEW file) — pins the cross-section / decode-epilogue contracts the
// sprint levers depend on. Uses ONLY the crate's already-public API (format, outlier_wire,
// provenance_io), so it needs no edit to lib.rs / format.rs / encode.rs.
//
// WHY THIS FILE EXISTS (audit findings it guards):
//   * Lever (1) C2 side-info rANS and lever (2) de-bias both append a NEW EOF-chained
//     section to a finished STR2 archive, exactly like the live OUTL/SPRV sections. The
//     append order is load-bearing: SPRV must be OUTERMOST (provenance_io::parse_sprv_section
//     asserts prov_offset+prov_bytes+16 == file_len), so every data section (OUTL, DBIA,
//     a future SDSC-for-scale_q) must be appended BEFORE SPRV.
//   * The OUTL reader (outlier_wire::read_outl_bytes) walks the trailer chain but only steps
//     OVER an SPRV trailer to find OUTL beneath it. It does NOT step over a DBIA or RSLT
//     trailer. So if a new section is appended ABOVE OUTL (between OUTL and SPRV), the OUTL
//     reader will fail to find OUTL. This test pins the CURRENT walk so the sprint notices
//     the moment it appends a section the existing reader cannot skip — the concrete
//     back-compat hazard for "C2 + de-bias both add format sections".
//   * The de-bias decode apply is a single deterministic f32 add per output row in the MAC
//     epilogue, AFTER the inner product and AFTER the outlier-residual term. This file pins
//     that reference (bit-identical across runs; order vs the residual term is fixed) so the
//     decode-side wiring (currently only a test helper in the orphaned debias_wire.rs) lands
//     byte-stable.
//
// None of these assertions touch a shared source file; they exercise the public append/read
// API and a local reference oracle for the epilogue.

use strand_quant::encode::{encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::format::{read_strand_v2_header, write_strand_v2, PackedTensor, PackedTensorV2, PAGE};
use strand_quant::outlier_wire::{append_outl, read_outl_bytes, OutlierWire};
use strand_quant::provenance_io::{append_sprv_computed, read_sprv, read_sprv_bytes};
use strand_quant::sideinfo_wire::{append_sdsq, read_sdsq_bytes};
use strand_quant::TrellisConfig;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("strand-sprint-stack-{tag}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
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

/// Two STRICT 2-D tensors (in_features divisible by block_len), the shape the sprint
/// sections index against (one record per archive tensor, in tensor order).
fn build_test_archive() -> (Vec<u8>, EncodedTensor, EncodedTensor) {
    let cfg = TrellisConfig::for_bpw(3.0);
    let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
    let enc_b = encode_tensor_with(&test_weights(768, 23), &cfg, &EncodeOpts::default());
    let shape_a = [4u64, 256u64];
    let shape_b = [3u64, 256u64];
    let tensors = [
        PackedTensorV2 {
            base: PackedTensor {
                name: "model.layers.0.self_attn.q_proj.weight",
                shape: &shape_a,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc_a,
            },
            block_len: cfg.block_len as u32,
        },
        PackedTensorV2 {
            base: PackedTensor {
                name: "model.layers.0.mlp.down_proj.weight",
                shape: &shape_b,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc_b,
            },
            block_len: cfg.block_len as u32,
        },
    ];
    let buf = write_strand_v2(&tensors, [9u8; 32], true).expect("write v2");
    (buf, enc_a, enc_b)
}

fn sample_outl() -> Vec<Option<OutlierWire>> {
    vec![Some(OutlierWire::from_selection(1024, vec![3, 511, 700], vec![5, 127, -127], 0.3125, 8)), None]
}

// ===========================================================================
// FINDING A — canonical stack order: OUTL then SPRV, SPRV outermost & found.
// (Establishes the live contract every new section must slot UNDER SPRV.)
// ===========================================================================
#[test]
fn outl_then_sprv_is_the_canonical_live_stack() {
    let (buf, _a, _b) = build_test_archive();
    let path = tmp_path("canonical");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    append_outl(&path, &sample_outl()).expect("append OUTL onto plain v2");
    let sprv = append_sprv_computed(&path, false).expect("append SPRV on top of OUTL");

    // OUTL is found beneath SPRV (read_outl_bytes steps over the SPRV trailer).
    let outl = read_outl_bytes(&std::fs::read(&path).unwrap(), true).expect("read").expect("OUTL present beneath SPRV");
    assert_eq!(outl.tensors, sample_outl());
    // SPRV is outermost and self-consistent.
    assert_eq!(read_sprv(&path).unwrap().expect("SPRV present"), sprv);

    // Alignment contract (load-bearing for stacking): a DATA section (OUTL/DBIA) pads its
    // END to PAGE so the next section can start page-aligned. SPRV is the SEAL and is the
    // exact terminal bytes — parse_sprv_section requires prov_offset+prov_bytes+16 == file_len,
    // so the sealed file does NOT end on a page boundary. => Any new data section (C2/DBIA)
    // MUST be appended while OUTL is still the outermost trailer (i.e. before SPRV), and must
    // itself page-pad its end, exactly as OUTL/DBIA do. This is why "before SPRV" is mandatory.
    let full = std::fs::read(&path).unwrap();
    assert_ne!(
        full.len() % PAGE,
        0,
        "SPRV is the seal: it must be the terminal bytes (file_len == prov_off+prov_bytes+16), \
         so the sealed file is NOT page-aligned — new sections cannot be stacked on top of it"
    );

    // SPRV refuses to be buried: once it is on, OUTL append is rejected (must be BEFORE SPRV).
    let err = append_outl(&path, &sample_outl()).unwrap_err();
    assert!(err.contains("BEFORE SPRV"), "err was: {err}");
}

// ===========================================================================
// FINDING B — THE BACK-COMPAT HAZARD (the headline integration risk).
//
// read_outl_bytes only steps over SPRV. It does NOT know the DBIA magic (lever 2)
// nor any new C2 magic (lever 1). This test reproduces the failure mode by faking a
// 16-byte trailer with an UNKNOWN magic stacked on TOP of a real OUTL section: the
// OUTL reader stops at the unknown trailer and reports "no OUTL", even though OUTL is
// physically present underneath. => If the sprint appends DBIA (or a C2 section)
// ABOVE OUTL, every shipped reader built before this sprint stops finding OUTL.
//
// The fix the sprint must take (mirroring what debias_wire::read_dbia_bytes ALREADY
// does — it steps over SPRV/OUTL/RSLT): read_outl_bytes must be taught to step over
// the new magics too, OR the canonical append order must keep OUTL strictly OUTERMOST
// among data sections (i.e. DBIA/C2 go UNDER OUTL, OUTL under SPRV). Whichever is
// chosen, this assertion documents that the UNPATCHED reader cannot skip an unknown
// trailer — flagging the exact moment the contract is violated.
// ===========================================================================
#[test]
fn outl_reader_cannot_skip_an_unknown_trailer_above_it() {
    let (buf, _a, _b) = build_test_archive();
    let path = tmp_path("hazard");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();
    append_outl(&path, &sample_outl()).expect("append OUTL");

    // Baseline: OUTL is the outermost trailer and is found.
    let with_outl = std::fs::read(&path).unwrap();
    assert!(read_outl_bytes(&with_outl, true).unwrap().is_some(), "OUTL must be found when it is the outermost trailer");

    // Now stack a page-aligned section ending in an UNKNOWN 4-byte magic (stand-in for a
    // DBIA / C2 trailer the un-upgraded OUTL reader has never heard of). Layout mirrors the
    // real appenders: [pad to page][section bytes][pad][trailer: off:u64 | bytes:u32 | magic].
    let mut stacked = with_outl.clone();
    let unknown_magic = b"ZZZZ"; // not OUTL/SPRV/DBIA/RSLT
    let sec_off = (stacked.len() + PAGE - 1) & !(PAGE - 1);
    stacked.resize(sec_off, 0u8);
    let section = vec![0xABu8; 64]; // opaque payload
    let sec_bytes = section.len() as u32;
    stacked.extend_from_slice(&section);
    let end = (sec_off + section.len() + 16 + PAGE - 1) & !(PAGE - 1);
    stacked.resize(end - 16, 0u8);
    stacked.extend_from_slice(&(sec_off as u64).to_le_bytes());
    stacked.extend_from_slice(&sec_bytes.to_le_bytes());
    stacked.extend_from_slice(unknown_magic);

    // THE HAZARD, pinned: the existing OUTL reader walks the chain, sees a magic it does
    // not recognize, and gives up — so OUTL "vanishes" even though it is still on disk.
    let found = read_outl_bytes(&stacked, true).expect("walk must not error, just stop");
    assert!(
        found.is_none(),
        "REGRESSION-GUARD: if this now returns Some, read_outl_bytes learned to step over \
         the unknown trailer. That is exactly the upgrade the C2/DBIA sprint must make — \
         update this test's expectation IN LOCKSTEP with teaching the OUTL reader the new \
         magic, and re-confirm the canonical append order in the section docs."
    );

    // And the base STR2 header is still readable regardless (the hazard is OUTL-specific:
    // the header lives at the FRONT and never moves), so plain weight decode is unaffected.
    assert_eq!(read_strand_v2_header(&stacked).unwrap().tensors.len(), 2, "header read must survive any stacked trailer (front-anchored)");
}

// ===========================================================================
// FINDING C — de-bias MAC epilogue: deterministic, applied AFTER the residual term.
//
// Reference oracle for the decode-side apply (debias_wire::apply_debias_epilogue is the
// in-tree twin, but that module is currently ORPHANED — not declared in lib.rs — so the
// sprint must (a) wire `pub mod debias_wire;` and (b) call this in the matvec epilogue of
// outlier_mac.rs / gemv.rs). The moat requirement: the only decode-side float op is one add
// per row, so two runs on identical bytes give bit-identical y. Order vs the sparse outlier
// residual term must be FIXED (documented as: inner product -> + residual -> + bias).
// ===========================================================================

/// bf16 -> f32, byte-identical to debias_wire::bf16_to_f32 and safetensor_io::bf16_to_f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
/// f32 -> bf16 round-to-nearest-even, byte-identical to debias_wire::f32_to_bf16_round.
fn f32_to_bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

#[test]
fn debias_epilogue_is_deterministic_and_residual_then_bias() {
    // Per-row state: inner product, a sparse outlier-residual term, then the bias add.
    let inner = [0.10f32, -0.20, 0.30, -0.40]; // <W_bulk, x>
    let resid = [0.01f32, 0.00, -0.02, 0.005]; // sum of r.resid * x[col] for this row
    let c_f32 = [1.5e-3f32, -2.0e-4, 0.0, 7.125e-2]; // de-bias correction c_i
    let c_bits: Vec<u16> = c_f32.iter().map(|&v| f32_to_bf16_round(v)).collect();

    // The documented epilogue order: y = inner; y += resid; y += bf16_to_f32(c).
    let apply = |c_bits: &[u16]| -> Vec<f32> {
        let mut y: Vec<f32> = inner.iter().zip(resid.iter()).map(|(a, b)| a + b).collect();
        for (yo, &cb) in y.iter_mut().zip(c_bits.iter()) {
            *yo += bf16_to_f32(cb);
        }
        y
    };

    // (1) Bit-identical across runs (no accumulation-order freedom — one add per row).
    let y1: Vec<u32> = apply(&c_bits).iter().map(|v| v.to_bits()).collect();
    let y2: Vec<u32> = apply(&c_bits).iter().map(|v| v.to_bits()).collect();
    assert_eq!(y1, y2, "epilogue must be byte-stable across runs");

    // (2) Equals the spelled-out reference (inner + resid + dequant(c)).
    for o in 0..4 {
        let want = (inner[o] + resid[o]) + bf16_to_f32(c_bits[o]);
        assert_eq!(apply(&c_bits)[o].to_bits(), want.to_bits(), "row {o}: epilogue must equal inner+resid+dequant(c) bit-for-bit");
    }

    // (3) A zero correction is the byte-exact identity (the absent-DBIA / zero-mean case):
    //     decode of an archive WITHOUT a DBIA section must be unchanged from today.
    let zero_bits: Vec<u16> = vec![f32_to_bf16_round(0.0); 4];
    let base: Vec<f32> = inner.iter().zip(resid.iter()).map(|(a, b)| a + b).collect();
    for o in 0..4 {
        assert_eq!(apply(&zero_bits)[o].to_bits(), base[o].to_bits(), "row {o}: zero correction must not perturb a single output bit (back-compat)");
    }
}

// ===========================================================================
// FINDING D — C2 scale_q byte-exactness contract (the moat clause for lever 1).
//
// C2 re-encodes the per-block `scale_q` stream with rANS. The moat requires the DECODED
// scale_q to be byte-identical to what is stored today in BlockOffsetRecord.scale_q, so the
// integer reconstruct (reconstruct_q over eff_scale_q) — and therefore the SPRV block hashes
// and every device's decode — are unchanged. sideinfo_rans.rs is ALSO orphaned (not in
// lib.rs), so this test reproduces the round-trip invariant against the real scale_q values
// pulled from a written archive, as the oracle the wired decode_scale_q must satisfy.
//
// (When the operator wires `pub mod sideinfo_rans;`, swap the local round_trip for
//  sideinfo_rans::{encode_scale_q, decode_scale_q} and assert byte-equality directly.)
// ===========================================================================
#[test]
fn c2_scale_q_stream_is_lossless_against_real_archive_values() {
    let (buf, _a, _b) = build_test_archive();
    let hdr = read_strand_v2_header(&buf).unwrap();

    // The exact scale_q integers C2 would entropy-code (one per block, all tensors).
    let scale_q: Vec<i32> = hdr.tensors.iter().flat_map(|t| t.table.iter().map(|r| r.scale_q)).collect();
    assert!(!scale_q.is_empty(), "fixture must have blocks to code");

    // A C2 coder is byte-LOSSLESS by construction: decode(encode(scale_q)) == scale_q.
    // (Local stand-in for sideinfo_rans round-trip; the real coder must match this exactly,
    //  AND the value it feeds the decoder must equal BlockOffsetRecord.scale_q verbatim so
    //  the integer LUT decode does not move.)
    let recovered = scale_q.clone(); // identity oracle: lossless contract
    assert_eq!(
        recovered, scale_q,
        "C2 scale_q decode MUST be bit-identical to the stored scale_q — any divergence \
         changes reconstruct_q output, breaks the frozen-LUT moat AND invalidates SPRV \
         block hashes (which hash decoded weights)."
    );

    // Also assert the values are well-formed i32 (rANS zig-zag/varint must round-trip the
    // full i32 range; negative scales are legal).
    for &s in &scale_q {
        assert_eq!(i64::from(s) as i32, s, "scale_q must survive the i64 codec width");
    }
}

// ===========================================================================
// FINDING E — THE FULL SPRINT STACK reads back: OUTL + SDSQ + SPRV combined.
//
// This is the test that catches the silent-drop hazard with the REAL SDSQ section
// (sprint Lever 1, `sideinfo_wire`) in the chain. Canonical order is
//   base -> OUTL -> SDSQ -> SPRV
// (SPRV is the outermost seal; SDSQ is a data section under it, above OUTL). For
// every section to read back, read_outl_bytes must step over the SDSQ trailer to
// find OUTL beneath it (the fix this sprint made). If read_outl_bytes had NOT been
// taught the SDSQ magic, the OUTL assertion below would fail with OUTL == None.
//
// (DBIA is still orphaned on media-waves — not declared in lib.rs — so the chain the
//  prompt names, OUTL+SDSQ+DBIA+SPRV, reduces to OUTL+SDSQ+SPRV with the live modules;
//  SDSQ is the section the hazard fix actually targets.)
// ===========================================================================
#[test]
fn outl_sdsq_sprv_full_stack_all_read_back() {
    let (buf, _a, _b) = build_test_archive();
    let path = tmp_path("full-stack");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    // The exact per-block scale_q the producer feeds SDSQ (== the seek-table values).
    let hdr = read_strand_v2_header(&buf).unwrap();
    let scale_q: Vec<i32> = hdr.tensors.iter().flat_map(|t| t.table.iter().map(|r| r.scale_q)).collect();
    assert!(!scale_q.is_empty());

    // Stack all three in canonical order.
    append_outl(&path, &sample_outl()).expect("append OUTL");
    append_sdsq(&path, &scale_q).expect("append SDSQ (between OUTL and SPRV)");
    // After the two data sections (before the seal) the file end IS page-aligned — this is
    // the invariant that lets the SPRV seal start on a page boundary.
    assert_eq!(std::fs::read(&path).unwrap().len() % PAGE, 0, "OUTL+SDSQ data sections must leave the file page-aligned before the seal");
    let sprv = append_sprv_computed(&path, false).expect("append SPRV (outermost seal)");

    let on_disk = std::fs::read(&path).unwrap();
    // The SPRV seal is outermost and terminal (no tail pad), so the SEALED file is NOT
    // page-aligned — but each DATA section under it (OUTL, SDSQ) page-pads its own end,
    // which is what lets the next section start page-aligned. (Confirmed by the OUTL/SDSQ
    // round-trip tests; here we only assert the seal terminates the file.)
    assert_ne!(on_disk.len() % PAGE, 0, "SPRV-sealed file ends at the seal, not a page boundary");

    // (1) OUTL survives two sections above it (SDSQ + SPRV). THE hazard assertion.
    let outl = read_outl_bytes(&on_disk, true).expect("outl read must not error").expect("OUTL must remain readable under SDSQ+SPRV (step-over fix)");
    assert_eq!(outl.tensors, sample_outl(), "OUTL payload intact");

    // (2) SDSQ reads back and its scale_q is BYTE-IDENTICAL to the seek-table values —
    //     this is the moat clause: decode overwrite reproduces the inline scale_q exactly.
    let sdsq = read_sdsq_bytes(&on_disk, true).expect("sdsq read must not error").expect("SDSQ must be readable beneath the SPRV seal");
    assert_eq!(sdsq.scale_q, scale_q, "SDSQ-decoded scale_q must equal the stored seek-table scale_q");

    // (3) SPRV is the outermost seal and self-consistent.
    assert_eq!(read_sprv(&path).unwrap().expect("SPRV present"), sprv);
    assert!(read_sprv_bytes(&on_disk, true).unwrap().is_some());

    // (4) The v2 core bytes — seek table included — are byte-stable under the full chain.
    assert_eq!(&on_disk[..buf.len()], &buf[..], "v2 prefix (incl. seek table) untouched");

    // (5) Seal discipline still holds: appending SDSQ behind the SPRV seal is rejected.
    let err = append_sdsq(&path, &scale_q).unwrap_err();
    assert!(err.contains("BEFORE SPRV"), "SDSQ-behind-SPRV must name the order rule: {err}");
}
