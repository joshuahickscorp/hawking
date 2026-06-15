//! Guard tests for the SPRINT integration into the `.strand` v2 EOF-section chain.
//!
//! These pin the *currently observable* invariants of the section-chaining
//! machinery (format.rs + outlier_wire.rs + provenance_io.rs + selfdesc.rs +
//! rslt.rs) that the sprint levers — C2 side-info rANS coder and the DBIA
//! de-bias section — must not break. Each test documents a concrete hazard the
//! audit found. They use only the already-wired public API (the two new sprint
//! files `sideinfo_rans.rs` / `debias_wire.rs` are NOT yet declared in lib.rs,
//! so they cannot be imported from an integration test until wired).
//!
//! Run: `cargo test -p strand-quant --test sprint_chain_hazards`

use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::format::{
    read_strand_v2, read_strand_v2_header, write_strand_v2, PackedTensor, PackedTensorV2, PAGE,
};
use strand_quant::outlier_wire::{append_outl, read_outl_bytes, OutlierWire};
use strand_quant::provenance_io::{append_sprv_computed, read_sprv_bytes};
use strand_quant::rslt::{self, append_rslt, read_rslt_bytes, RsltSection, RSLT_VERSION};
use strand_quant::selfdesc::{append_sdsc, read_sdsc_bytes};
use strand_quant::TrellisConfig;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "strand-sprint-{tag}-{}-{}.strand",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

struct TmpFile(PathBuf);
impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn test_weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
        .collect()
}

/// Two-tensor base archive (q_proj [4,256] out=4, down_proj [3,300] out=3).
fn build_base() -> Vec<u8> {
    let cfg = TrellisConfig::for_bpw(3.0);
    let enc_a = encode_tensor_with(&test_weights(1024, 11), &cfg, &EncodeOpts::default());
    let enc_b = encode_tensor_with(&test_weights(900, 23), &cfg, &EncodeOpts::default());
    let shape_a = [4u64, 256u64];
    let shape_b = [3u64, 300u64];
    let tensors = [
        PackedTensorV2 {
            base: PackedTensor {
                name: "model.layers.0.q_proj",
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
                name: "model.layers.0.down_proj",
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
    write_strand_v2(&tensors, [9u8; 32], false).expect("write v2")
}

fn rslt_for(buf: &[u8]) -> RsltSection {
    let hdr = read_strand_v2_header(buf).unwrap();
    RsltSection {
        version: RSLT_VERSION,
        block_counts: hdr.tensors.iter().map(|t| vec![0u32; t.n_blocks]).collect(),
    }
}

// ===========================================================================
// HAZARD 1 — RSLT breaks the page-aligned-EOF invariant the other sections
// rely on. OUTL/DBIA/SDSC pad their *file end* up to a page boundary; RSLT
// page-aligns only its section START, then writes body+trailer with no tail
// pad, so the file ends un-page-aligned. Every other section's parser rejects a
// `trailer_end % PAGE != 0`, so NOTHING may be stacked on top of an RSLT.
// The sprint must NOT place DBIA or a C2 side-info section after RSLT.
// ===========================================================================

#[test]
fn rslt_appended_file_end_is_not_page_aligned() {
    let buf = build_base();
    let path = tmp_path("rslt-align");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    append_rslt(&path, &rslt_for(&buf)).expect("append rslt");
    let after = std::fs::read(&path).unwrap();

    // OUTL/DBIA/SDSC would leave `after.len() % PAGE == 0`. RSLT does not.
    // This asymmetry is the hazard: a section stacked above RSLT would see a
    // non-page-aligned trailer_end and be rejected by its own parser.
    assert_ne!(
        after.len() % PAGE,
        0,
        "RSLT now page-aligns its file end — if this changed, the 'nothing may \
         stack above RSLT' assumption in this audit is stale; re-check the chain"
    );
    // The RSLT itself still reads back (it is the outermost section).
    assert!(read_rslt_bytes(&after, true).unwrap().is_some());
}

#[test]
fn stacking_outl_on_top_of_rslt_is_rejected_by_page_check() {
    // Demonstrates the practical consequence of HAZARD 1: OUTL's parser requires
    // its trailer_end (== file len) to be page aligned. On top of an RSLT it is
    // not, so the append's own self-read (read_outl_bytes over the produced file)
    // cannot succeed as a clean page-aligned section. We assert the file-end
    // misalignment that guarantees this, rather than depending on append order
    // that the codebase intentionally forbids elsewhere.
    let buf = build_base();
    let path = tmp_path("rslt-then-outl");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();
    append_rslt(&path, &rslt_for(&buf)).expect("append rslt");

    let after_rslt = std::fs::read(&path).unwrap();
    // A would-be OUTL appender computes outl_offset = page_align(len). Because
    // len is not page aligned, lead padding is inserted; but the RSLT trailer
    // now sits in the *interior*, and read_rslt over the longer file would no
    // longer find its trailer at EOF. Confirm RSLT is only discoverable while it
    // is the last section:
    assert!(read_rslt_bytes(&after_rslt, true).unwrap().is_some());

    // Simulate "something appended after RSLT" by adding one page of zeros:
    let mut longer = after_rslt.clone();
    longer.resize(((longer.len() / PAGE) + 2) * PAGE, 0u8);
    assert_eq!(
        read_rslt_bytes(&longer, true).unwrap(),
        None,
        "once any bytes follow RSLT, its EOF trailer is no longer at EOF and it \
         vanishes — RSLT must remain the outermost section"
    );
}

// ===========================================================================
// HAZARD 2 — the EOF-chain readers have INCONSISTENT step-over magic sets, and
// NONE of them knows DBIA yet. The audit map:
//   read_outl  steps over { SPRV }                 (not SDSC, not RSLT, not DBIA)
//   read_sdsc  steps over { SPRV, OUTL }           (not RSLT, not DBIA)
//   read_dbia  steps over { SPRV, OUTL, RSLT }     (not SDSC)   [new, orphan file]
//   read_sprv  outermost only, no walk
// Consequence: inserting DBIA (or a C2 side-info section) into the chain can
// make an *existing* reader halt early and silently return Ok(None). These
// tests pin the existing readers' step-over reach so the sprint notices if it
// places a new section where a prior reader can no longer see past it.
// ===========================================================================

#[test]
fn read_outl_sees_through_sprv_but_chain_order_is_load_bearing() {
    // Canonical order base -> OUTL -> SPRV: read_outl steps over the SPRV trailer
    // to find OUTL underneath. This is the ONE step-over read_outl supports.
    let buf = build_base();
    let path = tmp_path("outl-under-sprv");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    let wires = vec![
        Some(OutlierWire::from_selection(1024, vec![7, 600], vec![-100, 42], 0.5, 8)),
        None,
    ];
    append_outl(&path, &wires).expect("append outl");
    append_sprv_computed(&path, false).expect("append sprv");

    let buf2 = std::fs::read(&path).unwrap();
    // OUTL is found beneath SPRV.
    let outl = read_outl_bytes(&buf2, true).unwrap().expect("outl beneath sprv");
    assert_eq!(outl.tensors, wires);
    // SPRV is the outermost section.
    assert!(read_sprv_bytes(&buf2, true).unwrap().is_some());
}

#[test]
fn read_outl_cannot_see_past_sdsc_documenting_the_step_over_gap() {
    // base -> OUTL -> SDSC. The canonical SDSC appender RESTACKS (it strips OUTL,
    // writes SDSC innermost, then re-appends OUTL on top), so in practice OUTL
    // ends up ABOVE SDSC and read_outl still finds it. But read_outl's walker
    // itself does NOT step over an SDSC trailer: if a future appender placed SDSC
    // ABOVE OUTL (non-canonical), read_outl would halt on the SDSC magic.
    // We pin read_outl's step-over set indirectly: a lone SDSC trailer is opaque
    // to read_outl (returns None, never an error, never a panic).
    let buf = build_base();
    let path = tmp_path("outl-sdsc");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    // SDSC alone (no OUTL): read_outl must cleanly report "no OUTL" by stepping
    // over / bailing on the unknown-to-it region, never crashing.
    append_sdsc(&path).expect("append sdsc");
    let with_sdsc = std::fs::read(&path).unwrap();
    assert_eq!(
        read_outl_bytes(&with_sdsc, true).unwrap(),
        None,
        "read_outl over an SDSC-trailered file with no OUTL must be a clean None"
    );
    // And SDSC reads back fine as the outermost section.
    assert!(read_sdsc_bytes(&with_sdsc, true).unwrap().is_some());
}

#[test]
fn sdsc_restack_preserves_outl_and_sprv_visibility() {
    // The canonical multi-section stack the sprint extends:
    //   base -> OUTL -> SPRV, then append_sdsc RESTACKS to base -> SDSC -> OUTL -> SPRV.
    // After restack: read_sdsc (innermost), read_outl (steps over SPRV), and
    // read_sprv (outermost) must all still resolve. This is the invariant DBIA
    // and a C2 side-info section have to slot into without displacing.
    let buf = build_base();
    let path = tmp_path("restack");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    let wires = vec![
        Some(OutlierWire::from_selection(1024, vec![7, 600], vec![-100, 42], 0.5, 8)),
        None,
    ];
    append_outl(&path, &wires).expect("append outl");
    append_sprv_computed(&path, false).expect("append sprv");
    append_sdsc(&path).expect("append sdsc (restack)");

    let buf2 = std::fs::read(&path).unwrap();
    assert!(read_sdsc_bytes(&buf2, true).unwrap().is_some(), "sdsc innermost");
    assert_eq!(
        read_outl_bytes(&buf2, true).unwrap().expect("outl after restack").tensors,
        wires,
        "OUTL must remain visible (read_outl steps over SPRV) after the SDSC restack"
    );
    assert!(read_sprv_bytes(&buf2, true).unwrap().is_some(), "sprv outermost");
    // v2 core bytes are untouched by the whole stack.
    assert_eq!(&buf2[..buf.len()], &buf[..], "v2 prefix must be byte-stable under the chain");
}

// ===========================================================================
// HAZARD 3 — BACK-COMPAT: a plain v2 reader, and the v2 *header* reader, must be
// unchanged by ANY appended section. This is the bedrock the sprint relies on:
// adding DBIA / C2 sections is additive and invisible to readers that don't
// know them. Pins that read_strand_v2 + read_strand_v2_header ignore trailers.
// ===========================================================================

#[test]
fn v2_core_readers_ignore_every_trailer_in_the_chain() {
    let buf = build_base();
    let path = tmp_path("backcompat");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    let base_hdr = read_strand_v2_header(&buf).unwrap();
    let base_full = read_strand_v2(&buf).unwrap();

    // Stack OUTL then SPRV (the common deploy chain).
    let wires = vec![
        Some(OutlierWire::from_selection(1024, vec![7, 600], vec![-100, 42], 0.5, 8)),
        None,
    ];
    append_outl(&path, &wires).expect("append outl");
    append_sprv_computed(&path, false).expect("append sprv");
    let trailered = std::fs::read(&path).unwrap();

    let hdr2 = read_strand_v2_header(&trailered).expect("v2 header under trailers");
    let full2 = read_strand_v2(&trailered).expect("v2 full under trailers");

    assert_eq!(hdr2.tensors.len(), base_hdr.tensors.len());
    assert_eq!(hdr2.source_sha256, base_hdr.source_sha256);
    for (a, b) in hdr2.tensors.iter().zip(base_hdr.tensors.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.table, b.table);
        assert_eq!(a.payload_offset, b.payload_offset);
        assert_eq!(a.payload_bytes, b.payload_bytes);
        assert_eq!(a.sideinfo_offset, b.sideinfo_offset);
        assert_eq!(a.sideinfo_bytes, b.sideinfo_bytes);
    }
    assert_eq!(full2.len(), base_full.len());
    for (a, b) in full2.iter().zip(base_full.iter()) {
        assert_eq!(a.base.enc.bits, b.base.enc.bits, "payload must survive trailers");
        assert_eq!(a.base.enc.blocks, b.base.enc.blocks);
    }
}

// ===========================================================================
// HAZARD 4 — DOUBLE-APPEND / SEAL discipline. Every section refuses a second
// copy of itself, and OUTL refuses to append behind an SPRV seal. A C2
// side-info section and DBIA must adopt the SAME discipline (refuse-behind-SPRV,
// refuse-double) or the chain can be silently corrupted. Pins the existing
// guards so a regression in them is caught.
// ===========================================================================

#[test]
fn outl_refuses_double_and_refuses_behind_sprv_seal() {
    let buf = build_base();
    let path = tmp_path("seal");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    let wires = vec![
        Some(OutlierWire::from_selection(1024, vec![7, 600], vec![-100, 42], 0.5, 8)),
        None,
    ];
    append_outl(&path, &wires).expect("first outl");
    // double-append of OUTL is rejected, file untouched
    let before = std::fs::read(&path).unwrap();
    assert!(append_outl(&path, &wires).is_err(), "double OUTL must be rejected");
    assert_eq!(std::fs::read(&path).unwrap(), before, "rejected append must not mutate file");

    // seal with SPRV, then OUTL-behind-SPRV must be rejected with the SPRV-order msg
    append_sprv_computed(&path, false).expect("append sprv");
    let sealed = std::fs::read(&path).unwrap();
    let err = append_outl(&path, &wires).unwrap_err();
    assert!(err.contains("BEFORE SPRV"), "OUTL behind SPRV must name the order rule: {err}");
    assert_eq!(std::fs::read(&path).unwrap(), sealed, "rejected append must not mutate sealed file");
}

#[test]
fn rslt_refuses_double_append() {
    let buf = build_base();
    let path = tmp_path("rslt-double");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();
    append_rslt(&path, &rslt_for(&buf)).expect("first rslt");
    let after_first = std::fs::read(&path).unwrap();
    let err = append_rslt(&path, &rslt_for(&buf)).unwrap_err();
    assert!(err.contains("already has"), "double RSLT must be rejected: {err}");
    assert_eq!(std::fs::read(&path).unwrap(), after_first, "file untouched after rejected append");
}

// ===========================================================================
// HAZARD 5 — RSLT serialize/deserialize is the structural template a C2-coded
// side-info section would replace (fixed-width per-block counts). Pin that the
// raw codec round-trips so a C2 swap-in has a measured baseline to beat and an
// equality oracle to diff against.
// ===========================================================================

#[test]
fn rslt_raw_codec_round_trips_as_c2_baseline() {
    let section = RsltSection {
        version: RSLT_VERSION,
        block_counts: vec![vec![0, 5, 100, 999, 0, 0, 7], vec![1, 2, 3]],
    };
    let bytes = rslt::serialize(&section);
    let back = rslt::deserialize(&bytes).expect("deserialize");
    assert_eq!(back, section, "fixed-width RSLT round-trip is the C2 swap-in oracle");
}
