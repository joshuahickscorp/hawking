//! Sprint-integration AUDIT tests for the EOF-chained `.strand` v2 side sections.
//!
//! These are NEW, standalone tests (they touch no shared source files). They pin the
//! integration hazards found while auditing the encode path for the sprint levers
//! C2 (sideinfo rANS), de-bias DBIA, and the OUTL/SPRV stack. Each test is written to
//! PASS on the *correct* end state and to FAIL loudly on the wire-order bug that the
//! current `outlier_wire::read_outl_bytes` walk would produce if DBIA is appended
//! between OUTL and SPRV (the order the `debias_wire.rs` docstring calls canonical).
//!
//! NOTE: SDSQ (sprint Lever 1, `sideinfo_wire` over `sideinfo_rans`) IS now declared in
//! `strand-quant/src/lib.rs`, so the section-sandwich test below is ACTIVE and exercises
//! the real silent-drop hazard with SDSQ in the chain. DBIA (`debias_wire`) is still NOT
//! declared on the `media-waves` branch; SDSQ is the live analog and the regression
//! backstop for the chain walk. The OUTL <-> SPRV stacking test also runs today.

use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2, PAGE};
use strand_quant::outlier_wire::{append_outl, read_outl_bytes, OutlierWire};
use strand_quant::provenance_io::{append_sprv_computed, read_sprv_bytes};
use strand_quant::sideinfo_wire::{append_sdsq, read_sdsq_bytes};
use strand_quant::trellis::TrellisConfig;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "strand-sprint-audit-{tag}-{}-{}.strand",
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

fn weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
}

fn two_tensor_archive() -> Vec<u8> {
    let cfg = TrellisConfig::for_bpw(3.0);
    let enc_a = encode_tensor_with(&weights(1024, 11), &cfg, &EncodeOpts::default());
    let enc_b = encode_tensor_with(&weights(900, 23), &cfg, &EncodeOpts::default());
    let shape_a = [4u64, 256u64];
    let shape_b = [900u64];
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
    write_strand_v2(&tensors, [9u8; 32], true).expect("write v2")
}

fn sample_outl() -> Vec<Option<OutlierWire>> {
    vec![
        Some(OutlierWire::from_selection(1024, vec![700, 3, 511], vec![-127, 5, 127], 0.3125, 8)),
        None,
    ]
}

/// Every block's scale_q in archive tensor/block order — the stream SDSQ codes.
fn archive_scale_q(buf: &[u8]) -> Vec<i32> {
    let hdr = strand_quant::format::read_strand_v2_header(buf).unwrap();
    hdr.tensors.iter().flat_map(|t| t.table.iter().map(|r| r.scale_q)).collect()
}

/// BASELINE (must pass today): OUTL then SPRV. Both sections must remain readable
/// after stacking, and the v2 bytes must be untouched. This is the contract every
/// new section (DBIA, an eventual C2 section) must also satisfy.
#[test]
fn outl_then_sprv_both_readable() {
    let buf = two_tensor_archive();
    let path = tmp_path("outl-sprv");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    append_outl(&path, &sample_outl()).expect("append outl");
    append_sprv_computed(&path, false).expect("append sprv on outl-trailered file");

    let on_disk = std::fs::read(&path).unwrap();
    // SPRV is the OUTERMOST seal: parse_sprv_section requires prov_off+prov_bytes+16 ==
    // file_len, and the SPRV appender pads only its START to a page (no tail pad). So a
    // sealed file is NOT page-aligned at EOF (matches `outl_then_sprv_is_the_canonical_
    // live_stack` in sprint_section_stacking.rs, which asserts the same `!= 0`).
    assert_ne!(on_disk.len() % PAGE, 0, "a SPRV-sealed file ends at the seal, not a page boundary");
    assert_eq!(&on_disk[..buf.len()], &buf[..], "stacking must not touch v2 bytes");

    let outl = read_outl_bytes(&on_disk, true).expect("outl read").expect("outl present under sprv");
    assert_eq!(outl.tensors, sample_outl(), "OUTL must survive under the SPRV trailer");
    let sprv = read_sprv_bytes(&on_disk, true).expect("sprv read").expect("sprv outermost");
    assert_eq!(sprv.tensors.len(), 2);
}

/// AUDIT HAZARD (the load-bearing one). `outlier_wire::read_outl_bytes` walks the EOF
/// trailer chain stepping over **SPRV only** (it returns None on any other non-OUTL
/// magic). The `debias_wire.rs` docstring declares the canonical append order to be
/// `base -> OUTL -> DBIA -> SPRV`. With DBIA sandwiched between OUTL and SPRV, the OUTL
/// walk sees: SPRV (step over) -> DBIA magic -> "not OUTL, not SPRV" -> `Ok(None)`.
/// OUTL then reads as ABSENT with no error, and `loader::StrandModel::from_mmap` (which
/// calls `read_outl_bytes(.., true)`) silently drops the outlier channel -> wrong
/// dequantized weights, no diagnostic.
///
/// This is the regression that pins the fix that landed for sprint Lever 1. The chosen
/// fix is option (a): teach `read_outl_bytes` (and `selfdesc`'s SDSC walkers) to step
/// over the SDSQ magic, exactly as `read_dbia_bytes` steps over SPRV/OUTL/RSLT. The
/// canonical append order is `base -> [OUTL] -> SDSQ -> SPRV` (SPRV is the outermost
/// seal; SDSQ is a data section under it), so OUTL ends up beneath SDSQ and the OUTL
/// reader MUST step over the SDSQ trailer or it silently drops the outlier channel.
///
/// ACTIVE: `sideinfo_wire` is declared (`pub mod sideinfo_wire;`). This is the live
/// in-tree analog of the DBIA-sandwich case (DBIA is still orphaned on `media-waves`).
#[test]
fn outl_survives_sdsq_sandwiched_under_sprv() {
    let buf = two_tensor_archive();
    let path = tmp_path("outl-sdsq-sprv");
    let _g = TmpFile(path.clone());
    std::fs::write(&path, &buf).unwrap();

    // base -> OUTL -> SDSQ -> SPRV (SDSQ sandwiched between OUTL and the seal).
    append_outl(&path, &sample_outl()).expect("outl");
    append_sdsq(&path, &archive_scale_q(&buf)).expect("sdsq between outl and sprv");
    append_sprv_computed(&path, false).expect("sprv outermost");

    let on_disk = std::fs::read(&path).unwrap();
    // ALL THREE must read back. Before teaching read_outl_bytes the SDSQ magic, the
    // next assertion failed (OUTL read as None — the silent-drop the audit warns of).
    let outl = read_outl_bytes(&on_disk, true).unwrap();
    assert!(
        outl.is_some(),
        "OUTL must remain readable with SDSQ sandwiched under SPRV (read_outl_bytes \
         must step over the SDSQ trailer, else it silently drops the outlier channel)"
    );
    assert_eq!(outl.unwrap().tensors, sample_outl(), "OUTL payload intact under SDSQ+SPRV");
    let sdsq = read_sdsq_bytes(&on_disk, true).unwrap();
    assert!(sdsq.is_some(), "SDSQ readable beneath the SPRV seal");
    assert_eq!(sdsq.unwrap().scale_q, archive_scale_q(&buf), "SDSQ scale_q intact");
    assert!(read_sprv_bytes(&on_disk, true).unwrap().is_some(), "SPRV readable (outermost)");

    // v2 core bytes (the seek table included) are byte-stable under the whole stack.
    assert_eq!(&on_disk[..buf.len()], &buf[..], "v2 prefix must be untouched by the chain");
}
