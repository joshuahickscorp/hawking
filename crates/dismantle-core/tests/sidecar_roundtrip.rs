//! Confirmation test for the `.dismantle` sidecar (Track 4.1/4.2): lock in the
//! three invariants the bake→load path depends on so they can't silently
//! regress.
//!
//!   (1) ROUND-TRIP: predec scale entries written by `SidecarWriter` and read
//!       back by `read_predec_entries` are byte-for-byte the same values that
//!       `predecode_q4_k_scale_table` produces in memory at load time. (The
//!       loader, `ensure_q4k_predec_cache`, uploads exactly these to a
//!       PinnedBuffer — so equality here == bit-identical decode.)
//!   (2) HEADER ROUND-TRIP: the JSON header survives the write/read cycle.
//!   (3) HASH-MISMATCH REJECTS: `check_sidecar_compatibility` flags a stale
//!       sidecar (GGUF hash differs) as fatal, and the matching-hash case is
//!       loadable — this is the guard that stops the engine from using stale
//!       predec data against a different GGUF.
//!
//! Pure CPU (no Metal context, no model load): `predecode_q4_k_scale_table` is
//! byte math and the sidecar reader/writer are plain file I/O. Gates with:
//!
//!   cargo test -p dismantle-core --test sidecar_roundtrip
//!
//! NOTE: `predecode_q4_k_scale_table` is re-exported under
//! `dismantle_core::kernels` only on macOS (it lives in the macOS-gated
//! `metal_dispatch` module), so the round-trip test is macOS-gated. The
//! compatibility-check test is platform-independent and always runs.

use dismantle_core::sidecar::{
    check_sidecar_compatibility, read_predec_entries, sidecar_path_for, SidecarCompat,
    SidecarContents, SidecarHeader, SidecarProfile, SidecarQuality, SidecarWriter,
    SIDECAR_VERSION,
};

/// Deterministic Q4_K blocks (144 bytes each) with realistic header scales and
/// packed sub-block bytes — same shape as the production GGUF tensor slices fed
/// to `predecode_q4_k_scale_table`. Mirrors the helper in
/// `tests/predec_f16_scale_table.rs`.
fn make_q4k_bytes(n_blocks: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.012_f32 + (b % 7) as f32 * 0.001;
        let dmin = ((b % 5) as f32 - 2.0) * 0.002;
        bytes[off..off + 2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = ((i * 31 + b * 17) & 0xFF) as u8;
        }
    }
    bytes
}

fn header_for(gguf_hash: &str, shader_hash: &str) -> SidecarHeader {
    SidecarHeader {
        version: SIDECAR_VERSION,
        source_gguf_hash: gguf_hash.to_string(),
        tokenizer_hash: "tok123".to_string(),
        shader_hash: shader_hash.to_string(),
        bake_profile: SidecarProfile::Fast,
        contents: SidecarContents {
            q4k_predec_scales: true,
            ..Default::default()
        },
        quality: SidecarQuality {
            quality_gate_passed: true,
            quality_gate_spec: "predec scales are bit-identical".to_string(),
            ..Default::default()
        },
        bake_device: "test-device".to_string(),
        bake_time_secs: 0,
        tier_map: None,
    }
}

/// (1)+(2): bake → load round-trip equals the in-memory predecode, and the
/// header survives the JSON round-trip. macOS-only (predecode entry point is
/// macOS-gated).
#[cfg(target_os = "macos")]
#[test]
fn bake_load_roundtrip_equals_in_memory_predecode() {
    use dismantle_core::kernels::predecode_q4_k_scale_table;

    // Two "tensors" at distinct GGUF offsets, like q_proj and ffn_down.
    let t0 = make_q4k_bytes(48);
    let t1 = make_q4k_bytes(80);
    let off0: u64 = 0x1000;
    let off1: u64 = 0x9abc;

    // In-memory predecode — the source of truth (what ensure_q4k_predec_cache
    // computes when there is no sidecar).
    let mem0 = predecode_q4_k_scale_table(&t0);
    let mem1 = predecode_q4_k_scale_table(&t1);
    assert_eq!(mem0.len(), 48 * 16, "predec table is 16 f32 per 144-byte block");
    assert_eq!(mem1.len(), 80 * 16);

    // Bake: write exactly those tables into a sidecar (what bake_sidecar_predec
    // does — it calls the same predecode fn on the same bytes).
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.dismantle");
    let header = header_for("deadbeef_gguf_hash", "shader_abc");
    let writer = SidecarWriter {
        path: path.clone(),
        predec_entries: vec![(off0, mem0.clone()), (off1, mem1.clone())],
        header: header.clone(),
    };
    let written = writer.write().expect("sidecar write");
    assert!(written > 0, "writer reported {written} bytes");

    // Load: read the entries back.
    let (read_header, entries) = read_predec_entries(&path).expect("read predec entries");

    // (2) Header round-trips through JSON.
    assert_eq!(read_header.version, header.version);
    assert_eq!(read_header.source_gguf_hash, header.source_gguf_hash);
    assert_eq!(read_header.tokenizer_hash, header.tokenizer_hash);
    assert_eq!(read_header.shader_hash, header.shader_hash);
    assert_eq!(read_header.bake_profile, header.bake_profile);
    assert!(read_header.contents.q4k_predec_scales);

    // (1) Entries are byte-for-byte the in-memory predecode, keyed by offset.
    let map: std::collections::HashMap<usize, Vec<f32>> = entries.into_iter().collect();
    assert_eq!(map.len(), 2, "both tensors round-trip");
    let r0 = map.get(&(off0 as usize)).expect("entry off0");
    let r1 = map.get(&(off1 as usize)).expect("entry off1");
    assert_eq!(r0.len(), mem0.len());
    assert_eq!(r1.len(), mem1.len());
    // bit-identical: compare raw f32 bits, not approximate.
    for (i, (&a, &b)) in r0.iter().zip(mem0.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "off0 scale[{i}] not bit-identical");
    }
    for (i, (&a, &b)) in r1.iter().zip(mem1.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "off1 scale[{i}] not bit-identical");
    }
}

/// (3) HASH-MISMATCH REJECTS: a sidecar baked against one GGUF must be flagged
/// fatal when loaded against a different GGUF; matching hash is loadable.
/// Platform-independent (pure compatibility check).
#[test]
fn hash_mismatch_is_fatal_and_match_is_loadable() {
    let header = header_for("hash_of_gguf_A", "shader_A");

    // Matching GGUF + shader → Compatible, loadable, not fatal.
    let ok = check_sidecar_compatibility(&header, "hash_of_gguf_A", "shader_A");
    assert!(matches!(ok, SidecarCompat::Compatible), "got {ok:?}");
    assert!(ok.is_loadable());
    assert!(!ok.is_fatal());

    // Different GGUF hash → GgufHashMismatch, fatal, NOT loadable. This is the
    // guard that stops stale predec scales from being applied to a new GGUF.
    let stale = check_sidecar_compatibility(&header, "hash_of_gguf_B", "shader_A");
    assert!(
        matches!(stale, SidecarCompat::GgufHashMismatch { .. }),
        "stale GGUF must be a hash mismatch, got {stale:?}"
    );
    assert!(stale.is_fatal(), "GGUF hash mismatch must be fatal");
    assert!(!stale.is_loadable(), "stale sidecar must NOT load");

    // Shader-hash mismatch is non-fatal (data still valid) but flagged loadable.
    let shader_drift = check_sidecar_compatibility(&header, "hash_of_gguf_A", "shader_B");
    assert!(
        matches!(shader_drift, SidecarCompat::ShaderHashMismatch { .. }),
        "got {shader_drift:?}"
    );
    assert!(shader_drift.is_loadable(), "shader drift is non-fatal");
    assert!(!shader_drift.is_fatal());

    // Version newer than this binary understands → fatal.
    let mut future = header.clone();
    future.version = SIDECAR_VERSION + 1;
    let too_new = check_sidecar_compatibility(&future, "hash_of_gguf_A", "shader_A");
    assert!(matches!(too_new, SidecarCompat::VersionTooNew { .. }), "got {too_new:?}");
    assert!(too_new.is_fatal());
}

/// Bad magic bytes are rejected by the reader (corrupt/foreign file guard).
/// Platform-independent.
#[test]
fn bad_magic_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("not_a_sidecar.dismantle");
    std::fs::write(&path, b"NOTDSMTL........junk").expect("write junk");
    let err = read_predec_entries(&path);
    assert!(err.is_err(), "reader must reject a file with bad magic");
}

/// `sidecar_path_for` derives `<stem>.dismantle` from a GGUF path.
/// Platform-independent.
#[test]
fn sidecar_path_derivation() {
    let p = sidecar_path_for(std::path::Path::new("models/qwen2.5-3b-q4_k_m.gguf"));
    assert_eq!(p, std::path::PathBuf::from("models/qwen2.5-3b-q4_k_m.dismantle"));
}
