//! Admission-trust receipt tests. Isolated tempfile trees only; never the
//! sealed 43-layer artifact.

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4TensorMetadata, DeepSeekV4VerifyMode,
    ADMISSION_TRUST_RECEIPT_NAME, ADMISSION_TRUST_SCHEMA,
};
use hawking_core::gravity_deepseek_v4_admission_trust::{
    file_identity, load_admission_receipt, DeepSeekV4AdmissionLoad,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn isolated_tensor(name: &str, segment: DeepSeekV4Segment) -> DeepSeekV4TensorMetadata {
    let bytes = segment.bytes;
    DeepSeekV4TensorMetadata {
        name: name.to_owned(),
        dtype: "I8".to_owned(),
        shape: vec![bytes],
        data_offsets: [0, bytes],
        bytes,
        source_file_start: 0,
        source_file_end: bytes,
        source_shard: "model-00001-of-00046.safetensors".to_owned(),
        segments: vec![segment],
    }
}

fn bind(
    root: &Path,
    name: &str,
    payload: &[u8],
    mode: DeepSeekV4VerifyMode,
) -> DeepSeekV4FullStreamReader {
    let (segment, spec) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(root, payload)
            .expect("write isolated chunk");
    let mut tensors = BTreeMap::new();
    tensors.insert(name.to_owned(), isolated_tensor(name, segment));
    DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture_with_verify_mode(
        root,
        tensors,
        [spec],
        mode,
    )
    .expect("bind isolated fixture")
}

fn rebind(
    root: &Path,
    name: &str,
    segment: DeepSeekV4Segment,
    mode: DeepSeekV4VerifyMode,
) -> DeepSeekV4FullStreamReader {
    let spec = hawking_core::gravity_deepseek_v4::DeepSeekV4ChunkSpec {
        relative: segment.chunk_relpath.clone(),
        sha256: segment.sha256.clone(),
        bytes: segment.bytes,
    };
    let mut tensors = BTreeMap::new();
    tensors.insert(name.to_owned(), isolated_tensor(name, segment));
    DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture_with_verify_mode(
        root,
        tensors,
        [spec],
        mode,
    )
    .expect("rebind isolated fixture")
}

#[cfg(unix)]
fn set_mtime_ns(path: &Path, mtime_ns: i128) {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    #[repr(C)]
    struct TimeSpec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    extern "C" {
        fn utimensat(dirfd: i32, path: *const i8, times: *const TimeSpec, flag: i32) -> i32;
    }
    const AT_FDCWD: i32 = -2;
    let sec = (mtime_ns / 1_000_000_000) as i64;
    let nsec = (mtime_ns % 1_000_000_000) as i64;
    let times = [
        TimeSpec {
            tv_sec: sec,
            tv_nsec: nsec,
        },
        TimeSpec {
            tv_sec: sec,
            tv_nsec: nsec,
        },
    ];
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("path cstring");
    let rc = unsafe { utimensat(AT_FDCWD, c_path.as_ptr(), times.as_ptr(), 0) };
    assert_eq!(rc, 0, "utimensat failed: {}", std::io::Error::last_os_error());
}

#[test]
fn valid_receipt_fast_path_skips_hash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    let seal = sealer.seal_admission_trust().expect("seal");
    assert!(seal.path.ends_with(ADMISSION_TRUST_RECEIPT_NAME));
    assert_eq!(seal.chunk_count, 1);
    assert_eq!(seal.bytes_hashed, payload.len() as u64);

    let trusted = bind(
        tmp.path(),
        "probe.weight",
        &payload,
        DeepSeekV4VerifyMode::Admission,
    );
    assert!(trusted.chunk_verification_stats().admission_receipt_loaded);
    let bytes = trusted
        .read_verified_full("probe.weight", payload.len())
        .expect("trusted read");
    assert_eq!(bytes, payload);
    let stats = trusted.chunk_verification_stats();
    assert_eq!(stats.hash_invocations, 0, "valid receipt must skip SHA-256");
    assert_eq!(stats.admission_trust_hits, 1);
    assert_eq!(stats.admission_trust_fallbacks, 0);
    assert_eq!(stats.bytes_hashed, 0);
    assert_eq!(stats.chunks_verified, 1);
}

#[test]
fn missing_receipt_falls_back_to_hash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"no receipt here".to_vec();
    let reader = bind(
        tmp.path(),
        "probe.weight",
        &payload,
        DeepSeekV4VerifyMode::Admission,
    );
    assert!(!reader.chunk_verification_stats().admission_receipt_loaded);
    let bytes = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("hash fallback");
    assert_eq!(bytes, payload);
    let stats = reader.chunk_verification_stats();
    assert_eq!(stats.hash_invocations, 1);
    assert_eq!(stats.admission_trust_hits, 0);
    assert_eq!(stats.bytes_hashed, payload.len() as u64);
}

#[test]
fn corrupted_receipt_is_rejected_and_hashes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"receipt will be corrupted".to_vec();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let path = tmp.path().join(ADMISSION_TRUST_RECEIPT_NAME);
    let mut raw = fs::read(&path).expect("read receipt");
    let last = raw.len() - 2;
    raw[last] ^= 0x01;
    fs::write(&path, &raw).expect("corrupt receipt");

    let loaded = load_admission_receipt(
        tmp.path(),
        sealer.manifest_seal_sha256(),
        sealer.content_addressed_chunk_sha256(),
        1,
        payload.len() as u64,
    );
    match loaded {
        DeepSeekV4AdmissionLoad::Rejected(reason) => {
            assert!(
                reason.contains("seal mismatch") || reason.contains("not valid JSON"),
                "unexpected reject: {reason}"
            );
        }
        other => panic!("expected rejected receipt, got {other:?}"),
    }

    let reader = bind(
        tmp.path(),
        "probe.weight",
        &payload,
        DeepSeekV4VerifyMode::Admission,
    );
    assert!(!reader.chunk_verification_stats().admission_receipt_loaded);
    let bytes = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("fallback hash after corrupt receipt");
    assert_eq!(bytes, payload);
    assert_eq!(reader.chunk_verification_stats().hash_invocations, 1);
}

#[test]
fn per_chunk_invariant_mismatch_falls_back_to_hash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = vec![0x11u8; 2048];
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    let (segment, _) = {
        // Re-discover the written segment from a trusted read path.
        let first = sealer
            .read_verified_full("probe.weight", payload.len())
            .expect("sealer read");
        assert_eq!(first, payload);
        sealer.seal_admission_trust().expect("seal");
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(tmp.path(), &payload)
            .expect("existing chunk")
    };

    // Same bytes, restored content, but a newer mtime: identity fails, hash succeeds.
    let chunk_path = tmp.path().join(&segment.chunk_relpath);
    fs::write(&chunk_path, &payload).expect("touch mtime");
    let reader = rebind(
        tmp.path(),
        "probe.weight",
        segment,
        DeepSeekV4VerifyMode::Admission,
    );
    assert!(reader.chunk_verification_stats().admission_receipt_loaded);
    let bytes = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("mtime mismatch must hash, not fail");
    assert_eq!(bytes, payload);
    let stats = reader.chunk_verification_stats();
    assert_eq!(stats.hash_invocations, 1);
    assert_eq!(stats.admission_trust_hits, 0);
    assert_eq!(stats.admission_trust_fallbacks, 1);
}

#[test]
fn tamper_after_seal_is_detected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"flip one byte after seal".to_vec();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let (segment, _) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(tmp.path(), &payload)
            .expect("existing");
    let chunk_path = tmp.path().join(&segment.chunk_relpath);
    let mut flipped = payload.clone();
    flipped[0] ^= 0xff;
    fs::write(&chunk_path, &flipped).expect("tamper");

    let reader = rebind(
        tmp.path(),
        "probe.weight",
        segment,
        DeepSeekV4VerifyMode::Admission,
    );
    let err = reader
        .read_verified_full("probe.weight", payload.len())
        .expect_err("tamper must hard-fail");
    let message = format!("{err}");
    assert!(
        message.contains("sha256 differs from sealed segment digest"),
        "unexpected error: {message}"
    );
}

#[test]
fn delete_and_truncate_hard_fail() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = vec![0x5au8; 1024];
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let (segment, spec) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(tmp.path(), &payload)
            .expect("existing");
    let chunk_path = tmp.path().join(&segment.chunk_relpath);

    // Bind first (size still matches), then truncate the live file.
    let live = rebind(
        tmp.path(),
        "probe.weight",
        segment.clone(),
        DeepSeekV4VerifyMode::Admission,
    );
    fs::write(&chunk_path, &payload[..64]).expect("truncate");
    let err = live
        .read_verified_full("probe.weight", payload.len())
        .expect_err("truncate must hard-fail");
    assert!(
        format!("{err}").contains("byte size") || format!("{err}").contains("differs from sealed"),
        "unexpected truncate error: {err}"
    );

    // Restore so a fresh bind can open, then delete the live file.
    fs::write(&chunk_path, &payload).expect("restore");
    let live = rebind(
        tmp.path(),
        "probe.weight",
        segment,
        DeepSeekV4VerifyMode::Admission,
    );
    let _ = spec;
    fs::remove_file(&chunk_path).expect("delete");
    let err = live
        .read_verified_full("probe.weight", payload.len())
        .expect_err("delete must hard-fail");
    assert!(
        format!("{err}").contains("cannot inspect") || format!("{err}").contains("must be a regular"),
        "unexpected delete error: {err}"
    );
}

#[test]
fn full_mode_hashes_even_with_valid_receipt() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"full mode ignores receipt".to_vec();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let reader = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    let bytes = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("full read");
    assert_eq!(bytes, payload);
    let stats = reader.chunk_verification_stats();
    assert_eq!(stats.hash_invocations, 1);
    assert_eq!(stats.admission_trust_hits, 0);
    assert!(!stats.admission_receipt_loaded);
}

#[test]
fn stale_receipt_wrong_digest_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"stale digest".to_vec();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let loaded = load_admission_receipt(
        tmp.path(),
        sealer.manifest_seal_sha256(),
        "00".repeat(32).as_str(),
        1,
        payload.len() as u64,
    );
    match loaded {
        DeepSeekV4AdmissionLoad::Rejected(reason) => {
            assert!(
                reason.contains("content_addressed_chunk_sha256"),
                "unexpected reject: {reason}"
            );
        }
        other => panic!("expected stale reject, got {other:?}"),
    }
}

#[test]
fn clone_view_remaps_to_sealed_inode() {
    let src = tempfile::tempdir().expect("src");
    let view = tempfile::tempdir().expect("view");
    let payload: Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
    let sealer = bind(src.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let (segment, spec) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(src.path(), &payload)
            .expect("src chunk");

    let dest = view.path().join(&segment.chunk_relpath);
    fs::create_dir_all(dest.parent().unwrap()).expect("view prefix");
    fs::copy(src.path().join(&segment.chunk_relpath), &dest).expect("copy clone");
    fs::copy(
        src.path().join(ADMISSION_TRUST_RECEIPT_NAME),
        view.path().join(ADMISSION_TRUST_RECEIPT_NAME),
    )
    .expect("copy receipt");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "probe.weight".to_owned(),
        isolated_tensor("probe.weight", segment),
    );
    let reader = DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture_with_verify_mode(
        view.path(),
        tensors,
        [spec],
        DeepSeekV4VerifyMode::Admission,
    )
    .expect("bind view");
    assert!(reader.chunk_verification_stats().admission_receipt_loaded);
    let bytes = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("remap read");
    assert_eq!(bytes, payload);
    let stats = reader.chunk_verification_stats();
    assert_eq!(
        stats.hash_invocations, 0,
        "clone view must remap to the sealed inode and skip hash"
    );
    assert_eq!(stats.admission_trust_hits, 1);
}

#[cfg(unix)]
#[test]
fn metadata_preserving_bitflip_is_the_residual_window() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"mtime-preserving flip".to_vec();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let (segment, _) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(tmp.path(), &payload)
            .expect("existing");
    let chunk_path = tmp.path().join(&segment.chunk_relpath);
    let before = file_identity(&chunk_path, "chunk").expect("stat before");
    let mut flipped = payload.clone();
    flipped[3] ^= 0x01;
    fs::write(&chunk_path, &flipped).expect("flip");
    set_mtime_ns(&chunk_path, before.mtime_ns);
    let after = file_identity(&chunk_path, "chunk").expect("stat after");
    assert_eq!(after.bytes, before.bytes);
    assert_eq!(after.mtime_ns, before.mtime_ns);
    assert_eq!(after.inode, before.inode);

    let trusted = rebind(
        tmp.path(),
        "probe.weight",
        segment.clone(),
        DeepSeekV4VerifyMode::Admission,
    );
    let trusted_bytes = trusted
        .read_verified_full("probe.weight", payload.len())
        .expect("admission path cannot see a metadata-preserving flip");
    assert_eq!(
        trusted_bytes, flipped,
        "residual window: admission trusts identity-matched bytes"
    );
    assert_eq!(trusted.chunk_verification_stats().hash_invocations, 0);

    let full = rebind(tmp.path(), "probe.weight", segment, DeepSeekV4VerifyMode::Full);
    let err = full
        .read_verified_full("probe.weight", payload.len())
        .expect_err("VERIFY=full must close the residual window");
    assert!(format!("{err}").contains("sha256 differs from sealed segment digest"));
}

#[test]
fn receipt_schema_is_the_v1_contract() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"schema contract".to_vec();
    let sealer = bind(tmp.path(), "probe.weight", &payload, DeepSeekV4VerifyMode::Full);
    sealer.seal_admission_trust().expect("seal");
    let raw = fs::read(tmp.path().join(ADMISSION_TRUST_RECEIPT_NAME)).expect("read");
    let value: serde_json::Value = serde_json::from_slice(&raw).expect("json");
    assert_eq!(
        value.get("schema").and_then(|v| v.as_str()),
        Some(ADMISSION_TRUST_SCHEMA)
    );
    assert!(value.get("content_addressed_chunk_sha256").is_some());
    assert!(value.get("table_sha256").is_some());
    assert!(value.get("seal_sha256").is_some());
    assert!(value.get("verifier_version").is_some());
    assert!(value.get("chunks").and_then(|v| v.as_array()).is_some());
    assert!(value.get("threat_model").is_some());
}
