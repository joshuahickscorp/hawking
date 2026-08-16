//! Artifact-index format, validity binding, and JSON fallback.

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4SourceIdentity,
    DeepSeekV4TensorMetadata, DeepSeekV4VerifyMode,
};
use hawking_core::gravity_deepseek_v4_admission_trust::DeepSeekV4ChunkFileIdentity;
use hawking_core::gravity_deepseek_v4_artifact_index::{
    load_artifact_index, tensor_maps_structurally_equal, write_artifact_index, DeepSeekV4IndexLoad,
    IndexBuildInput, ARTIFACT_INDEX_NAME,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_chunk(root: &Path, payload: &[u8]) -> (String, String) {
    let digest = sha256_hex(payload);
    let relative = format!("chunks/{}/{}", &digest[..2], digest);
    let path = root.join(&relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, payload).unwrap();
    (relative, digest)
}

fn identity_of(root: &Path, relative: &str) -> DeepSeekV4ChunkFileIdentity {
    hawking_core::gravity_deepseek_v4_admission_trust::file_identity(&root.join(relative), "chunk")
        .unwrap()
}

fn fake_sealed_tree() -> (tempfile::TempDir, IndexBuildInputOwned) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::write(
        root.join("manifest.json"),
        br#"{"schema":"test","tensors":{}}"#,
    )
    .unwrap();
    fs::write(root.join("stream-ranges.jsonl"), b"{\"i\":0}\n").unwrap();
    fs::write(root.join("stream-journal.json"), b"{\"ok\":true}\n").unwrap();

    let weight: Vec<u8> = (0u8..=255).cycle().take(32 * 32).collect();
    let scale: Vec<u8> = vec![0x70u8; 32 * 2];
    let (w_rel, w_sha) = write_chunk(root, &weight);
    let (s_rel, s_sha) = write_chunk(root, &scale);

    let mut w_id = identity_of(root, &w_rel);
    w_id.key = w_rel.clone();
    let mut s_id = identity_of(root, &s_rel);
    s_id.key = s_rel.clone();

    let w_tensor = DeepSeekV4TensorMetadata {
        name: "probe.weight".into(),
        dtype: "I8".into(),
        shape: vec![32, 32],
        data_offsets: [0, 1024],
        bytes: 1024,
        source_file_start: 0,
        source_file_end: 1024,
        source_shard: "model-00001-of-00046.safetensors".into(),
        segments: vec![DeepSeekV4Segment {
            bytes: 1024,
            chunk_relpath: w_rel.clone(),
            sha256: w_sha.clone(),
            source_file_start: 0,
            source_file_end: 1024,
            tensor_start: 0,
            tensor_end: 1024,
            row_start: 0,
            row_count: 32,
        }],
    };
    let s_tensor = DeepSeekV4TensorMetadata {
        name: "probe.scale".into(),
        dtype: "F8_E8M0".into(),
        shape: vec![32, 2],
        data_offsets: [0, 64],
        bytes: 64,
        source_file_start: 0,
        source_file_end: 64,
        source_shard: "model-00001-of-00046.safetensors".into(),
        segments: vec![DeepSeekV4Segment {
            bytes: 64,
            chunk_relpath: s_rel.clone(),
            sha256: s_sha.clone(),
            source_file_start: 0,
            source_file_end: 64,
            tensor_start: 0,
            tensor_end: 64,
            row_start: 0,
            row_count: 32,
        }],
    };
    let mut tensors = BTreeMap::new();
    tensors.insert("probe.scale".into(), s_tensor);
    tensors.insert("probe.weight".into(), w_tensor);
    let mut chunks = BTreeMap::new();
    chunks.insert(w_rel.clone(), (w_sha, 1024u64));
    chunks.insert(s_rel.clone(), (s_sha, 64u64));
    let mut identities = BTreeMap::new();
    identities.insert(w_rel.clone(), w_id);
    identities.insert(s_rel.clone(), s_id);
    let mut meta = BTreeMap::new();
    meta.insert("config.json".into(), "ab".repeat(32));

    let owned = IndexBuildInputOwned {
        source: DeepSeekV4SourceIdentity {
            repository: "deepseek-ai/DeepSeek-V4-Flash".into(),
            revision: "60d8d70770c6776ff598c94bb586a859a38244f1".into(),
        },
        manifest_seal: "11".repeat(32),
        manifest_file: "22".repeat(32),
        restart_seal: "33".repeat(32),
        chunk_digest: "44".repeat(32),
        tensors,
        chunks,
        identities,
        source_metadata: meta,
        table_sha: "55".repeat(32),
        verifier: "0.2.2".into(),
    };
    (tmp, owned)
}

struct IndexBuildInputOwned {
    source: DeepSeekV4SourceIdentity,
    manifest_seal: String,
    manifest_file: String,
    restart_seal: String,
    chunk_digest: String,
    tensors: BTreeMap<String, DeepSeekV4TensorMetadata>,
    chunks: BTreeMap<String, (String, u64)>,
    identities: BTreeMap<String, DeepSeekV4ChunkFileIdentity>,
    source_metadata: BTreeMap<String, String>,
    table_sha: String,
    verifier: String,
}

impl IndexBuildInputOwned {
    fn input<'a>(&'a self, root: &'a Path) -> IndexBuildInput<'a> {
        IndexBuildInput {
            source_root: root,
            _reader_root: root,
            source: &self.source,
            manifest_seal_sha256: &self.manifest_seal,
            manifest_file_sha256: &self.manifest_file,
            restart_seal_sha256: &self.restart_seal,
            content_addressed_chunk_sha256: &self.chunk_digest,
            tensor_bytes: 1088,
            tensors: &self.tensors,
            chunks: &self.chunks,
            identities: &self.identities,
            source_metadata_sha256: &self.source_metadata,
            table_sha256: &self.table_sha,
            sealed_at_unix_ms: 1,
            verifier_version: &self.verifier,
        }
    }
}

#[test]
fn index_roundtrip_matches_tensor_map() {
    let (tmp, owned) = fake_sealed_tree();
    let seal = write_artifact_index(owned.input(tmp.path())).expect("write index");
    assert!(seal.path.ends_with(ARTIFACT_INDEX_NAME));
    assert!(seal.bytes > 512);
    match load_artifact_index(tmp.path()) {
        DeepSeekV4IndexLoad::Loaded(loaded) => {
            tensor_maps_structurally_equal(&owned.tensors, &loaded.tensors).unwrap();
            assert_eq!(owned.chunks, loaded.chunks);
            assert_eq!(loaded.admission.chunk_count, 2);
            let reader = DeepSeekV4FullStreamReader::try_admit_from_artifact_index(
                tmp.path(),
                DeepSeekV4VerifyMode::Admission,
            )
            .expect("admit")
            .expect("index hit");
            assert!(reader.chunk_verification_stats().artifact_index_loaded);
            assert!(reader.chunk_verification_stats().admission_receipt_loaded);
            reader.structural_map_eq(&reader).unwrap();
        }
        other => panic!("expected loaded index, got {other:?}"),
    }
}

#[test]
fn truncated_index_falls_back() {
    let (tmp, owned) = fake_sealed_tree();
    let seal = write_artifact_index(owned.input(tmp.path())).expect("write");
    let raw = fs::read(&seal.path).unwrap();
    fs::write(&seal.path, &raw[..200]).unwrap();
    match load_artifact_index(tmp.path()) {
        DeepSeekV4IndexLoad::Rejected(reason) => {
            assert!(
                reason.contains("truncated")
                    || reason.contains("escapes")
                    || reason.contains("seal"),
                "{reason}"
            );
        }
        other => panic!("expected reject, got {other:?}"),
    }
    assert!(DeepSeekV4FullStreamReader::try_admit_from_artifact_index(
        tmp.path(),
        DeepSeekV4VerifyMode::Admission,
    )
    .unwrap()
    .is_none());
}

#[test]
fn corrupt_index_falls_back() {
    let (tmp, owned) = fake_sealed_tree();
    let seal = write_artifact_index(owned.input(tmp.path())).expect("write");
    let mut raw = fs::read(&seal.path).unwrap();
    let last = raw.len() - 1;
    raw[last] ^= 0xff;
    fs::write(&seal.path, &raw).unwrap();
    match load_artifact_index(tmp.path()) {
        DeepSeekV4IndexLoad::Rejected(reason) => {
            assert!(
                reason.contains("seal") || reason.contains("magic"),
                "{reason}"
            );
        }
        other => panic!("expected reject, got {other:?}"),
    }
}

#[test]
fn stale_mtime_falls_back() {
    let (tmp, owned) = fake_sealed_tree();
    write_artifact_index(owned.input(tmp.path())).expect("write");
    fs::write(
        tmp.path().join("manifest.json"),
        br#"{"schema":"test","tensors":{},"x":1}"#,
    )
    .unwrap();
    match load_artifact_index(tmp.path()) {
        DeepSeekV4IndexLoad::Rejected(reason) => {
            assert!(
                reason.contains("identity") || reason.contains("mtime"),
                "{reason}"
            );
        }
        other => panic!("expected stale reject, got {other:?}"),
    }
}

#[test]
fn deleted_index_falls_back() {
    let (tmp, owned) = fake_sealed_tree();
    let seal = write_artifact_index(owned.input(tmp.path())).expect("write");
    fs::remove_file(&seal.path).unwrap();
    match load_artifact_index(tmp.path()) {
        DeepSeekV4IndexLoad::Missing => {}
        other => panic!("expected missing, got {other:?}"),
    }
    assert!(DeepSeekV4FullStreamReader::try_admit_from_artifact_index(
        tmp.path(),
        DeepSeekV4VerifyMode::Admission,
    )
    .unwrap()
    .is_none());
}
