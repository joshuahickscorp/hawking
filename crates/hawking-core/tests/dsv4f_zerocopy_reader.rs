//! Integrity and zero-copy tests for the DeepSeek-V4 verified-once reader.
//!
//! These tests never touch the sealed artifact. Every chunk lives in a
//! tempfile tree and is discarded with the test.

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4TensorMetadata,
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

fn bind_one_tensor(
    root: &Path,
    name: &str,
    payload: &[u8],
) -> (DeepSeekV4FullStreamReader, DeepSeekV4Segment) {
    let (segment, spec) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(root, payload)
            .expect("write isolated chunk");
    let mut tensors = BTreeMap::new();
    tensors.insert(name.to_owned(), isolated_tensor(name, segment.clone()));
    let reader = DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture(root, tensors, [spec])
        .expect("bind isolated fixture");
    (reader, segment)
}

#[test]
fn corrupted_chunk_still_raises() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"verified-once integrity payload for dsv4f reader";
    let (segment, spec) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(tmp.path(), payload)
            .expect("write isolated chunk");
    let chunk_path = tmp.path().join(&spec.relative);

    let mut corrupted = payload.to_vec();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0xff;
    fs::write(&chunk_path, &corrupted).expect("mutate temp-copy chunk");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "probe.weight".to_owned(),
        isolated_tensor("probe.weight", segment),
    );
    let reader =
        DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture(tmp.path(), tensors, [spec])
            .expect("bind isolated fixture over mutated copy");

    let err = reader
        .read_verified_full("probe.weight", payload.len())
        .expect_err("corrupted chunk must hard-fail");
    let message = format!("{err}");
    assert!(
        message.contains("sha256 differs from sealed segment digest"),
        "unexpected error: {message}"
    );
    assert_eq!(reader.chunk_verification_stats().chunks_verified, 0);
}

#[test]
fn truncated_chunk_still_raises() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = vec![0x5au8; 4096];
    let (segment, spec) =
        DeepSeekV4FullStreamReader::write_isolated_content_addressed_chunk(tmp.path(), &payload)
            .expect("write isolated chunk");
    let chunk_path = tmp.path().join(&spec.relative);
    fs::write(&chunk_path, &payload[..1024]).expect("truncate temp-copy chunk");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "probe.weight".to_owned(),
        isolated_tensor("probe.weight", segment),
    );
    let bind_err =
        DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture(tmp.path(), tensors, [spec])
            .expect_err("bind must refuse a size-mismatched chunk");
    let message = format!("{bind_err}");
    assert!(
        message.contains("bytes"),
        "unexpected bind error: {message}"
    );
}

#[test]
fn truncated_after_bind_still_raises_on_read() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = vec![0x3cu8; 4096];
    let (reader, segment) = bind_one_tensor(tmp.path(), "probe.weight", &payload);
    fs::write(tmp.path().join(&segment.chunk_relpath), &payload[..512])
        .expect("truncate after bind");
    let err = reader
        .read_verified_full("probe.weight", payload.len())
        .expect_err("truncated chunk must hard-fail on read");
    let message = format!("{err}");
    assert!(
        message.contains("byte size") || message.contains("differs from sealed"),
        "unexpected error: {message}"
    );
}

#[test]
fn verified_once_second_read_does_not_rehash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
    let (reader, _) = bind_one_tensor(tmp.path(), "probe.weight", &payload);

    let first = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("first read");
    assert_eq!(first, payload);
    let after_first = reader.chunk_verification_stats();
    assert_eq!(after_first.hash_invocations, 1);
    assert_eq!(after_first.cache_hits, 0);
    assert_eq!(after_first.chunks_verified, 1);
    assert_eq!(after_first.bytes_hashed, payload.len() as u64);

    let second = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("second read");
    assert_eq!(second, payload);
    let after_second = reader.chunk_verification_stats();
    assert_eq!(
        after_second.hash_invocations, 1,
        "second read of the same chunk must not re-hash"
    );
    assert_eq!(after_second.cache_hits, 1);
    assert_eq!(after_second.chunks_verified, 1);
    assert_eq!(after_second.bytes_hashed, payload.len() as u64);

    let slice = reader
        .read_verified_range("probe.weight", 100..200, 100)
        .expect("range read of verified chunk");
    assert_eq!(slice, &payload[100..200]);
    let after_range = reader.chunk_verification_stats();
    assert_eq!(after_range.hash_invocations, 1);
    assert_eq!(after_range.cache_hits, 2);
}

#[test]
fn zero_copy_view_matches_copying_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload: Vec<u8> = (0u32..4096).flat_map(|n| n.to_le_bytes()).collect();
    let (reader, _) = bind_one_tensor(tmp.path(), "probe.weight", &payload);

    let copied = reader
        .read_verified_full("probe.weight", payload.len())
        .expect("copying path");
    let view = reader
        .read_verified_full_view("probe.weight", payload.len())
        .expect("zero-copy path");

    assert_eq!(copied.as_slice(), payload.as_slice());
    assert_eq!(view.as_bytes(), copied.as_slice());
    assert!(
        view.is_zero_copy(),
        "single-chunk full tensor must be an mmap view"
    );
    assert_eq!(view.len(), payload.len());

    let range_copied = reader
        .read_verified_range("probe.weight", 64..192, 128)
        .expect("copying range");
    let range_view = reader
        .read_verified_range_view("probe.weight", 64..192, 128)
        .expect("zero-copy range");
    assert_eq!(range_view.as_bytes(), range_copied.as_slice());
    assert_eq!(range_view.as_bytes(), &payload[64..192]);
    assert!(range_view.is_zero_copy());
}

#[test]
fn mutation_after_verify_is_not_silently_trusted_by_a_fresh_reader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let payload = b"fresh-reader must re-check a mutated temp copy".to_vec();
    let (reader, segment) = bind_one_tensor(tmp.path(), "probe.weight", &payload);
    reader
        .read_verified_full("probe.weight", payload.len())
        .expect("first reader verifies");

    let chunk_path = tmp.path().join(&segment.chunk_relpath);
    let mut mutated = payload.clone();
    mutated[0] ^= 0x01;
    fs::write(&chunk_path, &mutated).expect("mutate after first verify");

    let spec = hawking_core::gravity_deepseek_v4::DeepSeekV4ChunkSpec {
        relative: segment.chunk_relpath.clone(),
        sha256: segment.sha256.clone(),
        bytes: segment.bytes,
    };
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "probe.weight".to_owned(),
        isolated_tensor("probe.weight", segment),
    );
    let fresh =
        DeepSeekV4FullStreamReader::bind_isolated_integrity_fixture(tmp.path(), tensors, [spec])
            .expect("fresh reader binds the mutated copy");
    let err = fresh
        .read_verified_full("probe.weight", payload.len())
        .expect_err("fresh reader must re-hash and fail");
    assert!(format!("{err}").contains("sha256 differs from sealed segment digest"));
}
