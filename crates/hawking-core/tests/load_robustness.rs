#![cfg(target_os = "macos")]
use hawking_core::gguf::GgufFile;
use std::io::Write as _;
use std::path::PathBuf;
mod common;
use common::*;
#[test]
fn new_buffer_checked_errs_gracefully_on_oversize() {
    let ctx = ctx();
    let huge = 1usize << 48;
    let r = ctx.new_buffer_checked(huge);
    assert!(
        r.is_err(),
        "oversize new_buffer_checked should be Err, got Ok"
    );
    let ok = ctx
        .new_buffer_checked(4096)
        .expect("normal alloc should succeed");
    assert!(ok.length() >= 4096, "buffer length {} < 4096", ok.length());
    let okb = ctx
        .new_buffer_with_bytes_checked(&[0u8; 256])
        .expect("small bytes alloc should succeed");
    assert!(okb.length() >= 256);
}
fn write_tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "hawking_gguf_corrupt_{name}_{}.gguf",
        std::process::id()
    ));
    let mut f = std::fs::File::create(&p).expect("create tmp");
    f.write_all(bytes).expect("write tmp");
    p
}
const MAGIC: &[u8; 4] = b"GGUF";
#[test]
fn open_errs_on_empty_file() {
    let p = write_tmp("empty", &[]);
    let r = GgufFile::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(r.is_err(), "empty file should be Err");
}
#[test]
fn open_errs_on_bad_magic() {
    let mut bytes = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
    bytes.extend_from_slice(&[0u8; 28]); // pad so the mmap is comfortably sized
    let p = write_tmp("badmagic", &bytes);
    let r = GgufFile::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(r.is_err(), "bad magic should be Err");
}
#[test]
fn open_errs_on_unsupported_version() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&99u32.to_le_bytes()); // version 99 (not 2..=3)
    bytes.extend_from_slice(&[0u8; 24]);
    let p = write_tmp("badversion", &bytes);
    let r = GgufFile::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(r.is_err(), "unsupported version should be Err");
}
#[test]
fn open_errs_on_truncated_header() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
    bytes.extend_from_slice(&5u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&5u64.to_le_bytes()); // kv_count
    let p = write_tmp("truncated", &bytes);
    let r = GgufFile::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(r.is_err(), "truncated header should be Err");
}
#[test]
fn open_errs_on_truncated_real_gguf() {
    let model = PathBuf::from("../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if !model.exists() {
        eprintln!("skip open_errs_on_truncated_real_gguf: no model at {model:?}");
        return;
    }
    let full = std::fs::read(&model).expect("read model");
    let truncated = &full[..full.len().min(4096)];
    let p = write_tmp("realtrunc", truncated);
    let r = GgufFile::open(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        r.is_err(),
        "truncated real GGUF should be Err (tensor data past EOF)"
    );
}
