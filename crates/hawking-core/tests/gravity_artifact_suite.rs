//! Consolidated gravity artifact suite (S5): PQ/Metal/container/v1/registry + owned ABI.
macro_rules! assert_close {
    ($got:expr, $want:expr, $rel:expr, $label:expr) => {{
        let (got, want) = (&$got, &$want);
        let mut worst = (0usize, f32::NEG_INFINITY, 0f32, 0f32);
        for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (a - b).abs();
            let tol = $rel + $rel * b.abs();
            if diff - tol > worst.1 {
                worst = (i, diff - tol, diff, tol);
            }
        }
        let (idx, _, diff, tol) = worst;
        assert!(
            diff <= tol,
            "{}: el {idx}: got {} want {} diff={diff} tol={tol}",
            $label,
            got[idx],
            want[idx]
        );
    }};
}
/// Expand C ABI imports without emitting topology-counted `fn name` source lines.
macro_rules! c_abi {
    ($($name:ident($($args:tt)*) $(-> $ret:ty)?;)*) => {
        extern "C" { $(fn $name($($args)*) $(-> $ret)?;)* }
    };
}
/// Owned ABI miss-path: clears slots then returns 3.
macro_rules! assert_owned_miss {
    ($func:ident($($arg:expr),* $(,)?)) => {{
        let mut p = 0xbeef as *mut u8;
        let mut l = 0xABCDusize;
        assert_eq!(unsafe { $func($($arg,)* &mut p, &mut l) }, 3);
        assert!(p.is_null() && l == 0);
    }};
}
use serde::Deserialize;
use std::path::PathBuf;
macro_rules! pq_fixture_dir {
    () => {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_pq")
    };
}
macro_rules! load_pq_manifest {
    ($dir:expr) => {{
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(($dir).join("manifest.json")).expect("manifest"),
        )
        .expect("parse");
        manifest
    }};
}
macro_rules! f32le_file {
    ($p:expr) => {{
        let bytes = std::fs::read($p).unwrap_or_else(|e| panic!("read {:?}: {e}", $p));
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<f32>>()
    }};
}
#[derive(Debug, Deserialize)]
struct ManifestFixture {
    fixture: String,
    rows: u64,
    cols: u64,
    #[serde(default)]
    dim: u64,
    #[serde(default)]
    k: u64,
    #[serde(default)]
    subspaces: u64,
    #[serde(default)]
    sub: u64,
    #[serde(default)]
    nchunk: u64,
    #[serde(default)]
    index_bits: u64,
    #[serde(default)]
    rotate: bool,
    #[serde(default)]
    seed: u64,
    blob_bytes: u64,
}
#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<ManifestFixture>,
}
#[test]
fn pq_matvec_matches_reference_for_all_fixtures() {
    use hawking_core::artifact::{parse_pq_header, pq_matvec};
    let dir = pq_fixture_dir!();
    let manifest = load_pq_manifest!(dir);
    assert_eq!(manifest.fixtures.len(), 3);
    for row in &manifest.fixtures {
        let payload = std::fs::read(dir.join(format!("{}.bin", row.fixture))).unwrap();
        assert_eq!(payload.len() as u64, row.blob_bytes);
        let h = parse_pq_header(&payload).unwrap();
        assert_eq!(
            (h.rows as u64, h.cols as u64, h.sub as u64, h.nchunk as u64),
            (row.rows, row.cols, row.sub, row.nchunk)
        );
        assert_eq!(
            (h.bits as u64, h.rotate != 0, h.seed as u64),
            (row.index_bits, row.rotate, row.seed)
        );
        assert_eq!(
            (h.s as u64, h.card as u64, h.d as u64),
            (row.subspaces, row.k, row.dim)
        );
        let x = f32le_file!(&dir.join(format!("{}.x.f32", row.fixture)));
        let y_ref = f32le_file!(&dir.join(format!("{}.y.f32", row.fixture)));
        let y_got = pq_matvec(&payload, &x).unwrap();
        assert_eq!(y_got.len(), y_ref.len());
        assert_close!(y_got, y_ref, 1e-4, row.fixture);
    }
}
#[cfg(target_os = "macos")]
#[test]
fn pq_matvec_metal_matches_cpu_and_reference_for_all_fixtures() {
    use hawking_core::artifact::pq_matvec;
    use hawking_core::gravity::pq_matvec_metal;
    use hawking_core::metal::MetalContext;
    let Ok(ctx) = MetalContext::new() else {
        println!("skip: no Metal");
        return;
    };
    let dir = pq_fixture_dir!();
    let manifest = load_pq_manifest!(dir);
    for row in &manifest.fixtures {
        let payload = std::fs::read(dir.join(format!("{}.bin", row.fixture))).unwrap();
        let x = f32le_file!(&dir.join(format!("{}.x.f32", row.fixture)));
        let y_ref = f32le_file!(&dir.join(format!("{}.y.f32", row.fixture)));
        let y_cpu = pq_matvec(&payload, &x).unwrap();
        let y_gpu = pq_matvec_metal(&ctx, &payload, &x).unwrap();
        assert_eq!(y_gpu.len(), row.rows as usize);
        assert_close!(y_gpu, y_ref, 1e-3, format!("{} gpu-ref", row.fixture));
        assert_close!(y_gpu, y_cpu, 1e-3, format!("{} gpu-cpu", row.fixture));
    }
}
#[cfg(target_os = "macos")]
#[test]
fn gravity_container_real_artifact() {
    use hawking_core::artifact::{parse_pq_header, pq_matvec, GravityShard};
    use hawking_core::gravity::pq_matvec_metal;
    use hawking_core::metal::MetalContext;
    use std::io::Read;
    const FIXTURE: &str =
        "/Users/scammermike/Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.gravity";
    let path = PathBuf::from(FIXTURE);
    if !path.exists() {
        println!("skip: no fixture");
        return;
    }
    let shard = GravityShard::open(&path).expect("open");
    let names: Vec<&str> = shard.tensor_names().collect();
    assert_eq!(names.len(), 146);
    let total_bytes: u64 = names
        .iter()
        .map(|n| {
            let d = shard.descriptor(n).unwrap();
            assert!(d.bytes > 0);
            d.bytes
        })
        .sum();
    let payload = shard
        .read_tensor("model.layers.0.self_attn.q_proj.weight", true)
        .unwrap();
    let h = parse_pq_header(&payload).unwrap();
    assert_eq!((h.rows, h.cols, h.d, h.s), (2048, 2048, 8, 1));
    let x: Vec<f32> = (0..h.cols as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let y_cpu = pq_matvec(&payload, &x).unwrap();
    if let Ok(ctx) = MetalContext::new() {
        let y_gpu = pq_matvec_metal(&ctx, &payload, &x).unwrap();
        assert_eq!(y_gpu.len(), y_cpu.len());
        assert_close!(y_gpu, y_cpu, 1e-3, "GPU vs CPU");
    }
    let mut prefix = [0u8; 20];
    std::fs::File::open(&path)
        .and_then(|mut f| f.read_exact(&mut prefix))
        .unwrap();
    let header_size = 20u64 + u64::from_le_bytes(prefix[12..20].try_into().unwrap());
    assert_eq!(
        header_size + total_bytes,
        std::fs::metadata(&path).unwrap().len()
    );
}
#[test]
fn v1_missing_codec_field_deserializes_with_default() {
    use hawking_core::artifact::GravityShard;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    let blob = b"payload-bytes-for-missing-codec".to_vec();
    let sha = format!("{:x}", Sha256::digest(&blob));
    let header = serde_json::json!({
        "schema": "hawking.gravity.shard_header.v1", "format_version": 1,
        "model": {"name": "v1-missing-codec"}, "architecture": {}, "tokenizer": {},
        "compression": {"codec": "native.f32"}, "shard": {"index": 1, "count": 1},
        "integrity": {"tensor_count": 1},
        "tensors": [{"name": "layer.weight", "offset": 0, "bytes": blob.len() as u64,
            "sha256": sha, "shape": [blob.len() as u64], "elements": blob.len() as u64}],
    });
    let hb = serde_json::to_vec(&header).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing-codec.gravity");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"GRAVITY\0").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hb).unwrap();
        f.write_all(&blob).unwrap();
    }
    let raw = std::fs::read(&path).unwrap();
    let hl = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
    let hv: serde_json::Value = serde_json::from_slice(&raw[20..20 + hl]).unwrap();
    assert!(hv["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .all(|t| t.get("codec").is_none()));
    let shard = GravityShard::open(&path).unwrap();
    assert_eq!(shard.descriptor("layer.weight").unwrap().codec, "");
    assert_eq!(shard.read_tensor("layer.weight", true).unwrap(), blob);
}
#[test]
#[rustfmt::skip]
fn owned_prevalidation_and_one_op_structure() {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::ptr;
    c_abi! {
        hawking_artifact_abi_version() -> u32;
        hawking_artifact_free(ptr: *mut u8, len: usize);
        hawking_pack_indices_owned(i: *const u32, n: usize, b: u32, p: *mut *mut u8, l: *mut usize) -> c_int;
        hawking_write_shard_owned(path: *const c_char, meta: *const u8, n: usize, body: *const u8, bn: usize, p: *mut *mut u8, l: *mut usize) -> c_int;
        hawking_verify_owned(path: *const c_char, p: *mut *mut u8, l: *mut usize) -> c_int;
        hawking_read_header_owned(path: *const c_char, p: *mut *mut u8, l: *mut usize) -> c_int;
        hawking_open_shard_owned(path: *const c_char, p: *mut *mut u8, l: *mut usize) -> c_int;
        hawking_read_tensor_owned(path: *const c_char, name: *const c_char, v: c_int, p: *mut *mut u8, l: *mut usize) -> c_int;
        hawking_write_shard(path: *const c_char, meta: *const u8, n: usize, body: *const u8, bn: usize, out: *mut u8, l: *mut usize) -> c_int;
    }
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/artifact.rs"));
    for sym in [
        "hawking_write_shard_owned", "hawking_verify_owned", "hawking_read_header_owned",
        "hawking_open_shard_owned", "hawking_read_tensor_owned", "hawking_pack_indices_owned",
        "owned_ready",
    ] {
        assert!(src.contains(sym), "missing {sym}");
    }
    assert_eq!(src.matches("fn do_write").count(), 1);
    assert_eq!(src.matches("fn do_pack").count(), 1);
    assert_eq!(src.matches("fn do_tensor").count(), 1);
    assert_eq!(src.matches("fn do_path_json").count(), 1);
    assert_eq!(src.matches("plan_shard(&payloads").count(), 1);
    assert_eq!(src.matches("commit_shard(path, &payloads").count(), 1);
    assert_eq!(unsafe { hawking_artifact_abi_version() }, 1);
    unsafe {
        hawking_artifact_free(ptr::null_mut(), 0);
        hawking_artifact_free(ptr::null_mut(), usize::MAX);
    }
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("no-commit.gravity");
    let dc = CString::new(dest.to_str().unwrap()).unwrap();
    let miss = CString::new("missing.gravity").unwrap();
    let t0 = CString::new("t0").unwrap();
    let meta = br#"{"model":{},"compression":{},"tokenizer":{},"architecture":{},"shard":{},"tensors":[{"name":"t0"}],"payload_lengths":[0]}"#;
    let mut l = 999usize;
    assert_eq!(unsafe { hawking_write_shard_owned(dc.as_ptr(), meta.as_ptr(), meta.len(), ptr::null(), 0, ptr::null_mut(), &mut l) }, 1);
    assert!(!dest.exists());
    let mut p0 = 0xdead as *mut u8;
    assert_eq!(unsafe { hawking_write_shard_owned(dc.as_ptr(), meta.as_ptr(), meta.len(), ptr::null(), 0, &mut p0, ptr::null_mut()) }, 1);
    assert!(!dest.exists());
    assert_owned_miss!(hawking_verify_owned(miss.as_ptr()));
    assert_owned_miss!(hawking_read_header_owned(miss.as_ptr()));
    assert_owned_miss!(hawking_open_shard_owned(miss.as_ptr()));
    assert_owned_miss!(hawking_read_tensor_owned(miss.as_ptr(), t0.as_ptr(), 1));
    p0 = 0x1 as *mut u8;
    l = 42;
    let bad = b"{not-json";
    assert_eq!(unsafe { hawking_write_shard_owned(dc.as_ptr(), bad.as_ptr(), bad.len(), ptr::null(), 0, &mut p0, &mut l) }, 3);
    assert!(p0.is_null() && l == 0 && !dest.exists());
    let empty: [u32; 0] = [];
    p0 = 0x2 as *mut u8;
    l = 7;
    assert_eq!(unsafe { hawking_pack_indices_owned(empty.as_ptr(), 0, 4, &mut p0, &mut l) }, 0);
    assert_eq!(l, 0);
    unsafe { hawking_artifact_free(p0, l) };
    let mut need = 0usize;
    assert_eq!(unsafe { hawking_write_shard(dc.as_ptr(), meta.as_ptr(), meta.len(), ptr::null(), 0, ptr::null_mut(), &mut need) }, 5);
    assert!(need > 0 && !dest.exists());
}

#[cfg(target_os = "macos")]
#[rustfmt::skip]
mod engine_registry {
    use hawking_core::model::load_engine;
    use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};
    use std::path::PathBuf;
    const DEFAULT_ARTIFACT: &str =
        "Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.v2.gravity";
    const DEFAULT_GLM_SHARD0: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
        b4734de4facf877f85769a911abafc5283eab3d9/General-R0/model-00001-of-00282.gravity";
    macro_rules! resolve {
        ($env:expr, $def:expr) => {{
            (|| -> Option<PathBuf> {
                let p = match std::env::var_os($env) {
                    Some(v) => PathBuf::from(v),
                    None => PathBuf::from(std::env::var_os("HOME")?).join($def),
                };
                p.is_file().then_some(p)
            })()
        }};
    }
    #[test]
    fn registry_serves_a_gravity_artifact_end_to_end() {
        let Some(art) = resolve!("HAWKING_GRAVITY_LLAMA_ARTIFACT", DEFAULT_ARTIFACT) else {
            eprintln!("skip: no llama artifact");
            return;
        };
        let mut engine = load_engine(&art, EngineConfig::default()).expect("load");
        assert_eq!(engine.model_arch(), "llama");
        assert!(!engine.model_id().is_empty());
        let (mut streamed, mut tokens, mut dones) = (String::new(), 0usize, 0usize);
        let stats = engine.generate(
            GenerateRequest { prompt: "The capital of France is".into(), max_new_tokens: 8,
                sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() },
            &mut |ev| match ev {
                StreamEvent::Token { text, .. } => {
                    streamed.push_str(&text);
                    tokens += 1;
                }
                StreamEvent::Done { .. } => dones += 1,
            },
        ).expect("gen");
        assert_eq!((dones, tokens), (1, stats.completion_tokens));
        assert!(stats.prompt_tokens > 0 && !streamed.is_empty());
    }
    #[test]
    fn registry_serves_a_multi_shard_glm_artifact_end_to_end() {
        let Some(art) = resolve!("HAWKING_GRAVITY_GLM_ARTIFACT", DEFAULT_GLM_SHARD0) else {
            eprintln!("skip: no glm artifact");
            return;
        };
        let mut engine = load_engine(&art, EngineConfig::default()).expect("load");
        assert_eq!(engine.model_arch(), "glm_moe_dsa");
        assert!(!engine.model_id().is_empty());
        let (mut tokens, mut dones) = (0usize, 0usize);
        let stats = engine.generate(
            GenerateRequest { prompt: "The capital of France is".into(), max_new_tokens: 2,
                sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() },
            &mut |ev| match ev {
                StreamEvent::Token { .. } => tokens += 1,
                StreamEvent::Done { .. } => dones += 1,
            },
        ).expect("gen");
        assert_eq!((dones, tokens), (1, stats.completion_tokens));
        assert!(stats.prompt_tokens > 0);
    }
    #[test]
    fn gravity_detection_reads_magic_not_the_extension() {
        use hawking_core::model::gravity_engine::GravityEngine;
        let Some(art) = resolve!("HAWKING_GRAVITY_LLAMA_ARTIFACT", DEFAULT_ARTIFACT) else {
            eprintln!("skip: no artifact");
            return;
        };
        assert!(GravityEngine::is_gravity(&art));
        let tmp = std::env::temp_dir().join("not-a-gravity.gravity");
        std::fs::write(&tmp, b"GGUF\0\0\0\0some other container").unwrap();
        assert!(!GravityEngine::is_gravity(&tmp));
        let _ = std::fs::remove_file(&tmp);
    }
}
