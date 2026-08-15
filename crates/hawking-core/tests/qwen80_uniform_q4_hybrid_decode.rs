//! Required velocity-track tests for the Qwen80 uniform-Q4 hybrid decode loop.
//!
//! 1. Multi-token state advance fails if state is reset between tokens.
//! 2. Same prompt + greedy is deterministic.
//! 3. Artifact binding: catalog count is 74,391 when the sealed body is
//!    present; a missing or short tensor raises rather than zero-filling.

use hawking_core::model::qwen80_device_expert_table::QWEN80_EXPERT_TABLE_KERNELS;
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_tokenizer, discover_qwen80_uniform_q4_root, generate_greedy,
    load_qwen80_tokenizer, qwen80_fixture_advance_hybrid_state, qwen80_fixture_greedy_token,
    render_qwen80_source_user_chat, Qwen80HybridDecodeState, Qwen80UniformQ4HybridDecodeSession,
    Qwen80UniformQ4StreamingCatalog, QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
};
use hawking_core::model::qwen_complete_binary::{
    pack_uniform_q4_group64, QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT, QWEN80_UNIFORM_Q4_SCHEMA,
    QWEN80_UNIFORM_Q4_TENSOR_EXT,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use tempfile::TempDir;

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn multi_token_state_advance_fails_if_reset_between_tokens() {
    let mut sequential = Qwen80HybridDecodeState::new(16).unwrap();
    let mut seq = Vec::new();
    for _ in 0..4 {
        seq.push(qwen80_fixture_advance_hybrid_state(&mut sequential, 21, false).unwrap());
    }
    assert_ne!(seq[0], seq[1]);
    assert_ne!(seq[1], seq[2]);

    let mut reset = Qwen80HybridDecodeState::new(16).unwrap();
    let mut rst = Vec::new();
    for _ in 0..4 {
        rst.push(qwen80_fixture_advance_hybrid_state(&mut reset, 21, true).unwrap());
    }
    assert_eq!(
        rst[0], rst[1],
        "reset-each-token must repeat per-token state"
    );
    assert_ne!(
        seq, rst,
        "decoding N tokens with a state reset between tokens must fail this test"
    );

    let mut mixed = Qwen80HybridDecodeState::new(16).unwrap();
    let t0 = qwen80_fixture_advance_hybrid_state(&mut mixed, 4, false).unwrap();
    let t1 = qwen80_fixture_advance_hybrid_state(&mut mixed, 5, false).unwrap();
    let mut same = Qwen80HybridDecodeState::new(16).unwrap();
    let s0 = qwen80_fixture_advance_hybrid_state(&mut same, 4, false).unwrap();
    let s1 = qwen80_fixture_advance_hybrid_state(&mut same, 4, false).unwrap();
    assert_eq!(t0, s0);
    assert_ne!(t1, s1);
}

#[test]
fn greedy_same_prompt_is_deterministic() {
    let mut a = Qwen80HybridDecodeState::new(16).unwrap();
    let mut b = Qwen80HybridDecodeState::new(16).unwrap();
    let mut ids_a = Vec::new();
    let mut ids_b = Vec::new();
    let mut token = 8u32;
    for _ in 0..5 {
        qwen80_fixture_advance_hybrid_state(&mut a, token, false).unwrap();
        let next = qwen80_fixture_greedy_token(&a, token);
        ids_a.push(next);
        token = next;
    }
    token = 8;
    for _ in 0..5 {
        qwen80_fixture_advance_hybrid_state(&mut b, token, false).unwrap();
        let next = qwen80_fixture_greedy_token(&b, token);
        ids_b.push(next);
        token = next;
    }
    assert_eq!(ids_a, ids_b);
    assert_eq!(a.fingerprint_sha256(), b.fingerprint_sha256());
}

#[test]
fn artifact_binding_74391_and_missing_or_short_raises() {
    if let Some(root) = discover_qwen80_uniform_q4_root() {
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        assert_eq!(
            catalog.tensor_count(),
            QWEN80_UNIFORM_Q4_EXPECTED_TENSOR_COUNT
        );
        assert!(
            (catalog.complete_physical_bpw - QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW).abs() < 1e-6
        );
        let missing = catalog.read_payload("this.tensor.does.not.exist");
        assert!(missing.is_err(), "missing tensor must raise, not zero-fill");
    }

    let temp = TempDir::new().unwrap();
    let root = temp.path().join("q4");
    let tensors = root.join("tensors");
    fs::create_dir_all(&tensors).unwrap();
    let name = "model.norm.weight";
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.02).collect();
    let (payload, _) = pack_uniform_q4_group64(&values, &[64]).unwrap();
    let hashed = format!(
        "{}.{QWEN80_UNIFORM_Q4_TENSOR_EXT}",
        sha256_hex(name.as_bytes())
    );
    let path = tensors.join(hashed);
    fs::write(&path, &payload).unwrap();
    let manifest = json!({
        "schema": QWEN80_UNIFORM_Q4_SCHEMA,
        "seal_sha256": "bb".repeat(32),
        "complete_physical_bpw_ledger": {
            "complete_physical_bpw": QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW,
            "tensor_payload_bytes": payload.len()
        },
        "tensors": [{
            "tensor_name": name,
            "shape": [64],
            "elements": 64,
            "artifact_path": path,
            "artifact_bytes": payload.len(),
            "artifact_sha256": sha256_hex(&payload),
        }]
    });
    let manifest_path =
        root.join("QWEN80_UNIFORM_Q4_GROUP64_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json");
    let mut file = fs::File::create(&manifest_path).unwrap();
    file.write_all(serde_json::to_vec(&manifest).unwrap().as_slice())
        .unwrap();
    let catalog = Qwen80UniformQ4StreamingCatalog::open_manifest(&manifest_path).unwrap();
    assert_eq!(catalog.tensor_count(), 1);
    assert!(catalog.read_payload("absent.weight").is_err());
    fs::write(&path, &payload[..16]).unwrap();
    let short = catalog.read_payload(name).unwrap_err();
    let message = format!("{short}");
    assert!(
        message.contains("short") || message.contains("truncated"),
        "short tensor must raise, got {message}"
    );
}

#[test]
fn greedy_baseline_prompt_yields_hello_how_tokens() {
    let Some(root) = discover_qwen80_uniform_q4_root() else {
        return;
    };
    let Some(tokenizer_path) = discover_qwen80_tokenizer() else {
        return;
    };
    let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
    let tokenizer = load_qwen80_tokenizer(&tokenizer_path).unwrap();
    let mut session = Qwen80UniformQ4HybridDecodeSession::new(catalog, 64).unwrap();
    let prompt = render_qwen80_source_user_chat("Hi");
    let result = generate_greedy(&mut session, &tokenizer, &prompt, 3).unwrap();
    assert_eq!(result.generated_token_ids, vec![9707, 0, 2585]);
}

#[test]
fn qwen80_expert_table_kernels_are_named() {
    assert_eq!(QWEN80_EXPERT_TABLE_KERNELS.len(), 5);
    for kernel in QWEN80_EXPERT_TABLE_KERNELS {
        assert!(
            kernel.starts_with("qwen80_expert_table_"),
            "{kernel} is not a Q80 expert-table kernel"
        );
    }
}
