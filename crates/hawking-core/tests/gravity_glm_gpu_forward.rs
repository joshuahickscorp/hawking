#![cfg(target_os = "macos")]
use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use std::path::PathBuf;
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm")
}
#[derive(serde::Deserialize)]
struct Reference {
    tokens: Vec<u32>,
    argmax: u32,
    top5: Vec<u32>,
}
fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .expect("no NaN in logits")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}
#[test]
fn gravity_glm_gpu_forward_matches_frozen_oracle() {
    let dir = fixtures_dir();
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).expect("read ref_glm"))
            .expect("parse ref_glm");
    let want: Vec<f32> = std::fs::read(dir.join("ref_logits.f32"))
        .expect("read ref logits")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let model = GravityGlmGpu::open_dir(&dir, true).expect("open GLM artifact on device");
    assert_eq!(
        model.arch.vocab_size,
        want.len(),
        "vocab vs reference logits"
    );
    let (got, _trace) = model.forward(&reference.tokens).expect("gpu forward");
    assert_eq!(got.len(), want.len(), "logit count");
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-2 + 1e-3 * b.abs();
        assert!(
            (a - b).abs() <= tol,
            "logit {i}: got {a}, want {b}, diff {} > tol {tol}",
            (a - b).abs()
        );
    }
    let got_top5 = top_k(&got, 5);
    assert_eq!(got_top5[0], reference.argmax, "argmax");
    assert_eq!(got_top5, reference.top5, "top-5");
}
#[test]
fn gravity_glm_gpu_incremental_decode_matches_full_replay() {
    let dir = fixtures_dir();
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).expect("read ref_glm"))
            .expect("parse ref_glm");
    let model = GravityGlmGpu::open_dir(&dir, true).expect("open GLM artifact on device");
    let tokens = &reference.tokens;
    assert!(
        tokens.len() >= 3,
        "fixture needs at least 3 tokens to split"
    );
    let (want, _) = model.forward(tokens).expect("full replay");
    let split = tokens.len() - 2;
    let (mut got, _) = model.forward(&tokens[..split]).expect("prefill");
    for (i, &t) in tokens[split..].iter().enumerate() {
        got = model.forward_at(&[t], split + i).expect("extend").0;
    }
    assert_eq!(got.len(), want.len());
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert_eq!(
        got, want,
        "incremental decode must exactly match full replay"
    );
}
#[test]
fn gravity_glm_gpu_forward_resets_between_requests() {
    let dir = fixtures_dir();
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).expect("read ref_glm"))
            .expect("parse ref_glm");
    let model = GravityGlmGpu::open_dir(&dir, true).expect("open GLM artifact on device");
    let (first, _) = model.forward(&reference.tokens).expect("first request");
    let (second, _) = model
        .forward(&reference.tokens)
        .expect("second request on the same resident model");
    assert_eq!(
        first, second,
        "two identical requests on one resident model must produce identical logits"
    );
}
