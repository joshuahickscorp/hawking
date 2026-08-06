#![cfg(target_os = "macos")]
use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm::{
    estimate_host_state_waits_per_token, estimate_resident_waits_per_token, GravityGlm,
};
use hawking_core::metal::MetalContext;
use std::path::PathBuf;
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm")
}
fn top1(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| {
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(j))
        })
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}
fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .expect("no NaN")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}
fn prompts(base: &[u32]) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    out.push(base.to_vec());
    if base.len() >= 2 {
        out.push(base[1..].to_vec());
        out.push(base[..base.len() - 1].to_vec());
        let mut rev = base.to_vec();
        rev.reverse();
        out.push(rev);
    }
    out.push(vec![0]);
    out.push(vec![1, 2, 3]);
    out.push(vec![7, 7, 7, 7]);
    out.push(vec![100, 200, 300, 400, 500]);
    out
}
#[test]
fn resident_matches_host_state_over_several_prompts() {
    let dir = fixtures_dir();
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            return;
        }
    };
    let host = GravityGlm::open(&dir.join("glm52-tiny-R0.gravity"), true).expect("host open");
    let host_gpu = GravityGlmGpu::open_dir_with_budget_resident(
        MetalContext::new().expect("second ctx"),
        &dir,
        true,
        256 * 1024 * 1024,
        false, // host-state GPU path (oracle for GPU weight layout)
    )
    .expect("host-state gpu open");
    let resident = GravityGlmGpu::open_dir_with_budget_resident(
        ctx,
        &dir,
        true,
        256 * 1024 * 1024,
        true, // resident path under test
    )
    .expect("resident open");
    assert!(resident.resident_state_enabled());
    assert!(!host_gpu.resident_state_enabled());
    let base: Vec<u32> = {
        #[derive(serde::Deserialize)]
        struct Ref {
            tokens: Vec<u32>,
        }
        let r: Ref =
            serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).expect("ref_glm"))
                .expect("parse");
        r.tokens
    };
    let mut any_waits = None;
    for (pi, prompt) in prompts(&base).into_iter().enumerate() {
        if prompt.is_empty() {
            continue;
        }
        if prompt.iter().any(|&t| t as usize >= host.arch.vocab_size) {
            continue;
        }
        let (cpu_logits, cpu_trace) = host.forward(&prompt).expect("cpu forward");
        let (host_gpu_logits, host_gpu_trace) =
            host_gpu.forward(&prompt).expect("host-state gpu forward");
        let (res_logits, res_trace, waits) = resident
            .forward_resident_counted(&prompt)
            .expect("resident forward");
        any_waits = Some(waits);
        let host_tok = top1(&host_gpu_logits);
        let res_tok = top1(&res_logits);
        assert_eq!(
            res_tok, host_tok,
            "prompt {pi} {prompt:?}: resident argmax {res_tok} != host-state gpu {host_tok}"
        );
        assert_eq!(
            top_k(&res_logits, 5),
            top_k(&host_gpu_logits, 5),
            "prompt {pi}: top-5 tokens diverge"
        );
        assert_eq!(
            res_trace.final_topk, host_gpu_trace.final_topk,
            "prompt {pi}: final DSA top-k"
        );
        assert_eq!(
            res_trace.expert_choices, host_gpu_trace.expert_choices,
            "prompt {pi}: expert choices"
        );
        assert_eq!(
            res_logits, host_gpu_logits,
            "prompt {pi}: logits must be bit-identical on the native-heavy fixture"
        );
        let _ = (cpu_logits, cpu_trace);
    }
    let waits = any_waits.expect("ran at least one prompt");
}
#[test]
fn resident_incremental_decode_matches_full_replay() {
    let dir = fixtures_dir();
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            return;
        }
    };
    let model =
        GravityGlmGpu::open_dir_with_budget_resident(ctx, &dir, true, 256 * 1024 * 1024, true)
            .expect("open");
    #[derive(serde::Deserialize)]
    struct Ref {
        tokens: Vec<u32>,
    }
    let reference: Ref =
        serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).unwrap()).unwrap();
    let tokens = &reference.tokens;
    assert!(tokens.len() >= 3);
    let (want, _) = model.forward(tokens).expect("full");
    let split = tokens.len() - 2;
    let (mut got, _) = model.forward(&tokens[..split]).expect("prefill");
    for (i, &t) in tokens[split..].iter().enumerate() {
        got = model.forward_at(&[t], split + i).expect("extend").0;
    }
    assert_eq!(
        got, want,
        "incremental resident decode must match full replay"
    );
}
#[test]
fn static_wait_estimates_are_exported_for_the_controller() {
    let dir = fixtures_dir();
    let host = GravityGlm::open(&dir.join("glm52-tiny-R0.gravity"), false).unwrap();
    let h = estimate_host_state_waits_per_token(&host.arch);
    let r = estimate_resident_waits_per_token(&host.arch);
    assert!(h > r);
    assert!(h >= 30 && h <= 80, "tiny host waits {h}");
}
