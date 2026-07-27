//! §4.4: local `.gravity` -> reviewed adapter registry -> hawking-core ->
//! a complete token -> streaming output, with no source weights consulted.
//!
//! The registry is the part under test. A runtime reachable only through its
//! own example binary is a benchmark, not a production path, and the campaign
//! forbids exactly that.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use hawking_core::model::load_engine;
use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};

const DEFAULT_ARTIFACT: &str =
    "Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.v2.gravity";

fn artifact_path() -> Option<PathBuf> {
    let p = match std::env::var_os("HAWKING_GRAVITY_LLAMA_ARTIFACT") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(std::env::var_os("HOME")?).join(DEFAULT_ARTIFACT),
    };
    p.is_file().then_some(p)
}

#[test]
fn registry_serves_a_gravity_artifact_end_to_end() {
    let Some(art) = artifact_path() else {
        eprintln!("skipping registry serve: no llama32-1b .gravity artifact on disk");
        return;
    };

    // Dispatch by container magic, not by file extension or a caller hint.
    let mut engine = load_engine(&art, EngineConfig::default()).expect("registry load");
    assert_eq!(engine.model_arch(), "llama");
    assert!(!engine.model_id().is_empty());

    let mut streamed = String::new();
    let mut token_events = 0usize;
    let mut done_events = 0usize;
    let stats = engine
        .generate(
            GenerateRequest {
                prompt: "The capital of France is".to_string(),
                max_new_tokens: 8,
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &mut |ev| match ev {
                StreamEvent::Token { text, .. } => {
                    streamed.push_str(&text);
                    token_events += 1;
                }
                StreamEvent::Done { .. } => done_events += 1,
            },
        )
        .expect("generate");

    assert_eq!(done_events, 1, "exactly one Done event");
    assert_eq!(
        token_events, stats.completion_tokens,
        "every counted token was streamed"
    );
    assert!(stats.prompt_tokens > 0, "prompt was tokenized");
    assert!(
        !streamed.is_empty(),
        "streamed text is empty; the engine produced ids but no text"
    );
    // The artifact at this rate is degenerate, which is a sealed finding and
    // not this test's business. What is this test's business is that real text
    // came out of the registry path at all.
    eprintln!(
        "registry serve: {} prompt + {} completion tokens, prefill {:.0} ms, decode {:.0} ms, \
         text {streamed:?}",
        stats.prompt_tokens, stats.completion_tokens, stats.prefill_ms, stats.decode_ms
    );
}

const DEFAULT_GLM_SHARD0: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/General-R0/model-00001-of-00282.gravity";

fn glm_artifact_path() -> Option<PathBuf> {
    let p = match std::env::var_os("HAWKING_GRAVITY_GLM_ARTIFACT") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(std::env::var_os("HOME")?).join(DEFAULT_GLM_SHARD0),
    };
    p.is_file().then_some(p)
}

/// The same registry path GLM's own example binary exercises, but through
/// `load_engine` -- a multi-shard architecture reachable only from an
/// example is still not a production path. `weights` names one shard file;
/// the engine must find the other 281 in its parent directory on its own.
///
/// Bounded to 2 completion tokens: each one is real seconds at this artifact
/// size (per-tensor Metal upload on first touch), and this test's business
/// is that the registry reaches the multi-shard GPU path end to end, not a
/// throughput number -- `gravity_glm_tps` owns that separately.
#[test]
fn registry_serves_a_multi_shard_glm_artifact_end_to_end() {
    let Some(art) = glm_artifact_path() else {
        eprintln!("skipping GLM registry serve: no GLM-5.2 General-R0 artifact on disk");
        return;
    };

    let mut engine = load_engine(&art, EngineConfig::default()).expect("registry load");
    assert_eq!(engine.model_arch(), "glm_moe_dsa");
    assert!(!engine.model_id().is_empty());

    let mut streamed = String::new();
    let mut token_events = 0usize;
    let mut done_events = 0usize;
    let stats = engine
        .generate(
            GenerateRequest {
                prompt: "The capital of France is".to_string(),
                max_new_tokens: 2,
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &mut |ev| match ev {
                StreamEvent::Token { text, .. } => {
                    streamed.push_str(&text);
                    token_events += 1;
                }
                StreamEvent::Done { .. } => done_events += 1,
            },
        )
        .expect("generate");

    assert_eq!(done_events, 1, "exactly one Done event");
    assert_eq!(
        token_events, stats.completion_tokens,
        "every counted token was streamed"
    );
    assert!(stats.prompt_tokens > 0, "prompt was tokenized");
    // R0 fails the screening gate (sealed, M03) -- degenerate output here is
    // the known finding, not a regression. This test's business is that the
    // registry produced it through the real multi-shard path, at all.
    eprintln!(
        "GLM registry serve: {} prompt + {} completion tokens, prefill {:.0} ms, decode {:.0} ms, \
         text {streamed:?}",
        stats.prompt_tokens, stats.completion_tokens, stats.prefill_ms, stats.decode_ms
    );
}

/// A GGUF must not be mistaken for a `.gravity`, and vice versa. The check
/// reads the container magic, so a renamed file still routes correctly.
#[test]
fn gravity_detection_reads_magic_not_the_extension() {
    use hawking_core::model::gravity_engine::GravityEngine;

    let Some(art) = artifact_path() else {
        eprintln!("skipping magic detection: no artifact on disk");
        return;
    };
    assert!(GravityEngine::is_gravity(&art));

    let tmp = std::env::temp_dir().join("not-a-gravity.gravity");
    std::fs::write(&tmp, b"GGUF\0\0\0\0some other container").expect("write decoy");
    assert!(
        !GravityEngine::is_gravity(&tmp),
        "a .gravity extension over non-gravity bytes was accepted"
    );
    let _ = std::fs::remove_file(&tmp);
}
