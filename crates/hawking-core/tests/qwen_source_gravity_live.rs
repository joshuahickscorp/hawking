//! Opt-in full Qwen source-preserving `.gravity` execution gate.
//!
//! This test is intentionally inert without both immutable input paths.  When
//! enabled it proves only source/artifact/tokenizer binding plus a complete
//! CPU direct-grammar forward.  It is not a Metal, latency, TG, or capability
//! promotion.

use hawking_core::gguf::GgufFile;
use hawking_core::gravity::GravityShard;
use hawking_core::gravity_llama::GravityLlama;
use hawking_core::numeric_parity::{format_score_line, score_against_f64, score_pair, Bounds};
use hawking_core::tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn required_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| path.is_file())
}

fn sha256_file(path: &PathBuf) -> String {
    let mut file = File::open(path).expect("open immutable source");
    let mut hasher = Sha256::new();
    let mut bytes = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut bytes).expect("read immutable source");
        if read == 0 {
            break;
        }
        hasher.update(&bytes[..read]);
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn source_preserving_qwen_gravity_executes_complete_cpu_forward() {
    let Some(source) = required_path("HAWKING_QWEN_GRAVITY_SOURCE") else {
        eprintln!("skipping Qwen Gravity live gate: HAWKING_QWEN_GRAVITY_SOURCE is absent");
        return;
    };
    let Some(artifact) = required_path("HAWKING_QWEN_GRAVITY_ARTIFACT") else {
        eprintln!("skipping Qwen Gravity live gate: HAWKING_QWEN_GRAVITY_ARTIFACT is absent");
        return;
    };
    let Some(tokenizer_path) = required_path("HAWKING_QWEN_GRAVITY_TOKENIZER") else {
        eprintln!("skipping Qwen Gravity live gate: HAWKING_QWEN_GRAVITY_TOKENIZER is absent");
        return;
    };

    let shard = GravityShard::open(&artifact).expect("open sealed Qwen Gravity artifact");
    let expected_source_sha = shard
        .extra
        .get("model")
        .and_then(|model| model.get("source_gguf_sha256"))
        .and_then(serde_json::Value::as_str)
        .expect("artifact source hash binding");
    assert_eq!(
        sha256_file(&source),
        expected_source_sha,
        "source body changed"
    );
    let expected_tokenizer_sha = shard
        .extra
        .get("tokenizer")
        .and_then(|tokenizer| tokenizer.get("sha256"))
        .and_then(serde_json::Value::as_str)
        .expect("artifact tokenizer hash binding");
    assert_eq!(
        sha256_file(&tokenizer_path),
        expected_tokenizer_sha,
        "tokenizer changed"
    );
    drop(shard);

    let prompt = "2 + 2 =";
    let gguf = GgufFile::open(&source).expect("open Qwen GGUF tokenizer authority");
    let source_tokens = Tokenizer::from_gguf(&gguf)
        .expect("build GGUF tokenizer")
        .encode(prompt, true)
        .expect("encode prompt with GGUF tokenizer");
    let sidecar_tokens = Tokenizer::from_file(&tokenizer_path)
        .expect("build hash-bound tokenizer sidecar")
        .encode(prompt, true)
        .expect("encode prompt with sidecar tokenizer");
    assert_eq!(source_tokens, sidecar_tokens, "tokenizer ids differ");
    assert!(!source_tokens.is_empty());

    let started = Instant::now();
    let model = GravityLlama::open(&artifact, true).expect("lazy-open executable Qwen artifact");
    let logits = model
        .forward(&source_tokens)
        .expect("complete Qwen CPU direct-grammar forward");
    let elapsed = started.elapsed();
    assert_eq!(logits.len(), model.arch.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    let greedy = logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("nonempty vocabulary")
        .0;
    eprintln!(
        "QWEN_SOURCE_GRAVITY_CPU_COMPLETE prompt_tokens={} vocab={} greedy_token={} elapsed_ms={} metal_dispatches=0 cpu_reference_fallback=NOT_APPLICABLE promotion=FORBIDDEN",
        source_tokens.len(),
        logits.len(),
        greedy,
        elapsed.as_millis(),
    );

    // The independent full f64 transformer is opt-in: it is reference-only
    // and intentionally much slower than decode.  It reads the same packed
    // source bytes directly, without lifting the CPU f32 logits.  A passing
    // CPU score is V2.1 evidence, never a TG or capability promotion.
    let f64_authority = if std::env::var_os("HAWKING_QWEN_GRAVITY_V21").is_some() {
        let reference = model
            .forward_f64_authority(&source_tokens)
            .expect("independent Qwen f64 complete forward");
        let score = score_against_f64(
            &logits,
            &reference,
            &Bounds::full_forward_logits(),
            "qwen_source_gravity_cpu_f32",
        );
        assert!(
            score.pass,
            "Qwen CPU f32 failed V2.1 against packed-byte f64 authority: {} failures={:?}",
            format_score_line(&score),
            score.failures
        );
        eprintln!(
            "QWEN_SOURCE_GRAVITY_CPU_V21 prompt_tokens={} {} promotion=FORBIDDEN",
            source_tokens.len(),
            format_score_line(&score),
        );
        Some(reference)
    } else {
        None
    };

    // The full GPU comparison is deliberately opt-in because it reads the
    // complete vocabulary logits back for a diagnostic, not a serving path.
    // CPU f32 is not an FP64 V2.1 authority, so this asserts only exact
    // same-artifact discrete agreement and reports continuous drift.
    #[cfg(target_os = "macos")]
    if std::env::var_os("HAWKING_QWEN_GRAVITY_GPU_COMPARE").is_some() {
        use hawking_core::gravity_llama::gpu::GravityLlamaGpu;
        use hawking_core::metal::MetalContext;
        use hawking_core::numeric_parity::top_k_indices_f32;

        let gpu = GravityLlamaGpu::open_with(
            MetalContext::new().expect("Metal context"),
            &artifact,
            true,
        )
        .expect("open resident Qwen Gravity GPU artifact");
        let (gpu_logits, stats) = gpu
            .forward(&source_tokens)
            .expect("complete Qwen GPU direct-grammar forward");
        let gpu_greedy = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("nonempty GPU vocabulary")
            .0;
        let top5_cpu = top_k_indices_f32(&logits, 5);
        let top5_gpu = top_k_indices_f32(&gpu_logits, 5);
        let max_abs = logits
            .iter()
            .zip(&gpu_logits)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        let reference_l2 = logits
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let error_l2 = logits
            .iter()
            .zip(&gpu_logits)
            .map(|(left, right)| (*left as f64 - *right as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative_l2 = error_l2 / reference_l2.max(f64::MIN_POSITIVE);
        assert_eq!(gpu_greedy, greedy, "same-artifact GPU greedy decision");
        assert_eq!(top5_gpu, top5_cpu, "same-artifact GPU top-5 decisions");
        assert!(stats.dispatches > 0, "GPU comparison emitted no dispatches");
        if let Some(reference) = f64_authority.as_deref() {
            let score = score_pair(
                &logits,
                &gpu_logits,
                reference,
                &Bounds::full_forward_logits(),
            );
            assert!(
                score.pass,
                "Qwen CPU/GPU failed V2.1 against packed-byte f64 authority: host={:?} device={:?}",
                score.host.failures, score.device.failures
            );
            eprintln!(
                "QWEN_SOURCE_GRAVITY_GPU_V21 prompt_tokens={} host={} device={} dispatches={} fallback=0 promotion=FORBIDDEN",
                source_tokens.len(),
                format_score_line(&score.host),
                format_score_line(&score.device),
                stats.dispatches,
            );
        }
        eprintln!(
            "QWEN_SOURCE_GRAVITY_GPU_COMPARE prompt_tokens={} greedy_token={} top5={:?} max_abs={:.9e} relative_l2={:.9e} dispatches={} command_buffers={} cpu_fallback=0 parity_authority=CPU_F32_ONLY_V21_NOT_CLAIMED",
            source_tokens.len(),
            gpu_greedy,
            top5_gpu,
            max_abs,
            relative_l2,
            stats.dispatches,
            stats.command_buffers,
        );
    }
}
