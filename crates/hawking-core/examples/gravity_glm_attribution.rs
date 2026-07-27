//! Attribution instrument: is Math-Preserve's degenerate output the artifact
//! or the path that serves it?
//!
//! Serving `GLM-5.2-H0.98-Math-Preserve.gravity` over HTTP produced
//! byte-identical `99999999` (token id 24, eight times) for two semantically
//! unrelated prompts. Prompt-independence is consistent with BOTH a collapsed
//! artifact and a path that never conditions on the prompt, so it discriminates
//! neither. This prints the intermediate state that does.
//!
//! For each prompt it reports:
//!   1. the token ids the artifact's own tokenizer produces  -- if these are
//!      empty or identical across prompts, the prompt never reached the model
//!      and the answer is H2 without any logit work;
//!   2. the top-5 next-token logits after a full prefill    -- if these are
//!      identical across prompts while the ids differ, the model is not
//!      conditioning; if they differ but are degenerate, the artifact is;
//!   3. simple health statistics of the logit vector        -- a constant or
//!      non-finite vector is a different failure from a peaked-but-wrong one.
//!
//!     cargo run --release -p hawking-core --example gravity_glm_attribution
//!
//! Read-only with respect to the artifact. Does not modify `gravity_glm.rs`.

use std::path::PathBuf;

const DEFAULT_MODEL_DIR: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/GLM-5.2-H0.98-Math-Preserve.gravity";

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::gravity_glm::gpu::GravityGlmGpu;
    use hawking_core::tokenizer::Tokenizer;

    let mut dir: Option<PathBuf> = None;
    let mut verify_hash = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--verify-hash" => verify_hash = true,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let dir = dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(DEFAULT_MODEL_DIR)
    });

    let tok_path = dir.join("tokenizer/tokenizer.json");
    eprintln!("tokenizer: {}", tok_path.display());
    let tokenizer = Tokenizer::from_file(&tok_path)?;

    eprintln!("opening {} (verify_hash={verify_hash})...", dir.display());
    let model = GravityGlmGpu::open_dir(&dir, verify_hash)?;
    eprintln!(
        "opened | layers={} hidden={} routed_experts={} vocab={}",
        model.arch.n_layers, model.arch.hidden, model.arch.n_routed_experts, model.arch.vocab_size
    );

    // Deliberately raw prompts, not chat-templated: this asks what the model
    // does with tokens, separating that from any template question.
    // The last two are the SERVED forms: what `render_glm_chat` emits for the
    // same user message, per its own tests -- `[gMASK]<sop><|system|>Reasoning
    // Effort: Max` ... `<|user|>{msg}<|assistant|><think>`. Including them here
    // is what separates "the artifact is wrong" from "the serve path is wrong",
    // since the HTTP request went through the template and this instrument's
    // raw prompts did not.
    // A bisect over the control-token prefix, holding the user text fixed. Raw
    // text conditions (different argmax per prompt); the fully templated form
    // collapses onto digits. Adding one control construct at a time says which
    // one causes it.
    let prompts = [
        "The capital of France is",
        "[gMASK]<sop>The capital of France is",
        "[gMASK]<sop><|user|>The capital of France is<|assistant|>",
        "[gMASK]<sop><|user|>The capital of France is<|assistant|><think>",
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>The capital of France is<|assistant|><think>",
    ];

    let mut all_top1 = Vec::new();
    for p in prompts {
        let ids = tokenizer.encode(p, true)?;
        println!("\n=== prompt: {p:?}");
        println!("  n_tokens: {}", ids.len());
        println!("  ids: {:?}", &ids[..ids.len().min(24)]);
        let decoded: Vec<String> = ids
            .iter()
            .take(12)
            .map(|&i| tokenizer.decode_one(i).unwrap_or_else(|_| "<?>".into()))
            .collect();
        println!("  pieces: {decoded:?}");

        let (logits, _) = model.forward(&ids)?;

        let finite = logits.iter().filter(|v| v.is_finite()).count();
        let (mut lo, mut hi, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &v in &logits {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
                sum += v as f64;
            }
        }
        let mean = sum / finite.max(1) as f64;
        println!(
            "  logits: len={} finite={} min={lo:.6} max={hi:.6} mean={mean:.6} spread={:.6}",
            logits.len(),
            finite,
            hi - lo
        );

        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        print!("  top5:");
        for &i in idx.iter().take(5) {
            let piece = tokenizer.decode_one(i as u32).unwrap_or_else(|_| "<?>".into());
            print!(" [{i}]{piece:?}={:.5}", logits[i]);
        }
        println!();
        all_top1.push((p, idx[0], logits[idx[0]], hi - lo));
    }

    println!("\n=== attribution");
    let distinct: std::collections::BTreeSet<usize> =
        all_top1.iter().map(|(_, i, _, _)| *i).collect();
    if distinct.len() == 1 {
        let (_, id, _, spread) = all_top1[0];
        println!("  every prompt has the SAME argmax: id {id}");
        if spread < 1e-3 {
            println!("  and the logit vector is essentially FLAT (spread {spread:.6})");
            println!("  => the model is not producing a meaningful distribution at all");
        } else {
            println!("  but the logit vector is NOT flat (spread {spread:.6})");
            println!("  => a real, peaked, prompt-independent distribution: collapse, not a dead vector");
        }
    } else {
        println!("  argmax DIFFERS across prompts: {distinct:?}");
        println!("  => the model DOES condition on its input at this level;");
        println!("     a constant HTTP answer would then be a defect above this layer");
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_glm_attribution requires macOS/Metal");
}
