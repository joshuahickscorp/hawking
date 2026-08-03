//! CPU correctness probe for a source-preserving DeepSeek Gravity artifact.

use hawking_core::gravity_deepseek::GravityDeepSeek;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut artifact = None::<PathBuf>;
    let mut tokens_file = None::<PathBuf>;
    let mut generate = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--tokens-file" => tokens_file = args.next().map(PathBuf::from),
            "--generate" => generate = args.next().ok_or("--generate needs a count")?.parse()?,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    let tokens_file = tokens_file.ok_or("--tokens-file is required")?;
    let tokens: Vec<u32> = std::fs::read_to_string(tokens_file)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("tokens file is empty".into());
    }
    let model = GravityDeepSeek::open(&artifact, true)?;
    let mut logits = model.forward(&tokens)?;
    let mut generated = Vec::with_capacity(generate);
    for _ in 0..generate {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as u32)
            .ok_or("empty logits")?;
        generated.push(next);
        logits = model.forward_at(&[next], tokens.len() + generated.len() - 1)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hawking.gravity.deepseek_probe.v1",
            "artifact": artifact,
            "prompt_tokens": tokens,
            "generated_tokens": generated,
            "architecture": {
                "layers": model.arch.n_layers,
                "hidden": model.arch.hidden,
                "experts": model.arch.n_routed_experts,
                "experts_per_token": model.arch.num_experts_per_tok,
            },
            "note": "CPU raw-quant source adapter probe; not a TPS or rung claim"
        }))?
    );
    Ok(())
}
