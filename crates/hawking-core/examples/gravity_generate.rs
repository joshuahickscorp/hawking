//! End to end: prompt in, streamed text out, entirely from a `.gravity` artifact.
//!
//! This is §4.4's production path with nothing stubbed -- local `.gravity`, the resident
//! Metal runtime, a real tokenizer, a real sampler, tokens emitted as they are produced.
//! No source weights are opened at any point; the artifact is the only thing on disk the
//! model reads.
//!
//! The tokenizer directory comes from the artifact's own header rather than a flag,
//! because a container that needs to be told where its tokenizer lives is not
//! self-describing. A missing tokenizer is an error, not a fall back to raw token ids:
//! printing ids and calling it output would hide exactly the failure this proves absent.
//!
//!     cargo run --release -p hawking-core --example gravity_generate -- \
//!         --prompt "The capital of France is" --tokens 24

use std::path::PathBuf;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    use std::time::Instant;

    use hawking_core::gravity::GravityShard;
    use hawking_core::gravity_llama::gpu::GravityLlamaGpu;
    use hawking_core::metal::MetalContext;
    use hawking_core::sample::Sampler;
    use hawking_core::SamplingParams;
    use hawking_core::tokenizer::Tokenizer;

    let mut artifact: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 24usize;
    let mut temperature = 0.0f32;
    let mut seed = 20260724u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--prompt" => prompt = args.next().ok_or("--prompt needs a value")?,
            "--tokens" => max_tokens = args.next().ok_or("--tokens needs a value")?.parse()?,
            "--temperature" => {
                temperature = args.next().ok_or("--temperature needs a value")?.parse()?
            }
            "--seed" => seed = args.next().ok_or("--seed needs a value")?.parse()?,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let artifact = artifact.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join("Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.v2.gravity")
    });
    if !artifact.is_file() {
        return Err(format!("no artifact at {artifact:?}").into());
    }

    // The tokenizer is named by the artifact, not by the caller.
    let shard = GravityShard::open(&artifact)?;
    let tok_dir = shard
        .extra
        .get("tokenizer")
        .and_then(|t| t.get("dir"))
        .and_then(serde_json::Value::as_str)
        .ok_or("artifact header declares no tokenizer directory")?;
    let tok_path = PathBuf::from(tok_dir).join(
        shard
            .extra
            .get("tokenizer")
            .and_then(|t| t.get("source"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("tokenizer.json"),
    );
    if !tok_path.is_file() {
        return Err(format!(
            "artifact names tokenizer {tok_path:?}, which is not present; refusing to \
             emit raw token ids in its place"
        )
        .into());
    }
    let tokenizer = Tokenizer::from_file(&tok_path)?;
    drop(shard);

    let ctx = MetalContext::new()?;
    let model = GravityLlamaGpu::open_with(ctx, &artifact, true)?;

    let mut ids = tokenizer.encode(&prompt, true)?;
    let prompt_len = ids.len();
    eprintln!(
        "artifact {} | {} prompt tokens | tokenizer {}",
        artifact.file_name().unwrap().to_string_lossy(),
        prompt_len,
        tok_path.file_name().unwrap().to_string_lossy()
    );
    print!("{prompt}");
    std::io::stdout().flush()?;

    let mut sampler = Sampler::new(seed);
    let params = SamplingParams {
        temperature,
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut emitted = 0usize;

    // Prefill the whole prompt once, then extend one token at a time. Each
    // position owns its cache slot, so continuing is a matter of not
    // resetting -- replaying the prefix would reach identical logits, slower.
    let mut logits = model.forward(&ids)?.0;
    let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut pos = ids.len();

    for _ in 0..max_tokens {
        let next = sampler.sample(&mut logits, &params);
        if tokenizer.is_eog(next) {
            break;
        }
        print!("{}", tokenizer.decode_one(next)?);
        std::io::stdout().flush()?;
        ids.push(next);
        emitted += 1;
        logits = model.forward_at(&[next], pos)?.0;
        pos += 1;
    }
    println!();

    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "\n{emitted} tokens in {wall:.2}s | ttft {ttft_ms:.0} ms | \
         {:.2} tok/s end-to-end (incremental decode)",
        emitted as f64 / wall
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_generate drives the Metal runtime; it only runs on macOS");
}
