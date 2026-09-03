//! Does resuming from a prefix checkpoint produce the SAME tokens as a full prefill?
//!
//! The argument for correctness is that the DeltaNet carry after tokens 0..N is
//! a function of those tokens alone, so two prompts sharing a prefix have
//! identical state at N by construction, and KV is positionally indexed so the
//! entries at 0..N are already the same bytes whichever request wrote them.
//!
//! That is an argument. This is the measurement. An argument that a cache is
//! sound is worth exactly as much as a receipt saying a test passed.
//!
//! The bar is BIT-IDENTITY, not similarity: resuming must produce the same
//! token ids as prefilling the whole prompt, or the cache is silently changing
//! what the model says.
//!
//!     cargo run -p hawking-core --profile release-fast \
//!       --example qwen38_prefix_checkpoint_parity -- \
//!       --artifact <root> --tokenizer <tokenizer.json> [--max-new 24]
//!
//! Exit 0 only on bit-identical output.

#[cfg(target_os = "macos")]
fn main() {
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy_reusing, load_qwen38_tokenizer, Qwen38HybridDecodeSession,
        Qwen38HybridWeights,
    };
    use std::sync::Arc;

    let mut artifact = String::new();
    let mut tokenizer_path = String::new();
    let mut max_new = 24usize;
    let mut max_seq_len = 8192usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--artifact" => { artifact = args[i + 1].clone(); i += 2; }
            "--tokenizer" => { tokenizer_path = args[i + 1].clone(); i += 2; }
            "--max-new" => { max_new = args[i + 1].parse().unwrap_or(24); i += 2; }
            "--max-seq-len" => { max_seq_len = args[i + 1].parse().unwrap_or(8192); i += 2; }
            other => { eprintln!("unknown argument {other}"); std::process::exit(2); }
        }
    }
    if artifact.is_empty() || tokenizer_path.is_empty() {
        eprintln!("--artifact and --tokenizer are required");
        std::process::exit(2);
    }

    let weights = Arc::new(
        Qwen38HybridWeights::load(&artifact).unwrap_or_else(|e| {
            eprintln!("load artifact: {e}");
            std::process::exit(1);
        }),
    );
    let tokenizer = load_qwen38_tokenizer(&tokenizer_path).unwrap_or_else(|e| {
        eprintln!("load tokenizer: {e}");
        std::process::exit(1);
    });
    let mut session = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), max_seq_len)
        .unwrap_or_else(|e| {
            eprintln!("attach session: {e}");
            std::process::exit(1);
        });

    // A shared prefix, then two different continuations -- the exact shape of
    // two goals that share a system prompt and tool catalog.
    let shared = "You are a careful engineering assistant. Answer precisely.\n\n";
    let tail_a = "Question: name one property of a prefix cache.\nAnswer:";

    let prefix_ids = tokenizer.encode(shared, false).expect("encode prefix");
    let full_ids = tokenizer
        .encode(&format!("{shared}{tail_a}"), false)
        .expect("encode full");

    // ---- Baseline: prefill the WHOLE prompt from a clean session.
    session.reset();
    let baseline = generate_greedy_reusing(&mut session, &full_ids, max_new, 0)
        .expect("baseline generation");
    let baseline_new = baseline.new_tokens().to_vec();

    // ---- Checkpointed: prefill only the shared prefix, snapshot, then resume.
    session.reset();
    let _warm = generate_greedy_reusing(&mut session, &prefix_ids, 1, 0)
        .expect("prefix generation");
    let checkpoint = session.prefix_checkpoint().expect("checkpoint");
    println!(
        "checkpoint: position={} rec_state={} f32 conv_state={} f32",
        checkpoint.position,
        checkpoint.rec_state.len(),
        checkpoint.conv_state.len()
    );

    // Something else entirely, to prove the restore is doing the work rather
    // than the session merely still holding the right state by accident.
    session.reset();
    let _other = generate_greedy_reusing(
        &mut session,
        &tokenizer.encode("Completely unrelated text about weather.", false).unwrap(),
        4,
        0,
    );

    session.restore_prefix(&checkpoint).expect("restore");
    let resumed = generate_greedy_reusing(
        &mut session,
        &full_ids,
        max_new,
        checkpoint.position.min(full_ids.len().saturating_sub(1)),
    )
    .expect("resumed generation");
    let resumed_new = resumed.new_tokens().to_vec();

    println!("prefix tokens        : {}", prefix_ids.len());
    println!("full prompt tokens   : {}", full_ids.len());
    println!("baseline generated   : {baseline_new:?}");
    println!("resumed  generated   : {resumed_new:?}");
    println!(
        "prefill steps: baseline {} vs resumed {} (skipped {})",
        full_ids.len(),
        full_ids.len() - checkpoint.position.min(full_ids.len().saturating_sub(1)),
        checkpoint.position.min(full_ids.len().saturating_sub(1)),
    );

    if baseline_new == resumed_new {
        println!("\nPARITY: BIT-IDENTICAL. The checkpoint is sound on this body.");
    } else {
        println!("\nPARITY: DIVERGED. The checkpoint is NOT sound; do not enable it.");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen38 native decode is Metal-only");
    std::process::exit(2);
}
