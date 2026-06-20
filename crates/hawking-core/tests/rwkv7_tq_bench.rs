//! Bench scaffold for RWKV-7 TQ (trellis-quant) throughput and memory.
//!
//! All tests are `#[ignore]` stubs — they hold signatures and reporting format
//! for three bench gates that gate TQ deployment:
//!
//! 1. Single-stream decode tps vs the Q4_K_M baseline.
//! 2. Resident memory footprint vs Q4_K_M (target: ≤ 70% RSS at comparable
//!    quality, i.e. the sub-4-bit byte-cut pays for itself in memory before
//!    any tps claim).
//! 3. Speculative-decode accepted tps under TQ draft + Q4_K_M verify (the
//!    hawking moat lever: TQ's reduced byte footprint keeps the draft cheap
//!    enough to lift accepted tps above non-spec Q4_K_M).
//!
//! Run with:
//!
//! ```sh
//! RWKV7_TQ_MODEL=/path/to/model.tq \
//! RWKV7_Q4K_MODEL=/path/to/baseline.gguf \
//!   cargo test -p hawking-core --features tq --test rwkv7_tq_bench \
//!     -- --nocapture --ignored
//! ```

#![cfg(feature = "tq")]
#![allow(dead_code)]

/// Print a single bench result line in the canonical hawking format.
///
/// `mode` describes the configuration being measured (e.g. `"TQ k=3 L=7"`).
/// `tps` is decode tokens-per-second (greedy, single stream, warm).
/// `rss_mb` is resident set size in MiB after model load.
/// `bpw` is effective bits-per-weight for the run.
/// `accepted_tps` is the accepted tokens-per-second under speculative decode,
/// or `None` if this measurement is a non-spec run.
fn print_bench_result(mode: &str, tps: f32, rss_mb: f32, bpw: f32, accepted_tps: Option<f32>) {
    let spec_col = match accepted_tps {
        Some(a) => format!("  accepted_tps={a:.2}"),
        None => String::new(),
    };
    println!(
        "[rwkv7_tq_bench]  {mode:<30}  tps={tps:6.2}  rss={rss_mb:7.1} MiB  bpw={bpw:.2}{spec_col}"
    );
}

/// Measure single-stream greedy decode tps for the TQ model and print the
/// result alongside the Q4_K_M baseline.
///
/// Requires:
/// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
/// - `RWKV7_Q4K_MODEL`: path to the Q4_K_M GGUF baseline (for comparison).
///
/// Warm run: 3 primer tokens discarded, then 128 tokens timed.
#[test]
#[ignore = "stub — implement after TqPreparedGpu dispatch and Metal RWKV-7 forward are wired"]
fn tq_single_stream_tps() {
    let tq_path = std::env::var("RWKV7_TQ_MODEL")
        .expect("RWKV7_TQ_MODEL must be set");
    let _q4k_path = std::env::var("RWKV7_Q4K_MODEL")
        .unwrap_or_else(|_| "(not set — skipping baseline)".to_string());

    println!("STUB: not yet implemented");
    println!("  tq_path = {tq_path}");
    println!("  Would time 128-token greedy decode (3-token warm-up discarded)");
    println!("  and call print_bench_result for both TQ and Q4_K_M.");

    // Example of what the output call will look like once wired:
    // print_bench_result("TQ k=3 L=7 (0.4B)", tq_tps, tq_rss_mb, 3.0, None);
    // print_bench_result("Q4_K_M baseline (0.4B)", q4k_tps, q4k_rss_mb, 4.5, None);
}

/// Measure resident memory (RSS) for TQ vs Q4_K_M after model load + one
/// forward pass (to ensure all Metal buffers are allocated).
///
/// Requires:
/// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
/// - `RWKV7_Q4K_MODEL`: path to the Q4_K_M GGUF baseline.
///
/// Gate: `rss_tq_mb <= rss_q4k_mb * 0.70` (30% reduction target).
#[test]
#[ignore = "stub — implement after TQ loader and Metal buffer allocation are wired"]
fn tq_resident_memory_vs_q4k() {
    let tq_path = std::env::var("RWKV7_TQ_MODEL")
        .expect("RWKV7_TQ_MODEL must be set");
    let q4k_path = std::env::var("RWKV7_Q4K_MODEL")
        .expect("RWKV7_Q4K_MODEL must be set");

    println!("STUB: not yet implemented");
    println!("  tq_path  = {tq_path}");
    println!("  q4k_path = {q4k_path}");
    println!("  Would measure RSS after load + 1 forward pass for each model.");
    println!("  Gate: rss_tq <= rss_q4k * 0.70");

    // Example output call once wired:
    // print_bench_result("TQ k=3 L=7 (0.4B)", 0.0, tq_rss_mb, 3.0, None);
    // print_bench_result("Q4_K_M baseline (0.4B)", 0.0, q4k_rss_mb, 4.5, None);
    // assert!(tq_rss_mb <= q4k_rss_mb * 0.70, ...);
}

/// Measure accepted tokens-per-second under speculative decode with TQ draft +
/// Q4_K_M verifier. Prints absolute accepted tps and the lift ratio over
/// non-spec Q4_K_M decode.
///
/// Requires:
/// - `RWKV7_TQ_DRAFT_MODEL`: path to the small-TQ draft `.tq` artifact.
/// - `RWKV7_Q4K_MODEL`: path to the Q4_K_M verifier GGUF.
///
/// Bench length: 256 tokens, K=4 (speculate 4 tokens per verify step).
#[test]
#[ignore = "stub — implement after TQ speculative decode is wired in the RWKV-7 pipeline"]
fn tq_spec_decode_accepted_tps() {
    let draft_path = std::env::var("RWKV7_TQ_DRAFT_MODEL")
        .expect("RWKV7_TQ_DRAFT_MODEL must be set");
    let q4k_path = std::env::var("RWKV7_Q4K_MODEL")
        .expect("RWKV7_Q4K_MODEL must be set");

    println!("STUB: not yet implemented");
    println!("  draft_path = {draft_path}");
    println!("  q4k_path   = {q4k_path}");
    println!("  Would run 256-token speculative decode (K=4) and report accepted tps.");

    // Example output call once wired:
    // let accepted_tps: f32 = /* measured */;
    // let draft_bpw: f32 = /* from artifact header */;
    // print_bench_result("TQ draft + Q4K verify (K=4)", accepted_tps, rss_mb, draft_bpw, Some(accepted_tps));
}
