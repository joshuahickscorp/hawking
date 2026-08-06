#![cfg(feature = "tq")]
#![allow(dead_code)]
fn print_bench_result(mode: &str, tps: f32, rss_mb: f32, bpw: f32, accepted_tps: Option<f32>) {
    let spec_col = match accepted_tps {
        Some(a) => format!("  accepted_tps={a:.2}"),
        None => String::new(),
    };
}
#[test]
#[ignore = "stub — implement after TqPreparedGpu dispatch and Metal RWKV-7 forward are wired"]
fn tq_single_stream_tps() {
    let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
    let _q4k_path = std::env::var("RWKV7_Q4K_MODEL")
        .unwrap_or_else(|_| "(not set — skipping baseline)".to_string());
}
#[test]
#[ignore = "stub — implement after TQ loader and Metal buffer allocation are wired"]
fn tq_resident_memory_vs_q4k() {
    let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
    let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");
}
#[test]
#[ignore = "stub — implement after TQ speculative decode is wired in the RWKV-7 pipeline"]
fn tq_spec_decode_accepted_tps() {
    let draft_path =
        std::env::var("RWKV7_TQ_DRAFT_MODEL").expect("RWKV7_TQ_DRAFT_MODEL must be set");
    let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");
}
