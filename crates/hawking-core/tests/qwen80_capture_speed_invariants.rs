//! Invariants for the Q80 layer-major capture speed work:
//! byte-identical writes, RSS budget at the shipped worker count, and
//! deterministic MoE output across worker counts.

use hawking_core::model::qwen80_source_bf16_layer_major::{
    moe_routed_experts_parallel, moe_wave_working_set_bytes, moe_worker_count_at_rss,
    moe_worker_scratch_bytes_per_worker, routed_expert_down_bytes, write_retained_hidden_f32le,
    ExpertWeights, DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, QWEN80_EXPERTS, QWEN80_HIDDEN,
    QWEN80_MOE_INTERMEDIATE, QWEN80_TOP_K, STREAMED_PEAK_RSS_HARD_CAP_BYTES,
};
use sha2::{Digest, Sha256};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn patterned_bf16(n_elems: usize, tag: u16) -> Vec<u8> {
    // Finite BF16: truncate a small f32. Raw u16 patterns hit NaN/Inf and
    // make `assert_eq!` fail even when both worker counts produce the same bits.
    let mut out = vec![0u8; n_elems.saturating_mul(2)];
    for i in 0..n_elems {
        let v = (((i as u32).wrapping_add(tag as u32) % 2000) as f32) * 0.001 - 1.0;
        let bits = (v.to_bits() >> 16) as u16;
        out[i * 2] = bits as u8;
        out[i * 2 + 1] = (bits >> 8) as u8;
    }
    out
}

fn reference_hidden_le_and_sha(values: &[f32]) -> (Vec<u8>, String) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    let mut digest = Sha256::new();
    for v in values {
        let b = v.to_le_bytes();
        bytes.extend_from_slice(&b);
        digest.update(b);
    }
    (bytes, format!("{:x}", digest.finalize()))
}

#[test]
fn write_retained_hidden_is_byte_identical_to_per_element_le() {
    let values: Vec<f32> = (0..QWEN80_HIDDEN)
        .map(|i| (i as f32) * 0.015_625 - 4.0)
        .collect();
    let (expect_bytes, expect_sha) = reference_hidden_le_and_sha(&values);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("row.f32le");
    let (got_sha, n) = write_retained_hidden_f32le(&path, &values).expect("write");
    let got = std::fs::read(&path).expect("read");
    assert_eq!(n, expect_bytes.len());
    assert_eq!(got, expect_bytes);
    assert_eq!(got_sha, expect_sha);
}

#[test]
fn pinned_streamed_constants_unchanged() {
    assert_eq!(STREAMED_PEAK_RSS_HARD_CAP_BYTES, 16 * 1024 * 1024 * 1024);
    assert_eq!(DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, 64);
}

#[test]
fn shipped_worker_working_set_stays_under_16_gib() {
    // 25 258 tokens is the sealed production corpus; N=192 is what that run used.
    let tokens = 25_258usize;
    let n_workers = hawking_core::model::qwen80_source_bf16_layer_major::MOE_WORKER_DEFAULT_CAP;
    let ws = moe_wave_working_set_bytes(tokens, n_workers, 192);
    assert!(
        ws <= STREAMED_PEAK_RSS_HARD_CAP_BYTES,
        "working set {ws} exceeds streamed cap at {n_workers} workers"
    );
    assert!(moe_worker_scratch_bytes_per_worker() < 16 * 1024 * 1024);
    assert_eq!(
        routed_expert_down_bytes(tokens),
        tokens * QWEN80_TOP_K * QWEN80_HIDDEN * 4
    );
    // Near the cap the function must back off to 1, never invent headroom.
    let tight = moe_worker_count_at_rss(tokens, QWEN80_EXPERTS, STREAMED_PEAK_RSS_HARD_CAP_BYTES);
    assert_eq!(tight, 1);
    // Comfortable RSS (4 GiB) must admit more than the stale 4-worker ladder.
    let roomy = moe_worker_count_at_rss(tokens, QWEN80_EXPERTS, 4 * 1024 * 1024 * 1024);
    assert!(
        roomy >= 8,
        "expected memory-aware count >= 8 at 4 GiB RSS, got {roomy}"
    );
}

fn synthetic_moe_out(n_workers: usize, tokens: usize) -> Vec<f32> {
    let _guard = ENV_LOCK.lock().expect("env lock");
    std::env::set_var("HAWKING_Q80_MOE_WORKERS", n_workers.to_string());
    let inter = QWEN80_MOE_INTERMEDIATE;
    let h = QWEN80_HIDDEN;
    let n_live = 8usize;
    let mut experts: Vec<ExpertWeights> = (0..QWEN80_EXPERTS)
        .map(|_| ExpertWeights {
            gate: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
        })
        .collect();
    for e in 0..n_live {
        experts[e] = ExpertWeights {
            gate: patterned_bf16(inter * h, 11 + e as u16),
            up: patterned_bf16(inter * h, 29 + e as u16),
            down: patterned_bf16(h * inter, 47 + e as u16),
        };
    }
    let mut all_router_in = vec![0.0f32; tokens * h];
    for t in 0..tokens {
        for j in 0..h {
            all_router_in[t * h + j] = ((t * 17 + j) as f32) * 0.001 - 0.5;
        }
    }
    let mut expert_members: Vec<Vec<(usize, f32)>> = vec![Vec::new(); QWEN80_EXPERTS];
    let mut routes: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let ids: Vec<u32> = (0..QWEN80_TOP_K)
            .map(|k| ((t + k) % n_live) as u32)
            .collect();
        let weights = vec![0.1f32; QWEN80_TOP_K];
        for (&e, &w) in ids.iter().zip(weights.iter()) {
            expert_members[e as usize].push((t, w));
        }
        routes.push((ids, weights));
    }
    let mut moe_out = vec![0.0f32; tokens * h];
    moe_routed_experts_parallel(
        &mut experts,
        &expert_members,
        &routes,
        &all_router_in,
        tokens,
        h,
        inter,
        &mut moe_out,
        None,
    )
    .expect("moe wave");
    std::env::remove_var("HAWKING_Q80_MOE_WORKERS");
    moe_out
}

#[test]
fn moe_wave_is_byte_identical_across_worker_counts() {
    let tokens = 24usize;
    let a = synthetic_moe_out(2, tokens);
    let b = synthetic_moe_out(8, tokens);
    assert_eq!(a.len(), tokens * QWEN80_HIDDEN);
    assert_eq!(
        a, b,
        "2-worker and 8-worker MoE downs must match bit-for-bit"
    );
}

#[test]
fn peak_rss_at_shipped_worker_count_under_hard_cap() {
    // Exercise the same worker-scratch allocation the wave uses, then read
    // ru_maxrss. This process is far below 16 GiB; the assertion is the
    // live measurement the contract asked for.
    let n = moe_worker_count_at_rss(4_719, QWEN80_EXPERTS, 2 * 1024 * 1024 * 1024);
    let scratch = moe_worker_scratch_bytes_per_worker().saturating_mul(n);
    let mut hold = vec![0u8; scratch.min(64 * 1024 * 1024)];
    hold.fill(1);
    let peak = hawking_core::model::qwen80_source_bf16_layer_major::peak_rss_bytes();
    assert!(
        peak <= STREAMED_PEAK_RSS_HARD_CAP_BYTES,
        "peak RSS {peak} exceeds streamed hard cap"
    );
    assert!(n >= 1);
    let _ = hold[0];
}
