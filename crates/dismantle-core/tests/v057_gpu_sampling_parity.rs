//! v0.5.7 GPU sampling parity tests.
//!
//! Tests for:
//! - parallel sample_argmax_f32 (tied values, empty, 102400 vocab)
//! - sample_topk for K ∈ {1, 8, 32, 64}
//! - sample_topp for P ∈ {0.1, 0.5, 0.9, 1.0}
//! - sample_multinomial vs CPU for fixed uniform variate
//! - sample_full_pipeline temperature=0 path matches argmax

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{
    gpu_argmax_logits_metal,
    sample_topk_metal, sample_topp_metal, sample_multinomial_metal,
    sample_full_pipeline_metal,
};
use dismantle_core::metal::MetalContext;

fn make_ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn cpu_argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv { best = i as u32; bv = v; }
    }
    best
}

fn cpu_softmax(vals: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = vals.iter().map(|&v| if temperature > 0.0 { v / temperature } else { v }).collect();
    let max_v = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn cpu_topk(logits: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
    let idx: Vec<u32> = top.iter().map(|&(i, _)| i as u32).collect();
    let val: Vec<f32> = top.iter().map(|&(_, v)| v).collect();
    (idx, val)
}

fn dispatch_topk(ctx: &MetalContext, logits: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let n = logits.len();
    let logits_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(logits));
    let topk_idx_buf = ctx.new_buffer(k * std::mem::size_of::<u32>());
    let topk_val_buf = ctx.new_buffer(k * std::mem::size_of::<f32>());
    sample_topk_metal(ctx, &logits_buf, &topk_idx_buf, &topk_val_buf, n, k).expect("topk");
    let idx_ptr = topk_idx_buf.contents() as *const u32;
    let val_ptr = topk_val_buf.contents() as *const f32;
    let idx: Vec<u32> = unsafe { std::slice::from_raw_parts(idx_ptr, k) }.to_vec();
    let val: Vec<f32> = unsafe { std::slice::from_raw_parts(val_ptr, k) }.to_vec();
    (idx, val)
}

// ── parallel_argmax_matches_cpu ──────────────────────────────────────────────

#[test]
fn parallel_argmax_matches_cpu_basic() {
    let ctx = make_ctx();
    for &vocab in &[1024usize, 8192, 102400] {
        let mut logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.001).collect();
        let target = vocab / 3 + 7;
        logits[target] = 9999.0;

        let cpu = cpu_argmax(&logits);
        let logits_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&logits));
        let out_buf = ctx.new_buffer(std::mem::size_of::<u32>());
        gpu_argmax_logits_metal(&ctx, &logits_buf, &out_buf, vocab).expect("argmax");
        let gpu = unsafe { *(out_buf.contents() as *const u32) };

        assert_eq!(gpu, cpu, "vocab={vocab} gpu={gpu} cpu={cpu}");
    }
}

#[test]
fn parallel_argmax_tied_values_lower_index_wins() {
    let ctx = make_ctx();
    // All same value → first (index 0) should win (lower index wins on tie)
    let n = 1024usize;
    let logits = vec![1.0f32; n];
    let logits_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&logits));
    let out_buf = ctx.new_buffer(std::mem::size_of::<u32>());
    gpu_argmax_logits_metal(&ctx, &logits_buf, &out_buf, n).expect("argmax");
    let gpu = unsafe { *(out_buf.contents() as *const u32) };
    assert_eq!(gpu, 0u32, "tied: lower index should win, got {gpu}");
}

#[test]
fn parallel_argmax_empty_input() {
    let ctx = make_ctx();
    let logits: Vec<f32> = vec![];
    let logits_buf = ctx.new_buffer(4); // 1 float min allocation
    let out_buf = ctx.new_buffer(std::mem::size_of::<u32>());
    // n=0: kernel writes token[0] = 0 and returns
    gpu_argmax_logits_metal(&ctx, &logits_buf, &out_buf, 0).expect("argmax empty");
    let gpu = unsafe { *(out_buf.contents() as *const u32) };
    assert_eq!(gpu, 0u32, "empty input should return 0");
    let _ = logits; // suppress unused warning
}

// ── sample_topk ─────────────────────────────────────────────────────────────

#[test]
fn topk_matches_cpu() {
    let ctx = make_ctx();
    for &vocab in &[1024usize, 102400] {
        for &k in &[1usize, 8, 32, 64] {
            // Logits with clear ordering: index i has value i * 0.001
            let logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.001).collect();

            let (cpu_idx, _cpu_val) = cpu_topk(&logits, k);
            let (gpu_idx, gpu_val) = dispatch_topk(&ctx, &logits, k);

            // Check all GPU indices are in the CPU top-K set
            let cpu_set: std::collections::HashSet<u32> = cpu_idx.iter().cloned().collect();
            for (j, &idx) in gpu_idx.iter().enumerate() {
                assert!(
                    cpu_set.contains(&idx),
                    "vocab={vocab} k={k}: gpu topk[{j}]={idx} not in cpu top-{k}"
                );
                // Value should be finite
                assert!(gpu_val[j].is_finite(), "vocab={vocab} k={k}: topk_val[{j}] not finite");
            }
            // GPU top-K should have exactly k unique indices
            let gpu_set: std::collections::HashSet<u32> = gpu_idx.iter().cloned().collect();
            assert_eq!(gpu_set.len(), k, "vocab={vocab} k={k}: expected {k} unique gpu indices");
        }
    }
}

#[test]
fn topk_k1_matches_argmax() {
    let ctx = make_ctx();
    let vocab = 4096usize;
    let mut logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.01).collect();
    logits[1234] = 999.0;

    let cpu_best = cpu_argmax(&logits);
    let (gpu_idx, _) = dispatch_topk(&ctx, &logits, 1);
    assert_eq!(gpu_idx[0], cpu_best, "topk k=1 should match argmax");
}

// ── sample_topp ─────────────────────────────────────────────────────────────

fn cpu_topp(topk_val: &[f32], top_p: f32, temperature: f32) -> (u32, f32) {
    let probs = cpu_softmax(topk_val, temperature);
    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len() as u32;
    let mut s_sum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = (i + 1) as u32;
            break;
        }
    }
    for &p in &probs[..cutoff as usize] {
        s_sum += p;
    }
    (cutoff, s_sum)
}

#[test]
fn topp_matches_cpu() {
    let ctx = make_ctx();
    let k = 32usize;
    // Top-K values: decreasing (first is most likely)
    let topk_val: Vec<f32> = (0..k).map(|i| (k - i) as f32 * 0.1).collect();
    let topk_idx: Vec<u32> = (0..k as u32).collect();

    for &top_p in &[0.1f32, 0.5, 0.9, 1.0] {
        for &temperature in &[1.0f32, 0.5] {
            let (cpu_sc, _cpu_ss) = cpu_topp(&topk_val, top_p, temperature);

            let tv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&topk_val));
            let ti_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&topk_idx));
            let sc_buf = ctx.new_buffer(std::mem::size_of::<u32>());
            let ss_buf = ctx.new_buffer(std::mem::size_of::<f32>());

            sample_topp_metal(&ctx, &tv_buf, &ti_buf, &sc_buf, &ss_buf, k, top_p, temperature)
                .expect("topp");

            let gpu_sc = unsafe { *(sc_buf.contents() as *const u32) };
            let gpu_ss = unsafe { *(ss_buf.contents() as *const f32) };

            assert_eq!(
                gpu_sc, cpu_sc,
                "topp p={top_p} t={temperature}: gpu_sc={gpu_sc} cpu_sc={cpu_sc}"
            );
            assert!(gpu_ss > 0.0,
                "topp p={top_p}: surviving_sum={gpu_ss} must be positive");
        }
    }
}

// ── sample_multinomial ───────────────────────────────────────────────────────

fn cpu_multinomial(topk_val: &[f32], topk_idx: &[u32], sc: u32, ss: f32, u: f32, temperature: f32) -> u32 {
    let sc = sc as usize;
    let scaled: Vec<f32> = topk_val[..sc].iter()
        .map(|&v| if temperature > 0.0 { v / temperature } else { v })
        .collect();
    let max_v = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let target = u * ss;
    let mut cumsum = 0.0f32;
    for i in 0..sc {
        cumsum += (scaled[i] - max_v).exp();
        if cumsum >= target { return topk_idx[i]; }
    }
    topk_idx[if sc > 0 { sc - 1 } else { 0 }]
}

#[test]
fn multinomial_matches_cpu_for_fixed_uniform() {
    let ctx = make_ctx();
    let k = 16usize;
    let topk_val: Vec<f32> = (0..k).map(|i| (k - i) as f32 * 0.1).collect();
    let topk_idx: Vec<u32> = (0..k as u32).collect();
    let temperature = 1.0f32;

    let (cpu_sc, _) = cpu_topp(&topk_val, 0.9, temperature);

    // Run topp to get surviving_sum
    let tv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&topk_val));
    let ti_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&topk_idx));
    let sc_buf = ctx.new_buffer(std::mem::size_of::<u32>());
    let ss_buf = ctx.new_buffer(std::mem::size_of::<f32>());
    sample_topp_metal(&ctx, &tv_buf, &ti_buf, &sc_buf, &ss_buf, k, 0.9, temperature).expect("topp");
    let gpu_sc = unsafe { *(sc_buf.contents() as *const u32) };
    let gpu_ss = unsafe { *(ss_buf.contents() as *const f32) };

    for &u in &[0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
        let cpu_tok = cpu_multinomial(&topk_val, &topk_idx, cpu_sc, gpu_ss, u, temperature);
        let out_buf = ctx.new_buffer(std::mem::size_of::<u32>());
        sample_multinomial_metal(
            &ctx, &tv_buf, &ti_buf, &sc_buf, &ss_buf, u, &out_buf, k, temperature,
        ).expect("multinomial");
        let gpu_tok = unsafe { *(out_buf.contents() as *const u32) };
        assert_eq!(gpu_tok, cpu_tok, "u={u}: gpu={gpu_tok} cpu={cpu_tok}");
    }
}

// ── sample_full_pipeline temperature=0 path ──────────────────────────────────

#[test]
fn full_pipeline_argmax_path() {
    let ctx = make_ctx();
    let vocab = 32000usize;
    let mut logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.001).collect();
    logits[5678] = 9999.0;

    let cpu_best = cpu_argmax(&logits);

    let logits_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&logits));
    let out_buf = ctx.new_buffer(std::mem::size_of::<u32>());

    // temperature=0 → falls back to greedy argmax
    sample_full_pipeline_metal(
        &ctx, &logits_buf, &out_buf, vocab,
        0.0,  // temperature=0 → argmax path
        0, 1.0, 0.5,
    ).expect("full_pipeline greedy");

    let gpu = unsafe { *(out_buf.contents() as *const u32) };
    assert_eq!(gpu, cpu_best, "full_pipeline temperature=0: gpu={gpu} cpu={cpu_best}");
}
