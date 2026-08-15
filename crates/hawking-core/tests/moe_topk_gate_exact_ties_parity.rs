//! Parallel `moe_topk_gate` (tie_epsilon == 0) must match the serial mask-
//! and-pick scan on deliberate exact ties — including ties that straddle the
//! top_k boundary. This is the property most likely to break under a
//! reduction rewrite and least likely to show up in a normal decode.
#![cfg(target_os = "macos")]

use hawking_core::kernels::{self, topk_softmax_batch};
use hawking_core::metal::TokenCommandBuffer;

mod common;
use common::*;

fn run_metal_topk(logits: &[f32], n_experts: usize, top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let ctx = ctx();
    let logits_buf = new_f32_buf(ctx, logits);
    let ids_buf = ctx.new_buffer(top_k * std::mem::size_of::<u32>());
    let weights_buf = ctx.new_buffer(top_k * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::moe_topk_gate_tcb(&mut tcb, &logits_buf, &ids_buf, &weights_buf, n_experts, top_k)
            .expect("moe_topk_gate_tcb encode");
        tcb.commit_and_wait().expect("moe_topk_gate commit");
    }
    let ids = unsafe { std::slice::from_raw_parts(ids_buf.contents() as *const u32, top_k).to_vec() };
    let weights =
        unsafe { std::slice::from_raw_parts(weights_buf.contents() as *const f32, top_k).to_vec() };
    (ids, weights)
}

fn assert_serial_parallel_match(logits: &[f32], n_experts: usize, top_k: usize, label: &str) {
    assert_eq!(logits.len(), n_experts, "{label}: logits len");
    // Ensure default tie policy: unit test must exercise the parallel path.
    assert!(
        std::env::var("HAWKING_DS_ROUTE_TIE_EPS").is_err()
            || std::env::var("HAWKING_DS_ROUTE_TIE_EPS").unwrap() == "0"
            || std::env::var("HAWKING_DS_ROUTE_TIE_EPS").unwrap() == "0.0",
        "{label}: HAWKING_DS_ROUTE_TIE_EPS must be unset or 0 so the parallel path runs"
    );

    let mut serial_ids = vec![0u32; top_k];
    let mut serial_weights = vec![0.0f32; top_k];
    topk_softmax_batch(
        logits,
        1,
        n_experts,
        top_k,
        &mut serial_ids,
        &mut serial_weights,
    );

    let (metal_ids, metal_weights) = run_metal_topk(logits, n_experts, top_k);

    // Ids are the load-bearing contract: parallel lex-max of (value, -index)
    // must match the serial left-to-right strict-`>` scan, including every
    // exact-tie pattern. Softmax *weights* may differ in the ULP sense from
    // the CPU oracle (Metal exp vs libm exp); that is not the property under
    // test here.
    assert_eq!(
        metal_ids, serial_ids,
        "{label}: expert ids mismatch\n  serial={serial_ids:?}\n  metal ={metal_ids:?}"
    );
    let mut max_w_err = 0.0f32;
    for (&sw, &mw) in serial_weights.iter().zip(metal_weights.iter()) {
        max_w_err = max_w_err.max((sw - mw).abs());
    }
    assert!(
        max_w_err <= 1e-5,
        "{label}: softmax weights drifted too far from CPU oracle (max_abs={max_w_err})"
    );
}

/// Several identical top logits; lowest indices must win, and the 8th/9th
/// experts that share the boundary value must resolve as serial does.
#[test]
fn exact_ties_across_topk_boundary_qwen30_shape() {
    let n_experts = 128usize;
    let top_k = 8usize;
    // Base floor so softmax is well-defined; then plant exact ties.
    let mut logits = vec![-3.0f32; n_experts];
    // Twelve experts share the same highest logit — top_k=8 must take the
    // eight lowest indices among them (0..7 if we assign them to 0..11).
    for i in 0..12 {
        logits[i] = 5.0;
    }
    // A second exact-tie cluster below the winners, and one unique mid value.
    for i in 20..28 {
        logits[i] = 1.0;
    }
    logits[40] = 2.0;
    // Scattered lower unique values so the rest of the vector is not flat.
    for i in 50..n_experts {
        logits[i] = -1.0 - ((i % 17) as f32) * 0.01;
    }

    assert_serial_parallel_match(&logits, n_experts, top_k, "boundary-ties-128x8");

    // Serial expectation for the planted pattern: indices 0..7.
    let (ids, _) = run_metal_topk(&logits, n_experts, top_k);
    assert_eq!(ids, (0..8).map(|i| i as u32).collect::<Vec<_>>());
}

/// Exact ties only after softmax would still be exact (equal logits → equal
/// probs). Mid-pack duplicate values must not reorder under the parallel tree.
#[test]
fn exact_ties_interior_and_tail_indices() {
    let n_experts = 128usize;
    let top_k = 8usize;
    let mut logits = vec![-4.0f32; n_experts];
    // Distinct descending winners for the first 5 slots.
    for (rank, &idx) in [90u32, 10, 50, 3, 70].iter().enumerate() {
        logits[idx as usize] = 10.0 - rank as f32;
    }
    // Three-way exact tie for the remaining top_k slots (need 3 more).
    // Lowest indices among {100, 25, 110, 15} should fill positions 5..7
    // → 15, 25, 100 (110 is the 4th and must fall outside top_k).
    for &idx in &[100usize, 25, 110, 15] {
        logits[idx] = 4.0;
    }

    assert_serial_parallel_match(&logits, n_experts, top_k, "interior-ties-128x8");

    let (ids, _) = run_metal_topk(&logits, n_experts, top_k);
    assert_eq!(ids[..5], [90, 10, 50, 3, 70]);
    assert_eq!(ids[5..], [15, 25, 100]);
}

/// DeepSeek-V2-Lite-ish shape (64 experts, top-6) with a full-vector tie.
#[test]
fn exact_all_equal_logits_lowest_indices_win() {
    let n_experts = 64usize;
    let top_k = 6usize;
    let logits = vec![0.25f32; n_experts];
    assert_serial_parallel_match(&logits, n_experts, top_k, "all-equal-64x6");
    let (ids, _) = run_metal_topk(&logits, n_experts, top_k);
    assert_eq!(ids, (0..6).map(|i| i as u32).collect::<Vec<_>>());
}

/// No ties: parallel path must still match serial on a typical random-ish
/// router vector (regression against pure reduction bugs).
#[test]
fn no_ties_randomish_matches_serial() {
    let n_experts = 128usize;
    let top_k = 8usize;
    let logits: Vec<f32> = (0..n_experts)
        .map(|i| ((i * 37 + 11) % n_experts) as f32 / n_experts as f32 * 12.0 - 6.0)
        .collect();
    // Confirm the fixture has unique values so we are not accidentally in
    // the tie suite above.
    let mut sorted = logits.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in sorted.windows(2) {
        assert!(w[0] < w[1], "fixture must be strictly increasing when sorted");
    }
    assert_serial_parallel_match(&logits, n_experts, top_k, "no-ties-128x8");
}
