//! Path B parity tests — every parallel-K kernel must match K sequential
//! single-token runs of the corresponding existing kernel (within atol=1e-3
//! fp16, matching the existing phase1 kernel parity gate).
//!
//! These tests are scaffolded as #[ignore] until the kernel bodies land.
//! Once a kernel is implemented, remove the #[ignore] for its test.

use dismantle_core::kernels::parallel_k;

#[test]
#[ignore = "Path B kernel not yet implemented; see reports/path_to_90/path_b/design.md"]
fn mla_decode_kbatch_matches_sequential_k4() {
    // Plan: construct K=4 random q_nope/q_rope queries, a fixed KV cache,
    // run mla_decode_kernel_fc K times sequentially to get the reference,
    // run mla_decode_kernel_fc_kbatch once to get the parallel result,
    // assert each (k, out_dim) element matches within 1e-3 fp16 atol.

    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    // Currently returns Unimplemented; once the kernel lands, remove
    // #[ignore] and replace with the real parity comparison.
    let result = parallel_k::mla_decode_kernel_fc_kbatch(
        &dummy, &dummy, &dummy, &dummy, &dummy, &mut out, 1, 4,
    );
    assert!(
        result.is_err(),
        "skeleton phase: kernel should still be Unimplemented",
    );
}

#[test]
#[ignore = "Path B kernel not yet implemented; see reports/path_to_90/path_b/design.md"]
fn gemv_q6_k_v3_kbatch_matches_sequential_k4() {
    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    let result = parallel_k::gemv_q6_k_v3_kbatch(
        &dummy, &dummy, &mut out, 1, 1, 4,
    );
    assert!(result.is_err());
}

#[test]
#[ignore = "Path B kernel not yet implemented; see reports/path_to_90/path_b/design.md"]
fn moe_block_kbatch_matches_sequential_k4() {
    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    let result = parallel_k::moe_block_batched_indexed_kbatch(
        &dummy, &dummy, &dummy, &dummy,
        &[0u32], &[0u32], &[0.0f32],
        &mut out, 1, 1, 4,
    );
    assert!(result.is_err());
}

#[test]
#[ignore = "Tree-decode extension not yet implemented; see reports/path_to_90/tree_decode/design.md"]
fn mla_decode_kbatch_tree_mask_matches_unmasked_when_all_zero() {
    // Once landed: a zero-bias mask (all 0s) should produce identical output
    // to the unmasked version. Validates the mask-arg surface is wired
    // without changing the math when mask is trivial.
    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    let zero_mask = vec![0.0f32; 16];
    let result = parallel_k::mla_decode_kernel_fc_kbatch_masked(
        &dummy, &dummy, &dummy, &dummy, &dummy, &mut out, &zero_mask, 1, 4,
    );
    assert!(result.is_err());
}
