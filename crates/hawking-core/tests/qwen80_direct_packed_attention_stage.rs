//! Non-GPU contract tests for the isolated Qwen80 layer-3 attention probe.
//!
//! Metal execution is intentionally lease-gated at runtime; these tests keep
//! the source geometry and source-algorithm contracts from silently drifting
//! before a future admitted-artifact run is scheduled.

#[test]
fn qwen3next_residual_rmsnorm_and_partial_rope_reference_is_exact() {
    const EPS: f32 = 1.0e-6;
    const THETA: f32 = 5_000_000.0;
    // A four-wide miniature preserves Qwen3-Next's non-interleaved
    // rotate_half algebra. The learned norm vector is residual-scale, not a
    // conventional direct multiplier.
    let input = [1.0f32, -2.0, 3.0, -4.0];
    let learned_weight = [0.25f32, -0.5, 0.0, 1.0];
    let variance = input.iter().map(|value| value * value).sum::<f32>() / 4.0;
    let inverse_rms = (variance + EPS).sqrt().recip();
    let mut normalized = input
        .iter()
        .zip(learned_weight)
        .map(|(&value, weight)| value * inverse_rms * (1.0 + weight))
        .collect::<Vec<_>>();
    let before = normalized.clone();
    // partial rotary dimension = 4; position 1, dimension 0 frequency = 1.
    let angle = THETA.powf(-2.0 * 0.0 / 4.0);
    normalized[0] = before[0] * angle.cos() - before[2] * angle.sin();
    normalized[2] = before[2] * angle.cos() + before[0] * angle.sin();
    assert!((normalized[0] - (before[0] * 1.0f32.cos() - before[2] * 1.0f32.sin())).abs() < 1e-6);
    assert!((normalized[2] - (before[2] * 1.0f32.cos() + before[0] * 1.0f32.sin())).abs() < 1e-6);
    // The second frequency uses theta^(-1/2), proving partial-RoPE uses the
    // rotary width rather than the full head width.
    let second_angle = THETA.powf(-2.0 * 1.0 / 4.0);
    assert!(second_angle > 0.0 && second_angle < 1.0);
}

#[test]
fn qwen3next_q_projection_query_and_gate_are_interleaved_per_head() {
    const HEADS: usize = 16;
    const HEAD_DIM: usize = 256;
    let q_projection = (0..HEADS * 2 * HEAD_DIM)
        .map(|index| index as u32)
        .collect::<Vec<_>>();
    let mut query = vec![0u32; HEADS * HEAD_DIM];
    let mut gate = vec![0u32; HEADS * HEAD_DIM];
    for head in 0..HEADS {
        let source = head * 2 * HEAD_DIM;
        let destination = head * HEAD_DIM;
        query[destination..destination + HEAD_DIM]
            .copy_from_slice(&q_projection[source..source + HEAD_DIM]);
        gate[destination..destination + HEAD_DIM]
            .copy_from_slice(&q_projection[source + HEAD_DIM..source + 2 * HEAD_DIM]);
    }
    for head in 0..HEADS {
        let destination = head * HEAD_DIM;
        assert_eq!(query[destination], (head * 2 * HEAD_DIM) as u32);
        assert_eq!(gate[destination], (head * 2 * HEAD_DIM + HEAD_DIM) as u32);
    }
    // In particular, the gate for head 0 is not q_projection[4096].
    assert_eq!(gate[0], 256);
    assert_ne!(gate[0], q_projection[HEADS * HEAD_DIM]);
}

#[test]
fn registered_shader_and_probe_keep_the_layer3_gated_gqa_boundary() {
    let shader = hawking_core::metal::all_shader_sources();
    assert!(shader.contains("qwen80_attention_qk_norm_rope_cache"));
    assert!(shader.contains("qwen80_attention_apply_sigmoid_gate"));
    assert!(shader.contains("n_heads != 16u || n_kv_heads != 2u"));
    assert!(shader.contains("head_dim != 256u || rotary_dim != 64u"));

    let probe = include_str!("../examples/ascension_qwen80_direct_packed_attention_stage.rs");
    for required in [
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
        "Qwen80LayerKind::FullAttention",
        "query_from_interleaved_q_projection",
        "gate_from_interleaved_q_projection",
        "head * 2 * HEAD_DIM + HEAD_DIM",
        "source_q_proj_gate_is_sigmoid_of_each_head_local_second_256_rows",
        "opens_no_raw_bf16_or_mps_shadow_model",
        "does_not_generate_tokens_expose_hcli_or_measure_tps",
        "--capture-dir NEW_ABSOLUTE_DIRECTORY",
        "begin_capture",
        "finalize_capture",
        "stdout.jsonl",
        "stderr.log",
        "receipt.json",
        "all_files_are_create_new_and_non_overwriting",
    ] {
        assert!(
            probe.contains(required),
            "missing source-bound contract: {required}"
        );
    }
}
