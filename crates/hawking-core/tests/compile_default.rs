//! Compile-time coverage for every default-feature hawking-core integration
//! source. The fast lane invokes this target with `cargo check`; it does not
//! execute the potentially stateful or hardware-backed test cases.

#[path = "add_rmsnorm_fused_q8_parity.rs"]
mod add_rmsnorm_fused_q8_parity;
#[path = "broker_kernel_ab_gate.rs"]
mod broker_kernel_ab_gate;
#[path = "chat_special_token_detok.rs"]
mod chat_special_token_detok;
#[path = "checkpoint_refuses_across_reset.rs"]
mod checkpoint_refuses_across_reset;
#[path = "chunk_workspace_residency.rs"]
mod chunk_workspace_residency;
#[path = "cpu_backend_parity.rs"]
mod cpu_backend_parity;
#[path = "cpu_backend_parity_deepseek.rs"]
mod cpu_backend_parity_deepseek;
#[path = "dsv4f_admission_trust.rs"]
mod dsv4f_admission_trust;
#[path = "dsv4f_artifact_index.rs"]
mod dsv4f_artifact_index;
#[path = "dsv4f_native_token_graph_e2e.rs"]
mod dsv4f_native_token_graph_e2e;
#[path = "dsv4f_streamed_native_e2e.rs"]
mod dsv4f_streamed_native_e2e;
#[path = "dsv4f_zerocopy_reader.rs"]
mod dsv4f_zerocopy_reader;
#[path = "e3_user_draft_gate_rule.rs"]
mod e3_user_draft_gate_rule;
#[path = "embed_rmsnorm_fused_parity.rs"]
mod embed_rmsnorm_fused_parity;
#[path = "gemm_q4k_v4r_predec_parity.rs"]
mod gemm_q4k_v4r_predec_parity;
#[path = "gemma2_smoke.rs"]
mod gemma2_smoke;
#[path = "gk_family_parity.rs"]
mod gk_family_parity;
#[path = "gravity_artifact_suite.rs"]
mod gravity_artifact_suite;
#[path = "gravity_glm_compact_mla_live_parity.rs"]
mod gravity_glm_compact_mla_live_parity;
#[path = "gravity_glm_compact_mla_parity.rs"]
mod gravity_glm_compact_mla_parity;
#[path = "gravity_glm_device_only_mlp.rs"]
mod gravity_glm_device_only_mlp;
#[path = "gravity_glm_expert_wave_parity.rs"]
mod gravity_glm_expert_wave_parity;
#[path = "gravity_glm_forward.rs"]
mod gravity_glm_forward;
#[path = "gravity_glm_gpu_forward.rs"]
mod gravity_glm_gpu_forward;
#[path = "gravity_glm_icb_encode_benchmark.rs"]
mod gravity_glm_icb_encode_benchmark;
#[path = "gravity_glm_lm_head_device_parity.rs"]
mod gravity_glm_lm_head_device_parity;
#[path = "gravity_glm_math_preserve_byte_census.rs"]
mod gravity_glm_math_preserve_byte_census;
#[path = "gravity_glm_resident_parity.rs"]
mod gravity_glm_resident_parity;
#[path = "gravity_llama_forward.rs"]
mod gravity_llama_forward;
#[path = "gravity_pq_kernel_registry.rs"]
mod gravity_pq_kernel_registry;
#[path = "gravity_pq_kernel_variants_metal.rs"]
mod gravity_pq_kernel_variants_metal;
#[path = "gravity_raw32_metal_parity.rs"]
mod gravity_raw32_metal_parity;
#[path = "gravity_residual_pq_metal.rs"]
mod gravity_residual_pq_metal;
#[path = "greedy_token_only_parity.rs"]
mod greedy_token_only_parity;
#[path = "integration_greedy_64.rs"]
mod integration_greedy_64;
#[path = "llama32_smoke.rs"]
mod llama32_smoke;
#[path = "llama_cpp_oracle.rs"]
mod llama_cpp_oracle;
#[path = "load_robustness.rs"]
mod load_robustness;
#[path = "mamba2_smoke.rs"]
mod mamba2_smoke;
#[path = "megakernel_2layer_parity.rs"]
mod megakernel_2layer_parity;
#[path = "mha_decode_f16kv_parity.rs"]
mod mha_decode_f16kv_parity;
#[path = "mha_decode_flash_int4kv_parity.rs"]
mod mha_decode_flash_int4kv_parity;
#[path = "mha_never_silently_overruns.rs"]
mod mha_never_silently_overruns;
#[path = "mixed_quant_store_build.rs"]
mod mixed_quant_store_build;
#[path = "moe_topk_gate_exact_ties_parity.rs"]
mod moe_topk_gate_exact_ties_parity;
#[path = "multiseq_aggregate_bench.rs"]
mod multiseq_aggregate_bench;
#[path = "multiseq_churn_parity.rs"]
mod multiseq_churn_parity;
#[path = "multiseq_decode_parity.rs"]
mod multiseq_decode_parity;
#[path = "multiseq_mha_parity.rs"]
mod multiseq_mha_parity;
#[path = "p3_batched_prefill_parity.rs"]
mod p3_batched_prefill_parity;
#[path = "pair_8r_parity.rs"]
mod pair_8r_parity;
#[path = "phase1_kernel_parity.rs"]
mod phase1_kernel_parity;
#[path = "phase2_foundation_parity.rs"]
mod phase2_foundation_parity;
#[path = "phase2_mla_metal_parity.rs"]
mod phase2_mla_metal_parity;
#[path = "phase2_weight_pinning_parity.rs"]
mod phase2_weight_pinning_parity;
#[path = "phi3_smoke.rs"]
mod phi3_smoke;
#[path = "prefill_slot_into_multiseq_parity.rs"]
mod prefill_slot_into_multiseq_parity;
#[path = "prefix_cache_e2e.rs"]
mod prefix_cache_e2e;
#[path = "prefix_cache_parity.rs"]
mod prefix_cache_parity;
#[path = "q3k_fused_2r_parity.rs"]
mod q3k_fused_2r_parity;
#[path = "q3k_predec_parity.rs"]
mod q3k_predec_parity;
#[path = "q4k_batched_gemm_parity.rs"]
mod q4k_batched_gemm_parity;
#[path = "q4k_fast_parity.rs"]
mod q4k_fast_parity;
#[path = "q6k_gemv_parity.rs"]
mod q6k_gemv_parity;
#[path = "q6k_swiglu_2r_parity.rs"]
mod q6k_swiglu_2r_parity;
#[path = "q6k_swiglu_4r_parity.rs"]
mod q6k_swiglu_4r_parity;
#[path = "qkv_concurrent_parity.rs"]
mod qkv_concurrent_parity;
#[path = "quantize_int8_kernel_parity.rs"]
mod quantize_int8_kernel_parity;
#[path = "qwen30_direct_packed_gate_up_swiglu_fused_metal.rs"]
mod qwen30_direct_packed_gate_up_swiglu_fused_metal;
#[path = "qwen38_batched_prefill_constrained.rs"]
mod qwen38_batched_prefill_constrained;
#[path = "qwen38_batched_prefill_greedy.rs"]
mod qwen38_batched_prefill_greedy;
#[path = "qwen38_prefill_gemm_parity.rs"]
mod qwen38_prefill_gemm_parity;
#[path = "qwen80_capture_speed_invariants.rs"]
mod qwen80_capture_speed_invariants;
#[path = "qwen80_direct_packed_attention_stage.rs"]
mod qwen80_direct_packed_attention_stage;
#[path = "qwen80_uniform_q4_hybrid_decode.rs"]
mod qwen80_uniform_q4_hybrid_decode;
#[path = "qwen_q4k_metal_pipeline_fixture.rs"]
mod qwen_q4k_metal_pipeline_fixture;
#[path = "qwen_source_gravity_live.rs"]
mod qwen_source_gravity_live;
#[path = "ram_prefix_cache_e2e.rs"]
mod ram_prefix_cache_e2e;
#[path = "rwkv7_metal_bench.rs"]
mod rwkv7_metal_bench;
#[path = "rwkv7_metal_parity.rs"]
mod rwkv7_metal_parity;
#[path = "rwkv7_multiseq_parity.rs"]
mod rwkv7_multiseq_parity;
#[path = "rwkv7_parity.rs"]
mod rwkv7_parity;
#[path = "rwkv7_prefill_slot_multiseq_parity.rs"]
mod rwkv7_prefill_slot_multiseq_parity;
#[path = "rwkv7_state_checkpoint_parity.rs"]
mod rwkv7_state_checkpoint_parity;
#[path = "rwkv_world_tokenizer_parity.rs"]
mod rwkv_world_tokenizer_parity;
#[path = "sidecar_tier_map_consume.rs"]
mod sidecar_tier_map_consume;
#[path = "tcb_dispatch_cost.rs"]
mod tcb_dispatch_cost;
#[path = "user_draft_parity_e2e.rs"]
mod user_draft_parity_e2e;
#[path = "w4a8_activation_distribution.rs"]
mod w4a8_activation_distribution;
#[path = "w4a8_per_channel_calibrate.rs"]
mod w4a8_per_channel_calibrate;
#[path = "w4a8_per_channel_lmhead_kernel_parity.rs"]
mod w4a8_per_channel_lmhead_kernel_parity;
#[path = "w4a8_per_channel_parity.rs"]
mod w4a8_per_channel_parity;
#[path = "w4a8_qwen3b_quality_gate.rs"]
mod w4a8_qwen3b_quality_gate;
