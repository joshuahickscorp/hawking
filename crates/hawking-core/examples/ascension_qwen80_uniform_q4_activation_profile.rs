//! Profile the five host-activation classes on the uniform-Q4 hybrid decode.
//!
//! This is a VELOCITY BASELINE instrument, not BASE_TRUE_TPS.
//!
//! ```text
//! cargo run --release -p hawking-core --example ascension_qwen80_uniform_q4_activation_profile -- \
//!   --prompt Hi --max-new-tokens 2
//! ```

use hawking_core::model::qwen80_complete_runtime::qwen80_assert_native_operator_composition_complete;
use hawking_core::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_tokenizer, discover_qwen80_uniform_q4_root, generate_greedy,
    load_qwen80_tokenizer, qwen80_default_tokenizer_path, render_qwen80_source_user_chat,
    Qwen80UniformQ4HybridDecodeSession, Qwen80UniformQ4StreamingCatalog,
};
use std::env;
use std::process;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    qwen80_assert_native_operator_composition_complete().map_err(|e| e.to_string())?;
    let mut prompt = "Hi".to_owned();
    let mut max_new_tokens = 2usize;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--prompt" => prompt = args.next().ok_or("missing --prompt")?,
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .ok_or("missing --max-new-tokens")?
                    .parse()
                    .map_err(|_| "bad --max-new-tokens")?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let root = discover_qwen80_uniform_q4_root().ok_or("uniform-q4 artifact not found")?;
    let tokenizer_path = discover_qwen80_tokenizer().unwrap_or_else(qwen80_default_tokenizer_path);
    let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).map_err(|e| e.to_string())?;
    let tokenizer = load_qwen80_tokenizer(&tokenizer_path).map_err(|e| e.to_string())?;
    let mut session =
        Qwen80UniformQ4HybridDecodeSession::new(catalog, 64).map_err(|e| e.to_string())?;
    let rendered = render_qwen80_source_user_chat(&prompt);
    let result = generate_greedy(&mut session, &tokenizer, &rendered, max_new_tokens)
        .map_err(|e| e.to_string())?;
    let act = &result.stages.activation;
    let counts = &result.activation_counts;
    let forwards =
        result.prompt_token_ids.len() + result.generated_token_ids.len().saturating_sub(1);
    println!("generated_token_ids={:?}", result.generated_token_ids);
    println!(
        "forwards={} generated={} prefill_secs={:.6} decode_secs={:.6} steady_state_tok_s={:.6}",
        forwards,
        result.generated_token_ids.len(),
        result.prefill_secs,
        result.decode_secs,
        result.steady_state_tok_s
    );
    println!(
        "fallbacks total={} host_activation={} host_q4_matvec={} host_expert_bind={} host_sample={} host_q4_vector_decode={}",
        result.fallbacks.total(),
        result.fallbacks.host_activation,
        result.fallbacks.host_q4_matvec,
        result.fallbacks.host_expert_payload_bind,
        result.fallbacks.host_sample,
        result.fallbacks.host_q4_vector_decode
    );
    println!(
        "native matvec={} embed={} expert_waves={} device_activation={}",
        result.native.q4_matvec_dispatches,
        result.native.q4_embedding_dispatches,
        result.native.expert_table_waves,
        result.native.device_activation_dispatches
    );
    println!(
        "class_secs shared_swiglu={:.6} shared_mlp_sandwich={:.6} deltanet_conv={:.6} deltanet_recurrent={:.6} gqa_input_layernorm={:.6} gqa_norm_rope={:.6} other_host_activation={:.6} metal_matvec_sync={:.6}",
        act.shared_swiglu_secs,
        act.shared_mlp_sandwich_secs,
        act.deltanet_conv_secs,
        act.deltanet_recurrent_secs,
        act.gqa_input_layernorm_secs,
        act.gqa_norm_rope_secs,
        act.other_host_activation_secs,
        act.metal_matvec_sync_secs
    );
    println!(
        "class_counts shared_swiglu={} deltanet_conv={} deltanet_recurrent={} gqa_input_layernorm={} gqa_norm_rope={} other_host_activation={}",
        counts.shared_swiglu,
        counts.deltanet_conv,
        counts.deltanet_recurrent,
        counts.gqa_input_layernorm,
        counts.gqa_norm_rope,
        counts.other_host_activation
    );
    if forwards > 0 {
        let n = forwards as f64;
        println!(
            "per_token_secs shared_swiglu={:.6} shared_mlp_sandwich={:.6} deltanet_conv={:.6} deltanet_recurrent={:.6} gqa_input_layernorm={:.6} gqa_norm_rope={:.6} other_host_activation={:.6} metal_matvec_sync={:.6}",
            act.shared_swiglu_secs / n,
            act.shared_mlp_sandwich_secs / n,
            act.deltanet_conv_secs / n,
            act.deltanet_recurrent_secs / n,
            act.gqa_input_layernorm_secs / n,
            act.gqa_norm_rope_secs / n,
            act.other_host_activation_secs / n,
            act.metal_matvec_sync_secs / n
        );
    }
    println!(
        "stage_secs embed={:.4} deltanet={:.4} gqa={:.4} moe_shared={:.4} moe_routed={:.4} moe_table_build={:.4} moe_norm_router={:.4} moe_combine={:.4} terminal={:.4} q4_matvec={:.4}",
        result.stages.embed_secs,
        result.stages.deltanet_secs,
        result.stages.gqa_secs,
        result.stages.moe_shared_secs,
        result.stages.moe_routed_secs,
        result.stages.moe_table_build_secs,
        result.stages.moe_norm_router_secs,
        result.stages.moe_combine_secs,
        result.stages.terminal_secs,
        result.stages.q4_matvec_secs
    );
    println!("peak_rss_bytes={}", result.peak_rss_bytes);
    if let Some(error) = &result.metal_error {
        println!("metal_error={error}");
    }
    Ok(())
}
