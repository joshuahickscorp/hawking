//! Hash-bound CPU stage trace for a DeepSeek Gravity artifact.
//!
//! This is an oracle-side companion to `HAWKING_DS_STAGE_TRACE`.  It emits the
//! same compact boundary names (length, bytes, SHA-256, finite first eight
//! values) so a resident Metal run can be compared without dumping tensors.

use hawking_core::gravity_deepseek::{DeepSeekLayerStageTrace, GravityDeepSeek};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn f32_summary(label: &str, values: &[f32]) -> serde_json::Value {
    let bytes = bytemuck::cast_slice(values);
    let first: Vec<serde_json::Value> = values
        .iter()
        .take(8)
        .map(|value| {
            if value.is_finite() {
                serde_json::json!(value)
            } else {
                serde_json::Value::Null
            }
        })
        .collect();
    serde_json::json!({
        "label": label,
        "len": values.len(),
        "bytes": bytes.len(),
        "sha256": format!("{:x}", Sha256::digest(bytes)),
        "first8": first,
    })
}

fn u32_summary(label: &str, values: &[u32]) -> serde_json::Value {
    serde_json::json!({
        "label": label,
        "len": values.len(),
        "bytes": values.len() * std::mem::size_of::<u32>(),
        "values": values,
    })
}

fn layer_json(layer: &DeepSeekLayerStageTrace) -> serde_json::Value {
    let routes: Vec<u32> = layer.routes.iter().map(|(id, _)| *id as u32).collect();
    let route_weights: Vec<f32> = layer.routes.iter().map(|(_, weight)| *weight).collect();
    serde_json::json!({
        "layer": layer.layer,
        "buffers": {
            "input": f32_summary("input", &layer.input),
            "attn_norm": f32_summary("attn_norm", &layer.attn_norm),
            "q": f32_summary("q", &layer.q),
            "kv_a": f32_summary("kv_a", &layer.kv_a),
            "c_kv": f32_summary("c_kv", &layer.c_kv),
            "k_pe": f32_summary("k_pe", &layer.k_pe),
            "context": f32_summary("context", &layer.context),
            "attn_out": f32_summary("attn_out", &layer.attn_out),
            "after_attention": f32_summary("after_attention", &layer.after_attention),
            "ffn_norm": f32_summary("ffn_norm", &layer.ffn_norm),
            "router_logits": f32_summary("router_logits", &layer.router_logits),
            "routes": u32_summary("routes", &routes),
            "route_weights": f32_summary("route_weights", &route_weights),
            "ffn_out": f32_summary("ffn_out", &layer.ffn_out),
            "output": f32_summary("output", &layer.output),
        },
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut artifact = None::<PathBuf>;
    let mut tokens_file = None::<PathBuf>;
    let mut out = None::<PathBuf>;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--tokens-file" => tokens_file = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    let tokens_file = tokens_file.ok_or("--tokens-file is required")?;
    let tokens: Vec<u32> = std::fs::read_to_string(tokens_file)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("tokens file is empty".into());
    }
    let model = GravityDeepSeek::open(&artifact, true)?;
    let (logits, trace) = model.forward_with_stage_trace(&tokens)?;
    let rows: Vec<serde_json::Value> = trace
        .iter()
        .map(|token| {
            serde_json::json!({
                "schema": "hawking.gravity.deepseek_stage_trace.v1",
                "token": token.token,
                "position": token.position,
                "layers": token.layers.iter().map(layer_json).collect::<Vec<_>>(),
                "final_norm": f32_summary("final_norm", &token.final_norm),
                "logits": f32_summary("logits", &token.logits),
            })
        })
        .collect();
    let value = serde_json::json!({
        "schema": "hawking.gravity.deepseek_stage_trace_bundle.v1",
        "artifact": artifact,
        "prompt_tokens": tokens,
        "architecture": {
            "layers": model.arch.n_layers,
            "hidden": model.arch.hidden,
            "experts": model.arch.n_routed_experts,
            "experts_per_token": model.arch.num_experts_per_tok,
        },
        "logits": f32_summary("logits", &logits),
        "tokens": rows,
        "note": "CPU raw-quant oracle; not a TPS or TG-rung claim",
    });
    let text = serde_json::to_string_pretty(&value)?;
    if let Some(path) = out {
        std::fs::write(path, text)?;
    } else {
        println!("{text}");
    }
    Ok(())
}
