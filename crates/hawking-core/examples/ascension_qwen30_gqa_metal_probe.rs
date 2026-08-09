//! Exact Qwen3-Coder-30B GQA cached-attention component probe.
//!
//! This binds the direct Metal MHA primitive to the locally pinned official
//! 32-query-head / 4-KV-head / 128-dimension Qwen30 configuration and checks
//! it against a deterministic CPU oracle.  It does not project Q/K/V, apply
//! RoPE, execute layers or experts, generate a token, or measure TPS.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::metal::MetalContext;
    use hawking_core::model::qwen_moe::{
        dispatch_qwen30_gqa_attention_component, qwen30_gqa_topology_from_hf_config,
        QWEN30_CODER_GQA_TOPOLOGY,
    };
    use serde_json::{json, Value};
    use sha2::Digest;
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const SEQUENCE_LENGTH: usize = 64;

    fn parse_out() -> Result<PathBuf, Box<dyn Error>> {
        let mut values = env::args().skip(1);
        if values.next().as_deref() != Some("--out") {
            return Err("usage: ascension_qwen30_gqa_metal_probe --out <receipt.json>".into());
        }
        Ok(PathBuf::from(values.next().ok_or("missing output path")?))
    }

    fn source_config() -> Result<(Value, PathBuf, String), Box<dyn Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(
                "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct/config.json",
            );
        let raw = fs::read(&path)?;
        let sha256 = format!("{:x}", sha2::Sha256::digest(&raw));
        Ok((serde_json::from_slice(&raw)?, path, sha256))
    }

    fn cpu_oracle(query: &[f32], key_cache: &[f32], value_cache: &[f32]) -> Vec<f32> {
        let topology = QWEN30_CODER_GQA_TOPOLOGY;
        let mut output = vec![0.0; topology.n_heads * topology.head_dim];
        let scale = 1.0 / (topology.head_dim as f32).sqrt();
        let group_size = topology.n_heads / topology.n_kv_heads;
        for head in 0..topology.n_heads {
            let kv_head = head / group_size;
            let q_base = head * topology.head_dim;
            let mut scores = Vec::with_capacity(SEQUENCE_LENGTH);
            for token in 0..SEQUENCE_LENGTH {
                let kv_base = (token * topology.n_kv_heads + kv_head) * topology.head_dim;
                let dot = (0..topology.head_dim)
                    .map(|dimension| query[q_base + dimension] * key_cache[kv_base + dimension])
                    .sum::<f32>();
                scores.push(dot * scale);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normalizer = scores
                .iter()
                .map(|score| (score - max_score).exp())
                .sum::<f32>();
            for (token, score) in scores.iter().enumerate() {
                let weight = (*score - max_score).exp() / normalizer;
                let kv_base = (token * topology.n_kv_heads + kv_head) * topology.head_dim;
                for dimension in 0..topology.head_dim {
                    output[q_base + dimension] += weight * value_cache[kv_base + dimension];
                }
            }
        }
        output
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let out = parse_out()?;
        let (config, config_path, config_sha256) = source_config()?;
        let topology = qwen30_gqa_topology_from_hf_config(&config)?;
        let query = (0..topology.n_heads * topology.head_dim)
            .map(|index| ((index * 13 % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let kv_values = SEQUENCE_LENGTH * topology.n_kv_heads * topology.head_dim;
        let key_cache = (0..kv_values)
            .map(|index| ((index * 17 % 509) as f32 - 254.0) / 509.0)
            .collect::<Vec<_>>();
        let value_cache = (0..kv_values)
            .map(|index| ((index * 19 % 389) as f32 - 194.0) / 389.0)
            .collect::<Vec<_>>();
        let expected = cpu_oracle(&query, &key_cache, &value_cache);
        let metal = MetalContext::new()?;
        let observed = dispatch_qwen30_gqa_attention_component(
            &metal,
            topology,
            &query,
            &key_cache,
            &value_cache,
            SEQUENCE_LENGTH,
        )?;
        let max_abs_error = expected
            .iter()
            .zip(&observed)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        if max_abs_error > 3e-5 {
            return Err(
                format!("Qwen30 GQA Metal parity failed: max_abs_error={max_abs_error}").into(),
            );
        }
        let report = json!({
            "schema": "hawking.ascension.qwen30_gqa_metal_component_probe.v1",
            "status": "PASS_DIRECT_METAL_QWEN30_GQA_ATTENTION_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE",
            "device": metal.device_name(),
            "official_config": {"path": config_path, "sha256": config_sha256},
            "official_qwen30_geometry": {"query_heads": topology.n_heads, "kv_heads": topology.n_kv_heads, "head_dim": topology.head_dim, "sequence_length": SEQUENCE_LENGTH},
            "max_abs_output_error": max_abs_error,
            "claim_boundary": {"component_uses_deterministic_oracle_inputs_not_model_weights": true, "qkv_projection_rope_kv_write_moe_residual_and_decoder_not_run": true, "not_100_tps_or_tg3": true, "not_full_model_or_manager_qualification": true}
        });
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &out,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
