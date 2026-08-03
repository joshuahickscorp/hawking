//! Diagnostic CPU trace for a source-quantized Gravity Llama shard.
//!
//! This intentionally uses the existing scalar dequant/GEMV authority and
//! only runs layer 0.  It is not a performance path; it answers whether the
//! container's row geometry and raw bytes reproduce the source checkpoint
//! surfaces before Metal arithmetic is considered.

use std::path::PathBuf;

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mean + eps).sqrt().recip();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

fn sum(x: &[f32]) -> f64 {
    x.iter().map(|v| *v as f64).sum()
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::attn::mha_decode_step;
    use hawking_core::gguf::GgmlType;
    use hawking_core::gravity::{widen_native, GravityShard};
    use hawking_core::gravity_llama::GravityLlamaArch;
    use hawking_core::kernels::{gemv_f32, rope_inplace_normal_with_factors, silu_mul};
    use hawking_core::quant;

    let mut artifact = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    let shard = GravityShard::open(&artifact)?;
    let arch = GravityLlamaArch::from_header(&shard.extra)?;
    let tokens = [128000u32, 29531u32];

    let bytes_for = |name: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(shard.read_tensor(name, true)?)
    };
    let dense = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let d = shard.descriptor(name).ok_or("missing dense tensor")?;
        Ok(widen_native(&d.codec, &bytes_for(name)?)?)
    };
    let matvec = |name: &str, x: &[f32]| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let d = shard.descriptor(name).ok_or("missing projection")?;
        let shape = &d.shape;
        if shape.len() != 2 {
            return Err(format!("{name}: bad shape {shape:?}").into());
        }
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        let payload = bytes_for(name)?;
        let mut out = vec![0.0f32; rows];
        match d.codec.as_str() {
            "ggml.q4_k" | "ggml.q6_k" => {
                let (dtype, block_bytes) = if d.codec == "ggml.q4_k" {
                    (GgmlType::Q4_K, 144usize)
                } else {
                    (GgmlType::Q6_K, 210usize)
                };
                let row_bytes = cols / 256 * block_bytes;
                let mut row = vec![0.0f32; cols];
                for r in 0..rows {
                    quant::dequant_into(
                        dtype,
                        &payload[r * row_bytes..(r + 1) * row_bytes],
                        &mut row,
                    )?;
                    let mut acc = 0.0f32;
                    for c in 0..cols {
                        acc += row[c] * x[c];
                    }
                    out[r] = acc;
                }
            }
            codec if codec.starts_with("native.") => {
                gemv_f32(&dense(name)?, rows, cols, x, &mut out);
            }
            other => return Err(format!("{name}: unsupported codec {other}").into()),
        }
        Ok(out)
    };
    let embed_desc = shard
        .descriptor("model.embed_tokens.weight")
        .ok_or("missing embedding")?;
    let embed_bytes = bytes_for("model.embed_tokens.weight")?;
    let embed_row = |token: u32| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cols = arch.hidden;
        let row_bytes = cols / 256 * 144;
        let mut out = vec![0.0f32; cols];
        quant::dequant_into(
            GgmlType::Q4_K,
            &embed_bytes[token as usize * row_bytes..(token as usize + 1) * row_bytes],
            &mut out,
        )?;
        Ok(out)
    };

    let input_norm = dense("model.layers.0.input_layernorm.weight")?;
    let post_norm = dense("model.layers.0.post_attention_layernorm.weight")?;
    let final_norm = dense("model.norm.weight")?;
    let mut keys = Vec::<f32>::new();
    let mut values = Vec::<f32>::new();
    let kv_width = arch.n_kv_heads * arch.head_dim;
    let mut rows = Vec::new();

    for (pos, &token) in tokens.iter().enumerate() {
        let mut x = embed_row(token)?;
        let embedding_sum = sum(&x);
        let x_norm = rmsnorm(&x, &input_norm, arch.rms_norm_eps);
        let mut q = matvec("model.layers.0.self_attn.q_proj.weight", &x_norm)?;
        let mut k = matvec("model.layers.0.self_attn.k_proj.weight", &x_norm)?;
        let v = matvec("model.layers.0.self_attn.v_proj.weight", &x_norm)?;
        let q_raw_sum = sum(&q);
        let k_raw_sum = sum(&k);
        let v_raw_sum = sum(&v);
        for head in q.chunks_exact_mut(arch.head_dim) {
            rope_inplace_normal_with_factors(
                head,
                pos as u32,
                arch.rope_theta,
                arch.rope_scaling,
                arch.rope_freq_factors.as_deref(),
            );
        }
        for head in k.chunks_exact_mut(arch.head_dim) {
            rope_inplace_normal_with_factors(
                head,
                pos as u32,
                arch.rope_theta,
                arch.rope_scaling,
                arch.rope_freq_factors.as_deref(),
            );
        }
        keys.extend_from_slice(&k);
        values.extend_from_slice(&v);
        let q_rope_sum = sum(&q);
        let k_rope_sum = sum(&k);
        let mut attn = vec![0.0f32; arch.n_heads * arch.head_dim];
        mha_decode_step(
            &q,
            &keys,
            &values,
            arch.n_heads,
            arch.n_kv_heads,
            arch.head_dim,
            pos + 1,
            &mut attn,
        )?;
        let o = matvec("model.layers.0.self_attn.o_proj.weight", &attn)?;
        for i in 0..arch.hidden {
            x[i] += o[i];
        }
        let ffn_input_sum = sum(&x);
        let ffn_norm = rmsnorm(&x, &post_norm, arch.rms_norm_eps);
        let gate = matvec("model.layers.0.mlp.gate_proj.weight", &ffn_norm)?;
        let up = matvec("model.layers.0.mlp.up_proj.weight", &ffn_norm)?;
        let mut act = vec![0.0f32; gate.len()];
        silu_mul(&gate, &up, &mut act);
        let f = matvec("model.layers.0.mlp.down_proj.weight", &act)?;
        for i in 0..arch.hidden {
            x[i] += f[i];
        }
        let residual = x;
        let _ = final_norm; // loaded to validate the final native organ exists
        rows.push(serde_json::json!({
            "position": pos,
            "token_id": token,
            "embedding_sum": embedding_sum,
            "attn_norm_sum": sum(&x_norm),
            "q_raw_sum": q_raw_sum,
            "k_raw_sum": k_raw_sum,
            "v_raw_sum": v_raw_sum,
            "q_rope_sum": q_rope_sum,
            "k_rope_sum": k_rope_sum,
            "attn_out_sum": sum(&o),
            "ffn_input_sum": ffn_input_sum,
            "ffn_norm_sum": sum(&ffn_norm),
            "ffn_gate_sum": sum(&gate),
            "ffn_up_sum": sum(&up),
            "ffn_swiglu_sum": sum(&act),
            "ffn_out_sum": sum(&f),
            "layer_out_sum": sum(&residual),
            "kv_width": kv_width,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hawking.gravity.llama_cpu_layer0_trace.v1",
            "artifact": artifact,
            "architecture": {"hidden": arch.hidden, "layers": arch.n_layers},
            "rows": rows,
            "note": "scalar dequant/GEMV diagnostic only; not a throughput or rung claim"
        }))?
    );
    let _ = embed_desc;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_llama_cpu_trace is intended for the macOS Gravity artifact");
}
