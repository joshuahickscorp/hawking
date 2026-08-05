//! Device learned-bias Gate route kernel parity vs F64 host oracle.
//!
//! Dispatches `deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority` on
//! real Metal with synthetic logits + a real layer-3 `gate.bias` from the
//! admitted artifact. Host F64 oracle implements source Gate.forward for
//! non-hash layers:
//!   original = sqrt(softplus(logits));
//!   indices  = topk(original + bias, k=6);
//!   weights  = gather(original) / sum * route_scale
//!
//! This is the route kernel only — not a full P6 MoE composition (expert
//! selection is dynamic, so full learned MoE needs a two-phase load).
//!
//! Usage:
//!   cargo run -p hawking-core --example gravity_deepseek_v4_learned_bias_route_metal -- \
//!     --artifact <full-43-layer-stream.gravity> --out <receipt.json>

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_learned_bias_route_metal requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_layer0_moe::{ACTIVATED_EXPERTS, ROUTED_EXPERTS, ROUTE_SCALE};
    use hawking_core::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
    use hawking_core::metal::MetalContext;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    const KERNEL: &str = "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority";
    const LAYER: usize = 3;
    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.learned_bias_route_metal.v1";
    const PARITY: &str = "NUMERIC_PARITY_V2_1_ONLY";

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let wall = Instant::now();
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(&reader)?;
        let plan = catalog.plan(LAYER)?;
        if plan.gate_mode.as_str() != "learned_scores_with_selection_bias" {
            return Err(format!("layer {LAYER} is not learned-bias gate").into());
        }
        // BOS window attention admits; MoE still refuses (expected).
        plan.require_bos_window_attention_device()?;
        if plan.require_moe_device().is_ok() {
            return Err("expected layer-3 MoE plan to still refuse full P6 composition".into());
        }

        let bias_name = format!("layers.{LAYER}.ffn.gate.bias");
        let bias_meta = reader.tensor_metadata(&bias_name)?;
        if bias_meta.dtype != "F32" || bias_meta.shape.as_slice() != [ROUTED_EXPERTS as u64] {
            return Err("layer-3 gate.bias geometry is not F32[256]".into());
        }
        let bias_bytes = reader.read_verified_full(&bias_name, bias_meta.bytes as usize)?;
        let mut bias = vec![0f32; ROUTED_EXPERTS];
        for (i, slot) in bias.iter_mut().enumerate() {
            *slot = f32::from_le_bytes(bias_bytes[i * 4..i * 4 + 4].try_into()?);
        }

        // Deterministic synthetic logits (not from a real activation — this
        // seals the route kernel alone).
        let mut logits = vec![0f32; ROUTED_EXPERTS];
        for i in 0..ROUTED_EXPERTS {
            logits[i] = ((i as f32) * 0.017 - 1.3).sin() * 2.5 + (i as f32 % 7.0) * 0.11;
        }

        let (oracle_ids, oracle_weights, oracle_scores) = host_learned_route_f64(&logits, &bias)?;

        let metal = MetalContext::new()?;
        let logits_bytes = bytemuck_f32(&logits);
        let bias_bytes_upload = bytemuck_f32(&bias);
        let logits_buf = metal.new_buffer_with_bytes_checked(&logits_bytes)?;
        let bias_buf = metal.new_buffer_with_bytes_checked(&bias_bytes_upload)?;
        let ids_buf = metal.new_buffer_checked(ACTIVATED_EXPERTS * 4)?;
        let weights_buf = metal.new_buffer_checked(ACTIVATED_EXPERTS * 4)?;
        let scores_buf = metal.new_buffer_checked(ROUTED_EXPERTS * 4)?;
        let valid_buf = metal.new_buffer_checked(4)?;
        let expert_count = ROUTED_EXPERTS as u32;
        let top_k = ACTIVATED_EXPERTS as u32;
        let route_scale = ROUTE_SCALE as f32;

        let timing = metal.dispatch_batch_timed(|batch| {
            batch.dispatch_threads(KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&logits_buf), 0);
                encoder.set_buffer(1, Some(&bias_buf), 0);
                encoder.set_buffer(2, Some(&ids_buf), 0);
                encoder.set_buffer(3, Some(&weights_buf), 0);
                encoder.set_buffer(4, Some(&scores_buf), 0);
                encoder.set_buffer(5, Some(&valid_buf), 0);
                set_u32(encoder, 6, &expert_count);
                set_u32(encoder, 7, &top_k);
                set_f32(encoder, 8, &route_scale);
            })
        })?;

        let valid = read_u32(&valid_buf, 1)?[0];
        if valid != 1 {
            return Err(format!("learned-bias route kernel valid code {valid}").into());
        }
        let device_ids = read_u32(&ids_buf, ACTIVATED_EXPERTS)?;
        let device_weights = read_f32(&weights_buf, ACTIVATED_EXPERTS)?;
        let device_scores = read_f32(&scores_buf, ROUTED_EXPERTS)?;

        let mut id_match = true;
        for i in 0..ACTIVATED_EXPERTS {
            if device_ids[i] != oracle_ids[i] {
                id_match = false;
            }
        }
        let mut max_weight_abs = 0f64;
        let mut max_score_abs = 0f64;
        for i in 0..ACTIVATED_EXPERTS {
            max_weight_abs =
                max_weight_abs.max((device_weights[i] as f64 - oracle_weights[i]).abs());
        }
        for i in 0..ROUTED_EXPERTS {
            max_score_abs = max_score_abs.max((device_scores[i] as f64 - oracle_scores[i]).abs());
        }
        // F32 Metal vs F64 host: require exact ID match and tight weight/score tol.
        if !id_match {
            return Err(format!(
                "selected expert IDs differ: device={device_ids:?} oracle={oracle_ids:?}"
            )
            .into());
        }
        if max_weight_abs > 1e-5 || max_score_abs > 1e-5 {
            return Err(format!(
                "weight/score abs err too large: weight={max_weight_abs:.3e} score={max_score_abs:.3e}"
            )
            .into());
        }
        if timing.compute_dispatches == 0 {
            return Err("zero metal dispatches".into());
        }

        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let receipt = json!({
            "schema": RECEIPT_SCHEMA,
            "status": "PASS_LEARNED_BIAS_ROUTE_KERNEL_METAL",
            "artifact": {
                "path": args.artifact.display().to_string(),
                "manifest_seal_sha256": catalog.identity().manifest_seal_sha256,
            },
            "scope": {
                "layer": LAYER,
                "kernel": KERNEL,
                "expert_count": ROUTED_EXPERTS,
                "top_k": ACTIVATED_EXPERTS,
                "route_scale": ROUTE_SCALE,
                "bias_tensor": bias_name,
                "logits": "synthetic_deterministic_not_live_activation",
            },
            "metal": {
                "metal_dispatches": timing.compute_dispatches as usize,
                "command_buffers": timing.command_buffers as usize,
                "cpu_visible_waits": 1,
                "fallback": 0,
            },
            "parity": {
                "classification": PARITY,
                "exact_storage": false,
                "selected_ids_exact_match": true,
                "selected_ids": device_ids,
                "max_abs_weight_err_f64": max_weight_abs,
                "max_abs_score_err_f64": max_score_abs,
            },
            "honesty": {
                "full_p6_moe_composed": false,
                "reason": "dynamic topk requires two-phase expert load; this receipt seals the route kernel only",
                "serve_endpoint_flipped": false,
            },
            "wall_time_ms": wall_ms,
        });

        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        let pretty = serde_json::to_string_pretty(&receipt)?;
        let mut file = File::create(&args.out)?;
        file.write_all(pretty.as_bytes())?;
        file.write_all(b"\n")?;
        let seal = format!("{:x}", Sha256::digest(pretty.as_bytes()));
        println!("{pretty}");
        println!("receipt_path: {}", args.out.display());
        println!("receipt_sha256: {seal}");
        println!(
            "metal_dispatches: {} selected_ids: {:?}",
            timing.compute_dispatches, device_ids
        );
        println!("status: PASS_LEARNED_BIAS_ROUTE_KERNEL_METAL");
        println!("parity: {PARITY}");
        Ok(())
    }

    fn host_learned_route_f64(
        logits: &[f32],
        bias: &[f32],
    ) -> ProbeResult<([u32; ACTIVATED_EXPERTS], [f64; ACTIVATED_EXPERTS], Vec<f64>)> {
        let mut original = Vec::with_capacity(ROUTED_EXPERTS);
        for &logit in logits {
            let x = logit as f64;
            let softplus = if x > 20.0 {
                x
            } else if x >= 0.0 {
                x + (-x).exp().ln_1p()
            } else {
                x.exp().ln_1p()
            };
            let score = softplus.sqrt();
            if !(score.is_finite() && score > 0.0) {
                return Err("oracle score not positive finite".into());
            }
            original.push(score);
        }
        let mut selected = Vec::new();
        for _ in 0..ACTIVATED_EXPERTS {
            let mut best_id = None;
            let mut best_sel = f64::NEG_INFINITY;
            for expert in 0..ROUTED_EXPERTS {
                if selected.contains(&(expert as u32)) {
                    continue;
                }
                let sel = original[expert] + bias[expert] as f64;
                if best_id.is_none()
                    || sel > best_sel
                    || (sel == best_sel && (expert as u32) < best_id.unwrap())
                {
                    best_sel = sel;
                    best_id = Some(expert as u32);
                }
            }
            selected.push(best_id.ok_or("oracle topk exhausted")?);
        }
        let mut weights = [0f64; ACTIVATED_EXPERTS];
        let mut sum = 0f64;
        for (i, &id) in selected.iter().enumerate() {
            weights[i] = original[id as usize];
            sum += weights[i];
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err("oracle weight sum invalid".into());
        }
        for w in &mut weights {
            *w = (*w / sum) * ROUTE_SCALE as f64;
        }
        let ids: [u32; ACTIVATED_EXPERTS] = selected.try_into().unwrap();
        Ok((ids, weights, original))
    }

    fn bytemuck_f32(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_f32(buf: &metal::Buffer, n: usize) -> ProbeResult<Vec<f32>> {
        let ptr = buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, n * 4) };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into()?));
        }
        Ok(out)
    }

    fn read_u32(buf: &metal::Buffer, n: usize) -> ProbeResult<Vec<u32>> {
        let ptr = buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, n * 4) };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into()?));
        }
        Ok(out)
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(index, std::mem::size_of::<u32>() as u64, value as *const u32 as *const _);
    }

    fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
        encoder.set_bytes(index, std::mem::size_of::<f32>() as u64, value as *const f32 as *const _);
    }

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut out = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => {
                    artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?));
                }
                "--out" => {
                    out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        Ok(Args {
            artifact: artifact.ok_or("--artifact is required")?,
            out: out.ok_or("--out is required")?,
        })
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
