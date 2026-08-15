//! Exact Qwen3-Next Gated DeltaNet cached-decode component probe.
//!
//! Checks the direct Metal recurrence against a CPU oracle at the official
//! Qwen3-Coder-Next state geometry. It is deliberately not a model/runtime
//! benchmark: no Qwen weight is loaded, no convolution/projection/MoE is run,
//! and no TPS gate can be inferred from this receipt.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::kernels::qwen_next_gated_delta_decode_single_tcb;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use metal::Buffer;
    use serde_json::json;
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    const HEADS: usize = 32;
    const KEY_DIM: usize = 128;
    const VALUE_DIM: usize = 128;

    fn parse_out() -> Result<PathBuf, Box<dyn Error>> {
        let mut values = env::args().skip(1);
        if values.next().as_deref() != Some("--out") {
            return Err(
                "usage: ascension_qwen_next_deltanet_metal_probe --out <receipt.json>".into(),
            );
        }
        Ok(PathBuf::from(values.next().ok_or("missing output path")?))
    }

    fn l2_normalized(seed: usize, scale: f32) -> Vec<f32> {
        let mut row = (0..KEY_DIM)
            .map(|index| (((seed * 97 + index * 31) % 4093) as f32 / 2048.0) - 1.0)
            .collect::<Vec<_>>();
        let norm = row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        for value in &mut row {
            *value = *value / norm * scale;
        }
        row
    }

    fn cpu_oracle(
        mut state: Vec<f32>,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        decay: &[f32],
        beta: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut output = vec![0.0f32; HEADS * VALUE_DIM];
        for head in 0..HEADS {
            let state_base = head * KEY_DIM * VALUE_DIM;
            let key_base = head * KEY_DIM;
            let value_base = head * VALUE_DIM;
            for vi in 0..VALUE_DIM {
                let mut kv_mem = 0.0;
                for ki in 0..KEY_DIM {
                    let index = state_base + ki * VALUE_DIM + vi;
                    state[index] *= decay[head];
                    kv_mem += state[index] * key[key_base + ki];
                }
                let delta = (value[value_base + vi] - kv_mem) * beta[head];
                for ki in 0..KEY_DIM {
                    let index = state_base + ki * VALUE_DIM + vi;
                    state[index] += key[key_base + ki] * delta;
                }
            }
            for vi in 0..VALUE_DIM {
                for ki in 0..KEY_DIM {
                    output[value_base + vi] +=
                        state[state_base + ki * VALUE_DIM + vi] * query[key_base + ki];
                }
            }
        }
        (state, output)
    }

    fn copy_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let out = parse_out()?;
        let mut query = Vec::with_capacity(HEADS * KEY_DIM);
        let mut key = Vec::with_capacity(HEADS * KEY_DIM);
        let mut value = Vec::with_capacity(HEADS * VALUE_DIM);
        let mut decay = Vec::with_capacity(HEADS);
        let mut beta = Vec::with_capacity(HEADS);
        for head in 0..HEADS {
            query.extend(l2_normalized(head, 1.0 / (KEY_DIM as f32).sqrt()));
            key.extend(l2_normalized(head + 47, 1.0));
            value.extend(
                (0..VALUE_DIM).map(|index| ((head * 17 + index * 11) % 127) as f32 / 127.0 - 0.5),
            );
            decay.push(0.65 + (head % 7) as f32 * 0.03);
            beta.push(0.20 + (head % 5) as f32 * 0.07);
        }
        let state = (0..HEADS * KEY_DIM * VALUE_DIM)
            .map(|index| ((index * 13 % 101) as f32 - 50.0) * 0.0005)
            .collect::<Vec<_>>();
        let (expected_state, expected_output) =
            cpu_oracle(state.clone(), &query, &key, &value, &decay, &beta);
        let metal = MetalContext::new()?;
        let state_buf = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&state))?;
        let query_buf = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&query))?;
        let key_buf = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&key))?;
        let value_buf = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&value))?;
        let decay_buf = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&decay))?;
        let beta_buf = metal.new_buffer_with_bytes_checked(bytemuck::cast_slice(&beta))?;
        let output_buf =
            metal.new_buffer_checked(HEADS * VALUE_DIM * std::mem::size_of::<f32>())?;
        let mut tcb = TokenCommandBuffer::new(&metal);
        qwen_next_gated_delta_decode_single_tcb(
            &mut tcb,
            &state_buf,
            &query_buf,
            &key_buf,
            &value_buf,
            &decay_buf,
            &beta_buf,
            &output_buf,
            HEADS,
            KEY_DIM,
            VALUE_DIM,
        )?;
        tcb.commit_and_wait()?;
        let observed_state = copy_f32(&state_buf, expected_state.len());
        let observed_output = copy_f32(&output_buf, expected_output.len());
        let max_state_error = expected_state
            .iter()
            .zip(&observed_state)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_output_error = expected_output
            .iter()
            .zip(&observed_output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if max_state_error > 2e-5 || max_output_error > 2e-5 {
            return Err(format!("Qwen-Next DeltaNet parity failed: state={max_state_error}, output={max_output_error}").into());
        }
        let report = json!({
            "schema": "hawking.ascension.qwen_next_deltanet_metal_component_probe.v1",
            "status": "PASS_DIRECT_METAL_QWEN_NEXT_GATED_DELTANET_RECURRENCE_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE",
            "device": metal.device_name(),
            "official_qwen_next_geometry": {"heads": HEADS, "key_head_dim": KEY_DIM, "value_head_dim": VALUE_DIM, "recurrent_state_elements_per_layer": HEADS * KEY_DIM * VALUE_DIM},
            "max_abs_state_error": max_state_error,
            "max_abs_output_error": max_output_error,
            "claim_boundary": {"component_uses_deterministic_oracle_inputs_not_model_weights": true, "projections_convolution_gated_norm_moe_and_decoder_not_run": true, "not_100_tps_or_tg3": true, "not_full_model_or_manager_qualification": true}
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
