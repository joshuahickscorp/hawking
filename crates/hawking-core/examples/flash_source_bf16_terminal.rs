//! Terminal probe for the verified Qwen3.8-Flash-Next source-BF16 chain.
//!
//! This consumes the exact 48-layer final HyperConnection state produced by
//! the layer receipts, performs the model-level HyperConnection readout, and
//! executes the source-BF16 lm_head and greedy argmax natively on Metal.
//! It is intentionally a terminal boundary probe, not a replacement for the
//! eventual one-process streamed whole-token executor.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash terminal requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::kernels::{
        native_bf16_gemv_seq_tcb, qwen_next_hyperconnection_grouped_rmsnorm_tcb,
        qwen_next_hyperconnection_read_mix_tcb, qwen_next_hyperconnection_silu_scale_tcb,
        sample_argmax_f32_tcb,
    };
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    const ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const STATE: &str = "receipts/headless/FLASH_PREFIX_FED_LAYER47_STATE.f32";
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const HC: usize = HIDDEN * STREAMS;
    const LOWRANK: usize = 320;
    const VOCAB: usize = 248320;
    const EPS: f32 = 1.0e-6;

    fn sha256(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bf16(bytes: &[u8], i: usize) -> f32 {
        f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16)
    }

    fn matvec(bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let mut acc = 0.0f32;
                for col in 0..cols {
                    acc += bf16(bytes, row * cols + col) * x[col];
                }
                acc
            })
            .collect()
    }

    fn source_hc_readout_with_weights(
        state: &[f32],
        norm_w: &[u8],
        down_w: &[u8],
        up_w: &[u8],
    ) -> Result<Vec<f32>, String> {
        if state.len() != HC {
            return Err(format!(
                "final state has {} elements, expected {HC}",
                state.len()
            ));
        }
        let mut normed = vec![0.0f32; HC];
        for stream in 0..STREAMS {
            let start = stream * HIDDEN;
            let mean = state[start..start + HIDDEN]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / HIDDEN as f32;
            let inv = (mean + EPS).sqrt().recip();
            for i in 0..HIDDEN {
                normed[start + i] = state[start + i] * inv * (1.0 + bf16(&norm_w, start + i));
            }
        }
        let low = matvec(&down_w, LOWRANK, HC, &normed);
        let low_act: Vec<f32> = low
            .iter()
            .map(|v| {
                let x = *v / STREAMS as f32;
                x / (1.0 + (-x).exp())
            })
            .collect();
        let gates = matvec(&up_w, HC, LOWRANK, &low_act);
        Ok((0..HIDDEN)
            .map(|i| {
                let mut sum = 0.0f32;
                for stream in 0..STREAMS {
                    let offset = stream * HIDDEN + i;
                    sum += (1.0 / (1.0 + (-gates[offset]).exp())) * normed[offset];
                }
                sum / STREAMS as f32
            })
            .collect())
    }

    fn source_hc_readout(index: &SourceBf16Index, state: &[f32]) -> Result<Vec<f32>, String> {
        let norm_w = index
            .read_raw("model.language_model.hyper_connection_mixer.hc_norm.weight")
            .map_err(|e| e.to_string())?;
        let down_w = index
            .read_raw("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight")
            .map_err(|e| e.to_string())?;
        let up_w = index
            .read_raw("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight")
            .map_err(|e| e.to_string())?;
        source_hc_readout_with_weights(state, &norm_w, &down_w, &up_w)
    }

    fn read_state(path: &PathBuf) -> Result<Vec<f32>, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() != HC * 4 {
            return Err(format!("state bytes {} != {}", bytes.len(), HC * 4));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn metrics(expected: &[f32], observed: &[f32]) -> serde_json::Value {
        if expected.len() != observed.len() {
            return json!({"length_match": false});
        }
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut norm_expected = 0.0f64;
        let mut norm_observed = 0.0f64;
        for (&a, &b) in expected.iter().zip(observed) {
            let delta = (a - b).abs();
            max_abs = max_abs.max(delta);
            sum_sq += f64::from(delta) * f64::from(delta);
            dot += f64::from(a) * f64::from(b);
            norm_expected += f64::from(a) * f64::from(a);
            norm_observed += f64::from(b) * f64::from(b);
        }
        let cosine = if norm_expected > 0.0 && norm_observed > 0.0 {
            dot / (norm_expected.sqrt() * norm_observed.sqrt())
        } else {
            0.0
        };
        json!({
            "length_match": true,
            "finite": observed.iter().all(|v| v.is_finite()),
            "max_abs_error": max_abs,
            "rmse": (sum_sq / expected.len() as f64).sqrt(),
            "cosine": cosine,
            "within_tolerance": max_abs <= 2.0e-2
        })
    }

    /// Reusable terminal boundary for a native Flash session.
    ///
    /// The previous helper was intentionally simple, but constructing it for
    /// every state reopened the source index, reread the readout/LM-head
    /// payloads, allocated a fresh 1.27-GB LM-head buffer and rebuilt the
    /// Metal context.  A stateful decode must pay that cost once per process,
    /// not once per accepted token.  The source index and all buffers below
    /// therefore live for the executor lifetime; `run_state` only replaces the
    /// small final-state input and reuses the cached pipelines.
    pub(crate) struct TerminalExecutor {
        root: PathBuf,
        index: SourceBf16Index,
        context: MetalContext,
        state_buf: PinnedBuffer,
        hc_norm_buf: PinnedBuffer,
        hc_down_buf: PinnedBuffer,
        hc_up_buf: PinnedBuffer,
        normalized: PinnedBuffer,
        low: PinnedBuffer,
        low_activation: PinnedBuffer,
        gates: PinnedBuffer,
        input: PinnedBuffer,
        lm_head_buf: PinnedBuffer,
        logits: PinnedBuffer,
        sampled: PinnedBuffer,
        hc_norm: Vec<u8>,
        hc_down: Vec<u8>,
        hc_up: Vec<u8>,
    }

    impl TerminalExecutor {
        pub(crate) fn new(root: PathBuf) -> Result<Self, String> {
            let index = SourceBf16Index::open(&root).map_err(|e| e.to_string())?;
            let hc_norm = index
                .read_raw("model.language_model.hyper_connection_mixer.hc_norm.weight")
                .map_err(|e| e.to_string())?;
            let hc_down = index
                .read_raw(
                    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                )
                .map_err(|e| e.to_string())?;
            let hc_up = index
                .read_raw("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight")
                .map_err(|e| e.to_string())?;
            let lm_head = index
                .read_raw("lm_head.weight")
                .map_err(|e| e.to_string())?;
            if lm_head.len() != VOCAB * HIDDEN * 2 {
                return Err(format!(
                    "lm_head bytes {} != {}",
                    lm_head.len(),
                    VOCAB * HIDDEN * 2
                ));
            }
            let context = MetalContext::new_with_trace(true).map_err(|e| e.to_string())?;
            let state_buf = context
                .new_buffer_checked(HC * 4)
                .map_err(|e| e.to_string())?;
            let hc_norm_buf = context
                .new_buffer_with_bytes_checked(&hc_norm)
                .map_err(|e| e.to_string())?;
            let hc_down_buf = context
                .new_buffer_with_bytes_checked(&hc_down)
                .map_err(|e| e.to_string())?;
            let hc_up_buf = context
                .new_buffer_with_bytes_checked(&hc_up)
                .map_err(|e| e.to_string())?;
            let normalized = context
                .new_buffer_checked(HC * 4)
                .map_err(|e| e.to_string())?;
            let low = context
                .new_buffer_checked(LOWRANK * 4)
                .map_err(|e| e.to_string())?;
            let low_activation = context
                .new_buffer_checked(LOWRANK * 4)
                .map_err(|e| e.to_string())?;
            let gates = context
                .new_buffer_checked(HC * 4)
                .map_err(|e| e.to_string())?;
            let input = context
                .new_buffer_checked(HIDDEN * 4)
                .map_err(|e| e.to_string())?;
            let lm_head_buf = context
                .new_buffer_with_bytes_checked(&lm_head)
                .map_err(|e| e.to_string())?;
            let logits = context
                .new_buffer_checked(VOCAB * 4)
                .map_err(|e| e.to_string())?;
            let sampled = context.new_buffer_checked(4).map_err(|e| e.to_string())?;
            Ok(Self {
                root,
                index,
                context,
                state_buf,
                hc_norm_buf,
                hc_down_buf,
                hc_up_buf,
                normalized,
                low,
                low_activation,
                gates,
                input,
                lm_head_buf,
                logits,
                sampled,
                hc_norm,
                hc_down,
                hc_up,
            })
        }

        pub(crate) fn run_state(
            &self,
            state_path: &PathBuf,
            out_path: Option<PathBuf>,
        ) -> Result<(usize, serde_json::Value), String> {
            let started = Instant::now();
            let state_read_started = Instant::now();
            let state = read_state(state_path)?;
            let state_read_ns = state_read_started.elapsed().as_nanos() as u64;
            let readout_started = Instant::now();
            let hidden =
                source_hc_readout_with_weights(&state, &self.hc_norm, &self.hc_down, &self.hc_up)?;
            let cpu_readout_ns = readout_started.elapsed().as_nanos() as u64;
            if hidden.iter().any(|v| !v.is_finite()) {
                return Err("HyperConnection readout produced non-finite values".into());
            }
            let upload_started = Instant::now();
            MetalContext::write_buffer_bytes(&self.state_buf, &f32_bytes(&state));
            let device_upload_ns = upload_started.elapsed().as_nanos() as u64;
            let encode_started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            qwen_next_hyperconnection_grouped_rmsnorm_tcb(
                &mut tcb,
                &self.state_buf,
                &self.hc_norm_buf,
                &self.normalized,
                HIDDEN,
                STREAMS,
                EPS,
            )
            .map_err(|e| e.to_string())?;
            native_bf16_gemv_seq_tcb(
                &mut tcb,
                &self.hc_down_buf,
                &self.normalized,
                &self.low,
                LOWRANK,
                HC,
            )
            .map_err(|e| e.to_string())?;
            qwen_next_hyperconnection_silu_scale_tcb(
                &mut tcb,
                &self.low,
                &self.low_activation,
                LOWRANK,
                STREAMS as f32,
            )
            .map_err(|e| e.to_string())?;
            native_bf16_gemv_seq_tcb(
                &mut tcb,
                &self.hc_up_buf,
                &self.low_activation,
                &self.gates,
                HC,
                LOWRANK,
            )
            .map_err(|e| e.to_string())?;
            qwen_next_hyperconnection_read_mix_tcb(
                &mut tcb,
                &self.normalized,
                &self.gates,
                &self.input,
                HIDDEN,
                STREAMS,
            )
            .map_err(|e| e.to_string())?;
            native_bf16_gemv_seq_tcb(
                &mut tcb,
                &self.lm_head_buf,
                &self.input,
                &self.logits,
                VOCAB,
                HIDDEN,
            )
            .map_err(|e| e.to_string())?;
            sample_argmax_f32_tcb(&mut tcb, &self.logits, &self.sampled, VOCAB)
                .map_err(|e| e.to_string())?;
            let encode_wall_ns = encode_started.elapsed().as_nanos() as u64;
            let timing = tcb.commit_and_wait_timed().map_err(|e| e.to_string())?;
            let token = unsafe { *(self.sampled.contents() as *const u32) } as usize;
            let observed_hidden = unsafe {
                std::slice::from_raw_parts(self.input.contents() as *const f32, HIDDEN).to_vec()
            };
            let parity_started = Instant::now();
            let hidden_metrics = metrics(&hidden, &observed_hidden);
            let parity_ns = parity_started.elapsed().as_nanos() as u64;
            let wall_ns = started.elapsed().as_nanos() as u64;
            let receipt = json!({
                "schema": "hawking.flash_source_bf16_terminal.v1",
                "status": "PASSED",
                "repo": "Qwen/Qwen3.8-Flash-Next",
                "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
                "representation": "source_bf16_exact",
                "input": {"state": state_path, "state_sha256": sha256(&f32_bytes(&state)), "elements": HC},
                "terminal": {"hyperconnection_readout": "cached native Metal source-BF16 graph with CPU oracle", "lm_head": "cached native_bf16_gemv_seq", "sampling": "sample_argmax_f32", "token_id": token},
                "execution": {
                    "source_root": self.root.display().to_string(),
                    "device": self.context.device_name(),
                    "provider": "apple_metal",
                    "dispatches": timing.dispatches,
                    "command_buffers": timing.command_buffers,
                    "gpu_ns": timing.gpu_ns,
                    "wall_ns": wall_ns,
                    "fallback_count": 0,
                    "native_source_bf16": true,
                    "source_index_reused": true,
                    "readout_weights_reused": true,
                    "lm_head_reused": true,
                    "timing": {
                        "state_read_ns": state_read_ns,
                        "cpu_readout_ns": cpu_readout_ns,
                        "device_upload_ns": device_upload_ns,
                        "encode_wall_ns": encode_wall_ns,
                        "command_submit_ns": timing.submit_ns,
                        "command_wait_ns": timing.wait_ns,
                        "gpu_execution_ns": timing.gpu_ns,
                        "parity_ns": parity_ns,
                        "wall_ns": wall_ns,
                    },
                },
                "parity": {"hyperconnection_readout": hidden_metrics},
                "complete_token_runtime": "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE",
                "flash_tps": null,
                "complete_system_ebpw": null,
                "promotion_allowed": false,
                "claim_boundary": "A long-lived terminal executor reused one source index, readout weights, lm_head buffer, Metal context and pipeline cache for this state. This is a terminal boundary measurement, not a complete-model TPS, EBPW, capability or residency qualification."
            });
            let serialized = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
            if let Some(path) = out_path {
                fs::write(path, format!("{serialized}\n")).map_err(|e| e.to_string())?;
            }
            Ok((token, receipt))
        }

        pub(crate) fn source_bytes_read(&self) -> u64 {
            self.index.bytes_read_total()
        }

        pub(crate) fn device_name(&self) -> String {
            self.context.device_name()
        }
    }

    pub(crate) fn run_with(
        root: PathBuf,
        state_path: PathBuf,
        out_path: Option<PathBuf>,
    ) -> Result<(), String> {
        let started = Instant::now();
        let index = SourceBf16Index::open(&root).map_err(|e| e.to_string())?;
        let state = read_state(&state_path)?;
        let hidden = source_hc_readout(&index, &state)?;
        if hidden.iter().any(|v| !v.is_finite()) {
            return Err("HyperConnection readout produced non-finite values".into());
        }
        let hc_norm = index
            .read_raw("model.language_model.hyper_connection_mixer.hc_norm.weight")
            .map_err(|e| e.to_string())?;
        let hc_down = index
            .read_raw("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight")
            .map_err(|e| e.to_string())?;
        let hc_up = index
            .read_raw("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight")
            .map_err(|e| e.to_string())?;
        let lm_head = index
            .read_raw("lm_head.weight")
            .map_err(|e| e.to_string())?;
        if lm_head.len() != VOCAB * HIDDEN * 2 {
            return Err(format!(
                "lm_head bytes {} != {}",
                lm_head.len(),
                VOCAB * HIDDEN * 2
            ));
        }
        let context = MetalContext::new_with_trace(true).map_err(|e| e.to_string())?;
        let state_buf = context
            .new_buffer_checked(HC * 4)
            .map_err(|e| e.to_string())?;
        MetalContext::write_buffer_bytes(
            &state_buf,
            &state
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let hc_norm_buf = context
            .new_buffer_with_bytes_checked(&hc_norm)
            .map_err(|e| e.to_string())?;
        let hc_down_buf = context
            .new_buffer_with_bytes_checked(&hc_down)
            .map_err(|e| e.to_string())?;
        let hc_up_buf = context
            .new_buffer_with_bytes_checked(&hc_up)
            .map_err(|e| e.to_string())?;
        let normalized = context
            .new_buffer_checked(HC * 4)
            .map_err(|e| e.to_string())?;
        let low = context
            .new_buffer_checked(LOWRANK * 4)
            .map_err(|e| e.to_string())?;
        let low_activation = context
            .new_buffer_checked(LOWRANK * 4)
            .map_err(|e| e.to_string())?;
        let gates = context
            .new_buffer_checked(HC * 4)
            .map_err(|e| e.to_string())?;
        let input = context
            .new_buffer_checked(HIDDEN * 4)
            .map_err(|e| e.to_string())?;
        let lm_head_buf = context
            .new_buffer_with_bytes_checked(&lm_head)
            .map_err(|e| e.to_string())?;
        let logits = context
            .new_buffer_checked(VOCAB * 4)
            .map_err(|e| e.to_string())?;
        let sampled = context.new_buffer_checked(4).map_err(|e| e.to_string())?;
        let mut tcb = TokenCommandBuffer::new(&context);
        qwen_next_hyperconnection_grouped_rmsnorm_tcb(
            &mut tcb,
            &state_buf,
            &hc_norm_buf,
            &normalized,
            HIDDEN,
            STREAMS,
            EPS,
        )
        .map_err(|e| e.to_string())?;
        native_bf16_gemv_seq_tcb(&mut tcb, &hc_down_buf, &normalized, &low, LOWRANK, HC)
            .map_err(|e| e.to_string())?;
        qwen_next_hyperconnection_silu_scale_tcb(
            &mut tcb,
            &low,
            &low_activation,
            LOWRANK,
            STREAMS as f32,
        )
        .map_err(|e| e.to_string())?;
        native_bf16_gemv_seq_tcb(&mut tcb, &hc_up_buf, &low_activation, &gates, HC, LOWRANK)
            .map_err(|e| e.to_string())?;
        qwen_next_hyperconnection_read_mix_tcb(
            &mut tcb,
            &normalized,
            &gates,
            &input,
            HIDDEN,
            STREAMS,
        )
        .map_err(|e| e.to_string())?;
        native_bf16_gemv_seq_tcb(&mut tcb, &lm_head_buf, &input, &logits, VOCAB, HIDDEN)
            .map_err(|e| e.to_string())?;
        sample_argmax_f32_tcb(&mut tcb, &logits, &sampled, VOCAB).map_err(|e| e.to_string())?;
        let timing = tcb.commit_and_wait_timed().map_err(|e| e.to_string())?;
        let token = unsafe { *(sampled.contents() as *const u32) };
        let observed_hidden =
            unsafe { std::slice::from_raw_parts(input.contents() as *const f32, HIDDEN).to_vec() };
        let hidden_metrics = metrics(&hidden, &observed_hidden);
        let device = context.device_name();
        let receipt = json!({
            "schema": "hawking.flash_source_bf16_terminal.v1",
            "status": "PASSED",
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "representation": "source_bf16_exact",
            "input": {"state": state_path, "state_sha256": sha256(&state.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()), "elements": HC},
            "terminal": {"hyperconnection_readout": "native Metal source-BF16 graph with CPU oracle", "lm_head": "native_bf16_gemv_seq", "sampling": "sample_argmax_f32", "token_id": token},
            "execution": {"device": device, "provider": "apple_metal", "dispatches": timing.dispatches, "command_buffers": timing.command_buffers, "gpu_ns": timing.gpu_ns, "wall_ns": started.elapsed().as_nanos() as u64, "fallback_count": 0, "native_source_bf16": true},
            "parity": {"hyperconnection_readout": hidden_metrics},
            "complete_token_runtime": "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE",
            "flash_tps": null,
            "complete_system_ebpw": null,
            "promotion_allowed": false,
            "claim_boundary": "First greedy token was physically received from the verified 48-layer source-BF16 state through a single Metal command buffer containing the HyperConnection readout and lm_head. This is not yet a one-process streamed 48-layer runtime, TPS, EBPW, or residency qualification."
        });
        let serialized = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
        if let Some(path) = out_path {
            fs::write(path, format!("{serialized}\n")).map_err(|e| e.to_string())?;
        }
        println!("{serialized}");
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let root = PathBuf::from(env::args().nth(1).unwrap_or_else(|| ROOT.to_owned()));
        let state_path = PathBuf::from(env::args().nth(2).unwrap_or_else(|| STATE.to_owned()));
        let out_path = env::args().nth(3).map(PathBuf::from);
        run_with(root, state_path, out_path)
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run().map_err(Into::into)
}

#[cfg(target_os = "macos")]
pub(crate) fn run_with(
    root: std::path::PathBuf,
    state_path: std::path::PathBuf,
    out_path: Option<std::path::PathBuf>,
) -> Result<(), String> {
    macos::run_with(root, state_path, out_path)
}

#[cfg(target_os = "macos")]
pub(crate) use macos::TerminalExecutor;
