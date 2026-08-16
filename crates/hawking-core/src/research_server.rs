//! Persistent local research server for Q80 and DSV4F dirty Tier-1 loops.
//!
//! A long-lived process keeps the Metal device/library/PSOs, artifact
//! manifests, and one admitted model resident so an experiment does not
//! re-pay process startup, shader compile, or model admission. Every timing
//! this module emits is labelled `DIRTY_TIER1`: good for keep/reject, never
//! for promotion to BASE_TRUE_TPS.
//!
//! Transport is stdin/stdout or a unix-domain socket. There is no TCP
//! listener and no auth surface.

use crate::gravity_deepseek_v4::{DeepSeekV4FullStreamReader, PINNED_REPOSITORY, PINNED_REVISION};
use crate::gravity_deepseek_v4_layer_source_anchors::verify_deepseek_v4_layer_source_anchors;
use crate::gravity_deepseek_v4_streamed_forward::{
    discover_sealed_dsv4f_artifact, peak_rss_bytes, prepare_sealed_admission_root,
    run_streamed_forward_admitted, SealedAdmissionRoot, StreamedForwardConfig,
};
use crate::gravity_deepseek_v4_streamed_native::{
    StreamedNativeSession, ORACLE_GREEDY_LOGIT, ORACLE_GREEDY_TOKEN_ID,
};
use crate::model::qwen80_complete_runtime::qwen80_assert_native_operator_composition_complete;
use crate::model::qwen80_uniform_q4_hybrid_decode::{
    discover_qwen80_uniform_q4_root, load_qwen80_tokenizer, qwen80_default_tokenizer_path,
    render_qwen80_source_user_chat, Qwen80UniformQ4HybridDecodeSession,
    Qwen80UniformQ4StreamingCatalog, QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Every timing this harness reports. Not BASE_TRUE_TPS.
pub const TIMING_CLASS: &str = "DIRTY_TIER1";
/// Known-good Q80 greedy continuation of prompt `"Hi"` (chat-templated).
pub const Q80_KNOWN_GOOD_HI_TOKENS: [u32; 3] = [9707, 0, 2585];
/// Known-good DSV4F BOS greedy token.
pub const DSV4F_KNOWN_GOOD_TOKEN: u32 = ORACLE_GREEDY_TOKEN_ID;
/// Known-good DSV4F BOS greedy logit.
pub const DSV4F_KNOWN_GOOD_LOGIT: f32 = ORACLE_GREEDY_LOGIT;

const DEFAULT_Q80_MAX_SEQ: usize = 64;
const DEFAULT_DSV4F_MAX_LAYER: usize = 42;

/// One resident process.
pub struct ResearchServer {
    loaded: Option<LoadedModel>,
    model_admit_count: u64,
    kernel_reload_count: u64,
    greedy_count: u64,
    last_prompt: Option<String>,
    last_raw: bool,
    started: Instant,
}

enum LoadedModel {
    Q80(Q80Resident),
    Dsv4f(Dsv4fResident),
    #[cfg(test)]
    Fixture(FixtureResident),
}

struct Q80Resident {
    session: Qwen80UniformQ4HybridDecodeSession,
    tokenizer: Tokenizer,
    artifact: PathBuf,
    load_secs: f64,
    shader_secs: f64,
    catalog_secs: f64,
}

struct Dsv4fResident {
    admission: SealedAdmissionRoot,
    reader: DeepSeekV4FullStreamReader,
    metal: Option<StreamedNativeSession>,
    use_metal: bool,
    max_layer: usize,
    artifact: PathBuf,
    load_secs: f64,
    shader_secs: f64,
    admit_secs: f64,
}

#[cfg(test)]
struct FixtureResident {
    name: String,
}

/// Parsed request. Line protocol or a single JSON object.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Load(LoadSpec),
    Greedy {
        prompt: String,
        n_tokens: usize,
        raw: bool,
    },
    Time {
        n_tokens: usize,
    },
    ReloadKernels,
    Status,
    Quit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoadSpec {
    Q80 {
        artifact: Option<PathBuf>,
        tokenizer: Option<PathBuf>,
        max_seq_len: usize,
    },
    Dsv4f {
        artifact: Option<PathBuf>,
        use_metal: bool,
        max_layer: usize,
    },
    #[cfg(test)]
    Fixture,
}

impl Default for ResearchServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ResearchServer {
    pub fn new() -> Self {
        crate::startup_timing::mark_process_start();
        Self {
            loaded: None,
            model_admit_count: 0,
            kernel_reload_count: 0,
            greedy_count: 0,
            last_prompt: None,
            last_raw: false,
            started: Instant::now(),
        }
    }

    pub fn model_admit_count(&self) -> u64 {
        self.model_admit_count
    }

    pub fn kernel_reload_count(&self) -> u64 {
        self.kernel_reload_count
    }

    pub fn greedy_count(&self) -> u64 {
        self.greedy_count
    }

    /// Handle one request line. JSON object or whitespace/quoted tokens.
    pub fn handle_line(&mut self, line: &str) -> Result<Value> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(error_value("empty command"));
        }
        let command = match parse_command(trimmed) {
            Ok(command) => command,
            Err(message) => return Ok(error_value(&message)),
        };
        match self.dispatch(command) {
            Ok(value) => Ok(value),
            Err(error) => Ok(error_value(&error.to_string())),
        }
    }

    fn dispatch(&mut self, command: Command) -> Result<Value> {
        match command {
            Command::Load(spec) => self.load(spec),
            Command::Greedy {
                prompt,
                n_tokens,
                raw,
            } => self.greedy(&prompt, n_tokens, raw),
            Command::Time { n_tokens } => self.time_tokens(n_tokens),
            Command::ReloadKernels => self.reload_kernels(),
            Command::Status => Ok(self.status_value()),
            Command::Quit => Ok(json!({
                "ok": true,
                "cmd": "quit",
                "timing_class": TIMING_CLASS,
                "uptime_secs": self.started.elapsed().as_secs_f64(),
                "quit": true,
            })),
        }
    }

    fn load(&mut self, spec: LoadSpec) -> Result<Value> {
        match spec {
            LoadSpec::Q80 {
                artifact,
                tokenizer,
                max_seq_len,
            } => self.load_q80(artifact, tokenizer, max_seq_len),
            LoadSpec::Dsv4f {
                artifact,
                use_metal,
                max_layer,
            } => self.load_dsv4f(artifact, use_metal, max_layer),
            #[cfg(test)]
            LoadSpec::Fixture => self.load_fixture(),
        }
    }

    fn load_q80(
        &mut self,
        artifact: Option<PathBuf>,
        tokenizer: Option<PathBuf>,
        max_seq_len: usize,
    ) -> Result<Value> {
        qwen80_assert_native_operator_composition_complete().map_err(|error| {
            Error::Model(format!(
                "q80 hybrid operator composition incomplete: {error}"
            ))
        })?;
        let root = artifact
            .or_else(discover_qwen80_uniform_q4_root)
            .ok_or_else(|| {
                Error::Model("q80 uniform-q4 artifact root not found; pass load q80 <path>".into())
            })?;
        let tokenizer_path = tokenizer.unwrap_or_else(discover_qwen80_tokenizer);
        let wall = Instant::now();

        let catalog_started = Instant::now();
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root)?;
        let catalog_secs = catalog_started.elapsed().as_secs_f64();
        let tensor_count = catalog.tensor_count();
        let manifest_seal = catalog.manifest_seal_sha256.clone();
        let bpw = catalog.complete_physical_bpw;

        let tokenizer = load_qwen80_tokenizer(&tokenizer_path)?;

        let shader_started = Instant::now();
        let session = Qwen80UniformQ4HybridDecodeSession::new(catalog, max_seq_len)?;
        let shader_secs = shader_started.elapsed().as_secs_f64();
        let metal_error = session.metal_error.clone();
        let load_secs = wall.elapsed().as_secs_f64();

        self.loaded = Some(LoadedModel::Q80(Q80Resident {
            session,
            tokenizer,
            artifact: root.clone(),
            load_secs,
            shader_secs,
            catalog_secs,
        }));
        self.model_admit_count = self.model_admit_count.saturating_add(1);

        Ok(json!({
            "ok": true,
            "cmd": "load",
            "timing_class": TIMING_CLASS,
            "model": "q80",
            "artifact": root,
            "tensor_count": tensor_count,
            "manifest_seal_sha256": manifest_seal,
            "complete_physical_bpw": bpw,
            "claim": QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
            "max_seq_len": max_seq_len,
            "model_admit_count": self.model_admit_count,
            "catalog_secs": catalog_secs,
            "shader_or_session_secs": shader_secs,
            "load_secs": load_secs,
            "peak_rss_bytes": peak_rss_bytes(),
            "metal_error": metal_error,
            "note": "DIRTY_TIER1 cold start includes catalog parse and Metal library compile",
        }))
    }

    fn load_dsv4f(
        &mut self,
        artifact: Option<PathBuf>,
        use_metal: bool,
        max_layer: usize,
    ) -> Result<Value> {
        let root = artifact
            .or_else(discover_sealed_dsv4f_artifact)
            .ok_or_else(|| {
                Error::Gravity("dsv4f sealed artifact not found; pass load dsv4f <path>".into())
            })?;
        let wall = Instant::now();
        let admit_started = Instant::now();
        let admission = prepare_sealed_admission_root(&root)?;
        let reader = DeepSeekV4FullStreamReader::admit(&admission.path)?;
        let anchors = verify_deepseek_v4_layer_source_anchors(&reader)?;
        if anchors.identity().repository != PINNED_REPOSITORY
            || anchors.identity().revision != PINNED_REVISION
        {
            return Err(Error::Gravity(
                "dsv4f load refused a reader whose source identity is not pinned".into(),
            ));
        }
        let admit_secs = admit_started.elapsed().as_secs_f64();
        let view = admission.view.clone();
        let tensor_count = reader.tensor_count();
        let manifest_seal = reader.manifest_seal_sha256().to_owned();

        let shader_started = Instant::now();
        let metal = if use_metal {
            Some(StreamedNativeSession::new()?)
        } else {
            None
        };
        let shader_secs = shader_started.elapsed().as_secs_f64();
        let load_secs = wall.elapsed().as_secs_f64();

        self.loaded = Some(LoadedModel::Dsv4f(Dsv4fResident {
            admission,
            reader,
            metal,
            use_metal,
            max_layer,
            artifact: root.clone(),
            load_secs,
            shader_secs,
            admit_secs,
        }));
        self.model_admit_count = self.model_admit_count.saturating_add(1);

        Ok(json!({
            "ok": true,
            "cmd": "load",
            "timing_class": TIMING_CLASS,
            "model": "dsv4f",
            "artifact": root,
            "admission_view": view,
            "tensor_count": tensor_count,
            "manifest_seal_sha256": manifest_seal,
            "use_metal": use_metal,
            "max_layer": max_layer,
            "model_admit_count": self.model_admit_count,
            "admit_secs": admit_secs,
            "shader_secs": shader_secs,
            "load_secs": load_secs,
            "peak_rss_bytes": peak_rss_bytes(),
            "note": "DIRTY_TIER1 cold start includes sealed-stream admit and optional Metal compile",
        }))
    }

    #[cfg(test)]
    fn load_fixture(&mut self) -> Result<Value> {
        self.loaded = Some(LoadedModel::Fixture(FixtureResident {
            name: "fixture".to_owned(),
        }));
        self.model_admit_count = self.model_admit_count.saturating_add(1);
        Ok(json!({
            "ok": true,
            "cmd": "load",
            "timing_class": TIMING_CLASS,
            "model": "fixture",
            "model_admit_count": self.model_admit_count,
            "note": "test-only fixture; not a real model admit",
        }))
    }

    fn greedy(&mut self, prompt: &str, n_tokens: usize, raw: bool) -> Result<Value> {
        if n_tokens == 0 {
            return Err(Error::Model("n_tokens must be positive".into()));
        }
        let admit_before = self.model_admit_count;
        let outcome = match self.loaded.as_mut() {
            Some(LoadedModel::Q80(resident)) => greedy_q80(resident, prompt, n_tokens, raw)?,
            Some(LoadedModel::Dsv4f(resident)) => greedy_dsv4f(resident, n_tokens)?,
            #[cfg(test)]
            Some(LoadedModel::Fixture(_)) => greedy_fixture(prompt, n_tokens)?,
            None => {
                return Err(Error::Model(
                    "no model loaded; run load q80|dsv4f first".into(),
                ))
            }
        };
        if self.model_admit_count != admit_before {
            return Err(Error::Model(
                "greedy re-admitted the model; research server residency is broken".into(),
            ));
        }
        self.greedy_count = self.greedy_count.saturating_add(1);
        self.last_prompt = Some(prompt.to_owned());
        self.last_raw = raw;

        let mut value = outcome;
        value["ok"] = json!(true);
        value["cmd"] = json!("greedy");
        value["timing_class"] = json!(TIMING_CLASS);
        value["model_admit_count"] = json!(self.model_admit_count);
        value["greedy_count"] = json!(self.greedy_count);
        value["peak_rss_bytes"] = json!(peak_rss_bytes());
        value["claim"] = json!("DIRTY_TIER1_NOT_BASE_TRUE_TPS");
        Ok(value)
    }

    fn time_tokens(&mut self, n_tokens: usize) -> Result<Value> {
        if n_tokens == 0 {
            return Err(Error::Model("n_tokens must be positive".into()));
        }
        let prompt = self.last_prompt.clone().unwrap_or_else(|| "Hi".to_owned());
        let raw = self.last_raw;
        let admit_before = self.model_admit_count;
        let value = match self.loaded.as_mut() {
            Some(LoadedModel::Q80(resident)) => time_q80(resident, &prompt, n_tokens, raw)?,
            Some(LoadedModel::Dsv4f(resident)) => greedy_dsv4f(resident, n_tokens)?,
            #[cfg(test)]
            Some(LoadedModel::Fixture(_)) => greedy_fixture(&prompt, n_tokens)?,
            None => {
                return Err(Error::Model(
                    "no model loaded; run load q80|dsv4f first".into(),
                ))
            }
        };
        if self.model_admit_count != admit_before {
            return Err(Error::Model(
                "time re-admitted the model; research server residency is broken".into(),
            ));
        }
        self.greedy_count = self.greedy_count.saturating_add(1);
        let mut out = value;
        out["ok"] = json!(true);
        out["cmd"] = json!("time");
        out["timing_class"] = json!(TIMING_CLASS);
        out["model_admit_count"] = json!(self.model_admit_count);
        out["greedy_count"] = json!(self.greedy_count);
        out["peak_rss_bytes"] = json!(peak_rss_bytes());
        out["claim"] = json!("DIRTY_TIER1_NOT_BASE_TRUE_TPS");
        Ok(out)
    }

    fn reload_kernels(&mut self) -> Result<Value> {
        let admit_before = self.model_admit_count;
        let started = Instant::now();
        let detail = match self.loaded.as_mut() {
            Some(LoadedModel::Q80(resident)) => {
                let report = resident.session.reload_kernels()?;
                json!({
                    "model": "q80",
                    "rebuilt_metal": report.rebuilt_metal,
                    "catalog_reopened": report.catalog_reopened,
                    "metal_error": report.metal_error,
                    "reload_inner_secs": report.elapsed_secs,
                    "skipped": [
                        "model_admit",
                        "catalog_open",
                        "manifest_parse",
                        "tokenizer_load",
                    ],
                    "rebuilt": ["metal_device_library", "pipeline_state_objects"],
                })
            }
            Some(LoadedModel::Dsv4f(resident)) => {
                let shader_started = Instant::now();
                if resident.use_metal {
                    resident.metal = Some(StreamedNativeSession::new()?);
                }
                json!({
                    "model": "dsv4f",
                    "rebuilt_metal": resident.metal.is_some(),
                    "catalog_reopened": false,
                    "reload_inner_secs": shader_started.elapsed().as_secs_f64(),
                    "skipped": [
                        "model_admit",
                        "sealed_stream_admit",
                        "admission_view",
                        "manifest_parse",
                    ],
                    "rebuilt": ["metal_device_library", "pipeline_state_objects"],
                })
            }
            #[cfg(test)]
            Some(LoadedModel::Fixture(_)) => json!({
                "model": "fixture",
                "rebuilt_metal": false,
                "catalog_reopened": false,
                "skipped": ["model_admit"],
                "rebuilt": [],
            }),
            None => return Err(Error::Model("no model loaded; nothing to reload".into())),
        };
        if self.model_admit_count != admit_before {
            return Err(Error::Model(
                "reload-kernels re-admitted the model; residency is broken".into(),
            ));
        }
        self.kernel_reload_count = self.kernel_reload_count.saturating_add(1);
        Ok(json!({
            "ok": true,
            "cmd": "reload-kernels",
            "timing_class": TIMING_CLASS,
            "model_admit_count": self.model_admit_count,
            "kernel_reload_count": self.kernel_reload_count,
            "elapsed_secs": started.elapsed().as_secs_f64(),
            "peak_rss_bytes": peak_rss_bytes(),
            "detail": detail,
        }))
    }

    fn status_value(&self) -> Value {
        let (model, extra) = match &self.loaded {
            Some(LoadedModel::Q80(resident)) => (
                "q80",
                json!({
                    "artifact": resident.artifact,
                    "tensor_count": resident.session.catalog().tensor_count(),
                    "manifest_seal_sha256": resident.session.catalog().manifest_seal_sha256,
                    "metal_error": resident.session.metal_error,
                    "native": {
                        "q4_matvec_dispatches": resident.session.native.q4_matvec_dispatches,
                        "q4_embedding_dispatches": resident.session.native.q4_embedding_dispatches,
                        "q4_decode_vector_dispatches": resident.session.native.q4_decode_vector_dispatches,
                    },
                    "fallbacks": {
                        "total": resident.session.fallbacks.total(),
                        "host_q4_matvec": resident.session.fallbacks.host_q4_matvec,
                        "host_q4_embedding_gather": resident.session.fallbacks.host_q4_embedding_gather,
                        "host_q4_vector_decode": resident.session.fallbacks.host_q4_vector_decode,
                        "host_activation": resident.session.fallbacks.host_activation,
                        "host_expert_payload_bind": resident.session.fallbacks.host_expert_payload_bind,
                        "host_sample": resident.session.fallbacks.host_sample,
                    },
                    "cold_load_secs": resident.load_secs,
                    "catalog_secs": resident.catalog_secs,
                    "shader_or_session_secs": resident.shader_secs,
                    "claim": QWEN80_UNIFORM_Q4_VELOCITY_NOT_BASE_TRUE_TPS,
                }),
            ),
            Some(LoadedModel::Dsv4f(resident)) => (
                "dsv4f",
                json!({
                    "artifact": resident.artifact,
                    "admission_view": resident.admission.view,
                    "tensor_count": resident.reader.tensor_count(),
                    "manifest_seal_sha256": resident.reader.manifest_seal_sha256(),
                    "use_metal": resident.use_metal,
                    "max_layer": resident.max_layer,
                    "metal_dispatches": resident.metal.as_ref().map(|s| s.metal_dispatches()).unwrap_or(0),
                    "fallbacks": resident.metal.as_ref().map(|s| s.fallbacks()).unwrap_or(0),
                    "cold_load_secs": resident.load_secs,
                    "admit_secs": resident.admit_secs,
                    "shader_secs": resident.shader_secs,
                }),
            ),
            #[cfg(test)]
            Some(LoadedModel::Fixture(resident)) => ("fixture", json!({ "name": resident.name })),
            None => ("none", json!({})),
        };
        json!({
            "ok": true,
            "cmd": "status",
            "timing_class": TIMING_CLASS,
            "model": model,
            "model_admit_count": self.model_admit_count,
            "kernel_reload_count": self.kernel_reload_count,
            "greedy_count": self.greedy_count,
            "peak_rss_bytes": peak_rss_bytes(),
            "uptime_secs": self.started.elapsed().as_secs_f64(),
            "loaded": extra,
        })
    }
}

fn greedy_q80(
    resident: &mut Q80Resident,
    prompt: &str,
    n_tokens: usize,
    raw: bool,
) -> Result<Value> {
    let rendered = if raw {
        prompt.to_owned()
    } else {
        render_qwen80_source_user_chat(prompt)
    };
    let prompt_token_ids = resident.tokenizer.encode(&rendered, false)?;
    if prompt_token_ids.is_empty() {
        return Err(Error::Model(
            "prompt tokenization produced no tokens".into(),
        ));
    }
    if prompt_token_ids.len() + n_tokens > resident.session.state.max_seq_len {
        return Err(Error::Model(
            "prompt + n_tokens exceeds session max_seq_len".into(),
        ));
    }
    resident.session.reset_state();

    let mut generated = Vec::with_capacity(n_tokens);
    let mut per_token_secs = Vec::with_capacity(n_tokens);
    let wall = Instant::now();
    let prefill_started = Instant::now();
    let mut next = 0u32;
    for &token in &prompt_token_ids {
        next = resident.session.forward_token(token)?;
    }
    let prefill_secs = prefill_started.elapsed().as_secs_f64();
    generated.push(next);
    per_token_secs.push(prefill_secs);

    for _ in 1..n_tokens {
        if resident.tokenizer.is_eog(next) {
            break;
        }
        let step = Instant::now();
        next = resident.session.forward_token(next)?;
        per_token_secs.push(step.elapsed().as_secs_f64());
        generated.push(next);
    }
    let wall_secs = wall.elapsed().as_secs_f64();
    let generated_text = resident.tokenizer.decode(&generated, true)?;
    let decode_secs: f64 = per_token_secs.iter().skip(1).sum();
    let metal_q4_matvec_used = resident.session.native.q4_matvec_dispatches > 0;

    Ok(json!({
        "model": "q80",
        "prompt": rendered,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "per_token_secs": per_token_secs,
        "prefill_secs": prefill_secs,
        "decode_secs": decode_secs,
        "wall_secs": wall_secs,
        "metal_q4_matvec_used": metal_q4_matvec_used,
        "metal_error": resident.session.metal_error,
        "native": {
            "q4_matvec_dispatches": resident.session.native.q4_matvec_dispatches,
            "q4_embedding_dispatches": resident.session.native.q4_embedding_dispatches,
            "q4_decode_vector_dispatches": resident.session.native.q4_decode_vector_dispatches,
        },
        "fallbacks_total": resident.session.fallbacks.total(),
    }))
}

fn time_q80(resident: &mut Q80Resident, prompt: &str, n_tokens: usize, raw: bool) -> Result<Value> {
    // Prefill is paid once so the reported number is decode-only DIRTY_TIER1.
    let rendered = if raw {
        prompt.to_owned()
    } else {
        render_qwen80_source_user_chat(prompt)
    };
    let prompt_token_ids = resident.tokenizer.encode(&rendered, false)?;
    if prompt_token_ids.is_empty() {
        return Err(Error::Model(
            "prompt tokenization produced no tokens".into(),
        ));
    }
    if prompt_token_ids.len() + n_tokens > resident.session.state.max_seq_len {
        return Err(Error::Model(
            "prompt + n_tokens exceeds session max_seq_len".into(),
        ));
    }
    resident.session.reset_state();
    let prefill_started = Instant::now();
    let mut next = 0u32;
    for &token in &prompt_token_ids {
        next = resident.session.forward_token(token)?;
    }
    let prefill_secs = prefill_started.elapsed().as_secs_f64();

    let mut generated = Vec::with_capacity(n_tokens);
    let mut per_token_secs = Vec::with_capacity(n_tokens);
    generated.push(next);
    let decode_started = Instant::now();
    for _ in 1..n_tokens {
        if resident.tokenizer.is_eog(next) {
            break;
        }
        let step = Instant::now();
        next = resident.session.forward_token(next)?;
        per_token_secs.push(step.elapsed().as_secs_f64());
        generated.push(next);
    }
    let decode_secs = decode_started.elapsed().as_secs_f64();
    let generated_text = resident.tokenizer.decode(&generated, true)?;
    Ok(json!({
        "model": "q80",
        "prompt": rendered,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "prefill_secs": prefill_secs,
        "decode_secs": decode_secs,
        "per_decode_token_secs": per_token_secs,
        "steady_state_tokens": per_token_secs.len(),
        "note": "DIRTY_TIER1 decode-only; prefill is reported separately and is not the keep/reject clock",
    }))
}

fn greedy_dsv4f(resident: &mut Dsv4fResident, n_tokens: usize) -> Result<Value> {
    // Streamed DSV4F is BOS / position-0: one greedy token per forward.
    // `n_tokens` repeats that dirty BOS decode for keep/reject timing.
    let repeats = n_tokens.max(1);
    let mut tokens = Vec::with_capacity(repeats);
    let mut logits = Vec::with_capacity(repeats);
    let mut per_token_secs = Vec::with_capacity(repeats);
    let wall = Instant::now();
    for _ in 0..repeats {
        let mut config = StreamedForwardConfig::for_layers(resident.max_layer)?;
        config.use_metal = resident.use_metal;
        config.compute_final_head = true;
        let step = Instant::now();
        let report = run_streamed_forward_admitted(
            &resident.admission,
            &resident.reader,
            resident.metal.as_mut(),
            config,
        )?;
        per_token_secs.push(step.elapsed().as_secs_f64());
        if let Some(reason) = report.stop_reason {
            return Err(Error::Gravity(format!(
                "dsv4f streamed forward stopped: {reason}"
            )));
        }
        let greedy = report.greedy.ok_or_else(|| {
            Error::Gravity("dsv4f streamed forward produced no greedy token".into())
        })?;
        tokens.push(greedy.token_id);
        logits.push(greedy.logit);
    }
    Ok(json!({
        "model": "dsv4f",
        "generated_token_ids": tokens,
        "logits": logits,
        "per_token_secs": per_token_secs,
        "wall_secs": wall.elapsed().as_secs_f64(),
        "bos_only": true,
        "note": "DSV4F streamed path is BOS/position-0; n_tokens repeats that one-token decode",
        "use_metal": resident.use_metal,
        "admission_view": resident.admission.view,
    }))
}

#[cfg(test)]
fn greedy_fixture(prompt: &str, n_tokens: usize) -> Result<Value> {
    let rendered = render_qwen80_source_user_chat(prompt);
    let tokens: Vec<u32> = Q80_KNOWN_GOOD_HI_TOKENS
        .iter()
        .copied()
        .cycle()
        .take(n_tokens)
        .collect();
    Ok(json!({
        "model": "fixture",
        "prompt": rendered,
        "generated_token_ids": tokens,
        "per_token_secs": vec![0.0; tokens.len()],
        "wall_secs": 0.0,
    }))
}

fn discover_qwen80_tokenizer() -> PathBuf {
    let candidates = [
        qwen80_default_tokenizer_path(),
        PathBuf::from(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next/tokenizer.json",
        ),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(qwen80_default_tokenizer_path)
}

fn error_value(message: &str) -> Value {
    json!({
        "ok": false,
        "timing_class": TIMING_CLASS,
        "error": message,
    })
}

/// Parse one request line.
pub fn parse_command(line: &str) -> std::result::Result<Command, String> {
    let trimmed = line.trim();
    if trimmed.starts_with('{') {
        return parse_json_command(trimmed);
    }
    let tokens = tokenize_line(trimmed);
    if tokens.is_empty() {
        return Err("empty command".into());
    }
    match tokens[0].as_str() {
        "load" => parse_load(&tokens[1..]),
        "greedy" => parse_greedy(&tokens[1..]),
        "time" => {
            if tokens.len() != 2 {
                return Err("usage: time <n_tokens>".into());
            }
            let n_tokens = parse_n_tokens(&tokens[1])?;
            Ok(Command::Time { n_tokens })
        }
        "reload-kernels" | "reload_kernels" => Ok(Command::ReloadKernels),
        "status" => Ok(Command::Status),
        "quit" | "exit" => Ok(Command::Quit),
        other => Err(format!(
            "unknown command {other:?}; expected load|greedy|time|reload-kernels|status|quit"
        )),
    }
}

fn parse_json_command(line: &str) -> std::result::Result<Command, String> {
    let value: Value =
        serde_json::from_str(line).map_err(|error| format!("json command: {error}"))?;
    let cmd = value
        .get("cmd")
        .or_else(|| value.get("op"))
        .and_then(Value::as_str)
        .ok_or_else(|| "json command missing cmd".to_owned())?;
    match cmd {
        "load" => {
            let model = value
                .get("model")
                .and_then(Value::as_str)
                .ok_or_else(|| "json load missing model".to_owned())?;
            match model {
                "q80" | "qwen80" => Ok(Command::Load(LoadSpec::Q80 {
                    artifact: json_path(&value, "artifact"),
                    tokenizer: json_path(&value, "tokenizer"),
                    max_seq_len: json_usize(&value, "max_seq_len").unwrap_or(DEFAULT_Q80_MAX_SEQ),
                })),
                "dsv4f" | "deepseek-v4" | "deepseek_v4" => Ok(Command::Load(LoadSpec::Dsv4f {
                    artifact: json_path(&value, "artifact"),
                    use_metal: value
                        .get("metal")
                        .or_else(|| value.get("use_metal"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    max_layer: json_usize(&value, "max_layer").unwrap_or(DEFAULT_DSV4F_MAX_LAYER),
                })),
                #[cfg(test)]
                "fixture" => Ok(Command::Load(LoadSpec::Fixture)),
                other => Err(format!("unknown model {other:?}")),
            }
        }
        "greedy" => {
            let prompt = value
                .get("prompt")
                .and_then(Value::as_str)
                .ok_or_else(|| "json greedy missing prompt".to_owned())?
                .to_owned();
            let n_tokens = json_usize(&value, "n_tokens")
                .or_else(|| json_usize(&value, "tokens"))
                .ok_or_else(|| "json greedy missing n_tokens".to_owned())?;
            if n_tokens == 0 {
                return Err("n_tokens must be positive".into());
            }
            Ok(Command::Greedy {
                prompt,
                n_tokens,
                raw: value.get("raw").and_then(Value::as_bool).unwrap_or(false),
            })
        }
        "time" => {
            let n_tokens = json_usize(&value, "n_tokens")
                .or_else(|| json_usize(&value, "tokens"))
                .ok_or_else(|| "json time missing n_tokens".to_owned())?;
            if n_tokens == 0 {
                return Err("n_tokens must be positive".into());
            }
            Ok(Command::Time { n_tokens })
        }
        "reload-kernels" | "reload_kernels" => Ok(Command::ReloadKernels),
        "status" => Ok(Command::Status),
        "quit" | "exit" => Ok(Command::Quit),
        other => Err(format!("unknown json cmd {other:?}")),
    }
}

fn parse_load(args: &[String]) -> std::result::Result<Command, String> {
    if args.is_empty() {
        return Err(
            "usage: load q80|dsv4f [path] [--metal] [--max-seq-len N] [--max-layer N]".into(),
        );
    }
    let model = args[0].as_str();
    let rest = &args[1..];
    match model {
        "q80" | "qwen80" => {
            let mut artifact = None;
            let mut tokenizer = None;
            let mut max_seq_len = DEFAULT_Q80_MAX_SEQ;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--artifact" | "--artifact-root" => {
                        i += 1;
                        artifact =
                            Some(PathBuf::from(rest.get(i).ok_or("--artifact needs a path")?));
                    }
                    "--tokenizer" => {
                        i += 1;
                        tokenizer = Some(PathBuf::from(
                            rest.get(i).ok_or("--tokenizer needs a path")?,
                        ));
                    }
                    "--max-seq-len" => {
                        i += 1;
                        max_seq_len = rest
                            .get(i)
                            .ok_or("--max-seq-len needs N")?
                            .parse()
                            .map_err(|_| "--max-seq-len must be an integer")?;
                    }
                    flag if flag.starts_with('-') => {
                        return Err(format!("unknown load q80 flag {flag}"));
                    }
                    path => {
                        if artifact.is_some() {
                            return Err("load q80 got two artifact paths".into());
                        }
                        artifact = Some(PathBuf::from(path));
                    }
                }
                i += 1;
            }
            Ok(Command::Load(LoadSpec::Q80 {
                artifact,
                tokenizer,
                max_seq_len,
            }))
        }
        "dsv4f" | "deepseek-v4" | "deepseek_v4" => {
            let mut artifact = None;
            let mut use_metal = false;
            let mut max_layer = DEFAULT_DSV4F_MAX_LAYER;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--artifact" => {
                        i += 1;
                        artifact =
                            Some(PathBuf::from(rest.get(i).ok_or("--artifact needs a path")?));
                    }
                    "--metal" => use_metal = true,
                    "--max-layer" => {
                        i += 1;
                        max_layer = rest
                            .get(i)
                            .ok_or("--max-layer needs N")?
                            .parse()
                            .map_err(|_| "--max-layer must be an integer")?;
                    }
                    flag if flag.starts_with('-') => {
                        return Err(format!("unknown load dsv4f flag {flag}"));
                    }
                    path => {
                        if artifact.is_some() {
                            return Err("load dsv4f got two artifact paths".into());
                        }
                        artifact = Some(PathBuf::from(path));
                    }
                }
                i += 1;
            }
            Ok(Command::Load(LoadSpec::Dsv4f {
                artifact,
                use_metal,
                max_layer,
            }))
        }
        #[cfg(test)]
        "fixture" => Ok(Command::Load(LoadSpec::Fixture)),
        other => Err(format!("unknown model {other:?}; expected q80|dsv4f")),
    }
}

fn parse_greedy(args: &[String]) -> std::result::Result<Command, String> {
    let mut raw = false;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--raw" => raw = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown greedy flag {flag}"));
            }
            value => positional.push(value.to_owned()),
        }
        i += 1;
    }
    if positional.len() != 2 {
        return Err("usage: greedy <prompt> <n_tokens> [--raw]".into());
    }
    let n_tokens = parse_n_tokens(&positional[1])?;
    Ok(Command::Greedy {
        prompt: positional[0].clone(),
        n_tokens,
        raw,
    })
}

fn parse_n_tokens(text: &str) -> std::result::Result<usize, String> {
    let n: usize = text
        .parse()
        .map_err(|_| format!("n_tokens must be a positive integer, got {text:?}"))?;
    if n == 0 {
        return Err("n_tokens must be positive".into());
    }
    Ok(n)
}

fn json_path(value: &Value, key: &str) -> Option<PathBuf> {
    value.get(key).and_then(Value::as_str).map(PathBuf::from)
}

fn json_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(|item| {
        item.as_u64()
            .map(|n| n as usize)
            .or_else(|| item.as_str().and_then(|s| s.parse().ok()))
    })
}

fn tokenize_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '"' {
            chars.next();
            let mut token = String::new();
            while let Some(next) = chars.next() {
                if next == '\\' {
                    if let Some(escaped) = chars.next() {
                        token.push(escaped);
                    }
                } else if next == '"' {
                    break;
                } else {
                    token.push(next);
                }
            }
            tokens.push(token);
        } else {
            let mut token = String::new();
            while let Some(&next) = chars.peek() {
                if next.is_whitespace() {
                    break;
                }
                token.push(next);
                chars.next();
            }
            tokens.push(token);
        }
    }
    tokens
}

/// Serve line-oriented JSON on stdin/stdout until `quit` or EOF.
/// Returns `true` when the client sent `quit`.
pub fn serve_stdio<R, W>(server: &mut ResearchServer, input: R, mut output: W) -> Result<bool>
where
    R: BufRead,
    W: Write,
{
    for line in input.lines() {
        let line = line?;
        let reply = server.handle_line(&line)?;
        writeln!(output, "{reply}")?;
        output.flush()?;
        if reply.get("quit").and_then(Value::as_bool) == Some(true) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Bind a unix-domain socket and serve one client at a time. Local only.
pub fn serve_unix_socket(server: &mut ResearchServer, path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::net::UnixListener;
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let listener = UnixListener::bind(path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(path, perms)?;
        }
        eprintln!(
            "research-server DIRTY_TIER1 listening on unix:{}",
            path.display()
        );
        for incoming in listener.incoming() {
            let stream = incoming?;
            let mut reader = BufReader::new(stream.try_clone()?);
            let mut writer = stream;
            match serve_stdio(server, &mut reader, &mut writer) {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => eprintln!("research-server client error: {error}"),
            }
        }
        let _ = std::fs::remove_file(path);
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (server, path);
        Err(Error::Unimplemented(
            "unix-domain sockets are required for the research server socket transport",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_line_commands() {
        assert_eq!(
            parse_command("load q80").unwrap(),
            Command::Load(LoadSpec::Q80 {
                artifact: None,
                tokenizer: None,
                max_seq_len: DEFAULT_Q80_MAX_SEQ,
            })
        );
        assert_eq!(
            parse_command("greedy Hi 3").unwrap(),
            Command::Greedy {
                prompt: "Hi".into(),
                n_tokens: 3,
                raw: false,
            }
        );
        assert_eq!(
            parse_command(r#"greedy "Hello there" 4 --raw"#).unwrap(),
            Command::Greedy {
                prompt: "Hello there".into(),
                n_tokens: 4,
                raw: true,
            }
        );
        assert_eq!(
            parse_command("time 6").unwrap(),
            Command::Time { n_tokens: 6 }
        );
        assert_eq!(
            parse_command("reload-kernels").unwrap(),
            Command::ReloadKernels
        );
        assert_eq!(parse_command("status").unwrap(), Command::Status);
        assert_eq!(parse_command("quit").unwrap(), Command::Quit);
    }

    #[test]
    fn parse_json_commands() {
        let cmd = parse_command(r#"{"cmd":"greedy","prompt":"Hi","n_tokens":3}"#).unwrap();
        assert_eq!(
            cmd,
            Command::Greedy {
                prompt: "Hi".into(),
                n_tokens: 3,
                raw: false,
            }
        );
    }

    #[test]
    fn fixture_greedy_is_deterministic_and_does_not_readmit() {
        let mut server = ResearchServer::new();
        let load = server.handle_line("load fixture").unwrap();
        assert_eq!(load["ok"], true);
        assert_eq!(server.model_admit_count(), 1);

        let first = server.handle_line("greedy Hi 3").unwrap();
        let second = server.handle_line("greedy Hi 3").unwrap();
        assert_eq!(first["ok"], true);
        assert_eq!(second["ok"], true);
        assert_eq!(first["generated_token_ids"], second["generated_token_ids"]);
        assert_eq!(first["generated_token_ids"], json!([9707, 0, 2585]));
        assert_eq!(server.model_admit_count(), 1);
        assert_eq!(server.greedy_count(), 2);
        assert_eq!(first["timing_class"], TIMING_CLASS);
        assert_eq!(second["timing_class"], TIMING_CLASS);

        let reload = server.handle_line("reload-kernels").unwrap();
        assert_eq!(reload["ok"], true);
        assert_eq!(server.model_admit_count(), 1);
        assert_eq!(server.kernel_reload_count(), 1);
        assert_eq!(reload["detail"]["catalog_reopened"], false);

        let status = server.handle_line("status").unwrap();
        assert_eq!(status["timing_class"], TIMING_CLASS);
        assert_eq!(status["model_admit_count"], 1);
    }

    #[test]
    fn every_response_is_labelled_dirty_tier1() {
        let mut server = ResearchServer::new();
        for line in ["status", "greedy Hi 3", "not-a-command", ""] {
            let reply = server.handle_line(line).unwrap();
            assert_eq!(
                reply["timing_class"], TIMING_CLASS,
                "reply for {line:?} missing DIRTY_TIER1: {reply}"
            );
        }
    }
}
