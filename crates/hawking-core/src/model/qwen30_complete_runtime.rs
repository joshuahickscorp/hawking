//! Native Metal execution path for an admitted Qwen3-Coder-30B-A3B direct
//! complete-binary artifact.
//!
//! This module is intentionally narrower than the generic [`crate::Engine`]
//! dispatch: a complete-binary artifact is a catalog directory, not a GGUF or
//! a `.gravity` container, and it must be opened with protected admission
//! bindings.  The runtime therefore has an explicit constructor which:
//!
//! 1. re-admits the exact sealed artifact;
//! 2. loads the source-adjacent Qwen config/tokenizer by an exact path; and
//! 3. validates every one of the 18,867 expected Qwen30 tensor names/shapes.
//!
//! Once constructed, every model weight used by the token graph comes from an
//! immutable `HQ30G1B1` compact-payload snapshot validated during this
//! process's full admission and is consumed by native Metal. The raw BF16
//! source is never opened by this runtime. Host work is limited to protected
//! catalog I/O at process admission, Metal command scheduling, and reading the eight
//! *device-produced* route ids / one sampled id; it does not evaluate a
//! projection, norm, attention, router, expert, or sampler fallback.
//!
//! This is a baseline composition path, not a performance receipt.  It is
//! deliberately fail-closed on unsupported context sizes, non-greedy sampling,
//! and invalid device results.  A caller may call [`generate_greedy`] only
//! after it has elected to spend the considerable I/O/residency cost of the
//! first all-layer token; no constructor or preflight result implies that a
//! prompt has generated coherently or that TPS has been measured.

#[cfg(target_os = "macos")]
use super::qwen30_quality_repack_diagnostic::{
    Qwen30QualityRepackDiagnosticCatalog, Qwen30QualityRepackSparseGateUpDevicePair,
    Qwen30QualityRepackSparseGateUpDispatch,
};
use super::qwen_complete_binary::{
    admit_complete_binary_artifact, admit_qwen30_quality_repack_artifact, CompleteBinaryAdmission,
    CompleteBinaryArtifact, CompleteBinaryHeader, Qwen30QualityRepackAdmission,
    QwenCompleteBinaryModel,
};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
#[cfg(target_os = "macos")]
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use crate::kernels::{
    add_inplace_metal_tcb, mha_decode_f32_tcb, moe_topk_gate_tcb, rmsnorm_metal_buf_tcb,
    rope_qk_kv_append_vbias_f32_tcb,
};
#[cfg(target_os = "macos")]
use crate::metal::{DispatchSample, MetalContext, PinnedBuffer, TokenCommandBuffer};

// The generic kernel dispatcher keeps its scalar-binding helper private.  This
// runtime has a handful of bespoke kernels, so retain the same direct Metal
// ABI spelling locally rather than routing their scalar controls through a
// host-side decoded tensor or exposing a broad new public helper.
#[cfg(target_os = "macos")]
trait QwenCompleteSetScalar {
    fn qwen_set_u32(&self, index: u64, value: u32);
    fn qwen_set_f32(&self, index: u64, value: f32);
}

#[cfg(target_os = "macos")]
impl QwenCompleteSetScalar for ::metal::ComputeCommandEncoderRef {
    #[inline(always)]
    fn qwen_set_u32(&self, index: u64, value: u32) {
        self.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    #[inline(always)]
    fn qwen_set_f32(&self, index: u64, value: f32) {
        self.set_bytes(
            index,
            std::mem::size_of::<f32>() as u64,
            &value as *const f32 as *const _,
        );
    }
}

/// Current context ceiling of the materialized device GQA kernel.  This is a
/// native runtime support limit, not a claim that the source's 262K context
/// has been qualified.  The kernel uses one shared score vector per query
/// head; accepting a larger value would become an invalid Metal dispatch.
pub const QWEN30_COMPLETE_NATIVE_MAX_CONTEXT: usize = 4096;

const QWEN30_MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const QWEN30_REPOSITORY: &str = "Qwen/Qwen3-Coder-30B-A3B-Instruct";
const QWEN30_ARCHITECTURE: &str = "Qwen3MoeForCausalLM";
const QWEN30_MODEL_TYPE: &str = "qwen3_moe";
const QWEN30_LAYERS: usize = 48;
const QWEN30_COMPLETE_TENSOR_COUNT: usize = 18_867;
const QWEN30_HIDDEN: usize = 2048;
const QWEN30_HEADS: usize = 32;
const QWEN30_KV_HEADS: usize = 4;
const QWEN30_HEAD_DIM: usize = 128;
const QWEN30_EXPERTS: usize = 128;
const QWEN30_TOP_K: usize = 8;
const QWEN30_MOE_INTERMEDIATE: usize = 768;
const QWEN30_GROUP_SIZE: usize = 128;
const QWEN30_VOCAB: usize = 151_936;
const QWEN30_ROPE_THETA: f32 = 10_000_000.0;
const QWEN30_RMS_EPS: f32 = 1.0e-6;

fn model_error(message: impl Into<String>) -> Error {
    Error::Model(format!(
        "qwen30 complete native runtime: {}",
        message.into()
    ))
}

/// Opt-out for the serial multi-dispatch encoder on the Q30 token path.
///
/// Default **on**: each layer attention/router wave and each selected-expert
/// wave encode into one Metal compute encoder (serial dispatch type) instead
/// of opening/closing an encoder per kernel.  Set
/// `HAWKING_QWEN30_SERIAL_ENCODER=0` / `false` / `off` / `no` to restore the
/// historical per-dispatch encoder shape for A/B.  Under
/// `HAWKING_TCB_TRACE=gpu` / `gpu_prod` the TCB already ignores serial groups
/// so timestamp attribution is unchanged.
#[cfg(target_os = "macos")]
fn qwen30_serial_encoder_enabled() -> bool {
    match std::env::var("HAWKING_QWEN30_SERIAL_ENCODER") {
        Ok(raw) => {
            let trimmed = raw.trim();
            !(trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        }
        Err(_) => true,
    }
}

/// Receipt from [`Qwen30CompleteNativeRuntime::prewarm_static_decoded_vectors`].
/// Diagnostic / load-path only; not a TPS claim.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30StaticDecodePrewarmReport {
    pub catalog_vectors: usize,
    pub already_resident: usize,
    pub decoded_now: usize,
    pub dispatches: usize,
    pub command_buffers: usize,
    pub serial_encoder: bool,
}

fn usize_field(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| model_error(format!("config missing unsigned {field:?}")))
}

fn finite_f32_field(value: &Value, field: &str) -> Result<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
        .ok_or_else(|| model_error(format!("config missing finite numeric {field:?}")))
}

fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| model_error(format!("config missing non-empty string {field:?}")))
}

fn u32_checked(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| model_error(format!("{label} exceeds Metal uint ABI")))
}

fn bytes_for_f32(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| model_error(format!("{label} byte count overflows usize")))
}

/// The exact Qwen configuration required by the admitted 30B candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30CompleteRuntimeConfig {
    pub model_id: String,
    pub source_repository: String,
    pub source_revision: String,
    pub layers: usize,
    pub hidden: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate: usize,
    pub vocab_size: usize,
    pub rope_theta_bits: u32,
    pub rms_norm_eps_bits: u32,
    pub source_max_position_embeddings: usize,
}

impl Qwen30CompleteRuntimeConfig {
    /// Parse only the exact upstream Qwen30 geometry.  A close-looking Qwen
    /// config is refused before any artifact tensor can be addressed.
    pub fn from_source_config(
        document: &Value,
        source_repository: &str,
        source_revision: &str,
    ) -> Result<Self> {
        let architectures = document
            .get("architectures")
            .and_then(Value::as_array)
            .ok_or_else(|| model_error("config missing architectures array"))?;
        if !architectures
            .iter()
            .any(|value| value.as_str() == Some(QWEN30_ARCHITECTURE))
        {
            return Err(model_error(format!(
                "config architectures does not contain {QWEN30_ARCHITECTURE}"
            )));
        }
        if string_field(document, "model_type")? != QWEN30_MODEL_TYPE {
            return Err(model_error("config model_type is not qwen3_moe"));
        }
        if source_repository != QWEN30_REPOSITORY {
            return Err(model_error(format!(
                "artifact repository {source_repository:?} is not {QWEN30_REPOSITORY:?}"
            )));
        }
        if source_revision.is_empty() {
            return Err(model_error("artifact source revision is empty"));
        }
        let exact = [
            ("num_hidden_layers", QWEN30_LAYERS),
            ("hidden_size", QWEN30_HIDDEN),
            ("num_attention_heads", QWEN30_HEADS),
            ("num_key_value_heads", QWEN30_KV_HEADS),
            ("head_dim", QWEN30_HEAD_DIM),
            ("num_experts", QWEN30_EXPERTS),
            ("num_experts_per_tok", QWEN30_TOP_K),
            ("moe_intermediate_size", QWEN30_MOE_INTERMEDIATE),
            ("decoder_sparse_step", 1),
            ("vocab_size", QWEN30_VOCAB),
        ];
        for (field, expected) in exact {
            let observed = usize_field(document, field)?;
            if observed != expected {
                return Err(model_error(format!(
                    "config {field}={observed}, expected exact Qwen30 value {expected}"
                )));
            }
        }
        if string_field(document, "hidden_act")? != "silu" {
            return Err(model_error(
                "Qwen30 runtime requires the source SiLU activation",
            ));
        }
        if document.get("norm_topk_prob").and_then(Value::as_bool) != Some(true) {
            return Err(model_error(
                "Qwen30 runtime requires source norm_topk_prob=true for top-8 route weights",
            ));
        }
        if document.get("tie_word_embeddings").and_then(Value::as_bool) != Some(false) {
            return Err(model_error(
                "Qwen30 runtime requires untied embed/lm_head source tensors",
            ));
        }
        if document.get("attention_bias").and_then(Value::as_bool) != Some(false) {
            return Err(model_error(
                "Qwen30 runtime requires source attention_bias=false; projection biases would need explicit catalog support",
            ));
        }
        if document.get("rope_scaling") != Some(&Value::Null) {
            return Err(model_error(
                "Qwen30 runtime only admits the source's unscaled RoPE configuration",
            ));
        }
        if document.get("use_sliding_window").and_then(Value::as_bool) != Some(false) {
            return Err(model_error(
                "Qwen30 runtime requires source use_sliding_window=false for its full causal KV kernel",
            ));
        }
        if document
            .get("mlp_only_layers")
            .and_then(Value::as_array)
            .map(|layers| !layers.is_empty())
            .unwrap_or(true)
        {
            return Err(model_error(
                "Qwen30 runtime requires source mlp_only_layers=[] so all 48 layers include attention",
            ));
        }
        let rope_theta = finite_f32_field(document, "rope_theta")?;
        let rms_norm_eps = finite_f32_field(document, "rms_norm_eps")?;
        if rope_theta.to_bits() != QWEN30_ROPE_THETA.to_bits() {
            return Err(model_error(format!(
                "config rope_theta={rope_theta:?} differs from Qwen30's exact 10000000.0"
            )));
        }
        if rms_norm_eps.to_bits() != QWEN30_RMS_EPS.to_bits() {
            return Err(model_error(format!(
                "config rms_norm_eps={rms_norm_eps:?} differs from Qwen30's exact 1e-6"
            )));
        }
        let source_max_position_embeddings = usize_field(document, "max_position_embeddings")?;
        if source_max_position_embeddings == 0 {
            return Err(model_error(
                "config max_position_embeddings must be non-zero",
            ));
        }
        Ok(Self {
            model_id: QWEN30_MODEL_ID.to_owned(),
            source_repository: source_repository.to_owned(),
            source_revision: source_revision.to_owned(),
            layers: QWEN30_LAYERS,
            hidden: QWEN30_HIDDEN,
            attention_heads: QWEN30_HEADS,
            key_value_heads: QWEN30_KV_HEADS,
            head_dim: QWEN30_HEAD_DIM,
            experts: QWEN30_EXPERTS,
            experts_per_token: QWEN30_TOP_K,
            moe_intermediate: QWEN30_MOE_INTERMEDIATE,
            vocab_size: QWEN30_VOCAB,
            rope_theta_bits: rope_theta.to_bits(),
            rms_norm_eps_bits: rms_norm_eps.to_bits(),
            source_max_position_embeddings,
        })
    }

    pub fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    pub fn rms_norm_eps(&self) -> f32 {
        f32::from_bits(self.rms_norm_eps_bits)
    }

    pub fn q_dim(&self) -> usize {
        self.attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.key_value_heads * self.head_dim
    }
}

/// The packed projection geometry chosen for a bounded native execution.
///
/// `ScalarControl` is the admitted control path.  `SimdgroupCandidate` has a
/// separate direct-packed Metal-vs-CPU parity test and is deliberately opt-in
/// so a kernel experiment cannot silently alter an existing runtime receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen30PackedMatvecKernel {
    ScalarControl,
    SimdgroupCandidate,
}

impl Qwen30PackedMatvecKernel {
    pub fn receipt_name(self) -> &'static str {
        match self {
            Self::ScalarControl => "scalar_one_thread_per_row_control",
            Self::SimdgroupCandidate => "simdgroup_eight_rows_per_threadgroup_candidate",
        }
    }
}

/// The routed-expert gate/up/SwiGLU topology used inside a Qwen30 MoE layer.
///
/// This is intentionally independent of [`Qwen30PackedMatvecKernel`].  The
/// latter applies to every packed projection in the graph, while this choice
/// can replace only the three-dispatch gate/up/activation chain for a selected
/// routed expert.  Keeping the controls separate prevents a component trial
/// from silently changing attention, router, down-projection, or lm-head
/// arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen30GateUpSwiGluKernel {
    /// Current independent packed gate projection, packed up projection, and
    /// offset-aware SwiGLU control sequence.
    ThreeDispatchControl,
    /// One direct-packed gate/up reduction with in-kernel SwiGLU.  This mode
    /// is a candidate only and carries no implied numerical admission.
    FusedCandidate,
    /// Candidate output drives the native down projection while the exact
    /// three-dispatch control activation is also calculated and compared from
    /// device buffers after every selected routed expert.  This is a bounded
    /// diagnostic parity mode, never a production throughput path.
    FusedCandidateWithDeviceControlParity,
    /// Separate paired gate/up topology candidate that preserves the scalar
    /// control's non-FMA increasing-column recurrence under explicit Metal
    /// no-contract/no-reassociate pragmas. It is diagnostic-only and always
    /// retains device-control parity.
    PairedScalarOrderCandidateWithDeviceControlParity,
    /// The separately named no-parity execution path for a later production
    /// requalification. It invokes the exact same scalar-order shader as the
    /// parity candidate but omits the retained control activation and its
    /// post-command-buffer readback. This enum alone is not a promotion: a
    /// new executable binding, complete native chain, profile, and HCLI gates
    /// must all be earned before it may serve.
    PairedScalarOrderProductionNoParity,
}

impl Qwen30GateUpSwiGluKernel {
    pub fn receipt_name(self) -> &'static str {
        match self {
            Self::ThreeDispatchControl => "three_dispatch_direct_packed_gate_up_swiglu_control",
            Self::FusedCandidate => "fused_direct_packed_gate_up_swiglu_candidate",
            Self::FusedCandidateWithDeviceControlParity => {
                "fused_direct_packed_gate_up_swiglu_candidate_with_device_control_parity"
            }
            Self::PairedScalarOrderCandidateWithDeviceControlParity => {
                "paired_direct_packed_gate_up_swiglu_scalar_order_candidate_with_device_control_parity"
            }
            Self::PairedScalarOrderProductionNoParity => {
                "paired_direct_packed_gate_up_swiglu_scalar_order_production_no_parity"
            }
        }
    }

    fn requires_device_control_parity(self) -> bool {
        matches!(
            self,
            Self::FusedCandidateWithDeviceControlParity
                | Self::PairedScalarOrderCandidateWithDeviceControlParity
        )
    }
}

/// Runtime choices which do not alter the artifact or source architecture.
#[derive(Clone, Debug)]
pub struct Qwen30CompleteRuntimeOptions {
    /// Requested native GQA cache length.  It must fit both the artifact's
    /// source context maximum and the current materialized Metal attention
    /// kernel's 4096-position support ceiling.
    pub max_seq_len: usize,
    /// Collect Metal per-dispatch trace samples for a profiler invocation.
    pub trace_dispatch: bool,
    /// Packed projection kernel geometry.  It is recorded in all execution
    /// receipts and defaults to the scalar control until a candidate earns
    /// component and complete-token parity evidence.
    pub packed_matvec_kernel: Qwen30PackedMatvecKernel,
    /// Routed-expert gate/up/SwiGLU topology.  It remains the independent
    /// three-dispatch control unless a caller explicitly runs a bounded
    /// candidate/parity experiment.
    pub gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel,
}

impl Default for Qwen30CompleteRuntimeOptions {
    fn default() -> Self {
        Self {
            max_seq_len: 256,
            trace_dispatch: false,
            packed_matvec_kernel: Qwen30PackedMatvecKernel::ScalarControl,
            gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel::ThreeDispatchControl,
        }
    }
}

/// Immutable evidence returned by structural admission.  This is intentionally
/// not a generation, capability, HCLI, TPS, or TG receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30CompleteRuntimePreflight {
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub source_revision: String,
    pub config_path: PathBuf,
    pub config_sha256: String,
    pub tokenizer_path: PathBuf,
    pub tokenizer_sha256: String,
    /// The exact source `chat_template.jinja` that the limited native
    /// user-message renderer validates before it can format a prompt.
    pub source_user_chat_template_path: PathBuf,
    pub source_user_chat_template_sha256: String,
    /// `tokenizer_config.json` must carry byte-identical chat-template text;
    /// otherwise the runtime refuses rather than guessing a prompt format.
    pub tokenizer_config_path: PathBuf,
    pub tokenizer_config_sha256: String,
    /// Source tokenizer IDs directly addressable for text prompt/decode. The
    /// Qwen30 LM head has a larger reserved tail; that tail is never silently
    /// remapped by this native runtime.
    pub tokenizer_addressable_vocab: usize,
    pub tensor_count: usize,
    pub tensor_payload_bytes: u64,
    pub source_weight_elements: u64,
    pub direct_layout_group_size: usize,
    /// The admission-time whole-artifact scan retained an immutable direct
    /// payload snapshot for every catalog tensor. The preflight process drops
    /// the snapshots on exit; a native runtime re-admits and retains its own
    /// snapshots before executing a token.
    pub verified_payload_count: usize,
    pub complete_verified_payload_cache_at_admission: bool,
}

/// Exact binding for the source's simple one-user-message chat-template path.
///
/// Qwen's shipped Jinja template has rich system/tool branches.  The bounded
/// native runtime intentionally supports only the no-system/no-tools user
/// branch here; it validates the source's structural anchors and renders that
/// branch exactly instead of pretending it implements a generic Jinja engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30SourceUserChatTemplate {
    pub source_template_path: PathBuf,
    pub source_template_sha256: String,
    pub tokenizer_config_path: PathBuf,
    pub tokenizer_config_sha256: String,
}

/// A completed full native greedy token.  Its timing is diagnostic only.  No
/// rate is computed here so a caller cannot confuse an individual cold-token
/// baseline with a clean sustained BASE_TRUE_TPS measurement.
#[derive(Clone, Debug)]
pub struct Qwen30NativeGreedyStep {
    pub position: usize,
    pub token_id: u32,
    pub elapsed: Duration,
    pub command_buffers: usize,
    pub metal_dispatches: usize,
    pub host_route_id_readbacks: usize,
    pub host_sample_id_readbacks: usize,
    /// Present only in the bounded fused-candidate diagnostic mode.  The
    /// comparison reads native device-produced activations; it never opens or
    /// computes against raw BF16 source weights.
    pub gate_up_swiglu_device_control_parity: Option<Qwen30GateUpSwiGluDeviceParity>,
    /// Ordered, source-owned host-wall intervals for this exact token. They
    /// are emitted only by the diagnostic entrypoint and deliberately keep a
    /// combined command-buffer wait as an explicit topology/synchronization
    /// interval rather than assigning that wait to a guessed individual
    /// kernel. They are not a TPS measurement.
    pub host_stage_intervals: Vec<Qwen30HostStageInterval>,
}

/// One complete native token together with the device-selected MoE route at
/// each layer.  This is an opt-in diagnostic observation surface: the host
/// only copies the IDs that the normal token graph already reads in order to
/// bind resident expert slabs.  It never computes router scores or routes.
///
/// The type is intentionally separate from [`Qwen30NativeGreedyStep`] so the
/// serving/token-rate path does not accumulate per-layer diagnostic state.
#[derive(Clone, Debug)]
pub struct Qwen30NativeRouteCaptureStep {
    pub greedy: Qwen30NativeGreedyStep,
    pub selected_expert_ids_per_layer: Vec<[u32; QWEN30_TOP_K]>,
}

/// One non-overlapping host-wall interval from a complete native token.
///
/// The profiler consumes these offsets only when its timer origin is the same
/// `forward_token_greedy` invocation.  `command_graph_transition_gap` is an
/// intentional named bucket for serial host setup/submission/wait spans whose
/// several device operators execute in one command buffer and cannot be
/// honestly split into synthetic per-operator host times.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30HostStageInterval {
    pub bucket: String,
    pub label: String,
    pub start_us: u64,
    pub end_us: u64,
}

fn duration_us(value: Duration) -> u64 {
    u64::try_from(value.as_micros()).unwrap_or(u64::MAX)
}

/// Record a partition of the complete-token host wall without silently
/// assigning gaps to a semantic model operator.  The caller supplies exact
/// source-side timing boundaries; gaps are retained as the explicit command
/// graph/synchronization bucket so the ledger is both non-overlapping and
/// auditable.
struct Qwen30HostStageRecorder {
    started: Instant,
    enabled: bool,
    last_end_us: u64,
    intervals: Vec<Qwen30HostStageInterval>,
}

impl Qwen30HostStageRecorder {
    fn new(started: Instant, enabled: bool) -> Self {
        Self {
            started,
            enabled,
            last_end_us: 0,
            intervals: Vec::new(),
        }
    }

    fn elapsed_us(&self) -> u64 {
        duration_us(self.started.elapsed())
    }

    fn append_gap_until(&mut self, start_us: u64, next_label: &str) {
        if start_us > self.last_end_us {
            self.intervals.push(Qwen30HostStageInterval {
                bucket: "command_graph_transition_gap".to_string(),
                label: format!("serial host setup/scheduling gap before {next_label}"),
                start_us: self.last_end_us,
                end_us: start_us,
            });
            self.last_end_us = start_us;
        }
    }

    fn measure<T>(
        &mut self,
        bucket: &str,
        label: impl Into<String>,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.enabled {
            return operation();
        }
        let label = label.into();
        let start_us = self.elapsed_us();
        self.append_gap_until(start_us, &label);
        let result = operation();
        let end_us = self.elapsed_us().max(start_us);
        if end_us > start_us {
            self.intervals.push(Qwen30HostStageInterval {
                bucket: bucket.to_string(),
                label,
                start_us,
                end_us,
            });
        }
        self.last_end_us = self.last_end_us.max(end_us);
        result
    }

    fn finish(mut self, elapsed: Duration) -> Vec<Qwen30HostStageInterval> {
        if !self.enabled {
            return Vec::new();
        }
        let total_us = duration_us(elapsed);
        self.append_gap_until(total_us, "complete-token return");
        debug_assert!(self
            .intervals
            .windows(2)
            .all(|pair| pair[0].end_us <= pair[1].start_us));
        debug_assert!(self
            .intervals
            .last()
            .is_none_or(|interval| interval.end_us <= total_us));
        self.intervals
    }
}

/// Device-buffer comparison facts from one complete Qwen30 token in the
/// fused gate/up/SwiGLU diagnostic mode.  A value is returned only after all
/// 48 layers and their eight selected experts were compared within tolerance.
#[derive(Clone, Debug)]
pub struct Qwen30GateUpSwiGluDeviceParity {
    pub layers_compared: usize,
    pub routed_experts_compared: usize,
    pub activation_values_compared: usize,
    pub max_abs_error: f32,
    pub tolerance_max_abs: f32,
}

/// Actual native greedy-generation result.  The result says nothing about
/// coherence; that must be established by separate capability evidence.
#[derive(Clone, Debug)]
pub struct Qwen30NativeGeneration {
    pub prompt_token_ids: Vec<u32>,
    pub completion_token_ids: Vec<u32>,
    pub completion_text: String,
    pub ended_on_eog: bool,
    /// Full native forwards which consumed source-template prompt tokens.
    /// They are retained so a candidate parity invocation can account for the
    /// entire prompt path rather than only feedback forwards.
    pub prefill_steps: Vec<Qwen30NativeGreedyStep>,
    pub steps: Vec<Qwen30NativeGreedyStep>,
}

/// One native, layer-0-only router observation for an already-tokenized
/// prompt.  This is a deliberately partial diagnostic primitive: it executes
/// the direct-packed embedding plus layer-0 attention/post-attention norm and
/// router on Metal, then reads the eight device-selected route ids, their
/// normalized device weights, and the router input hidden vector.  It does
/// not execute an MLP, later layer, final norm, lm_head, sampler, or token
/// feedback loop.
///
/// The bounded HQ30GR2 quality branch changes only the L0/E0 gate/up weights,
/// which occur *after* this router decision.  Consequently this capture can
/// establish whether the changed expert is on an exact current input route
/// without trying to treat a partial observation as a full candidate runtime.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Qwen30Layer0RouterCapture {
    pub position: usize,
    pub input_token_id: u32,
    pub selected_expert_ids: [u32; QWEN30_TOP_K],
    pub normalized_route_weights: [f32; QWEN30_TOP_K],
    /// Exact device-produced L0 post-attention RMSNorm output fed to the
    /// layer-0 router.  Returning this shared-memory copy enables a separate
    /// source-bound CPU representation diagnostic; it is never used by this
    /// runtime for model math or fallback computation.
    pub router_input_hidden: Vec<f32>,
}

/// One layer's router observation from a full all-layer diagnostic forward.
/// Captured after the layer's post-attention RMSNorm + router top-k, before
/// the selected expert gate/up/down wave mutates residual state.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Qwen30LayerRouterCapture {
    pub layer: usize,
    pub selected_expert_ids: [u32; QWEN30_TOP_K],
    pub normalized_route_weights: [f32; QWEN30_TOP_K],
    /// Device-produced post-attention RMSNorm buffer at this layer (router
    /// input). Host copy only; never fed back into native model math.
    pub router_input_hidden: Vec<f32>,
}

/// Full 48-layer router+hidden diagnostic for one tokenized input token.
/// Executes the complete residual stack (embedding through every expert wave)
/// so deeper-layer hiddens are causally real. Final norm / lm_head / sampler
/// still run via the shared greedy path; their outputs are not retained here.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Qwen30AllLayerRouterCaptureStep {
    pub position: usize,
    pub input_token_id: u32,
    pub layers: Vec<Qwen30LayerRouterCapture>,
}

/// Diagnostic-only profiler data from one or more actual native Metal token
/// executions.  It deliberately exposes counts and raw completed-dispatch
/// samples rather than calculating a TPS value; clean sustained throughput is
/// a separate benchmark authority.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Qwen30NativeProfilerSnapshot {
    pub dispatch_samples: Vec<DispatchSample>,
    pub buffers_created: usize,
    pub bytes_allocated: usize,
    pub command_buffers_committed: usize,
}

fn tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut expected = BTreeMap::new();
    expected.insert(
        "model.embed_tokens.weight".into(),
        vec![QWEN30_VOCAB, QWEN30_HIDDEN],
    );
    for layer in 0..QWEN30_LAYERS {
        let prefix = format!("model.layers.{layer}");
        expected.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![QWEN30_HIDDEN],
        );
        expected.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![QWEN30_HIDDEN],
        );
        expected.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![QWEN30_HEAD_DIM],
        );
        expected.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![QWEN30_HEAD_DIM],
        );
        expected.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![QWEN30_HEADS * QWEN30_HEAD_DIM, QWEN30_HIDDEN],
        );
        expected.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![QWEN30_KV_HEADS * QWEN30_HEAD_DIM, QWEN30_HIDDEN],
        );
        expected.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![QWEN30_KV_HEADS * QWEN30_HEAD_DIM, QWEN30_HIDDEN],
        );
        expected.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![QWEN30_HIDDEN, QWEN30_HEADS * QWEN30_HEAD_DIM],
        );
        expected.insert(
            format!("{prefix}.mlp.gate.weight"),
            vec![QWEN30_EXPERTS, QWEN30_HIDDEN],
        );
        for expert in 0..QWEN30_EXPERTS {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            expected.insert(
                format!("{expert_prefix}.gate_proj.weight"),
                vec![QWEN30_MOE_INTERMEDIATE, QWEN30_HIDDEN],
            );
            expected.insert(
                format!("{expert_prefix}.up_proj.weight"),
                vec![QWEN30_MOE_INTERMEDIATE, QWEN30_HIDDEN],
            );
            expected.insert(
                format!("{expert_prefix}.down_proj.weight"),
                vec![QWEN30_HIDDEN, QWEN30_MOE_INTERMEDIATE],
            );
        }
    }
    expected.insert("model.norm.weight".into(), vec![QWEN30_HIDDEN]);
    expected.insert("lm_head.weight".into(), vec![QWEN30_VOCAB, QWEN30_HIDDEN]);
    expected
}

fn validate_complete_catalog(
    artifact: &CompleteBinaryArtifact,
    config: &Qwen30CompleteRuntimeConfig,
) -> Result<()> {
    if artifact.model != QwenCompleteBinaryModel::Qwen30Coder {
        return Err(model_error(
            "complete artifact is not the Qwen30 model family",
        ));
    }
    if artifact.source_revision != config.source_revision
        || artifact.source_revision.is_empty()
        || config.source_repository != QWEN30_REPOSITORY
    {
        return Err(model_error(
            "artifact and source configuration revision binding disagrees",
        ));
    }
    let expected = tensor_shapes();
    if artifact.tensors.len() != expected.len() {
        return Err(model_error(format!(
            "admitted artifact tensor count {} does not equal required Qwen30 count {}",
            artifact.tensors.len(),
            expected.len()
        )));
    }
    let actual: HashSet<&str> = artifact.tensors.keys().map(String::as_str).collect();
    let required: HashSet<&str> = expected.keys().map(String::as_str).collect();
    if actual != required {
        let missing = expected
            .keys()
            .find(|name| !actual.contains(name.as_str()))
            .cloned();
        let unexpected = artifact
            .tensors
            .keys()
            .find(|name| !required.contains(name.as_str()))
            .cloned();
        return Err(model_error(format!(
            "admitted Qwen30 catalog tensor set mismatch; missing={missing:?} unexpected={unexpected:?}"
        )));
    }
    for (name, shape) in expected {
        let tensor = artifact.tensors.get(&name).ok_or_else(|| {
            model_error(format!("required tensor {name:?} vanished after set check"))
        })?;
        if tensor.header.shape != shape || tensor.header.group_size != QWEN30_GROUP_SIZE {
            return Err(model_error(format!(
                "tensor {name:?} has shape {:?}/group {} but requires {:?}/{}",
                tensor.header.shape, tensor.header.group_size, shape, QWEN30_GROUP_SIZE
            )));
        }
    }
    Ok(())
}

fn regular_bytes(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| model_error(format!("cannot stat {label} {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(model_error(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let raw = fs::read(path)
        .map_err(|error| model_error(format!("cannot read {label} {}: {error}", path.display())))?;
    if raw.len() as u64 != metadata.len() {
        return Err(model_error(format!("{label} changed while being read")));
    }
    Ok(raw)
}

fn source_sidecar_path(artifact: &CompleteBinaryArtifact, filename: &str) -> Result<PathBuf> {
    let root = artifact
        .source_index_path
        .parent()
        .ok_or_else(|| model_error("admitted source index has no parent directory"))?;
    let source_root = fs::canonicalize(root).map_err(|error| {
        model_error(format!(
            "cannot canonicalize admitted source directory {}: {error}",
            root.display()
        ))
    })?;
    let path = source_root.join(filename);
    let candidate = fs::canonicalize(&path)
        .map_err(|error| model_error(format!("cannot canonicalize source {filename}: {error}")))?;
    if candidate.parent() != Some(source_root.as_path()) {
        return Err(model_error(format!(
            "source {filename} is not a direct admitted-source child"
        )));
    }
    Ok(candidate)
}

fn parse_source_config(artifact: &CompleteBinaryArtifact) -> Result<(PathBuf, String, Value)> {
    let path = source_sidecar_path(artifact, "config.json")?;
    let raw = regular_bytes(&path, "source config")?;
    let document: Value = serde_json::from_slice(&raw)
        .map_err(|error| model_error(format!("source config is invalid JSON: {error}")))?;
    Ok((path, format!("{:x}", Sha256::digest(&raw)), document))
}

fn tokenizer_from_source(
    artifact: &CompleteBinaryArtifact,
) -> Result<(PathBuf, String, Tokenizer, usize)> {
    let path = source_sidecar_path(artifact, "tokenizer.json")?;
    let raw = regular_bytes(&path, "source tokenizer")?;
    let tokenizer = Tokenizer::from_file(&path)?;
    let addressable_vocab = tokenizer.vocab_size();
    if addressable_vocab == 0 || addressable_vocab > QWEN30_VOCAB {
        return Err(model_error(format!(
            "source tokenizer vocab {addressable_vocab} is outside permitted 1..={QWEN30_VOCAB} model vocabulary"
        )));
    }
    // The source's text tokenizer intentionally covers a contiguous 151,669
    // ID prefix while this Coder LM head has 151,936 rows. The remaining
    // reserved rows have no tokenizer.json text mapping. Retaining that
    // distinction is safer than inventing synthetic tokens: a sampled tail
    // ID causes generation to refuse rather than decode through a substitute.
    Ok((
        path,
        format!("{:x}", Sha256::digest(&raw)),
        tokenizer,
        addressable_vocab,
    ))
}

fn source_user_chat_template_from_source(
    artifact: &CompleteBinaryArtifact,
) -> Result<Qwen30SourceUserChatTemplate> {
    let source_template_path = source_sidecar_path(artifact, "chat_template.jinja")?;
    let source_template_raw = regular_bytes(&source_template_path, "source chat template")?;
    let source_template = std::str::from_utf8(&source_template_raw).map_err(|error| {
        model_error(format!(
            "source chat template is not valid UTF-8 at {}: {error}",
            source_template_path.display()
        ))
    })?;
    // This renderer supports *only* the exact no-system/no-tools, one-user
    // path below.  These anchors encode that source branch and make a future
    // upstream template rewrite fail closed rather than silently changing the
    // prompt token stream.
    for required in [
        "{%- for message in loop_messages %}",
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}",
        "{%- if add_generation_prompt %}",
        "{{- '<|im_start|>assistant\\n' }}",
    ] {
        if !source_template.contains(required) {
            return Err(model_error(format!(
                "source chat template does not contain required one-user branch anchor {required:?}"
            )));
        }
    }
    let tokenizer_config_path = source_sidecar_path(artifact, "tokenizer_config.json")?;
    let tokenizer_config_raw = regular_bytes(&tokenizer_config_path, "source tokenizer config")?;
    let tokenizer_config: Value =
        serde_json::from_slice(&tokenizer_config_raw).map_err(|error| {
            model_error(format!(
                "source tokenizer config is invalid JSON at {}: {error}",
                tokenizer_config_path.display()
            ))
        })?;
    let configured_template = tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
        .ok_or_else(|| model_error("source tokenizer config has no string chat_template"))?;
    if configured_template.as_bytes() != source_template_raw {
        return Err(model_error(
            "source tokenizer_config chat_template differs from source chat_template.jinja",
        ));
    }
    Ok(Qwen30SourceUserChatTemplate {
        source_template_path,
        source_template_sha256: format!("{:x}", Sha256::digest(&source_template_raw)),
        tokenizer_config_path,
        tokenizer_config_sha256: format!("{:x}", Sha256::digest(&tokenizer_config_raw)),
    })
}

fn render_source_user_chat_template(user_content: &str) -> String {
    // The source template's no-system/no-tools branch loops the single user
    // message, then appends its generation prompt.  No escaping is applied by
    // that branch, so retain user content byte-for-byte.
    format!("<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n")
}

/// Bind the exact artifact/config/tokenizer/canonical tensor catalog without
/// constructing a Metal cache.  Useful for an independent preflight worker.
pub fn preflight_complete_runtime(
    manifest_path: impl AsRef<Path>,
    admission: &CompleteBinaryAdmission,
) -> Result<Qwen30CompleteRuntimePreflight> {
    let artifact = admit_complete_binary_artifact(manifest_path, admission)?;
    let (config_path, config_sha256, source_config) = parse_source_config(&artifact)?;
    let config = Qwen30CompleteRuntimeConfig::from_source_config(
        &source_config,
        QWEN30_REPOSITORY,
        &artifact.source_revision,
    )?;
    let (tokenizer_path, tokenizer_sha256, _tokenizer, tokenizer_addressable_vocab) =
        tokenizer_from_source(&artifact)?;
    let source_user_chat_template = source_user_chat_template_from_source(&artifact)?;
    validate_complete_catalog(&artifact, &config)?;
    if artifact.verified_payload_count() != QWEN30_COMPLETE_TENSOR_COUNT
        || !artifact.has_complete_verified_payload_cache()
    {
        return Err(model_error(
            "preflight full admission did not retain one immutable verified payload snapshot per Qwen30 tensor",
        ));
    }
    let verified_payload_count = artifact.verified_payload_count();
    let complete_verified_payload_cache_at_admission =
        artifact.has_complete_verified_payload_cache();
    Ok(Qwen30CompleteRuntimePreflight {
        manifest_path: artifact.manifest_path,
        manifest_seal_sha256: artifact.manifest_seal_sha256,
        source_revision: artifact.source_revision,
        config_path,
        config_sha256,
        tokenizer_path,
        tokenizer_sha256,
        source_user_chat_template_path: source_user_chat_template.source_template_path,
        source_user_chat_template_sha256: source_user_chat_template.source_template_sha256,
        tokenizer_config_path: source_user_chat_template.tokenizer_config_path,
        tokenizer_config_sha256: source_user_chat_template.tokenizer_config_sha256,
        tokenizer_addressable_vocab,
        tensor_count: artifact.tensors.len(),
        tensor_payload_bytes: artifact.tensor_payload_bytes,
        source_weight_elements: artifact.source_weight_elements,
        direct_layout_group_size: QWEN30_GROUP_SIZE,
        verified_payload_count,
        complete_verified_payload_cache_at_admission,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct GpuBinaryTensor {
    signs: PinnedBuffer,
    scales: PinnedBuffer,
    header: CompleteBinaryHeader,
}

#[cfg(target_os = "macos")]
impl GpuBinaryTensor {
    fn rows_cols(&self, name: &str) -> Result<(usize, usize)> {
        if self.header.shape.len() != 2 {
            return Err(model_error(format!("{name:?} is not a matrix tensor")));
        }
        Ok((self.header.shape[0], self.header.shape[1]))
    }
}

/// Host-side binding for one device-selected routed expert.  The sparse
/// variant is deliberately possible only for the separately admitted
/// HQ30GR2 L0/E0 candidate; ordinary runtime routes retain three direct
/// HQ30G1B1 tensors.
#[cfg(target_os = "macos")]
enum Qwen30RoutedExpertWeights {
    Direct {
        gate: GpuBinaryTensor,
        up: GpuBinaryTensor,
        down: GpuBinaryTensor,
    },
    QualitySparseGateUp {
        down: GpuBinaryTensor,
    },
}

#[cfg(target_os = "macos")]
struct DeviceWorkspace {
    x: PinnedBuffer,
    x_norm: PinnedBuffer,
    q: PinnedBuffer,
    k: PinnedBuffer,
    v: PinnedBuffer,
    attention: PinnedBuffer,
    attention_projection: PinnedBuffer,
    router_logits: PinnedBuffer,
    route_ids: PinnedBuffer,
    route_weights: PinnedBuffer,
    expert_gate: PinnedBuffer,
    expert_up: PinnedBuffer,
    expert_activation: PinnedBuffer,
    /// Reserved solely for the fused gate/up candidate's diagnostic control
    /// comparison.  The production candidate does not read it, and the
    /// normal control path continues to use `expert_activation`.
    expert_activation_control: PinnedBuffer,
    expert_output: PinnedBuffer,
    final_logits: PinnedBuffer,
    sampled_token: PinnedBuffer,
    invalid_f32_flag: PinnedBuffer,
    key_cache: PinnedBuffer,
    value_cache: PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl DeviceWorkspace {
    fn new(
        context: &MetalContext,
        max_seq_len: usize,
        config: &Qwen30CompleteRuntimeConfig,
    ) -> Result<Self> {
        let f32_buf = |elements: usize, label: &str| -> Result<PinnedBuffer> {
            context.new_buffer_checked(bytes_for_f32(elements, label)?)
        };
        let kv_elements = config
            .layers
            .checked_mul(max_seq_len)
            .and_then(|value| value.checked_mul(config.kv_dim()))
            .ok_or_else(|| model_error("KV cache element count overflows usize"))?;
        let expert_mid = config
            .experts_per_token
            .checked_mul(config.moe_intermediate)
            .ok_or_else(|| model_error("expert intermediate workspace overflows usize"))?;
        let expert_hidden = config
            .experts_per_token
            .checked_mul(config.hidden)
            .ok_or_else(|| model_error("expert output workspace overflows usize"))?;
        Ok(Self {
            x: f32_buf(config.hidden, "residual workspace")?,
            x_norm: f32_buf(config.hidden, "normalized workspace")?,
            q: f32_buf(config.q_dim(), "query workspace")?,
            k: f32_buf(config.kv_dim(), "key workspace")?,
            v: f32_buf(config.kv_dim(), "value workspace")?,
            attention: f32_buf(config.q_dim(), "attention workspace")?,
            attention_projection: f32_buf(config.hidden, "attention projection workspace")?,
            router_logits: f32_buf(config.experts, "router workspace")?,
            route_ids: context.new_buffer_checked(
                config
                    .experts_per_token
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| model_error("route id workspace byte count overflows usize"))?,
            )?,
            route_weights: f32_buf(config.experts_per_token, "route weight workspace")?,
            expert_gate: f32_buf(expert_mid, "expert gate workspace")?,
            expert_up: f32_buf(expert_mid, "expert up workspace")?,
            expert_activation: f32_buf(expert_mid, "expert activation workspace")?,
            expert_activation_control: f32_buf(
                expert_mid,
                "fused gate/up diagnostic control activation workspace",
            )?,
            expert_output: f32_buf(expert_hidden, "expert output workspace")?,
            final_logits: f32_buf(config.vocab_size, "final logits workspace")?,
            sampled_token: context.new_buffer_checked(std::mem::size_of::<u32>())?,
            invalid_f32_flag: context.new_buffer_checked(std::mem::size_of::<u32>())?,
            key_cache: f32_buf(kv_elements, "native Qwen30 key cache")?,
            value_cache: f32_buf(kv_elements, "native Qwen30 value cache")?,
        })
    }
}

/// Direct Metal runtime for the admitted Qwen30 artifact.  It is deliberately
/// macOS-only: lack of Metal is an error, never a CPU fallback.
#[cfg(target_os = "macos")]
pub struct Qwen30CompleteNativeRuntime {
    artifact: CompleteBinaryArtifact,
    pub config: Qwen30CompleteRuntimeConfig,
    tokenizer: Tokenizer,
    tokenizer_addressable_vocab: usize,
    source_user_chat_template: Qwen30SourceUserChatTemplate,
    context: MetalContext,
    workspace: DeviceWorkspace,
    packed_tensors: HashMap<String, GpuBinaryTensor>,
    decoded_vectors: HashMap<String, PinnedBuffer>,
    packed_matvec_kernel: Qwen30PackedMatvecKernel,
    gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel,
    /// Present only for the separately admitted HQ30GR2 diagnostic body.
    /// The ordinary direct runtime always leaves this absent, so live serving
    /// cannot silently switch representations.
    quality_sparse_gate_up: Option<Qwen30QualityRepackSparseGateUpDevicePair>,
    /// Counts actual route-major HQ30GR2 sparse gate/up encodes for the
    /// separately admitted diagnostic body.  It is never populated by the
    /// direct serving runtime and exists only to make the bounded all-layer
    /// candidate receipt distinguish an armed override from one actually
    /// reached through a device-selected L0/E0 route.
    quality_sparse_gate_up_interception_count: Cell<usize>,
    /// Host-wall intervals are diagnostic only and are enabled alongside the
    /// already explicit dispatch trace. Serving and clean-rate candidates do
    /// not pay timer/ledger collection overhead.
    trace_host_stages: bool,
    /// Enabled only by the bounded candidate/control comparison executor.
    /// Normal serving leaves this false, so retaining diagnostic route
    /// membership cannot silently become part of a production token path.
    diagnostic_route_capture_enabled: bool,
    diagnostic_selected_expert_ids: Vec<[u32; QWEN30_TOP_K]>,
    /// Enabled only by the all-layer activation capture diagnostic.  When set,
    /// each layer's router-input hidden and route membership are retained
    /// after the router command buffer completes and before the expert wave.
    /// Serving never enables this.
    diagnostic_router_hidden_capture_enabled: bool,
    diagnostic_layer_router_captures: Vec<Qwen30LayerRouterCapture>,
    max_seq_len: usize,
    next_position: usize,
}

#[cfg(target_os = "macos")]
impl Qwen30CompleteNativeRuntime {
    /// Re-admit a physical artifact then allocate the native device workspace.
    /// This never calls the raw BF16 source loader and fails if Metal cannot be
    /// initialized.
    pub fn load(
        manifest_path: impl AsRef<Path>,
        admission: &CompleteBinaryAdmission,
        options: Qwen30CompleteRuntimeOptions,
    ) -> Result<Self> {
        let artifact = admit_complete_binary_artifact(manifest_path, admission)?;
        Self::from_admitted_direct_artifact(artifact, options)
    }

    /// Construct from an already admitted direct-body artifact.  The public
    /// `load` path supplies only a normal admitted HQ30G1B1 body.  A separate
    /// HQ30GR2 diagnostic constructor may provide a crate-private direct-base
    /// view, but only after its own full candidate admission and typed sparse
    /// gate/up override have been established.
    fn from_admitted_direct_artifact(
        artifact: CompleteBinaryArtifact,
        options: Qwen30CompleteRuntimeOptions,
    ) -> Result<Self> {
        if options.max_seq_len == 0 || options.max_seq_len > QWEN30_COMPLETE_NATIVE_MAX_CONTEXT {
            return Err(model_error(format!(
                "requested max_seq_len={} is outside native GQA support 1..={QWEN30_COMPLETE_NATIVE_MAX_CONTEXT}",
                options.max_seq_len
            )));
        }
        if artifact.verified_payload_count() != QWEN30_COMPLETE_TENSOR_COUNT
            || !artifact.has_complete_verified_payload_cache()
        {
            return Err(model_error(
                "full direct artifact admission did not retain one immutable verified payload snapshot per Qwen30 tensor",
            ));
        }
        let (_config_path, _config_sha256, source_config) = parse_source_config(&artifact)?;
        let config = Qwen30CompleteRuntimeConfig::from_source_config(
            &source_config,
            QWEN30_REPOSITORY,
            &artifact.source_revision,
        )?;
        if options.max_seq_len > config.source_max_position_embeddings {
            return Err(model_error(format!(
                "requested max_seq_len={} exceeds source config maximum {}",
                options.max_seq_len, config.source_max_position_embeddings
            )));
        }
        let (_tokenizer_path, _tokenizer_sha256, tokenizer, tokenizer_addressable_vocab) =
            tokenizer_from_source(&artifact)?;
        let source_user_chat_template = source_user_chat_template_from_source(&artifact)?;
        validate_complete_catalog(&artifact, &config)?;
        let context = MetalContext::new_with_trace(options.trace_dispatch)?;
        let workspace = DeviceWorkspace::new(&context, options.max_seq_len, &config)?;
        Ok(Self {
            artifact,
            config,
            tokenizer,
            tokenizer_addressable_vocab,
            source_user_chat_template,
            context,
            workspace,
            packed_tensors: HashMap::new(),
            decoded_vectors: HashMap::new(),
            packed_matvec_kernel: options.packed_matvec_kernel,
            gate_up_swiglu_kernel: options.gate_up_swiglu_kernel,
            quality_sparse_gate_up: None,
            quality_sparse_gate_up_interception_count: Cell::new(0),
            trace_host_stages: options.trace_dispatch,
            diagnostic_route_capture_enabled: false,
            diagnostic_selected_expert_ids: Vec::new(),
            diagnostic_router_hidden_capture_enabled: false,
            diagnostic_layer_router_captures: Vec::new(),
            max_seq_len: options.max_seq_len,
            next_position: 0,
        })
    }

    /// Reset only native device state.  The compact weight cache remains bound
    /// to the exact admission and is reused across sessions.
    pub fn reset(&mut self) {
        let zero_kv = vec![0u8; self.workspace.key_cache.length() as usize];
        MetalContext::write_buffer_bytes(&self.workspace.key_cache, &zero_kv);
        MetalContext::write_buffer_bytes(&self.workspace.value_cache, &zero_kv);
        self.diagnostic_selected_expert_ids.clear();
        self.diagnostic_layer_router_captures.clear();
        self.quality_sparse_gate_up_interception_count.set(0);
        self.next_position = 0;
    }

    pub fn position(&self) -> usize {
        self.next_position
    }

    /// Run one complete direct-packed native token while retaining the exact
    /// device-selected expert IDs for each of its 48 layers.  This is limited
    /// to explicitly invoked diagnostic callers; the ordinary greedy API does
    /// not retain this extra observation state.
    pub fn forward_token_greedy_with_route_capture(
        &mut self,
        token: u32,
    ) -> Result<Qwen30NativeRouteCaptureStep> {
        if self.diagnostic_route_capture_enabled {
            return Err(model_error(
                "diagnostic route capture is already active for another native token",
            ));
        }
        self.diagnostic_selected_expert_ids.clear();
        self.diagnostic_route_capture_enabled = true;
        let result = self.forward_token_greedy(token);
        self.diagnostic_route_capture_enabled = false;
        let selected_expert_ids_per_layer =
            std::mem::take(&mut self.diagnostic_selected_expert_ids);
        let greedy = result?;
        if selected_expert_ids_per_layer.len() != self.config.layers {
            return Err(model_error(format!(
                "diagnostic route capture retained {} layers, expected {}",
                selected_expert_ids_per_layer.len(),
                self.config.layers
            )));
        }
        Ok(Qwen30NativeRouteCaptureStep {
            greedy,
            selected_expert_ids_per_layer,
        })
    }

    /// Run one complete 48-layer native token and retain every layer's
    /// device-selected route IDs/weights plus the router-input hidden vector.
    ///
    /// This is the activation-capture primitive for all-layer surplus fitting.
    /// It deliberately reuses the exact residual/expert stack of
    /// [`Self::forward_token_greedy`] so deeper-layer hiddens are causally real
    /// (unlike [`Self::capture_layer0_router_for_token`], which stops before the
    /// L0 expert wave). Serving never enables this path.
    pub fn capture_all_layers_router_for_token(
        &mut self,
        token: u32,
    ) -> Result<Qwen30AllLayerRouterCaptureStep> {
        if self.diagnostic_route_capture_enabled || self.diagnostic_router_hidden_capture_enabled {
            return Err(model_error(
                "all-layer router capture is already active for another native token",
            ));
        }
        self.diagnostic_selected_expert_ids.clear();
        self.diagnostic_layer_router_captures.clear();
        self.diagnostic_route_capture_enabled = true;
        self.diagnostic_router_hidden_capture_enabled = true;
        let result = self.forward_token_greedy(token);
        self.diagnostic_route_capture_enabled = false;
        self.diagnostic_router_hidden_capture_enabled = false;
        let layers = std::mem::take(&mut self.diagnostic_layer_router_captures);
        let _selected = std::mem::take(&mut self.diagnostic_selected_expert_ids);
        let greedy = result?;
        if layers.len() != self.config.layers {
            return Err(model_error(format!(
                "all-layer router capture retained {} layers, expected {}",
                layers.len(),
                self.config.layers
            )));
        }
        for (expected, row) in layers.iter().enumerate() {
            if row.layer != expected {
                return Err(model_error(format!(
                    "all-layer router capture layer order broken: expected {expected}, got {}",
                    row.layer
                )));
            }
            if row.router_input_hidden.len() != self.config.hidden {
                return Err(model_error(format!(
                    "all-layer router capture layer {} hidden width {} != {}",
                    row.layer,
                    row.router_input_hidden.len(),
                    self.config.hidden
                )));
            }
        }
        Ok(Qwen30AllLayerRouterCaptureStep {
            position: greedy.position,
            input_token_id: token,
            layers,
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of source tokenizer IDs that can be losslessly decoded to text.
    /// It can be smaller than the model LM-head row count because Qwen30's
    /// source model reserves a tail with no tokenizer.json text mapping.
    pub fn tokenizer_addressable_vocab(&self) -> usize {
        self.tokenizer_addressable_vocab
    }

    /// Exact source binding for the bounded one-user-message chat path.
    pub fn source_user_chat_template(&self) -> &Qwen30SourceUserChatTemplate {
        &self.source_user_chat_template
    }

    /// Render the source's user-only chat-template branch.  This is not a
    /// generic template engine: supplying system messages, tools, or roles
    /// other than one user message belongs to a later, explicitly validated
    /// template implementation.
    pub fn render_source_user_chat_prompt(&self, user_content: &str) -> String {
        render_source_user_chat_template(user_content)
    }

    pub fn artifact_manifest_seal(&self) -> &str {
        &self.artifact.manifest_seal_sha256
    }

    /// Number of immutable direct payload snapshots verified during this
    /// process's full artifact admission. It must remain the complete sealed
    /// catalog count; a lazy subset is not accepted for production execution.
    pub fn verified_payload_count(&self) -> usize {
        self.artifact.verified_payload_count()
    }

    /// Whether the runtime owns a complete immutable direct-payload catalog
    /// for this process. This is an integrity/residency fact only, not a TPS
    /// result.
    pub fn has_complete_verified_payload_cache(&self) -> bool {
        self.artifact.has_complete_verified_payload_cache()
    }

    pub fn packed_matvec_kernel(&self) -> Qwen30PackedMatvecKernel {
        self.packed_matvec_kernel
    }

    /// Current routed-expert gate/up/SwiGLU topology.  A caller must retain
    /// the receipt name in any candidate result so it cannot be mistaken for
    /// the scalar model control.
    pub fn gate_up_swiglu_kernel(&self) -> Qwen30GateUpSwiGluKernel {
        self.gate_up_swiglu_kernel
    }

    /// Drain diagnostic dispatch/allocation data accumulated since the last
    /// call. This has no throughput interpretation on its own: callers must
    /// use a dedicated clean sustained benchmark before publishing TPS.
    pub fn drain_profiler(&self) -> Qwen30NativeProfilerSnapshot {
        let (buffers_created, bytes_allocated, command_buffers_committed) =
            self.context.drain_stats();
        Qwen30NativeProfilerSnapshot {
            dispatch_samples: self.context.drain_trace(),
            buffers_created,
            bytes_allocated,
            command_buffers_committed,
        }
    }

    fn packed_tensor(&mut self, name: &str) -> Result<GpuBinaryTensor> {
        if let Some(tensor) = self.packed_tensors.get(name) {
            return Ok(tensor.clone());
        }
        // The complete artifact was SHA-256 scanned before this runtime was
        // constructed. Retain and consume that immutable admission snapshot
        // here rather than reopening/re-hashing a payload on a token path.
        // A restart re-enters `admit_complete_binary_artifact` and revalidates
        // every one of the 18,867 source-bound payloads.
        let payload = self.artifact.verified_tensor_payload(name)?;
        let header = self.artifact.tensor(name)?.header.clone();
        if payload.len() != header.payload_bytes {
            return Err(model_error(format!(
                "verified immutable payload {name:?} has {} bytes but admitted header requires {}",
                payload.len(),
                header.payload_bytes
            )));
        }
        if header.group_size != QWEN30_GROUP_SIZE {
            return Err(model_error(format!(
                "tensor {name:?} group size {} is not the admitted Qwen30 group size {QWEN30_GROUP_SIZE}",
                header.group_size
            )));
        }
        let scales = payload
            .get(header.scale_offset..header.sign_offset)
            .ok_or_else(|| model_error(format!("tensor {name:?} scales are truncated")))?;
        let signs = payload
            .get(header.sign_offset..header.payload_bytes)
            .ok_or_else(|| model_error(format!("tensor {name:?} signs are truncated")))?;
        let expected_sign_bytes = header
            .groups
            .checked_mul(header.group_size / 8)
            .ok_or_else(|| model_error(format!("tensor {name:?} sign byte count overflow")))?;
        if signs.len() != expected_sign_bytes || scales.len() != header.groups * 2 {
            return Err(model_error(format!(
                "tensor {name:?} compact sections disagree with checked header"
            )));
        }
        let tensor = GpuBinaryTensor {
            signs: self.context.new_buffer_with_bytes_checked(signs)?,
            scales: self.context.new_buffer_with_bytes_checked(scales)?,
            header,
        };
        self.packed_tensors.insert(name.to_owned(), tensor.clone());
        Ok(tensor)
    }

    /// Decode one 1-D packed vector into a cached f32 buffer, or return the
    /// existing cache entry.  A cache miss used to open a dedicated command
    /// buffer and `commit_and_wait` per vector (193 extra CBs on a cold first
    /// token).  Prefer [`Self::ensure_decoded_vector_on_tcb`] so the decode is
    /// folded into the caller's graph; this standalone path remains for
    /// diagnostic callers that do not already own a token command buffer.
    fn decoded_vector(&mut self, name: &str, elements: usize) -> Result<PinnedBuffer> {
        if let Some(buffer) = self.decoded_vectors.get(name) {
            return Ok(buffer.clone());
        }
        // Clone the Arc-backed context so the TCB does not borrow `self.context`
        // while `ensure_decoded_vector_on_tcb` needs `&mut self`.
        let context = self.context.clone();
        let mut tcb = TokenCommandBuffer::new(&context);
        let output = self.ensure_decoded_vector_on_tcb(&mut tcb, name, elements)?;
        if tcb.dispatch_count() > 0 {
            tcb.commit_and_wait()?;
        }
        Ok(output)
    }

    /// Ensure `name` is resident as a decoded f32 vector.  On a cache miss the
    /// decode kernel is encoded into `tcb` and the result is retained for the
    /// process lifetime of this runtime; on a hit this is a pure HashMap
    /// lookup with zero Metal work.  The caller remains responsible for the
    /// eventual `commit_and_wait` that covers the folded decode.
    fn ensure_decoded_vector_on_tcb(
        &mut self,
        tcb: &mut TokenCommandBuffer<'_>,
        name: &str,
        elements: usize,
    ) -> Result<PinnedBuffer> {
        if let Some(buffer) = self.decoded_vectors.get(name) {
            return Ok(buffer.clone());
        }
        let tensor = self.packed_tensor(name)?;
        if tensor.header.shape != [elements] {
            return Err(model_error(format!(
                "vector {name:?} has shape {:?}, expected [{elements}]",
                tensor.header.shape
            )));
        }
        let output = self
            .context
            .new_buffer_checked(bytes_for_f32(elements, "decoded vector")?)?;
        let threads = u32_checked(elements, "decoded vector elements")?;
        tcb.dispatch_threads(
            "qwen_complete_binary_decode_vector",
            (threads, 1, 1),
            (threads.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&tensor.signs), 0);
                encoder.set_buffer(1, Some(&tensor.scales), 0);
                encoder.set_buffer(2, Some(&output), 0);
                encoder.qwen_set_u32(3, threads);
                encoder.qwen_set_u32(4, QWEN30_GROUP_SIZE as u32);
            },
        )?;
        self.decoded_vectors.insert(name.to_owned(), output.clone());
        Ok(output)
    }

    /// Names and element counts of every static RMSNorm vector the all-layer
    /// token graph will need.  Expert and projection tensors stay packed and
    /// are not listed here.
    fn static_decoded_vector_catalog(&self) -> Vec<(String, usize)> {
        let mut catalog =
            Vec::with_capacity(self.config.layers.saturating_mul(4).saturating_add(1));
        for layer in 0..self.config.layers {
            catalog.push((
                Self::layer_name(layer, "input_layernorm.weight"),
                self.config.hidden,
            ));
            catalog.push((
                Self::layer_name(layer, "post_attention_layernorm.weight"),
                self.config.hidden,
            ));
            catalog.push((
                Self::layer_name(layer, "self_attn.q_norm.weight"),
                self.config.head_dim,
            ));
            catalog.push((
                Self::layer_name(layer, "self_attn.k_norm.weight"),
                self.config.head_dim,
            ));
        }
        catalog.push(("model.norm.weight".to_owned(), self.config.hidden));
        catalog
    }

    /// Decode every static RMSNorm vector into one command buffer (one fence).
    ///
    /// Component-only / load-path helper.  The sealed complete-token profile
    /// recorded 291 command buffers on a cold first token because each of the
    /// 193 vector decodes paid its own `commit_and_wait`.  After this prewarm
    /// a warm token's structural graph is embed + 48×(attn/router) +
    /// 48×(experts) + final_head = 98 CBs until the host route-id roundtrip
    /// itself is removed.  Not a TPS measurement.
    pub fn prewarm_static_decoded_vectors(&mut self) -> Result<Qwen30StaticDecodePrewarmReport> {
        let catalog = self.static_decoded_vector_catalog();
        let catalog_len = catalog.len();
        let already_resident = catalog
            .iter()
            .filter(|(name, _)| self.decoded_vectors.contains_key(name))
            .count();
        // Clone context: TCB must not hold a borrow of `self.context` across
        // `&mut self` cache inserts inside `ensure_decoded_vector_on_tcb`.
        let context = self.context.clone();
        let mut tcb = TokenCommandBuffer::new(&context);
        let serial = qwen30_serial_encoder_enabled();
        if serial {
            tcb.begin_serial_group()?;
        }
        let mut decoded_now = 0usize;
        for (name, elements) in &catalog {
            if self.decoded_vectors.contains_key(name) {
                continue;
            }
            self.ensure_decoded_vector_on_tcb(&mut tcb, name, *elements)?;
            decoded_now = decoded_now.saturating_add(1);
        }
        if serial {
            tcb.end_concurrent_group()?;
        }
        let dispatches = tcb.dispatch_count();
        let command_buffers = if dispatches > 0 {
            tcb.commit_and_wait()?;
            1usize
        } else {
            0usize
        };
        Ok(Qwen30StaticDecodePrewarmReport {
            catalog_vectors: catalog_len,
            already_resident,
            decoded_now,
            dispatches,
            command_buffers,
            serial_encoder: serial,
        })
    }

    /// Whether the production token path currently folds multi-dispatch waves
    /// into one serial compute encoder.  Diagnostic `HAWKING_TCB_TRACE=gpu*`
    /// modes ignore the serial group (no-op inside TCB) so per-kernel
    /// timestamps remain available.
    pub fn serial_encoder_enabled(&self) -> bool {
        qwen30_serial_encoder_enabled()
    }

    fn dispatch_embedding(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        embedding: &GpuBinaryTensor,
        token: u32,
    ) -> Result<()> {
        if embedding.header.shape != [self.config.vocab_size, self.config.hidden] {
            return Err(model_error(
                "embedding tensor shape changed after catalog admission",
            ));
        }
        let hidden = u32_checked(self.config.hidden, "embedding hidden")?;
        tcb.dispatch_threads(
            "qwen_complete_binary_embedding_lookup",
            (hidden, 1, 1),
            (hidden.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&embedding.signs), 0);
                encoder.set_buffer(1, Some(&embedding.scales), 0);
                encoder.set_buffer(2, Some(&self.workspace.x), 0);
                encoder.qwen_set_u32(3, token);
                encoder.qwen_set_u32(4, hidden);
                encoder.qwen_set_u32(5, self.config.vocab_size as u32);
                encoder.qwen_set_u32(6, QWEN30_GROUP_SIZE as u32);
            },
        )
    }

    fn dispatch_binary_matvec(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        weight: &GpuBinaryTensor,
        input: &PinnedBuffer,
        output: &PinnedBuffer,
        input_offset_bytes: usize,
        output_offset_bytes: usize,
    ) -> Result<()> {
        let (rows, cols) = weight.rows_cols("direct binary projection")?;
        if weight.header.group_size != QWEN30_GROUP_SIZE {
            return Err(model_error("direct binary projection group size drifted"));
        }
        let required_input = bytes_for_f32(cols, "direct binary projection input")?;
        let input_end = input_offset_bytes
            .checked_add(required_input)
            .ok_or_else(|| model_error("direct binary projection input offset overflows usize"))?;
        if input_end > input.length() as usize {
            return Err(model_error(
                "direct binary projection input range exceeds workspace",
            ));
        }
        let required_output = bytes_for_f32(rows, "direct binary projection output")?;
        let end = output_offset_bytes
            .checked_add(required_output)
            .ok_or_else(|| model_error("direct binary projection output offset overflows usize"))?;
        if end > output.length() as usize {
            return Err(model_error(
                "direct binary projection output range exceeds workspace",
            ));
        }
        let rows_u32 = u32_checked(rows, "direct binary projection rows")?;
        let cols_u32 = u32_checked(cols, "direct binary projection cols")?;
        let groups = cols.div_ceil(QWEN30_GROUP_SIZE);
        let (kernel, grid, threads_per_threadgroup) = match self.packed_matvec_kernel {
            Qwen30PackedMatvecKernel::ScalarControl => (
                "qwen_binary_sign_scale_matvec",
                (rows_u32, 1, 1),
                (rows_u32.min(256).max(1), 1, 1),
            ),
            Qwen30PackedMatvecKernel::SimdgroupCandidate => {
                let groups_of_rows = rows.div_ceil(8);
                let grid_x = groups_of_rows
                    .checked_mul(256)
                    .ok_or_else(|| model_error("simdgroup projection grid overflows usize"))?;
                (
                    "qwen_binary_sign_scale_matvec_simdgroup_candidate",
                    (u32_checked(grid_x, "simdgroup projection grid")?, 1, 1),
                    (256, 1, 1),
                )
            }
        };
        tcb.dispatch_threads(kernel, grid, threads_per_threadgroup, |encoder| {
            encoder.set_buffer(0, Some(&weight.signs), 0);
            encoder.set_buffer(1, Some(&weight.scales), 0);
            encoder.set_buffer(2, Some(input), input_offset_bytes as u64);
            encoder.set_buffer(3, Some(output), output_offset_bytes as u64);
            encoder.qwen_set_u32(4, rows_u32);
            encoder.qwen_set_u32(5, cols_u32);
            encoder.qwen_set_u32(6, QWEN30_GROUP_SIZE as u32);
            encoder.qwen_set_u32(7, groups as u32);
        })
    }

    fn dispatch_rmsnorm_rows(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        weight: &PinnedBuffer,
        output: &PinnedBuffer,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let rows_u32 = u32_checked(rows, "Q/K RMSNorm rows")?;
        let width_u32 = u32_checked(width, "Q/K RMSNorm width")?;
        tcb.dispatch_threads(
            "qwen_complete_rmsnorm_rows_f32",
            (256, rows_u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(weight), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.qwen_set_u32(3, rows_u32);
                encoder.qwen_set_u32(4, width_u32);
                encoder.qwen_set_f32(5, self.config.rms_norm_eps());
                encoder.set_threadgroup_memory_length(0, 256 * std::mem::size_of::<f32>() as u64);
            },
        )
    }

    fn dispatch_normalize_route_weights(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
        tcb.dispatch_threads(
            "qwen_complete_normalize_route_weights",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.route_weights), 0);
                encoder.qwen_set_u32(1, self.config.experts_per_token as u32);
            },
        )
    }

    fn dispatch_silu_offset(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        offset_bytes: usize,
        output: &PinnedBuffer,
    ) -> Result<()> {
        let elements = self.config.moe_intermediate;
        let elements_u32 = u32_checked(elements, "expert activation elements")?;
        let required = bytes_for_f32(elements, "expert activation output")?;
        let end = offset_bytes
            .checked_add(required)
            .ok_or_else(|| model_error("expert activation output offset overflows usize"))?;
        if end > output.length() as usize {
            return Err(model_error(
                "expert activation output range exceeds its route-major workspace",
            ));
        }
        tcb.dispatch_threads(
            "qwen_complete_silu_mul_offset",
            (elements_u32, 1, 1),
            (elements_u32.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.expert_gate), offset_bytes as u64);
                encoder.set_buffer(1, Some(&self.workspace.expert_up), offset_bytes as u64);
                encoder.set_buffer(2, Some(output), offset_bytes as u64);
                encoder.qwen_set_u32(3, elements_u32);
            },
        )
    }

    /// Encode the isolated routed-expert candidate.  It has a deliberately
    /// narrower ABI than `dispatch_binary_matvec`: exactly two admitted
    /// `[768, 2048]` HQ30G1B1 matrices, one normalized hidden input, and a
    /// route-major 768-value activation output.  Any other geometry is a
    /// runtime error rather than an accidental broad fusion.
    fn dispatch_gate_up_swiglu_fused_candidate(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        gate: &GpuBinaryTensor,
        up: &GpuBinaryTensor,
        output: &PinnedBuffer,
        output_offset_bytes: usize,
    ) -> Result<()> {
        let gate_shape = gate.rows_cols("fused gate/up gate projection")?;
        let up_shape = up.rows_cols("fused gate/up up projection")?;
        let expected = (self.config.moe_intermediate, self.config.hidden);
        if gate_shape != expected || up_shape != expected {
            return Err(model_error(format!(
                "fused gate/up candidate requires exact {:?} gate/up geometry, got gate={gate_shape:?}, up={up_shape:?}",
                expected
            )));
        }
        if gate.header.group_size != QWEN30_GROUP_SIZE || up.header.group_size != QWEN30_GROUP_SIZE
        {
            return Err(model_error(
                "fused gate/up candidate requires exact HQ30G1B1 group_size=128 for both projections",
            ));
        }
        let required = bytes_for_f32(self.config.moe_intermediate, "fused gate/up activation")?;
        let end = output_offset_bytes
            .checked_add(required)
            .ok_or_else(|| model_error("fused gate/up activation offset overflows usize"))?;
        if end > output.length() as usize {
            return Err(model_error(
                "fused gate/up activation range exceeds its route-major workspace",
            ));
        }
        let rows = u32_checked(self.config.moe_intermediate, "fused gate/up rows")?;
        let cols = u32_checked(self.config.hidden, "fused gate/up cols")?;
        tcb.dispatch_threads(
            "qwen_direct_packed_gate_up_swiglu_fused_candidate",
            (rows, 1, 1),
            (rows.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&gate.signs), 0);
                encoder.set_buffer(1, Some(&gate.scales), 0);
                encoder.set_buffer(2, Some(&up.signs), 0);
                encoder.set_buffer(3, Some(&up.scales), 0);
                encoder.set_buffer(4, Some(&self.workspace.x_norm), 0);
                encoder.set_buffer(5, Some(output), output_offset_bytes as u64);
                encoder.qwen_set_u32(6, rows);
                encoder.qwen_set_u32(7, cols);
                encoder.qwen_set_u32(8, QWEN30_GROUP_SIZE as u32);
            },
        )
    }

    /// Encode the separate scalar-order-preserving paired gate/up candidate.
    /// It shares only the narrow two-projection ABI with the older explicit
    /// FMA experiment. Its shader is separately named and retains scalar
    /// non-FMA accumulation, so prior FMA rejection evidence cannot be
    /// reused as parity evidence for this topology trial.
    fn dispatch_gate_up_swiglu_paired_scalar_order_candidate(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        gate: &GpuBinaryTensor,
        up: &GpuBinaryTensor,
        output: &PinnedBuffer,
        output_offset_bytes: usize,
    ) -> Result<()> {
        let gate_shape = gate.rows_cols("paired scalar-order gate/up gate projection")?;
        let up_shape = up.rows_cols("paired scalar-order gate/up up projection")?;
        let expected = (self.config.moe_intermediate, self.config.hidden);
        if gate_shape != expected || up_shape != expected {
            return Err(model_error(format!(
                "paired scalar-order gate/up candidate requires exact {:?} gate/up geometry, got gate={gate_shape:?}, up={up_shape:?}",
                expected
            )));
        }
        if gate.header.group_size != QWEN30_GROUP_SIZE || up.header.group_size != QWEN30_GROUP_SIZE
        {
            return Err(model_error(
                "paired scalar-order gate/up candidate requires exact HQ30G1B1 group_size=128 for both projections",
            ));
        }
        let required = bytes_for_f32(
            self.config.moe_intermediate,
            "paired scalar-order gate/up activation",
        )?;
        let end = output_offset_bytes.checked_add(required).ok_or_else(|| {
            model_error("paired scalar-order gate/up activation offset overflows usize")
        })?;
        if end > output.length() as usize {
            return Err(model_error(
                "paired scalar-order gate/up activation range exceeds its route-major workspace",
            ));
        }
        let rows = u32_checked(
            self.config.moe_intermediate,
            "paired scalar-order gate/up rows",
        )?;
        let cols = u32_checked(self.config.hidden, "paired scalar-order gate/up cols")?;
        tcb.dispatch_threads(
            "qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate",
            (rows, 1, 1),
            (rows.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&gate.signs), 0);
                encoder.set_buffer(1, Some(&gate.scales), 0);
                encoder.set_buffer(2, Some(&up.signs), 0);
                encoder.set_buffer(3, Some(&up.scales), 0);
                encoder.set_buffer(4, Some(&self.workspace.x_norm), 0);
                encoder.set_buffer(5, Some(output), output_offset_bytes as u64);
                encoder.qwen_set_u32(6, rows);
                encoder.qwen_set_u32(7, cols);
                encoder.qwen_set_u32(8, QWEN30_GROUP_SIZE as u32);
            },
        )
    }

    /// Encode either the retained independent control topology or the bounded
    /// fusion candidate.  In diagnostic parity mode both paths operate on the
    /// *same* direct-packed weights and device-resident normalized input; the
    /// candidate activation is the only value consumed by the subsequent down
    /// projection, while the control activation remains available for a
    /// post-completion device-buffer comparison.
    fn dispatch_expert_gate_up_swiglu(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        gate: &GpuBinaryTensor,
        up: &GpuBinaryTensor,
        output_offset_bytes: usize,
    ) -> Result<()> {
        match self.gate_up_swiglu_kernel {
            Qwen30GateUpSwiGluKernel::ThreeDispatchControl => {
                self.dispatch_binary_matvec(
                    tcb,
                    gate,
                    &self.workspace.x_norm,
                    &self.workspace.expert_gate,
                    0,
                    output_offset_bytes,
                )?;
                self.dispatch_binary_matvec(
                    tcb,
                    up,
                    &self.workspace.x_norm,
                    &self.workspace.expert_up,
                    0,
                    output_offset_bytes,
                )?;
                self.dispatch_silu_offset(
                    tcb,
                    output_offset_bytes,
                    &self.workspace.expert_activation,
                )
            }
            Qwen30GateUpSwiGluKernel::FusedCandidate => self
                .dispatch_gate_up_swiglu_fused_candidate(
                    tcb,
                    gate,
                    up,
                    &self.workspace.expert_activation,
                    output_offset_bytes,
                ),
            Qwen30GateUpSwiGluKernel::FusedCandidateWithDeviceControlParity => {
                self.dispatch_binary_matvec(
                    tcb,
                    gate,
                    &self.workspace.x_norm,
                    &self.workspace.expert_gate,
                    0,
                    output_offset_bytes,
                )?;
                self.dispatch_binary_matvec(
                    tcb,
                    up,
                    &self.workspace.x_norm,
                    &self.workspace.expert_up,
                    0,
                    output_offset_bytes,
                )?;
                self.dispatch_silu_offset(
                    tcb,
                    output_offset_bytes,
                    &self.workspace.expert_activation_control,
                )?;
                self.dispatch_gate_up_swiglu_fused_candidate(
                    tcb,
                    gate,
                    up,
                    &self.workspace.expert_activation,
                    output_offset_bytes,
                )
            }
            Qwen30GateUpSwiGluKernel::PairedScalarOrderCandidateWithDeviceControlParity => {
                self.dispatch_binary_matvec(
                    tcb,
                    gate,
                    &self.workspace.x_norm,
                    &self.workspace.expert_gate,
                    0,
                    output_offset_bytes,
                )?;
                self.dispatch_binary_matvec(
                    tcb,
                    up,
                    &self.workspace.x_norm,
                    &self.workspace.expert_up,
                    0,
                    output_offset_bytes,
                )?;
                self.dispatch_silu_offset(
                    tcb,
                    output_offset_bytes,
                    &self.workspace.expert_activation_control,
                )?;
                self.dispatch_gate_up_swiglu_paired_scalar_order_candidate(
                    tcb,
                    gate,
                    up,
                    &self.workspace.expert_activation,
                    output_offset_bytes,
                )
            }
            Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity => self
                .dispatch_gate_up_swiglu_paired_scalar_order_candidate(
                    tcb,
                    gate,
                    up,
                    &self.workspace.expert_activation,
                    output_offset_bytes,
                ),
        }
    }

    fn quality_sparse_gate_up_applies(&self, layer: usize, expert: u32) -> bool {
        self.quality_sparse_gate_up.is_some() && layer == 0 && expert == 0
    }

    fn dispatch_quality_sparse_gate_up(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        output_offset_bytes: usize,
    ) -> Result<()> {
        {
            let pair = self.quality_sparse_gate_up.as_ref().ok_or_else(|| {
                model_error(
                    "quality sparse gate/up route was selected without a typed candidate device pair",
                )
            })?;
            pair.encode(
                tcb,
                &self.workspace.x_norm,
                &self.workspace.expert_activation,
                output_offset_bytes,
            )?;
        }
        let next_count = self
            .quality_sparse_gate_up_interception_count
            .get()
            .checked_add(1)
            .ok_or_else(|| model_error("HQ30GR2 sparse gate/up interception count overflowed"))?;
        self.quality_sparse_gate_up_interception_count
            .set(next_count);
        Ok(())
    }

    /// Compare exactly the selected route-major activation buffers after the
    /// command buffer which produced them has completed.  This is a numerical
    /// parity oracle only: it reads two Metal buffers generated from the same
    /// admitted packed bodies and never calls host model math or a BF16 path.
    fn compare_fused_gate_up_device_control(
        &self,
        layer: usize,
    ) -> Result<Qwen30GateUpSwiGluDeviceParity> {
        const TOLERANCE: f32 = 4.0e-3;
        let route_count = self.config.experts_per_token;
        let elements_per_route = self.config.moe_intermediate;
        let elements = route_count
            .checked_mul(elements_per_route)
            .ok_or_else(|| model_error("fused gate/up parity element count overflows usize"))?;
        let candidate = unsafe {
            std::slice::from_raw_parts(
                self.workspace.expert_activation.contents() as *const f32,
                elements,
            )
        };
        let control = unsafe {
            std::slice::from_raw_parts(
                self.workspace.expert_activation_control.contents() as *const f32,
                elements,
            )
        };
        let mut max_abs_error = 0.0f32;
        for (index, (&candidate_value, &control_value)) in
            candidate.iter().zip(control.iter()).enumerate()
        {
            if !candidate_value.is_finite() || !control_value.is_finite() {
                let route = index / elements_per_route;
                return Err(model_error(format!(
                    "fused gate/up parity produced a non-finite activation at layer {layer}, route {route}, element {}",
                    index % elements_per_route
                )));
            }
            let error = (candidate_value - control_value).abs();
            if error > max_abs_error {
                max_abs_error = error;
            }
            if error > TOLERANCE {
                let route = index / elements_per_route;
                return Err(model_error(format!(
                    "fused gate/up parity exceeded tolerance at layer {layer}, route {route}, element {}: {error} > {TOLERANCE}",
                    index % elements_per_route
                )));
            }
        }
        Ok(Qwen30GateUpSwiGluDeviceParity {
            layers_compared: 1,
            routed_experts_compared: route_count,
            activation_values_compared: elements,
            max_abs_error,
            tolerance_max_abs: TOLERANCE,
        })
    }

    fn dispatch_weighted_expert_add(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
        let hidden = u32_checked(self.config.hidden, "expert combine hidden")?;
        tcb.dispatch_threads(
            "qwen_complete_weighted_expert_add",
            (hidden, 1, 1),
            (hidden.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.expert_output), 0);
                encoder.set_buffer(1, Some(&self.workspace.route_weights), 0);
                encoder.set_buffer(2, Some(&self.workspace.x), 0);
                encoder.qwen_set_u32(3, hidden);
                encoder.qwen_set_u32(4, self.config.experts_per_token as u32);
            },
        )
    }

    /// Assert on the device that a final f32 tensor stayed finite. This is a
    /// fail-closed integrity check, not an alternate sampler or a CPU model
    /// calculation: the host reads one device-written flag only after all
    /// Metal execution is complete.
    fn dispatch_finite_check(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        elements: usize,
    ) -> Result<()> {
        let elements_u32 = u32_checked(elements, "finite-check elements")?;
        tcb.dispatch_threads(
            "qwen_complete_any_nonfinite_f32",
            (elements_u32, 1, 1),
            (elements_u32.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(&self.workspace.invalid_f32_flag), 0);
                encoder.qwen_set_u32(2, elements_u32);
            },
        )
    }

    fn assert_final_logits_finite(&self) -> Result<()> {
        let invalid = unsafe { *(self.workspace.invalid_f32_flag.contents() as *const u32) };
        if invalid != 0 {
            return Err(model_error(
                "native Metal final logits contain a non-finite value; refusing sampled token",
            ));
        }
        Ok(())
    }

    /// Copy device-produced final logits for a bounded, separately admitted
    /// diagnostic only.  This does not perform host inference or sampling;
    /// callers must bind any use to the completed native forward which wrote
    /// this workspace and must not treat the snapshot as HCLI/TPS evidence.
    pub fn diagnostic_final_logits_f32(&self) -> Result<Vec<f32>> {
        self.assert_final_logits_finite()?;
        let logits = unsafe {
            std::slice::from_raw_parts(
                self.workspace.final_logits.contents() as *const f32,
                self.config.vocab_size,
            )
        };
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "device final-logit diagnostic copy contains a non-finite value",
            ));
        }
        Ok(logits.to_vec())
    }

    fn route_ids(&self) -> Result<[u32; QWEN30_TOP_K]> {
        let ids = unsafe {
            std::slice::from_raw_parts(
                self.workspace.route_ids.contents() as *const u32,
                self.config.experts_per_token,
            )
        };
        let mut output = [0u32; QWEN30_TOP_K];
        output.copy_from_slice(ids);
        let mut seen = HashSet::new();
        for &id in &output {
            if id as usize >= self.config.experts || !seen.insert(id) {
                return Err(model_error(format!(
                    "device router produced invalid/duplicate expert id {id}"
                )));
            }
        }
        Ok(output)
    }

    /// Read the eight device-normalized route weights after a completed router
    /// command buffer.  This is an observation-only shared-memory read; the
    /// host neither recomputes router scores nor normalizes the weights.
    fn route_weights(&self) -> Result<[f32; QWEN30_TOP_K]> {
        let weights = unsafe {
            std::slice::from_raw_parts(
                self.workspace.route_weights.contents() as *const f32,
                self.config.experts_per_token,
            )
        };
        let mut output = [0.0f32; QWEN30_TOP_K];
        output.copy_from_slice(weights);
        if output
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(model_error(
                "device router produced a non-finite or negative normalized route weight",
            ));
        }
        let total = output.iter().copied().sum::<f32>();
        if !total.is_finite() || (total - 1.0).abs() > 2.0e-3 {
            return Err(model_error(format!(
                "device router normalized route weights sum to {total}, not one"
            )));
        }
        Ok(output)
    }

    /// Copy the exact device-produced router input vector only for a bounded
    /// diagnostic record.  The next operator receives this same shared buffer
    /// on device; the returned host copy is never fed back into native model
    /// execution.
    fn router_input_hidden(&self) -> Result<Vec<f32>> {
        let values = unsafe {
            std::slice::from_raw_parts(
                self.workspace.x_norm.contents() as *const f32,
                self.config.hidden,
            )
        };
        if values.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "device L0 router input hidden state contains a non-finite value",
            ));
        }
        Ok(values.to_vec())
    }

    /// Execute exactly the native portion needed to observe an L0 router
    /// decision for one token.  This intentionally excludes the L0 expert
    /// wave and all later layers: the layer-0 router runs before the isolated
    /// HQ30GR2 L0/E0 gate/up residual, so its route membership is causally
    /// common to the control and candidate representations.
    pub fn capture_layer0_router_for_token(
        &mut self,
        token: u32,
    ) -> Result<Qwen30Layer0RouterCapture> {
        if token as usize >= self.config.vocab_size {
            return Err(model_error(format!(
                "L0 router capture token {token} is outside model vocabulary"
            )));
        }
        if self.next_position >= self.max_seq_len {
            return Err(model_error(format!(
                "L0 router capture KV cache is full at position {}; reset before continuing",
                self.next_position
            )));
        }
        let position = self.next_position;
        let embedding = self.packed_tensor("model.embed_tokens.weight")?;
        {
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.dispatch_embedding(&mut tcb, &embedding, token)?;
            tcb.commit_and_wait()?;
        }

        let layer = 0usize;
        let input_norm = self.decoded_vector(
            &Self::layer_name(layer, "input_layernorm.weight"),
            self.config.hidden,
        )?;
        let post_norm = self.decoded_vector(
            &Self::layer_name(layer, "post_attention_layernorm.weight"),
            self.config.hidden,
        )?;
        let q_norm = self.decoded_vector(
            &Self::layer_name(layer, "self_attn.q_norm.weight"),
            self.config.head_dim,
        )?;
        let k_norm = self.decoded_vector(
            &Self::layer_name(layer, "self_attn.k_norm.weight"),
            self.config.head_dim,
        )?;
        let q = self.packed_tensor(&Self::layer_name(layer, "self_attn.q_proj.weight"))?;
        let k = self.packed_tensor(&Self::layer_name(layer, "self_attn.k_proj.weight"))?;
        let v = self.packed_tensor(&Self::layer_name(layer, "self_attn.v_proj.weight"))?;
        let o = self.packed_tensor(&Self::layer_name(layer, "self_attn.o_proj.weight"))?;
        let router = self.packed_tensor(&Self::layer_name(layer, "mlp.gate.weight"))?;
        let kv_offset = position
            .checked_mul(self.config.kv_dim())
            .ok_or_else(|| model_error("L0 router capture KV offset overflows usize"))?;
        {
            let mut tcb = TokenCommandBuffer::new(&self.context);
            rmsnorm_metal_buf_tcb(
                &mut tcb,
                &self.workspace.x,
                &input_norm,
                self.config.rms_norm_eps(),
                self.config.hidden,
                &self.workspace.x_norm,
            )?;
            self.dispatch_binary_matvec(
                &mut tcb,
                &q,
                &self.workspace.x_norm,
                &self.workspace.q,
                0,
                0,
            )?;
            self.dispatch_binary_matvec(
                &mut tcb,
                &k,
                &self.workspace.x_norm,
                &self.workspace.k,
                0,
                0,
            )?;
            self.dispatch_binary_matvec(
                &mut tcb,
                &v,
                &self.workspace.x_norm,
                &self.workspace.v,
                0,
                0,
            )?;
            self.dispatch_rmsnorm_rows(
                &mut tcb,
                &self.workspace.q,
                &q_norm,
                &self.workspace.q,
                self.config.attention_heads,
                self.config.head_dim,
            )?;
            self.dispatch_rmsnorm_rows(
                &mut tcb,
                &self.workspace.k,
                &k_norm,
                &self.workspace.k,
                self.config.key_value_heads,
                self.config.head_dim,
            )?;
            rope_qk_kv_append_vbias_f32_tcb(
                &mut tcb,
                &self.workspace.q,
                &self.workspace.k,
                &self.workspace.v,
                None,
                None,
                None,
                &self.workspace.key_cache,
                &self.workspace.value_cache,
                self.config.attention_heads,
                self.config.key_value_heads,
                self.config.head_dim,
                position as u32,
                self.config.rope_theta(),
                self.config.kv_dim(),
                kv_offset,
            )?;
            mha_decode_f32_tcb(
                &mut tcb,
                &self.workspace.q,
                &self.workspace.key_cache,
                0,
                &self.workspace.value_cache,
                0,
                &self.workspace.attention,
                position + 1,
                self.config.head_dim,
                self.config.attention_heads,
                self.config.key_value_heads,
            )?;
            self.dispatch_binary_matvec(
                &mut tcb,
                &o,
                &self.workspace.attention,
                &self.workspace.attention_projection,
                0,
                0,
            )?;
            add_inplace_metal_tcb(
                &mut tcb,
                &self.workspace.x,
                &self.workspace.attention_projection,
                self.config.hidden,
            )?;
            rmsnorm_metal_buf_tcb(
                &mut tcb,
                &self.workspace.x,
                &post_norm,
                self.config.rms_norm_eps(),
                self.config.hidden,
                &self.workspace.x_norm,
            )?;
            self.dispatch_binary_matvec(
                &mut tcb,
                &router,
                &self.workspace.x_norm,
                &self.workspace.router_logits,
                0,
                0,
            )?;
            moe_topk_gate_tcb(
                &mut tcb,
                &self.workspace.router_logits,
                &self.workspace.route_ids,
                &self.workspace.route_weights,
                self.config.experts,
                self.config.experts_per_token,
            )?;
            self.dispatch_normalize_route_weights(&mut tcb)?;
            tcb.commit_and_wait()?;
        }
        let selected_expert_ids = self.route_ids()?;
        let normalized_route_weights = self.route_weights()?;
        let router_input_hidden = self.router_input_hidden()?;
        self.next_position = self.next_position.saturating_add(1);
        Ok(Qwen30Layer0RouterCapture {
            position,
            input_token_id: token,
            selected_expert_ids,
            normalized_route_weights,
            router_input_hidden,
        })
    }

    fn sampled_id(&self) -> Result<u32> {
        let id = unsafe { *(self.workspace.sampled_token.contents() as *const u32) };
        if id as usize >= self.config.vocab_size {
            return Err(model_error(format!(
                "device sampler emitted token id {id} outside vocabulary"
            )));
        }
        Ok(id)
    }

    fn layer_name(layer: usize, suffix: &str) -> String {
        format!("model.layers.{layer}.{suffix}")
    }

    /// Execute one exact full 48-layer greedy token graph using only compact
    /// admitted artifact weights and native Metal operators.  This is the
    /// primitive used by [`generate_greedy`].  It performs no CPU model
    /// fallback and refuses rather than estimating token rate.
    pub fn forward_token_greedy(&mut self, token: u32) -> Result<Qwen30NativeGreedyStep> {
        if token as usize >= self.config.vocab_size {
            return Err(model_error(format!(
                "input token {token} is outside the source vocabulary"
            )));
        }
        if self.next_position >= self.max_seq_len {
            return Err(model_error(format!(
                "native KV cache is full at position {}; reset or use a supported larger max_seq_len",
                self.max_seq_len
            )));
        }
        let started = Instant::now();
        let mut host_stages = Qwen30HostStageRecorder::new(started, self.trace_host_stages);
        let position = self.next_position;
        let mut command_buffers = 0usize;
        let mut metal_dispatches = 0usize;
        let mut fused_gate_up_parity = self
            .gate_up_swiglu_kernel
            .requires_device_control_parity()
            .then(|| Qwen30GateUpSwiGluDeviceParity {
                layers_compared: 0,
                routed_experts_compared: 0,
                activation_values_compared: 0,
                max_abs_error: 0.0,
                tolerance_max_abs: 4.0e-3,
            });
        if self.diagnostic_route_capture_enabled {
            self.diagnostic_selected_expert_ids.clear();
        }
        if self.diagnostic_router_hidden_capture_enabled {
            self.diagnostic_layer_router_captures.clear();
        }

        let serial_encoder = qwen30_serial_encoder_enabled();
        host_stages.measure(
            "embedding",
            "embedding direct-packed lookup plus command-buffer submit/wait",
            || {
                let embedding = self.packed_tensor("model.embed_tokens.weight")?;
                let mut tcb = TokenCommandBuffer::new(&self.context);
                if serial_encoder {
                    tcb.begin_serial_group()?;
                }
                self.dispatch_embedding(&mut tcb, &embedding, token)?;
                if serial_encoder {
                    tcb.end_concurrent_group()?;
                }
                metal_dispatches = metal_dispatches.saturating_add(tcb.dispatch_count());
                tcb.commit_and_wait()?;
                command_buffers = command_buffers.saturating_add(1);
                Ok(())
            },
        )?;

        for layer in 0..self.config.layers {
            host_stages.measure(
                "command_graph_transition_gap",
                format!(
                    "layer {layer} combined norm/QKV/KV/attention/router command graph prepare-submit-wait"
                ),
                || {
                    let input_norm_name = Self::layer_name(layer, "input_layernorm.weight");
                    let post_norm_name = Self::layer_name(layer, "post_attention_layernorm.weight");
                    let q_norm_name = Self::layer_name(layer, "self_attn.q_norm.weight");
                    let k_norm_name = Self::layer_name(layer, "self_attn.k_norm.weight");
                    let q_name = Self::layer_name(layer, "self_attn.q_proj.weight");
                    let k_name = Self::layer_name(layer, "self_attn.k_proj.weight");
                    let v_name = Self::layer_name(layer, "self_attn.v_proj.weight");
                    let o_name = Self::layer_name(layer, "self_attn.o_proj.weight");
                    let router_name = Self::layer_name(layer, "mlp.gate.weight");

                    // Packed weight lookup is host-side catalog/cache only; the
                    // RMSNorm vectors fold any cold decode into the layer TCB
                    // below so a cold first token does not pay 4 dedicated CBs
                    // per layer.
                    let q = self.packed_tensor(&q_name)?;
                    let k = self.packed_tensor(&k_name)?;
                    let v = self.packed_tensor(&v_name)?;
                    let o = self.packed_tensor(&o_name)?;
                    let router = self.packed_tensor(&router_name)?;

                    let layer_cache_elements = layer
                        .checked_mul(self.max_seq_len)
                        .and_then(|value| value.checked_mul(self.config.kv_dim()))
                        .ok_or_else(|| model_error("layer KV cache offset overflows usize"))?;
                    let kv_offset = layer_cache_elements
                        .checked_add(
                            position
                                .checked_mul(self.config.kv_dim())
                                .ok_or_else(|| model_error("position KV cache offset overflows usize"))?,
                        )
                        .ok_or_else(|| model_error("layer-position KV cache offset overflows usize"))?;
                    let cache_offset_bytes =
                        bytes_for_f32(layer_cache_elements, "layer KV cache byte offset")?;

                    // Clone context so folding cold vector decode (`&mut self`)
                    // can coexist with the layer TokenCommandBuffer lifetime.
                    let context = self.context.clone();
                    let mut tcb = TokenCommandBuffer::new(&context);
                    if serial_encoder {
                        tcb.begin_serial_group()?;
                    }
                    let input_norm =
                        self.ensure_decoded_vector_on_tcb(&mut tcb, &input_norm_name, self.config.hidden)?;
                    let post_norm =
                        self.ensure_decoded_vector_on_tcb(&mut tcb, &post_norm_name, self.config.hidden)?;
                    let q_norm =
                        self.ensure_decoded_vector_on_tcb(&mut tcb, &q_norm_name, self.config.head_dim)?;
                    let k_norm =
                        self.ensure_decoded_vector_on_tcb(&mut tcb, &k_norm_name, self.config.head_dim)?;
                    rmsnorm_metal_buf_tcb(
                        &mut tcb,
                        &self.workspace.x,
                        &input_norm,
                        self.config.rms_norm_eps(),
                        self.config.hidden,
                        &self.workspace.x_norm,
                    )?;
                    self.dispatch_binary_matvec(
                        &mut tcb,
                        &q,
                        &self.workspace.x_norm,
                        &self.workspace.q,
                        0,
                        0,
                    )?;
                    self.dispatch_binary_matvec(
                        &mut tcb,
                        &k,
                        &self.workspace.x_norm,
                        &self.workspace.k,
                        0,
                        0,
                    )?;
                    self.dispatch_binary_matvec(
                        &mut tcb,
                        &v,
                        &self.workspace.x_norm,
                        &self.workspace.v,
                        0,
                        0,
                    )?;
                    self.dispatch_rmsnorm_rows(
                        &mut tcb,
                        &self.workspace.q,
                        &q_norm,
                        &self.workspace.q,
                        self.config.attention_heads,
                        self.config.head_dim,
                    )?;
                    self.dispatch_rmsnorm_rows(
                        &mut tcb,
                        &self.workspace.k,
                        &k_norm,
                        &self.workspace.k,
                        self.config.key_value_heads,
                        self.config.head_dim,
                    )?;
                    // Qwen3's `rotate_half` pairing is split-half; the shared
                    // Qwen GQA device kernel matches that layout.  Qwen30 has no
                    // Q/K/V projection biases, so the supplied optional bias
                    // buffers are deliberately absent.
                    rope_qk_kv_append_vbias_f32_tcb(
                        &mut tcb,
                        &self.workspace.q,
                        &self.workspace.k,
                        &self.workspace.v,
                        None,
                        None,
                        None,
                        &self.workspace.key_cache,
                        &self.workspace.value_cache,
                        self.config.attention_heads,
                        self.config.key_value_heads,
                        self.config.head_dim,
                        position as u32,
                        self.config.rope_theta(),
                        self.config.kv_dim(),
                        kv_offset,
                    )?;
                    mha_decode_f32_tcb(
                        &mut tcb,
                        &self.workspace.q,
                        &self.workspace.key_cache,
                        cache_offset_bytes,
                        &self.workspace.value_cache,
                        cache_offset_bytes,
                        &self.workspace.attention,
                        position + 1,
                        self.config.head_dim,
                        self.config.attention_heads,
                        self.config.key_value_heads,
                    )?;
                    self.dispatch_binary_matvec(
                        &mut tcb,
                        &o,
                        &self.workspace.attention,
                        &self.workspace.attention_projection,
                        0,
                        0,
                    )?;
                    add_inplace_metal_tcb(
                        &mut tcb,
                        &self.workspace.x,
                        &self.workspace.attention_projection,
                        self.config.hidden,
                    )?;
                    rmsnorm_metal_buf_tcb(
                        &mut tcb,
                        &self.workspace.x,
                        &post_norm,
                        self.config.rms_norm_eps(),
                        self.config.hidden,
                        &self.workspace.x_norm,
                    )?;
                    self.dispatch_binary_matvec(
                        &mut tcb,
                        &router,
                        &self.workspace.x_norm,
                        &self.workspace.router_logits,
                        0,
                        0,
                    )?;
                    moe_topk_gate_tcb(
                        &mut tcb,
                        &self.workspace.router_logits,
                        &self.workspace.route_ids,
                        &self.workspace.route_weights,
                        self.config.experts,
                        self.config.experts_per_token,
                    )?;
                    self.dispatch_normalize_route_weights(&mut tcb)?;
                    if serial_encoder {
                        tcb.end_concurrent_group()?;
                    }
                    metal_dispatches = metal_dispatches.saturating_add(tcb.dispatch_count());
                    tcb.commit_and_wait()?;
                    command_buffers = command_buffers.saturating_add(1);
                    Ok(())
                },
            )?;

            // The device selects and normalizes routes.  The host only uses
            // the eight ids to bind those exact resident expert tensor slabs;
            // it does not recompute scores, weights, or activations.
            //
            // This host roundtrip is the structural reason the warm path still
            // pays ~2 command buffers per layer: expert weight buffers cannot
            // be bound until the device-written route ids are visible.  See
            // the S-bucket mechanism table (device-indexed expert argument
            // buffers) for the elimination path.
            let route_weights = host_stages.measure(
                "router",
                format!(
                    "layer {layer} device route-id readback and selected direct-packed expert lookup"
                ),
                || {
                    let route_ids = self.route_ids()?;
                    if self.diagnostic_route_capture_enabled {
                        self.diagnostic_selected_expert_ids.push(route_ids);
                    }
                    // Hidden/route capture must happen before the expert wave
                    // reuses x_norm. Observation-only; never fed back.
                    if self.diagnostic_router_hidden_capture_enabled {
                        let router_input_hidden = self.router_input_hidden()?;
                        let normalized_route_weights = self.route_weights()?;
                        self.diagnostic_layer_router_captures
                            .push(Qwen30LayerRouterCapture {
                                layer,
                                selected_expert_ids: route_ids,
                                normalized_route_weights,
                                router_input_hidden,
                            });
                    }
                    let mut route_weights = Vec::with_capacity(self.config.experts_per_token);
                    for &expert in &route_ids {
                        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
                        let down = self.packed_tensor(&format!("{prefix}.down_proj.weight"))?;
                        if self.quality_sparse_gate_up_applies(layer, expert) {
                            route_weights.push(Qwen30RoutedExpertWeights::QualitySparseGateUp {
                                down,
                            });
                        } else {
                            route_weights.push(Qwen30RoutedExpertWeights::Direct {
                                gate: self.packed_tensor(&format!("{prefix}.gate_proj.weight"))?,
                                up: self.packed_tensor(&format!("{prefix}.up_proj.weight"))?,
                                down,
                            });
                        }
                    }
                    Ok(route_weights)
                },
            )?;
            host_stages.measure(
                "command_graph_transition_gap",
                format!(
                    "layer {layer} selected-expert gate/up/down/combine wave command graph submit/wait"
                ),
                || {
                    let mut tcb = TokenCommandBuffer::new(&self.context);
                    if serial_encoder {
                        tcb.begin_serial_group()?;
                    }
                    for (route, weights) in route_weights.iter().enumerate() {
                        let mid_offset = bytes_for_f32(
                            route
                                .checked_mul(self.config.moe_intermediate)
                                .ok_or_else(|| {
                                    model_error("expert route intermediate offset overflows usize")
                                })?,
                            "expert route intermediate offset",
                        )?;
                        let hidden_offset = bytes_for_f32(
                            route.checked_mul(self.config.hidden).ok_or_else(|| {
                                model_error("expert route hidden offset overflows usize")
                            })?,
                            "expert route hidden offset",
                        )?;
                        let down = match weights {
                            Qwen30RoutedExpertWeights::Direct { gate, up, down } => {
                                self.dispatch_expert_gate_up_swiglu(&mut tcb, gate, up, mid_offset)?;
                                down
                            }
                            Qwen30RoutedExpertWeights::QualitySparseGateUp { down } => {
                                self.dispatch_quality_sparse_gate_up(&mut tcb, mid_offset)?;
                                down
                            }
                        };
                        self.dispatch_binary_matvec(
                            &mut tcb,
                            down,
                            &self.workspace.expert_activation,
                            &self.workspace.expert_output,
                            mid_offset,
                            hidden_offset,
                        )?;
                    }
                    self.dispatch_weighted_expert_add(&mut tcb)?;
                    if serial_encoder {
                        tcb.end_concurrent_group()?;
                    }
                    metal_dispatches = metal_dispatches.saturating_add(tcb.dispatch_count());
                    tcb.commit_and_wait()?;
                    command_buffers = command_buffers.saturating_add(1);
                    Ok(())
                },
            )?;
            if let Some(cumulative) = fused_gate_up_parity.as_mut() {
                host_stages.measure(
                    "command_graph_transition_gap",
                    format!("layer {layer} fused gate/up device-control parity readback"),
                    || {
                        let observed = self.compare_fused_gate_up_device_control(layer)?;
                        cumulative.layers_compared = cumulative
                            .layers_compared
                            .saturating_add(observed.layers_compared);
                        cumulative.routed_experts_compared = cumulative
                            .routed_experts_compared
                            .saturating_add(observed.routed_experts_compared);
                        cumulative.activation_values_compared = cumulative
                            .activation_values_compared
                            .saturating_add(observed.activation_values_compared);
                        cumulative.max_abs_error =
                            cumulative.max_abs_error.max(observed.max_abs_error);
                        if observed.tolerance_max_abs.to_bits()
                            != cumulative.tolerance_max_abs.to_bits()
                        {
                            return Err(model_error(
                                "fused gate/up parity tolerance changed within one native token",
                            ));
                        }
                        Ok(())
                    },
                )?;
            }
        }

        host_stages.measure(
            "final_head",
            "final norm/lm-head/finite guard/argmax command graph submit/wait",
            || {
                let lm_head = self.packed_tensor("lm_head.weight")?;
                MetalContext::write_buffer_bytes(
                    &self.workspace.invalid_f32_flag,
                    &0u32.to_ne_bytes(),
                );
                let context = self.context.clone();
                let mut tcb = TokenCommandBuffer::new(&context);
                if serial_encoder {
                    tcb.begin_serial_group()?;
                }
                let final_norm = self.ensure_decoded_vector_on_tcb(
                    &mut tcb,
                    "model.norm.weight",
                    self.config.hidden,
                )?;
                rmsnorm_metal_buf_tcb(
                    &mut tcb,
                    &self.workspace.x,
                    &final_norm,
                    self.config.rms_norm_eps(),
                    self.config.hidden,
                    &self.workspace.x_norm,
                )?;
                self.dispatch_binary_matvec(
                    &mut tcb,
                    &lm_head,
                    &self.workspace.x_norm,
                    &self.workspace.final_logits,
                    0,
                    0,
                )?;
                self.dispatch_finite_check(
                    &mut tcb,
                    &self.workspace.final_logits,
                    self.config.vocab_size,
                )?;
                crate::kernels::sample_argmax_f32_tcb(
                    &mut tcb,
                    &self.workspace.final_logits,
                    &self.workspace.sampled_token,
                    self.config.vocab_size,
                )?;
                if serial_encoder {
                    tcb.end_concurrent_group()?;
                }
                metal_dispatches = metal_dispatches.saturating_add(tcb.dispatch_count());
                tcb.commit_and_wait()?;
                command_buffers = command_buffers.saturating_add(1);
                Ok(())
            },
        )?;
        let token_id = host_stages.measure(
            "sampling",
            "final finite flag and sampled-token device readback",
            || {
                self.assert_final_logits_finite()?;
                self.sampled_id()
            },
        )?;
        host_stages.measure("sampling", "advance native autoregressive position", || {
            self.next_position = self.next_position.saturating_add(1);
            Ok(())
        })?;
        let elapsed = started.elapsed();
        let host_stage_intervals = host_stages.finish(elapsed);
        Ok(Qwen30NativeGreedyStep {
            position,
            token_id,
            elapsed,
            command_buffers,
            metal_dispatches,
            host_route_id_readbacks: self.config.layers,
            host_sample_id_readbacks: 1,
            gate_up_swiglu_device_control_parity: fused_gate_up_parity,
            host_stage_intervals,
        })
    }

    /// Run prompt prefill followed by a native greedy autoregressive loop.
    /// Greedy sampling is deliberate: it keeps the full 151,936-logit vector
    /// on device and reads only the sampled id.  Temperature/top-k/top-p HCLI
    /// sampling is a separate device sampler implementation and is not
    /// silently delegated to the CPU here.
    pub fn generate_greedy(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<Qwen30NativeGeneration> {
        let prompt_token_ids = self.tokenizer.encode(prompt, true)?;
        if prompt_token_ids.is_empty() {
            return Err(model_error("prompt tokenization produced no tokens"));
        }
        if prompt_token_ids.len().saturating_add(max_new_tokens) > self.max_seq_len {
            return Err(model_error(format!(
                "prompt length {} + max_new_tokens {} exceeds native context {}",
                prompt_token_ids.len(),
                max_new_tokens,
                self.max_seq_len
            )));
        }
        self.reset();
        let mut next = 0u32;
        let mut prefill_steps = Vec::with_capacity(prompt_token_ids.len());
        for &token in &prompt_token_ids {
            let step = self.forward_token_greedy(token)?;
            next = step.token_id;
            prefill_steps.push(step);
        }
        let mut completion_token_ids = Vec::with_capacity(max_new_tokens);
        let mut steps = Vec::with_capacity(max_new_tokens);
        let mut ended_on_eog = false;
        for _ in 0..max_new_tokens {
            let emitted = next;
            if emitted as usize >= self.tokenizer_addressable_vocab {
                return Err(model_error(format!(
                    "device sampler emitted model token {emitted} in the LM-head reserved tail; source tokenizer only addresses 0..{} and no token remapping is permitted",
                    self.tokenizer_addressable_vocab.saturating_sub(1)
                )));
            }
            completion_token_ids.push(emitted);
            if self.tokenizer.is_eog(emitted) {
                ended_on_eog = true;
                break;
            }
            let step = self.forward_token_greedy(emitted)?;
            next = step.token_id;
            steps.push(step);
        }
        let completion_text = self.tokenizer.decode(&completion_token_ids, true)?;
        Ok(Qwen30NativeGeneration {
            prompt_token_ids,
            completion_token_ids,
            completion_text,
            ended_on_eog,
            prefill_steps,
            steps,
        })
    }

    /// Generate through the exact validated one-user source chat-template
    /// branch.  The forward path remains the same direct packed Metal model;
    /// only prompt formatting differs from the raw-text diagnostic helper.
    pub fn generate_source_user_chat_greedy(
        &mut self,
        user_content: &str,
        max_new_tokens: usize,
    ) -> Result<Qwen30NativeGeneration> {
        let rendered = self.render_source_user_chat_prompt(user_content);
        self.generate_greedy(&rendered, max_new_tokens)
    }
}

/// Separately admitted HQ30GR2 all-layer diagnostic runtime.
///
/// This is intentionally not a serving type: it exposes exact token forwards
/// and device-logit snapshots for one bounded source-template experiment, but
/// no HTTP/HCLI adapter or general text-generation helper.  It keeps the
/// ordinary direct Qwen30 runtime unmodified in production: the sparse
/// representation is selected only by this explicit constructor after a full
/// candidate admission and typed L0/E0 gate/up device upload.
#[cfg(target_os = "macos")]
pub struct Qwen30QualityRepackNativeDiagnosticRuntime {
    inner: Qwen30CompleteNativeRuntime,
    /// Retained solely for the device-parity CPU oracle.  It owns the exact
    /// immutable HQ30GR2 admission snapshots; it is never exposed to serving
    /// and no token path reads a candidate file from it.
    catalog: Qwen30QualityRepackDiagnosticCatalog,
    sparse_gate_up_dispatch: Qwen30QualityRepackSparseGateUpDispatch,
}

#[cfg(target_os = "macos")]
impl Qwen30QualityRepackNativeDiagnosticRuntime {
    /// Re-admit the exact HQ30GR2 candidate and construct the isolated
    /// all-layer diagnostic graph.  This must be called only from a future
    /// explicitly leased candidate diagnostic process; no ordinary server or
    /// runtime watcher calls it.
    pub fn load(
        manifest_path: impl AsRef<Path>,
        admission: &Qwen30QualityRepackAdmission,
        options: Qwen30CompleteRuntimeOptions,
    ) -> Result<Self> {
        if options.packed_matvec_kernel != Qwen30PackedMatvecKernel::ScalarControl
            || options.gate_up_swiglu_kernel != Qwen30GateUpSwiGluKernel::ThreeDispatchControl
        {
            return Err(model_error(
                "HQ30GR2 all-layer diagnostic requires the scalar direct control topology for every unchanged organ",
            ));
        }
        let candidate = admit_qwen30_quality_repack_artifact(manifest_path, admission)?;
        let catalog = Qwen30QualityRepackDiagnosticCatalog::from_admitted(candidate)?;
        let sparse_gate_up_dispatch = catalog.sparse_gate_up_dispatch()?;
        let direct_base = catalog.direct_base_view_for_diagnostic_runtime()?;
        let mut inner =
            Qwen30CompleteNativeRuntime::from_admitted_direct_artifact(direct_base, options)?;
        let pair = Qwen30QualityRepackSparseGateUpDevicePair::upload(&inner.context, &catalog)?;
        if pair.specification() != &sparse_gate_up_dispatch {
            return Err(model_error(
                "HQ30GR2 sparse gate/up device upload differs from its typed dispatch contract",
            ));
        }
        inner.quality_sparse_gate_up = Some(pair);
        Ok(Self {
            inner,
            catalog,
            sparse_gate_up_dispatch,
        })
    }

    pub fn artifact_manifest_seal(&self) -> &str {
        self.inner.artifact_manifest_seal()
    }

    pub fn sparse_gate_up_dispatch(&self) -> &Qwen30QualityRepackSparseGateUpDispatch {
        &self.sparse_gate_up_dispatch
    }

    /// Number of actual candidate sparse gate/up encodes since the last
    /// diagnostic reset.  This is a structural witness only: it makes no
    /// quality, coherence, performance, or serving claim.
    pub fn sparse_gate_up_interception_count(&self) -> usize {
        self.inner.quality_sparse_gate_up_interception_count.get()
    }

    pub fn source_user_chat_template(&self) -> &Qwen30SourceUserChatTemplate {
        self.inner.source_user_chat_template()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn position(&self) -> usize {
        self.inner.position()
    }

    /// Exact-format CPU reference for one typed L0/E0 gate/up/SwiGLU vector.
    /// It is provided only so a separately leased device process can prove
    /// the sparse dispatch before it permits an all-layer candidate forward.
    /// It is not called by, or available through, any production/server path.
    pub fn sparse_gate_up_cpu_oracle_f64(&self, input: &[f64]) -> Result<Vec<f64>> {
        self.catalog.sparse_gate_up_cpu_oracle_f64(input)
    }

    /// Execute precisely the selected sparse L0/E0 gate/up/SwiGLU pair on
    /// Metal for a supplied diagnostic input.  This does not execute an
    /// attention block, expert down projection, residual, layer, logits, or
    /// sampler.  Callers must establish CPU/device parity from this output
    /// before calling [`Self::forward_token_diagnostic`].
    pub fn sparse_gate_up_device_pair_for_input_diagnostic(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let specification = &self.sparse_gate_up_dispatch;
        if input.len() != specification.cols {
            return Err(model_error(format!(
                "HQ30GR2 sparse gate/up device input has {} values, expected {}",
                input.len(),
                specification.cols
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "HQ30GR2 sparse gate/up device input contains a non-finite value",
            ));
        }
        let byte_len = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                model_error("HQ30GR2 sparse gate/up input byte count overflows usize")
            })?;
        let bytes = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, byte_len) };
        MetalContext::write_buffer_bytes(&self.inner.workspace.x_norm, bytes);
        let mut tcb = TokenCommandBuffer::new(&self.inner.context);
        self.inner.dispatch_quality_sparse_gate_up(&mut tcb, 0)?;
        tcb.commit_and_wait()?;
        let output = unsafe {
            std::slice::from_raw_parts(
                self.inner.workspace.expert_activation.contents() as *const f32,
                specification.rows,
            )
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "HQ30GR2 sparse gate/up device output contains a non-finite value",
            ));
        }
        Ok(output.to_vec())
    }

    /// One complete native 48-layer candidate forward.  It is deliberately
    /// named diagnostic rather than generation: callers must supply the exact
    /// sealed input IDs and must not expose its sampled ID as chat output.
    pub fn forward_token_diagnostic(&mut self, token: u32) -> Result<Qwen30NativeGreedyStep> {
        self.inner.forward_token_greedy(token)
    }

    /// The candidate equivalent of the control's opt-in route capture.  It
    /// remains non-serving and exposes only device-selected IDs; it never
    /// offers a CPU router or alternate generation path.
    pub fn forward_token_diagnostic_with_route_capture(
        &mut self,
        token: u32,
    ) -> Result<Qwen30NativeRouteCaptureStep> {
        self.inner.forward_token_greedy_with_route_capture(token)
    }

    /// Device-produced final logits from the most recent diagnostic forward.
    /// This is a bounded observation surface, not an endpoint result.
    pub fn diagnostic_final_logits_f32(&self) -> Result<Vec<f32>> {
        self.inner.diagnostic_final_logits_f32()
    }

    pub fn drain_profiler(&self) -> Qwen30NativeProfilerSnapshot {
        self.inner.drain_profiler()
    }
}

#[cfg(not(target_os = "macos"))]
pub struct Qwen30CompleteNativeRuntime;

#[cfg(not(target_os = "macos"))]
impl Qwen30CompleteNativeRuntime {
    pub fn load(
        _manifest_path: impl AsRef<Path>,
        _admission: &CompleteBinaryAdmission,
        _options: Qwen30CompleteRuntimeOptions,
    ) -> Result<Self> {
        Err(Error::Metal(
            "Qwen30 complete native runtime is Metal-only and requires macOS".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn source_config() -> Value {
        json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_hidden_layers": 48,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "vocab_size": 151936,
            "hidden_act": "silu",
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "rope_scaling": null,
            "use_sliding_window": false,
            "mlp_only_layers": [],
            "rope_theta": 10000000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
        })
    }

    #[test]
    fn exact_qwen30_source_config_is_required() {
        let config = Qwen30CompleteRuntimeConfig::from_source_config(
            &source_config(),
            QWEN30_REPOSITORY,
            "pinned-revision",
        )
        .unwrap();
        assert_eq!(config.layers, 48);
        assert_eq!(config.q_dim(), 4096);
        assert_eq!(config.kv_dim(), 512);
        let mut bad = source_config();
        bad["num_experts_per_tok"] = json!(10);
        assert!(Qwen30CompleteRuntimeConfig::from_source_config(
            &bad,
            QWEN30_REPOSITORY,
            "pinned-revision"
        )
        .is_err());
        let mut no_norm_topk = source_config();
        no_norm_topk["norm_topk_prob"] = json!(false);
        assert!(Qwen30CompleteRuntimeConfig::from_source_config(
            &no_norm_topk,
            QWEN30_REPOSITORY,
            "pinned-revision"
        )
        .is_err());
    }

    #[test]
    fn fused_gate_up_candidate_is_explicit_and_default_runtime_keeps_control() {
        let options = Qwen30CompleteRuntimeOptions::default();
        assert_eq!(
            options.gate_up_swiglu_kernel,
            Qwen30GateUpSwiGluKernel::ThreeDispatchControl
        );
        assert_eq!(
            Qwen30GateUpSwiGluKernel::ThreeDispatchControl.receipt_name(),
            "three_dispatch_direct_packed_gate_up_swiglu_control"
        );
        assert_eq!(
            Qwen30GateUpSwiGluKernel::FusedCandidate.receipt_name(),
            "fused_direct_packed_gate_up_swiglu_candidate"
        );
        assert_eq!(
            Qwen30GateUpSwiGluKernel::FusedCandidateWithDeviceControlParity.receipt_name(),
            "fused_direct_packed_gate_up_swiglu_candidate_with_device_control_parity"
        );
        assert_eq!(
            Qwen30GateUpSwiGluKernel::PairedScalarOrderCandidateWithDeviceControlParity
                .receipt_name(),
            "paired_direct_packed_gate_up_swiglu_scalar_order_candidate_with_device_control_parity"
        );
        assert_eq!(
            Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity.receipt_name(),
            "paired_direct_packed_gate_up_swiglu_scalar_order_production_no_parity"
        );
        assert!(!Qwen30GateUpSwiGluKernel::ThreeDispatchControl.requires_device_control_parity());
        assert!(!Qwen30GateUpSwiGluKernel::FusedCandidate.requires_device_control_parity());
        assert!(
            Qwen30GateUpSwiGluKernel::FusedCandidateWithDeviceControlParity
                .requires_device_control_parity()
        );
        assert!(
            Qwen30GateUpSwiGluKernel::PairedScalarOrderCandidateWithDeviceControlParity
                .requires_device_control_parity()
        );
        assert!(
            !Qwen30GateUpSwiGluKernel::PairedScalarOrderProductionNoParity
                .requires_device_control_parity()
        );
    }

    #[test]
    fn fused_gate_up_candidate_kernel_is_embedded_in_the_native_metal_library() {
        let source = crate::metal::all_shader_sources();
        assert!(source.contains("kernel void qwen_direct_packed_gate_up_swiglu_fused_candidate("));
    }

    #[test]
    fn paired_scalar_order_candidate_is_separate_and_forbids_explicit_fma() {
        let source = crate::metal::SHADER_QWEN_DIRECT_PACKED_GATE_UP_SWIGLU_PAIRED_SCALAR_ORDER;
        assert!(source.contains(
            "kernel void qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate("
        ));
        assert!(source.contains("#pragma clang fp contract(off)"));
        assert!(source.contains("#pragma clang fp reassociate(off)"));
        assert!(source.contains("gate_sum = gate_sum + gate_product"));
        assert!(!source.contains("fma("));
    }

    #[test]
    fn expected_tensor_catalog_is_complete_and_exact_count() {
        let tensors = tensor_shapes();
        assert_eq!(tensors.len(), 18_867);
        assert_eq!(
            tensors.get("model.layers.47.mlp.experts.127.down_proj.weight"),
            Some(&vec![2048, 768])
        );
        assert_eq!(
            tensors.get("model.layers.0.self_attn.q_proj.weight"),
            Some(&vec![4096, 2048])
        );
        assert_eq!(tensors.get("lm_head.weight"), Some(&vec![151_936, 2048]));
    }

    #[test]
    fn native_context_limit_is_explicitly_below_source_limit() {
        let config = Qwen30CompleteRuntimeConfig::from_source_config(
            &source_config(),
            QWEN30_REPOSITORY,
            "pinned-revision",
        )
        .unwrap();
        assert!(QWEN30_COMPLETE_NATIVE_MAX_CONTEXT < config.source_max_position_embeddings);
        assert_eq!(Qwen30CompleteRuntimeOptions::default().max_seq_len, 256);
    }

    #[test]
    fn host_stage_ledger_closes_serial_gaps_without_overlap() {
        let mut recorder = Qwen30HostStageRecorder {
            started: Instant::now(),
            enabled: true,
            last_end_us: 0,
            intervals: Vec::new(),
        };
        recorder.append_gap_until(5, "embedding");
        recorder.intervals.push(Qwen30HostStageInterval {
            bucket: "embedding".to_string(),
            label: "exact embedding span".to_string(),
            start_us: 5,
            end_us: 12,
        });
        recorder.last_end_us = 12;
        recorder.append_gap_until(15, "final head");
        let intervals = recorder.finish(Duration::from_micros(20));
        assert_eq!(intervals.first().map(|interval| interval.start_us), Some(0));
        assert_eq!(intervals.last().map(|interval| interval.end_us), Some(20));
        assert!(intervals
            .windows(2)
            .all(|pair| pair[0].end_us <= pair[1].start_us));
        assert!(intervals.iter().any(|interval| {
            interval.bucket == "command_graph_transition_gap"
                && interval.label.contains("serial host setup/scheduling gap")
        }));
    }

    #[test]
    fn bounded_source_user_chat_template_renders_exact_user_branch() {
        assert_eq!(
            render_source_user_chat_template("write one line"),
            "<|im_start|>user\nwrite one line<|im_end|>\n<|im_start|>assistant\n"
        );
        // The source Jinja user branch does not escape a message body. This
        // regression guard keeps the native limited renderer byte-for-byte
        // aligned with that declared behavior rather than adding a guessed
        // escape layer.
        assert_eq!(
            render_source_user_chat_template("a\nb"),
            "<|im_start|>user\na\nb<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn direct_binary_vector_decode_and_finite_guard_execute_on_metal() {
        use half::f16;

        let context = MetalContext::new().expect("native Metal context");
        // The fixed group format always retains 128 sign bits, even when a
        // logical vector has fewer elements. The first four little-endian
        // bits encode +, -, +, - at scale 2.5.
        let mut signs = vec![0u8; QWEN30_GROUP_SIZE / 8];
        signs[0] = 0b0000_0101;
        let scale = f16::from_f32(2.5).to_bits().to_le_bytes();
        let signs = context
            .new_buffer_with_bytes_checked(&signs)
            .expect("packed signs buffer");
        let scales = context
            .new_buffer_with_bytes_checked(&scale)
            .expect("packed scales buffer");
        let output = context
            .new_buffer_checked(4 * std::mem::size_of::<f32>())
            .expect("decoded vector buffer");
        let invalid = context
            .new_buffer_checked(std::mem::size_of::<u32>())
            .expect("finite flag buffer");
        MetalContext::write_buffer_bytes(&invalid, &0u32.to_ne_bytes());

        let mut tcb = TokenCommandBuffer::new(&context);
        tcb.dispatch_threads(
            "qwen_complete_binary_decode_vector",
            (4, 1, 1),
            (4, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&signs), 0);
                encoder.set_buffer(1, Some(&scales), 0);
                encoder.set_buffer(2, Some(&output), 0);
                encoder.qwen_set_u32(3, 4);
                encoder.qwen_set_u32(4, QWEN30_GROUP_SIZE as u32);
            },
        )
        .expect("direct packed decode dispatch");
        tcb.dispatch_threads(
            "qwen_complete_any_nonfinite_f32",
            (4, 1, 1),
            (4, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&output), 0);
                encoder.set_buffer(1, Some(&invalid), 0);
                encoder.qwen_set_u32(2, 4);
            },
        )
        .expect("finite guard dispatch");
        tcb.commit_and_wait()
            .expect("native Metal command completion");

        let decoded = unsafe { std::slice::from_raw_parts(output.contents() as *const f32, 4) };
        assert_eq!(decoded, &[2.5, -2.5, 2.5, -2.5]);
        assert_eq!(unsafe { *(invalid.contents() as *const u32) }, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn direct_packed_matvec_honors_route_major_input_buffer_offset() {
        use half::f16;

        // Two adjacent four-value activation slices model the first two
        // route-major expert slots. The direct-packed matvec must consume the
        // second slice when its buffer binding carries the 4-f32 byte offset;
        // silently binding offset zero would reproduce the former MoE defect.
        let context = MetalContext::new().expect("native Metal context");
        let mut signs = vec![0u8; QWEN30_GROUP_SIZE / 8];
        signs[0] = 0b0000_0101; // +, -, +, - at scale 1.0
        let scales = f16::from_f32(1.0).to_bits().to_le_bytes();
        let route_major_input = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let signs_buffer = context
            .new_buffer_with_bytes_checked(&signs)
            .expect("packed signs buffer");
        let scales_buffer = context
            .new_buffer_with_bytes_checked(&scales)
            .expect("packed scales buffer");
        let input_buffer = context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&route_major_input))
            .expect("route-major input buffer");
        let output = context
            .new_buffer_checked(std::mem::size_of::<f32>())
            .expect("matvec output");
        let mut tcb = TokenCommandBuffer::new(&context);
        tcb.dispatch_threads(
            "qwen_binary_sign_scale_matvec",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&signs_buffer), 0);
                encoder.set_buffer(1, Some(&scales_buffer), 0);
                encoder.set_buffer(
                    2,
                    Some(&input_buffer),
                    (4 * std::mem::size_of::<f32>()) as u64,
                );
                encoder.set_buffer(3, Some(&output), 0);
                encoder.qwen_set_u32(4, 1);
                encoder.qwen_set_u32(5, 4);
                encoder.qwen_set_u32(6, QWEN30_GROUP_SIZE as u32);
                encoder.qwen_set_u32(7, 1);
            },
        )
        .expect("direct packed route-major matvec dispatch");
        tcb.commit_and_wait()
            .expect("direct packed route-major matvec completion");
        let observed = unsafe { *(output.contents() as *const f32) };
        assert!(
            (observed - -20.0).abs() <= 1.0e-6,
            "route-major input offset was not honored: observed={observed} expected=-20"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn binary_simdgroup_candidate_matches_scalar_and_packed_cpu_oracle() {
        use half::f16;

        // Nine rows deliberately cross the eight-rows-per-threadgroup tail;
        // three 128-value groups exercise the fixed admitted Qwen geometry.
        let (rows, cols, group_size) = (9usize, 384usize, QWEN30_GROUP_SIZE);
        let groups_per_row = cols.div_ceil(group_size);
        let mut signs = vec![0u8; rows * cols / 8];
        for flat in 0..rows * cols {
            if (flat * 17 + 11) % 5 >= 2 {
                signs[flat >> 3] |= 1u8 << (flat & 7);
            }
        }
        let scale_bits = (0..rows * groups_per_row)
            .map(|index| f16::from_f32(0.015625 * (1 + index) as f32).to_bits())
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| ((index * 37 % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; rows];
        for row in 0..rows {
            for col in 0..cols {
                let flat = row * cols + col;
                let scale =
                    f16::from_bits(scale_bits[row * groups_per_row + col / group_size]).to_f32();
                expected[row] += if ((signs[flat >> 3] >> (flat & 7)) & 1) != 0 {
                    scale * input[col]
                } else {
                    -scale * input[col]
                };
            }
        }

        let context = MetalContext::new().expect("native Metal context");
        let signs_buffer = context
            .new_buffer_with_bytes_checked(&signs)
            .expect("packed signs buffer");
        let scales_buffer = context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&scale_bits))
            .expect("packed scale buffer");
        let input_buffer = context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&input))
            .expect("input buffer");
        let scalar_output = context
            .new_buffer_checked(rows * std::mem::size_of::<f32>())
            .expect("scalar output buffer");
        let simdgroup_output = context
            .new_buffer_checked(rows * std::mem::size_of::<f32>())
            .expect("simdgroup output buffer");
        let mut tcb = TokenCommandBuffer::new(&context);
        tcb.dispatch_threads(
            "qwen_binary_sign_scale_matvec",
            (rows as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&signs_buffer), 0);
                encoder.set_buffer(1, Some(&scales_buffer), 0);
                encoder.set_buffer(2, Some(&input_buffer), 0);
                encoder.set_buffer(3, Some(&scalar_output), 0);
                encoder.qwen_set_u32(4, rows as u32);
                encoder.qwen_set_u32(5, cols as u32);
                encoder.qwen_set_u32(6, group_size as u32);
                encoder.qwen_set_u32(7, groups_per_row as u32);
            },
        )
        .expect("scalar packed binary dispatch");
        tcb.dispatch_threads(
            "qwen_binary_sign_scale_matvec_simdgroup_candidate",
            (rows.div_ceil(8).saturating_mul(256) as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&signs_buffer), 0);
                encoder.set_buffer(1, Some(&scales_buffer), 0);
                encoder.set_buffer(2, Some(&input_buffer), 0);
                encoder.set_buffer(3, Some(&simdgroup_output), 0);
                encoder.qwen_set_u32(4, rows as u32);
                encoder.qwen_set_u32(5, cols as u32);
                encoder.qwen_set_u32(6, group_size as u32);
                encoder.qwen_set_u32(7, groups_per_row as u32);
            },
        )
        .expect("simdgroup packed binary dispatch");
        tcb.commit_and_wait()
            .expect("packed binary candidate command completion");

        let scalar =
            unsafe { std::slice::from_raw_parts(scalar_output.contents() as *const f32, rows) };
        let simdgroup =
            unsafe { std::slice::from_raw_parts(simdgroup_output.contents() as *const f32, rows) };
        for row in 0..rows {
            let scalar_error = (scalar[row] - expected[row]).abs();
            let simdgroup_error = (simdgroup[row] - expected[row]).abs();
            let candidate_delta = (simdgroup[row] - scalar[row]).abs();
            assert!(
                scalar_error <= 3e-5 && simdgroup_error <= 3e-5 && candidate_delta <= 3e-5,
                "row {row}: cpu={}, scalar={}, simdgroup={}, scalar_error={scalar_error}, simdgroup_error={simdgroup_error}, candidate_delta={candidate_delta}",
                expected[row], scalar[row], simdgroup[row],
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn static_decoded_vector_catalog_matches_cold_token_vector_count() {
        // 48 layers × (input_ln, post_ln, q_norm, k_norm) + final norm = 193.
        // The sealed complete-token profile's 291 CBs = 193 cold vector-decode
        // CBs + 98 structural graph CBs; this catalog is the first term.
        let expected = QWEN30_LAYERS * 4 + 1;
        assert_eq!(expected, 193);
        // Shape of the public report type is part of the falsifier surface.
        let report = Qwen30StaticDecodePrewarmReport {
            catalog_vectors: expected,
            already_resident: 0,
            decoded_now: expected,
            dispatches: expected,
            command_buffers: 1,
            serial_encoder: true,
        };
        assert_eq!(report.catalog_vectors, 193);
        assert_eq!(report.command_buffers, 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn serial_encoder_env_opt_out_is_recognized() {
        // Default-on, explicit off for A/B against the historical per-dispatch
        // encoder shape.  Restoring the prior env value keeps the test
        // hermetic under parallel cargo test runners that share the process
        // environment only within a single binary.
        let previous = std::env::var("HAWKING_QWEN30_SERIAL_ENCODER").ok();
        std::env::remove_var("HAWKING_QWEN30_SERIAL_ENCODER");
        assert!(qwen30_serial_encoder_enabled());
        for off in ["0", "false", "OFF", "no"] {
            std::env::set_var("HAWKING_QWEN30_SERIAL_ENCODER", off);
            assert!(
                !qwen30_serial_encoder_enabled(),
                "expected serial encoder disabled for {off:?}"
            );
        }
        std::env::set_var("HAWKING_QWEN30_SERIAL_ENCODER", "1");
        assert!(qwen30_serial_encoder_enabled());
        match previous {
            Some(value) => std::env::set_var("HAWKING_QWEN30_SERIAL_ENCODER", value),
            None => std::env::remove_var("HAWKING_QWEN30_SERIAL_ENCODER"),
        }
    }

    /// Component-only topology microbench: multi-CB vs multi-encoder-in-one-CB
    /// vs one serial encoder.  Does **not** claim clean TPS and does not take
    /// an exclusive GPU lease; it only proves bit-identity and prints wall
    /// times so a human can rank the three shapes against the S-bucket
    /// hypothesis without a full Q30 token.
    #[cfg(target_os = "macos")]
    #[test]
    fn component_command_topology_serial_vs_split_encoder_and_multi_cb() {
        use half::f16;

        const N: usize = 64;
        const ELEMENTS: u32 = 256;

        let context = match MetalContext::new() {
            Ok(context) => context,
            Err(error) => {
                // Sandboxed CI / seats without a Metal device must not fail
                // the compile-gated suite; the human gate profile runs this
                // as a real component microbench.
                eprintln!(
                    "component_only command topology skipped (no Metal device): {error}"
                );
                return;
            }
        };
        let mut signs = vec![0u8; QWEN30_GROUP_SIZE / 8];
        signs[0] = 0b0000_0101;
        let scale = f16::from_f32(1.0).to_bits().to_le_bytes();
        let signs_buf = context
            .new_buffer_with_bytes_checked(&signs)
            .expect("signs");
        let scales_buf = context
            .new_buffer_with_bytes_checked(&scale)
            .expect("scales");
        let mut outputs = Vec::with_capacity(N);
        for _ in 0..N {
            outputs.push(
                context
                    .new_buffer_checked(ELEMENTS as usize * std::mem::size_of::<f32>())
                    .expect("output"),
            );
        }

        let encode_one = |tcb: &mut TokenCommandBuffer<'_>, out: &PinnedBuffer| {
            tcb.dispatch_threads(
                "qwen_complete_binary_decode_vector",
                (ELEMENTS, 1, 1),
                (ELEMENTS.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&signs_buf), 0);
                    encoder.set_buffer(1, Some(&scales_buf), 0);
                    encoder.set_buffer(2, Some(out), 0);
                    encoder.qwen_set_u32(3, ELEMENTS);
                    encoder.qwen_set_u32(4, QWEN30_GROUP_SIZE as u32);
                },
            )
            .expect("decode dispatch");
        };

        // Warm the pipeline once so the measured passes are not dominated by
        // first-shader compile.
        {
            let mut warm = TokenCommandBuffer::new(&context);
            encode_one(&mut warm, &outputs[0]);
            warm.commit_and_wait().expect("warm");
        }

        let zero_all = || {
            let zeros = vec![0u8; ELEMENTS as usize * std::mem::size_of::<f32>()];
            for out in &outputs {
                MetalContext::write_buffer_bytes(out, &zeros);
            }
        };

        let head4 = |buf: &PinnedBuffer| -> [f32; 4] {
            let slice = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, 4) };
            [slice[0], slice[1], slice[2], slice[3]]
        };
        let assert_all_match = |label: &str, expected: [f32; 4]| {
            for (index, out) in outputs.iter().enumerate() {
                let observed = head4(out);
                assert_eq!(
                    observed, expected,
                    "{label}: topology shape diverged at buffer {index}"
                );
            }
        };
        let expected_values = [1.0f32, -1.0, 1.0, -1.0];

        // Shape A: N command buffers, one dispatch each.
        zero_all();
        let t0 = Instant::now();
        for out in &outputs {
            let mut tcb = TokenCommandBuffer::new(&context);
            encode_one(&mut tcb, out);
            tcb.commit_and_wait().expect("multi-cb wait");
        }
        let multi_cb_us = t0.elapsed().as_micros();
        assert_all_match("multi_cb", expected_values);

        // Shape B: one CB, N separate encoders (historical TCB default).
        zero_all();
        let t0 = Instant::now();
        {
            let mut tcb = TokenCommandBuffer::new(&context);
            for out in &outputs {
                encode_one(&mut tcb, out);
            }
            tcb.commit_and_wait().expect("multi-encoder wait");
        }
        let multi_encoder_us = t0.elapsed().as_micros();
        assert_all_match("multi_encoder", expected_values);

        // Shape C: one CB, one serial encoder, N dispatches.
        zero_all();
        let t0 = Instant::now();
        {
            let mut tcb = TokenCommandBuffer::new(&context);
            tcb.begin_serial_group().expect("serial group");
            for out in &outputs {
                encode_one(&mut tcb, out);
            }
            tcb.end_concurrent_group().expect("end serial group");
            tcb.commit_and_wait().expect("serial wait");
        }
        let serial_us = t0.elapsed().as_micros();
        assert_all_match("serial_encoder", expected_values);

        // Component-only numbers for the human report.  Not a gate.
        eprintln!(
            "component_only command topology N={N}: multi_cb_us={multi_cb_us} multi_encoder_one_cb_us={multi_encoder_us} serial_one_encoder_us={serial_us}"
        );
        // All three completed and matched the packed oracle.  We do not assert
        // a speedup here — ranking is the human's A/B against the S bucket.
        assert!(multi_cb_us > 0 && multi_encoder_us > 0 && serial_us > 0);
    }
}
