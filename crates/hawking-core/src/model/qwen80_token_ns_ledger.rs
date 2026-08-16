//! Per-decode-token nanosecond ledger for the Qwen80 uniform-Q4 hybrid vehicle.
//!
//! Production command-buffer shape is preserved. GPU time is
//! `MTLCommandBuffer.GPUEndTime − GPUStartTime` after wait; it is never a
//! CPU-wait proxy. Mixed CBs are reported as mixed — their GPU time is not
//! proportionally split across operator classes.
//!
//! Every named stage is either measured in nanoseconds or explicitly
//! `not_applicable` / `absorbed` with a reason. Silent zeros are a
//! measurement bug. The identity partition (`in_identity_sum`) must sum to
//! the token wall; the residual is named, never dropped.

use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Instant;

use super::qwen80_complete_runtime::{
    Qwen80CanonicalGqaLayout, Qwen80CanonicalLinearDeltaNetLayout, QWEN80_EXPERTS, QWEN80_HIDDEN,
    QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_TOP_K, QWEN80_VOCAB,
};
use super::qwen_complete_binary::UNIFORM_Q4_GROUP_SIZE;

pub const QWEN80_TOKEN_NS_LEDGER_SCHEMA: &str = "hawking.ascension.qwen80_token_ns_ledger.v2";
pub const QWEN80_TOKEN_NS_LEDGER_ENV: &str = "HAWKING_QWEN80_TOKEN_NS_LEDGER";

/// M3 Ultra 96 GB unified — published peak memory bandwidth.
pub const M3_ULTRA_96GB_PEAK_GB_S: f64 = 819.0;

const Q4_BYTES_PER_GROUP: u64 = (UNIFORM_Q4_GROUP_SIZE as u64) / 2 + 2;

/// Identity-partition phase names. These are a closed serial cover of the
/// token wall on the production device path. Nested / absorbed stages are
/// catalogued separately and do not appear here.
pub const IDENTITY_PHASES: &[&str] = &[
    "embed",
    "prefix_deltanet",
    "prefix_gqa",
    "router_readback",
    "host_topk",
    "moe_table_build",
    "suffix",
    "terminal",
    "logits_readback",
    "host_argmax",
    "inter_phase_gap",
];

/// The eleven printed stage names from the velocity baseline. A stage that
/// is absorbed into a mixed CB or is not on the production path must still
/// appear in the catalog with a reason — it may not silently read zero.
pub const NAMED_STAGES: &[&str] = &[
    "embed",
    "deltanet",
    "gqa",
    "moe_norm_router",
    "moe_shared",
    "moe_table_build",
    "moe_routed",
    "moe_combine",
    "terminal",
    "q4_matvec",
    "host_expert_bind",
];

/// Host-activation class names from `Qwen80ActivationClassTimes`.
pub const ACTIVATION_CLASSES: &[&str] = &[
    "shared_swiglu",
    "shared_mlp_sandwich",
    "deltanet_conv",
    "deltanet_recurrent",
    "gqa_input_layernorm",
    "gqa_norm_rope",
    "other_host_activation",
    "metal_matvec_sync",
];

pub fn qwen80_token_ns_ledger_enabled() -> bool {
    match std::env::var(QWEN80_TOKEN_NS_LEDGER_ENV) {
        Ok(raw) => {
            let trimmed = raw.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        }
        Err(_) => false,
    }
}

pub fn q4_payload_bytes(elements: u64) -> u64 {
    let groups = elements.div_ceil(UNIFORM_Q4_GROUP_SIZE as u64);
    groups.saturating_mul(Q4_BYTES_PER_GROUP)
}

pub fn q4_matrix_bytes(rows: usize, cols: usize) -> u64 {
    q4_payload_bytes((rows as u64).saturating_mul(cols as u64))
}

/// Geometry-derived weight traffic for one decode token on this vehicle.
pub fn theoretical_weight_bytes_per_token() -> TheoreticalWeightBytes {
    let expert_proj = q4_matrix_bytes(QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN);
    let dn_qkvz = q4_matrix_bytes(12_288, QWEN80_HIDDEN);
    let dn_ba = q4_matrix_bytes(64, QWEN80_HIDDEN);
    let dn_out = q4_matrix_bytes(QWEN80_HIDDEN, 4_096);
    let gqa_q = q4_matrix_bytes(8_192, QWEN80_HIDDEN);
    let gqa_k = q4_matrix_bytes(512, QWEN80_HIDDEN);
    let gqa_v = q4_matrix_bytes(512, QWEN80_HIDDEN);
    let gqa_o = q4_matrix_bytes(QWEN80_HIDDEN, 4_096);
    let n_dn = 36u64;
    let n_gqa = 12u64;
    let n_layer = QWEN80_LAYERS as u64;
    let attention = n_dn * (dn_qkvz + dn_ba + dn_out) + n_gqa * (gqa_q + gqa_k + gqa_v + gqa_o);
    let router = n_layer * q4_matrix_bytes(QWEN80_EXPERTS, QWEN80_HIDDEN);
    let shared = n_layer * 3 * expert_proj;
    let routed = n_layer * (QWEN80_TOP_K as u64) * 3 * expert_proj;
    let shared_gate = n_layer * q4_matrix_bytes(1, QWEN80_HIDDEN);
    let lm_head = q4_matrix_bytes(QWEN80_VOCAB, QWEN80_HIDDEN);
    let embed = q4_matrix_bytes(1, QWEN80_HIDDEN);
    let total = attention + router + shared + routed + shared_gate + lm_head + embed;
    TheoreticalWeightBytes {
        embed_bytes: embed,
        attention_deltanet_gqa_bytes: attention,
        router_bytes: router,
        shared_expert_bytes: shared,
        routed_expert_bytes: routed,
        shared_expert_gate_bytes: shared_gate,
        lm_head_bytes: lm_head,
        total_bytes: total,
        note: "Q4 group-64 codes+f16 scales only; activations and compact-slab host memcpy are separate",
    }
}

/// Resident activation workspace + per-token readback traffic.
///
/// Workspace is allocated once and reused; it is not a per-token malloc.
/// Readbacks are the host-visible copies that force a CB wait (router
/// logits + lm-head logits).
pub fn theoretical_temp_bytes(max_seq_len: usize) -> TheoreticalTempBytes {
    let f = 4u64;
    let hidden = (QWEN80_HIDDEN as u64).saturating_mul(f);
    let mid = (QWEN80_MOE_INTERMEDIATE as u64).saturating_mul(f);
    let n_dn = 36u64;
    let n_gqa = 12u64;
    let linear = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    let gqa = Qwen80CanonicalGqaLayout::source_exact();
    let qkvz = linear
        .qkvz_projection_elements()
        .map(|n| n as u64 * f)
        .unwrap_or(12_288 * f);
    let ba = linear
        .ba_projection_elements()
        .map(|n| n as u64 * f)
        .unwrap_or(64 * f);
    let value = linear
        .value_elements()
        .map(|n| n as u64 * f)
        .unwrap_or(4_096 * f);
    let conv = linear
        .conv_state_elements()
        .map(|n| n_dn.saturating_mul(n as u64).saturating_mul(f))
        .unwrap_or(0);
    let rec = linear
        .recurrent_state_elements()
        .map(|n| n_dn.saturating_mul(n as u64).saturating_mul(f))
        .unwrap_or(0);
    let q_proj = (gqa.q_proj_rows as u64).saturating_mul(f);
    let kv = (gqa.kv_dim as u64).saturating_mul(f);
    let query = (gqa.query_dim as u64).saturating_mul(f);
    let seq = max_seq_len.max(1) as u64;
    let gqa_cache = n_gqa
        .saturating_mul(seq)
        .saturating_mul(gqa.kv_dim as u64)
        .saturating_mul(f)
        .saturating_mul(2);
    let scratch = hidden * 6
        + mid * 3
        + qkvz
        + ba
        + value * 6
        + q_proj
        + kv * 2
        + query * 3
        + (QWEN80_EXPERTS as u64) * f
        + (QWEN80_VOCAB as u64) * f
        + f;
    let workspace = scratch + conv + rec + gqa_cache;
    let readback = (QWEN80_EXPERTS as u64) * f * (QWEN80_LAYERS as u64)
        + (QWEN80_VOCAB as u64) * f;
    TheoreticalTempBytes {
        workspace_bytes: workspace,
        readback_bytes_per_token: readback,
        total_bytes: workspace.saturating_add(readback),
        max_seq_len: seq,
        note: "workspace is resident and reused; readback is 48×512 router logits + one vocab lm-head snapshot",
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TheoreticalWeightBytes {
    pub embed_bytes: u64,
    pub attention_deltanet_gqa_bytes: u64,
    pub router_bytes: u64,
    pub shared_expert_bytes: u64,
    pub routed_expert_bytes: u64,
    pub shared_expert_gate_bytes: u64,
    pub lm_head_bytes: u64,
    pub total_bytes: u64,
    pub note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct TheoreticalTempBytes {
    pub workspace_bytes: u64,
    pub readback_bytes_per_token: u64,
    pub total_bytes: u64,
    pub max_seq_len: u64,
    pub note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CommandBufferRecord {
    pub name: String,
    pub layer: Option<u32>,
    pub operator_classes: Vec<String>,
    pub submit_ns: u64,
    pub gpu_ns: Option<u64>,
    pub gpu_start_s: Option<f64>,
    pub gpu_end_s: Option<f64>,
    pub cpu_wait_ns: u64,
    pub encode_ns: u64,
    pub dispatches: u64,
    pub force_sync_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostSyncRecord {
    pub name: String,
    pub layer: Option<u32>,
    pub reason: String,
    pub cpu_block_ns: u64,
    pub bytes: u64,
    pub direction: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostWorkRecord {
    pub name: String,
    pub layer: Option<u32>,
    pub ns: u64,
    pub bytes: u64,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SubstageRecord {
    pub stage: String,
    pub substage: String,
    pub ns: u64,
    pub calls: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenIdentity {
    pub wall_ns: u64,
    pub sum_identity_phases_ns: u64,
    pub residual_ns: i64,
    pub residual_name: &'static str,
    pub residual_reason: &'static str,
    pub identity_holds: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TokenTotals {
    pub total_token_ns: u64,
    pub total_gpu_busy_ns: u64,
    pub total_gpu_idle_ns: u64,
    pub total_gpu_gap_ns: u64,
    pub total_cpu_critical_ns: u64,
    pub total_dispatches: u64,
    pub total_command_buffers: u64,
    pub total_sync_points: u64,
    pub total_readbacks: u64,
    pub total_buffer_creations: u64,
    pub total_buffer_rebinds: u64,
    pub dram_bytes_per_token: u64,
    pub temp_bytes_per_token: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen80TokenNsToken {
    pub kind: &'static str,
    pub position: u32,
    pub wall_ns: u64,
    pub command_buffers: Vec<CommandBufferRecord>,
    pub host_syncs: Vec<HostSyncRecord>,
    pub host_work: Vec<HostWorkRecord>,
    pub substages: Vec<SubstageRecord>,
    pub phases: BTreeMap<String, u64>,
    pub phase_calls: BTreeMap<String, u64>,
    pub weight_bytes_observed: u64,
    pub buffer_creations: u64,
    pub buffer_rebinds: u64,
    pub identity: TokenIdentity,
    pub totals: TokenTotals,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OperatorClassLine {
    pub class: String,
    pub calls: u64,
    pub ns_total: u64,
    pub ns_per_call: f64,
    pub source: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageLedgerRow {
    pub stage: String,
    pub substage: String,
    pub status: &'static str,
    pub reason: String,
    pub calls_per_token: f64,
    pub ns_per_call: f64,
    pub ns_per_token: f64,
    pub pct_of_token: f64,
    pub resource_class: &'static str,
    pub serial_vs_overlappable: &'static str,
    pub removable_vs_necessary: &'static str,
    pub confidence: &'static str,
    pub measurement_method: String,
    pub in_identity_sum: bool,
    pub gpu_ns_per_token: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IdentityReport {
    pub n: usize,
    pub mean_wall_ns: f64,
    pub mean_sum_identity_ns: f64,
    pub mean_residual_ns: f64,
    pub residual_name: &'static str,
    pub residual_reason: &'static str,
    pub identity_holds_all: bool,
    pub per_token: Vec<TokenIdentityLine>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenIdentityLine {
    pub kind: &'static str,
    pub position: u32,
    pub wall_ns: u64,
    pub sum_identity_phases_ns: u64,
    pub residual_ns: i64,
    pub identity_holds: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Diagnosis {
    pub verdict: String,
    pub rationale: String,
    pub gpu_execution_ns: u64,
    pub cpu_wait_ns: u64,
    pub submit_ns: u64,
    pub encode_ns: u64,
    pub host_sync_ns: u64,
    pub host_work_ns: u64,
    pub wall_ns: u64,
    pub gpu_busy_fraction_of_wall: f64,
    pub wait_minus_gpu_ns: Option<i64>,
    pub command_buffers_per_token: f64,
    pub dispatches_per_token: f64,
    pub weight_bytes_per_token: u64,
    pub implied_gb_s_from_gpu: Option<f64>,
    pub implied_gb_s_from_wall: f64,
    pub peak_memory_gb_s: f64,
    pub bandwidth_fraction_of_peak_from_gpu: Option<f64>,
    pub cannot_split: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen80TokenNsLedger {
    pub schema: &'static str,
    pub vehicle: &'static str,
    pub production_cb_shape: bool,
    pub gpu_timestamp_authority: &'static str,
    pub box_note: &'static str,
    pub theoretical_weight_bytes: TheoreticalWeightBytes,
    pub theoretical_temp_bytes: TheoreticalTempBytes,
    pub tokens: Vec<Qwen80TokenNsToken>,
    pub steady_state_mean: Option<SteadyStateMean>,
    pub ranked_aggregate: Vec<OperatorClassLine>,
    pub stage_table: Vec<StageLedgerRow>,
    pub identity: Option<IdentityReport>,
    pub totals_mean_decode: Option<TokenTotals>,
    pub diagnosis: Option<Diagnosis>,
}

/// Compact receipt: stage table + identity + per-token totals.
/// Omits the per-CB dump that made the v1 ledger 1.2 MiB.
#[derive(Clone, Debug, Serialize)]
pub struct Qwen80TokenNsCompactLedger {
    pub schema: &'static str,
    pub measurement_label: String,
    pub measured_commit: String,
    pub vehicle: &'static str,
    pub production_cb_shape: bool,
    pub gpu_timestamp_authority: &'static str,
    pub box_note: &'static str,
    pub theoretical_weight_bytes: TheoreticalWeightBytes,
    pub theoretical_temp_bytes: TheoreticalTempBytes,
    pub stage_table: Vec<StageLedgerRow>,
    pub identity: Option<IdentityReport>,
    pub totals_mean_decode: Option<TokenTotals>,
    pub per_token: Vec<CompactToken>,
    pub diagnosis: Option<Diagnosis>,
    pub catalog_complete: bool,
    pub silent_zero_stages: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompactToken {
    pub kind: &'static str,
    pub position: u32,
    pub wall_ns: u64,
    pub identity: TokenIdentity,
    pub totals: TokenTotals,
    pub phases: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SteadyStateMean {
    pub n: usize,
    pub wall_ns: f64,
    pub gpu_execution_ns: f64,
    pub cpu_wait_ns: f64,
    pub submit_ns: f64,
    pub encode_ns: f64,
    pub host_sync_ns: f64,
    pub host_work_ns: f64,
    pub command_buffers: f64,
    pub dispatches: f64,
    pub weight_bytes: f64,
}

#[derive(Debug, Default)]
pub struct Qwen80TokenNsSession {
    pub enabled: bool,
    current: Option<TokenBuilder>,
    pub finished: Vec<Qwen80TokenNsToken>,
    /// Optional override written into the compact receipt.
    pub measured_commit: String,
    pub measurement_label: String,
}

#[derive(Debug)]
struct TokenBuilder {
    kind: &'static str,
    position: u32,
    started: Instant,
    last_mark: Instant,
    command_buffers: Vec<CommandBufferRecord>,
    host_syncs: Vec<HostSyncRecord>,
    host_work: Vec<HostWorkRecord>,
    substages: BTreeMap<(String, String), (u64, u64, u64)>,
    phases: BTreeMap<&'static str, u64>,
    phase_calls: BTreeMap<&'static str, u64>,
    weight_bytes_observed: u64,
    buffer_creations: u64,
    buffer_rebinds: u64,
}

impl Qwen80TokenNsSession {
    pub fn from_env() -> Self {
        Self {
            enabled: qwen80_token_ns_ledger_enabled(),
            current: None,
            finished: Vec::new(),
            measured_commit: String::new(),
            measurement_label: "DIRTY_ENGINEERING".to_owned(),
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn begin(&mut self, kind: &'static str, position: u32) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        self.current = Some(TokenBuilder {
            kind,
            position,
            started: now,
            last_mark: now,
            command_buffers: Vec::new(),
            host_syncs: Vec::new(),
            host_work: Vec::new(),
            substages: BTreeMap::new(),
            phases: BTreeMap::new(),
            phase_calls: BTreeMap::new(),
            weight_bytes_observed: 0,
            buffer_creations: 0,
            buffer_rebinds: 0,
        });
    }

    /// Close the current identity phase. Elapsed since the previous mark
    /// (or `begin`) is added to `name`. Call after each serial segment.
    pub fn close_phase(&mut self, name: &'static str) {
        if let Some(cur) = self.current.as_mut() {
            let ns = cur.last_mark.elapsed().as_nanos() as u64;
            cur.last_mark = Instant::now();
            *cur.phases.entry(name).or_insert(0) = cur
                .phases
                .get(name)
                .copied()
                .unwrap_or(0)
                .saturating_add(ns);
            *cur.phase_calls.entry(name).or_insert(0) =
                cur.phase_calls.get(name).copied().unwrap_or(0).saturating_add(1);
        }
    }

    /// Test / synthetic: add `ns` to an identity phase without using Instant.
    pub fn record_phase_ns(&mut self, name: &'static str, ns: u64) {
        if let Some(cur) = self.current.as_mut() {
            *cur.phases.entry(name).or_insert(0) =
                cur.phases.get(name).copied().unwrap_or(0).saturating_add(ns);
            *cur.phase_calls.entry(name).or_insert(0) =
                cur.phase_calls.get(name).copied().unwrap_or(0).saturating_add(1);
        }
    }

    pub fn record_substage(
        &mut self,
        stage: impl Into<String>,
        substage: impl Into<String>,
        ns: u64,
        calls: u64,
        bytes: u64,
    ) {
        if let Some(cur) = self.current.as_mut() {
            let key = (stage.into(), substage.into());
            let slot = cur.substages.entry(key).or_insert((0, 0, 0));
            slot.0 = slot.0.saturating_add(ns);
            slot.1 = slot.1.saturating_add(calls);
            slot.2 = slot.2.saturating_add(bytes);
        }
    }

    pub fn add_buffer_creations(&mut self, n: u64) {
        if let Some(cur) = self.current.as_mut() {
            cur.buffer_creations = cur.buffer_creations.saturating_add(n);
        }
    }

    pub fn add_buffer_rebinds(&mut self, n: u64) {
        if let Some(cur) = self.current.as_mut() {
            cur.buffer_rebinds = cur.buffer_rebinds.saturating_add(n);
        }
    }

    pub fn end(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(cur) = self.current.take() {
            self.finished.push(seal_token(cur, None));
        }
    }

    /// Seal the current token with an explicit wall (tests). `None` uses Instant.
    pub fn end_with_wall_ns(&mut self, wall_ns: u64) {
        if !self.enabled {
            return;
        }
        if let Some(cur) = self.current.take() {
            self.finished.push(seal_token(cur, Some(wall_ns)));
        }
    }

    pub fn record_cb(
        &mut self,
        name: impl Into<String>,
        layer: Option<u32>,
        operator_classes: &[&str],
        timing: crate::metal::CommandBufferTiming,
        force_sync_reason: impl Into<String>,
    ) {
        if let Some(cur) = self.current.as_mut() {
            cur.command_buffers.push(CommandBufferRecord {
                name: name.into(),
                layer,
                operator_classes: operator_classes.iter().map(|s| (*s).to_string()).collect(),
                submit_ns: timing.submit_ns,
                gpu_ns: timing.gpu_ns,
                gpu_start_s: timing.gpu_start_s,
                gpu_end_s: timing.gpu_end_s,
                cpu_wait_ns: timing.wait_ns,
                encode_ns: timing.encode_ns,
                dispatches: timing.dispatches,
                force_sync_reason: force_sync_reason.into(),
            });
        }
    }

    pub fn record_sync(
        &mut self,
        name: impl Into<String>,
        layer: Option<u32>,
        reason: impl Into<String>,
        cpu_block_ns: u64,
        bytes: u64,
        direction: &'static str,
    ) {
        if let Some(cur) = self.current.as_mut() {
            cur.host_syncs.push(HostSyncRecord {
                name: name.into(),
                layer,
                reason: reason.into(),
                cpu_block_ns,
                bytes,
                direction,
            });
        }
    }

    pub fn record_host_work(
        &mut self,
        name: impl Into<String>,
        layer: Option<u32>,
        ns: u64,
        bytes: u64,
        note: impl Into<String>,
    ) {
        if let Some(cur) = self.current.as_mut() {
            cur.host_work.push(HostWorkRecord {
                name: name.into(),
                layer,
                ns,
                bytes,
                note: note.into(),
            });
        }
    }

    pub fn add_weight_bytes(&mut self, bytes: u64) {
        if let Some(cur) = self.current.as_mut() {
            cur.weight_bytes_observed = cur.weight_bytes_observed.saturating_add(bytes);
        }
    }

    pub fn finish_report(&self) -> Qwen80TokenNsLedger {
        self.finish_report_with_seq(64)
    }

    pub fn finish_report_with_seq(&self, max_seq_len: usize) -> Qwen80TokenNsLedger {
        let theoretical = theoretical_weight_bytes_per_token();
        let temp = theoretical_temp_bytes(max_seq_len);
        let steady: Vec<&Qwen80TokenNsToken> = self
            .finished
            .iter()
            .filter(|t| t.kind == "decode")
            .collect();
        let source: Vec<&Qwen80TokenNsToken> = if steady.is_empty() {
            self.finished.iter().collect()
        } else {
            steady
        };
        let steady_state_mean = if source.is_empty() {
            None
        } else {
            Some(SteadyStateMean {
                n: source.len(),
                wall_ns: mean(source.iter().map(|t| t.wall_ns as f64)),
                gpu_execution_ns: mean(source.iter().map(|t| sum_gpu(t) as f64)),
                cpu_wait_ns: mean(source.iter().map(|t| sum_wait(t) as f64)),
                submit_ns: mean(source.iter().map(|t| sum_submit(t) as f64)),
                encode_ns: mean(source.iter().map(|t| sum_encode(t) as f64)),
                host_sync_ns: mean(source.iter().map(|t| sum_sync(t) as f64)),
                host_work_ns: mean(source.iter().map(|t| sum_host_work(t) as f64)),
                command_buffers: mean(source.iter().map(|t| t.command_buffers.len() as f64)),
                dispatches: mean(source.iter().map(|t| sum_disp(t) as f64)),
                weight_bytes: mean(source.iter().map(|t| t.weight_bytes_observed as f64)),
            })
        };
        let ranked = rank_aggregates(&source);
        let diagnosis = steady_state_mean
            .as_ref()
            .map(|mean| diagnose(mean, &theoretical, &source));
        let stage_table = build_stage_table(&source, &theoretical, &temp);
        let identity = if source.is_empty() {
            None
        } else {
            Some(build_identity_report(&source))
        };
        let totals_mean_decode = if source.is_empty() {
            None
        } else {
            Some(mean_totals(&source))
        };
        Qwen80TokenNsLedger {
            schema: QWEN80_TOKEN_NS_LEDGER_SCHEMA,
            vehicle: "uniform-q4-group64-v1 + hybrid token graph (device activations + 512-way expert table)",
            production_cb_shape: true,
            gpu_timestamp_authority:
                "completed MTLCommandBuffer GPUStartTime/GPUEndTime only; never a CPU-wait proxy",
            box_note: "Apple M3 Ultra, 60 GPU cores, 96 GB unified, 819 GB/s published peak",
            theoretical_weight_bytes: theoretical,
            theoretical_temp_bytes: temp,
            tokens: self.finished.clone(),
            steady_state_mean,
            ranked_aggregate: ranked,
            stage_table,
            identity,
            totals_mean_decode,
            diagnosis,
        }
    }

    pub fn compact_receipt(&self) -> Qwen80TokenNsCompactLedger {
        let report = self.finish_report();
        let silent = report
            .stage_table
            .iter()
            .filter(|row| {
                row.stage != "bytes"
                    && row.ns_per_token == 0.0
                    && row.status != "not_applicable"
                    && row.status != "absorbed"
            })
            .map(|row| format!("{}.{}", row.stage, row.substage))
            .collect::<Vec<_>>();
        let catalog_complete = catalog_is_complete(&report.stage_table) && silent.is_empty();
        Qwen80TokenNsCompactLedger {
            schema: report.schema,
            measurement_label: if self.measurement_label.is_empty() {
                "DIRTY_ENGINEERING".to_owned()
            } else {
                self.measurement_label.clone()
            },
            measured_commit: self.measured_commit.clone(),
            vehicle: report.vehicle,
            production_cb_shape: report.production_cb_shape,
            gpu_timestamp_authority: report.gpu_timestamp_authority,
            box_note: report.box_note,
            theoretical_weight_bytes: report.theoretical_weight_bytes,
            theoretical_temp_bytes: report.theoretical_temp_bytes,
            stage_table: report.stage_table,
            identity: report.identity,
            totals_mean_decode: report.totals_mean_decode,
            per_token: self
                .finished
                .iter()
                .map(|t| CompactToken {
                    kind: t.kind,
                    position: t.position,
                    wall_ns: t.wall_ns,
                    identity: t.identity.clone(),
                    totals: t.totals.clone(),
                    phases: t.phases.clone(),
                })
                .collect(),
            diagnosis: report.diagnosis,
            catalog_complete,
            silent_zero_stages: silent,
        }
    }
}

fn seal_token(mut cur: TokenBuilder, wall_override: Option<u64>) -> Qwen80TokenNsToken {
    // Synthetic tokens pass an explicit wall; do not mix Instant leftover into that sum.
    let leftover = if wall_override.is_some() {
        0
    } else {
        cur.last_mark.elapsed().as_nanos() as u64
    };
    if leftover > 0 {
        *cur.phases.entry("inter_phase_gap").or_insert(0) = cur
            .phases
            .get("inter_phase_gap")
            .copied()
            .unwrap_or(0)
            .saturating_add(leftover);
        *cur.phase_calls.entry("inter_phase_gap").or_insert(0) = cur
            .phase_calls
            .get("inter_phase_gap")
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
    }
    let wall_ns = wall_override.unwrap_or(cur.started.elapsed().as_nanos() as u64);
    let phases: BTreeMap<String, u64> = cur
        .phases
        .iter()
        .map(|(k, v)| ((*k).to_owned(), *v))
        .collect();
    let phase_calls: BTreeMap<String, u64> = cur
        .phase_calls
        .iter()
        .map(|(k, v)| ((*k).to_owned(), *v))
        .collect();
    let substages: Vec<SubstageRecord> = cur
        .substages
        .into_iter()
        .map(|((stage, substage), (ns, calls, bytes))| SubstageRecord {
            stage,
            substage,
            ns,
            calls,
            bytes,
        })
        .collect();
    let identity = identity_of(wall_ns, &phases);
    let mut token = Qwen80TokenNsToken {
        kind: cur.kind,
        position: cur.position,
        wall_ns,
        command_buffers: cur.command_buffers,
        host_syncs: cur.host_syncs,
        host_work: cur.host_work,
        substages,
        phases,
        phase_calls,
        weight_bytes_observed: cur.weight_bytes_observed,
        buffer_creations: cur.buffer_creations,
        buffer_rebinds: cur.buffer_rebinds,
        identity,
        totals: TokenTotals::default(),
    };
    token.totals = totals_of(&token);
    token
}

fn identity_of(wall_ns: u64, phases: &BTreeMap<String, u64>) -> TokenIdentity {
    let sum: u64 = IDENTITY_PHASES
        .iter()
        .map(|name| phases.get(*name).copied().unwrap_or(0))
        .sum();
    let residual = wall_ns as i64 - sum as i64;
    // Instant close_phase leaves a few hundred ns of accounting noise.
    // A residual larger than 50 µs is a missing mark, not clock jitter.
    let holds = residual.abs() <= 50_000;
    TokenIdentity {
        wall_ns,
        sum_identity_phases_ns: sum,
        residual_ns: residual,
        residual_name: "inter_phase_gap_plus_seal_jitter",
        residual_reason: "wall − sum(identity phases). inter_phase_gap is the named leftover (position++, rss cap, unmarked host). Residual after that is Instant close/seal jitter; >50µs means a phase was not closed.",
        identity_holds: holds,
    }
}

fn gpu_accounting(token: &Qwen80TokenNsToken) -> (u64, u64, u64) {
    let busy = sum_gpu(token);
    let mut pairs: Vec<(f64, f64)> = token
        .command_buffers
        .iter()
        .filter_map(|cb| match (cb.gpu_start_s, cb.gpu_end_s) {
            (Some(s), Some(e)) if e > s => Some((s, e)),
            _ => None,
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut gap = 0u64;
    for window in pairs.windows(2) {
        let delta = window[1].0 - window[0].1;
        if delta > 0.0 {
            gap = gap.saturating_add((delta * 1_000_000_000.0) as u64);
        }
    }
    let idle = token.wall_ns.saturating_sub(busy);
    (busy, idle, gap)
}

fn totals_of(token: &Qwen80TokenNsToken) -> TokenTotals {
    let (busy, idle, gap) = gpu_accounting(token);
    let cpu_critical = [
        "router_readback",
        "host_topk",
        "moe_table_build",
        "logits_readback",
        "host_argmax",
        "inter_phase_gap",
    ]
    .iter()
    .map(|n| token.phases.get(*n).copied().unwrap_or(0))
    .sum::<u64>()
    .saturating_add(sum_encode(token));
    let readbacks = token
        .host_syncs
        .iter()
        .filter(|s| s.direction == "device_to_host")
        .count() as u64;
    let temp = theoretical_temp_bytes(64);
    TokenTotals {
        total_token_ns: token.wall_ns,
        total_gpu_busy_ns: busy,
        total_gpu_idle_ns: idle,
        total_gpu_gap_ns: gap,
        total_cpu_critical_ns: cpu_critical,
        total_dispatches: sum_disp(token),
        total_command_buffers: token.command_buffers.len() as u64,
        total_sync_points: token.command_buffers.len() as u64,
        total_readbacks: readbacks,
        total_buffer_creations: token.buffer_creations,
        total_buffer_rebinds: token.buffer_rebinds,
        dram_bytes_per_token: token
            .weight_bytes_observed
            .max(theoretical_weight_bytes_per_token().total_bytes),
        temp_bytes_per_token: temp.total_bytes,
    }
}

fn mean_totals(tokens: &[&Qwen80TokenNsToken]) -> TokenTotals {
    let n = tokens.len().max(1) as u64;
    let sum = |f: fn(&Qwen80TokenNsToken) -> u64| -> u64 {
        tokens.iter().map(|t| f(t)).sum::<u64>() / n
    };
    TokenTotals {
        total_token_ns: sum(|t| t.totals.total_token_ns),
        total_gpu_busy_ns: sum(|t| t.totals.total_gpu_busy_ns),
        total_gpu_idle_ns: sum(|t| t.totals.total_gpu_idle_ns),
        total_gpu_gap_ns: sum(|t| t.totals.total_gpu_gap_ns),
        total_cpu_critical_ns: sum(|t| t.totals.total_cpu_critical_ns),
        total_dispatches: sum(|t| t.totals.total_dispatches),
        total_command_buffers: sum(|t| t.totals.total_command_buffers),
        total_sync_points: sum(|t| t.totals.total_sync_points),
        total_readbacks: sum(|t| t.totals.total_readbacks),
        total_buffer_creations: sum(|t| t.totals.total_buffer_creations),
        total_buffer_rebinds: sum(|t| t.totals.total_buffer_rebinds),
        dram_bytes_per_token: sum(|t| t.totals.dram_bytes_per_token),
        temp_bytes_per_token: sum(|t| t.totals.temp_bytes_per_token),
    }
}

fn build_identity_report(tokens: &[&Qwen80TokenNsToken]) -> IdentityReport {
    let per_token: Vec<TokenIdentityLine> = tokens
        .iter()
        .map(|t| TokenIdentityLine {
            kind: t.kind,
            position: t.position,
            wall_ns: t.identity.wall_ns,
            sum_identity_phases_ns: t.identity.sum_identity_phases_ns,
            residual_ns: t.identity.residual_ns,
            identity_holds: t.identity.identity_holds,
        })
        .collect();
    IdentityReport {
        n: tokens.len(),
        mean_wall_ns: mean(tokens.iter().map(|t| t.identity.wall_ns as f64)),
        mean_sum_identity_ns: mean(
            tokens
                .iter()
                .map(|t| t.identity.sum_identity_phases_ns as f64),
        ),
        mean_residual_ns: mean(tokens.iter().map(|t| t.identity.residual_ns as f64)),
        residual_name: "inter_phase_gap_plus_seal_jitter",
        residual_reason: tokens
            .first()
            .map(|t| t.identity.residual_reason)
            .unwrap_or(""),
        identity_holds_all: per_token.iter().all(|t| t.identity_holds),
        per_token,
    }
}

fn phase_mean(tokens: &[&Qwen80TokenNsToken], name: &str) -> (f64, f64) {
    if tokens.is_empty() {
        return (0.0, 0.0);
    }
    let ns: f64 = tokens
        .iter()
        .map(|t| t.phases.get(name).copied().unwrap_or(0) as f64)
        .sum::<f64>()
        / tokens.len() as f64;
    let calls: f64 = tokens
        .iter()
        .map(|t| t.phase_calls.get(name).copied().unwrap_or(0) as f64)
        .sum::<f64>()
        / tokens.len() as f64;
    (ns, calls)
}

fn substage_mean(tokens: &[&Qwen80TokenNsToken], stage: &str, sub: &str) -> (f64, f64) {
    if tokens.is_empty() {
        return (0.0, 0.0);
    }
    let ns: f64 = tokens
        .iter()
        .map(|t| {
            t.substages
                .iter()
                .filter(|s| s.stage == stage && s.substage == sub)
                .map(|s| s.ns as f64)
                .sum::<f64>()
        })
        .sum::<f64>()
        / tokens.len() as f64;
    let calls: f64 = tokens
        .iter()
        .map(|t| {
            t.substages
                .iter()
                .filter(|s| s.stage == stage && s.substage == sub)
                .map(|s| s.calls as f64)
                .sum::<f64>()
        })
        .sum::<f64>()
        / tokens.len() as f64;
    (ns, calls)
}

fn mixed_gpu(tokens: &[&Qwen80TokenNsToken], pred: impl Fn(&CommandBufferRecord) -> bool) -> Option<f64> {
    if tokens.is_empty() {
        return None;
    }
    if !tokens.iter().all(|t| {
        t.command_buffers
            .iter()
            .filter(|cb| pred(cb))
            .all(|cb| cb.gpu_ns.is_some())
    }) {
        return None;
    }
    let ns: f64 = tokens
        .iter()
        .map(|t| {
            t.command_buffers
                .iter()
                .filter(|cb| pred(cb))
                .map(|cb| cb.gpu_ns.unwrap_or(0) as f64)
                .sum::<f64>()
        })
        .sum::<f64>()
        / tokens.len() as f64;
    Some(ns)
}

fn row(
    stage: &str,
    substage: &str,
    status: &'static str,
    reason: impl Into<String>,
    calls: f64,
    ns: f64,
    wall: f64,
    resource: &'static str,
    serial: &'static str,
    removable: &'static str,
    confidence: &'static str,
    method: impl Into<String>,
    in_identity: bool,
    gpu_ns: Option<f64>,
) -> StageLedgerRow {
    let ns_per_call = if calls > 0.0 { ns / calls } else { 0.0 };
    let pct = if wall > 0.0 { 100.0 * ns / wall } else { 0.0 };
    StageLedgerRow {
        stage: stage.to_owned(),
        substage: substage.to_owned(),
        status,
        reason: reason.into(),
        calls_per_token: calls,
        ns_per_call,
        ns_per_token: ns,
        pct_of_token: pct,
        resource_class: resource,
        serial_vs_overlappable: serial,
        removable_vs_necessary: removable,
        confidence,
        measurement_method: method.into(),
        in_identity_sum: in_identity,
        gpu_ns_per_token: gpu_ns,
    }
}

fn build_stage_table(
    tokens: &[&Qwen80TokenNsToken],
    theoretical: &TheoreticalWeightBytes,
    temp: &TheoreticalTempBytes,
) -> Vec<StageLedgerRow> {
    let wall = if tokens.is_empty() {
        0.0
    } else {
        mean(tokens.iter().map(|t| t.wall_ns as f64))
    };
    let (embed_ns, embed_calls) = phase_mean(tokens, "embed");
    let (dn_ns, dn_calls) = phase_mean(tokens, "prefix_deltanet");
    let (gqa_ns, gqa_calls) = phase_mean(tokens, "prefix_gqa");
    let (rb_ns, rb_calls) = phase_mean(tokens, "router_readback");
    let (topk_ns, topk_calls) = phase_mean(tokens, "host_topk");
    let (tbl_ns, tbl_calls) = phase_mean(tokens, "moe_table_build");
    let (suf_ns, suf_calls) = phase_mean(tokens, "suffix");
    let (term_ns, term_calls) = phase_mean(tokens, "terminal");
    let (logit_ns, logit_calls) = phase_mean(tokens, "logits_readback");
    let (arg_ns, arg_calls) = phase_mean(tokens, "host_argmax");
    let (gap_ns, gap_calls) = phase_mean(tokens, "inter_phase_gap");

    let embed_gpu = mixed_gpu(tokens, |cb| cb.name == "embed");
    let dn_gpu = mixed_gpu(tokens, |cb| {
        cb.name.contains(".prefix") && cb.operator_classes.iter().any(|c| c == "deltanet")
    });
    let gqa_gpu = mixed_gpu(tokens, |cb| {
        cb.name.contains(".prefix") && cb.operator_classes.iter().any(|c| c == "gqa")
    });
    let suf_gpu = mixed_gpu(tokens, |cb| cb.name.contains(".suffix"));
    let term_gpu = mixed_gpu(tokens, |cb| cb.name == "terminal");

    let (swiglu_ns, swiglu_calls) = substage_mean(tokens, "activation", "shared_swiglu_encode");
    let (sandwich_ns, sandwich_calls) =
        substage_mean(tokens, "activation", "shared_mlp_sandwich_encode");
    let (conv_ns, conv_calls) = substage_mean(tokens, "activation", "deltanet_conv_encode");
    let (rec_ns, rec_calls) = substage_mean(tokens, "activation", "deltanet_recurrent_encode");
    let (gqa_ln_ns, gqa_ln_calls) = substage_mean(tokens, "activation", "gqa_input_layernorm_encode");
    let (rope_ns, rope_calls) = substage_mean(tokens, "activation", "gqa_norm_rope_encode");
    let (other_ns, other_calls) = substage_mean(tokens, "activation", "other_host_activation");
    let (q4_enc_ns, q4_enc_calls) = substage_mean(tokens, "q4_matvec", "encode");
    let (norm_enc_ns, norm_enc_calls) = substage_mean(tokens, "moe_norm_router", "encode");
    let (shared_enc_ns, shared_enc_calls) = substage_mean(tokens, "moe_shared", "encode");
    let (routed_enc_ns, routed_enc_calls) = substage_mean(tokens, "moe_routed", "encode");

    let mut rows = vec![
        row(
            "embed",
            "gather_cb",
            "measured",
            "device embedding lookup command buffer (encode+submit+wait)",
            embed_calls,
            embed_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary",
            "high",
            "host Instant around embed_into; GPU ns = GPUEndTime-GPUStartTime after wait",
            true,
            embed_gpu,
        ),
        row(
            "deltanet",
            "prefix_cb_mixed",
            "measured",
            "mixed prefix CB for 36 DeltaNet layers: mixer + shared-expert + router + norm. GPU time is not split.",
            dn_calls,
            dn_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary (mixer); CB split removable if device top-k lands",
            "high",
            "host Instant around prefix encode+commit_and_wait; GPU ns is mixed",
            true,
            dn_gpu,
        ),
        row(
            "gqa",
            "prefix_cb_mixed",
            "measured",
            "mixed prefix CB for 12 GQA layers: mixer + shared-expert + router + norm. GPU time is not split.",
            gqa_calls,
            gqa_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary (mixer); CB split removable if device top-k lands",
            "high",
            "host Instant around prefix encode+commit_and_wait; GPU ns is mixed",
            true,
            gqa_gpu,
        ),
        row(
            "moe_norm_router",
            "gpu_in_prefix",
            "absorbed",
            "post-attn RMSNorm + router Q4 matvec are encoded into the mixed prefix CB. Isolated GPU ns is not measurable without splitting the CB (forbidden: would change token identity). Host extract is identity rows router_readback + host_topk.",
            norm_enc_calls,
            norm_enc_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary",
            "medium",
            "CPU Instant around encode_residual_rmsnorm + router matvec only; GPU absorbed into prefix",
            false,
            None,
        ),
        row(
            "moe_shared",
            "gpu_in_prefix",
            "absorbed",
            "shared-expert SwiGLU MLP is encoded into the mixed prefix CB. Isolated GPU ns is not measurable.",
            shared_enc_calls,
            shared_enc_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary",
            "medium",
            "CPU Instant around encode_shared_mlp only; GPU absorbed into prefix",
            false,
            None,
        ),
        row(
            "moe_table_build",
            "host_address_table",
            "measured",
            "rewrite the live top-10 expert gpuAddress table (or first-touch triplet upload). Payloads stay resident.",
            tbl_calls,
            tbl_ns,
            wall,
            "CPU",
            "serial",
            "removable if a 512-way resident table needs no per-token rewrite",
            "high",
            "host Instant around ensure_selected_expert_table + route id/weight writes",
            true,
            None,
        ),
        row(
            "moe_routed",
            "gpu_in_suffix",
            "absorbed",
            "512-way expert-table wave is encoded into the mixed suffix CB with combine. Isolated GPU ns is not measurable.",
            routed_enc_calls,
            routed_enc_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary",
            "medium",
            "CPU Instant around dispatch_qwen80_device_expert_table_tcb only; GPU absorbed into suffix",
            false,
            None,
        ),
        row(
            "moe_combine",
            "suffix_cb_mixed",
            "measured",
            "mixed suffix CB: routed-expert table + shared-gate + residual add. GPU time is not split. Named moe_combine because that is the historical bucket; moe_routed is absorbed here.",
            suf_calls,
            suf_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary (combine); CB split removable with device top-k + fused layer CB",
            "high",
            "host Instant around suffix encode+commit_and_wait; GPU ns is mixed",
            true,
            suf_gpu,
        ),
        row(
            "terminal",
            "norm_lm_head_cb",
            "measured",
            "final RMSNorm + lm-head Q4 matvec command buffer (encode+submit+wait). Logits snapshot and argmax are separate identity phases.",
            term_calls,
            term_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary",
            "high",
            "host Instant around terminal encode+commit_and_wait; GPU ns = GPUEndTime-GPUStartTime",
            true,
            term_gpu,
        ),
        row(
            "q4_matvec",
            "fused_into_mixed_cbs",
            "absorbed",
            "Q4 matvecs are encoded into embed/prefix/suffix/terminal CBs. Isolated per-matvec GPU ns would require per-kernel counter samples that change token identity on this GPU family.",
            q4_enc_calls,
            q4_enc_ns,
            wall,
            "GPU",
            "serial",
            "physically necessary",
            "medium",
            "CPU Instant around encode_q4_matvec; GPU absorbed. Dispatch count is native.q4_matvec_dispatches",
            false,
            None,
        ),
        row(
            "host_expert_bind",
            "host_payload_fallback",
            "not_applicable",
            "512-way device expert table is the production path. host_expert_payload_bind is the host fallback and is not entered. Table rewrite is moe_table_build, not this stage.",
            0.0,
            0.0,
            wall,
            "CPU",
            "serial",
            "not on production path",
            "high",
            "not entered when qwen80_device_expert_table_enabled(); zero is N/A, not a measurement bug",
            false,
            None,
        ),
        row(
            "router_readback",
            "memcpy_512_f32",
            "measured",
            "prefix CB wait has already returned; this is the host memcpy of 512 router logits that forces CPU top-k.",
            rb_calls,
            rb_ns,
            wall,
            "synchronization",
            "serial",
            "removable if device top-k consumes on-device logits",
            "high",
            "host Instant around snapshot_f32(router_logits)",
            true,
            None,
        ),
        row(
            "host_topk",
            "softmax_top10",
            "measured",
            "host softmax + top-10 + renormalize on 512 logits. Serial: suffix cannot encode until route ids exist.",
            topk_calls,
            topk_ns,
            wall,
            "CPU",
            "serial",
            "removable if device top-k lands",
            "high",
            "host Instant around source_qwen80_topk_router",
            true,
            None,
        ),
        row(
            "logits_readback",
            "memcpy_vocab_f32",
            "measured",
            "terminal CB wait has returned; host memcpy of vocab logits for greedy argmax.",
            logit_calls,
            logit_ns,
            wall,
            "synchronization",
            "serial",
            "removable if device argmax / sampled-id feedback",
            "high",
            "host Instant around snapshot_f32(logits)",
            true,
            None,
        ),
        row(
            "host_argmax",
            "lowest_id_greedy",
            "measured",
            "lowest-id-wins greedy over tokenizer vocab on the host.",
            arg_calls,
            arg_ns,
            wall,
            "CPU",
            "serial",
            "removable if device argmax",
            "high",
            "host Instant around the argmax scan",
            true,
            None,
        ),
        row(
            "inter_phase_gap",
            "unmarked_host",
            "measured",
            "named residual: position increment, rss cap, layer-loop overhead, and any host between closed phases. This is the 0.37 s class of cost the old stage print hid.",
            gap_calls,
            gap_ns,
            wall,
            "CPU",
            "serial",
            "mostly bookkeeping; investigate if it grows",
            "high",
            "token wall minus closed phases; sealed into this bucket at end()",
            true,
            None,
        ),
        row(
            "activation",
            "shared_swiglu",
            if swiglu_calls > 0.0 { "measured" } else { "absorbed" },
            "device silu_mul encode wall. GPU is inside the mixed prefix CB.",
            swiglu_calls,
            swiglu_ns,
            wall,
            "CPU",
            "serial",
            "physically necessary",
            "medium",
            "host Instant around encode_silu_mul; GPU absorbed into prefix",
            false,
            None,
        ),
        row(
            "activation",
            "shared_mlp_sandwich",
            if sandwich_calls > 0.0 {
                "measured"
            } else {
                "absorbed"
            },
            "device shared-expert sandwich encode wall (3 Q4 + silu). GPU absorbed into prefix.",
            sandwich_calls,
            sandwich_ns,
            wall,
            "CPU",
            "serial",
            "physically necessary",
            "medium",
            "host Instant around encode_shared_mlp",
            false,
            None,
        ),
        row(
            "activation",
            "deltanet_conv",
            if conv_calls > 0.0 { "measured" } else { "absorbed" },
            "device qwen80_qkvz_rearrange_conv_l2_f32 encode wall. GPU absorbed into prefix.",
            conv_calls,
            conv_ns,
            wall,
            "CPU",
            "serial",
            "physically necessary",
            "medium",
            "host Instant around the conv dispatch",
            false,
            None,
        ),
        row(
            "activation",
            "deltanet_recurrent",
            if rec_calls > 0.0 { "measured" } else { "absorbed" },
            "device qwen80_gated_delta_decode_tg encode wall. GPU absorbed into prefix.",
            rec_calls,
            rec_ns,
            wall,
            "CPU",
            "serial",
            "physically necessary",
            "medium",
            "host Instant around the recurrent dispatch",
            false,
            None,
        ),
        row(
            "activation",
            "gqa_input_layernorm",
            if gqa_ln_calls > 0.0 { "measured" } else { "absorbed" },
            "device residual RMSNorm encode on GQA layers. GPU absorbed into prefix.",
            gqa_ln_calls,
            gqa_ln_ns,
            wall,
            "CPU",
            "serial",
            "physically necessary",
            "medium",
            "host Instant around encode_residual_rmsnorm in encode_gqa_mixer",
            false,
            None,
        ),
        row(
            "activation",
            "gqa_norm_rope",
            if rope_calls > 0.0 { "measured" } else { "absorbed" },
            "device qwen80_gqa_qk_norm_rope_cache_f32 encode wall. GPU absorbed into prefix.",
            rope_calls,
            rope_ns,
            wall,
            "CPU",
            "serial",
            "physically necessary",
            "medium",
            "host Instant around the rope dispatch",
            false,
            None,
        ),
        row(
            "activation",
            "other_host_activation",
            if other_calls > 0.0 { "measured" } else { "measured" },
            "on the device path this is the host top-k (also identity host_topk). Host-path residual adds / gated-rms live here instead.",
            other_calls.max(topk_calls),
            if other_ns > 0.0 { other_ns } else { topk_ns },
            wall,
            "CPU",
            "serial",
            "top-k removable if device top-k lands",
            "high",
            "host Instant around source_qwen80_topk_router (device path) or host residual ops (host path)",
            false,
            None,
        ),
        row(
            "activation",
            "metal_matvec_sync",
            "not_applicable",
            "per-matvec commit_and_wait exists only on the host-activation fallback path. Production fused CBs sync once per prefix/suffix/embed/terminal; that wait is the parent mixed-CB wait, not this class.",
            0.0,
            0.0,
            wall,
            "synchronization",
            "serial",
            "not on production fused path",
            "high",
            "not entered when device activations are live; zero is N/A",
            false,
            None,
        ),
    ];

    // Geometry-derived byte rows so DRAM / TEMP cannot silently read zero.
    rows.push(row(
        "bytes",
        "dram_weight_traffic",
        "geometry",
        format!(
            "Q4 weight traffic per token; theoretical {} B, observed mean {} B",
            theoretical.total_bytes,
            if tokens.is_empty() {
                0
            } else {
                (mean(tokens.iter().map(|t| t.weight_bytes_observed as f64))) as u64
            }
        ),
        1.0,
        0.0,
        wall,
        "RAM/DRAM",
        "overlappable with compute in principle; today serial behind CB waits",
        "physically necessary (weights must be read)",
        "high",
        "geometry q4_matrix_bytes; observed via add_weight_bytes on encode",
        false,
        None,
    ));
    rows.push(row(
        "bytes",
        "temp_workspace_plus_readback",
        "geometry",
        format!(
            "resident workspace {} B + readback {} B/token",
            temp.workspace_bytes, temp.readback_bytes_per_token
        ),
        1.0,
        0.0,
        wall,
        "RAM/DRAM",
        "workspace resident; readbacks serial",
        "workspace necessary; readbacks removable with on-device top-k/argmax",
        "high",
        "geometry of DeviceActivationWorkspace + 48*512*4 + vocab*4",
        false,
        None,
    ));
    rows
}

fn catalog_is_complete(rows: &[StageLedgerRow]) -> bool {
    let stages: Vec<&str> = rows.iter().map(|r| r.stage.as_str()).collect();
    let substages: Vec<(&str, &str)> = rows
        .iter()
        .map(|r| (r.stage.as_str(), r.substage.as_str()))
        .collect();
    NAMED_STAGES.iter().all(|s| stages.contains(s))
        && ACTIVATION_CLASSES
            .iter()
            .all(|c| substages.iter().any(|(st, sub)| *st == "activation" && sub == c))
        && [
            "router_readback",
            "host_topk",
            "logits_readback",
            "host_argmax",
            "inter_phase_gap",
        ]
        .iter()
        .all(|s| stages.contains(s))
}

fn mean<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let v: Vec<f64> = iter.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn sum_gpu(t: &Qwen80TokenNsToken) -> u64 {
    t.command_buffers
        .iter()
        .map(|cb| cb.gpu_ns.unwrap_or(0))
        .sum()
}

fn sum_wait(t: &Qwen80TokenNsToken) -> u64 {
    t.command_buffers.iter().map(|cb| cb.cpu_wait_ns).sum()
}

fn sum_submit(t: &Qwen80TokenNsToken) -> u64 {
    t.command_buffers.iter().map(|cb| cb.submit_ns).sum()
}

fn sum_encode(t: &Qwen80TokenNsToken) -> u64 {
    t.command_buffers.iter().map(|cb| cb.encode_ns).sum()
}

fn sum_disp(t: &Qwen80TokenNsToken) -> u64 {
    t.command_buffers.iter().map(|cb| cb.dispatches).sum()
}

fn sum_sync(t: &Qwen80TokenNsToken) -> u64 {
    t.host_syncs.iter().map(|s| s.cpu_block_ns).sum()
}

fn sum_host_work(t: &Qwen80TokenNsToken) -> u64 {
    t.host_work.iter().map(|s| s.ns).sum()
}

fn rank_aggregates(tokens: &[&Qwen80TokenNsToken]) -> Vec<OperatorClassLine> {
    let mut map: BTreeMap<String, (u64, u64, &'static str)> = BTreeMap::new();
    let bump = |map: &mut BTreeMap<String, (u64, u64, &'static str)>,
                key: &str,
                ns: u64,
                calls: u64,
                source: &'static str| {
        let slot = map.entry(key.to_owned()).or_insert((0, 0, source));
        slot.0 = slot.0.saturating_add(calls);
        slot.1 = slot.1.saturating_add(ns);
    };
    for token in tokens {
        for cb in &token.command_buffers {
            let label = if cb.operator_classes.len() == 1 {
                cb.operator_classes[0].clone()
            } else {
                format!("cb_mixed:{}", cb.name)
            };
            let ns = cb.cpu_wait_ns;
            bump(
                &mut map,
                &format!("command_buffer_wait:{label}"),
                ns,
                1,
                "host Instant around wait_until_completed (includes GPU work + queue delay)",
            );
            if let Some(gpu) = cb.gpu_ns {
                bump(
                    &mut map,
                    &format!("command_buffer_gpu:{label}"),
                    gpu,
                    1,
                    "MTLCommandBuffer GPUEndTime-GPUStartTime",
                );
            }
            bump(
                &mut map,
                &format!("command_buffer_submit:{label}"),
                cb.submit_ns,
                1,
                "host Instant around commit",
            );
            if cb.encode_ns > 0 {
                bump(
                    &mut map,
                    &format!("command_buffer_encode:{label}"),
                    cb.encode_ns,
                    1,
                    "host Instant around encode (not cost-ledger; always recorded when token_ns is on)",
                );
            }
        }
        for sync in &token.host_syncs {
            bump(
                &mut map,
                &format!("host_sync:{}", sync.name),
                sync.cpu_block_ns,
                1,
                "host work that required a prior GPU wait",
            );
        }
        for work in &token.host_work {
            bump(
                &mut map,
                &format!("host_work:{}", work.name),
                work.ns,
                1,
                "host-visible per-layer or per-expert work",
            );
        }
        for (name, ns) in &token.phases {
            bump(
                &mut map,
                &format!("phase:{name}"),
                *ns,
                1,
                "closed identity phase (serial partition of the token wall)",
            );
        }
    }
    let n = tokens.len().max(1) as u64;
    let mut lines: Vec<OperatorClassLine> = map
        .into_iter()
        .map(|(class, (calls, ns_total, source))| {
            let calls_per_token = calls / n;
            let ns_per_call = if calls == 0 {
                0.0
            } else {
                ns_total as f64 / calls as f64
            };
            OperatorClassLine {
                class,
                calls: calls_per_token,
                ns_total: ns_total / n,
                ns_per_call,
                source,
            }
        })
        .collect();
    lines.sort_by(|a, b| b.ns_total.cmp(&a.ns_total));
    lines
}

fn diagnose(
    mean: &SteadyStateMean,
    theoretical: &TheoreticalWeightBytes,
    tokens: &[&Qwen80TokenNsToken],
) -> Diagnosis {
    let wall = mean.wall_ns;
    let gpu = mean.gpu_execution_ns;
    let wait = mean.cpu_wait_ns;
    let submit = mean.submit_ns;
    let gpu_frac = if wall > 0.0 { gpu / wall } else { 0.0 };
    let wait_minus_gpu = if tokens.iter().all(|t| {
        t.command_buffers
            .iter()
            .all(|cb| cb.gpu_ns.is_some())
    }) {
        Some((wait - gpu) as i64)
    } else {
        None
    };
    let implied_gpu = if gpu > 0.0 {
        Some(mean.weight_bytes.max(theoretical.total_bytes as f64) / gpu)
    } else {
        None
    };
    let implied_wall = if wall > 0.0 {
        mean.weight_bytes.max(theoretical.total_bytes as f64) / wall
    } else {
        0.0
    };
    let bw_frac = implied_gpu.map(|g| g / M3_ULTRA_96GB_PEAK_GB_S);

    let mut cannot_split = Vec::new();
    cannot_split.push(
        "prefix CBs mix attention/DeltaNet + shared-expert + router matvec; GPU time is not split"
            .to_owned(),
    );
    cannot_split.push(
        "suffix CBs mix routed-expert matvecs + silu + combine; GPU time is not split".to_owned(),
    );
    cannot_split.push(
        "per-kernel GPU timestamps require intrusive counter-sample attachments that change token identity on this GPU family"
            .to_owned(),
    );

    let missing_gpu = tokens
        .iter()
        .any(|t| t.command_buffers.iter().any(|cb| cb.gpu_ns.is_none()));
    if missing_gpu {
        cannot_split.push(
            "one or more command buffers lacked GPUStartTime/GPUEndTime; those GPU ns are reported as missing, not proxied"
                .to_owned(),
        );
    }

    let (verdict, rationale) = if missing_gpu && gpu_frac < 0.05 {
        (
            "cannot_separate".to_owned(),
            "GPU timestamps were missing on the production path, so GPU-bound vs sync-bound cannot be separated."
                .to_owned(),
        )
    } else if bw_frac.unwrap_or(0.0) > 0.35 {
        (
            "bandwidth-bound".to_owned(),
            format!(
                "implied {:.1} GB/s from GPU timestamps vs {:.0} GB/s peak ({:.1}%); weight traffic dominates kernel time.",
                implied_gpu.unwrap_or(0.0),
                M3_ULTRA_96GB_PEAK_GB_S,
                bw_frac.unwrap_or(0.0) * 100.0
            ),
        )
    } else if mean.host_work_ns + mean.host_sync_ns > 0.35 * wall {
        (
            "sync-bound".to_owned(),
            format!(
                "host-visible work + post-wait sync is {:.1} ms ({:.0}% of wall). GPU starves on route extraction / expert packing / scalar readbacks.",
                (mean.host_work_ns + mean.host_sync_ns) / 1e6,
                (mean.host_work_ns + mean.host_sync_ns) / wall * 100.0
            ),
        )
    } else if gpu_frac >= 0.70 {
        (
            "gpu-bound".to_owned(),
            format!(
                "GPU execution is {:.0}% of token wall ({:.1} ms GPU / {:.1} ms wall). Kernels genuinely occupy the device.",
                gpu_frac * 100.0,
                gpu / 1e6,
                wall / 1e6
            ),
        )
    } else if wait_minus_gpu.unwrap_or(0) as f64 > 0.40 * wall {
        (
            "sync-bound".to_owned(),
            format!(
                "CPU wait exceeds GPU execution by {:.1} ms ({:.0}% of wall). GPU is idle while the host blocks on readbacks or queue drain.",
                (wait - gpu) / 1e6,
                (wait - gpu) / wall * 100.0
            ),
        )
    } else if mean.command_buffers >= 32.0 && gpu_frac < 0.55 {
        (
            "dispatch-latency-bound".to_owned(),
            format!(
                "{:.0} command buffers and {:.0} dispatches per token; GPU busy only {:.0}% of wall. Many small submits leave the GPU idle between CBs.",
                mean.command_buffers,
                mean.dispatches,
                gpu_frac * 100.0
            ),
        )
    } else {
        (
            "mixed".to_owned(),
            format!(
                "GPU {:.0}% of wall, wait-minus-gpu {:.1} ms, host work {:.1} ms, {:.0} CBs. No single class exceeds the decision thresholds.",
                gpu_frac * 100.0,
                (wait - gpu) / 1e6,
                mean.host_work_ns / 1e6,
                mean.command_buffers
            ),
        )
    };

    Diagnosis {
        verdict,
        rationale,
        gpu_execution_ns: gpu as u64,
        cpu_wait_ns: wait as u64,
        submit_ns: submit as u64,
        encode_ns: mean.encode_ns as u64,
        host_sync_ns: mean.host_sync_ns as u64,
        host_work_ns: mean.host_work_ns as u64,
        wall_ns: wall as u64,
        gpu_busy_fraction_of_wall: gpu_frac,
        wait_minus_gpu_ns: wait_minus_gpu,
        command_buffers_per_token: mean.command_buffers,
        dispatches_per_token: mean.dispatches,
        weight_bytes_per_token: mean.weight_bytes.max(theoretical.total_bytes as f64) as u64,
        implied_gb_s_from_gpu: implied_gpu,
        implied_gb_s_from_wall: implied_wall,
        peak_memory_gb_s: M3_ULTRA_96GB_PEAK_GB_S,
        bandwidth_fraction_of_peak_from_gpu: bw_frac,
        cannot_split,
    }
}

/// Human table for stderr. One line per catalog row. Numbers are ns.
pub fn format_stage_table(rows: &[StageLedgerRow], wall_ns: f64) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "stage                    substage                      status           calls  ns/call        ns/token    %    class            identity\n"
    ));
    for row in rows {
        if row.stage == "bytes" {
            continue;
        }
        out.push_str(&format!(
            "{:<24} {:<29} {:<16} {:>5.1} {:>10.0} {:>14.0} {:>5.1} {:<16} {}\n",
            row.stage,
            row.substage,
            row.status,
            row.calls_per_token,
            row.ns_per_call,
            row.ns_per_token,
            row.pct_of_token,
            row.resource_class,
            if row.in_identity_sum { "SUM" } else { "nested" },
        ));
    }
    let sum: f64 = rows
        .iter()
        .filter(|r| r.in_identity_sum)
        .map(|r| r.ns_per_token)
        .sum();
    out.push_str(&format!(
        "SUM(identity stages)={:.0} ns   wall={:.0} ns   residual={:.0} ns ({:.3}% of wall)\n",
        sum,
        wall_ns,
        wall_ns - sum,
        if wall_ns > 0.0 {
            100.0 * (wall_ns - sum) / wall_ns
        } else {
            0.0
        }
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_session() -> Qwen80TokenNsSession {
        let mut s = Qwen80TokenNsSession::from_env();
        s.enable();
        s.measurement_label = "DIRTY_ENGINEERING".to_owned();
        s.measured_commit = "test".to_owned();
        s.begin("decode", 3);
        s.record_phase_ns("embed", 1_000_000);
        s.record_phase_ns("prefix_deltanet", 40_000_000);
        s.record_phase_ns("prefix_gqa", 12_000_000);
        s.record_phase_ns("router_readback", 2_000_000);
        s.record_phase_ns("host_topk", 3_000_000);
        s.record_phase_ns("moe_table_build", 80_000_000);
        s.record_phase_ns("suffix", 20_000_000);
        s.record_phase_ns("terminal", 5_000_000);
        s.record_phase_ns("logits_readback", 1_500_000);
        s.record_phase_ns("host_argmax", 500_000);
        s.record_phase_ns("inter_phase_gap", 4_000_000);
        s.record_substage("activation", "shared_swiglu_encode", 100_000, 48, 0);
        s.record_substage("activation", "shared_mlp_sandwich_encode", 200_000, 48, 0);
        s.record_substage("activation", "deltanet_conv_encode", 80_000, 36, 0);
        s.record_substage("activation", "deltanet_recurrent_encode", 90_000, 36, 0);
        s.record_substage("activation", "gqa_input_layernorm_encode", 20_000, 12, 0);
        s.record_substage("activation", "gqa_norm_rope_encode", 30_000, 12, 0);
        s.record_substage("activation", "other_host_activation", 3_000_000, 48, 0);
        s.record_substage("q4_matvec", "encode", 400_000, 200, 0);
        s.record_substage("moe_norm_router", "encode", 50_000, 48, 0);
        s.record_substage("moe_shared", "encode", 60_000, 48, 0);
        s.record_substage("moe_routed", "encode", 70_000, 48, 0);
        s.add_weight_bytes(theoretical_weight_bytes_per_token().total_bytes);
        let wall: u64 = 1_000_000
            + 40_000_000
            + 12_000_000
            + 2_000_000
            + 3_000_000
            + 80_000_000
            + 20_000_000
            + 5_000_000
            + 1_500_000
            + 500_000
            + 4_000_000;
        s.end_with_wall_ns(wall);
        s
    }

    #[test]
    fn theoretical_weight_bytes_are_nonzero() {
        let w = theoretical_weight_bytes_per_token();
        assert!(w.total_bytes > 1_000_000_000, "total={}", w.total_bytes);
        assert!(w.embed_bytes > 0);
        assert!(w.lm_head_bytes > 0);
        assert!(w.routed_expert_bytes > w.shared_expert_bytes);
    }

    #[test]
    fn theoretical_temp_bytes_are_nonzero() {
        let t = theoretical_temp_bytes(64);
        assert!(t.workspace_bytes > 0);
        assert!(t.readback_bytes_per_token > 0);
        assert_eq!(
            t.readback_bytes_per_token,
            (QWEN80_EXPERTS * QWEN80_LAYERS + QWEN80_VOCAB) as u64 * 4
        );
    }

    #[test]
    fn catalog_covers_every_named_stage_and_activation() {
        let s = closed_session();
        let compact = s.compact_receipt();
        assert!(
            compact.catalog_complete,
            "silent zeros: {:?}",
            compact.silent_zero_stages
        );
        assert!(compact.silent_zero_stages.is_empty());
        let names: Vec<&str> = compact.stage_table.iter().map(|r| r.stage.as_str()).collect();
        for stage in NAMED_STAGES {
            assert!(names.contains(stage), "missing named stage {stage}");
        }
        for class in ACTIVATION_CLASSES {
            assert!(
                compact
                    .stage_table
                    .iter()
                    .any(|r| r.stage == "activation" && r.substage == *class),
                "missing activation class {class}"
            );
        }
    }

    #[test]
    fn absorbed_and_na_may_be_zero_measured_may_not() {
        let s = closed_session();
        let compact = s.compact_receipt();
        for row in &compact.stage_table {
            if row.stage == "bytes" {
                continue;
            }
            match row.status {
                "measured" => assert!(
                    row.ns_per_token > 0.0,
                    "{}.{} measured but ns/token=0",
                    row.stage,
                    row.substage
                ),
                "not_applicable" | "absorbed" => {}
                other => panic!("unexpected status {other}"),
            }
        }
        let bind = compact
            .stage_table
            .iter()
            .find(|r| r.stage == "host_expert_bind")
            .unwrap();
        assert_eq!(bind.status, "not_applicable");
        assert_eq!(bind.ns_per_token, 0.0);
        let sync = compact
            .stage_table
            .iter()
            .find(|r| r.stage == "activation" && r.substage == "metal_matvec_sync")
            .unwrap();
        assert_eq!(sync.status, "not_applicable");
    }

    #[test]
    fn identity_phases_sum_to_wall() {
        let s = closed_session();
        let token = &s.finished[0];
        assert_eq!(token.identity.wall_ns, token.identity.sum_identity_phases_ns);
        assert_eq!(token.identity.residual_ns, 0);
        assert!(token.identity.identity_holds);
        let compact = s.compact_receipt();
        let id = compact.identity.as_ref().unwrap();
        assert!(id.identity_holds_all);
        assert!((id.mean_wall_ns - id.mean_sum_identity_ns).abs() < 1.0);
        let sum_rows: f64 = compact
            .stage_table
            .iter()
            .filter(|r| r.in_identity_sum)
            .map(|r| r.ns_per_token)
            .sum();
        assert!(
            (sum_rows - id.mean_wall_ns).abs() < 1.0,
            "sum_rows={sum_rows} wall={}",
            id.mean_wall_ns
        );
    }

    #[test]
    fn totals_are_populated() {
        let s = closed_session();
        let t = &s.finished[0].totals;
        assert_eq!(t.total_token_ns, s.finished[0].wall_ns);
        assert!(t.dram_bytes_per_token > 0);
        assert!(t.temp_bytes_per_token > 0);
        // synthetic token has no CBs, so GPU busy is honestly zero
        assert_eq!(t.total_gpu_busy_ns, 0);
        assert_eq!(t.total_gpu_idle_ns, t.total_token_ns);
    }

    #[test]
    fn compact_receipt_omits_command_buffers() {
        let s = closed_session();
        let text = serde_json::to_string(&s.compact_receipt()).unwrap();
        assert!(
            !text.contains("\"command_buffers\""),
            "compact receipt must not dump per-CB records"
        );
        assert!(text.contains("stage_table"));
        assert!(text.len() < 200_000, "compact receipt grew to {}", text.len());
    }

    #[test]
    fn missing_phase_breaks_identity() {
        let mut s = Qwen80TokenNsSession::from_env();
        s.enable();
        s.begin("decode", 0);
        s.record_phase_ns("embed", 100);
        // deliberately omit the rest
        s.end_with_wall_ns(10_000_000);
        assert!(!s.finished[0].identity.identity_holds);
        assert_eq!(s.finished[0].identity.residual_ns, 10_000_000 - 100);
    }
}
