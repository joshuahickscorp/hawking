//! Per-decode-token nanosecond ledger for the Qwen80 uniform-Q4 hybrid vehicle.
//!
//! Production command-buffer shape is preserved. GPU time is
//! `MTLCommandBuffer.GPUEndTime − GPUStartTime` after wait; it is never a
//! CPU-wait proxy. Mixed CBs are reported as mixed — their GPU time is not
//! proportionally split across operator classes.

use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Instant;

use super::qwen80_complete_runtime::{
    QWEN80_EXPERTS, QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_TOP_K,
    QWEN80_VOCAB,
};
use super::qwen_complete_binary::UNIFORM_Q4_GROUP_SIZE;

pub const QWEN80_TOKEN_NS_LEDGER_SCHEMA: &str = "hawking.ascension.qwen80_token_ns_ledger.v1";
pub const QWEN80_TOKEN_NS_LEDGER_ENV: &str = "HAWKING_QWEN80_TOKEN_NS_LEDGER";

/// M3 Ultra 96 GB unified — published peak memory bandwidth.
pub const M3_ULTRA_96GB_PEAK_GB_S: f64 = 819.0;

const Q4_BYTES_PER_GROUP: u64 = (UNIFORM_Q4_GROUP_SIZE as u64) / 2 + 2;

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
pub struct CommandBufferRecord {
    pub name: String,
    pub layer: Option<u32>,
    pub operator_classes: Vec<String>,
    pub submit_ns: u64,
    pub gpu_ns: Option<u64>,
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
pub struct Qwen80TokenNsToken {
    pub kind: &'static str,
    pub position: u32,
    pub wall_ns: u64,
    pub command_buffers: Vec<CommandBufferRecord>,
    pub host_syncs: Vec<HostSyncRecord>,
    pub host_work: Vec<HostWorkRecord>,
    pub weight_bytes_observed: u64,
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
    pub tokens: Vec<Qwen80TokenNsToken>,
    pub steady_state_mean: Option<SteadyStateMean>,
    pub ranked_aggregate: Vec<OperatorClassLine>,
    pub diagnosis: Option<Diagnosis>,
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
}

#[derive(Debug)]
struct TokenBuilder {
    kind: &'static str,
    position: u32,
    started: Instant,
    command_buffers: Vec<CommandBufferRecord>,
    host_syncs: Vec<HostSyncRecord>,
    host_work: Vec<HostWorkRecord>,
    weight_bytes_observed: u64,
}

impl Qwen80TokenNsSession {
    pub fn from_env() -> Self {
        Self {
            enabled: qwen80_token_ns_ledger_enabled(),
            current: None,
            finished: Vec::new(),
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn begin(&mut self, kind: &'static str, position: u32) {
        if !self.enabled {
            return;
        }
        self.current = Some(TokenBuilder {
            kind,
            position,
            started: Instant::now(),
            command_buffers: Vec::new(),
            host_syncs: Vec::new(),
            host_work: Vec::new(),
            weight_bytes_observed: 0,
        });
    }

    pub fn end(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(cur) = self.current.take() {
            self.finished.push(Qwen80TokenNsToken {
                kind: cur.kind,
                position: cur.position,
                wall_ns: cur.started.elapsed().as_nanos() as u64,
                command_buffers: cur.command_buffers,
                host_syncs: cur.host_syncs,
                host_work: cur.host_work,
                weight_bytes_observed: cur.weight_bytes_observed,
            });
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
        let theoretical = theoretical_weight_bytes_per_token();
        let steady: Vec<&Qwen80TokenNsToken> = self
            .finished
            .iter()
            .filter(|t| t.kind == "decode")
            .collect();
        let steady_state_mean = if steady.is_empty() {
            None
        } else {
            Some(SteadyStateMean {
                n: steady.len(),
                wall_ns: mean(steady.iter().map(|t| t.wall_ns as f64)),
                gpu_execution_ns: mean(steady.iter().map(|t| sum_gpu(t) as f64)),
                cpu_wait_ns: mean(steady.iter().map(|t| sum_wait(t) as f64)),
                submit_ns: mean(steady.iter().map(|t| sum_submit(t) as f64)),
                encode_ns: mean(steady.iter().map(|t| sum_encode(t) as f64)),
                host_sync_ns: mean(steady.iter().map(|t| sum_sync(t) as f64)),
                host_work_ns: mean(steady.iter().map(|t| sum_host_work(t) as f64)),
                command_buffers: mean(steady.iter().map(|t| t.command_buffers.len() as f64)),
                dispatches: mean(steady.iter().map(|t| sum_disp(t) as f64)),
                weight_bytes: mean(steady.iter().map(|t| t.weight_bytes_observed as f64)),
            })
        };
        let ranked = rank_aggregates(&steady);
        let diagnosis = steady_state_mean
            .as_ref()
            .map(|mean| diagnose(mean, &theoretical, &steady));
        Qwen80TokenNsLedger {
            schema: QWEN80_TOKEN_NS_LEDGER_SCHEMA,
            vehicle: "uniform-q4-group64-v1 + hybrid token graph (device activations + 512-way expert table)",
            production_cb_shape: true,
            gpu_timestamp_authority:
                "completed MTLCommandBuffer GPUStartTime/GPUEndTime only; never a CPU-wait proxy",
            box_note: "Apple M3 Ultra, 60 GPU cores, 96 GB unified, 819 GB/s published peak",
            theoretical_weight_bytes: theoretical,
            tokens: self.finished.clone(),
            steady_state_mean,
            ranked_aggregate: ranked,
            diagnosis,
        }
    }
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
    // bytes / ns = GB/s.
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

    let missing_gpu = tokens.iter().any(|t| {
        t.command_buffers.iter().any(|cb| cb.gpu_ns.is_none())
    });
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
