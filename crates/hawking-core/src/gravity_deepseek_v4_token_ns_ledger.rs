//! Per-token nanosecond ledger for the DeepSeek-V4-Flash native BOS graph.
//!
//! This is a diagnostic surface. It does not change operator semantics.
//! GPU timestamps are command-buffer `GPUStartTime`/`GPUEndTime` after wait;
//! missing timestamps are reported, never proxied from CPU wait.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;

use crate::metal::MetalBatchTiming;

pub const TOKEN_NS_LEDGER_SCHEMA: &str = "hawking.gravity.deepseek_v4.token_ns_ledger.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenBodyDiagnosis {
    GpuBound,
    DispatchLatencyBound,
    SyncBound,
    IoBound,
    Mixed,
    Insufficient,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageRow {
    pub name: String,
    pub ns: u64,
    pub ns_per_token: u64,
    pub pct_body: f64,
    pub calls: u64,
    pub calls_per_token: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBufferRow {
    pub name: String,
    pub layer: Option<usize>,
    pub force: String,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub host_wall_ns: u64,
    pub gpu_ns: Option<u64>,
    pub gpu_start_ns: Option<u64>,
    pub gpu_end_ns: Option<u64>,
    pub cpu_minus_gpu_ns: Option<i64>,
    pub dispatches: u64,
    pub encoders: u64,
}

/// Device-timeline gaps from consecutive command-buffer
/// `GPUStartTime`/`GPUEndTime` pairs. Intra-CB kernel gaps are not visible
/// on this surface: MetalBatchTiming is one pair per command buffer.
#[derive(Debug, Clone, Serialize)]
pub struct GpuGapAccounting {
    pub timestamp_authority: &'static str,
    pub intra_cb_kernel_gaps_visible: bool,
    pub observed_timestamp_pairs: u64,
    pub missing_timestamp_pairs: u64,
    pub first_gpu_start_ns: Option<u64>,
    pub last_gpu_end_ns: Option<u64>,
    pub device_span_ns: Option<u64>,
    pub gpu_busy_ns: u64,
    pub inter_cb_device_gap_ns: u64,
    pub inter_cb_device_overlap_ns: u64,
    pub inter_cb_gap_count: u64,
    pub gpu_idle_in_span_ns: Option<u64>,
    pub gpu_idle_fraction_of_span: Option<f64>,
    pub host_between_cb_exclusive_ns: u64,
    pub occupancy_proxy_gpu_ns_per_encoder: Option<f64>,
    pub production_encoders: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostSyncRow {
    pub name: String,
    pub layer: Option<usize>,
    pub force: String,
    pub block_ns: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteReadbackRow {
    pub layer: usize,
    pub index: usize,
    pub ns: u64,
    pub blocks: String,
    pub force: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IsolatedKernelRow {
    pub name: String,
    pub layer: usize,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: Option<u64>,
    pub dispatches: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlaDispatchSpec {
    pub name: String,
    pub kernel: String,
    pub invocations_per_layer: u64,
    pub invocations_per_token: u64,
    pub threads: u64,
    pub threadgroups: u64,
    pub threads_per_threadgroup: u64,
    pub simdgroups_per_threadgroup: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub approx_flops: u64,
    pub occupancy_proxy: f64,
    pub isolated_gpu_ns_mean: Option<u64>,
    pub isolated_ns_per_invocation: Option<u64>,
    pub memory_stall_proxy_gbps: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlaPipelineLimit {
    pub kernel: String,
    pub thread_execution_width: u64,
    pub max_total_threads_per_threadgroup: u64,
    pub static_threadgroup_memory_length: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlaKvState {
    pub persistent_device_addressable: bool,
    pub compressed_indexer_loaded: bool,
    pub bos_window_only: bool,
    pub bytes_written_per_layer: u64,
    pub bytes_read_per_layer: u64,
    pub bytes_written_per_token: u64,
    pub bytes_read_per_token: u64,
    pub device_copies_per_token: u64,
    pub host_syncs_per_token: u64,
    pub host_sync_bytes_per_token: u64,
    pub scratch_overwritten_per_layer: bool,
    pub rebuild_rebind_reallocate_per_token: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlaCensus {
    pub serial_group: bool,
    pub layers: u64,
    pub attn_command_buffers: u64,
    pub attn_encoders: u64,
    pub attn_dispatches: u64,
    pub attn_gpu_ns: u64,
    pub attn_wait_ns: u64,
    pub attn_encode_ns: u64,
    pub attn_gpu_ns_min: Option<u64>,
    pub attn_gpu_ns_p50: Option<u64>,
    pub attn_gpu_ns_max: Option<u64>,
    pub attn_gpu_ns_per_invocation: Option<u64>,
    pub layers_gpu_over_20ms: u64,
    pub structural_blocker: bool,
    pub isolated_mla_gpu_sum_ns: u64,
    pub dispatch_gap_proxy_ns: Option<i64>,
    pub limit_class: String,
    pub limit_proof: String,
    pub dispatches: Vec<MlaDispatchSpec>,
    pub kv_state: MlaKvState,
    pub pipeline_limits: Vec<MlaPipelineLimit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetalVsHost {
    pub metal_encode_ns: u64,
    pub metal_submit_ns: u64,
    pub metal_wait_ns: u64,
    pub metal_gpu_ns: u64,
    pub metal_gpu_ns_observed_cbs: u64,
    pub metal_gpu_ns_missing_cbs: u64,
    pub host_exclusive_ns: u64,
    pub host_between_first_last_cb_gap_ns: u64,
    pub production_command_buffers: u64,
    pub production_dispatches: u64,
    pub isolated_probe_overhead_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenNsLedger {
    pub schema: &'static str,
    pub body_ns: u64,
    pub init_ns: u64,
    pub wall_ns: u64,
    pub verify_ns: u64,
    pub diagnosis: TokenBodyDiagnosis,
    pub diagnosis_proof: String,
    pub stages: Vec<StageRow>,
    pub operator_classes: Vec<StageRow>,
    pub command_buffers: Vec<CommandBufferRow>,
    pub host_syncs: Vec<HostSyncRow>,
    pub route_id_readbacks: Vec<RouteReadbackRow>,
    pub isolated_kernels: Vec<IsolatedKernelRow>,
    pub metal_vs_host: MetalVsHost,
    pub host_wall_classes: Vec<StageRow>,
    pub reader_parallel_sum: Vec<StageRow>,
    pub gpu_gaps: GpuGapAccounting,
    pub mla: Option<MlaCensus>,
}

#[derive(Debug, Clone, Default)]
struct StageAccum {
    ns: u64,
    calls: u64,
    bytes: u64,
}

#[derive(Debug)]
pub struct TokenNsCollector {
    stages: BTreeMap<String, StageAccum>,
    command_buffers: Vec<CommandBufferRow>,
    host_syncs: Vec<HostSyncRow>,
    route_id_readbacks: Vec<RouteReadbackRow>,
    isolated_kernels: Vec<IsolatedKernelRow>,
    probe_overhead_ns: u64,
    last_cb_end: Option<Instant>,
    between_cb_gap_ns: u64,
    mla_serial_group: bool,
    mla_dispatch_specs: Vec<MlaDispatchSpec>,
    mla_kv_state: Option<MlaKvState>,
    mla_pipeline_limits: Vec<MlaPipelineLimit>,
}

impl Default for TokenNsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenNsCollector {
    pub fn new() -> Self {
        Self {
            stages: BTreeMap::new(),
            command_buffers: Vec::new(),
            host_syncs: Vec::new(),
            route_id_readbacks: Vec::new(),
            isolated_kernels: Vec::new(),
            probe_overhead_ns: 0,
            last_cb_end: None,
            between_cb_gap_ns: 0,
            mla_serial_group: false,
            mla_dispatch_specs: Vec::new(),
            mla_kv_state: None,
            mla_pipeline_limits: Vec::new(),
        }
    }

    pub fn record_mla_static(
        &mut self,
        serial_group: bool,
        specs: Vec<MlaDispatchSpec>,
        kv_state: MlaKvState,
        pipeline_limits: Vec<MlaPipelineLimit>,
    ) {
        self.mla_serial_group = serial_group;
        self.mla_dispatch_specs = specs;
        self.mla_kv_state = Some(kv_state);
        self.mla_pipeline_limits = pipeline_limits;
    }

    pub fn add_stage(&mut self, name: &str, ns: u64, calls: u64, bytes: u64) {
        let row = self.stages.entry(name.to_owned()).or_default();
        row.ns = row.ns.saturating_add(ns);
        row.calls = row.calls.saturating_add(calls);
        row.bytes = row.bytes.saturating_add(bytes);
    }

    pub fn time<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        self.time_bytes(name, 0, f)
    }

    pub fn time_bytes<T>(&mut self, name: &str, bytes: u64, f: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let out = f();
        self.add_stage(name, started.elapsed().as_nanos() as u64, 1, bytes);
        out
    }

    pub fn time_result<T, E>(
        &mut self,
        name: &str,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        self.time_bytes_result(name, 0, f)
    }

    pub fn time_bytes_result<T, E>(
        &mut self,
        name: &str,
        bytes: u64,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let started = Instant::now();
        let out = f();
        self.add_stage(name, started.elapsed().as_nanos() as u64, 1, bytes);
        out
    }

    pub fn record_cb(
        &mut self,
        name: &str,
        layer: Option<usize>,
        force: &str,
        dispatches: u64,
        timing: &MetalBatchTiming,
    ) {
        let encode_ns = timing.encode_us.saturating_mul(1_000);
        let submit_ns = timing.submit_us.saturating_mul(1_000);
        let wait_ns = timing.wait_us.saturating_mul(1_000);
        let host_wall_ns = timing.host_wall_us.saturating_mul(1_000);
        if let Some(prev) = self.last_cb_end.take() {
            // Host exclusive gap: time after the previous wait returned until
            // this CB's encode started. host_wall covers encode+submit+wait,
            // so subtract it from the interval that ends at this record_cb
            // call (which is after wait).
            let since_prev = prev.elapsed().as_nanos() as u64;
            self.between_cb_gap_ns = self
                .between_cb_gap_ns
                .saturating_add(since_prev.saturating_sub(host_wall_ns));
        }
        let gpu_ns = timing.gpu_duration_us.map(|us| us.saturating_mul(1_000));
        let cpu_minus_gpu_ns = gpu_ns.map(|gpu| wait_ns as i64 - gpu as i64);
        self.add_stage("metal.encode", encode_ns, 1, 0);
        self.add_stage("metal.submit", submit_ns, 1, 0);
        self.add_stage("metal.wait", wait_ns, 1, 0);
        // host_wall starts at submit_batch and ends after wait. The gap vs
        // encode+submit+wait is host work between commit and wait (shared I/O
        // during attn GPU, prefetch_fill during moe GPU). Raw observation;
        // TOKEN_NS partitions it into exclusive serial classes.
        let overlap_ns = host_wall_ns.saturating_sub(
            encode_ns
                .saturating_add(submit_ns)
                .saturating_add(wait_ns),
        );
        self.add_stage("metal.cb_overlap_host", overlap_ns, 1, 0);
        if let Some(gpu) = gpu_ns {
            self.add_stage("metal.gpu", gpu, 1, 0);
        } else {
            self.add_stage("metal.gpu_missing", 0, 1, 0);
        }
        self.command_buffers.push(CommandBufferRow {
            name: name.to_owned(),
            layer,
            force: force.to_owned(),
            encode_ns,
            submit_ns,
            wait_ns,
            host_wall_ns,
            gpu_ns,
            gpu_start_ns: timing.gpu_start_ns,
            gpu_end_ns: timing.gpu_end_ns,
            cpu_minus_gpu_ns,
            dispatches,
            encoders: timing.compute_encoders,
        });
        self.last_cb_end = Some(Instant::now());
    }

    pub fn record_sync(&mut self, name: &str, layer: Option<usize>, force: &str, block_ns: u64, bytes: u64) {
        self.add_stage(name, block_ns, 1, bytes);
        self.host_syncs.push(HostSyncRow {
            name: name.to_owned(),
            layer,
            force: force.to_owned(),
            block_ns,
            bytes,
        });
    }

    pub fn record_route_readback(
        &mut self,
        layer: usize,
        ns: u64,
        blocks: &str,
        force: &str,
    ) {
        let index = self.route_id_readbacks.len();
        self.route_id_readbacks.push(RouteReadbackRow {
            layer,
            index,
            ns,
            blocks: blocks.to_owned(),
            force: force.to_owned(),
        });
        self.add_stage("host.route_id_readback", ns, 1, 6 * 4);
    }

    pub fn record_isolated(&mut self, name: &str, layer: usize, overhead_ns: u64, timing: &MetalBatchTiming) {
        self.probe_overhead_ns = self.probe_overhead_ns.saturating_add(overhead_ns);
        self.isolated_kernels.push(IsolatedKernelRow {
            name: name.to_owned(),
            layer,
            encode_ns: timing.encode_us.saturating_mul(1_000),
            submit_ns: timing.submit_us.saturating_mul(1_000),
            wait_ns: timing.wait_us.saturating_mul(1_000),
            gpu_ns: timing.gpu_duration_us.map(|us| us.saturating_mul(1_000)),
            dispatches: timing.compute_dispatches,
        });
    }

    pub fn probe_overhead_ns(&self) -> u64 {
        self.probe_overhead_ns
    }

    pub fn finish(
        self,
        body_ns: u64,
        init_ns: u64,
        wall_ns: u64,
        verify_ns: u64,
    ) -> TokenNsLedger {
        let production_command_buffers = self.command_buffers.len() as u64;
        let production_dispatches = self
            .command_buffers
            .iter()
            .map(|cb| cb.dispatches)
            .sum::<u64>();
        let metal_encode_ns = stage_ns(&self.stages, "metal.encode");
        let metal_submit_ns = stage_ns(&self.stages, "metal.submit");
        let metal_wait_ns = stage_ns(&self.stages, "metal.wait");
        let metal_gpu_ns = stage_ns(&self.stages, "metal.gpu");
        let metal_gpu_ns_observed_cbs = self
            .command_buffers
            .iter()
            .filter(|cb| cb.gpu_ns.is_some())
            .count() as u64;
        let metal_gpu_ns_missing_cbs = production_command_buffers
            .saturating_sub(metal_gpu_ns_observed_cbs);
        let metal_host_wall_ns = self
            .command_buffers
            .iter()
            .map(|cb| cb.host_wall_ns)
            .sum::<u64>();
        let host_exclusive_ns = body_ns.saturating_sub(metal_host_wall_ns);

        let stages = finalize_stages(&self.stages, body_ns);
        let operator_classes = operator_class_rows(&self.stages, &self.isolated_kernels, body_ns);
        let host_wall_classes = host_wall_class_rows(&self.stages, body_ns, host_exclusive_ns);
        let reader_parallel_sum = reader_parallel_sum_rows(&self.stages, body_ns);
        let (diagnosis, diagnosis_proof) = diagnose(
            body_ns,
            host_exclusive_ns,
            metal_wait_ns,
            metal_gpu_ns,
            metal_gpu_ns_observed_cbs,
            production_command_buffers,
            verify_ns,
            &self.command_buffers,
            &self.isolated_kernels,
        );
        let gpu_gaps = account_gpu_gaps(&self.command_buffers, self.between_cb_gap_ns);
        let mla = build_mla_census(
            self.mla_serial_group,
            &self.mla_dispatch_specs,
            self.mla_kv_state,
            &self.mla_pipeline_limits,
            &self.command_buffers,
            &self.isolated_kernels,
        );

        TokenNsLedger {
            schema: TOKEN_NS_LEDGER_SCHEMA,
            body_ns,
            init_ns,
            wall_ns,
            verify_ns,
            diagnosis,
            diagnosis_proof,
            stages,
            operator_classes,
            command_buffers: self.command_buffers,
            host_syncs: self.host_syncs,
            route_id_readbacks: self.route_id_readbacks,
            isolated_kernels: self.isolated_kernels,
            metal_vs_host: MetalVsHost {
                metal_encode_ns,
                metal_submit_ns,
                metal_wait_ns,
                metal_gpu_ns,
                metal_gpu_ns_observed_cbs,
                metal_gpu_ns_missing_cbs,
                host_exclusive_ns,
                host_between_first_last_cb_gap_ns: self.between_cb_gap_ns,
                production_command_buffers,
                production_dispatches,
                isolated_probe_overhead_ns: self.probe_overhead_ns,
            },
            host_wall_classes,
            reader_parallel_sum,
            gpu_gaps,
            mla,
        }
    }
}

fn account_gpu_gaps(cbs: &[CommandBufferRow], host_between_cb_exclusive_ns: u64) -> GpuGapAccounting {
    let mut observed = 0u64;
    let mut missing = 0u64;
    let mut first_start = None;
    let mut last_end = None;
    let mut busy = 0u64;
    let mut gap = 0u64;
    let mut overlap = 0u64;
    let mut gap_count = 0u64;
    let mut prev_end: Option<u64> = None;
    let mut encoders = 0u64;
    for cb in cbs {
        encoders = encoders.saturating_add(cb.encoders);
        match (cb.gpu_start_ns, cb.gpu_end_ns, cb.gpu_ns) {
            (Some(start), Some(end), gpu) if end > start => {
                observed += 1;
                busy = busy.saturating_add(gpu.unwrap_or(end - start));
                if first_start.is_none() {
                    first_start = Some(start);
                }
                last_end = Some(end);
                if let Some(prev) = prev_end {
                    if start > prev {
                        gap = gap.saturating_add(start - prev);
                        gap_count += 1;
                    } else if start < prev {
                        overlap = overlap.saturating_add(prev - start);
                    }
                }
                prev_end = Some(end);
            }
            _ => missing += 1,
        }
    }
    let device_span_ns = match (first_start, last_end) {
        (Some(s), Some(e)) if e > s => Some(e - s),
        _ => None,
    };
    let gpu_idle_in_span_ns = device_span_ns.map(|span| span.saturating_sub(busy));
    let gpu_idle_fraction_of_span = device_span_ns.map(|span| {
        if span == 0 {
            0.0
        } else {
            span.saturating_sub(busy) as f64 / span as f64
        }
    });
    let occupancy_proxy_gpu_ns_per_encoder = if encoders == 0 {
        None
    } else {
        Some(busy as f64 / encoders as f64)
    };
    GpuGapAccounting {
        timestamp_authority: "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wall proxy",
        intra_cb_kernel_gaps_visible: false,
        observed_timestamp_pairs: observed,
        missing_timestamp_pairs: missing,
        first_gpu_start_ns: first_start,
        last_gpu_end_ns: last_end,
        device_span_ns,
        gpu_busy_ns: busy,
        inter_cb_device_gap_ns: gap,
        inter_cb_device_overlap_ns: overlap,
        inter_cb_gap_count: gap_count,
        gpu_idle_in_span_ns,
        gpu_idle_fraction_of_span,
        host_between_cb_exclusive_ns,
        occupancy_proxy_gpu_ns_per_encoder,
        production_encoders: encoders,
    }
}

fn stage_ns(stages: &BTreeMap<String, StageAccum>, name: &str) -> u64 {
    stages.get(name).map(|s| s.ns).unwrap_or(0)
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (whole as f64)
    }
}

fn finalize_stages(stages: &BTreeMap<String, StageAccum>, body_ns: u64) -> Vec<StageRow> {
    let mut rows: Vec<StageRow> = stages
        .iter()
        .map(|(name, acc)| StageRow {
            name: name.clone(),
            ns: acc.ns,
            ns_per_token: acc.ns,
            pct_body: pct(acc.ns, body_ns),
            calls: acc.calls,
            calls_per_token: acc.calls,
            bytes: acc.bytes,
        })
        .collect();
    rows.sort_by(|a, b| b.ns.cmp(&a.ns).then_with(|| a.name.cmp(&b.name)));
    rows
}

fn operator_class_rows(
    stages: &BTreeMap<String, StageAccum>,
    isolated: &[IsolatedKernelRow],
    body_ns: u64,
) -> Vec<StageRow> {
    let mut classes: BTreeMap<&'static str, StageAccum> = BTreeMap::new();
    let bump = |map: &mut BTreeMap<&'static str, StageAccum>, name: &'static str, ns: u64, calls: u64, bytes: u64| {
        let row = map.entry(name).or_default();
        row.ns = row.ns.saturating_add(ns);
        row.calls = row.calls.saturating_add(calls);
        row.bytes = row.bytes.saturating_add(bytes);
    };
    for (name, acc) in stages {
        let class = if name.starts_with("host.mla.") || name.starts_with("cb.attn") {
            Some("mla")
        } else if name.starts_with("host.routed") || name.starts_with("host.expert") {
            Some("routed_experts")
        } else if name.starts_with("host.shared") {
            Some("shared_expert")
        } else if name.contains("rmsnorm") || name.contains("norm") {
            Some("norms")
        } else if name.starts_with("host.router")
            || name.starts_with("cb.route")
            || name.contains("route_id")
        {
            Some("router_gate")
        } else if name.contains("moe_combine") || name.starts_with("cb.moe") {
            Some("moe_combine_and_ffn")
        } else if name.starts_with("host.lm_head") || name.starts_with("cb.lm_head") {
            Some("lm_head")
        } else if name.starts_with("host.mhc") {
            Some("mhc_host")
        } else {
            None
        };
        if let Some(class) = class {
            bump(&mut classes, class, acc.ns, acc.calls, acc.bytes);
        }
    }
    for row in isolated {
        let class = isolated_class(&row.name);
        bump(
            &mut classes,
            class,
            row.gpu_ns.unwrap_or(row.wait_ns),
            1,
            0,
        );
        let fine = isolated_fine(&row.name);
        bump(&mut classes, fine, row.gpu_ns.unwrap_or(row.wait_ns), 1, 0);
    }
    let mut rows: Vec<StageRow> = classes
        .into_iter()
        .map(|(name, acc)| StageRow {
            name: name.to_owned(),
            ns: acc.ns,
            ns_per_token: acc.ns,
            pct_body: pct(acc.ns, body_ns),
            calls: acc.calls,
            calls_per_token: acc.calls,
            bytes: acc.bytes,
        })
        .collect();
    rows.sort_by(|a, b| b.ns.cmp(&a.ns).then_with(|| a.name.cmp(&b.name)));
    rows
}

fn host_wall_class(name: &str) -> Option<&'static str> {
    if name.starts_with("metal.")
        || name.starts_with("reader.")
        || name.starts_with("probe.")
        || name.ends_with("_bytes")
        || name.ends_with("_prefetched")
        || name == "host.attn_weight_io_prefetch"
    {
        return None;
    }
    if name.contains("identity") || name.contains("digest") || name.contains("verify") {
        Some("validation_identity")
    } else if name.contains("mmap") {
        Some("mmap_page_fault")
    } else if name.contains("memcpy")
        || name.contains("upload")
        || name.contains("prefetch_fill")
        || name.contains("owned_copy")
    {
        Some("memcpy")
    } else if name.contains("tensor_lookup")
        || name.contains("path_resolve")
        || name.contains("address")
    {
        Some("address_lookup")
    } else if name.contains("decode") || name.contains("table") {
        Some("table_construction")
    } else if name.contains("route") {
        Some("route_preparation")
    } else if name.contains("mhc") || name.contains("rmsnorm") {
        Some("state_movement")
    } else if name.contains("readback") || name.contains("sync") {
        Some("synchronization")
    } else if name.contains("_io") || name.contains("embed") {
        Some("file_source_reads")
    } else if name.contains("buffer") || name.contains("ledger") {
        Some("buffer_lifecycle")
    } else {
        None
    }
}

fn host_wall_class_rows(
    stages: &BTreeMap<String, StageAccum>,
    body_ns: u64,
    host_exclusive_ns: u64,
) -> Vec<StageRow> {
    let mut classes: BTreeMap<&'static str, StageAccum> = BTreeMap::new();
    for (name, acc) in stages {
        if let Some(class) = host_wall_class(name) {
            let row = classes.entry(class).or_default();
            row.ns = row.ns.saturating_add(acc.ns);
            row.calls = row.calls.saturating_add(acc.calls);
            row.bytes = row.bytes.saturating_add(acc.bytes);
        }
    }
    let accounted: u64 = classes.values().map(|row| row.ns).sum();
    classes.insert(
        "unaccounted_host_exclusive",
        StageAccum {
            ns: host_exclusive_ns.saturating_sub(accounted),
            calls: 1,
            bytes: 0,
        },
    );
    finalize_stages(
        &classes
            .into_iter()
            .map(|(name, acc)| (name.to_owned(), acc))
            .collect(),
        body_ns,
    )
}

fn reader_parallel_sum_rows(stages: &BTreeMap<String, StageAccum>, body_ns: u64) -> Vec<StageRow> {
    let mut selected: BTreeMap<String, StageAccum> = BTreeMap::new();
    for (name, acc) in stages {
        if name.starts_with("reader.") {
            selected.insert(name.clone(), acc.clone());
        }
    }
    finalize_stages(&selected, body_ns)
}

fn isolated_class(name: &str) -> &'static str {
    if name.contains("wq_a")
        || name.contains("wq_b")
        || name.contains("wkv")
        || name.contains("wo_")
        || name.contains("attn.")
        || name.contains("kv_qat")
        || name.contains("per_head")
    {
        "isolated.mla"
    } else if name.contains("w1") || name.contains("w2") || name.contains("w3") || name.contains("swiglu")
    {
        if name.contains("shared") {
            "isolated.shared_expert"
        } else {
            "isolated.routed_experts"
        }
    } else if name.contains("gate") || name.contains("route") || name.contains("pack") {
        "isolated.router_gate"
    } else if name.contains("combine") {
        "isolated.moe_combine"
    } else if name.contains("rmsnorm") || name.contains("act_quant") || name.contains("cast") {
        "isolated.ephemeral"
    } else if name.contains("lm_head") {
        "isolated.lm_head"
    } else {
        "isolated.other"
    }
}

fn isolated_fine(name: &str) -> &'static str {
    if name.contains("wq_a") {
        "isolated.mla.wq_a"
    } else if name.contains("wq_b") {
        "isolated.mla.wq_b"
    } else if name.contains("wkv") {
        "isolated.mla.wkv"
    } else if name.contains("wo_a") {
        "isolated.mla.wo_a"
    } else if name.contains("wo_b") {
        "isolated.mla.wo_b"
    } else if name.contains("w1") && name.contains("shared") {
        "isolated.shared.w1"
    } else if name.contains("w3") && name.contains("shared") {
        "isolated.shared.w3"
    } else if name.contains("w2") && name.contains("shared") {
        "isolated.shared.w2"
    } else if name.contains("w1") {
        "isolated.routed.w1"
    } else if name.contains("w3") {
        "isolated.routed.w3"
    } else if name.contains("w2") {
        "isolated.routed.w2"
    } else if name.contains("combine") {
        "isolated.moe_combine"
    } else if name.contains("gate") {
        "isolated.router.gate"
    } else if name.contains("lm_head") {
        "isolated.lm_head"
    } else {
        "isolated.other_fine"
    }
}

fn diagnose(
    body_ns: u64,
    host_exclusive_ns: u64,
    metal_wait_ns: u64,
    metal_gpu_ns: u64,
    gpu_observed: u64,
    command_buffers: u64,
    verify_ns: u64,
    cbs: &[CommandBufferRow],
    isolated: &[IsolatedKernelRow],
) -> (TokenBodyDiagnosis, String) {
    if body_ns == 0 {
        return (
            TokenBodyDiagnosis::Insufficient,
            "body_ns is 0; no token body was measured".to_owned(),
        );
    }
    if gpu_observed == 0 && !cbs.is_empty() {
        return (
            TokenBodyDiagnosis::Insufficient,
            format!(
                "GPU timestamps missing on all {command_buffers} production command buffers; cannot separate GPU-bound from wait-bound"
            ),
        );
    }
    let isolated_gpu: u64 = isolated.iter().filter_map(|k| k.gpu_ns).sum();
    let isolated_wait: u64 = isolated.iter().map(|k| k.wait_ns).sum();
    let mean_cb_gpu = if gpu_observed == 0 {
        0
    } else {
        metal_gpu_ns / gpu_observed
    };
    let mean_cb_wait = if command_buffers == 0 {
        0
    } else {
        metal_wait_ns / command_buffers
    };
    let wait_minus_gpu = metal_wait_ns.saturating_sub(metal_gpu_ns);
    let host_pct = pct(host_exclusive_ns, body_ns);
    let gpu_pct = pct(metal_gpu_ns, body_ns);
    let wait_pct = pct(metal_wait_ns, body_ns);
    let verify_pct = pct(verify_ns, body_ns);
    let extra_wait_pct = pct(wait_minus_gpu, body_ns);

    // Host exclusive = body time not inside commit+wait. I/O, MHC, memcpy, upload.
    if host_pct >= 55.0 || verify_pct >= 35.0 {
        return (
            TokenBodyDiagnosis::IoBound,
            format!(
                "host_exclusive={host_exclusive_ns}ns ({host_pct:.1}% of body) verify_ns={verify_ns} ({verify_pct:.1}%); metal_gpu={metal_gpu_ns}ns ({gpu_pct:.1}%) metal_wait={metal_wait_ns}ns ({wait_pct:.1}%). Weight streaming / host memcpy / MHC sit outside GPU occupancy."
            ),
        );
    }
    if extra_wait_pct >= 25.0 && metal_gpu_ns > 0 && wait_minus_gpu > metal_gpu_ns {
        return (
            TokenBodyDiagnosis::SyncBound,
            format!(
                "CPU wait {metal_wait_ns}ns exceeds GPU execution {metal_gpu_ns}ns by {wait_minus_gpu}ns ({extra_wait_pct:.1}% of body) across {command_buffers} CBs (mean wait {mean_cb_wait}ns, mean gpu {mean_cb_gpu}ns). GPU is idle while the CPU blocks on readbacks."
            ),
        );
    }
    if gpu_pct >= 55.0 {
        let isolated_note = if isolated_gpu > 0 {
            format!(
                " isolated-kernel GPU sum={isolated_gpu}ns wait={isolated_wait}ns"
            )
        } else {
            String::new()
        };
        // Many tiny CBs with high GPU occupancy can still be dispatch-bound
        // inside the CB (encoder gaps). Isolated kernels distinguish that.
        if !isolated.is_empty() {
            let isolated_gpu_per = isolated_gpu / isolated.len().max(1) as u64;
            if isolated_gpu_per < 200_000 && command_buffers >= 40 {
                return (
                    TokenBodyDiagnosis::DispatchLatencyBound,
                    format!(
                        "production GPU occupies {gpu_pct:.1}% of body but isolated kernels average {isolated_gpu_per}ns GPU each; {command_buffers} CBs / many encoders are the cost, not arithmetic.{isolated_note}"
                    ),
                );
            }
        }
        return (
            TokenBodyDiagnosis::GpuBound,
            format!(
                "metal_gpu={metal_gpu_ns}ns is {gpu_pct:.1}% of body ({mean_cb_gpu}ns/CB); host_exclusive only {host_pct:.1}%.{isolated_note}"
            ),
        );
    }
    if command_buffers >= 40 && mean_cb_gpu < 2_000_000 && gpu_pct < 55.0 {
        return (
            TokenBodyDiagnosis::DispatchLatencyBound,
            format!(
                "{command_buffers} command buffers, mean GPU {mean_cb_gpu}ns, GPU {gpu_pct:.1}% of body, host {host_pct:.1}%, extra wait {extra_wait_pct:.1}%. Many small submits; GPU is not doing 3s of arithmetic."
            ),
        );
    }
    (
        TokenBodyDiagnosis::Mixed,
        format!(
            "no single class clears the bar: host_exclusive={host_pct:.1}% gpu={gpu_pct:.1}% wait={wait_pct:.1}% extra_wait={extra_wait_pct:.1}% verify={verify_pct:.1}% cbs={command_buffers} isolated_gpu={isolated_gpu}"
        ),
    )
}

fn isolated_mean_gpu(isolated: &[IsolatedKernelRow], name: &str) -> Option<u64> {
    let mut sum = 0u64;
    let mut n = 0u64;
    for row in isolated {
        if row.name == name {
            if let Some(gpu) = row.gpu_ns {
                sum = sum.saturating_add(gpu);
                n += 1;
            }
        }
    }
    if n == 0 {
        None
    } else {
        Some(sum / n)
    }
}

fn build_mla_census(
    serial_group: bool,
    specs: &[MlaDispatchSpec],
    kv_state: Option<MlaKvState>,
    pipeline_limits: &[MlaPipelineLimit],
    cbs: &[CommandBufferRow],
    isolated: &[IsolatedKernelRow],
) -> Option<MlaCensus> {
    let attn: Vec<&CommandBufferRow> = cbs.iter().filter(|cb| cb.name == "attn").collect();
    if attn.is_empty() && specs.is_empty() {
        return None;
    }
    let layers = attn.len() as u64;
    let attn_encoders = attn.iter().map(|cb| cb.encoders).sum::<u64>();
    let attn_dispatches = attn.iter().map(|cb| cb.dispatches).sum::<u64>();
    let attn_gpu_ns = attn.iter().filter_map(|cb| cb.gpu_ns).sum::<u64>();
    let attn_wait_ns = attn.iter().map(|cb| cb.wait_ns).sum::<u64>();
    let attn_encode_ns = attn.iter().map(|cb| cb.encode_ns).sum::<u64>();
    let mut gpus: Vec<u64> = attn.iter().filter_map(|cb| cb.gpu_ns).collect();
    gpus.sort_unstable();
    let attn_gpu_ns_min = gpus.first().copied();
    let attn_gpu_ns_max = gpus.last().copied();
    let attn_gpu_ns_p50 = if gpus.is_empty() {
        None
    } else {
        Some(gpus[gpus.len() / 2])
    };
    let layers_gpu_over_20ms = gpus.iter().filter(|&&ns| ns >= 20_000_000).count() as u64;
    let attn_gpu_ns_per_invocation = if attn_dispatches == 0 {
        None
    } else {
        Some(attn_gpu_ns / attn_dispatches)
    };

    let mut dispatches = specs.to_vec();
    for spec in &mut dispatches {
        let mean = isolated_mean_gpu(isolated, &format!("isolated.{}", spec.name))
            .or_else(|| isolated_mean_gpu(isolated, &format!("isolated.mla.{}", spec.name)))
            .or_else(|| isolated_mean_gpu(isolated, &spec.name));
        spec.isolated_gpu_ns_mean = mean;
        spec.isolated_ns_per_invocation = mean.map(|ns| {
            if spec.invocations_per_layer == 0 {
                ns
            } else {
                ns / spec.invocations_per_layer
            }
        });
        spec.memory_stall_proxy_gbps = mean.and_then(|ns| {
            if ns == 0 {
                None
            } else {
                Some((spec.bytes_read as f64 + spec.bytes_written as f64) / (ns as f64))
            }
        });
    }

    let isolated_mla_gpu_sum_ns = isolated
        .iter()
        .filter(|row| {
            row.name.contains("mla.")
                || row.name.contains("attn.")
                || row.name.contains("kv_qat")
                || row.name.contains("per_head")
                || row.name.contains("act_quant")
                || row.name.contains("rmsnorm")
                || row.name.contains("cast.")
                || row.name.contains("mla.chain")
        })
        .filter(|row| !row.name.contains("chain"))
        .filter_map(|row| row.gpu_ns)
        .sum::<u64>();

    let dispatch_gap_proxy_ns = if isolated_mla_gpu_sum_ns > 0 && layers > 0 {
        let probed_layers = isolated
            .iter()
            .filter(|row| row.name.contains("mla.wo_a") || row.name == "isolated.mla.wo_a")
            .map(|row| row.layer)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            .max(1) as u64;
        let isolated_per_layer = isolated_mla_gpu_sum_ns / probed_layers;
        Some(attn_gpu_ns as i64 - (isolated_per_layer.saturating_mul(layers)) as i64)
    } else {
        None
    };

    let mean_threads = if dispatches.is_empty() {
        0.0
    } else {
        dispatches.iter().map(|d| d.threads as f64).sum::<f64>() / dispatches.len() as f64
    };
    let mean_occupancy = if dispatches.is_empty() {
        0.0
    } else {
        dispatches.iter().map(|d| d.occupancy_proxy).sum::<f64>() / dispatches.len() as f64
    };
    let bytes_per_token: u64 = dispatches
        .iter()
        .map(|d| (d.bytes_read + d.bytes_written).saturating_mul(d.invocations_per_token))
        .sum();
    let bandwidth_proxy_gbps = if attn_gpu_ns == 0 {
        0.0
    } else {
        bytes_per_token as f64 / attn_gpu_ns as f64
    };
    let encoders_per_cb = if layers == 0 {
        0.0
    } else {
        attn_encoders as f64 / layers as f64
    };

    let (limit_class, limit_proof) = if layers_gpu_over_20ms > 0 {
        (
            "structural_blocker".to_owned(),
            format!(
                "{layers_gpu_over_20ms} attn CBs have GPU >= 20ms (min={:?} p50={:?} max={:?}); a single GPU component above 20ms is a structural blocker, not a tuning target",
                attn_gpu_ns_min, attn_gpu_ns_p50, attn_gpu_ns_max
            ),
        )
    } else if encoders_per_cb >= 8.0 && mean_occupancy < 0.25 {
        (
            "gap_and_occupancy".to_owned(),
            format!(
                "attn uses {encoders_per_cb:.1} encoders/CB, mean occupancy_proxy={mean_occupancy:.3} (mean threads={mean_threads:.0}), realized {bandwidth_proxy_gbps:.1} GB/s vs ~750 GB/s ceiling ({:.2}%); serial-group={} . Matches the system-wide 0.79% bandwidth / ~51% idle picture: gaps + starved grids, not a full-width bandwidth roof.",
                100.0 * bandwidth_proxy_gbps / 750.0,
                serial_group
            ),
        )
    } else if encoders_per_cb >= 8.0 {
        (
            "gap".to_owned(),
            format!(
                "attn uses {encoders_per_cb:.1} encoders/CB for {attn_dispatches} dispatches; realized {bandwidth_proxy_gbps:.1} GB/s. Encoder begin/end is the tax, not arithmetic. serial_group={serial_group}"
            ),
        )
    } else if mean_occupancy < 0.25 {
        (
            "occupancy".to_owned(),
            format!(
                "mean occupancy_proxy={mean_occupancy:.3} (mean threads={mean_threads:.0}); WQ-A/WKV/RMSNorm grids are hundreds-to-1k threads. serial_group={serial_group} encoders/CB={encoders_per_cb:.1} realized {bandwidth_proxy_gbps:.1} GB/s"
            ),
        )
    } else if bandwidth_proxy_gbps >= 200.0 {
        (
            "bandwidth".to_owned(),
            format!(
                "realized {bandwidth_proxy_gbps:.1} GB/s on {bytes_per_token} bytes; occupancy_proxy={mean_occupancy:.3} encoders/CB={encoders_per_cb:.1}"
            ),
        )
    } else {
        (
            "mixed".to_owned(),
            format!(
                "encoders/CB={encoders_per_cb:.1} occupancy_proxy={mean_occupancy:.3} realized {bandwidth_proxy_gbps:.1} GB/s serial_group={serial_group}"
            ),
        )
    };

    Some(MlaCensus {
        serial_group,
        layers,
        attn_command_buffers: layers,
        attn_encoders,
        attn_dispatches,
        attn_gpu_ns,
        attn_wait_ns,
        attn_encode_ns,
        attn_gpu_ns_min,
        attn_gpu_ns_p50,
        attn_gpu_ns_max,
        attn_gpu_ns_per_invocation,
        layers_gpu_over_20ms,
        structural_blocker: layers_gpu_over_20ms > 0,
        isolated_mla_gpu_sum_ns,
        dispatch_gap_proxy_ns,
        limit_class,
        limit_proof,
        dispatches,
        kv_state: kv_state.unwrap_or(MlaKvState {
            persistent_device_addressable: false,
            compressed_indexer_loaded: false,
            bos_window_only: true,
            bytes_written_per_layer: 0,
            bytes_read_per_layer: 0,
            bytes_written_per_token: 0,
            bytes_read_per_token: 0,
            device_copies_per_token: 0,
            host_syncs_per_token: 0,
            host_sync_bytes_per_token: 0,
            scratch_overwritten_per_layer: true,
            rebuild_rebind_reallocate_per_token: true,
            note: "census recorded without a kv_state payload".to_owned(),
        }),
        pipeline_limits: pipeline_limits.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_bound_when_host_exclusive_dominates() {
        let mut c = TokenNsCollector::new();
        c.add_stage("host.attn_weight_io", 2_000_000_000, 43, 8_000_000_000);
        let ledger = c.finish(3_000_000_000, 200_000_000, 3_200_000_000, 1_200_000_000);
        assert_eq!(ledger.diagnosis, TokenBodyDiagnosis::IoBound);
        assert!(
            ledger
                .host_wall_classes
                .iter()
                .any(|row| row.name == "file_source_reads" && row.ns == 2_000_000_000)
        );
    }

    #[test]
    fn host_wall_class_skips_metal_and_reader_sum() {
        assert_eq!(host_wall_class("metal.wait"), None);
        assert_eq!(host_wall_class("reader.mmap"), None);
        assert_eq!(host_wall_class("host.attn_weight_io_prefetch"), None);
        assert_eq!(host_wall_class("host.memcpy"), Some("memcpy"));
        assert_eq!(host_wall_class("host.mhc_pre"), Some("state_movement"));
        assert_eq!(host_wall_class("host.expert_slab_io"), Some("file_source_reads"));
    }

    #[test]
    fn mla_census_flags_encoder_gap_and_one_thread_occupancy() {
        let mut c = TokenNsCollector::new();
        c.record_mla_static(
            false,
            vec![MlaDispatchSpec {
                name: "rmsnorm.q".to_owned(),
                kernel: "deepseek_v4_p3a_rmsnorm_bf16_authority".to_owned(),
                invocations_per_layer: 1,
                invocations_per_token: 43,
                threads: 1,
                threadgroups: 1,
                threads_per_threadgroup: 1,
                simdgroups_per_threadgroup: 1,
                bytes_read: 4096,
                bytes_written: 2048,
                approx_flops: 3072,
                occupancy_proxy: (1.0f64 / 32_768.0).min(1.0),
                isolated_gpu_ns_mean: None,
                isolated_ns_per_invocation: None,
                memory_stall_proxy_gbps: None,
            }],
            MlaKvState {
                persistent_device_addressable: false,
                compressed_indexer_loaded: false,
                bos_window_only: true,
                bytes_written_per_layer: 1024,
                bytes_read_per_layer: 1024,
                bytes_written_per_token: 43 * 1024,
                bytes_read_per_token: 43 * 1024,
                device_copies_per_token: 0,
                host_syncs_per_token: 43,
                host_sync_bytes_per_token: 43 * 8192,
                scratch_overwritten_per_layer: true,
                rebuild_rebind_reallocate_per_token: true,
                note: "test".to_owned(),
            },
            Vec::new(),
        );
        for layer in 0..43 {
            c.record_cb(
                "attn",
                Some(layer),
                "test",
                17,
                &MetalBatchTiming {
                    encode_us: 50,
                    submit_us: 10,
                    wait_us: 2_000,
                    host_wall_us: 2_100,
                    gpu_duration_us: Some(4_500),
                    compute_encoders: 17,
                    compute_dispatches: 17,
                    ..MetalBatchTiming::default()
                },
            );
        }
        let ledger = c.finish(1_000_000_000, 160_000_000, 1_200_000_000, 0);
        let mla = ledger.mla.expect("census");
        assert_eq!(mla.layers, 43);
        assert_eq!(mla.attn_encoders, 43 * 17);
        assert!(!mla.structural_blocker);
        assert_eq!(mla.kv_state.device_copies_per_token, 0);
        assert!(
            mla.limit_class == "gap_and_occupancy" || mla.limit_class == "gap",
            "limit_class={}",
            mla.limit_class
        );
    }
}
