//! Adapters from the existing Q80 and DSV4F ledgers into TOKEN_NS.
//!
//! These do not change either collector. They classify each source row so a
//! DSV4F `host.expert_slab_io` and a Q80 `compact_expert_slab_pack` land on
//! the same `routed_expert` stage name.

use serde_json::Value;

use super::schema::{
    ClosureReport, Confidence, CriticalPath, EmitMeta, MeasurementLabel, RemovableOrNecessary,
    ResourceClass, SerialOrOverlappable, TokenNsDocument, TokenNsStage, TokenNsTotals,
    GPU_TIMESTAMP_AUTHORITY, TOKEN_NS_SCHEMA,
};
use super::served_weight::ServedWeightMetrics;
use crate::gravity_deepseek_v4_token_ns_ledger::TokenNsLedger as DsvLedger;
use crate::model::qwen38_token_ns_ledger::{
    Qwen38TokenNsLedger, QWEN38_TOKEN_NS_LEDGER_SCHEMA,
};
use crate::model::qwen80_token_ns_ledger::{
    Qwen80TokenNsLedger, Qwen80TokenNsToken, QWEN80_TOKEN_NS_LEDGER_SCHEMA,
};

#[derive(Clone, Copy)]
struct Class {
    stage: &'static str,
    resource: ResourceClass,
    serial: SerialOrOverlappable,
    removable: RemovableOrNecessary,
    method: &'static str,
}

fn dsv_class(name: &str) -> Class {
    if name.starts_with("reader.") {
        return Class {
            stage: "identity",
            resource: if name.contains("mmap") {
                ResourceClass::Io
            } else {
                ResourceClass::Cpu
            },
            serial: SerialOrOverlappable::ParallelSumNotLatency,
            removable: RemovableOrNecessary::Removable,
            method: "parallel worker-thread sum inside the reader; not token wall",
        };
    }
    if name.starts_with("probe.") {
        return Class {
            stage: "probe",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Removable,
            method: "diagnostic probe; excluded from body closure",
        };
    }
    if name.ends_with("_bytes") || name.ends_with("_prefetched") {
        return Class {
            stage: "counter",
            resource: ResourceClass::Dram,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Removable,
            method: "byte/hit counter, not a duration",
        };
    }
    if name == "metal.gpu" || name == "metal.gpu_missing" {
        return Class {
            stage: "metal_gpu",
            resource: ResourceClass::Gpu,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Necessary,
            method: GPU_TIMESTAMP_AUTHORITY,
        };
    }
    if name == "metal.encode" {
        return Class {
            stage: "metal_encode",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Necessary,
            method: "host Instant around Metal encode",
        };
    }
    if name == "metal.submit" {
        return Class {
            stage: "metal_submit",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Necessary,
            method: "host Instant around command-buffer commit",
        };
    }
    if name == "metal.wait" {
        return Class {
            stage: "metal_wait",
            resource: ResourceClass::Sync,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Necessary,
            method: "host Instant around wait_until_completed; includes GPU + queue delay",
        };
    }
    if name == "metal.cb_overlap_host" || name.starts_with("metal.cb_overlap.") {
        // Raw per-CB hole, or a derived exclusive partition. The adapter
        // inserts the exclusive serial partitions under the three class
        // names below; the raw total stays observational so it cannot
        // double-count with the partition.
        let exclusive = matches!(
            name,
            "metal.cb_overlap.hidden_io"
                | "metal.cb_overlap.buffer_lifecycle"
                | "metal.cb_overlap.other_cpu_feed"
        );
        return Class {
            stage: "cb_overlap",
            resource: if name.contains("hidden_io") {
                ResourceClass::Io
            } else if name.contains("buffer_lifecycle") {
                ResourceClass::Dram
            } else {
                ResourceClass::Cpu
            },
            serial: if exclusive {
                SerialOrOverlappable::Serial
            } else {
                SerialOrOverlappable::Overlappable
            },
            removable: RemovableOrNecessary::Removable,
            method: if exclusive {
                "derived exclusive partition of sum(cb.host_wall) − (encode+submit+wait); SERIAL on the token wall; GPU is concurrently busy"
            } else {
                "raw sum(cb.host_wall) − (encode+submit+wait); observational total, not exclusive"
            },
        };
    }
    if name.contains("expert_slab") {
        return Class {
            stage: "routed_expert",
            resource: ResourceClass::Io,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Removable,
            method: "host Instant around streamed top-6 expert read+fill; GPU-idle",
        };
    }
    if name.contains("memcpy") || name.contains("owned_copy") || name.contains("prefetch_fill") {
        return Class {
            stage: "memcpy",
            resource: ResourceClass::Dram,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Removable,
            method: "nested inside an I/O wall; do not add to the parent",
        };
    }
    if name.contains("attn_weight_io_prefetch") {
        return Class {
            stage: "attention",
            resource: ResourceClass::Io,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Removable,
            method: "same host wall as expert_slab_io; do not add",
        };
    }
    if name.starts_with("host.shared") || name.contains("moe_control_io") {
        return Class {
            stage: "shared_expert",
            resource: ResourceClass::Io,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Removable,
            method: "shared/control I/O overlapped with attention GPU",
        };
    }
    if name == "host.lm_head_upload" {
        return Class {
            stage: "lm_head",
            resource: ResourceClass::Dram,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Removable,
            method: "host write of an LM-head tile into a Metal buffer; GPU-idle; not nested in host.lm_head_io",
        };
    }
    if name.contains("upload") {
        return Class {
            stage: if name.contains("lm_head") {
                "lm_head"
            } else {
                "attention"
            },
            resource: ResourceClass::Dram,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Removable,
            method: "host write into a Metal buffer; nested in the parent I/O wall",
        };
    }
    if name.contains("mhc") {
        return Class {
            stage: "mhc",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Necessary,
            method: "host Instant; SHA-sealed host MHC between command buffers",
        };
    }
    if name.contains("rmsnorm") || name.contains("norm") {
        return Class {
            stage: "norm",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Necessary,
            method: "host Instant around RMSNorm",
        };
    }
    if name == "host.decode_in_cb_overlap" {
        return Class {
            stage: "table",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Overlappable,
            removable: RemovableOrNecessary::Necessary,
            method: "MHC/norm table decode inside preload_moe_io (during attn GPU); already inside metal.cb_overlap.other_cpu_feed",
        };
    }
    if name == "host.decode" {
        return Class {
            stage: "table",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Necessary,
            method: "host Instant around MHC/norm table construction",
        };
    }
    if name.contains("lm_head_io") {
        return Class {
            stage: "lm_head",
            resource: ResourceClass::Io,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Removable,
            method: "host Instant around streamed LM-head tile I/O",
        };
    }
    if name.contains("readback") || name.contains("_sync") {
        return Class {
            stage: "sync",
            resource: ResourceClass::Sync,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Conditional,
            method: "host Instant around a device-to-host readback",
        };
    }
    if name.contains("embed") {
        return Class {
            stage: "embed",
            resource: ResourceClass::Io,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Removable,
            method: "host Instant around embedding I/O",
        };
    }
    if name.contains("attn_weight_io") {
        return Class {
            stage: "attention",
            resource: ResourceClass::Io,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Removable,
            method: "host Instant around layer-0 attention weight I/O",
        };
    }
    Class {
        stage: "host",
        resource: ResourceClass::Cpu,
        serial: SerialOrOverlappable::Overlappable,
        removable: RemovableOrNecessary::Conditional,
        method: "host Instant; unclassified DSV4F row, treated as non-exclusive",
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CbHole {
    total: u64,
    attn: u64,
    moe: u64,
    other: u64,
}

fn cb_holes_from_rows<'a, I>(rows: I) -> CbHole
where
    I: IntoIterator<Item = (&'a str, u64, u64, u64, u64)>,
{
    let mut holes = CbHole::default();
    for (name, wall, encode, submit, wait) in rows {
        let hole = wall.saturating_sub(encode.saturating_add(submit).saturating_add(wait));
        holes.total = holes.total.saturating_add(hole);
        if name == "attn" {
            holes.attn = holes.attn.saturating_add(hole);
        } else if name == "moe" {
            holes.moe = holes.moe.saturating_add(hole);
        } else {
            holes.other = holes.other.saturating_add(hole);
        }
    }
    holes
}

fn cb_holes_from_ledger(ledger: &DsvLedger) -> CbHole {
    cb_holes_from_rows(ledger.command_buffers.iter().map(|cb| {
        (
            cb.name.as_str(),
            cb.host_wall_ns,
            cb.encode_ns,
            cb.submit_ns,
            cb.wait_ns,
        )
    }))
}

fn cb_holes_from_json(ledger: &Value) -> CbHole {
    let Some(arr) = ledger.get("command_buffers").and_then(|v| v.as_array()) else {
        return CbHole::default();
    };
    cb_holes_from_rows(arr.iter().map(|cb| {
        (
            json_str(cb, &["name"]),
            json_u64(cb, &["host_wall_ns"]),
            json_u64(cb, &["encode_ns"]),
            json_u64(cb, &["submit_ns"]),
            json_u64(cb, &["wait_ns"]),
        )
    }))
}

fn stage_ns_named(stages: &[TokenNsStage], name: &str) -> u64 {
    stages
        .iter()
        .filter(|s| s.substage == name)
        .map(|s| s.ns_per_token.round() as u64)
        .sum()
}

/// Partition the already-measured CB host-wall hole into exclusive serial
/// classes. The named overlappable rows (shared I/O, prefetch_fill) identify
/// the class; they stay overlappable so they are not double-counted.
fn ensure_dsv_overlap_serial(
    stages: &mut Vec<TokenNsStage>,
    holes: CbHole,
    total_token_ns: u64,
    commit: &str,
) {
    if holes.total == 0 {
        return;
    }
    if stages
        .iter()
        .any(|s| s.substage == "metal.cb_overlap.hidden_io")
    {
        return;
    }
    let shared_and_control = stage_ns_named(stages, "host.shared.w1_io")
        .saturating_add(stage_ns_named(stages, "host.shared.w2_io"))
        .saturating_add(stage_ns_named(stages, "host.shared.w3_io"))
        .saturating_add(stage_ns_named(stages, "host.moe_control_io"))
        .saturating_add(stage_ns_named(stages, "host.expert_slab_io_overlapped"));
    let prefetch_fill = stage_ns_named(stages, "host.mla.prefetch_fill");
    let hidden_io = holes.attn.min(shared_and_control);
    let buffer_lifecycle = holes.moe.min(prefetch_fill);
    let other = holes
        .total
        .saturating_sub(hidden_io)
        .saturating_sub(buffer_lifecycle);
    let commit = commit.to_owned();
    let push = |stages: &mut Vec<TokenNsStage>,
                substage: &str,
                ns: u64,
                resource: ResourceClass,
                method: &str| {
        if ns == 0 {
            return;
        }
        stages.push(TokenNsStage::new(
            "cb_overlap",
            substage,
            holes.total.max(1) as f64,
            ns as f64,
            total_token_ns,
            resource,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Removable,
            Confidence::Measured,
            method,
            commit.clone(),
        ));
    };
    push(
        stages,
        "metal.cb_overlap.hidden_io",
        hidden_io,
        ResourceClass::Io,
        "hidden IO: host.shared.w1/w2/w3 + moe_control (+ overlapped expert bind) during attn GPU; min(attn_hole, named I/O). GPU-overlapped but SERIAL on the token wall because body includes cb.host_wall",
    );
    push(
        stages,
        "metal.cb_overlap.buffer_lifecycle",
        buffer_lifecycle,
        ResourceClass::Dram,
        "buffer lifecycle: host.mla.prefetch_fill of next-layer attn weights during moe GPU; min(moe_hole, prefetch_fill)",
    );
    push(
        stages,
        "metal.cb_overlap.other_cpu_feed",
        other,
        ResourceClass::Cpu,
        "other, explicitly named: in-wall MHC/norm decode + Metal buffer writes + GPU-timestamp query after wait + Instant us truncation. remainder of host_wall − (encode+submit+wait) after hidden_io and buffer_lifecycle",
    );
}

fn q80_host_work_class(name: &str) -> Class {
    if name.contains("slab") || name.contains("pack") || name.contains("expert") {
        Class {
            stage: "routed_expert",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Removable,
            method: "host Instant around compact expert-slab pack",
        }
    } else {
        Class {
            stage: "host",
            resource: ResourceClass::Cpu,
            serial: SerialOrOverlappable::Serial,
            removable: RemovableOrNecessary::Conditional,
            method: "host Instant around host-visible work",
        }
    }
}

fn confidence(method_is_gpu: bool) -> Confidence {
    if method_is_gpu {
        Confidence::Measured
    } else {
        Confidence::Measured
    }
}

pub fn from_dsv4f_ledger(ledger: &DsvLedger, meta: &EmitMeta) -> TokenNsDocument {
    let total = ledger.body_ns;
    let mut stages = Vec::new();
    for row in &ledger.stages {
        let class = dsv_class(&row.name);
        stages.push(TokenNsStage::new(
            class.stage,
            row.name.clone(),
            row.calls_per_token as f64,
            row.ns_per_token as f64,
            total,
            class.resource,
            class.serial,
            class.removable,
            confidence(class.resource == ResourceClass::Gpu),
            class.method,
            meta.commit.clone(),
        ));
    }
    ensure_dsv_overlap_serial(
        &mut stages,
        cb_holes_from_ledger(ledger),
        total,
        &meta.commit,
    );
    let gpu_busy = ledger.metal_vs_host.metal_gpu_ns;
    let gpu_gap = ledger.gpu_gaps.inter_cb_device_gap_ns.max(
        ledger
            .metal_vs_host
            .host_between_first_last_cb_gap_ns,
    );
    let gpu_idle = ledger
        .gpu_gaps
        .gpu_idle_in_span_ns
        .unwrap_or_else(|| total.saturating_sub(gpu_busy.min(total)));
    let dram: u64 = ledger
        .stages
        .iter()
        .filter(|s| {
            s.name.contains("_io")
                && !s.name.contains("prefetch")
                && !s.name.starts_with("reader.")
        })
        .map(|s| s.bytes)
        .sum();
    let temp: u64 = ledger
        .stages
        .iter()
        .filter(|s| s.name.contains("memcpy") || s.name.contains("upload"))
        .map(|s| s.bytes)
        .sum();
    let mut notes = vec![
        format!(
            "TOTAL_TOKEN_NS is body_ns ({}), not wall_ns ({}) and not verify_ns ({}). verify_ns is a parallel sum.",
            ledger.body_ns, ledger.wall_ns, ledger.verify_ns
        ),
        format!(
            "metal.encode+submit+wait={} ns. The former residual hole sum(cb.host_wall)-(encode+submit+wait) is now exclusive serial metal.cb_overlap.* (hidden_io / buffer_lifecycle / other_cpu_feed).",
            ledger.metal_vs_host.metal_encode_ns
                + ledger.metal_vs_host.metal_submit_ns
                + ledger.metal_vs_host.metal_wait_ns
        ),
        "host.lm_head_upload is serial GPU-idle tile fill, not nested in host.lm_head_io.".to_owned(),
        "reader.* rows are parallel-thread sums. A parallel sum is not token latency.".to_owned(),
        format!("source diagnosis: {:?}", ledger.diagnosis),
    ];
    if ledger.gpu_gaps.observed_timestamp_pairs == 0 && !ledger.command_buffers.is_empty() {
        notes.push(
            "gpu_gaps missing or empty on this receipt; TOTAL_GPU_GAP_NS falls back to host_between_first_last_cb_gap_ns (host clock, not device)."
                .to_owned(),
        );
    }
    TokenNsDocument {
        schema: TOKEN_NS_SCHEMA,
        model: "dsv4f".to_owned(),
        vehicle: "native BOS token graph (streamed source, top-6 worklist)".to_owned(),
        source_schema: ledger.schema.to_owned(),
        source_path: meta.source_path.clone(),
        measurement_label: meta.measurement_label,
        commit: meta.commit.clone(),
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        residual_limit_fraction: meta.residual_limit_fraction,
        stages,
        totals: TokenNsTotals {
            total_token_ns: total,
            total_gpu_busy_ns: gpu_busy,
            total_gpu_idle_ns: gpu_idle,
            total_gpu_gap_ns: gpu_gap,
            total_cpu_critical_ns: ledger.metal_vs_host.host_exclusive_ns,
            total_dispatches: ledger.metal_vs_host.production_dispatches,
            total_command_buffers: ledger.metal_vs_host.production_command_buffers,
            total_sync_points: ledger.host_syncs.len() as u64,
            total_readbacks: ledger.route_id_readbacks.len() as u64
                + ledger
                    .stages
                    .iter()
                    .filter(|s| s.name.contains("readback"))
                    .map(|s| s.calls_per_token)
                    .sum::<u64>(),
            total_buffer_creations: 0,
            total_buffer_rebinds: 0,
            dram_bytes_per_token: dram,
            temp_bytes_per_token: temp,
        },
        served_weight: ServedWeightMetrics::default(),
        residual_ns: 0,
        physical_token_ns_override: meta.physical_token_ns,
        joules_per_token_in: meta.joules_per_token,
        closure: ClosureReport::compute(total, &[], meta.residual_limit_fraction),
        critical_path: CriticalPath {
            model: "dsv4f".to_owned(),
            token_definition: "body_ns = host_exclusive + sum(cb.host_wall_ns); init_ns is outside"
                .to_owned(),
            serial_ns: 0,
            overlappable_ns: 0,
            parallel_sum_ns: 0,
            top_serial_stages: Vec::new(),
            statement: "DSV4F critical path is GPU-idle expert_slab_io then metal.wait/GPU, plus host MHC between CBs. reader.path_resolve and verify_ns are parallel sums and are not the token."
                .to_owned(),
            warning: Some(
                "Any receipt that quotes path_resolve or verify_ns as token latency is presenting a parallel sum."
                    .to_owned(),
            ),
        },
        notes,
    }
    .seal()
}

fn q80_token_sums(tokens: &[&Qwen80TokenNsToken]) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    let n = tokens.len().max(1) as u64;
    let mut wall = 0u64;
    let mut gpu = 0u64;
    let mut wait = 0u64;
    let mut submit = 0u64;
    let mut encode = 0u64;
    let mut disp = 0u64;
    let mut cbs = 0u64;
    let mut syncs = 0u64;
    let mut host_work = 0u64;
    for t in tokens {
        wall += t.wall_ns;
        for cb in &t.command_buffers {
            gpu += cb.gpu_ns.unwrap_or(0);
            wait += cb.cpu_wait_ns;
            submit += cb.submit_ns;
            encode += cb.encode_ns;
            disp += cb.dispatches;
            cbs += 1;
        }
        syncs += t.host_syncs.len() as u64;
        host_work += t.host_work.iter().map(|w| w.ns).sum::<u64>();
    }
    (
        wall / n,
        gpu / n,
        wait / n,
        submit / n,
        encode / n,
        disp / n,
        cbs / n,
        syncs / n,
        host_work / n,
    )
}

pub fn from_q80_ledger(ledger: &Qwen80TokenNsLedger, meta: &EmitMeta) -> TokenNsDocument {
    let decode: Vec<&Qwen80TokenNsToken> = ledger
        .tokens
        .iter()
        .filter(|t| t.kind == "decode")
        .collect();
    let (wall, gpu, wait, submit, encode, disp, cbs, sync_calls, host_work) =
        if let Some(mean) = ledger.steady_state_mean.as_ref() {
            (
                mean.wall_ns.round() as u64,
                mean.gpu_execution_ns.round() as u64,
                mean.cpu_wait_ns.round() as u64,
                mean.submit_ns.round() as u64,
                mean.encode_ns.round() as u64,
                mean.dispatches.round() as u64,
                mean.command_buffers.round() as u64,
                decode
                    .iter()
                    .map(|t| t.host_syncs.len() as u64)
                    .sum::<u64>()
                    / decode.len().max(1) as u64,
                mean.host_work_ns.round() as u64,
            )
        } else {
            q80_token_sums(&decode)
        };
    let host_sync_ns = ledger
        .diagnosis
        .as_ref()
        .map(|d| d.host_sync_ns)
        .or_else(|| ledger.steady_state_mean.as_ref().map(|m| m.host_sync_ns.round() as u64))
        .unwrap_or(0);
    let total = wall;
    let commit = meta.commit.clone();
    let mut stages = vec![
        TokenNsStage::new(
            "metal_wait",
            "metal.wait",
            cbs as f64,
            wait as f64,
            total,
            ResourceClass::Sync,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "host Instant around wait_until_completed; includes GPU + queue delay",
            commit.clone(),
        ),
        TokenNsStage::new(
            "routed_expert",
            "host.compact_expert_slab_pack",
            48.0,
            host_work as f64,
            total,
            ResourceClass::Cpu,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Removable,
            Confidence::Measured,
            "host Instant around compact expert-slab pack (sum of host_work)",
            commit.clone(),
        ),
        TokenNsStage::new(
            "metal_submit",
            "metal.submit",
            cbs as f64,
            submit as f64,
            total,
            ResourceClass::Cpu,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "host Instant around command-buffer commit",
            commit.clone(),
        ),
        TokenNsStage::new(
            "metal_encode",
            "metal.encode",
            cbs as f64,
            encode as f64,
            total,
            ResourceClass::Cpu,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "host Instant around Metal encode",
            commit.clone(),
        ),
        TokenNsStage::new(
            "sync",
            "host.sync",
            sync_calls as f64,
            host_sync_ns as f64,
            total,
            ResourceClass::Sync,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Conditional,
            Confidence::Measured,
            "host Instant around post-wait sync / readback",
            commit.clone(),
        ),
        TokenNsStage::new(
            "metal_gpu",
            "metal.gpu",
            cbs as f64,
            gpu as f64,
            total,
            ResourceClass::Gpu,
            SerialOrOverlappable::Overlappable,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            GPU_TIMESTAMP_AUTHORITY,
            commit.clone(),
        ),
    ];
    for line in &ledger.ranked_aggregate {
        if line.class.starts_with("command_buffer_gpu:") {
            stages.push(TokenNsStage::new(
                "metal_gpu",
                line.class.clone(),
                line.calls as f64,
                line.ns_total as f64,
                total,
                ResourceClass::Gpu,
                SerialOrOverlappable::Overlappable,
                RemovableOrNecessary::Necessary,
                Confidence::Measured,
                line.source,
                commit.clone(),
            ));
        } else if line.class.starts_with("host_work:") {
            let class = q80_host_work_class(&line.class);
            stages.push(TokenNsStage::new(
                class.stage,
                line.class.clone(),
                line.calls as f64,
                line.ns_total as f64,
                total,
                class.resource,
                SerialOrOverlappable::Overlappable,
                class.removable,
                Confidence::Measured,
                line.source,
                commit.clone(),
            ));
        }
    }
    let dram = ledger
        .diagnosis
        .as_ref()
        .map(|d| d.weight_bytes_per_token)
        .unwrap_or(ledger.theoretical_weight_bytes.total_bytes);
    let wait_minus_gpu = wait.saturating_sub(gpu);
    TokenNsDocument {
        schema: TOKEN_NS_SCHEMA,
        model: "q80".to_owned(),
        vehicle: ledger.vehicle.to_owned(),
        source_schema: QWEN80_TOKEN_NS_LEDGER_SCHEMA.to_owned(),
        source_path: meta.source_path.clone(),
        measurement_label: meta.measurement_label,
        commit: meta.commit.clone(),
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        residual_limit_fraction: meta.residual_limit_fraction,
        stages,
        totals: TokenNsTotals {
            total_token_ns: total,
            total_gpu_busy_ns: gpu,
            total_gpu_idle_ns: total.saturating_sub(gpu),
            total_gpu_gap_ns: wait_minus_gpu,
            total_cpu_critical_ns: host_work + host_sync_ns + encode + submit,
            total_dispatches: disp,
            total_command_buffers: cbs,
            total_sync_points: cbs,
            total_readbacks: sync_calls,
            total_buffer_creations: 0,
            total_buffer_rebinds: 0,
            dram_bytes_per_token: dram,
            temp_bytes_per_token: 16_711_680,
        },
        served_weight: ServedWeightMetrics::default(),
        residual_ns: 0,
        physical_token_ns_override: meta.physical_token_ns,
        joules_per_token_in: meta.joules_per_token,
        closure: ClosureReport::compute(total, &[], meta.residual_limit_fraction),
        critical_path: CriticalPath {
            model: "q80".to_owned(),
            token_definition: "decode token wall_ns (steady-state mean of kind=decode)".to_owned(),
            serial_ns: 0,
            overlappable_ns: 0,
            parallel_sum_ns: 0,
            top_serial_stages: Vec::new(),
            statement: "Q80 decode critical path is host compact_expert_slab_pack plus metal.wait (which already contains GPU). Prefix/suffix CBs are mixed; GPU time is not split across operators."
                .to_owned(),
            warning: Some(
                "ASCENT_STATE 403 ms/token is a single CPU-wall run with 1637 fallbacks. It is not this document."
                    .to_owned(),
            ),
        },
        notes: vec![
            "metal.gpu is observational and excluded from closure because metal.wait already covers that wall."
                .to_owned(),
            "ranked host_work rows are emitted as overlappable observations so they are not double-counted with the exclusive host.compact_expert_slab_pack stage."
                .to_owned(),
            ledger
                .diagnosis
                .as_ref()
                .map(|d| format!("source verdict: {} — {}", d.verdict, d.rationale))
                .unwrap_or_else(|| "no diagnosis on this ledger".to_owned()),
        ],
    }
    .seal()
}

/// Adapt a sealed Qwen3.8 TOKEN_NS ledger. Every required component is a
/// serial exclusive row; the named residual is already inside `components`.
pub fn from_qwen38_ledger(ledger: &Qwen38TokenNsLedger, meta: &EmitMeta) -> TokenNsDocument {
    let total = ledger.median_wall_ns;
    let commit = meta.commit.clone();
    let stages: Vec<TokenNsStage> = ledger
        .components
        .iter()
        .map(|c| {
            TokenNsStage::new(
                c.component.clone(),
                c.component.clone(),
                c.dispatches as f64,
                c.ns_per_token,
                total,
                match c.resource_class {
                    "cpu" => ResourceClass::Cpu,
                    "sync" => ResourceClass::Sync,
                    "dram" => ResourceClass::Dram,
                    _ => ResourceClass::Gpu,
                },
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Necessary,
                match c.confidence {
                    "derived" => Confidence::Derived,
                    "estimated" => Confidence::Estimated,
                    _ => Confidence::Measured,
                },
                c.method.clone(),
                commit.clone(),
            )
        })
        .collect();
    TokenNsDocument {
        schema: TOKEN_NS_SCHEMA,
        model: "qwen38".to_owned(),
        vehicle: ledger.vehicle.to_owned(),
        source_schema: QWEN38_TOKEN_NS_LEDGER_SCHEMA.to_owned(),
        source_path: meta.source_path.clone(),
        measurement_label: meta.measurement_label,
        commit: meta.commit.clone(),
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        residual_limit_fraction: meta.residual_limit_fraction,
        stages,
        totals: TokenNsTotals {
            total_token_ns: total,
            total_gpu_busy_ns: ledger.median_gpu_ns,
            total_gpu_idle_ns: total.saturating_sub(ledger.median_gpu_ns),
            total_gpu_gap_ns: ledger.wait_minus_gpu_ns.max(0) as u64,
            total_cpu_critical_ns: ledger.median_encode_ns + ledger.median_submit_ns,
            total_dispatches: ledger.dispatches.total,
            total_command_buffers: ledger.dispatches.production_command_buffers,
            total_sync_points: 1,
            total_readbacks: 1,
            total_buffer_creations: 0,
            total_buffer_rebinds: 0,
            dram_bytes_per_token: ledger.weight_bytes.active_bytes,
            temp_bytes_per_token: ledger.state_bytes.total_rw_bytes,
        },
        served_weight: ServedWeightMetrics::default(),
        residual_ns: 0,
        physical_token_ns_override: meta.physical_token_ns,
        joules_per_token_in: meta.joules_per_token,
        closure: ClosureReport::compute(total, &[], meta.residual_limit_fraction),
        critical_path: CriticalPath {
            model: "qwen38".to_owned(),
            token_definition: "complete token wall = encode + submit + wait; GPU is GPUEnd-GPUStart inside wait".to_owned(),
            serial_ns: 0,
            overlappable_ns: 0,
            parallel_sum_ns: 0,
            top_serial_stages: Vec::new(),
            statement: "Qwen3.8 is GPU-bound: wait ≈ gpu + 0.4 ms. The exclusive cover is the twelve named physical components.".to_owned(),
            warning: Some(
                "uniform-q4-v1 is an oracle for profiling only, never an endpoint.".to_owned(),
            ),
        },
        notes: ledger.notes.clone(),
    }
    .seal()
}

fn json_u64(v: &Value, path: &[&str]) -> u64 {
    let mut cur = v;
    for key in path {
        cur = match cur.get(*key) {
            Some(next) => next,
            None => return 0,
        };
    }
    cur.as_u64()
        .or_else(|| cur.as_i64().map(|i| i.max(0) as u64))
        .or_else(|| cur.as_f64().map(|f| f.max(0.0).round() as u64))
        .unwrap_or(0)
}

fn json_f64(v: &Value, path: &[&str]) -> f64 {
    let mut cur = v;
    for key in path {
        cur = match cur.get(*key) {
            Some(next) => next,
            None => return 0.0,
        };
    }
    cur.as_f64()
        .or_else(|| cur.as_u64().map(|u| u as f64))
        .or_else(|| cur.as_i64().map(|i| i as f64))
        .unwrap_or(0.0)
}

fn json_str<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    let mut cur = v;
    for key in path {
        cur = match cur.get(*key) {
            Some(next) => next,
            None => return "",
        };
    }
    cur.as_str().unwrap_or("")
}

/// Adapt a DSV4F ledger JSON object (the ledger itself, or a native report
/// that nests `token_ns_ledger`).
pub fn from_dsv4f_json(root: &Value, meta: &EmitMeta) -> Result<TokenNsDocument, String> {
    let ledger = if root.get("body_ns").is_some() && root.get("stages").is_some() {
        root
    } else if root.get("token_ns_ledger").is_some() {
        root.get("token_ns_ledger")
            .ok_or_else(|| "token_ns_ledger missing".to_owned())?
    } else {
        return Err("not a DSV4F token-ns ledger or native report".to_owned());
    };
    let total = json_u64(ledger, &["body_ns"]);
    if total == 0 {
        return Err("body_ns is 0".to_owned());
    }
    let mut stages = Vec::new();
    if let Some(arr) = ledger.get("stages").and_then(|s| s.as_array()) {
        for row in arr {
            let name = json_str(row, &["name"]);
            if name.is_empty() {
                continue;
            }
            let class = dsv_class(name);
            let ns = json_f64(row, &["ns_per_token"]).max(json_f64(row, &["ns"]));
            let calls = json_f64(row, &["calls_per_token"]).max(json_f64(row, &["calls"]));
            stages.push(TokenNsStage::new(
                class.stage,
                name,
                calls,
                ns,
                total,
                class.resource,
                class.serial,
                class.removable,
                Confidence::Measured,
                class.method,
                meta.commit.clone(),
            ));
        }
    }
    ensure_dsv_overlap_serial(
        &mut stages,
        cb_holes_from_json(ledger),
        total,
        &meta.commit,
    );
    let gpu_busy = json_u64(ledger, &["metal_vs_host", "metal_gpu_ns"]);
    let gpu_gap = json_u64(ledger, &["gpu_gaps", "inter_cb_device_gap_ns"]).max(json_u64(
        ledger,
        &["metal_vs_host", "host_between_first_last_cb_gap_ns"],
    ));
    let gpu_idle = {
        let from_gaps = json_u64(ledger, &["gpu_gaps", "gpu_idle_in_span_ns"]);
        if from_gaps > 0 {
            from_gaps
        } else {
            total.saturating_sub(gpu_busy.min(total))
        }
    };
    let mut dram = 0u64;
    let mut temp = 0u64;
    if let Some(arr) = ledger.get("stages").and_then(|s| s.as_array()) {
        for row in arr {
            let name = json_str(row, &["name"]);
            let bytes = json_u64(row, &["bytes"]);
            if name.contains("_io") && !name.contains("prefetch") && !name.starts_with("reader.") {
                dram = dram.saturating_add(bytes);
            }
            if name.contains("memcpy") || name.contains("upload") {
                temp = temp.saturating_add(bytes);
            }
        }
    }
    let cbs = json_u64(ledger, &["metal_vs_host", "production_command_buffers"]);
    let disp = json_u64(ledger, &["metal_vs_host", "production_dispatches"]);
    let host_exclusive = json_u64(ledger, &["metal_vs_host", "host_exclusive_ns"]);
    let syncs = ledger
        .get("host_syncs")
        .and_then(|s| s.as_array())
        .map(|a| a.len() as u64)
        .unwrap_or(0);
    let readbacks = ledger
        .get("route_id_readbacks")
        .and_then(|s| s.as_array())
        .map(|a| a.len() as u64)
        .unwrap_or(0);
    let buffer_creates = json_u64(root, &["metal", "total_buffer_creations"]);
    let buffer_rebinds = json_u64(root, &["metal", "total_buffer_rebinds"]);
    let mapped = json_u64(root, &["chunk_verification", "host_read", "mapped_window_bytes"]);
    Ok(TokenNsDocument {
        schema: TOKEN_NS_SCHEMA,
        model: "dsv4f".to_owned(),
        vehicle: "native BOS token graph (streamed source, top-6 worklist)".to_owned(),
        source_schema: json_str(ledger, &["schema"]).to_owned(),
        source_path: meta.source_path.clone(),
        measurement_label: meta.measurement_label,
        commit: meta.commit.clone(),
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        residual_limit_fraction: meta.residual_limit_fraction,
        stages,
        totals: TokenNsTotals {
            total_token_ns: total,
            total_gpu_busy_ns: gpu_busy,
            total_gpu_idle_ns: gpu_idle,
            total_gpu_gap_ns: gpu_gap,
            total_cpu_critical_ns: host_exclusive,
            total_dispatches: disp,
            total_command_buffers: cbs,
            total_sync_points: if syncs > 0 { syncs } else { cbs },
            total_readbacks: readbacks,
            total_buffer_creations: buffer_creates,
            total_buffer_rebinds: buffer_rebinds,
            dram_bytes_per_token: if dram > 0 { dram } else { mapped },
            temp_bytes_per_token: temp,
        },
        served_weight: ServedWeightMetrics::default(),
        residual_ns: 0,
        physical_token_ns_override: meta.physical_token_ns,
        joules_per_token_in: meta.joules_per_token,
        closure: ClosureReport::compute(total, &[], meta.residual_limit_fraction),
        critical_path: CriticalPath {
            model: "dsv4f".to_owned(),
            token_definition: "body_ns = host_exclusive + sum(cb.host_wall_ns)".to_owned(),
            serial_ns: 0,
            overlappable_ns: 0,
            parallel_sum_ns: 0,
            top_serial_stages: Vec::new(),
            statement: "DSV4F critical path is GPU-idle expert_slab_io then metal.wait/GPU, plus host MHC between CBs. reader.path_resolve and verify_ns are parallel sums and are not the token."
                .to_owned(),
            warning: Some(
                "Any receipt that quotes path_resolve or verify_ns as token latency is presenting a parallel sum."
                    .to_owned(),
            ),
        },
        notes: vec![
            format!(
                "verify_ns={} is a parallel sum ({}% of body) and is not in TOTAL_TOKEN_NS.",
                json_u64(ledger, &["verify_ns"]),
                if total == 0 {
                    0.0
                } else {
                    json_u64(ledger, &["verify_ns"]) as f64 * 100.0 / total as f64
                }
            ),
            "Named serial now includes metal.cb_overlap.* = sum(cb.host_wall) − (encode+submit+wait), partitioned into hidden_io / buffer_lifecycle / other_cpu_feed. host.lm_head_upload is serial GPU-idle tile fill."
                .to_owned(),
        ],
    }
    .seal())
}

/// Adapt the Q80 TOKEN_NS ledger JSON (has diagnosis + steady_state_mean).
pub fn from_q80_json(root: &Value, meta: &EmitMeta) -> Result<TokenNsDocument, String> {
    if json_str(root, &["schema"]) != QWEN80_TOKEN_NS_LEDGER_SCHEMA
        && root.get("steady_state_mean").is_none()
        && root.get("diagnosis").is_none()
    {
        return Err("not a Q80 token-ns ledger".to_owned());
    }
    let total = json_u64(root, &["diagnosis", "wall_ns"])
        .max(json_u64(root, &["steady_state_mean", "wall_ns"]));
    if total == 0 {
        return Err("Q80 wall_ns is 0".to_owned());
    }
    let gpu = json_u64(root, &["diagnosis", "gpu_execution_ns"])
        .max(json_u64(root, &["steady_state_mean", "gpu_execution_ns"]));
    let wait = json_u64(root, &["diagnosis", "cpu_wait_ns"])
        .max(json_u64(root, &["steady_state_mean", "cpu_wait_ns"]));
    let submit = json_u64(root, &["diagnosis", "submit_ns"])
        .max(json_u64(root, &["steady_state_mean", "submit_ns"]));
    let encode = json_u64(root, &["diagnosis", "encode_ns"])
        .max(json_u64(root, &["steady_state_mean", "encode_ns"]));
    let host_work = json_u64(root, &["diagnosis", "host_work_ns"])
        .max(json_u64(root, &["steady_state_mean", "host_work_ns"]));
    let host_sync = json_u64(root, &["diagnosis", "host_sync_ns"])
        .max(json_u64(root, &["steady_state_mean", "host_sync_ns"]));
    let cbs = json_u64(root, &["diagnosis", "command_buffers_per_token"])
        .max(json_f64(root, &["steady_state_mean", "command_buffers"]).round() as u64);
    let disp = json_u64(root, &["diagnosis", "dispatches_per_token"])
        .max(json_f64(root, &["steady_state_mean", "dispatches"]).round() as u64);
    let dram = json_u64(root, &["diagnosis", "weight_bytes_per_token"])
        .max(json_u64(root, &["theoretical_weight_bytes", "total_bytes"]));
    let commit = meta.commit.clone();
    let stages = vec![
        TokenNsStage::new(
            "metal_wait",
            "metal.wait",
            cbs as f64,
            wait as f64,
            total,
            ResourceClass::Sync,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "host Instant around wait_until_completed; includes GPU + queue delay",
            commit.clone(),
        ),
        TokenNsStage::new(
            "routed_expert",
            "host.compact_expert_slab_pack",
            48.0,
            host_work as f64,
            total,
            ResourceClass::Cpu,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Removable,
            Confidence::Measured,
            "host Instant around compact expert-slab pack",
            commit.clone(),
        ),
        TokenNsStage::new(
            "metal_submit",
            "metal.submit",
            cbs as f64,
            submit as f64,
            total,
            ResourceClass::Cpu,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "host Instant around command-buffer commit",
            commit.clone(),
        ),
        TokenNsStage::new(
            "metal_encode",
            "metal.encode",
            cbs as f64,
            encode as f64,
            total,
            ResourceClass::Cpu,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            "host Instant around Metal encode",
            commit.clone(),
        ),
        TokenNsStage::new(
            "sync",
            "host.sync",
            1.0,
            host_sync as f64,
            total,
            ResourceClass::Sync,
            SerialOrOverlappable::Serial,
            RemovableOrNecessary::Conditional,
            Confidence::Measured,
            "host Instant around post-wait sync",
            commit.clone(),
        ),
        TokenNsStage::new(
            "metal_gpu",
            "metal.gpu",
            cbs as f64,
            gpu as f64,
            total,
            ResourceClass::Gpu,
            SerialOrOverlappable::Overlappable,
            RemovableOrNecessary::Necessary,
            Confidence::Measured,
            GPU_TIMESTAMP_AUTHORITY,
            commit,
        ),
    ];
    Ok(TokenNsDocument {
        schema: TOKEN_NS_SCHEMA,
        model: "q80".to_owned(),
        vehicle: json_str(root, &["vehicle"]).to_owned(),
        source_schema: json_str(root, &["schema"]).to_owned(),
        source_path: meta.source_path.clone(),
        measurement_label: meta.measurement_label,
        commit: meta.commit.clone(),
        gpu_timestamp_authority: GPU_TIMESTAMP_AUTHORITY,
        residual_limit_fraction: meta.residual_limit_fraction,
        stages,
        totals: TokenNsTotals {
            total_token_ns: total,
            total_gpu_busy_ns: gpu,
            total_gpu_idle_ns: total.saturating_sub(gpu),
            total_gpu_gap_ns: wait.saturating_sub(gpu),
            total_cpu_critical_ns: host_work + host_sync + encode + submit,
            total_dispatches: disp,
            total_command_buffers: cbs,
            total_sync_points: cbs,
            total_readbacks: 0,
            total_buffer_creations: 0,
            total_buffer_rebinds: 0,
            dram_bytes_per_token: dram,
            temp_bytes_per_token: 16_711_680,
        },
        served_weight: ServedWeightMetrics::default(),
        residual_ns: 0,
        physical_token_ns_override: meta.physical_token_ns,
        joules_per_token_in: meta.joules_per_token,
        closure: ClosureReport::compute(total, &[], meta.residual_limit_fraction),
        critical_path: CriticalPath {
            model: "q80".to_owned(),
            token_definition: "decode token wall_ns".to_owned(),
            serial_ns: 0,
            overlappable_ns: 0,
            parallel_sum_ns: 0,
            top_serial_stages: Vec::new(),
            statement: "Q80 decode critical path is host compact_expert_slab_pack plus metal.wait."
                .to_owned(),
            warning: None,
        },
        notes: vec![format!(
            "source verdict: {}",
            json_str(root, &["diagnosis", "verdict"])
        )],
    }
    .seal())
}

/// Adapt Q80_BASELINE_2026_08_16.json: the 15.23 s named-stage sum vs the
/// 15.60 s run. This is the residual the lane brief names.
pub fn from_q80_baseline_run_json(root: &Value, meta: &EmitMeta) -> Result<TokenNsDocument, String> {
    let prefill = json_f64(root, &["timing", "prefill_secs"]);
    let decode = json_f64(root, &["timing", "decode_secs"]);
    let tokens = json_f64(root, &["timing", "steady_state_tokens"]).max(1.0);
    let total_run_ns = ((prefill + decode) * 1e9).round() as u64;
    if total_run_ns == 0 {
        return Err("Q80 baseline run has no timing".to_owned());
    }
    let stages_obj = root
        .get("execution")
        .and_then(|e| e.get("stages"))
        .and_then(|s| s.as_object())
        .ok_or_else(|| "Q80 baseline missing execution.stages".to_owned())?;
    let mut stages = Vec::new();
    let commit = meta.commit.clone();
    for (name, val) in stages_obj {
        let secs = val.as_f64().unwrap_or(0.0);
        if secs <= 0.0 {
            continue;
        }
        let ns = secs * 1e9;
        let (stage, serial, removable, resource) = if name.contains("deltanet") {
            (
                "attention",
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Necessary,
                ResourceClass::Gpu,
            )
        } else if name.contains("gqa") {
            (
                "attention",
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Necessary,
                ResourceClass::Gpu,
            )
        } else if name.contains("table_build") {
            (
                "routed_expert",
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Removable,
                ResourceClass::Cpu,
            )
        } else if name.contains("combine") {
            (
                "moe_combine",
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Necessary,
                ResourceClass::Gpu,
            )
        } else if name.contains("embed") {
            (
                "embed",
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Necessary,
                ResourceClass::Io,
            )
        } else {
            (
                "host",
                SerialOrOverlappable::Serial,
                RemovableOrNecessary::Conditional,
                ResourceClass::Cpu,
            )
        };
        stages.push(TokenNsStage::new(
            stage,
            name.clone(),
            1.0,
            ns,
            total_run_ns,
            resource,
            serial,
            removable,
            Confidence::Measured,
            "host Instant bucket over the whole baseline run (prefill + decode); CPU wall, not GPU timestamps",
            commit.clone(),
        ));
    }
    let decode_ns_per_token = ((decode / tokens) * 1e9).round() as u64;
    Ok(TokenNsDocument {
        schema: TOKEN_NS_SCHEMA,
        model: "q80".to_owned(),
        vehicle: "uniform-Q4 group64 v1 hybrid greedy (Q80_BASELINE_2026_08_16 whole run)".to_owned(),
        source_schema: json_str(root, &["schema"]).to_owned(),
        source_path: meta.source_path.clone(),
        measurement_label: MeasurementLabel::DirtyEngineering,
        commit: meta.commit.clone(),
        gpu_timestamp_authority: "NOT GPU TIME. These buckets are CPU-wall Instant sums over the run.",
        residual_limit_fraction: meta.residual_limit_fraction,
        stages,
        totals: TokenNsTotals {
            total_token_ns: total_run_ns,
            total_gpu_busy_ns: 0,
            total_gpu_idle_ns: 0,
            total_gpu_gap_ns: 0,
            total_cpu_critical_ns: total_run_ns,
            total_dispatches: json_u64(root, &["execution", "native", "q4_matvec_dispatches"]),
            total_command_buffers: 0,
            total_sync_points: 0,
            total_readbacks: 0,
            total_buffer_creations: 0,
            total_buffer_rebinds: 0,
            dram_bytes_per_token: crate::model::qwen80_token_ns_ledger::theoretical_weight_bytes_per_token()
                .total_bytes,
            temp_bytes_per_token: 0,
        },
        served_weight: ServedWeightMetrics::default(),
        residual_ns: 0,
        physical_token_ns_override: meta.physical_token_ns.or(Some(decode_ns_per_token)),
        joules_per_token_in: meta.joules_per_token,
        closure: ClosureReport::compute(total_run_ns, &[], meta.residual_limit_fraction),
        critical_path: CriticalPath {
            model: "q80".to_owned(),
            token_definition: format!(
                "WHOLE RUN (prefill+decode), not per-token. decode/token ≈ {decode_ns_per_token} ns"
            ),
            serial_ns: 0,
            overlappable_ns: 0,
            parallel_sum_ns: 0,
            top_serial_stages: Vec::new(),
            statement: format!(
                "Named stage buckets sum to less than the run. moe_table_build dominates the run. Per-token decode is {decode_ns_per_token} ns and is a different object from this residual."
            ),
            warning: Some(
                "1637 fallbacks. No GPU timestamps. Single run. CROSS_ADVERSARIAL P1-Q80-403MS-NOT-A-LEGAL-MEASUREMENT."
                    .to_owned(),
            ),
        },
        notes: vec![
            format!(
                "decode_secs={decode} over {tokens} tokens → {decode_ns_per_token} ns/token CPU wall"
            ),
            format!(
                "fallbacks.total={}",
                json_u64(root, &["execution", "fallbacks", "total"])
            ),
        ],
    }
    .seal())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_deepseek_v4_token_ns_ledger::TokenNsCollector;

    fn meta() -> EmitMeta {
        EmitMeta::new("testcommit", "test.json")
    }

    #[test]
    fn dsv_reader_sum_is_not_closure() {
        let mut c = TokenNsCollector::new();
        c.add_stage("reader.path_resolve", 2_000_000_000, 3314, 0);
        c.add_stage("host.expert_slab_io", 400_000_000, 43, 1_000);
        c.add_stage("metal.wait", 250_000_000, 137, 0);
        let ledger = c.finish(1_000_000_000, 0, 1_200_000_000, 2_500_000_000);
        let doc = from_dsv4f_ledger(&ledger, &meta());
        let resolve = doc
            .stages
            .iter()
            .find(|s| s.substage == "reader.path_resolve")
            .expect("row");
        assert_eq!(
            resolve.serial_or_overlappable,
            SerialOrOverlappable::ParallelSumNotLatency
        );
        assert!(resolve.pct_of_token > 100.0);
        assert_eq!(doc.closure.sum_serial_stage_ns, 650_000_000);
        assert_eq!(doc.closure.residual_ns, 350_000_000);
        assert!(doc.closure.failed, "350 ms residual on a 1 s token fails 5%");
        assert_eq!(
            doc.closure.sum_serial_stage_ns + doc.closure.residual_ns,
            doc.totals.total_token_ns as i128
        );
        assert_eq!(doc.served_weight.active_weights_per_token, 12_748_587_008);
        assert!(doc.served_weight.honesty.caveat.contains("NOT latency"));
    }

    #[test]
    fn q80_wait_plus_host_work_closes_tightly() {
        let ledger = Qwen80TokenNsLedger {
            schema: QWEN80_TOKEN_NS_LEDGER_SCHEMA,
            vehicle: "test-vehicle",
            production_cb_shape: true,
            gpu_timestamp_authority: "test",
            box_note: "test",
            theoretical_weight_bytes:
                crate::model::qwen80_token_ns_ledger::theoretical_weight_bytes_per_token(),
            theoretical_temp_bytes:
                crate::model::qwen80_token_ns_ledger::theoretical_temp_bytes(64),
            tokens: Vec::new(),
            steady_state_mean: Some(
                crate::model::qwen80_token_ns_ledger::SteadyStateMean {
                    n: 2,
                    wall_ns: 1_000_000.0,
                    gpu_execution_ns: 400_000.0,
                    cpu_wait_ns: 500_000.0,
                    submit_ns: 10_000.0,
                    encode_ns: 5_000.0,
                    host_sync_ns: 1_000.0,
                    host_work_ns: 480_000.0,
                    command_buffers: 10.0,
                    dispatches: 20.0,
                    weight_bytes: 1_000.0,
                    gpu_gap_ns: 0.0,
                    gpu_gap_edges: 0.0,
                },
            ),
            ranked_aggregate: Vec::new(),
            stage_table: Vec::new(),
            identity: None,
            totals_mean_decode: None,
            diagnosis: None,
        };
        let doc = from_q80_ledger(&ledger, &meta());
        assert_eq!(doc.totals.total_token_ns, 1_000_000);
        assert_eq!(doc.closure.sum_serial_stage_ns, 996_000);
        assert_eq!(doc.closure.residual_ns, 4_000);
        assert!(!doc.closure.failed);
        assert!(
            doc.stages
                .iter()
                .any(|s| s.stage == "metal_gpu"
                    && s.serial_or_overlappable == SerialOrOverlappable::Overlappable)
        );
        assert_eq!(doc.served_weight.active_weights_per_token, 3_562_274_816);
        assert!(doc.served_weight.honesty.not_latency);
        assert!(doc.served_weight.fs_per_weight_served > 0.0);
        assert!(!doc.served_weight.energy_available);
    }

    #[test]
    fn q80_baseline_run_emits_the_known_residual() {
        let json = serde_json::json!({
            "schema": "hawking.ascension.qwen80_uniform_q4_velocity_baseline.v1",
            "timing": {
                "prefill_secs": 11.15855,
                "decode_secs": 4.437232,
                "steady_state_tokens": 11
            },
            "execution": {
                "stages": {
                    "deltanet_secs": 3.326900828,
                    "gqa_secs": 1.133528910,
                    "moe_combine_secs": 1.695811704,
                    "moe_table_build_secs": 9.077696634
                },
                "fallbacks": { "total": 1637 }
            }
        });
        let doc = from_q80_baseline_run_json(&json, &meta()).expect("adapt");
        assert!((doc.totals.total_token_ns as f64 - 15.595782e9).abs() < 2_000.0);
        assert!((doc.closure.sum_serial_stage_ns as f64 - 15.233938e9).abs() < 2_000.0);
        assert!(doc.closure.residual_ns > 360_000_000);
        assert!(doc.closure.residual_ns < 370_000_000);
        assert!(
            !doc.closure.failed,
            "2.3% residual must pass the 5% gate: {:?}",
            doc.closure.failure
        );
    }

    #[test]
    fn dsv_r6_shaped_residual_closes_under_five_percent() {
        // Authority R6 body and the two holes this lane named:
        //   sum(cb.host_wall)-(encode+submit+wait) = 221_456_000
        //   host.lm_head_upload = 49_767_957 (was misclassified as nested)
        let body = 1_024_054_500u64;
        let json = serde_json::json!({
            "schema": "hawking.gravity.deepseek_v4.token_ns_ledger.v1",
            "body_ns": body,
            "wall_ns": 1_183_504_667u64,
            "init_ns": 159_000_000u64,
            "verify_ns": 2_333_329_911u64,
            "metal_vs_host": {
                "metal_encode_ns": 4_674_000u64,
                "metal_submit_ns": 753_000u64,
                "metal_wait_ns": 243_374_000u64,
                "metal_gpu_ns": 396_549_000u64,
                "host_exclusive_ns": 553_797_500u64,
                "production_command_buffers": 137u64,
                "production_dispatches": 1857u64
            },
            "stages": [
                {"name": "host.expert_slab_io", "ns_per_token": 406_314_919u64, "calls_per_token": 43, "bytes": 1},
                {"name": "metal.wait", "ns_per_token": 243_374_000u64, "calls_per_token": 137, "bytes": 0},
                {"name": "host.lm_head_io", "ns_per_token": 36_883_000u64, "calls_per_token": 1, "bytes": 1},
                {"name": "host.mhc_pre", "ns_per_token": 25_438_623u64, "calls_per_token": 86, "bytes": 0},
                {"name": "host.decode", "ns_per_token": 19_065_715u64, "calls_per_token": 345, "bytes": 0},
                {"name": "host.attn_weight_io", "ns_per_token": 7_268_333u64, "calls_per_token": 1, "bytes": 1},
                {"name": "metal.encode", "ns_per_token": 4_674_000u64, "calls_per_token": 137, "bytes": 0},
                {"name": "host.mhc_post", "ns_per_token": 3_707_995u64, "calls_per_token": 86, "bytes": 0},
                {"name": "metal.submit", "ns_per_token": 753_000u64, "calls_per_token": 137, "bytes": 0},
                {"name": "host.rmsnorm", "ns_per_token": 726_175u64, "calls_per_token": 86, "bytes": 0},
                {"name": "host.lm_head.mhc_merge", "ns_per_token": 481_958u64, "calls_per_token": 1, "bytes": 0},
                {"name": "host.embed_io", "ns_per_token": 95_416u64, "calls_per_token": 1, "bytes": 1},
                {"name": "host.route_id_readback_sync", "ns_per_token": 70_046u64, "calls_per_token": 40, "bytes": 0},
                {"name": "host.moe_activation_readback", "ns_per_token": 49_085u64, "calls_per_token": 43, "bytes": 0},
                {"name": "host.attn_activation_readback", "ns_per_token": 47_878u64, "calls_per_token": 43, "bytes": 0},
                {"name": "host.lm_head_readback", "ns_per_token": 40_498u64, "calls_per_token": 8, "bytes": 0},
                {"name": "host.route_id_readback", "ns_per_token": 37_461u64, "calls_per_token": 40, "bytes": 0},
                {"name": "host.lm_head_upload", "ns_per_token": 49_767_957u64, "calls_per_token": 8, "bytes": 1},
                {"name": "host.shared.w1_io", "ns_per_token": 35_250_879u64, "calls_per_token": 43, "bytes": 1},
                {"name": "host.shared.w2_io", "ns_per_token": 32_568_457u64, "calls_per_token": 43, "bytes": 1},
                {"name": "host.shared.w3_io", "ns_per_token": 33_002_714u64, "calls_per_token": 43, "bytes": 1},
                {"name": "host.moe_control_io", "ns_per_token": 10_085_041u64, "calls_per_token": 43, "bytes": 1},
                {"name": "host.mla.prefetch_fill", "ns_per_token": 92_228_375u64, "calls_per_token": 42, "bytes": 1},
                {"name": "reader.path_resolve", "ns_per_token": 1_318_045_536u64, "calls_per_token": 3314, "bytes": 0}
            ],
            "command_buffers": [
                {
                    "name": "attn",
                    "host_wall_ns": 216_670_000u64,
                    "encode_ns": 1_510_000u64,
                    "submit_ns": 200_000u64,
                    "wait_ns": 85_910_000u64
                },
                {
                    "name": "moe",
                    "host_wall_ns": 191_530_000u64,
                    "encode_ns": 1_990_000u64,
                    "submit_ns": 300_000u64,
                    "wait_ns": 96_910_000u64
                },
                {
                    "name": "route",
                    "host_wall_ns": 52_660_000u64,
                    "encode_ns": 1_080_000u64,
                    "submit_ns": 200_000u64,
                    "wait_ns": 51_330_000u64
                },
                {
                    "name": "lm_head.tile0",
                    "host_wall_ns": 9_397_000u64,
                    "encode_ns": 94_000u64,
                    "submit_ns": 53_000u64,
                    "wait_ns": 9_224_000u64
                }
            ]
        });
        let doc = from_dsv4f_json(&json, &meta()).expect("adapt");
        let upload = doc
            .stages
            .iter()
            .find(|s| s.substage == "host.lm_head_upload")
            .expect("upload");
        assert_eq!(upload.serial_or_overlappable, SerialOrOverlappable::Serial);
        let hidden = doc
            .stages
            .iter()
            .find(|s| s.substage == "metal.cb_overlap.hidden_io")
            .expect("hidden_io");
        assert_eq!(hidden.serial_or_overlappable, SerialOrOverlappable::Serial);
        let resolve = doc
            .stages
            .iter()
            .find(|s| s.substage == "reader.path_resolve")
            .expect("resolve");
        assert_eq!(
            resolve.serial_or_overlappable,
            SerialOrOverlappable::ParallelSumNotLatency
        );
        assert!(
            doc.closure.identity_holds,
            "identity broken: {:?}",
            doc.closure
        );
        assert!(
            doc.closure.residual_within_limit,
            "residual {} (frac {}) must be under 5%: {:?}",
            doc.closure.residual_ns,
            doc.closure.residual_fraction,
            doc.closure.failure
        );
        assert!(!doc.closure.failed, "{:?}", doc.closure.failure);
        assert!(
            doc.closure.residual_ns >= 0,
            "exclusive set overcounted: {}",
            doc.closure.residual_ns
        );
    }

    #[test]
    fn collector_records_raw_cb_overlap_as_observation() {
        let mut c = TokenNsCollector::new();
        c.record_cb(
            "attn",
            Some(0),
            "test",
            1,
            &crate::metal::MetalBatchTiming {
                encode_us: 100,
                submit_us: 10,
                wait_us: 1_000,
                host_wall_us: 4_000,
                gpu_duration_us: Some(2_000),
                ..crate::metal::MetalBatchTiming::default()
            },
        );
        let ledger = c.finish(10_000_000, 0, 12_000_000, 0);
        let overlap = ledger
            .stages
            .iter()
            .find(|s| s.name == "metal.cb_overlap_host")
            .expect("raw overlap");
        assert_eq!(overlap.ns, 2_890_000);
        let doc = from_dsv4f_ledger(&ledger, &meta());
        let raw = doc
            .stages
            .iter()
            .find(|s| s.substage == "metal.cb_overlap_host")
            .expect("raw");
        assert_eq!(
            raw.serial_or_overlappable,
            SerialOrOverlappable::Overlappable
        );
        let other = doc
            .stages
            .iter()
            .find(|s| s.substage == "metal.cb_overlap.other_cpu_feed")
            .expect("partition");
        assert_eq!(other.serial_or_overlappable, SerialOrOverlappable::Serial);
        assert_eq!(other.ns_per_token, 2_890_000.0);
    }
}
