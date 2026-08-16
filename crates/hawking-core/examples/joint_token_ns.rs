//! Emit unified TOKEN_NS documents for Q80 and DSV4F from existing receipts,
//! enforce closure, reconcile lane claims against the whole token, and flag
//! receipts that present a parallel sum as token latency.
//!
//! No GPU. No runtime hot-path change.

use std::fs;
use std::path::{Path, PathBuf};

use hawking_core::token_ns::{
    ascent_2026_08_16_reports, flag_receipt, from_dsv4f_json, from_q80_baseline_run_json,
    from_q80_json, probe_energy_model, EmitMeta, MeasurementLabel, ReceiptFlag,
    AMORTIZED_CAVEAT, FS_PER_WEIGHT_SERVED_FIELD, TOKEN_NS_SCHEMA,
};

fn read_json(path: &Path) -> serde_json::Value {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn write_json(path: &Path, value: &serde_json::Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    let pretty = serde_json::to_string_pretty(value).expect("serialize");
    fs::write(path, pretty + "\n").unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

fn main() {
    let mut receipts = PathBuf::from("receipts/ascent-2026-08-16");
    let mut q80_ledger = PathBuf::from("receipts/QWEN80_TOKEN_NS_LEDGER.json");
    let mut out = PathBuf::from("receipts/ascent-2026-08-16");
    let mut commit = String::from("unknown");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipts-dir" => receipts = PathBuf::from(args.next().expect("--receipts-dir")),
            "--q80-ledger" => q80_ledger = PathBuf::from(args.next().expect("--q80-ledger")),
            "--out-dir" => out = PathBuf::from(args.next().expect("--out-dir")),
            "--commit" => commit = args.next().expect("--commit"),
            other => panic!("unknown arg {other}"),
        }
    }
    if commit == "unknown" {
        if let Ok(outp) = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
        {
            if outp.status.success() {
                commit = String::from_utf8_lossy(&outp.stdout).trim().to_owned();
            }
        }
    }

    let q80_path = if q80_ledger.exists() {
        q80_ledger
    } else {
        receipts.join("QWEN80_TOKEN_NS_LEDGER.json")
    };
    let q80_baseline_path = receipts.join("Q80_BASELINE_2026_08_16.json");
    let dsv_ledger_path = receipts.join("DSV4F_TOKEN_NS_LEDGER.json");
    let dsv_rep_path = receipts.join("dsv-host-wall-rep6.json");
    let dsv_authority = receipts.join("DSV4F_HOST_WALL_BASELINE.json");

    let mut q80_meta = EmitMeta::new(&commit, q80_path.display().to_string());
    q80_meta.measurement_label = MeasurementLabel::DirtyEngineering;
    let q80_doc = from_q80_json(&read_json(&q80_path), &q80_meta)
        .unwrap_or_else(|e| panic!("adapt Q80 ledger: {e}"));

    let mut q80_run_meta = EmitMeta::new(&commit, q80_baseline_path.display().to_string());
    q80_run_meta.measurement_label = MeasurementLabel::DirtyEngineering;
    let q80_run = from_q80_baseline_run_json(&read_json(&q80_baseline_path), &q80_run_meta)
        .unwrap_or_else(|e| panic!("adapt Q80 baseline: {e}"));

    let dsv_source = if dsv_rep_path.exists() {
        dsv_rep_path.clone()
    } else {
        dsv_ledger_path.clone()
    };
    let mut dsv_meta = EmitMeta::new(&commit, dsv_source.display().to_string());
    dsv_meta.measurement_label = MeasurementLabel::DirtyEngineering;
    let dsv_doc = from_dsv4f_json(&read_json(&dsv_source), &dsv_meta)
        .unwrap_or_else(|e| panic!("adapt DSV4F: {e}"));

    let q80_json = serde_json::to_value(&q80_doc).expect("q80 json");
    let q80_run_json = serde_json::to_value(&q80_run).expect("q80 run json");
    let dsv_json = serde_json::to_value(&dsv_doc).expect("dsv json");
    write_json(&out.join("TOKEN_NS_Q80.json"), &q80_json);
    write_json(&out.join("TOKEN_NS_Q80_BASELINE_RUN.json"), &q80_run_json);
    write_json(&out.join("TOKEN_NS_DSV4F.json"), &dsv_json);

    let reports = ascent_2026_08_16_reports();
    let reconcile_json = serde_json::to_value(&reports).expect("reconcile json");
    write_json(&out.join("TOKEN_NS_RECONCILE.json"), &reconcile_json);

    let mut flags: Vec<ReceiptFlag> = Vec::new();
    let scan_names = [
        "DSV4F_TOKEN_NS_LEDGER.json",
        "DSV4F_HOST_WALL_BASELINE.json",
        "dsv-host-wall-rep6.json",
        "dsv-cb-collapse.json",
        "q80-decode-kernels.json",
        "Q80_BASELINE_2026_08_16.json",
    ];
    for name in scan_names {
        let path = receipts.join(name);
        if path.exists() {
            flags.extend(flag_receipt(&path.display().to_string(), &read_json(&path)));
        }
    }
    flags.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then(a.kind.cmp(&b.kind))
            .then(a.detail.cmp(&b.detail))
    });
    flags.dedup_by(|a, b| a.path == b.path && a.kind == b.kind && a.detail == b.detail);

    let dsv_r = reports.iter().find(|r| r.model == "dsv4f").expect("dsv report");
    let q80_r = reports.iter().find(|r| r.model == "q80").expect("q80 report");
    let energy_probe = probe_energy_model();

    let receipt = serde_json::json!({
        "schema": "hawking.ascent.joint_token_ns.v1",
        "lane": "joint-token-ns",
        "date": "2026-08-16",
        "status": "SHIPPED",
        "timing_label": "N/A",
        "role": "WHOLE_TOKEN_LEDGER_AND_RECONCILER",
        "measurement_law": "This lane does not produce a new GPU timing. Every cited number keeps its original DIRTY_ENGINEERING / CLEAN_CANDIDATE / BASE_TRUE label. A CPU wall is never reported as GPU time.",
        "commit": commit,
        "token_ns_schema": TOKEN_NS_SCHEMA,
        "token_ns": {
            "q80_decode": {
                "source": q80_path.display().to_string(),
                "TOTAL_TOKEN_NS": q80_doc.totals.total_token_ns,
                "TOTAL_GPU_BUSY_NS": q80_doc.totals.total_gpu_busy_ns,
                "TOTAL_CPU_CRITICAL_NS": q80_doc.totals.total_cpu_critical_ns,
                "TOTAL_COMMAND_BUFFERS": q80_doc.totals.total_command_buffers,
                "TOTAL_DISPATCHES": q80_doc.totals.total_dispatches,
                "residual_ns": q80_doc.residual_ns,
                "residual_fraction": q80_doc.closure.residual_fraction,
                "closure_failed": q80_doc.closure.failed,
                "closure_failure": q80_doc.closure.failure,
                "label": q80_doc.measurement_label.as_str(),
                "document": "receipts/ascent-2026-08-16/TOKEN_NS_Q80.json",
                "served_weight": q80_doc.served_weight
            },
            "q80_baseline_run": {
                "source": q80_baseline_path.display().to_string(),
                "TOTAL_TOKEN_NS": q80_run.totals.total_token_ns,
                "sum_serial_stage_ns": q80_run.closure.sum_serial_stage_ns,
                "residual_ns": q80_run.residual_ns,
                "residual_fraction": q80_run.closure.residual_fraction,
                "closure_failed": q80_run.closure.failed,
                "note": "This is the 15.23 s named-stage sum vs the 15.60 s run. TOTAL_TOKEN_NS here is the WHOLE RUN, not per-token.",
                "document": "receipts/ascent-2026-08-16/TOKEN_NS_Q80_BASELINE_RUN.json",
                "served_weight_uses_decode_ns_not_whole_run": q80_run.served_weight.token_ns,
                "served_weight": q80_run.served_weight
            },
            "dsv4f_body": {
                "source": dsv_source.display().to_string(),
                "authority": dsv_authority.display().to_string(),
                "authority_body_ns_median": 1_037_764_208u64,
                "this_receipt_body_ns": dsv_doc.totals.total_token_ns,
                "TOTAL_GPU_BUSY_NS": dsv_doc.totals.total_gpu_busy_ns,
                "TOTAL_CPU_CRITICAL_NS": dsv_doc.totals.total_cpu_critical_ns,
                "TOTAL_COMMAND_BUFFERS": dsv_doc.totals.total_command_buffers,
                "TOTAL_DISPATCHES": dsv_doc.totals.total_dispatches,
                "residual_ns": dsv_doc.residual_ns,
                "residual_fraction": dsv_doc.closure.residual_fraction,
                "closure_failed": dsv_doc.closure.failed,
                "closure_failure": dsv_doc.closure.failure,
                "naive_all_stage_sum_ns": dsv_doc.closure.naive_all_stage_sum_ns,
                "naive_overcount_ns": dsv_doc.closure.naive_overcount_ns,
                "label": dsv_doc.measurement_label.as_str(),
                "document": "receipts/ascent-2026-08-16/TOKEN_NS_DSV4F.json",
                "served_weight": dsv_doc.served_weight
            }
        },
        "critical_path": {
            "dsv4f": dsv_doc.critical_path,
            "q80_decode": q80_doc.critical_path,
            "q80_baseline_run": q80_run.critical_path
        },
        "reconciliation": reports,
        "receipt_flags": flags,
        "honesty": {
            "dsv4f": dsv_r.honesty,
            "q80": q80_r.honesty,
            "plain": format!(
                "Lane wins do not add up to a measured token improvement. \
                 DSV4F: claimed {} ns against a {} ns token; measured delta 0 ns; discrepancy {} ns. \
                 Q80: claimed {} ns against a {} ns TOKEN_NS wall; measured delta 0 ns; discrepancy {} ns.",
                dsv_r.sum_claimed_ns,
                dsv_r.token_before_ns,
                dsv_r.discrepancy_ns,
                q80_r.sum_claimed_ns,
                q80_r.token_before_ns,
                q80_r.discrepancy_ns
            )
        },
        "served_weight_honesty": {
            "field": FS_PER_WEIGHT_SERVED_FIELD,
            "caveat": AMORTIZED_CAVEAT,
            "not_latency": true
        },
        "energy_investigation": energy_probe,
        "correctness": "N/A — schema/adapter/reconciler only. Existing ledgers and gates were not edited.",
        "authority_reps": {
            "dsv4f_host_wall_warm_body_ms": [1067, 1033, 1040, 1037, 1024],
            "dsv4f_host_wall_warm_gpu_ms": [407.6, 400.2, 399.0, 398.0, 396.5],
            "dsv4f_label": "DIRTY_ENGINEERING",
            "q80_token_ns_decode_wall_ns_mean": 559171655,
            "q80_registered_incumbent_ns": 403000000,
            "q80_registered_note": "P1-invalid: unpaired, 1637 fallbacks, no GPU timestamps"
        },
        "dsv4f_residual_decomposition": {
            "body_ns": dsv_doc.totals.total_token_ns,
            "named_serial_ns": dsv_doc.closure.sum_serial_stage_ns,
            "residual_ns": dsv_doc.residual_ns,
            "note": "Most of residual_ns is metal host_wall minus (encode+submit+wait): intra-CB host time that the named encode/submit/wait rows do not cover. The rest is host_exclusive not captured by the named serial host stages."
        },
        "next_bottleneck": {
            "dsv4f": {
                "name": "host.expert_slab_io (GPU-idle streamed top-6 expert read+fill) then the unattributed residual inside metal host_wall, then metal GPU across 137 CBs",
                "ns_per_token": 415_126_416u64,
                "residual_ns_this_receipt": dsv_doc.residual_ns,
                "gpu_busy_ns": dsv_doc.totals.total_gpu_busy_ns,
                "label": "DIRTY_ENGINEERING"
            },
            "q80": {
                "name": "host.compact_expert_slab_pack (~281 ms serial) then metal.wait (~274 ms, contains ~232 ms GPU)",
                "ns_per_token": q80_doc.totals.total_cpu_critical_ns,
                "gpu_busy_ns": q80_doc.totals.total_gpu_busy_ns,
                "label": "DIRTY_ENGINEERING"
            }
        }
    });
    write_json(&out.join("JOINT_TOKEN_NS.json"), &receipt);

    eprintln!(
        "Q80 decode TOTAL_TOKEN_NS={} residual={} failed={} | Q80 run residual={} failed={} | DSV4F body TOTAL_TOKEN_NS={} residual={} failed={} naive_overcount={}",
        q80_doc.totals.total_token_ns,
        q80_doc.residual_ns,
        q80_doc.closure.failed,
        q80_run.residual_ns,
        q80_run.closure.failed,
        dsv_doc.totals.total_token_ns,
        dsv_doc.residual_ns,
        dsv_doc.closure.failed,
        dsv_doc.closure.naive_overcount_ns
    );
    eprintln!(
        "DSV4F claimed={} measured_delta={} discrepancy={} | Q80 claimed={} measured_delta={} discrepancy={}",
        dsv_r.sum_claimed_ns,
        dsv_r.measured_delta_ns,
        dsv_r.discrepancy_ns,
        q80_r.sum_claimed_ns,
        q80_r.measured_delta_ns,
        q80_r.discrepancy_ns
    );
    eprintln!("receipt_flags={}", flags.len());
    eprintln!(
        "Q80 fs/weight_served={:.1} (amortized, NOT latency) floor={:?} distance={:?} | DSV4F fs/weight_served={:.1} floor={:?} distance={:?}",
        q80_doc.served_weight.fs_per_weight_served,
        q80_doc.served_weight.fs_per_weight_floor,
        q80_doc.served_weight.distance_from_floor,
        dsv_doc.served_weight.fs_per_weight_served,
        dsv_doc.served_weight.fs_per_weight_floor,
        dsv_doc.served_weight.distance_from_floor
    );
}
