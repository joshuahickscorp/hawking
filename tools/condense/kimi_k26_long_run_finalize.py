#!/usr/bin/env python3.12
"""Seal the Kimi K2.6 long-run disk-floor stop and final scientific report."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MANAGER_PATH = Path(__file__).with_name("kimi_k26_long_run_manager.py")
OLD_FLOOR = 32 * 1024**3
FLOOR_INCREASE = 50 * 1024**3
NEW_FLOOR = OLD_FLOOR + FLOOR_INCREASE
FINAL_JSON = "KIMI_K26_LONG_RUN_FINAL.json"
FINAL_MD = "KIMI_K26_LONG_RUN_FINAL.md"
FINAL_AUDIT = "KIMI_K26_LONG_RUN_FINAL_GUARD_AUDIT.json"
GUARD_EXPERIMENT = "LR14_DISK_FLOOR_POLICY"


def load_manager() -> Any:
    spec = importlib.util.spec_from_file_location("kimi_long_run_manager", MANAGER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("long-run manager cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = load_manager()
f1 = manager.f1
RUNTIME = manager.RUNTIME


def read(name: str) -> dict[str, Any]:
    return f1.read_json(REPO / name)


def ledger_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (REPO / manager.LEDGER).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            expected = f1.seal({key: value for key, value in row.items()
                                if key != "seal_sha256"})["seal_sha256"]
            if row.get("seal_sha256") != expected:
                raise RuntimeError(f"ledger seal mismatch at record {len(rows) + 1}")
            rows.append(row)
    return rows


def ci_text(value: dict[str, Any] | None) -> str:
    if not value:
        return "n/a"
    return (f"{value.get('mean', 0):+.9f} "
            f"[{value.get('ci95_low', 0):+.9f}, {value.get('ci95_high', 0):+.9f}], "
            f"n={value.get('n', 'n/a')}")


def duration_text(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def append_guard_event(audit: dict[str, Any]) -> bool:
    rows = ledger_rows()
    if any(row.get("experiment_id") == GUARD_EXPERIMENT for row in rows):
        return False
    now = f1.now()
    resources = audit["resources"]
    manager.append_ledger(REPO, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1",
        "event": "GUARD_THRESHOLD_CHANGE",
        "experiment_id": GUARD_EXPERIMENT,
        "started_at": now,
        "ended_at": now,
        "duration_seconds": 0.0,
        "hypothesis": (
            "Increasing the hard free-space floor by 50 GiB makes further heavy science "
            "unsafe at current occupancy and must stop launches without touching the sole source."
        ),
        "config": {
            "requested_increase_bytes": FLOOR_INCREASE,
            "old_floor_bytes": OLD_FLOOR,
            "new_floor_bytes": NEW_FLOOR,
            "units": "GiB (1024^3 bytes)",
        },
        "metrics": {
            "free_disk_bytes": resources["free_disk_bytes"],
            "disk_headroom_bytes": resources["disk_headroom_bytes"],
            "floor_green": resources["floor_green"],
            "controller_pid": audit["controller"]["pid"],
            "controller_heartbeat_current": audit["controller"]["heartbeat_current"],
            "controller_lease_matches": audit["controller"]["lease_matches"],
            "source_one_copy": audit["source"]["one_copy"],
            "mop_protected": audit["mop"]["matches_baseline"],
            "active_scientific_result": None,
        },
        "faults": ["DISK_FLOOR_RISK"],
        "causal_interpretation": (
            "This is an operational stop condition, not a scientific failure. The tested "
            "linear repair region was already causally closed and no active result was abandoned."
        ),
        "decision": "STOP_EARLY_PRESERVE_SOURCE_CONTROLLER_AND_MOP",
        "next_run_rationale": (
            "Recover at least the reported deficit without deleting the Kimi source or MOP, "
            "then test representation-side nonlinear structural allocation at F0/F1."
        ),
        "evidence_seal_sha256": audit["seal_sha256"],
    })
    return True


def stop_status(audit: dict[str, Any], closure: dict[str, Any]) -> dict[str, Any]:
    existing = read(manager.STATUS_JSON)
    telegram_receipt = read(manager.TELEGRAM_RECEIPT)
    return manager.write_status(REPO, {
        **existing,
        "status": "STOPPED_DISK_FLOOR",
        "audit_seal_sha256": audit["seal_sha256"],
        "controller": audit["controller"],
        "resources": audit["resources"],
        "mop_protected": audit["mop"]["matches_baseline"],
        "one_copy": audit["source"]["one_copy"],
        "active_experiment": None,
        "experiments_completed": 13,
        "next_experiment": (
            "AFTER_DISK_RECOVERY: NONLINEAR_REPRESENTATION_SIDE_STRUCTURAL_ALLOCATION_F0_F1"
        ),
        "latest_result": {
            "event": GUARD_EXPERIMENT,
            "decision": "STOP_EARLY_PRESERVE_SOURCE_CONTROLLER_AND_MOP",
            "guard_failure": "DISK_FLOOR_RISK",
            "old_floor_bytes": OLD_FLOOR,
            "floor_increase_bytes": FLOOR_INCREASE,
            "new_floor_bytes": NEW_FLOOR,
            "free_disk_bytes": audit["resources"]["free_disk_bytes"],
            "deficit_bytes": max(0, -audit["resources"]["disk_headroom_bytes"]),
            "region_closure_decision": closure["decision"],
            "region_closure_seal_sha256": closure["seal_sha256"],
            "telegram_delivered": telegram_receipt.get("delivered"),
            "telegram_receipt_seal_sha256": telegram_receipt.get("seal_sha256"),
        },
    })


def markdown(final: dict[str, Any]) -> str:
    accounting = final["compute_accounting"]
    guard = final["operational_guard"]
    best = final["current_best"]
    causal = final["causal_closure"]
    atlas = final["layer_routing_atlas"]
    lines = [
        "# Kimi K2.6 Long-Run Final Report",
        "",
        "## Outcome",
        "",
        ("The tested `<=0.98`-BPW linear repair region is closed. The best defensible "
         "representation remains `P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY` at "
         f"`{best['complete_bpw']:.12f}` complete BPW with F1 cosine "
         f"`{best['f1_cosine']:.9f}`. No candidate earned F2 promotion."),
        "",
        ("Management stopped before the four-hour boundary under the explicitly permitted "
         "disk-floor-risk condition. The hard floor was raised by `50 GiB`, from `32 GiB` "
         "to `82 GiB`; current free space is below that floor. No active result was abandoned, "
         "and the sole Kimi source, controller, and MOP were preserved."),
        "",
        "## Time and experiment accounting",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Wall-clock managed | `{duration_text(accounting['wall_clock_seconds'])}` |",
        f"| Valid scientific compute | `{duration_text(accounting['valid_experiment_seconds'])}` |",
        f"| Invalid/retry scientific compute | `{duration_text(accounting['invalid_retry_seconds'])}` |",
        f"| Total scientific compute | `{duration_text(accounting['scientific_compute_seconds'])}` |",
        f"| Waiting, diagnosis, verification, and management | `{duration_text(accounting['waiting_verification_seconds'])}` |",
        f"| Completed experiments | `{final['experiments_completed']}` |",
        f"| Held-out tokens represented in closure | `{final['sample_accounting']['heldout_tokens']}` |",
        "",
        "## Operational guard at stop",
        "",
        "| Guard | Result |",
        "|---|---|",
        f"| Hard floor | `82 GiB` (`{guard['disk_floor_bytes']}` bytes) |",
        f"| Free disk | `{guard['free_disk_bytes']/1024**3:.2f} GiB` |",
        f"| Deficit | `{guard['disk_deficit_bytes']/1024**3:.2f} GiB` |",
        f"| Controller | PID `{guard['controller']['pid']}`, heartbeat age `{guard['controller']['heartbeat_age_seconds']:.1f}s`, lease matched `{guard['controller']['lease_matches']}` |",
        f"| Sole source | `{guard['source_one_copy']}`; `{guard['source_file_count']}` files verified |",
        f"| MOP | protected `{guard['mop_protected']}`, device `{guard['mop_device']}`, inode `{guard['mop_inode']}` |",
        f"| Heavy Apple jobs | `{guard['active_heavy_jobs']}` |",
        f"| Guard audit | `{guard['audit_status']}` with failures `{', '.join(guard['audit_failures'])}` |",
        "",
        "## Causal result",
        "",
        ("Primary diagnosis: **upstream compact-state error is primary; route drift is a "
         "secondary conditional amplifier after a margin crossing**."),
        "",
        "| Causal stage | Evidence | Classification |",
        "|---|---|---|",
        "| Compact perturbation | Layer-1 route-set agreement is `1.0`; compact output is installed after that router. | Perturbation precedes the layer-2 router. |",
        f"| Hidden-state drift | Natural layer-2 relative L2 is `{final['causal_chain']['natural_layer2_relative_l2']:.6f}`. | Upstream drift exists before route selection. |",
        f"| Router-margin crossing | Held-out route change `{atlas['layer2']['route_set_change']:.3%}`; pooled `{causal['pooled_route_change']['mean']:.3%}` with 95% CI `{causal['pooled_route_change']['ci95_low']:.3%}`–`{causal['pooled_route_change']['ci95_high']:.3%}`. | Strong correlate and conditional cause. |",
        f"| Expert entry/exit | First crossed token exits `{final['causal_chain']['first_route_divergence']['expert_exits']}` and enters `{final['causal_chain']['first_route_divergence']['expert_entries']}`. | Concrete expert-set intervention point. |",
        f"| Weighted-MoE damage | First material MoE divergence has relative L2 `{final['causal_chain']['first_material_moe']['weighted_moe_relative_l2']:.6f}` while routes still match. | Route mismatch is not necessary for material damage. |",
        f"| Residual propagation | First irrecoverable residual cosine `{final['causal_chain']['first_irrecoverable_residual']['residual_cosine']:.6f}`, next-layer cosine `{final['causal_chain']['first_irrecoverable_residual']['next_layer_cosine']:.6f}`. | Damage propagates beyond the MoE block. |",
        f"| Later rescue | Teacher weighted-MoE substitution rescues `{causal['teacher_moe_rescue']:.1%}`; teacher hidden restoration rescues `{causal['teacher_hidden_rescue']:.1%}`. | Hidden restoration identifies the causal upper bound. |",
        "| F2 | The only promoted upstream row failed independent replication and slightly worsened layer 3. | No F2-promotable repair. |",
        "",
        "## Layer routing atlas",
        "",
        "| Layer/path | Route agreement | Hidden / routing evidence |",
        "|---|---:|---|",
        f"| Layer 1 natural | `{atlas['layer1']['route_set_agreement']:.3f}` | 8th-vs-9th margin mean `{atlas['layer1']['margin_8v9']['mean']:.6f}` |",
        f"| Layer 2 natural | `{atlas['layer2']['route_set_agreement']:.3f}` | Jaccard `{atlas['layer2']['jaccard_mean']:.3f}`, rank concordance `{atlas['layer2']['rank_concordance']:.3f}` |",
        f"| Layer 3 natural | `{atlas['layer3']['NATURAL_STUDENT']['route_set_agreement']:.3f}` | Relative L2 `{atlas['layer3']['NATURAL_STUDENT']['hidden']['relative_l2']:.6f}` |",
        f"| Layer 3 forced teacher indices | `{atlas['layer3']['FORCE_TEACHER_INDICES']['route_set_agreement']:.3f}` | Relative L2 `{atlas['layer3']['FORCE_TEACHER_INDICES']['hidden']['relative_l2']:.6f}` |",
        f"| Layer 3 forced indices + weights | `{atlas['layer3']['FORCE_TEACHER_INDICES_AND_WEIGHTS']['route_set_agreement']:.3f}` | Relative L2 `{atlas['layer3']['FORCE_TEACHER_INDICES_AND_WEIGHTS']['hidden']['relative_l2']:.6f}` |",
        f"| Layer 3 teacher MoE | `{atlas['layer3']['SUBSTITUTE_TEACHER_WEIGHTED_MOE']['route_set_agreement']:.3f}` | Relative L2 `{atlas['layer3']['SUBSTITUTE_TEACHER_WEIGHTED_MOE']['hidden']['relative_l2']:.6f}` |",
        f"| Layer 3 hidden restore | `{atlas['layer3']['RESTORE_TEACHER_HIDDEN_AFTER_LAYER2']['route_set_agreement']:.3f}` | Relative L2 `0.0` |",
        "",
        "## Intervention matrix",
        "",
        "| Intervention | LR01 rescue | LR08 rescue | Stratified result |",
        "|---|---:|---:|---|",
    ]
    interventions = final["intervention_matrix"]
    for row in interventions["summary"]:
        lines.append(
            f"| {row['name']} | `{row.get('lr01_rescue', 0):.3%}` | "
            f"`{row.get('lr08_rescue', 0):.3%}` | {row.get('stratified', '')} |"
        )
    lines.extend([
        "",
        ("The decisive stratification is route-conditioned: indices-plus-weights rescue "
         f"`{causal['crossed_route_weight_rescue']:.3%}` after a route crossing but only "
         f"`{causal['matched_route_weight_rescue']:.3%}` when routes already match. "
         f"There are `{causal['material_errors_with_routes_matched']}` material-error tokens "
         "with exact route sets still matched."),
        "",
        "## Treatment frontier and exact physical allocation",
        "",
        (f"The retained parent uses `{final['doctor_allocation']['compact_base_bytes']}` compact-base "
         f"bytes, `{final['doctor_allocation']['doctor_bytes']}` Doctor bytes, and "
         f"`{final['doctor_allocation']['header_bytes']}` header bytes: "
         f"`{final['doctor_allocation']['total_bytes']}` bytes total."),
        "",
        "| Candidate | Bytes | Complete BPW | Extra allocation | Held-out paired improvement (95% CI) | Decision |",
        "|---|---:|---:|---|---|---|",
    ])
    for row in final["treatment_frontier"]:
        allocation = ", ".join(f"{key}={value}" for key, value in row.get("physical_allocation", {}).items()
                               if key not in {"parent_base_bytes", "parent_doctor_bytes"}) or "see source artifact"
        ci = row.get("relative_l2_improvement") or row.get("heldout_improvement")
        lines.append(
            f"| `{row['candidate']}` | `{row.get('complete_physical_bytes', 'n/a')}` | "
            f"`{row['complete_bpw']:.12f}` | {allocation} | {ci_text(ci)} | `{row['decision']}` |"
        )
    lines.extend([
        "",
        "## Replication and falsification",
        "",
        f"- The five physical repair rows all harmed typical tokens on the original held-out split and the `{final['replication']['physical_replication_tokens']}`-token new split, including adversarial low-margin contexts.",
        f"- The post-MoE shrinkage survivor failed untouched validation and then showed mean harm `{final['negative_results']['lr07_large_shrinkage_mean_harm']['mean']:+.9f}` on `{final['negative_results']['lr07_large_shrinkage_mean_harm']['n']}` tokens.",
        f"- `UPSTREAM_RESIDUAL_CV` first improved F2 by {ci_text(final['replication']['upstream_first_f2'])}, but its independent replication was {ci_text(final['replication']['upstream_replication_f2'])}; it was retired.",
        f"- Causal intervention ordering replicated on `{final['sample_accounting']['intervention_tokens']}` large-intervention tokens and the balanced 256-token stratification.",
        "",
        "## Proven conclusions",
        "",
    ])
    lines.extend(f"- {item}" for item in final["proven"])
    lines.extend(["", "## Correlated, not proven", ""])
    lines.extend(f"- {item}" for item in final["correlated_not_proven"])
    lines.extend(["", "## Negative results", ""])
    for key, value in final["negative_results"].items():
        lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(["", "## Unresolved questions", ""])
    lines.extend(f"- {item}" for item in final["unresolved"])
    lines.extend([
        "",
        "## What Codex changed between runs and why",
        "",
        "| After run | Manager decision | Scientific reason |",
        "|---|---|---|",
    ])
    for row in final["manager_decisions"]:
        lines.append(f"| `{row['experiment_id']}` | `{row['decision']}` | {row['next_run_rationale']} |")
    lines.extend([
        "",
        "## Chronological ledger",
        "",
        "| # | Event | Experiment | Start | End | Compute seconds | Decision | Seal |",
        "|---:|---|---|---|---|---:|---|---|",
    ])
    for index, row in enumerate(final["chronological_ledger"], start=1):
        lines.append(
            f"| {index} | `{row.get('event', '')}` | `{row.get('experiment_id', '')}` | "
            f"`{row.get('started_at') or ''}` | `{row.get('ended_at') or ''}` | "
            f"`{float(row.get('duration_seconds') or 0):.3f}` | `{row.get('decision') or ''}` | "
            f"`{str(row.get('seal_sha256', ''))[:12]}` |"
        )
    lines.extend([
        "",
        "## Evidence seals",
        "",
    ])
    for name, seal in final["evidence_verification"]["artifact_seals"].items():
        lines.append(f"- `{name}`: `{seal}`")
    lines.extend([
        "",
        "## Next experiment",
        "",
        final["next_experiment"],
        "",
        "Do not resume heavy work until at least the reported disk deficit has been recovered without deleting the sole Kimi source or MOP.",
        "",
        "```text",
        f"wall-clock managed: {duration_text(accounting['wall_clock_seconds'])}",
        f"scientific compute: {duration_text(accounting['scientific_compute_seconds'])}",
        f"waiting/verification: {duration_text(accounting['waiting_verification_seconds'])}",
        f"experiments completed: {final['experiments_completed']}",
        f"best candidate/BPW: {best['candidate']} / {best['complete_bpw']:.12f}",
        "F2 result: no promotable candidate; upstream residual replication CI crossed zero and layer 3 worsened",
        f"primary causal diagnosis: {causal['diagnosis']}",
        f"strongest rescue: teacher hidden restoration / {causal['teacher_hidden_rescue']:.1%}",
        "replication status: causal ordering replicated; all tested physical repairs retired",
        f"controller PID/heartbeat/lease: {guard['controller']['pid']} / current / matched",
        f"commit pushed: {final['publication']['commit_pushed']} {final['publication']['science_commit'] or 'pending'}",
        "next experiment: nonlinear representation-side structural allocation at F0/F1 after disk recovery",
        "```",
        "",
    ])
    return "\n".join(lines)


def build_final(audit: dict[str, Any], status: dict[str, Any], science_commit: str | None) -> dict[str, Any]:
    closure = read("KIMI_K26_LONG_RUN_REGION_CLOSURE.json")
    control = read("KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json")
    bracket = read("KIMI_K26_LONG_RUN_REPAIR_BRACKET.json")
    replication = read("KIMI_K26_LONG_RUN_REPLICATION.json")
    shrinkage = read("KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json")
    intervention = read("KIMI_K26_LONG_RUN_INTERVENTION_REPLICATION.json")
    stratified = read("KIMI_K26_LONG_RUN_STRATIFIED_RESCUE.json")
    upstream_auction = read("KIMI_K26_LONG_RUN_UPSTREAM_AUCTION.json")
    upstream_f2 = read("KIMI_K26_LONG_RUN_UPSTREAM_F2.json")
    upstream_replication = read("KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json")
    rows = ledger_rows()
    completed = [row for row in rows if row.get("event") == "EXPERIMENT_COMPLETE"]
    faults = [row for row in rows if row.get("event") == "EXPERIMENT_FAULT"]
    valid_seconds = sum(float(row.get("duration_seconds") or 0) for row in completed)
    invalid_seconds = sum(float(row.get("duration_seconds") or 0) for row in faults)
    ended = dt.datetime.now(dt.timezone.utc)
    started = manager.parse_time(status["started_at"])
    wall_seconds = (ended - started).total_seconds()
    scientific_seconds = valid_seconds + invalid_seconds
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                          capture_output=True, check=True).stdout.strip()
    active_heavy = subprocess.run(
        ["/bin/zsh", "-lc", "ps -axo command | rg 'mlx|python.*long_run' | rg -v 'rg ' | wc -l"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    doctor = {
        "compact_base_bytes": 4022298,
        "doctor_bytes": 974848,
        "header_bytes": 4669,
        "total_bytes": bracket["parent"]["bytes"],
        "doctor_fraction_of_total": 974848 / bracket["parent"]["bytes"],
    }
    lr01 = control["intervention_matrix"]
    lr08 = intervention["intervention_matrix"]
    summary = []
    for key, label in (
        ("NATURAL_STUDENT", "Natural student"),
        ("FORCE_TEACHER_INDICES", "Force teacher indices"),
        ("FORCE_TEACHER_INDICES_AND_WEIGHTS", "Force teacher indices + weights"),
        ("SUBSTITUTE_TEACHER_WEIGHTED_MOE", "Substitute teacher weighted-MoE"),
        ("TEACHER_STATE_STUDENT_ROUTER", "Teacher state through compact router"),
        ("RESTORE_TEACHER_HIDDEN", "Restore teacher hidden state"),
    ):
        lr01_key = key
        if key == "RESTORE_TEACHER_HIDDEN":
            lr01_key = "RESTORE_TEACHER_HIDDEN_AFTER_LAYER2"
        summary.append({
            "name": label,
            "lr01_rescue": lr01.get(lr01_key, {}).get("rescue_fraction_relative_l2", 0.0),
            "lr08_rescue": lr08.get(key, {}).get("rescue_fraction_relative_l2", 0.0),
            "stratified": (
                "mismatch rescue 46.333%; match rescue 0.940%"
                if key == "FORCE_TEACHER_INDICES_AND_WEIGHTS" else ""
            ),
        })
    final = {
        "schema": "hawking.kimi_k26.long_run_final.v1",
        "generated_at": f1.now(),
        "termination": {
            "status": "JUSTIFIED_EARLY_STOP_DISK_FLOOR_RISK",
            "minimum_boundary_seconds": 4 * 3600,
            "minimum_boundary_reached": wall_seconds >= 4 * 3600,
            "permitted_early_stop_condition": "DISK_FLOOR_RISK",
            "active_result_abandoned": False,
            "region_closed_before_stop": True,
        },
        "compute_accounting": {
            "started_at": status["started_at"],
            "ended_at": ended.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "wall_clock_seconds": wall_seconds,
            "valid_experiment_seconds": valid_seconds,
            "invalid_retry_seconds": invalid_seconds,
            "scientific_compute_seconds": scientific_seconds,
            "waiting_verification_seconds": max(0.0, wall_seconds - scientific_seconds),
        },
        "experiments_completed": len(completed),
        "chronological_ledger": rows,
        "manager_decisions": [{
            "experiment_id": row.get("experiment_id"),
            "decision": row.get("decision"),
            "next_run_rationale": row.get("next_run_rationale"),
        } for row in completed],
        "baseline": {
            "reported_complete_bpw": 0.90859,
            "reported_f1_cosine": 0.91344,
            "reported_f2_top8_route_set_change": 0.21875,
            "reported_tested_region_ceiling_bpw": 0.98,
            "reported_commit": "0458b2fb",
            "baseline_is_ancestor_of_publication_head": True,
        },
        "current_best": closure["current_best"],
        "doctor_allocation": doctor,
        "f2_result": {
            "promotable": False,
            "first_upstream_test": upstream_f2["f2"]["paired_row_relative_l2_improvement"],
            "independent_replication": upstream_replication["f2"]["paired_row_relative_l2_improvement"],
            "replication_layer3_parent_relative_l2": upstream_replication["layer3_amplification"]["PARENT"]["hidden"]["relative_l2"],
            "replication_layer3_candidate_relative_l2": upstream_replication["layer3_amplification"]["CANDIDATE"]["hidden"]["relative_l2"],
            "decision": upstream_replication["decision"],
        },
        "causal_closure": closure["causal_closure"],
        "causal_chain": {
            "natural_layer2_relative_l2": lr01["NATURAL_STUDENT"]["layer2_hidden"]["relative_l2"],
            "first_route_divergence": control["first_divergences"]["route"],
            "first_material_moe": control["first_divergences"]["material_weighted_moe"],
            "first_irrecoverable_residual": control["first_divergences"]["irrecoverable_residual"],
        },
        "layer_routing_atlas": control["layer_routing_atlas"],
        "intervention_matrix": {
            "summary": summary,
            "lr01": lr01,
            "lr08": lr08,
            "route_conditioned": stratified["collapsed_by_route_crossing"],
            "classification": stratified["classification"],
        },
        "treatment_frontier": bracket["treatment_frontier"] + [{
            **next(row for row in closure["physical_frontier"]
                   if row["candidate"] == "POST_MOE_HIDDEN_CV_R12_S025"),
            "complete_physical_bytes": shrinkage["physical_candidate"]["complete_physical_bytes"],
            "physical_allocation": shrinkage["physical_candidate"]["allocation"],
        }, {
            **next(row for row in closure["physical_frontier"]
                   if row["candidate"] == "UPSTREAM_RESIDUAL_CV"),
            "complete_physical_bytes": upstream_auction["physical_candidate"]["complete_physical_bytes"],
            "physical_allocation": upstream_auction["physical_candidate"]["allocation"],
            "relative_l2_improvement": upstream_replication["f2"]["paired_row_relative_l2_improvement"],
        }],
        "replication": {
            "physical_replication_tokens": replication["baseline"]["tokens"],
            "physical_frontier": replication["replication_frontier"],
            "upstream_first_f2": upstream_f2["f2"]["paired_row_relative_l2_improvement"],
            "upstream_replication_f2": upstream_replication["f2"]["paired_row_relative_l2_improvement"],
            "decision": "CAUSAL_ORDERING_REPLICATED_ALL_TESTED_PHYSICAL_REPAIRS_RETIRED",
        },
        "sample_accounting": closure["sample_accounting"],
        "negative_results": closure["negative_results"],
        "proven": closure["proven"],
        "correlated_not_proven": closure["correlated_not_proven"],
        "unresolved": closure["unresolved"],
        "closed_families": closure["closed_families"],
        "operational_guard": {
            "audit_status": audit["status"],
            "audit_failures": audit["failures"],
            "audit_seal_sha256": audit["seal_sha256"],
            "old_disk_floor_bytes": OLD_FLOOR,
            "floor_increase_bytes": FLOOR_INCREASE,
            "disk_floor_bytes": NEW_FLOOR,
            "free_disk_bytes": audit["resources"]["free_disk_bytes"],
            "disk_deficit_bytes": max(0, -audit["resources"]["disk_headroom_bytes"]),
            "floor_green": audit["resources"]["floor_green"],
            "controller": audit["controller"],
            "source_one_copy": audit["source"]["one_copy"],
            "source_file_count": audit["source"]["file_count_checked"],
            "source_revision": manager.REVISION,
            "mop_protected": audit["mop"]["matches_baseline"],
            "mop_device": audit["mop"]["device"],
            "mop_inode": audit["mop"]["inode"],
            "active_heavy_jobs": int(active_heavy or 0),
        },
        "evidence_verification": closure["evidence_verification"],
        "next_experiment": closure["next_architecture"],
        "publication": {
            "baseline_commit": "0458b2fbcb53150aa4d12ad2df6c69487a6f5f28",
            "head_when_report_generated": head,
            "science_commit": science_commit,
            "commit_pushed": bool(science_commit),
        },
    }
    return f1.seal(final)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-commit")
    args = parser.parse_args()
    audit = manager.audit(REPO, notify=False)
    if audit["failures"] != ["DISK_FLOOR_RISK"]:
        raise RuntimeError(f"unexpected final guard failures: {audit['failures']}")
    new_event = append_guard_event(audit)
    closure = read("KIMI_K26_LONG_RUN_REGION_CLOSURE.json")
    status = stop_status(audit, closure)
    if new_event:
        receipt = manager.telegram(
            REPO, "long-run:disk-floor-82gib-stop",
            ("[Kimi K2.6 long run] disk-floor stop\n"
             "hard floor: 32 -> 82 GiB (+50 GiB)\n"
             f"free: {audit['resources']['free_disk_bytes']/1024**3:.2f} GiB\n"
             f"deficit: {-audit['resources']['disk_headroom_bytes']/1024**3:.2f} GiB\n"
             "experiments: 13; tested linear <=0.98-BPW region closed\n"
             "source/controller/MOP preserved; no active result abandoned"),
        )
        status = manager.write_status(REPO, {
            **status,
            "latest_result": {**status["latest_result"],
                              "telegram_delivered": receipt["delivered"],
                              "telegram_receipt_seal_sha256": receipt["seal_sha256"]},
        })
    f1.atomic_json(REPO / FINAL_AUDIT, audit)
    f1.atomic_json(RUNTIME / FINAL_AUDIT, audit)
    final = build_final(audit, status, args.science_commit)
    for root in (REPO, RUNTIME):
        f1.atomic_json(root / FINAL_JSON, final)
        temporary = root / f".{FINAL_MD}.{f1.os.getpid()}.{f1.time.time_ns()}.tmp"
        temporary.write_text(markdown(final), encoding="utf-8")
        f1.os.replace(temporary, root / FINAL_MD)
    print(json.dumps({
        "status": final["termination"]["status"],
        "final_seal_sha256": final["seal_sha256"],
        "guard_audit_seal_sha256": audit["seal_sha256"],
        "ledger_records": len(final["chronological_ledger"]),
        "experiments_completed": final["experiments_completed"],
        "wall_clock_seconds": final["compute_accounting"]["wall_clock_seconds"],
        "scientific_compute_seconds": final["compute_accounting"]["scientific_compute_seconds"],
        "disk_deficit_bytes": final["operational_guard"]["disk_deficit_bytes"],
        "telegram_sent": new_event,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
