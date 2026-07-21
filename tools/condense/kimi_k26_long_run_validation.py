#!/usr/bin/env python3.12
"""Untouched validation of the single LR04 preregistered post-MoE repair."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402
import kimi_k26_long_run_repair_bracket as bracket  # noqa: E402
import kimi_k26_long_run_replication as replication  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


EXPERIMENT_ID = "LR05_UNTOUCHED_SHRINKAGE_VALIDATION"
PROBES = [
    {"id": "validation_history", "domain": "prose",
     "text": "The printing press accelerated the circulation of technical and political texts across Europe."},
    {"id": "validation_biology", "domain": "prose",
     "text": "During transcription, RNA polymerase reads a DNA template and synthesizes a complementary RNA strand."},
    {"id": "validation_python", "domain": "code",
     "text": "def rolling_mean(values, width):\n    total = sum(values[:width])\n    result = [total / width]\n    return result"},
    {"id": "validation_sql", "domain": "code",
     "text": "SELECT customer_id, COUNT(*) AS n FROM orders WHERE paid = TRUE GROUP BY customer_id HAVING COUNT(*) > 3;"},
    {"id": "validation_algebra", "domain": "reasoning",
     "text": "If x + y = 11 and xy = 24, compute x squared plus y squared and explain each identity used."},
    {"id": "validation_probability", "domain": "reasoning",
     "text": "Two fair dice are rolled without observing order. Compare the probability of sum seven with sum nine."},
    {"id": "validation_logic", "domain": "reasoning",
     "text": "Every amber object is light, no light object is iron, and Q is amber. Determine what cannot be true of Q."},
    {"id": "validation_json", "domain": "tool_format", "chat": True,
     "text": "Answer as one JSON object containing result, assumptions, and confidence as a number from zero to one."},
    {"id": "validation_tool", "domain": "tool_format", "chat": True, "thinking": True,
     "text": "Check whether the process ID owns the lease and whether the heartbeat is younger than thirty seconds."},
    {"id": "validation_symbols", "domain": "prose",
     "text": "For ξ→0⁺, compare O(ξ²), o(ξ), and the oscillatory term sin(1/ξ)."},
    {"id": "validation_boundary", "domain": "reasoning",
     "text": "Candidates A and B differ by 0.00001 at the cutoff; state which evidence would justify swapping their rank."},
    {"id": "validation_markup", "domain": "tool_format", "chat": True,
     "text": "Return exactly <answer><status>valid</status><count>8</count></answer>."},
]


def prepare_requests(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = reference.KimiTokenizer(source)
    requests = []
    token_ids = []
    lengths = []
    records = []
    offset = 0
    for probe in PROBES:
        rendered = (tokenizer.user_prompt(probe["text"], thinking=bool(probe.get("thinking")))
                    if probe.get("chat") else probe["text"])
        ids = tokenizer.encode(rendered)
        requests.append({**probe, "rendered": rendered, "token_ids": ids})
        token_ids.extend(ids)
        lengths.append(len(ids))
        for position, token_id in enumerate(ids):
            records.append({"token_index": offset + position, "segment": probe["id"],
                            "domain": probe["domain"], "position": position,
                            "token_id": int(token_id),
                            "token_text": tokenizer.decode([int(token_id)])})
        offset += len(ids)
    return requests, {"token_ids": token_ids, "segment_lengths": lengths,
                      "token_records": records}


def summarize(
    teacher: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    teacher_route: dict[str, np.ndarray], baseline_route: dict[str, np.ndarray],
    mask: np.ndarray, seed: int,
) -> dict[str, Any]:
    base_rows = causal.row_metrics(teacher, baseline)
    candidate_rows = causal.row_metrics(teacher, candidate)
    route_exact = np.asarray([set(left) == set(right) for left, right in zip(
        teacher_route["indices"], baseline_route["indices"], strict=True,
    )])
    return {
        "tokens": int(np.sum(mask)),
        "baseline": {"hidden": f1.quality(teacher[mask], baseline[mask]),
                     "row_relative_l2_mean": float(np.mean(base_rows["relative_l2"][mask])),
                     "route_set_agreement": float(np.mean(route_exact[mask])),
                     "route_matches": int(np.sum(route_exact[mask]))},
        "candidate": {"hidden": f1.quality(teacher[mask], candidate[mask]),
                      "row_relative_l2_mean": float(np.mean(candidate_rows["relative_l2"][mask])),
                      "route_set_agreement_before_post_moe_repair":
                          float(np.mean(route_exact[mask])),
                      "route_matches_before_post_moe_repair": int(np.sum(route_exact[mask]))},
        "relative_l2_improvement": bracket.paired_interval(
            base_rows["relative_l2"][mask] - candidate_rows["relative_l2"][mask], seed,
        ),
    }


def run(repo: Path, source: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"pre-run guard audit failed: {audit['failures']}")
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**prior_status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": EXPERIMENT_ID,
                                "next_experiment": "LR05_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "The calibration-selected rank-12 quarter-strength repair improves untouched F2.",
        "config": {"seed": seed, "prompt_count": len(PROBES),
                   "candidate_count": 1, "refit": False, "retuning": False},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    lr04 = f1.read_json(repo / "KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json")
    payload_info = lr04["physical_candidate"]["payload"]
    header, payload_arrays = replication.read_payload(
        Path(payload_info["path"]), payload_info["sha256"],
    )
    model = replication.decode_model(payload_arrays, "post_moe_hidden")
    shrinkage = float(header["metadata"]["shrinkage"])
    requests, batch = prepare_requests(source)
    token_ids = batch["token_ids"]
    segment_lengths = batch["segment_lengths"]
    records = batch["token_records"]
    post_l1, x_l1, input_info = causal.capture_layer_one_inputs(
        source, token_ids, segment_lengths,
    )
    config = f1.read_json(source / "config.json")["text_config"]
    shard_l1 = reference.TensorShard(reference.shard_path(source, 2))
    routes_l1 = causal.route_diagnostics(x_l1, shard_l1, 1, config)
    del shard_l1
    parent_payload = (manager.RUNTIME / "f1_representation_bracket/doctor_auction/"
                      "P1_DUAL_PATH_RECOVERY_R16X2.k26f1")
    _, compact_output = causal.compact_expert_output(parent_payload, x_l1)
    layer1_arrays, layer1_details = causal.layer_one_paths(
        source, post_l1, x_l1, routes_l1, compact_output,
    )
    shard_l2 = reference.TensorShard(reference.shard_path(source, 3))
    teacher_post, teacher_x = causal.pre_moe(
        layer1_arrays["teacher_hidden_l1"], shard_l2, 2, config, segment_lengths,
    )
    student_post, student_x = causal.pre_moe(
        layer1_arrays["student_hidden_l1"], shard_l2, 2, config, segment_lengths,
    )
    teacher_route = causal.route_diagnostics(teacher_x, shard_l2, 2, config)
    student_route = causal.route_diagnostics(student_x, shard_l2, 2, config)
    teacher_feed_mx, _ = reference.routed_moe(
        mx.array(teacher_x).astype(mx.bfloat16), shard_l2, 2, config,
    )
    student_feed_mx, _ = reference.routed_moe(
        mx.array(student_x).astype(mx.bfloat16), shard_l2, 2, config,
    )
    teacher_hidden = causal.hidden_from_feed(
        teacher_post, np.asarray(teacher_feed_mx.astype(mx.float32), dtype=np.float32),
    )
    natural_hidden = causal.hidden_from_feed(
        student_post, np.asarray(student_feed_mx.astype(mx.float32), dtype=np.float32),
    )
    del shard_l2
    delta = bracket.predict(model, student_x)
    candidate_mx = (mx.array(natural_hidden).astype(mx.bfloat16) +
                    mx.array(delta * shrinkage).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(candidate_mx)
    candidate_hidden = np.asarray(candidate_mx.astype(mx.float32), dtype=np.float32)
    low_threshold = float(np.quantile(teacher_route["margin_8v9"], 0.25))
    low_margin = teacher_route["margin_8v9"] <= low_threshold
    all_mask = np.ones(len(token_ids), dtype=bool)
    all_summary = summarize(teacher_hidden, natural_hidden, candidate_hidden,
                            teacher_route, student_route, all_mask, seed)
    low_summary = summarize(teacher_hidden, natural_hidden, candidate_hidden,
                            teacher_route, student_route, low_margin, seed + 1)
    domain_summary = {}
    for index, domain in enumerate(sorted(set(record["domain"] for record in records))):
        mask = np.asarray([record["domain"] == domain for record in records])
        domain_summary[domain] = summarize(
            teacher_hidden, natural_hidden, candidate_hidden,
            teacher_route, student_route, mask, seed + 10 + index,
        )
    layer3 = replication.propagate_layer_three(
        source, {"TEACHER": teacher_hidden, "NATURAL_STUDENT": natural_hidden,
                 "POST_MOE_HIDDEN_CV_R12_S025": candidate_hidden},
        config, segment_lengths,
    )
    promoted = (
        all_summary["relative_l2_improvement"]["ci95_low"] > 0 and
        low_summary["relative_l2_improvement"]["ci95_low"] >= -0.001
    )
    arrays_to_save = {
        "teacher_x_l2": teacher_x, "student_x_l2": student_x,
        "teacher_hidden_l2": teacher_hidden, "natural_hidden_l2": natural_hidden,
        "candidate_hidden_l2": candidate_hidden,
        "teacher_route_indices_l2": teacher_route["indices"],
        "student_route_indices_l2": student_route["indices"],
        "teacher_margin_l2": teacher_route["margin_8v9"],
        "low_margin_mask": low_margin.astype(np.uint8),
    }
    capture_path = output_dir / "LR05_VALIDATION_CAPTURE.npz"
    temporary = capture_path.with_name(f".{capture_path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays_to_save)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, capture_path)
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_untouched_validation.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "The calibration-selected rank-12 quarter-strength repair improves untouched F2.",
        "config": {"seed": seed, "prompt_count": len(PROBES),
                   "segments": [probe["id"] for probe in PROBES],
                   "candidate_count": 1, "refit": False, "retuning": False,
                   "selection_parent": lr04["seal_sha256"]},
        "candidate": {"name": "POST_MOE_HIDDEN_CV_R12_S025",
                      "payload": payload_info,
                      "complete_physical_bytes": lr04["physical_candidate"][
                          "complete_physical_bytes"],
                      "complete_bpw": lr04["physical_candidate"]["complete_bpw"],
                      "doctor_allocation": lr04["physical_candidate"]["allocation"],
                      "shrinkage": shrinkage},
        "source": reference.source_identity(source),
        "tokenization": {"tokens": len(token_ids), "segment_lengths": segment_lengths,
                         "token_id_sha256": hashlib.sha256(
                             np.asarray(token_ids, dtype=np.int32).tobytes(),
                         ).hexdigest()},
        "input_capture": input_info, "layer1": layer1_details,
        "all_tokens": all_summary,
        "adversarial_low_margin": {"selection": "TEACHER_MARGIN_BOTTOM_QUARTILE",
                                   "threshold": low_threshold, **low_summary},
        "domain_summary": domain_summary,
        "layer3_amplification": layer3,
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": f1.sha256_file(capture_path)},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "evidence_parent": {"lr04_seal_sha256": lr04["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "decision": ("PROMOTE_TO_INDEPENDENT_REPLICATION" if promoted else
                     "RETIRE_SHRINKAGE_SURVIVOR_UNTOUCHED_FAILURE"),
        "causal_interpretation": (
            "The only calibration-CV survivor is judged once on an untouched split; route sets "
            "at layer 2 are unchanged by construction, so any rescue is attributable to the "
            "post-MoE hidden intervention rather than route selection."
        ),
        "next_run_rationale": (
            "Replicate with a new seed and adversarial margin distribution."
            if promoted else
            "The regularized post-MoE family is closed; use remaining time for causal replication."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": lr04["seal_sha256"],
        "candidate_hash": payload_info["sha256"],
        "physical_bytes": artifact["candidate"]["complete_physical_bytes"],
        "complete_bpw": artifact["candidate"]["complete_bpw"],
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": len(token_ids), "low_margin_tokens": int(np.sum(low_margin)),
                          "segments": len(PROBES)},
        "metrics": {"all_tokens": all_summary, "low_margin": low_summary,
                    "layer3": layer3},
        "confidence_intervals": {"all_tokens": all_summary["relative_l2_improvement"],
                                 "low_margin": low_summary["relative_l2_improvement"]},
        "faults": [], "retries": 0, "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "current_best_candidate": (artifact["candidate"]["name"] if promoted else
                                   prior_status["current_best_candidate"]),
        "current_best_bpw": (artifact["candidate"]["complete_bpw"] if promoted else
                             prior_status["current_best_bpw"]),
        "next_experiment": ("LR06_PROMOTED_ROW_REPLICATION" if promoted else
                            "LR06_CAUSAL_CLOSURE_REPLICATION"),
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "all_tokens": all_summary, "low_margin": low_summary,
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR05-complete",
        ("[Kimi K2.6 long run] LR05 untouched validation complete\n"
         f"tokens / low-margin: {len(token_ids)} / {int(np.sum(low_margin))}\n"
         f"mean per-token rescue: {all_summary['relative_l2_improvement']['mean']:.6f}\n"
         f"CI95 low: {all_summary['relative_l2_improvement']['ci95_low']:.6f}\n"
         f"decision: {artifact['decision']}\n"
         f"complete BPW: {artifact['candidate']['complete_bpw']:.6f}\n"
         f"free disk: {after['free_disk_bytes']/1024**3:.2f} GiB"),
    )
    manager.write_status(repo, {
        **status, "latest_result": {**status["latest_result"],
                                    "telegram_delivered": receipt["delivered"],
                                    "telegram_receipt_seal_sha256": receipt["seal_sha256"]},
    })
    gc.collect()
    mx.clear_cache()
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072105)
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.source.resolve(strict=True),
                       args.output_dir.resolve(), args.seed)
        print(json.dumps({"status": artifact["status"], "decision": artifact["decision"],
                          "tokens": artifact["tokenization"]["tokens"],
                          "seal_sha256": artifact["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
