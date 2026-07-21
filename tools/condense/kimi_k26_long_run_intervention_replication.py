#!/usr/bin/env python3.12
"""Large stratified replication of Kimi teacher/student router and MoE interventions."""
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
import kimi_k26_long_run_large_falsification as large  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402
import kimi_k26_long_run_repair_bracket as bracket  # noqa: E402
import kimi_k26_long_run_replication as replication  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


EXPERIMENT_ID = "LR08_LARGE_INTERVENTION_REPLICATION"
SUFFIXES = [
    " Audit the argument by naming a hidden assumption, a measurable consequence, and the smallest intervention that separates competing causes.",
    " Reframe the case for an adversarial reviewer: preserve exact quantities, identify the boundary condition, and state what evidence would reverse the decision.",
]


def probes() -> list[dict[str, Any]]:
    result = []
    for index, (domain, base) in enumerate(reversed(large.BASES)):
        for variant, suffix in enumerate(SUFFIXES):
            chat = domain == "tool_format"
            result.append({"id": f"intervention_{index:02d}_{variant}", "domain": domain,
                           "text": base + suffix, "chat": chat,
                           "thinking": bool(chat and variant == 0)})
    return result


def prepare_requests(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = reference.KimiTokenizer(source)
    requests = []
    token_ids = []
    lengths = []
    records = []
    offset = 0
    for probe in probes():
        rendered = (tokenizer.user_prompt(probe["text"], thinking=probe["thinking"])
                    if probe["chat"] else probe["text"])
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


def stratified_selection(
    mismatch: np.ndarray, margin: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    median = float(np.median(margin))
    rng = np.random.default_rng(seed)
    definitions = {
        "MISMATCH_LOW_MARGIN": mismatch & (margin <= median),
        "MISMATCH_HIGH_MARGIN": mismatch & (margin > median),
        "MATCH_LOW_MARGIN": ~mismatch & (margin <= median),
        "MATCH_HIGH_MARGIN": ~mismatch & (margin > median),
    }
    selected = np.zeros(mismatch.size, dtype=bool)
    record = {}
    for name, mask in definitions.items():
        available = np.where(mask)[0]
        count = min(64, available.size)
        chosen = np.sort(rng.choice(available, size=count, replace=False)) if count else available
        selected[chosen] = True
        record[name] = {"available": int(available.size), "selected": int(count),
                        "selected_indices_sha256": hashlib.sha256(
                            chosen.astype(np.int32).tobytes(),
                        ).hexdigest()}
    return selected, {"margin_median": median, "strata": record,
                      "total_selected": int(np.sum(selected))}


def intervention_metric(
    teacher: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    baseline_metric = f1.quality(teacher[selected], baseline[selected])
    metric = f1.quality(teacher[selected], candidate[selected])
    baseline_error = baseline_metric["relative_l2"]
    return {"hidden": metric,
            "rescue_fraction_relative_l2": float(
                1 - metric["relative_l2"] / (baseline_error + 1e-30)
            ),
            "baseline_relative_l2": baseline_error,
            "tokens": int(np.sum(selected))}


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
                                "next_experiment": "LR08_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "Large stratified interventions reproduce mixed state-first, route-secondary causality.",
        "config": {"seed": seed, "prompt_count": len(probes()),
                   "strata_cap": 64, "candidate_refit": False},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_law = f1.read_json(repo / "KIMI_K26_LONG_RUN_POOLED_CAUSAL_LAW.json")
    requests, batch = prepare_requests(source)
    token_ids = batch["token_ids"]
    lengths = batch["segment_lengths"]
    post_l1, x_l1, input_info = causal.capture_layer_one_inputs(source, token_ids, lengths)
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
    shard = reference.TensorShard(reference.shard_path(source, 3))
    teacher_post, teacher_x = causal.pre_moe(
        layer1_arrays["teacher_hidden_l1"], shard, 2, config, lengths,
    )
    student_post, student_x = causal.pre_moe(
        layer1_arrays["student_hidden_l1"], shard, 2, config, lengths,
    )
    teacher_route = causal.route_diagnostics(teacher_x, shard, 2, config)
    student_route = causal.route_diagnostics(student_x, shard, 2, config)
    teacher_feed_mx, _ = reference.routed_moe(
        mx.array(teacher_x).astype(mx.bfloat16), shard, 2, config,
    )
    student_feed_mx, _ = reference.routed_moe(
        mx.array(student_x).astype(mx.bfloat16), shard, 2, config,
    )
    teacher_feed = np.asarray(teacher_feed_mx.astype(mx.float32), dtype=np.float32)
    student_feed = np.asarray(student_feed_mx.astype(mx.float32), dtype=np.float32)
    teacher_hidden = causal.hidden_from_feed(teacher_post, teacher_feed)
    natural_hidden = causal.hidden_from_feed(student_post, student_feed)
    mismatch = np.asarray([set(left) != set(right) for left, right in zip(
        teacher_route["indices"], student_route["indices"], strict=True,
    )])
    selected, selection = stratified_selection(
        mismatch, teacher_route["margin_8v9"], seed,
    )
    student_scores_teacher_weights = causal.weights_for_indices(
        student_route["scores"], teacher_route["indices"], config,
    )
    force_indices_route = {
        "indices": student_route["indices"].copy(),
        "weights": student_route["weights"].copy(),
    }
    force_indices_route["indices"][selected] = teacher_route["indices"][selected]
    force_indices_route["weights"][selected] = student_scores_teacher_weights[selected]
    force_both_route = {
        "indices": student_route["indices"].copy(),
        "weights": student_route["weights"].copy(),
    }
    force_both_route["indices"][selected] = teacher_route["indices"][selected]
    force_both_route["weights"][selected] = teacher_route["weights"][selected]
    natural_custom_route = {"indices": student_route["indices"],
                            "weights": student_route["weights"]}
    natural_custom_feed = bracket.custom_routed_moe(
        student_x, shard, 2, natural_custom_route,
    )
    force_indices_feed = bracket.custom_routed_moe(
        student_x, shard, 2, force_indices_route,
    )
    force_both_feed = bracket.custom_routed_moe(
        student_x, shard, 2, force_both_route,
    )
    teacher_state_student_route = {"indices": teacher_route["indices"].copy(),
                                   "weights": teacher_route["weights"].copy()}
    teacher_state_student_route["indices"][selected] = student_route["indices"][selected]
    teacher_state_student_route["weights"][selected] = student_route["weights"][selected]
    teacher_state_student_feed = bracket.custom_routed_moe(
        teacher_x, shard, 2, teacher_state_student_route,
    )
    calibration_hidden = causal.hidden_from_feed(student_post, natural_custom_feed)
    force_indices_hidden = causal.hidden_from_feed(student_post, force_indices_feed)
    force_both_hidden = causal.hidden_from_feed(student_post, force_both_feed)
    teacher_state_student_hidden = causal.hidden_from_feed(
        teacher_post, teacher_state_student_feed,
    )
    substitute_feed = student_feed.copy()
    substitute_feed[selected] = teacher_feed[selected]
    substitute_hidden = causal.hidden_from_feed(student_post, substitute_feed)
    restore_hidden = natural_hidden.copy()
    restore_hidden[selected] = teacher_hidden[selected]
    interventions = {
        "NATURAL_STUDENT": intervention_metric(
            teacher_hidden, natural_hidden, natural_hidden, selected,
        ),
        "FORCE_TEACHER_INDICES": intervention_metric(
            teacher_hidden, natural_hidden, force_indices_hidden, selected,
        ),
        "FORCE_TEACHER_INDICES_AND_WEIGHTS": intervention_metric(
            teacher_hidden, natural_hidden, force_both_hidden, selected,
        ),
        "TEACHER_STATE_STUDENT_ROUTER": intervention_metric(
            teacher_hidden, natural_hidden, teacher_state_student_hidden, selected,
        ),
        "SUBSTITUTE_TEACHER_WEIGHTED_MOE": intervention_metric(
            teacher_hidden, natural_hidden, substitute_hidden, selected,
        ),
        "RESTORE_TEACHER_HIDDEN": intervention_metric(
            teacher_hidden, natural_hidden, restore_hidden, selected,
        ),
    }
    calibration = f1.quality(natural_hidden, calibration_hidden)
    layer3 = replication.propagate_layer_three(
        source, {"TEACHER": teacher_hidden, "NATURAL_STUDENT": natural_hidden,
                 "SUBSTITUTE_TEACHER_MOE_SELECTED": substitute_hidden},
        config, lengths,
    )
    del shard
    gc.collect()
    mx.clear_cache()
    lr01_interventions = parent_law["intervention_evidence"]
    replicated = (
        interventions["FORCE_TEACHER_INDICES_AND_WEIGHTS"][
            "rescue_fraction_relative_l2"] < 0.5 and
        interventions["SUBSTITUTE_TEACHER_WEIGHTED_MOE"][
            "rescue_fraction_relative_l2"] >
        interventions["FORCE_TEACHER_INDICES_AND_WEIGHTS"][
            "rescue_fraction_relative_l2"] and
        interventions["RESTORE_TEACHER_HIDDEN"]["rescue_fraction_relative_l2"] > 0.99
    )
    arrays_to_save = {
        "teacher_x_l2": teacher_x, "student_x_l2": student_x,
        "teacher_hidden_l2": teacher_hidden, "natural_hidden_l2": natural_hidden,
        "force_indices_hidden_l2": force_indices_hidden,
        "force_both_hidden_l2": force_both_hidden,
        "teacher_state_student_hidden_l2": teacher_state_student_hidden,
        "substitute_teacher_moe_hidden_l2": substitute_hidden,
        "selected_mask": selected.astype(np.uint8),
        "teacher_route_indices_l2": teacher_route["indices"],
        "student_route_indices_l2": student_route["indices"],
        "teacher_margin_l2": teacher_route["margin_8v9"],
    }
    capture_path = output_dir / "LR08_INTERVENTION_CAPTURE.npz"
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
        "schema": "hawking.kimi_k26.long_run_intervention_replication.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Large stratified interventions reproduce mixed state-first, route-secondary causality.",
        "config": {"seed": seed, "prompt_count": len(requests), "strata_cap": 64,
                   "prompt_construction": "REVERSED_LR07_BASES_WITH_NEW_SUFFIXES",
                   "selected_tokens": int(np.sum(selected))},
        "source": reference.source_identity(source),
        "tokenization": {"tokens": len(token_ids), "segment_lengths": lengths,
                         "token_id_sha256": hashlib.sha256(
                             np.asarray(token_ids, dtype=np.int32).tobytes(),
                         ).hexdigest()},
        "input_capture": input_info, "layer1": layer1_details,
        "baseline": {"hidden": f1.quality(teacher_hidden, natural_hidden),
                     "route_set_change": float(np.mean(mismatch)),
                     "route_mismatch_tokens": int(np.sum(mismatch))},
        "stratified_selection": selection,
        "counterfactual_recombination_calibration": {
            "manual_custom_vs_resident_natural": calibration,
            "lr01_numerical_floor_relative_l2": lr01_interventions[
                "counterfactual_numerical_floor_relative_l2"],
        },
        "intervention_matrix": interventions,
        "layer3_amplification": layer3,
        "replication_comparison": {"lr01": lr01_interventions,
                                   "lr08": {key: value["rescue_fraction_relative_l2"]
                                            for key, value in interventions.items()}},
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": f1.sha256_file(capture_path)},
        "evidence_parent": {"pooled_law_seal_sha256": parent_law["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": ("REPLICATE_MIXED_STATE_FIRST_ROUTE_SECONDARY_CAUSALITY" if replicated else
                     "CAUSAL_ORDER_NOT_REPLICATED_REOPEN_DIAGNOSIS"),
        "causal_interpretation": (
            "The stratified swap matrix tests route selection and weighted MoE output separately "
            "from upstream state restoration on a new long-context distribution."
        ),
        "next_run_rationale": (
            "Estimate intervention rescue by stratum and test the causal ordering under a new seed."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_INTERVENTION_REPLICATION.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": parent_law["seal_sha256"],
        "candidate_hash": None, "physical_bytes": None, "complete_bpw": None,
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": len(token_ids), "selected_tokens": int(np.sum(selected)),
                          "segments": len(requests)},
        "metrics": {"baseline": artifact["baseline"], "interventions": interventions,
                    "layer3": layer3, "calibration": calibration},
        "confidence_intervals": {}, "faults": [], "retries": 0,
        "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "primary_causal_diagnosis": "UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_AMPLIFIER",
        "next_experiment": "LR09_STRATIFIED_RESCUE_CONFIDENCE",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "tokens": len(token_ids), "selected_tokens": int(np.sum(selected)),
                          "interventions": {key: value["rescue_fraction_relative_l2"]
                                            for key, value in interventions.items()},
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR08-complete",
        ("[Kimi K2.6 long run] LR08 intervention replication complete\n"
         f"tokens / stratified interventions: {len(token_ids)} / {int(np.sum(selected))}\n"
         f"route+weight rescue: {interventions['FORCE_TEACHER_INDICES_AND_WEIGHTS']['rescue_fraction_relative_l2']:.3f}\n"
         f"teacher-MoE rescue: {interventions['SUBSTITUTE_TEACHER_WEIGHTED_MOE']['rescue_fraction_relative_l2']:.3f}\n"
         f"teacher-hidden rescue: {interventions['RESTORE_TEACHER_HIDDEN']['rescue_fraction_relative_l2']:.3f}\n"
         f"decision: {artifact['decision']}\n"
         f"free disk: {after['free_disk_bytes']/1024**3:.2f} GiB"),
    )
    manager.write_status(repo, {
        **status, "latest_result": {**status["latest_result"],
                                    "telegram_delivered": receipt["delivered"],
                                    "telegram_receipt_seal_sha256": receipt["seal_sha256"]},
    })
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072108)
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
