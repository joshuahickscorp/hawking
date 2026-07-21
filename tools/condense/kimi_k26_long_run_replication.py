#!/usr/bin/env python3.12
"""New-split and adversarial low-margin replication of the LR02 physical bracket."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import ml_dtypes
import mlx.core as mx
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402
import kimi_k26_long_run_repair_bracket as bracket  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


EXPERIMENT_ID = "LR03_NEW_SPLIT_ADVERSARIAL_REPLICATION"
PROBES = [
    {"id": "new_factual", "domain": "prose",
     "text": "The Danube passes through several European capitals before reaching the Black Sea."},
    {"id": "new_science", "domain": "prose",
     "text": "A photon in vacuum has energy proportional to frequency and momentum inversely proportional to wavelength."},
    {"id": "new_code", "domain": "code",
     "text": "def merge_intervals(items):\n    items = sorted(items)\n    out = []\n    for lo, hi in items:\n        pass"},
    {"id": "new_math", "domain": "reasoning",
     "text": "Let a_n = 3 a_{n-1} - 2 with a_0 = 4. Derive a closed form and verify it by induction."},
    {"id": "new_logic", "domain": "reasoning",
     "text": "Mira arrives before Chen, Chen before Ivo, and Ivo is not last. Which order constraints follow?"},
    {"id": "new_instruction", "domain": "tool_format", "chat": True,
     "text": "Return only JSON with keys decision, evidence, and uncertainty. Do not add markdown."},
    {"id": "new_tool_protocol", "domain": "tool_format", "chat": True, "thinking": True,
     "text": "Inspect the checksum, compare the lease owner, then report whether execution may proceed."},
    {"id": "new_rare_unicode", "domain": "prose",
     "text": "The notation ⟦α⊗β⟧, ψ̂, and ∮Γ f(z) dz mixes brackets, accents, and contours."},
    {"id": "adversarial_punctuation", "domain": "code",
     "text": "x={\"a\":[1,2,3],\"b\":None}; y=x.get(\"c\",[])[:0] # ???"},
    {"id": "adversarial_boundary", "domain": "reasoning",
     "text": "Exactly seven, almost eight, perhaps nine: rank the alternatives when the eighth and ninth scores differ by ε."},
    {"id": "adversarial_multilingual", "domain": "prose",
     "text": "Résumé: naïve façade; Ελληνικά, العربية, हिन्दी, 日本語, and emoji 🧭 coexist."},
    {"id": "adversarial_schema", "domain": "tool_format", "chat": True,
     "text": "Emit <result status=\"ok\"><value>0.0001</value></result> and nothing else."},
]


def prepare_requests(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = reference.KimiTokenizer(source)
    requests = []
    token_ids = []
    segment_lengths = []
    records = []
    offset = 0
    for probe in PROBES:
        rendered = (tokenizer.user_prompt(probe["text"], thinking=bool(probe.get("thinking")))
                    if probe.get("chat") else probe["text"])
        ids = tokenizer.encode(rendered)
        requests.append({**probe, "rendered": rendered, "token_ids": ids})
        token_ids.extend(ids)
        segment_lengths.append(len(ids))
        for position, token_id in enumerate(ids):
            records.append({"token_index": offset + position, "segment": probe["id"],
                            "domain": probe["domain"], "position": position,
                            "token_id": int(token_id),
                            "token_text": tokenizer.decode([int(token_id)])})
        offset += len(ids)
    return requests, {"token_ids": token_ids, "segment_lengths": segment_lengths,
                      "token_records": records}


def read_payload(path: Path, expected_hash: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if f1.sha256_file(path) != expected_hash:
        raise f1.F1Error(f"physical payload hash mismatch: {path.name}")
    raw = path.read_bytes()
    if not raw.startswith(bracket.PRELUDE):
        raise f1.F1Error(f"invalid physical payload prelude: {path.name}")
    cursor = len(bracket.PRELUDE)
    header_length = struct.unpack("<I", raw[cursor:cursor + 4])[0]
    cursor += 4
    header = json.loads(raw[cursor:cursor + header_length])
    data_start = cursor + header_length
    arrays = {}
    dtype_map = {"bfloat16": ml_dtypes.bfloat16}
    for component in header["components"]:
        dtype = (dtype_map[component["dtype"]] if component["dtype"] in dtype_map
                 else np.dtype(component["dtype"]))
        start = data_start + component["offset"]
        end = start + component["bytes"]
        arrays[component["name"]] = np.frombuffer(
            raw[start:end], dtype=dtype,
        ).reshape(component["shape"]).copy()
    return header, arrays


def decode_model(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, Any]:
    projection_q = arrays[f"{prefix}.projection_q"]
    projection_scale = arrays[f"{prefix}.projection_scale"]
    basis_q = arrays[f"{prefix}.basis_q"]
    basis_scale = arrays[f"{prefix}.basis_scale"]
    return {
        "projection": projection_q.astype(np.float32) * projection_scale[None, :],
        "basis": basis_q.astype(np.float32) * basis_scale[:, None],
        "bias": np.asarray(arrays[f"{prefix}.bias_bf16"], dtype=np.float32),
    }


def candidate_metrics(
    name: str, teacher_hidden: np.ndarray, natural_hidden: np.ndarray,
    candidate_hidden: np.ndarray, teacher_route: dict[str, np.ndarray],
    candidate_route: dict[str, np.ndarray], low_margin: np.ndarray,
    records: list[dict[str, Any]], prior: dict[str, Any], seed: int,
) -> dict[str, Any]:
    teacher_indices = teacher_route["indices"]
    candidate_indices = candidate_route["indices"]
    exact = np.asarray([set(left) == set(right) for left, right in zip(
        teacher_indices, candidate_indices, strict=True,
    )])
    baseline_rows = causal.row_metrics(teacher_hidden, natural_hidden)
    rows = causal.row_metrics(teacher_hidden, candidate_hidden)
    domains = {}
    for domain in sorted(set(record["domain"] for record in records)):
        mask = np.asarray([record["domain"] == domain for record in records])
        domains[domain] = {
            "tokens": int(np.sum(mask)),
            "row_relative_l2_mean": float(np.mean(rows["relative_l2"][mask])),
            "route_set_agreement": float(np.mean(exact[mask])),
        }
    return {
        "candidate": name, "complete_bpw": prior["complete_bpw"],
        "complete_physical_bytes": prior["complete_physical_bytes"],
        "payload_sha256": prior["physical_payload"]["sha256"],
        "all_tokens": {"hidden": f1.quality(teacher_hidden, candidate_hidden),
                       "row_relative_l2_mean": float(np.mean(rows["relative_l2"])),
                       "route_set_agreement": float(np.mean(exact)),
                       "route_matches": int(np.sum(exact)),
                       "relative_l2_improvement": bracket.paired_interval(
                           baseline_rows["relative_l2"] - rows["relative_l2"], seed,
                       )},
        "adversarial_low_margin": {
            "tokens": int(np.sum(low_margin)),
            "hidden": f1.quality(teacher_hidden[low_margin], candidate_hidden[low_margin]),
            "row_relative_l2_mean": float(np.mean(rows["relative_l2"][low_margin])),
            "route_set_agreement": float(np.mean(exact[low_margin])),
            "route_matches": int(np.sum(exact[low_margin])),
            "relative_l2_improvement": bracket.paired_interval(
                baseline_rows["relative_l2"][low_margin] - rows["relative_l2"][low_margin],
                seed + 100,
            ),
        },
        "domain_summary": domains,
        "prior_lr02_decision": prior["decision"],
        "prior_lr02_improvement": prior["relative_l2_improvement"],
    }


def propagate_layer_three(
    source: Path, variants: dict[str, np.ndarray], config: dict[str, Any],
    segment_lengths: list[int],
) -> dict[str, Any]:
    shard = reference.TensorShard(reference.shard_path(source, 4))
    outputs = {}
    infos = {}
    for name, hidden in variants.items():
        output, info = reference.layer_forward(
            mx.array(hidden).astype(mx.bfloat16), shard, 3, config, segment_lengths,
        )
        outputs[name] = np.asarray(output.astype(mx.float32), dtype=np.float32)
        infos[name] = info
        mx.clear_cache()
    teacher_routes = np.asarray(infos["TEACHER"]["moe"]["route_indices"])
    result = {}
    for name, output in outputs.items():
        routes = np.asarray(infos[name]["moe"]["route_indices"])
        exact = np.asarray([set(left) == set(right) for left, right in zip(
            teacher_routes, routes, strict=True,
        )])
        result[name] = {"hidden": f1.quality(outputs["TEACHER"], output),
                        "route_set_agreement": float(np.mean(exact)),
                        "route_matches": int(np.sum(exact)),
                        "hidden_sha256": hashlib.sha256(output.tobytes()).hexdigest()}
    del shard
    gc.collect()
    mx.clear_cache()
    return result


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
                                "next_experiment": "LR03_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "The LR02 held-out harm replicates on new contexts, especially low-margin tokens.",
        "config": {"seed": seed, "prompt_count": len(PROBES),
                   "payload_refit": False, "adversarial_selection": "TEACHER_MARGIN_BOTTOM_QUARTILE"},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    lr02 = f1.read_json(repo / "KIMI_K26_LONG_RUN_REPAIR_BRACKET.json")
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
    teacher_feed = np.asarray(teacher_feed_mx.astype(mx.float32), dtype=np.float32)
    student_feed = np.asarray(student_feed_mx.astype(mx.float32), dtype=np.float32)
    teacher_hidden = causal.hidden_from_feed(teacher_post, teacher_feed)
    natural_hidden = causal.hidden_from_feed(student_post, student_feed)
    low_threshold = float(np.quantile(teacher_route["margin_8v9"], 0.25))
    low_margin = teacher_route["margin_8v9"] <= low_threshold
    candidate_hidden = {}
    candidate_route = {}
    payload_headers = {}
    frontier_by_name = {row["candidate"]: row for row in lr02["treatment_frontier"]}

    def load(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
        row = frontier_by_name[name]
        header, values = read_payload(Path(row["physical_payload"]["path"]),
                                      row["physical_payload"]["sha256"])
        payload_headers[name] = header
        return header, values, row

    header, values, _ = load("FIRST_DIVERGENCE_PROTECTION_R12")
    risk_model = decode_model(values, "risk_state")
    threshold = float(header["metadata"]["activation_margin_threshold"])
    risk_mask = student_route["margin_8v9"] <= threshold
    risk_x = student_x.copy()
    risk_x[risk_mask] += bracket.predict(risk_model, student_x[risk_mask])
    risk_feed_mx, _ = reference.routed_moe(
        mx.array(risk_x).astype(mx.bfloat16), shard_l2, 2, config,
    )
    candidate_hidden["FIRST_DIVERGENCE_PROTECTION_R12"] = causal.hidden_from_feed(
        student_post, np.asarray(risk_feed_mx.astype(mx.float32), dtype=np.float32),
    )
    candidate_route["FIRST_DIVERGENCE_PROTECTION_R12"] = causal.route_diagnostics(
        risk_x, shard_l2, 2, config,
    )

    _, values, _ = load("PRE_ROUTER_STATE_R24")
    state_model = decode_model(values, "state")
    state_x = student_x + bracket.predict(state_model, student_x)
    state_feed_mx, _ = reference.routed_moe(
        mx.array(state_x).astype(mx.bfloat16), shard_l2, 2, config,
    )
    candidate_hidden["PRE_ROUTER_STATE_R24"] = causal.hidden_from_feed(
        student_post, np.asarray(state_feed_mx.astype(mx.float32), dtype=np.float32),
    )
    candidate_route["PRE_ROUTER_STATE_R24"] = causal.route_diagnostics(
        state_x, shard_l2, 2, config,
    )

    header, values, _ = load("LOW_MARGIN_ROUTER_R24")
    router_model = decode_model(values, "router")
    router_threshold = float(header["metadata"]["activation_margin_threshold"])
    router_mask = student_route["margin_8v9"] <= router_threshold
    router_logits = student_route["logits"].copy()
    router_logits[router_mask] += bracket.predict(router_model, student_x[router_mask])
    correction = np.asarray(shard_l2.mlx(
        f"{reference.PREFIX}.layers.2.mlp.gate.e_score_correction_bias"
    ).astype(mx.float32), dtype=np.float32)
    router_routes = bracket.route_from_logits(router_logits, correction, config)
    router_feed = bracket.custom_routed_moe(student_x, shard_l2, 2, router_routes)
    candidate_hidden["LOW_MARGIN_ROUTER_R24"] = causal.hidden_from_feed(
        student_post, router_feed,
    )
    candidate_route["LOW_MARGIN_ROUTER_R24"] = router_routes

    _, values, _ = load("WEIGHTED_MOE_OUTPUT_R24")
    moe_model = decode_model(values, "moe_output")
    moe_delta = bracket.predict(moe_model, student_x)
    moe_hidden_mx = (mx.array(natural_hidden).astype(mx.bfloat16) +
                     mx.array(moe_delta).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(moe_hidden_mx)
    candidate_hidden["WEIGHTED_MOE_OUTPUT_R24"] = np.asarray(
        moe_hidden_mx.astype(mx.float32), dtype=np.float32,
    )
    candidate_route["WEIGHTED_MOE_OUTPUT_R24"] = student_route

    _, values, _ = load("HYBRID_R12X2")
    hybrid_state = decode_model(values, "hybrid_state")
    hybrid_output = decode_model(values, "hybrid_output")
    hybrid_x = student_x + bracket.predict(hybrid_state, student_x)
    hybrid_feed_mx, _ = reference.routed_moe(
        mx.array(hybrid_x).astype(mx.bfloat16), shard_l2, 2, config,
    )
    hybrid_pre_hidden = causal.hidden_from_feed(
        student_post, np.asarray(hybrid_feed_mx.astype(mx.float32), dtype=np.float32),
    )
    hybrid_hidden_mx = (mx.array(hybrid_pre_hidden).astype(mx.bfloat16) + mx.array(
        bracket.predict(hybrid_output, student_x),
    ).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(hybrid_hidden_mx)
    candidate_hidden["HYBRID_R12X2"] = np.asarray(
        hybrid_hidden_mx.astype(mx.float32), dtype=np.float32,
    )
    candidate_route["HYBRID_R12X2"] = causal.route_diagnostics(
        hybrid_x, shard_l2, 2, config,
    )
    del shard_l2
    gc.collect()
    mx.clear_cache()

    baseline_exact = np.asarray([set(left) == set(right) for left, right in zip(
        teacher_route["indices"], student_route["indices"], strict=True,
    )])
    baseline_rows = causal.row_metrics(teacher_hidden, natural_hidden)
    baseline = {
        "tokens": len(token_ids), "hidden": f1.quality(teacher_hidden, natural_hidden),
        "row_relative_l2_mean": float(np.mean(baseline_rows["relative_l2"])),
        "route_set_agreement": float(np.mean(baseline_exact)),
        "route_matches": int(np.sum(baseline_exact)),
        "adversarial_low_margin": {
            "selection": "TEACHER_MARGIN_BOTTOM_QUARTILE",
            "threshold": low_threshold, "tokens": int(np.sum(low_margin)),
            "hidden": f1.quality(teacher_hidden[low_margin], natural_hidden[low_margin]),
            "row_relative_l2_mean": float(np.mean(baseline_rows["relative_l2"][low_margin])),
            "route_set_agreement": float(np.mean(baseline_exact[low_margin])),
            "route_matches": int(np.sum(baseline_exact[low_margin])),
        },
    }
    results = []
    for index, name in enumerate(frontier_by_name):
        results.append(candidate_metrics(
            name, teacher_hidden, natural_hidden, candidate_hidden[name], teacher_route,
            candidate_route[name], low_margin, records, frontier_by_name[name], seed + index,
        ))
    strongest = min(results, key=lambda row: row["all_tokens"]["hidden"]["relative_l2"])
    propagated = propagate_layer_three(
        source, {"TEACHER": teacher_hidden, "NATURAL_STUDENT": natural_hidden,
                 strongest["candidate"]: candidate_hidden[strongest["candidate"]]},
        config, segment_lengths,
    )
    for row in results:
        consistent_benefit = (
            row["prior_lr02_improvement"]["ci95_low"] > 0 and
            row["all_tokens"]["relative_l2_improvement"]["ci95_low"] > 0
        )
        row["replication_decision"] = (
            "PROMOTION_EVIDENCE" if consistent_benefit else
            "RETIRED_NO_TWO_SPLIT_BENEFIT"
        )
    arrays_to_save = {
        "teacher_x_l2": teacher_x, "student_x_l2": student_x,
        "teacher_hidden_l2": teacher_hidden, "natural_hidden_l2": natural_hidden,
        "teacher_route_indices_l2": teacher_route["indices"],
        "student_route_indices_l2": student_route["indices"],
        "teacher_margin_l2": teacher_route["margin_8v9"],
        "low_margin_mask": low_margin.astype(np.uint8),
        **{f"candidate_hidden_{name}": value for name, value in candidate_hidden.items()},
    }
    capture_path = output_dir / "LR03_REPLICATION_CAPTURE.npz"
    temporary = capture_path.with_name(f".{capture_path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays_to_save)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, capture_path)
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    any_promotable = any(row["replication_decision"] == "PROMOTION_EVIDENCE"
                         for row in results)
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_replication.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "The LR02 held-out harm replicates on new contexts, especially low-margin tokens.",
        "config": {"seed": seed, "prompt_count": len(PROBES),
                   "segments": [probe["id"] for probe in PROBES],
                   "payload_refit": False, "payload_selection_uses_replication": False,
                   "adversarial_selection": "TEACHER_MARGIN_BOTTOM_QUARTILE"},
        "source": reference.source_identity(source),
        "tokenization": {"tokens": len(token_ids), "segment_lengths": segment_lengths,
                         "token_id_sha256": hashlib.sha256(
                             np.asarray(token_ids, dtype=np.int32).tobytes(),
                         ).hexdigest()},
        "input_capture": input_info,
        "layer1": layer1_details, "baseline": baseline,
        "replication_frontier": results,
        "strongest_by_norm_weighted_f2": strongest["candidate"],
        "layer3_amplification": propagated,
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": f1.sha256_file(capture_path)},
        "evidence_parent": {"lr02_seal_sha256": lr02["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": ("REOPEN_FOR_INDEPENDENT_CONFIRMATION" if any_promotable else
                     "REPLICATED_NO_PROMOTABLE_REPAIR"),
        "causal_interpretation": (
            "A route-only or low-rank state/MoE repair must improve both the original held-out "
            "split and this untouched replication split; otherwise apparent norm-weighted rescue "
            "is not a robust capability-preserving representation."
        ),
        "next_run_rationale": (
            "Run shrinkage/rank boundary falsification from calibration-only cross-validation, "
            "then validate the single preregistered winner on another untouched split."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_REPLICATION.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": lr02["seal_sha256"],
        "candidate_hash": None, "physical_bytes": None, "complete_bpw": None,
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": len(token_ids), "low_margin_tokens": int(np.sum(low_margin)),
                          "segments": len(PROBES), "rows": len(results)},
        "metrics": {"baseline": baseline, "replication_frontier": results,
                    "layer3_amplification": propagated},
        "confidence_intervals": {row["candidate"]:
                                 row["all_tokens"]["relative_l2_improvement"] for row in results},
        "faults": [], "retries": 0, "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "next_experiment": "LR04_CALIBRATION_ONLY_SHRINKAGE_BOUNDARY",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "tokens": len(token_ids), "low_margin_tokens": int(np.sum(low_margin)),
                          "strongest_norm_weighted": strongest["candidate"],
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR03-complete",
        ("[Kimi K2.6 long run] LR03 new-split replication complete\n"
         f"tokens / low-margin: {len(token_ids)} / {int(np.sum(low_margin))}\n"
         f"baseline route change: {(1-baseline['route_set_agreement'])*100:.2f}%\n"
         f"decision: {artifact['decision']}\n"
         f"strongest norm-weighted row: {strongest['candidate']}\n"
         f"free disk: {after['free_disk_bytes']/1024**3:.2f} GiB\n"
         "next: calibration-only shrinkage/rank boundary"),
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
    parser.add_argument("--seed", type=int, default=26072103)
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
