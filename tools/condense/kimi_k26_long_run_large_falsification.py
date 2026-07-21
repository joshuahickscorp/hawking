#!/usr/bin/env python3.12
"""Large long-context held-out falsification of the pooled Kimi causal law."""
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
import kimi_k26_long_run_validation as validation  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


EXPERIMENT_ID = "LR07_LARGE_HELDOUT_FALSIFICATION"
BASES = [
    ("prose", "Ocean circulation redistributes heat, dissolved gases, nutrients, and salt. "
     "Surface winds initiate broad currents while temperature and salinity gradients drive "
     "deep overturning. Measurements from floats, satellites, and research vessels disagree "
     "slightly because they sample different scales and times. A careful summary must separate "
     "direct observations from model-dependent inference."),
    ("prose", "A municipal archive contains council minutes, engineering drawings, tax rolls, "
     "and correspondence written under changing administrative rules. Dates may refer to filing, "
     "approval, or later transcription. The historian cross-checks handwriting, paper, and cited "
     "events before treating any claim as contemporaneous evidence."),
    ("prose", "Photosynthesis couples photon absorption to charge separation and chemical storage. "
     "Pigment complexes transfer excitation toward reaction centers, but excess energy must be "
     "dissipated to avoid damage. Temperature, water availability, and carbon dioxide alter the "
     "observed rate, so a single limiting-factor explanation is rarely sufficient."),
    ("code", "A streaming parser receives arbitrary byte chunks and must emit complete records "
     "without losing delimiters split across reads. The implementation tracks quote state, escape "
     "state, nesting depth, and a bounded buffer. It rejects malformed input deterministically "
     "and reports the absolute byte offset of the first error."),
    ("code", "A concurrent work queue supports cancellation, retry limits, and idempotent result "
     "publication. Workers claim leases with monotonic deadlines, renew only while making progress, "
     "and write results through compare-and-swap. Recovery must distinguish an expired worker from "
     "a slow but healthy one without executing irreversible work twice."),
    ("code", "A numerical kernel computes stable log-sum-exp over rows containing finite values, "
     "negative infinity, and occasionally NaN. The contract defines propagation rules, empty-row "
     "behavior, dtype promotion, and a tolerance for reference comparisons. Tests cover extreme "
     "magnitudes and non-contiguous slices."),
    ("reasoning", "Four laboratories share two instruments under precedence, maintenance, and "
     "cool-down constraints. Atlas must finish before Boreal begins; Cygnus cannot overlap maintenance; "
     "Delta needs both instruments for its final stage. Determine which ordering claims follow in "
     "every feasible schedule and which depend on an unstated duration."),
    ("reasoning", "A bag contains red, blue, and green tokens. Two are drawn without replacement, "
     "a color-dependent token is then added, and a third draw is made. Compare conditional outcomes "
     "without assuming independence. State the sample space explicitly and identify which observations "
     "change the denominator."),
    ("reasoning", "An integer sequence is defined by alternating affine recurrences. On odd steps it "
     "triples and subtracts four; on even steps it halves after adding six. Determine fixed points, "
     "possible cycles, and the assumptions required for every term to remain integral."),
    ("tool_format", "The requested operation must first validate a content hash, confirm that the lease "
     "owner matches the running process, and check that free space remains above a hard floor. Return "
     "a machine-readable decision with separate evidence fields. Never infer success from the absence "
     "of an error message."),
    ("tool_format", "Transform the supplied observations into one JSON object with keys hypothesis, "
     "measurement, uncertainty, and next_test. Preserve numeric precision, represent unavailable values "
     "as null, and do not insert explanatory markdown. A failed invariant must set status to blocked "
     "and name the exact invariant."),
    ("tool_format", "Review a proposed database migration in three phases: reversible preflight, bounded "
     "execution, and postcondition verification. The output schema includes affected_rows, rollback, "
     "duration_ms, and evidence_hash. Do not claim rollback is possible unless the backup identity has "
     "been verified."),
    ("prose", "The expression ⟨ψ|H|ψ⟩ combines brackets, operators, and a normalized state. Nearby text "
     "may include ξ→0⁺, an integral around Γ, accented names, Arabic العربية, Japanese 日本語, and emoji. "
     "Token boundaries should not be mistaken for semantic boundaries when scripts and punctuation mix."),
    ("code", "Given rows like {id:17, score:0.00001, tags:[\"α\",\"β\"]}, sort by descending score, "
     "then ascending id, while preserving stability for exact ties. Serialize with canonical key order, "
     "reject duplicate identifiers, and compute a checksum over UTF-8 bytes rather than displayed glyphs."),
    ("reasoning", "At a top-eight cutoff, candidates in positions eight and nine differ by a very small "
     "margin. A perturbation changes both scores and normalization weights. Explain why a set change, a "
     "rank change without set change, and a weight change with neither rank nor set change are distinct "
     "events with different downstream consequences."),
    ("tool_format", "Emit exactly one XML element named result with attributes status and confidence. "
     "Its children must be evidence, diagnosis, and next-experiment in that order. Escape ampersands, "
     "preserve Unicode content, and include no preamble or trailing commentary."),
]
SUFFIXES = [
    " First list the observations, then the causal claims, then one falsifying test. Keep correlated evidence separate from interventions.",
    " Consider a boundary case in which two measurements differ by one part in one hundred thousand, and explain how confidence changes.",
]


def probes() -> list[dict[str, Any]]:
    result = []
    for index, (domain, base) in enumerate(BASES):
        for variant, suffix in enumerate(SUFFIXES):
            chat = domain == "tool_format"
            result.append({"id": f"large_{index:02d}_{variant}", "domain": domain,
                           "text": base + suffix, "chat": chat,
                           "thinking": bool(chat and variant == 1)})
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
                                "next_experiment": "LR07_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "The pooled mixed-causality law and shrinkage failure survive longer contexts.",
        "config": {"seed": seed, "prompt_count": len(probes()),
                   "candidate_count": 1, "refit": False, "retuning": False},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_law = f1.read_json(repo / "KIMI_K26_LONG_RUN_POOLED_CAUSAL_LAW.json")
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
    margin = teacher_route["margin_8v9"]
    low_threshold = float(np.quantile(margin, 0.25))
    low_margin = margin <= low_threshold
    all_mask = np.ones(len(token_ids), dtype=bool)
    all_summary = validation.summarize(
        teacher_hidden, natural_hidden, candidate_hidden,
        teacher_route, student_route, all_mask, seed,
    )
    low_summary = validation.summarize(
        teacher_hidden, natural_hidden, candidate_hidden,
        teacher_route, student_route, low_margin, seed + 1,
    )
    teacher_indices = teacher_route["indices"]
    student_indices = student_route["indices"]
    mismatch = np.asarray([set(left) != set(right) for left, right in zip(
        teacher_indices, student_indices, strict=True,
    )])
    baseline_rows = causal.row_metrics(teacher_hidden, natural_hidden)["relative_l2"]
    margin_quartile = np.digitize(margin, np.quantile(margin, [0.25, 0.5, 0.75]))
    atlas = {}
    for quartile in range(4):
        mask = margin_quartile == quartile
        atlas[str(quartile)] = {
            "tokens": int(np.sum(mask)), "route_change": float(np.mean(mismatch[mask])),
            "row_relative_l2_mean": float(np.mean(baseline_rows[mask])),
            "material_error_rate": float(np.mean(baseline_rows[mask] >= 0.05)),
        }
    domain_summary = {}
    for index, domain in enumerate(sorted(set(record["domain"] for record in records))):
        mask = np.asarray([record["domain"] == domain for record in records])
        domain_summary[domain] = validation.summarize(
            teacher_hidden, natural_hidden, candidate_hidden,
            teacher_route, student_route, mask, seed + 10 + index,
        )
    layer3 = replication.propagate_layer_three(
        source, {"TEACHER": teacher_hidden, "NATURAL_STUDENT": natural_hidden,
                 "POST_MOE_HIDDEN_CV_R12_S025": candidate_hidden},
        config, segment_lengths,
    )
    law_supported = (
        float(np.mean(mismatch[margin_quartile == 0])) >
        float(np.mean(mismatch[margin_quartile == 3])) and
        float(np.mean(baseline_rows[mismatch])) > float(np.mean(baseline_rows[~mismatch])) and
        all_summary["relative_l2_improvement"]["ci95_low"] <= 0
    )
    arrays_to_save = {
        "teacher_x_l2": teacher_x, "student_x_l2": student_x,
        "teacher_hidden_l2": teacher_hidden, "natural_hidden_l2": natural_hidden,
        "candidate_hidden_l2": candidate_hidden,
        "teacher_route_indices_l2": teacher_route["indices"],
        "student_route_indices_l2": student_route["indices"],
        "teacher_margin_l2": margin, "low_margin_mask": low_margin.astype(np.uint8),
    }
    capture_path = output_dir / "LR07_LARGE_CAPTURE.npz"
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
        "schema": "hawking.kimi_k26.long_run_large_falsification.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "The pooled mixed-causality law and shrinkage failure survive longer contexts.",
        "config": {"seed": seed, "prompt_count": len(requests),
                   "candidate_count": 1, "refit": False, "retuning": False,
                   "prompt_construction": "16_FIXED_BASES_X_2_PREREGISTERED_SUFFIXES"},
        "candidate": {"name": "POST_MOE_HIDDEN_CV_R12_S025", "payload": payload_info,
                      "complete_physical_bytes": lr04["physical_candidate"][
                          "complete_physical_bytes"],
                      "complete_bpw": lr04["physical_candidate"]["complete_bpw"],
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
        "margin_routing_atlas": atlas, "domain_summary": domain_summary,
        "route_mismatch_damage": {
            "mismatch_tokens": int(np.sum(mismatch)),
            "matched_tokens": int(np.sum(~mismatch)),
            "mismatch_row_relative_l2_mean": float(np.mean(baseline_rows[mismatch])),
            "matched_row_relative_l2_mean": float(np.mean(baseline_rows[~mismatch])),
            "material_errors_with_route_match": int(np.sum(
                (baseline_rows >= 0.05) & ~mismatch,
            )),
        },
        "layer3_amplification": layer3,
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": f1.sha256_file(capture_path)},
        "evidence_parent": {"pooled_law_seal_sha256": pooled_law["seal_sha256"],
                            "lr04_candidate_seal_sha256": lr04["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": ("STRENGTHEN_POOLED_CAUSAL_LAW" if law_supported else
                     "POOLED_LAW_FALSIFIED_REOPEN_REGION"),
        "causal_interpretation": (
            "Longer contexts independently test whether low margin remains a risk marker, route "
            "mismatch remains damaging but incomplete, and quarter-strength post-MoE recovery "
            "remains non-robust without any additional selection."
        ),
        "next_run_rationale": (
            "Repeat the large suite with a new prompt ordering/seed and intervention-stratified subset."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_LARGE_FALSIFICATION.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": pooled_law["seal_sha256"],
        "candidate_hash": payload_info["sha256"],
        "physical_bytes": artifact["candidate"]["complete_physical_bytes"],
        "complete_bpw": artifact["candidate"]["complete_bpw"],
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": len(token_ids), "low_margin_tokens": int(np.sum(low_margin)),
                          "segments": len(requests)},
        "metrics": {"all_tokens": all_summary, "low_margin": low_summary,
                    "margin_atlas": atlas, "route_mismatch_damage":
                    artifact["route_mismatch_damage"], "layer3": layer3},
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
        "next_experiment": "LR08_LARGE_SEED_REPLICATION",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "tokens": len(token_ids), "route_set_change": float(np.mean(mismatch)),
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR07-complete",
        ("[Kimi K2.6 long run] LR07 large falsification complete\n"
         f"tokens / segments: {len(token_ids)} / {len(requests)}\n"
         f"route change: {float(np.mean(mismatch))*100:.2f}%\n"
         f"mean shrinkage rescue: {all_summary['relative_l2_improvement']['mean']:.6f}\n"
         f"CI95 low: {all_summary['relative_l2_improvement']['ci95_low']:.6f}\n"
         f"decision: {artifact['decision']}\n"
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
    parser.add_argument("--seed", type=int, default=26072107)
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
