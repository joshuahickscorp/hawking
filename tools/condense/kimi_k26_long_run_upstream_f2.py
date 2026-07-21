#!/usr/bin/env python3.12
"""Untouched F2 test of the frozen LR10 upstream residual representation."""
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
import kimi_k26_long_run_upstream_auction as upstream  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


EXPERIMENT_ID = "LR11_UPSTREAM_RESIDUAL_HELDOUT_F2"
SUFFIX = (" Provide one counterexample, one intervention, and one quantitative stopping rule. "
          "Do not treat a ranking change as a causal explanation unless the downstream output changes.")
REPLICATION_SUFFIX = (
    " Audit the conclusion under a changed ordering and a new boundary example. Quantify both "
    "immediate rescue and next-layer amplification before deciding whether the candidate survives."
)


def prepare_requests(
    source: Path, *, replication_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = reference.KimiTokenizer(source)
    requests = []
    token_ids = []
    lengths = []
    records = []
    offset = 0
    source_bases = list(reversed(large.BASES)) if replication_run else large.BASES
    suffix = REPLICATION_SUFFIX if replication_run else SUFFIX
    for index, (domain, base) in enumerate(source_bases):
        chat = domain == "tool_format"
        thinking = bool(chat and index % 2)
        text = base + suffix
        rendered = tokenizer.user_prompt(text, thinking=thinking) if chat else text
        ids = tokenizer.encode(rendered)
        request = {"id": f"upstream_f2_{index:02d}", "domain": domain,
                   "text": text, "chat": chat, "thinking": thinking,
                   "rendered": rendered, "token_ids": ids}
        requests.append(request)
        token_ids.extend(ids)
        lengths.append(len(ids))
        for position, token_id in enumerate(ids):
            records.append({"token_index": offset + position, "segment": request["id"],
                            "domain": domain, "position": position, "token_id": int(token_id),
                            "token_text": tokenizer.decode([int(token_id)])})
        offset += len(ids)
    return requests, {"token_ids": token_ids, "segment_lengths": lengths,
                      "token_records": records}


def multi_layer_one(
    source: Path, post: np.ndarray, x: np.ndarray, routes: dict[str, np.ndarray],
    parent_output: np.ndarray, candidate_output: np.ndarray,
) -> dict[str, np.ndarray]:
    config = f1.read_json(source / "config.json")["text_config"]
    shard = reference.TensorShard(reference.shard_path(source, 2))
    x_mx = mx.array(x).astype(mx.bfloat16)
    teacher_feed, _ = reference.routed_moe(x_mx, shard, 1, config)
    native = causal.native_expert(
        x_mx, shard, 1, 0, np.arange(x.shape[0], dtype=np.int32),
    )
    indices = routes["indices"]
    weights = routes["weights"]
    tokens, slots = np.where(indices == 0)

    def hidden(compact: np.ndarray) -> np.ndarray:
        delta = mx.zeros_like(x_mx)
        if tokens.size:
            native_selected = mx.take(native, mx.array(tokens.astype(np.int32)), axis=0)
            compact_selected = mx.array(compact[tokens]).astype(native_selected.dtype)
            route_weight = mx.array(weights[tokens, slots], dtype=native_selected.dtype)[:, None]
            scatter = mx.array(np.eye(x.shape[0], dtype=np.float32)[tokens].T,
                               dtype=native_selected.dtype)
            delta = scatter @ ((compact_selected - native_selected) * route_weight)
        feed = teacher_feed + delta
        value = (mx.array(post).astype(mx.bfloat16) + feed).astype(mx.bfloat16)
        mx.eval(value)
        return np.asarray(value.astype(mx.float32), dtype=np.float32)

    teacher = (mx.array(post).astype(mx.bfloat16) + teacher_feed).astype(mx.bfloat16)
    mx.eval(teacher)
    result = {"TEACHER": np.asarray(teacher.astype(mx.float32), dtype=np.float32),
              "PARENT": hidden(parent_output), "CANDIDATE": hidden(candidate_output),
              "NATIVE_EXPERT0": np.asarray(native.astype(mx.float32), dtype=np.float32)}
    del shard, x_mx, teacher_feed, native, teacher
    gc.collect()
    mx.clear_cache()
    return result


def route_agreement(
    teacher: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
) -> tuple[float, int]:
    exact = np.asarray([set(left) == set(right) for left, right in zip(
        teacher["indices"], candidate["indices"], strict=True,
    )])
    return float(np.mean(exact)), int(np.sum(exact))


def run(
    repo: Path, source: Path, output_dir: Path, seed: int, replication_run: bool,
) -> dict[str, Any]:
    experiment_id = ("LR12_UPSTREAM_RESIDUAL_REPLICATION" if replication_run else
                     EXPERIMENT_ID)
    hypothesis = (
        "The LR11 upstream F2 gain replicates through layer 3 without route loss."
        if replication_run else
        "The microscopic LR10 held-out F1 gain survives untouched F2 propagation."
    )
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"pre-run guard audit failed: {audit['failures']}")
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**prior_status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": experiment_id,
                                "next_experiment": f"{experiment_id}_IN_PROGRESS",
                                "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": experiment_id, "started_at": started_at, "status": "RUNNING",
        "hypothesis": hypothesis,
        "config": {"seed": seed, "prompt_count": len(large.BASES),
                   "candidate_count": 1, "refit": False, "retuning": False},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    lr10 = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_AUCTION.json")
    payload_info = lr10["physical_candidate"]["payload"]
    header, payload_arrays = replication.read_payload(
        Path(payload_info["path"]), payload_info["sha256"],
    )
    model = replication.decode_model(payload_arrays, "upstream_residual")
    shrinkage = float(header["metadata"]["shrinkage"])
    requests, batch = prepare_requests(source, replication_run=replication_run)
    token_ids = batch["token_ids"]
    lengths = batch["segment_lengths"]
    records = batch["token_records"]
    post_l1, x_l1, input_info = causal.capture_layer_one_inputs(source, token_ids, lengths)
    config = f1.read_json(source / "config.json")["text_config"]
    shard_l1 = reference.TensorShard(reference.shard_path(source, 2))
    route_l1 = causal.route_diagnostics(x_l1, shard_l1, 1, config)
    del shard_l1
    parent_payload = (manager.RUNTIME / "f1_representation_bracket/doctor_auction/"
                      "P1_DUAL_PATH_RECOVERY_R16X2.k26f1")
    _, parent_output = causal.compact_expert_output(parent_payload, x_l1)
    feature = (parent_output if lr10["selected"]["architecture"] ==
               "PARENT_OUTPUT_RESIDUAL" else x_l1)
    candidate_output = upstream.apply_output(
        parent_output, bracket.predict(model, feature), shrinkage,
    )
    layer1 = multi_layer_one(
        source, post_l1, x_l1, route_l1, parent_output, candidate_output,
    )
    shard_l2 = reference.TensorShard(reference.shard_path(source, 3))
    pre = {}
    routes = {}
    hidden_l2 = {}
    for name in ("TEACHER", "PARENT", "CANDIDATE"):
        post, expert_input = causal.pre_moe(layer1[name], shard_l2, 2, config, lengths)
        route = causal.route_diagnostics(expert_input, shard_l2, 2, config)
        feed_mx, _ = reference.routed_moe(
            mx.array(expert_input).astype(mx.bfloat16), shard_l2, 2, config,
        )
        feed = np.asarray(feed_mx.astype(mx.float32), dtype=np.float32)
        pre[name] = {"post": post, "x": expert_input}
        routes[name] = route
        hidden_l2[name] = causal.hidden_from_feed(post, feed)
    del shard_l2
    gc.collect()
    mx.clear_cache()
    parent_agreement, parent_matches = route_agreement(routes["TEACHER"], routes["PARENT"])
    candidate_agreement, candidate_matches = route_agreement(
        routes["TEACHER"], routes["CANDIDATE"],
    )
    parent_rows = causal.row_metrics(hidden_l2["TEACHER"], hidden_l2["PARENT"])[
        "relative_l2"]
    candidate_rows = causal.row_metrics(hidden_l2["TEACHER"], hidden_l2["CANDIDATE"])[
        "relative_l2"]
    improvement = bracket.paired_interval(parent_rows - candidate_rows, seed)
    sentinel_tokens = np.any(route_l1["indices"] == 0, axis=1)
    domain_summary = {}
    for index, domain in enumerate(sorted(set(record["domain"] for record in records))):
        mask = np.asarray([record["domain"] == domain for record in records])
        domain_summary[domain] = bracket.paired_interval(
            parent_rows[mask] - candidate_rows[mask], seed + 10 + index,
        )
    layer3 = replication.propagate_layer_three(
        source, {"TEACHER": hidden_l2["TEACHER"], "PARENT": hidden_l2["PARENT"],
                 "CANDIDATE": hidden_l2["CANDIDATE"]}, config, lengths,
    )
    promoted = improvement["ci95_low"] > 0 and candidate_matches >= parent_matches
    if replication_run:
        promoted = bool(
            promoted and
            layer3["CANDIDATE"]["hidden"]["relative_l2"] <=
            layer3["PARENT"]["hidden"]["relative_l2"] and
            layer3["CANDIDATE"]["route_matches"] >= layer3["PARENT"]["route_matches"]
        )
    arrays_to_save = {
        "x_l1": x_l1, "post_l1": post_l1,
        "parent_output_l1": parent_output, "candidate_output_l1": candidate_output,
        "teacher_hidden_l1": layer1["TEACHER"], "parent_hidden_l1": layer1["PARENT"],
        "candidate_hidden_l1": layer1["CANDIDATE"],
        "teacher_hidden_l2": hidden_l2["TEACHER"], "parent_hidden_l2": hidden_l2["PARENT"],
        "candidate_hidden_l2": hidden_l2["CANDIDATE"],
        "teacher_route_indices_l2": routes["TEACHER"]["indices"],
        "parent_route_indices_l2": routes["PARENT"]["indices"],
        "candidate_route_indices_l2": routes["CANDIDATE"]["indices"],
    }
    capture_path = output_dir / (
        "LR12_UPSTREAM_REPLICATION_CAPTURE.npz" if replication_run else
        "LR11_UPSTREAM_F2_CAPTURE.npz"
    )
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
        "schema": "hawking.kimi_k26.long_run_upstream_f2.v1", "status": "PASS",
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "config": {"seed": seed, "prompt_count": len(requests), "candidate_count": 1,
                   "refit": False, "retuning": False,
                   "shared_parent_forward": True, "repeated_parent_forward": False},
        "candidate": {"name": "UPSTREAM_RESIDUAL_CV", "payload": payload_info,
                      "complete_physical_bytes": lr10["physical_candidate"][
                          "complete_physical_bytes"],
                      "complete_bpw": lr10["physical_candidate"]["complete_bpw"],
                      "shrinkage": shrinkage},
        "source": reference.source_identity(source),
        "tokenization": {"tokens": len(token_ids), "segment_lengths": lengths,
                         "token_id_sha256": hashlib.sha256(
                             np.asarray(token_ids, dtype=np.int32).tobytes(),
                         ).hexdigest()},
        "input_capture": input_info,
        "f1_untouched": {
            "all_tokens_parent": f1.quality(layer1["NATIVE_EXPERT0"], parent_output),
            "all_tokens_candidate": f1.quality(layer1["NATIVE_EXPERT0"], candidate_output),
            "routed_tokens": int(np.sum(sentinel_tokens)),
            "routed_parent": f1.quality(
                layer1["NATIVE_EXPERT0"][sentinel_tokens], parent_output[sentinel_tokens]),
            "routed_candidate": f1.quality(
                layer1["NATIVE_EXPERT0"][sentinel_tokens], candidate_output[sentinel_tokens]),
        },
        "f2": {
            "parent": {"hidden": f1.quality(hidden_l2["TEACHER"], hidden_l2["PARENT"]),
                       "row_relative_l2_mean": float(np.mean(parent_rows)),
                       "route_set_agreement": parent_agreement,
                       "route_matches": parent_matches},
            "candidate": {"hidden": f1.quality(
                                hidden_l2["TEACHER"], hidden_l2["CANDIDATE"]),
                          "row_relative_l2_mean": float(np.mean(candidate_rows)),
                          "route_set_agreement": candidate_agreement,
                          "route_matches": candidate_matches},
            "paired_row_relative_l2_improvement": improvement,
            "domain_improvement": domain_summary,
        },
        "layer3_amplification": layer3,
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": f1.sha256_file(capture_path)},
        "evidence_parent": {"lr10_seal_sha256": lr10["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": (("PROMOTE_UPSTREAM_RESIDUAL_REPLICATED" if replication_run else
                      "PROMOTE_UPSTREAM_RESIDUAL_TO_REPLICATION") if promoted else
                     ("RETIRE_UPSTREAM_RESIDUAL_REPLICATION_FAILED" if replication_run else
                      "RETIRE_UPSTREAM_RESIDUAL_F2_NOT_SIGNIFICANT")),
        "causal_interpretation": (
            "The candidate modifies the compact expert output before layer-1 residual propagation; "
            "F2 promotion requires a paired held-out token improvement without route loss."
        ),
        "next_run_rationale": (
            "Replicate the promoted upstream representation on a new split and seed."
            if promoted else
            "The microscopic F1 gain does not buy robust F2; close the tested linear representation region."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / (
        "KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json" if replication_run else
        "KIMI_K26_LONG_RUN_UPSTREAM_F2.json"
    )
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": experiment_id, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": lr10["parent"]["payload_sha256"],
        "candidate_hash": payload_info["sha256"],
        "physical_bytes": artifact["candidate"]["complete_physical_bytes"],
        "complete_bpw": artifact["candidate"]["complete_bpw"],
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": len(token_ids), "segments": len(requests)},
        "metrics": {"f1": artifact["f1_untouched"], "f2": artifact["f2"],
                    "layer3": layer3},
        "confidence_intervals": {"f2_improvement": improvement,
                                 "domain_improvement": domain_summary},
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
        "next_experiment": (("LR13_PROMOTED_BOUNDARY_REPLICATION" if promoted else
                             "LR13_REGION_CLOSURE_AUDIT") if replication_run else
                            ("LR12_UPSTREAM_RESIDUAL_REPLICATION" if promoted else
                             "LR12_REGION_CLOSURE_AUDIT")),
        "latest_result": {"experiment_id": experiment_id, "decision": artifact["decision"],
                          "tokens": len(token_ids), "f2_improvement": improvement,
                          "route_matches_parent": parent_matches,
                          "route_matches_candidate": candidate_matches,
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, ("long-run:LR12-complete" if replication_run else "long-run:LR11-complete"),
        (f"[Kimi K2.6 long run] {experiment_id} complete\n"
         f"tokens / segments: {len(token_ids)} / {len(requests)}\n"
         f"mean paired F2 rescue: {improvement['mean']:.8f}\n"
         f"CI95 low: {improvement['ci95_low']:.8f}\n"
         f"route matches parent/candidate: {parent_matches}/{candidate_matches}\n"
         f"decision: {artifact['decision']}"),
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
    parser.add_argument("--seed", type=int, default=26072111)
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.source.resolve(strict=True),
                       args.output_dir.resolve(), args.seed, args.replication)
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
