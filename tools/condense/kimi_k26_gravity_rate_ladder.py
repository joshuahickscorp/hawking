#!/usr/bin/env python3.12
"""Bounded exact-rate M5 contextual capability-density ladder for Kimi K2.6.

Fits one native teacher-free-at-inference low-rank linear expert student at each
of 0.75, 0.50, and 0.33 complete BPW.  Functional rank is maximized first.
Any remaining bytes are a deterministic zero reserve that is declared, hashed,
serialized, and billed so every physical file lands on its exact rational byte
ceiling.  LR11 supplies grouped fit/CV and LR12 is scored once without refit.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_final_chapter_manager as chapter  # noqa: E402
import kimi_k26_gravity_nonlinear as nonlinear  # noqa: E402
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


SCHEMA = "hawking.kimi_k26.gravity_rate_ladder.v1"
ARTIFACT = "KIMI_K26_GRAVITY_RATE_LADDER.json"
OUTPUT = chapter.RUNTIME / "final_chapter/gravity_rate_ladder"
LOGICAL_WEIGHTS = nonlinear.LOGICAL_WEIGHTS
HIDDEN = 7168
RATES = (
    {"label": "0.75", "numerator": 3, "denominator": 4, "slug": "075"},
    {"label": "0.50", "numerator": 1, "denominator": 2, "slug": "050"},
    {"label": "0.33", "numerator": 33, "denominator": 100, "slug": "033"},
)


class LadderError(RuntimeError):
    """Invalid split, physical payload, or rate-ladder state."""


def ceiling_bytes(rate: dict[str, Any]) -> int:
    return (LOGICAL_WEIGHTS * int(rate["numerator"]) //
            int(rate["denominator"]) // 8)


def experiment_id(rate: dict[str, Any]) -> str:
    return f"M5_NATIVE_LINEAR_RATE_{rate['slug']}"


def payload_metadata(
    rate: dict[str, Any], rank: int, capture_sha256: str,
    *, padding_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}.physical_payload", "oneoff_id": "M5",
        "candidate": experiment_id(rate), "family": "NATIVE_LOW_RANK_LINEAR_STUDENT",
        "revision": f1.REVISION, "layer": 1, "sentinel_expert": nonlinear.SENTINEL,
        "logical_weights_represented": LOGICAL_WEIGHTS,
        "target_complete_bpw": rate["label"],
        "exact_complete_ceiling_bytes": ceiling_bytes(rate),
        "rank": rank, "teacher_access_at_inference": False,
        "contextual_capture_sha256": capture_sha256,
        "inference_contract": (
            "output_mean + output_matrix(input_matrix(activation-input_mean)); "
            "rate_reserve is ignored"
        ),
        "rate_reserve": {
            "bytes": padding_bytes, "encoding": "DETERMINISTIC_ZERO_RESERVE",
            "runtime_effect": "NONE", "selection_order": (
                "functional rank maximized before deterministic reserve is allocated"
            ),
            "all_bytes_declared_hashed_and_billed": True,
        },
    }


def reserve_component(size: int) -> dict[str, Any]:
    if size < 0:
        raise LadderError("negative rate reserve")
    return {"name": "rate_reserve.zero", "role": "reserve",
            "encoding": "opaque_zero_reserve", "shape": [size], "data": bytes(size),
            "runtime_effect": "NONE", "deterministic": True}


def template_components(rank: int) -> list[dict[str, Any]]:
    return [
        nonlinear.component("model.input_mean", np.zeros(HIDDEN), "bfloat16", "base"),
        nonlinear.component("model.output_mean", np.zeros(HIDDEN), "bfloat16", "base"),
        nonlinear.component("model.input.q", np.zeros((rank, HIDDEN)), "int8", "base"),
        nonlinear.component("model.input.scale", np.ones(rank), "float32", "base"),
        nonlinear.component("model.output.q", np.zeros((HIDDEN, rank)), "int8", "base"),
        nonlinear.component("model.output.scale", np.ones(HIDDEN), "float32", "base"),
    ]


def write_exact_payload(
    path: Path, rate: dict[str, Any], rank: int, capture_sha256: str,
    functional_components: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target = ceiling_bytes(rate)
    reserve = 0
    for _ in range(12):
        metadata = payload_metadata(rate, rank, capture_sha256, padding_bytes=reserve)
        payload = f1.write_payload(path, metadata,
                                   functional_components + [reserve_component(reserve)])
        delta = target - int(payload["bytes"])
        if delta == 0:
            return {**payload, "target_bytes": target, "reserve_bytes": reserve,
                    "actual_complete_bpw": payload["bytes"] * 8 / LOGICAL_WEIGHTS,
                    "exact_target_bytes": True, "all_payload_bytes_counted": True}
        reserve += delta
        if reserve < 0:
            return None
    return None


def select_max_rank(
    output_dir: Path, rate: dict[str, Any], capture_sha256: str,
) -> tuple[int, dict[str, Any]]:
    target = ceiling_bytes(rate)
    fixed_estimate = HIDDEN * 8 + 16 * 1024
    upper = min(HIDDEN, max(1, (target - fixed_estimate) // (2 * HIDDEN)) + 8)
    proof = {}
    with tempfile.TemporaryDirectory(prefix="m5-rate-capacity-", dir=output_dir) as raw:
        temporary_root = Path(raw)
        for rank in range(upper, 0, -1):
            candidate_path = temporary_root / f"rank-{rank}.k26f1"
            payload = write_exact_payload(
                candidate_path, rate, rank, capture_sha256, template_components(rank),
            )
            if payload is not None:
                next_rank = rank + 1
                next_path = temporary_root / f"rank-{next_rank}.k26f1"
                next_payload = write_exact_payload(
                    next_path, rate, next_rank, capture_sha256,
                    template_components(next_rank),
                )
                proof = {
                    "selected_rank": rank,
                    "selected_functional_component_bytes": (
                        sum(len(value["data"]) for value in template_components(rank))
                    ),
                    "selected_reserve_bytes": payload["reserve_bytes"],
                    "next_rank": next_rank,
                    "next_rank_fits_exact_ceiling": next_payload is not None,
                    "search_upper_rank": upper,
                    "law": "highest rank admitting an exact self-contained payload",
                }
                if next_payload is not None:
                    raise LadderError("capacity search did not return the maximum rank")
                return rank, proof
    raise LadderError(f"no native linear rank fits rate {rate['label']}")


def train_linear(
    x: np.ndarray, target: np.ndarray, routed: np.ndarray,
    rank: int, steps: int, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch, device = nonlinear.torch_runtime()
    torch.manual_seed(seed)
    input_component, input_mean = nonlinear.physical_mean(
        "model.input_mean", np.mean(x, axis=0), "base",
    )
    output_component, output_mean = nonlinear.physical_mean(
        "model.output_mean", np.mean(target, axis=0), "base",
    )
    x_center = np.asarray(x - input_mean[None, :], dtype=np.float32)
    y_center = np.asarray(target - output_mean[None, :], dtype=np.float32)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(x.shape[1], rank, bias=False)
            self.output = torch.nn.Linear(rank, target.shape[1], bias=False)

        def forward(self, value: Any) -> Any:
            return self.output(self.input(value))

    model = Model().to(device)
    x_tensor = torch.tensor(x_center, device=device)
    y_tensor = torch.tensor(y_center, device=device)
    weight = torch.tensor(1.0 + 3.0 * routed.astype(np.float32), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    batch_size = min(128, x.shape[0])
    trace = []
    for step in range(max(1, steps)):
        indices = rng.choice(x.shape[0], size=batch_size, replace=x.shape[0] < batch_size)
        selected = torch.tensor(indices, dtype=torch.long, device=device)
        prediction = model(x_tensor[selected])
        truth = y_tensor[selected]
        row_weight = weight[selected]
        cosine = torch.nn.functional.cosine_similarity(prediction, truth, dim=-1, eps=1e-8)
        relative_mse = (prediction - truth).square().mean(dim=-1) / (
            truth.square().mean(dim=-1) + 1e-8
        )
        loss = ((1 - cosine) * row_weight).mean() + 0.25 * (relative_mse * row_weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step + 1 == steps or (step + 1) % 25 == 0:
            trace.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    input_value = model.input.weight.detach().cpu().numpy().astype(np.float32)
    output_value = model.output.weight.detach().cpu().numpy().astype(np.float32)
    del model, x_tensor, y_tensor, weight
    if device.type == "mps":
        torch.mps.empty_cache()
    input_components, input_decoded = nonlinear.quantized_matrix_components(
        "model.input", input_value, "base",
    )
    output_components, output_decoded = nonlinear.quantized_matrix_components(
        "model.output", output_value, "base",
    )
    physical = {"rank": rank, "input_mean": input_mean, "output_mean": output_mean,
                "input": input_decoded, "output": output_decoded,
                "steps": steps, "seed": seed, "loss_trace": trace}
    return physical, [input_component, output_component,
                      *input_components, *output_components]


def decode_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    header, components = f1.read_payload(path)
    by_name = {value["name"]: value for value in components}
    model = {
        "rank": int(header["rank"]),
        "input_mean": nonlinear.decode_component(by_name["model.input_mean"]),
        "output_mean": nonlinear.decode_component(by_name["model.output_mean"]),
        "input": nonlinear.decode_component(by_name["model.input.q"]) *
        nonlinear.decode_component(by_name["model.input.scale"])[:, None],
        "output": nonlinear.decode_component(by_name["model.output.q"]) *
        nonlinear.decode_component(by_name["model.output.scale"])[:, None],
    }
    reserve = by_name.get("rate_reserve.zero")
    if reserve is None or any(reserve["data"]):
        raise LadderError("missing or nonzero deterministic rate reserve")
    if len(reserve["data"]) != int(header["rate_reserve"]["bytes"]):
        raise LadderError("rate reserve header/data mismatch")
    return header, model


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    torch, device = nonlinear.torch_runtime()
    input_weight = torch.tensor(model["input"], device=device)
    output_weight = torch.tensor(model["output"], device=device)
    output_mean = torch.tensor(model["output_mean"], device=device)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            value = torch.tensor(
                np.asarray(x[start:start + 256] - model["input_mean"][None, :],
                           dtype=np.float32), device=device,
            )
            outputs.append(((value @ input_weight.T) @ output_weight.T + output_mean).
                           cpu().numpy())
    if device.type == "mps":
        torch.mps.empty_cache()
    return causal.bf16(np.concatenate(outputs, axis=0))


def grouped_cv(
    rate: dict[str, Any], rank: int, fit: dict[str, np.ndarray],
    seam: dict[str, np.ndarray], steps: int, seed: int,
) -> dict[str, Any]:
    groups = seam["fit_group_id"]
    routed = np.any(seam["fit_route_indices"] == nonlinear.SENTINEL, axis=1)
    prediction = np.zeros_like(seam["fit_native_output"])
    folds = []
    for fold in sorted(int(value) for value in np.unique(groups)):
        train = groups != fold
        heldout = groups == fold
        model, _ = train_linear(
            fit["x_l1"][train], seam["fit_native_output"][train], routed[train],
            rank, max(1, steps // 2), seed + fold * 1009,
        )
        prediction[heldout] = predict(model, fit["x_l1"][heldout])
        folds.append({
            "fold": fold, "tokens": int(np.sum(heldout)),
            "routed_tokens": int(np.sum(routed & heldout)),
            "candidate": nonlinear.masked_quality(
                seam["fit_native_output"], prediction, heldout,
            ),
            "paired_vs_parent": nonlinear.paired(
                seam["fit_native_output"], fit["parent_output_l1"], prediction,
                heldout, seed + fold,
            ),
        })
        del model
        gc.collect()
    return {
        "rate": rate["label"], "rank": rank,
        "grouping": "LEAVE_ONE_CONTEXT_SEGMENT_OUT_LR11",
        "folds": folds,
        "all_tokens": f1.quality(seam["fit_native_output"], prediction),
        "routed_tokens": nonlinear.masked_quality(
            seam["fit_native_output"], prediction, routed,
        ),
        "routed_paired_vs_parent": nonlinear.paired(
            seam["fit_native_output"], fit["parent_output_l1"], prediction,
            routed, seed + 90_000,
        ),
        "prediction_sha256": nonlinear.output_sha256(prediction),
    }


def frozen_score(
    model: dict[str, Any], score: dict[str, np.ndarray], seam: dict[str, np.ndarray],
    receipt: dict[str, Any], old: dict[str, np.ndarray], old_parent: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    candidate = predict(model, score["x_l1"])
    routed = np.any(seam["score_route_indices"] == nonlinear.SENTINEL, axis=1)
    state = nonlinear.candidate_hidden(
        score["teacher_hidden_l1"], seam["score_native_output"], candidate,
        seam["score_route_indices"], seam["score_route_weights"],
    )
    margin_threshold = float(np.quantile(seam["score_route_margin"], 0.25))
    low_margin = routed & (seam["score_route_margin"] <= margin_threshold)
    old_candidate = predict(model, old["score_x"])
    parent_quality = nonlinear.masked_quality(
        seam["score_native_output"], score["parent_output_l1"], routed,
    )
    candidate_quality = nonlinear.masked_quality(
        seam["score_native_output"], candidate, routed,
    )
    paired_output = nonlinear.paired(
        seam["score_native_output"], score["parent_output_l1"], candidate, routed, seed,
    )
    domain_names = receipt["inputs"]["score"]["split"]["domain_names"]
    metrics = {
        "frozen_split": "LR12_CHANGED_ORDER_AND_SUFFIX_NO_REFIT",
        "all_tokens": {"parent": f1.quality(
            seam["score_native_output"], score["parent_output_l1"]),
            "candidate": f1.quality(seam["score_native_output"], candidate)},
        "routed_tokens": {
            "parent": parent_quality, "candidate": candidate_quality,
            "paired_relative_l2_improvement": paired_output,
            "relative_l2_rescue_fraction": (
                paired_output["mean"] / (parent_quality.get("relative_l2", 0) + 1e-30)
            ),
            "cosine_gain": (candidate_quality.get("cosine_mean", 0) -
                            parent_quality.get("cosine_mean", 0)),
        },
        "immediate_residual_hidden": {
            "parent": nonlinear.masked_quality(
                score["teacher_hidden_l1"], score["parent_hidden_l1"], routed),
            "candidate": nonlinear.masked_quality(
                score["teacher_hidden_l1"], state, routed),
            "paired_relative_l2_improvement": nonlinear.paired(
                score["teacher_hidden_l1"], score["parent_hidden_l1"], state,
                routed, seed + 1,
            ),
        },
        "low_margin_routed": {
            "threshold": margin_threshold,
            "candidate": nonlinear.masked_quality(
                seam["score_native_output"], candidate, low_margin),
            "paired_relative_l2_improvement": nonlinear.paired(
                seam["score_native_output"], score["parent_output_l1"], candidate,
                low_margin, seed + 2,
            ),
        },
        "domains": nonlinear.domain_intervals(
            seam["score_native_output"], score["parent_output_l1"], candidate, routed,
            seam["score_domain_id"], domain_names, seed + 100,
        ),
        "isolated_regression": {
            "parent": f1.quality(old["score_teacher_output"], old_parent),
            "candidate": f1.quality(old["score_teacher_output"], old_candidate),
        },
        "candidate_output_sha256": nonlinear.output_sha256(candidate),
        "candidate_hidden_sha256": nonlinear.output_sha256(state),
    }
    return metrics, candidate


def classify(metrics: dict[str, Any], cv: dict[str, Any]) -> str:
    routed = metrics["routed_tokens"]
    candidate = routed["candidate"]
    parent = routed["parent"]
    hidden = metrics["immediate_residual_hidden"]["paired_relative_l2_improvement"]
    if (candidate["cosine_mean"] >= parent["cosine_mean"] - 0.02 and
            candidate["relative_l2"] <= parent["relative_l2"] * 1.05 and
            hidden["ci95_high"] >= 0 and cv["routed_paired_vs_parent"]["mean"] >= 0):
        return "SURVIVES_CONTEXTUAL_F1"
    if candidate["cosine_mean"] >= 0.20 and candidate["relative_l2"] <= 1.25:
        return "DEGRADED_BUT_NOT_IRRECOVERABLE_F1"
    return "IRRECOVERABLE_CONTEXTUAL_F1"


def valid_cached_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = f1.read_json(path)
    payload = value.get("physical_payload", {})
    payload_path = Path(payload.get("path", ""))
    if (value.get("status") == "PASS" and nonlinear.verify_seal(value) and
            payload_path.is_file() and f1.sha256_file(payload_path) == payload.get("sha256") and
            payload_path.stat().st_size == payload.get("target_bytes")):
        return value
    return None


def run_rate(
    repo: Path, rate: dict[str, Any], output_dir: Path, receipt: dict[str, Any],
    seam: dict[str, np.ndarray], fit: dict[str, np.ndarray], score: dict[str, np.ndarray],
    old: dict[str, np.ndarray], old_parent: np.ndarray, steps: int, seed: int,
) -> dict[str, Any]:
    row_id = experiment_id(rate)
    result_path = output_dir / f"{row_id}_RESULT.json"
    cached = valid_cached_row(result_path)
    if cached is not None:
        chapter.append_ledger({"event": "CACHE_REUSE", "at": f1.now(),
                               "experiment_id": row_id,
                               "evidence_seal_sha256": cached["seal_sha256"]})
        return cached
    hypothesis = (
        f"A native linear student at exactly {rate['label']} BPW reveals whether the "
        "contextual state trajectory remains physically plausible."
    )
    before, started = nonlinear.begin_lane(repo, row_id, hypothesis)
    rank, capacity_proof = select_max_rank(output_dir, rate, receipt["capture"]["sha256"])
    cv = grouped_cv(rate, rank, fit, seam, steps, seed)
    routed_fit = np.any(seam["fit_route_indices"] == nonlinear.SENTINEL, axis=1)
    model, functional_components = train_linear(
        fit["x_l1"], seam["fit_native_output"], routed_fit, rank, steps, seed + 500_000,
    )
    payload_path = output_dir / f"{row_id}.k26f1"
    payload = write_exact_payload(
        payload_path, rate, rank, receipt["capture"]["sha256"], functional_components,
    )
    if payload is None or payload["bytes"] != ceiling_bytes(rate):
        raise LadderError(f"failed to land {row_id} on its exact byte ceiling")
    header, decoded = decode_payload(payload_path)
    repeat_a = predict(decoded, score["x_l1"][:32])
    _, decoded_repeat = decode_payload(payload_path)
    repeat_b = predict(decoded_repeat, score["x_l1"][:32])
    if not np.array_equal(repeat_a, repeat_b):
        raise LadderError(f"{row_id} deterministic decode/execute failure")
    metrics, candidate = frozen_score(
        decoded, score, seam, receipt, old, old_parent, seed + 800_000,
    )
    verdict = classify(metrics, cv)
    functional_bytes = sum(len(value["data"]) for value in functional_components)
    result = f1.seal({
        "schema": f"{SCHEMA}.rate_result", "status": "PASS",
        "oneoff_id": "M5", "experiment_id": row_id, "sealed_at": f1.now(),
        "hypothesis": hypothesis, "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "split_law": {"fit_and_grouped_cv": "LR11_ONLY",
                      "frozen_score": "LR12_ONCE_NO_REFIT", "fit_score_overlap": 0},
        "physical_payload": payload,
        "physical_rate": {
            "target_complete_bpw": rate["label"],
            "target_bytes": ceiling_bytes(rate), "actual_bytes": payload["bytes"],
            "actual_complete_bpw": payload["actual_complete_bpw"],
            "functional_component_bytes": functional_bytes,
            "container_and_descriptor_bytes": (
                payload["bytes"] - functional_bytes - payload["reserve_bytes"]
            ),
            "deterministic_billed_reserve_bytes": payload["reserve_bytes"],
            "exact_ceiling_hit": payload["bytes"] == ceiling_bytes(rate),
            "all_headers_scales_matrices_and_reserve_billed": True,
        },
        "capacity_proof": capacity_proof,
        "training": {"rank": rank, "steps": steps, "seed": seed,
                     "loss_trace": model["loss_trace"]},
        "f0": {"deterministic_decode": True,
               "decode_probe_sha256": nonlinear.output_sha256(repeat_a),
               "teacher_access_at_inference": False,
               "payload_header_rate_reserve": header["rate_reserve"]},
        "grouped_cv": cv, "frozen_score_metrics": metrics,
        "verdict": verdict,
        "capability_density": {
            "contextual_routed_cosine_per_bpw": (
                metrics["routed_tokens"]["candidate"]["cosine_mean"] /
                payload["actual_complete_bpw"]
            ),
            "one_minus_relative_l2_per_bpw": (
                (1 - metrics["routed_tokens"]["candidate"]["relative_l2"]) /
                payload["actual_complete_bpw"]
            ),
        },
        "candidate_output_sha256": nonlinear.output_sha256(candidate),
        "claim_boundary": (
            "M5 CONTEXTUAL LAYER1 EXPERT0 F0/F1 RATE STRESS; not deployable full Kimi"
        ),
    })
    f1.atomic_json(result_path, result)
    nonlinear.finish_lane(
        repo, row_id, hypothesis, verdict,
        {"rank": rank, "rate": result["physical_rate"],
         "routed": metrics["routed_tokens"], "verdict": verdict},
        result["seal_sha256"], before, started,
        next_experiment="CONTINUE_M5_RATE_STRESS_LADDER",
        physical_bytes=payload["bytes"], complete_bpw=payload["actual_complete_bpw"],
        notify=(f"[Kimi Gravity M5] exact {rate['label']} BPW rung sealed\n"
                f"rank/reserve: {rank}/{payload['reserve_bytes']} bytes\n"
                f"verdict: {verdict}"),
    )
    return result


def valid_cached_artifact(repo: Path) -> dict[str, Any] | None:
    path = repo / ARTIFACT
    if not path.exists():
        return None
    value = f1.read_json(path)
    if (value.get("status") != "PASS" or not nonlinear.verify_seal(value) or
            value.get("decision_law_version") != "STRICT_FINAL_F1_V2"):
        return None
    for row in value.get("stress_ladder", []):
        payload = row["physical_payload"]
        path_value = Path(payload["path"])
        if (not path_value.is_file() or f1.sha256_file(path_value) != payload["sha256"] or
                path_value.stat().st_size != payload["target_bytes"]):
            return None
    return value


def run_ladder(
    repo: Path, source: Path, seam_dir: Path, output_dir: Path,
    steps: int, seed: int,
) -> dict[str, Any]:
    cached = valid_cached_artifact(repo)
    if cached is not None:
        chapter.append_ledger({"event": "CACHE_REUSE", "at": f1.now(),
                               "experiment_id": "M5_RATE_STRESS_LADDER",
                               "evidence_seal_sha256": cached["seal_sha256"]})
        return cached
    nonlinear.guard(repo)
    receipt, seam = nonlinear.load_seam(seam_dir)
    lr11_artifact = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_F2.json")
    lr12_artifact = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json")
    fit = nonlinear.load_original_capture(
        nonlinear.LR11_CAPTURE, lr11_artifact["capture"]["sha256"],
    )
    score = nonlinear.load_original_capture(
        nonlinear.LR12_CAPTURE, lr12_artifact["capture"]["sha256"],
    )
    old_receipt = f1.read_json(nonlinear.OLD_CAPTURE_RECEIPT)
    old = nonlinear.load_original_capture(
        nonlinear.OLD_CAPTURE, old_receipt["capture_sha256"],
    )
    _, old_parent = causal.compact_expert_output(nonlinear.PARENT_PAYLOAD, old["score_x"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, rate in enumerate(RATES):
        results.append(run_rate(
            repo, rate, output_dir, receipt, seam, fit, score, old, old_parent,
            steps, seed + index * 100_000,
        ))
    strict_rows = []
    for result in results:
        admitted, reasons = nonlinear.admission(
            result["frozen_score_metrics"], result["grouped_cv"],
            result["physical_payload"], True,
        )
        strict_rows.append({
            "admitted": admitted,
            "verdict": ("SURVIVES_STRICT_CONTEXTUAL_F1" if admitted else
                        "RETIRED_STRICT_CONTEXTUAL_F1"),
            "reasons": reasons,
        })
    verdicts = [value["verdict"] for value in strict_rows]
    decision = (
        "RATE_SURVIVOR_REQUIRES_NEW_CONTEXT_FALSIFICATION"
        if any(value["admitted"] for value in strict_rows) else
        "RATE_CURVE_ESTABLISHED_STRICT_F1_RETIRES_ALL"
    )
    artifact = f1.seal({
        "schema": f"{SCHEMA}.artifact", "status": "PASS", "oneoff_id": "M5",
        "experiment": "M5_AGGRESSIVE_RATE_STRESS_LADDER", "sealed_at": f1.now(),
        "decision_law_version": "STRICT_FINAL_F1_V2",
        "source": reference.source_identity(source),
        "contextual_capture_seal_sha256": receipt["seal_sha256"],
        "question": (
            "What is the contextual capability-density curve at 0.75, 0.50, and 0.33 BPW?"
        ),
        "stress_ladder": [{
            "rate": result["physical_rate"]["target_complete_bpw"],
            "rank": result["training"]["rank"],
            "physical_payload": result["physical_payload"],
            "physical_rate": result["physical_rate"],
            "grouped_cv": result["grouped_cv"],
            "frozen_score_metrics": result["frozen_score_metrics"],
            "capability_density": result["capability_density"],
            "screen_verdict": result["verdict"],
            "strict_final_verdict": strict["verdict"],
            "strict_admission_reasons": strict["reasons"],
            "seal_sha256": result["seal_sha256"],
        } for result, strict in zip(results, strict_rows, strict=True)],
        "decision": decision,
        "next_experiment": (
            "M5_RATE_SURVIVOR_NEW_CONTEXT_FALSIFICATION" if
            decision == "RATE_SURVIVOR_REQUIRES_NEW_CONTEXT_FALSIFICATION" else
            "M6_DOCTOR_REMOVAL_NATIVE_COMPACT_INVERSION"
        ),
        "provenance": {"parent_forwards_repeated": 0, "teacher_capture_repeated": 0,
                       "LR12_refits": 0, "exact_payloads": 3},
        "claim_boundary": (
            "BOUNDED M5 EXPERT0 F0/F1 CURVE; does not prove full-model capability"
        ),
    })
    chapter.mirror_json(ARTIFACT, artifact)
    chapter.append_ledger({
        "event": "ONEOFF_COMPLETE", "experiment_id": "M5_RATE_STRESS_LADDER",
        "at": f1.now(), "decision": decision,
        "rates": [value["physical_rate"] for value in results],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior = nonlinear.read_status(repo)
    chapter.write_status({
        **prior, "status": "MANAGING", "phase": "MAD_SCIENTIST_ONEOFFS",
        "active_heavy_lane": None, "experiments_completed": nonlinear.completed_count(repo),
        "next_experiment": artifact["next_experiment"],
        "latest_result": {"experiment_id": artifact["experiment"],
                          "decision": decision,
                          "verdicts": dict(zip((rate["label"] for rate in RATES), verdicts,
                                               strict=True)),
                          "evidence_seal_sha256": artifact["seal_sha256"]},
    })
    chapter.send_telegram(
        "gravity-final:M5-complete",
        ("[Kimi Gravity] M5 exact-rate ladder complete\n"
         f"0.75/0.50/0.33 verdicts: {' / '.join(verdicts)}\n"
         f"decision: {decision}\nnext: {artifact['next_experiment']}"),
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ladder",))
    parser.add_argument("--repo", type=Path, default=chapter.REPO)
    parser.add_argument("--source", type=Path, default=chapter.legacy.SNAPSHOT)
    parser.add_argument("--seam-dir", type=Path, default=nonlinear.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=26072165)
    args = parser.parse_args()
    try:
        result = run_ladder(
            args.repo.resolve(strict=True), args.source.resolve(strict=True),
            args.seam_dir.resolve(strict=True), args.output_dir.resolve(),
            args.steps, args.seed,
        )
        print(json.dumps({"status": result["status"], "experiment": result["experiment"],
                          "decision": result["decision"],
                          "seal_sha256": result["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "stage": args.stage,
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
