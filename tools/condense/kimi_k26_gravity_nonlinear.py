#!/usr/bin/env python3.12
"""Contextual F0/F1 nonlinear representation tournament for Kimi K2.6.

Stage ``capture`` reuses the sealed LR11/LR12 contextual inputs, recomputes only
the native expert-0 function and the layer-1 routes, and stores a sealed seam.
Stage ``tournament`` fits N1/N2/N4/N6 and mandatory ablations on grouped LR11
folds, freezes each row, scores it once on LR12, and writes exact self-contained
physical payloads.  Neither stage repeats a parent forward.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
import kimi_k26_final_chapter_manager as chapter  # noqa: E402
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_long_run_repair_bracket as bracket  # noqa: E402
import kimi_k26_long_run_upstream_f2 as upstream  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


SCHEMA = "hawking.kimi_k26.gravity_nonlinear.v1"
CAPTURE_ID = "NG0_CONTEXTUAL_EXPERT0_SEAM"
CAPTURE_JSON = "KIMI_K26_GRAVITY_CONTEXTUAL_SEAM.json"
CAPTURE_NPZ = "KIMI_K26_GRAVITY_CONTEXTUAL_SEAM.npz"
TOURNAMENT_JSON = "KIMI_K26_GRAVITY_NONLINEAR_TOURNAMENT.json"
PARENT_NAME = "P1_DUAL_PATH_RECOVERY_R16X2"
LOGICAL_WEIGHTS = 44_040_192
COMPLETE_CEILING_BPW = 0.98
COMPLETE_CEILING_BYTES = 5_394_923
SENTINEL = 0
DEFAULT_OUTPUT = chapter.RUNTIME / "final_chapter/gravity_nonlinear"
PARENT_PAYLOAD = (
    chapter.RUNTIME / "f1_representation_bracket/doctor_auction/"
    "P1_DUAL_PATH_RECOVERY_R16X2.k26f1"
)
PARENT_RESULT = (
    chapter.RUNTIME / "f1_representation_bracket/doctor_auction/"
    "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json"
)
OLD_CAPTURE = chapter.RUNTIME / "f1_representation_bracket/teacher_capture.npz"
OLD_CAPTURE_RECEIPT = chapter.RUNTIME / "f1_representation_bracket/teacher_capture.json"
LR11_CAPTURE = chapter.RUNTIME / "long_run/LR11/LR11_UPSTREAM_F2_CAPTURE.npz"
LR12_CAPTURE = chapter.RUNTIME / "long_run/LR12/LR12_UPSTREAM_REPLICATION_CAPTURE.npz"


class GravityError(RuntimeError):
    """Scientifically invalid or physically inadmissible tournament state."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def output_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value, dtype=np.float32).tobytes())


def verify_seal(value: dict[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return value.get("seal_sha256") == f1.seal(unsigned)["seal_sha256"]


def read_status(repo: Path) -> dict[str, Any]:
    path = repo / chapter.STATUS_JSON
    return f1.read_json(path) if path.exists() else {}


def ledger_has_complete(repo: Path, experiment_id: str) -> bool:
    path = repo / chapter.LEDGER
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (record.get("event") == "EXPERIMENT_COMPLETE" and
                record.get("experiment_id") == experiment_id):
            return True
    return False


def completed_count(repo: Path) -> int:
    path = repo / chapter.LEDGER
    if not path.exists():
        return 0
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "EXPERIMENT_COMPLETE":
            completed.add(str(record.get("experiment_id")))
    return len(completed)


def resource_snapshot() -> dict[str, Any]:
    return chapter.legacy.resource_snapshot()


def guard(repo: Path) -> dict[str, Any]:
    value = chapter.audit(repo)
    if value["status"] != "PASS":
        raise GravityError(f"final-chapter guard failed: {value['failures']}")
    return value


def begin_lane(repo: Path, experiment_id: str, hypothesis: str) -> tuple[dict[str, Any], float]:
    audit = guard(repo)
    before = resource_snapshot()
    if not ledger_has_complete(repo, experiment_id):
        chapter.append_ledger({
            "event": "EXPERIMENT_START", "experiment_id": experiment_id,
            "started_at": f1.now(), "hypothesis": hypothesis,
            "audit_seal_sha256": audit["seal_sha256"], "resources": before,
        })
    chapter.append_parallel({
        "event": "LANE_START", "at": f1.now(), "lane": "HEAVY_LANE",
        "pid": os.getpid(), "task": experiment_id, "resources": before,
        "contention_effect": "MEASURE_AFTER_COMPLETION",
    })
    prior = read_status(repo)
    chapter.write_status({
        **prior, "status": "RUNNING_EXPERIMENT", "phase": "NONLINEAR_F0_F1_TOURNAMENT",
        "active_heavy_lane": {"pid": os.getpid(), "task": experiment_id},
        "active_light_lanes": [], "next_experiment": f"{experiment_id}_IN_PROGRESS",
        "controller": audit["controller"], "resources": before,
        "one_copy": audit["source"]["one_copy"],
        "mop_protected": audit["mop"]["matches_baseline"],
    })
    return before, time.time()


def finish_lane(
    repo: Path, experiment_id: str, hypothesis: str, decision: str,
    metrics: dict[str, Any], evidence_seal: str, before: dict[str, Any], started: float,
    *, next_experiment: str, physical_bytes: int | None = None,
    complete_bpw: float | None = None, notify: str | None = None,
) -> None:
    after = resource_snapshot()
    duration = time.time() - started
    if not ledger_has_complete(repo, experiment_id):
        chapter.append_ledger({
            "event": "EXPERIMENT_COMPLETE", "experiment_id": experiment_id,
            "hypothesis": hypothesis, "started_at_epoch": started,
            "ended_at": f1.now(), "duration_seconds": duration,
            "physical_bytes": physical_bytes, "complete_bpw": complete_bpw,
            "metrics": metrics, "decision": decision,
            "evidence_seal_sha256": evidence_seal,
            "resources": {"before": before, "after": after},
            "faults": [], "retries": 0, "next_run_rationale": next_experiment,
        })
    chapter.append_parallel({
        "event": "LANE_COMPLETE", "at": f1.now(), "lane": "HEAVY_LANE",
        "pid": os.getpid(), "task": experiment_id, "runtime_seconds": duration,
        "resources": {"before": before, "after": after},
        "contention_effect": "NO_SECOND_HEAVY_LANE_LAUNCHED",
    })
    prior = read_status(repo)
    chapter.write_status({
        **prior, "status": "MANAGING", "phase": "NONLINEAR_F0_F1_TOURNAMENT",
        "active_heavy_lane": None, "active_light_lanes": [],
        "experiments_completed": completed_count(repo), "next_experiment": next_experiment,
        "latest_result": {"experiment_id": experiment_id, "decision": decision,
                          "metrics": metrics, "evidence_seal_sha256": evidence_seal},
        "resources": after,
    })
    if notify:
        chapter.send_telegram(f"gravity-final:{experiment_id}", notify)


def component(name: str, value: np.ndarray, encoding: str, role: str) -> dict[str, Any]:
    dtype_map = {
        "int8": np.int8, "float16": np.float16, "float32": np.float32,
        "bfloat16": ml_dtypes.bfloat16, "int32": np.int32,
    }
    if encoding not in dtype_map:
        raise GravityError(f"unsupported physical encoding: {encoding}")
    stored = np.ascontiguousarray(value, dtype=dtype_map[encoding])
    return {"name": name, "role": role, "encoding": encoding,
            "shape": list(stored.shape), "data": stored.tobytes(order="C")}


def decode_component(value: dict[str, Any]) -> np.ndarray:
    dtype_map = {
        "int8": np.int8, "float16": np.float16, "float32": np.float32,
        "bfloat16": ml_dtypes.bfloat16, "int32": np.int32,
    }
    return np.frombuffer(value["data"], dtype=dtype_map[value["encoding"]]).reshape(
        tuple(int(item) for item in value["shape"])
    ).astype(np.float32)


def quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(value, dtype=np.float32)
    scale = np.max(np.abs(source), axis=1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(source / scale[:, None]), -127, 127).astype(np.int8)
    decoded = quantized.astype(np.float32) * scale[:, None]
    return quantized, scale, decoded


def quantized_matrix_components(prefix: str, value: np.ndarray, role: str) -> tuple[
    list[dict[str, Any]], np.ndarray
]:
    quantized, scale, decoded = quantize_rows(value)
    return [component(f"{prefix}.q", quantized, "int8", role),
            component(f"{prefix}.scale", scale, "float32", role)], decoded


def physical_mean(prefix: str, value: np.ndarray, role: str) -> tuple[
    dict[str, Any], np.ndarray
]:
    stored = np.asarray(value, dtype=ml_dtypes.bfloat16)
    return component(prefix, stored, "bfloat16", role), stored.astype(np.float32)


def randomized_basis(value: np.ndarray, rank: int, seed: int) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    actual = min(rank, source.shape[0] - 1, source.shape[1])
    if actual < 1:
        raise GravityError("insufficient rows for randomized basis")
    width = min(actual + 8, source.shape[0], source.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, 1 / math.sqrt(source.shape[1]),
                       size=(source.shape[1], width)).astype(np.float32)
    projected = source @ omega
    q_value, _ = np.linalg.qr(projected, mode="reduced")
    small = q_value.T @ source
    _, _, right = np.linalg.svd(small, full_matrices=False)
    return np.asarray(right[:actual], dtype=np.float32)


def kmeans(value: np.ndarray, count: int, seed: int, iterations: int = 24) -> tuple[
    np.ndarray, np.ndarray
]:
    source = np.asarray(value, dtype=np.float32)
    count = min(count, source.shape[0])
    rng = np.random.default_rng(seed)
    centers = [source[int(rng.integers(0, source.shape[0]))]]
    for _ in range(1, count):
        distance = np.min(np.stack([
            np.sum((source - center[None, :]) ** 2, axis=1) for center in centers
        ]), axis=0)
        centers.append(source[int(np.argmax(distance))])
    center_array = np.stack(centers).astype(np.float32)
    labels = np.zeros(source.shape[0], dtype=np.int32)
    for _ in range(iterations):
        distance = np.sum((source[:, None, :] - center_array[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distance, axis=1).astype(np.int32)
        nearest = np.min(distance, axis=1)
        updated = center_array.copy()
        for index in range(count):
            selected = source[labels == index]
            updated[index] = (np.mean(selected, axis=0) if selected.size else
                              source[int(np.argmax(nearest))])
        if np.array_equal(updated, center_array):
            break
        center_array = updated
    return center_array, labels


def load_original_capture(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if f1.sha256_file(path) != expected_hash:
        raise GravityError(f"sealed capture hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def request_metadata(source: Path, replication: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    requests, batch = upstream.prepare_requests(source, replication_run=replication)
    records = batch["token_records"]
    segments = {request["id"]: index for index, request in enumerate(requests)}
    domains = {name: index for index, name in enumerate(sorted(
        {str(record["domain"]) for record in records}
    ))}
    arrays = {
        "group_id": np.asarray([segments[str(record["segment"])] for record in records],
                               dtype=np.int32),
        "domain_id": np.asarray([domains[str(record["domain"])] for record in records],
                                dtype=np.int32),
        "token_ids": np.asarray(batch["token_ids"], dtype=np.int32),
    }
    metadata = {"requests": [{key: value for key, value in request.items()
                               if key not in {"rendered", "token_ids"}}
                              for request in requests],
                "domain_names": {str(value): key for key, value in domains.items()},
                "segment_lengths": batch["segment_lengths"]}
    return arrays, metadata


def valid_cached_capture(output_dir: Path) -> dict[str, Any] | None:
    receipt_path = output_dir / CAPTURE_JSON
    payload_path = output_dir / CAPTURE_NPZ
    if not receipt_path.exists() or not payload_path.exists():
        return None
    value = f1.read_json(receipt_path)
    if (value.get("status") == "PASS" and verify_seal(value) and
            value.get("capture", {}).get("sha256") == f1.sha256_file(payload_path)):
        return value
    return None


def capture_stage(repo: Path, source: Path, output_dir: Path) -> dict[str, Any]:
    cached = valid_cached_capture(output_dir)
    if cached is not None:
        chapter.append_ledger({"event": "CACHE_REUSE", "at": f1.now(),
                               "experiment_id": CAPTURE_ID,
                               "evidence_seal_sha256": cached["seal_sha256"]})
        return cached
    hypothesis = (
        "Contextual native expert outputs reveal representation error hidden by the isolated F1 seam."
    )
    before, started = begin_lane(repo, CAPTURE_ID, hypothesis)
    output_dir.mkdir(parents=True, exist_ok=True)
    lr11_artifact = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_F2.json")
    lr12_artifact = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json")
    fit = load_original_capture(LR11_CAPTURE, lr11_artifact["capture"]["sha256"])
    score = load_original_capture(LR12_CAPTURE, lr12_artifact["capture"]["sha256"])
    fit_meta_arrays, fit_meta = request_metadata(source, False)
    score_meta_arrays, score_meta = request_metadata(source, True)
    for label, arrays, metadata_arrays, artifact in (
        ("fit", fit, fit_meta_arrays, lr11_artifact),
        ("score", score, score_meta_arrays, lr12_artifact),
    ):
        if arrays["x_l1"].shape[0] != metadata_arrays["token_ids"].size:
            raise GravityError(f"{label} token count does not match cached contextual states")
        digest = sha256_bytes(metadata_arrays["token_ids"].tobytes())
        if digest != artifact["tokenization"]["token_id_sha256"]:
            raise GravityError(f"{label} contextual token hash mismatch")
    parent_result = f1.read_json(PARENT_RESULT)
    if f1.sha256_file(PARENT_PAYLOAD) != parent_result["payload"]["sha256"]:
        raise GravityError("parent physical payload hash mismatch")
    base_weights, base_components = f1.decode_base_weights(PARENT_PAYLOAD)
    fit_base, _ = f1.expert_forward(fit["x_l1"], base_weights)
    score_base, _ = f1.expert_forward(score["x_l1"], base_weights)
    config = f1.read_json(source / "config.json")["text_config"]
    shard = reference.TensorShard(reference.shard_path(source, 2))
    fit_route = causal.route_diagnostics(fit["x_l1"], shard, 1, config)
    score_route = causal.route_diagnostics(score["x_l1"], shard, 1, config)
    fit_native_mx = causal.native_expert(
        mx.array(fit["x_l1"]).astype(mx.bfloat16), shard, 1, SENTINEL,
        np.arange(fit["x_l1"].shape[0], dtype=np.int32),
    )
    score_native_mx = causal.native_expert(
        mx.array(score["x_l1"]).astype(mx.bfloat16), shard, 1, SENTINEL,
        np.arange(score["x_l1"].shape[0], dtype=np.int32),
    )
    fit_native = np.asarray(fit_native_mx.astype(mx.float32), dtype=np.float32)
    score_native = np.asarray(score_native_mx.astype(mx.float32), dtype=np.float32)
    del shard, fit_native_mx, score_native_mx, base_weights
    gc.collect()
    mx.clear_cache()
    arrays_to_save = {
        "fit_native_output": fit_native, "score_native_output": score_native,
        "fit_base_output": fit_base, "score_base_output": score_base,
        "fit_route_indices": fit_route["indices"], "score_route_indices": score_route["indices"],
        "fit_route_weights": fit_route["weights"], "score_route_weights": score_route["weights"],
        "fit_route_margin": fit_route["margin_8v9"],
        "score_route_margin": score_route["margin_8v9"],
        "fit_group_id": fit_meta_arrays["group_id"],
        "score_group_id": score_meta_arrays["group_id"],
        "fit_domain_id": fit_meta_arrays["domain_id"],
        "score_domain_id": score_meta_arrays["domain_id"],
    }
    estimated = sum(value.nbytes for value in arrays_to_save.values()) + 1024 * 1024
    free = shutil.disk_usage(Path.home()).free
    if free - estimated <= chapter.FLOOR_BYTES:
        raise GravityError("contextual seam atomic write would cross the 5-GiB floor")
    capture_path = output_dir / CAPTURE_NPZ
    temporary = capture_path.with_name(f".{capture_path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays_to_save)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, capture_path)
    contextual_parent = {
        "fit": f1.quality(fit_native, fit["parent_output_l1"]),
        "score": f1.quality(score_native, score["parent_output_l1"]),
    }
    fit_routed = np.any(fit_route["indices"] == SENTINEL, axis=1)
    score_routed = np.any(score_route["indices"] == SENTINEL, axis=1)
    receipt = f1.seal({
        "schema": f"{SCHEMA}.contextual_seam", "status": "PASS",
        "experiment_id": CAPTURE_ID, "sealed_at": f1.now(),
        "source": reference.source_identity(source),
        "reuse": {"parent_forwards": 0, "teacher_captures": 0,
                  "native_expert_evaluations": 2, "router_evaluations": 2,
                  "serialized_base_payload_reused": True},
        "parent": {"name": PARENT_NAME, "payload_sha256": parent_result["payload"]["sha256"],
                   "physical_bytes": parent_result["payload"]["bytes"],
                   "complete_bpw": parent_result["physical_budget"]["actual_complete_bpw"],
                   "base_component_bytes": f1.role_bytes(base_components, "base")},
        "inputs": {
            "fit": {"path": str(LR11_CAPTURE), "sha256": lr11_artifact["capture"]["sha256"],
                    "tokens": int(fit["x_l1"].shape[0]), "split": fit_meta},
            "score": {"path": str(LR12_CAPTURE), "sha256": lr12_artifact["capture"]["sha256"],
                      "tokens": int(score["x_l1"].shape[0]), "split": score_meta},
        },
        "routing": {"fit_sentinel_tokens": int(np.sum(fit_routed)),
                    "score_sentinel_tokens": int(np.sum(score_routed))},
        "contextual_parent_metrics": contextual_parent,
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": f1.sha256_file(capture_path)},
        "claim_boundary": (
            "CONTEXTUAL_LAYER1_EXPERT0_FUNCTION_AND_ROUTES; no repeated parent forward"
        ),
    })
    f1.atomic_json(output_dir / CAPTURE_JSON, receipt)
    chapter.mirror_json(CAPTURE_JSON, receipt)
    finish_lane(
        repo, CAPTURE_ID, hypothesis, "CONTEXTUAL_SEAM_CACHED", receipt["routing"],
        receipt["seal_sha256"], before, started,
        next_experiment="N1_N2_N4_N6_CONTEXTUAL_F0_F1_TOURNAMENT",
        notify=("[Kimi Gravity] contextual nonlinear seam cached\n"
                f"fit/score tokens: {fit_native.shape[0]}/{score_native.shape[0]}\n"
                f"sentinel routed: {int(np.sum(fit_routed))}/{int(np.sum(score_routed))}\n"
                "next: exact-byte N1/N2/N4/N6 tournament"),
    )
    return receipt


def load_seam(output_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    receipt = valid_cached_capture(output_dir)
    if receipt is None:
        raise GravityError("valid contextual seam absent; run capture first")
    with np.load(output_dir / CAPTURE_NPZ, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    return receipt, arrays


def torch_runtime() -> tuple[Any, Any]:
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch, device


def train_torch_model(
    kind: str, x: np.ndarray, base: np.ndarray, target: np.ndarray,
    routed: np.ndarray, steps: int, seed: int,
) -> dict[str, Any]:
    torch, device = torch_runtime()
    torch.manual_seed(seed)
    x_mean = np.asarray(np.mean(x, axis=0), dtype=ml_dtypes.bfloat16).astype(np.float32)
    residual_target = target - base if kind.startswith("N1") else target
    output_mean = np.asarray(np.mean(residual_target, axis=0),
                             dtype=ml_dtypes.bfloat16).astype(np.float32)
    x_center = np.asarray(x - x_mean[None, :], dtype=np.float32)
    y_center = np.asarray(residual_target - output_mean[None, :], dtype=np.float32)
    if kind == "N1_GATED":
        input_rank, hidden, output_rank = 48, 96, 80

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(x.shape[1], input_rank, bias=False)
                self.activation = torch.nn.Linear(input_rank, hidden, bias=False)
                self.gate = torch.nn.Linear(input_rank, hidden, bias=False)
                self.coefficient = torch.nn.Linear(hidden, output_rank, bias=False)
                self.output = torch.nn.Linear(output_rank, target.shape[1], bias=False)

            def forward(self, value: Any) -> Any:
                z_value = self.input(value)
                hidden_value = torch.nn.functional.silu(self.activation(z_value))
                hidden_value = hidden_value * torch.sigmoid(self.gate(z_value))
                return self.output(self.coefficient(hidden_value))

        model = Model().to(device)
        with torch.no_grad():
            model.input.weight.copy_(torch.tensor(
                randomized_basis(x_center, input_rank, seed + 1), device=device))
            model.output.weight.copy_(torch.tensor(
                randomized_basis(y_center, output_rank, seed + 2).T, device=device))
        geometry = {"input_rank": input_rank, "hidden": hidden, "output_rank": output_rank}
    elif kind in {"N1_AFFINE", "N6_LINEAR"}:
        rank = 68 if kind == "N1_AFFINE" else 366

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(x.shape[1], rank, bias=False)
                self.output = torch.nn.Linear(rank, target.shape[1], bias=False)

            def forward(self, value: Any) -> Any:
                return self.output(self.input(value))

        model = Model().to(device)
        geometry = {"rank": rank}
    elif kind == "N6_SWIGLU":
        width = 244

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = torch.nn.Linear(x.shape[1], width, bias=False)
                self.up = torch.nn.Linear(x.shape[1], width, bias=False)
                self.down = torch.nn.Linear(width, target.shape[1], bias=False)

            def forward(self, value: Any) -> Any:
                return self.down(torch.nn.functional.silu(self.gate(value)) * self.up(value))

        model = Model().to(device)
        geometry = {"width": width}
    else:
        raise GravityError(f"unknown torch model kind: {kind}")
    x_tensor = torch.tensor(x_center, device=device)
    target_tensor = torch.tensor(y_center, device=device)
    weight = torch.tensor(1.0 + 3.0 * routed.astype(np.float32), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    batch_size = min(128, x.shape[0])
    loss_trace = []
    model.train()
    for step in range(max(1, steps)):
        indices = rng.choice(x.shape[0], size=batch_size, replace=x.shape[0] < batch_size)
        selected = torch.tensor(indices, device=device, dtype=torch.long)
        prediction = model(x_tensor[selected])
        truth = target_tensor[selected]
        row_weight = weight[selected]
        cosine = torch.nn.functional.cosine_similarity(prediction, truth, dim=-1, eps=1e-8)
        mse_rows = (prediction - truth).square().mean(dim=-1) / (
            truth.square().mean(dim=-1) + 1e-8
        )
        loss = ((1 - cosine) * row_weight).mean() + 0.25 * (mse_rows * row_weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step + 1 == steps or (step + 1) % 25 == 0:
            loss_trace.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    state = {name: value.detach().cpu().numpy().astype(np.float32)
             for name, value in model.state_dict().items()}
    del model, x_tensor, target_tensor, weight
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"kind": kind, "x_mean": x_mean, "output_mean": output_mean,
            "state": state, "geometry": geometry, "loss_trace": loss_trace}


def physicalize_torch(model: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    role = "doctor" if model["kind"].startswith("N1") else "base"
    components = []
    mean_component, x_mean = physical_mean("model.input_mean", model["x_mean"], role)
    components.append(mean_component)
    output_component, output_mean = physical_mean("model.output_mean", model["output_mean"], role)
    components.append(output_component)
    state = {}
    for name, value in model["state"].items():
        prefix = f"model.{name.removesuffix('.weight')}"
        packed, decoded = quantized_matrix_components(prefix, value, role)
        components.extend(packed)
        state[name] = decoded
    return {"kind": model["kind"], "x_mean": x_mean, "output_mean": output_mean,
            "state": state, "geometry": model["geometry"],
            "loss_trace": model["loss_trace"]}, components


def torch_predict(model: dict[str, Any], x: np.ndarray, base: np.ndarray) -> np.ndarray:
    torch, device = torch_runtime()
    x_center = torch.tensor(np.asarray(x - model["x_mean"][None, :], dtype=np.float32),
                            device=device)
    state = {name: torch.tensor(value, device=device) for name, value in model["state"].items()}
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            value = x_center[start:start + 256]
            kind = model["kind"]
            if kind == "N1_GATED":
                z_value = value @ state["input.weight"].T
                hidden = torch.nn.functional.silu(z_value @ state["activation.weight"].T)
                hidden = hidden * torch.sigmoid(z_value @ state["gate.weight"].T)
                prediction = (hidden @ state["coefficient.weight"].T) @ state["output.weight"].T
            elif kind in {"N1_AFFINE", "N6_LINEAR"}:
                prediction = (value @ state["input.weight"].T) @ state["output.weight"].T
            elif kind == "N6_SWIGLU":
                prediction = torch.nn.functional.silu(value @ state["gate.weight"].T)
                prediction = (prediction * (value @ state["up.weight"].T)) @ \
                    state["down.weight"].T
            else:
                raise GravityError(f"unknown decoded torch model kind: {kind}")
            prediction = prediction + torch.tensor(model["output_mean"], device=device)
            if kind.startswith("N1"):
                prediction = prediction + torch.tensor(base[start:start + 256], device=device)
            outputs.append(prediction.cpu().numpy())
    if device.type == "mps":
        torch.mps.empty_cache()
    return causal.bf16(np.concatenate(outputs, axis=0))


def fit_n2(
    x: np.ndarray, base: np.ndarray, target: np.ndarray, count: int, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    role = "doctor"
    residual = np.asarray(target - base, dtype=np.float32)
    mean_component, residual_mean = physical_mean(
        "model.residual_mean", np.mean(residual, axis=0), role,
    )
    centered = residual - residual_mean[None, :]
    if count == 1:
        return {"kind": "N2_K1", "residual_mean": residual_mean,
                "geometry": {"codebooks": 1}}, [mean_component]
    x_component, x_mean = physical_mean("model.input_mean", np.mean(x, axis=0), role)
    input_basis = randomized_basis(x - x_mean[None, :], 32, seed + 1)
    input_components, input_basis = quantized_matrix_components(
        "model.input_basis", input_basis, role,
    )
    output_basis = randomized_basis(centered, 80, seed + 2)
    output_components, output_basis = quantized_matrix_components(
        "model.output_basis", output_basis, role,
    )
    coefficient = centered @ output_basis.T
    prototypes, labels = kmeans(coefficient, count, seed + 3)
    z_value = (x - x_mean[None, :]) @ input_basis.T
    activation_centers = np.stack([
        (np.mean(z_value[labels == index], axis=0) if np.any(labels == index)
         else np.mean(z_value, axis=0))
        for index in range(count)
    ]).astype(np.float16)
    prototypes = np.asarray(prototypes, dtype=np.float16)
    components = [mean_component, x_component, *input_components, *output_components,
                  component("model.activation_centers", activation_centers, "float16", role),
                  component("model.output_codebooks", prototypes, "float16", role)]
    model = {"kind": "N2_K8", "residual_mean": residual_mean, "x_mean": x_mean,
             "input_basis": input_basis, "output_basis": output_basis,
             "activation_centers": activation_centers.astype(np.float32),
             "output_codebooks": prototypes.astype(np.float32),
             "geometry": {"codebooks": count, "input_rank": 32, "output_rank": 80}}
    return model, components


def predict_n2(model: dict[str, Any], x: np.ndarray, base: np.ndarray) -> np.ndarray:
    if model["kind"] == "N2_K1":
        return causal.bf16(base + model["residual_mean"][None, :])
    z_value = (x - model["x_mean"][None, :]) @ model["input_basis"].T
    distance = np.sum((z_value[:, None, :] - model["activation_centers"][None, :, :]) ** 2,
                      axis=2)
    labels = np.argmin(distance, axis=1)
    coefficient = model["output_codebooks"][labels]
    delta = model["residual_mean"][None, :] + coefficient @ model["output_basis"]
    return causal.bf16(base + delta)


def fit_n4(
    x: np.ndarray, base: np.ndarray, target: np.ndarray, spline: bool, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    role = "doctor"
    residual = np.asarray(target - base, dtype=np.float32)
    mean_component, residual_mean = physical_mean(
        "model.residual_mean", np.mean(residual, axis=0), role,
    )
    centered = residual - residual_mean[None, :]
    x_component, x_mean = physical_mean("model.input_mean", np.mean(x, axis=0), role)
    input_basis = randomized_basis(x - x_mean[None, :], 16, seed + 1)
    input_components, input_basis = quantized_matrix_components(
        "model.input_basis", input_basis, role,
    )
    output_basis = randomized_basis(centered, 80, seed + 2)
    output_components, output_basis = quantized_matrix_components(
        "model.output_basis", output_basis, role,
    )
    z_value = (x - x_mean[None, :]) @ input_basis.T
    coefficient = centered @ output_basis.T
    components = [mean_component, x_component, *input_components, *output_components]
    if not spline:
        design = np.concatenate((z_value, np.ones((z_value.shape[0], 1), dtype=np.float32)),
                                axis=1)
        ridge = float(np.trace(design.T @ design) / design.shape[1] * 1e-3)
        affine = np.linalg.solve(design.T @ design + np.eye(design.shape[1]) * ridge,
                                 design.T @ coefficient).astype(np.float16)
        components.append(component("model.affine", affine, "float16", role))
        return {"kind": "N4_AFFINE", "residual_mean": residual_mean, "x_mean": x_mean,
                "input_basis": input_basis, "output_basis": output_basis,
                "affine": affine.astype(np.float32),
                "geometry": {"input_rank": 16, "output_rank": 80}}, components
    knots = 16
    breakpoints = np.zeros((16, knots), dtype=np.float32)
    values = np.zeros((16, knots, 80), dtype=np.float32)
    remaining = coefficient.copy()
    for direction in range(16):
        coordinate = z_value[:, direction]
        point = np.quantile(coordinate, np.linspace(0, 1, knots)).astype(np.float32)
        point = np.maximum.accumulate(point + np.arange(knots, dtype=np.float32) * 1e-7)
        assignment = np.argmin(np.abs(coordinate[:, None] - point[None, :]), axis=1)
        table = np.stack([
            np.mean(remaining[assignment == index], axis=0)
            if np.any(assignment == index) else np.zeros(80, dtype=np.float32)
            for index in range(knots)
        ])
        predicted = interpolate_table(coordinate, point, table)
        remaining -= predicted
        breakpoints[direction] = point
        values[direction] = table
    breakpoints_stored = breakpoints.astype(np.float16)
    values_stored = values.astype(np.float16)
    components.extend([
        component("model.breakpoints", breakpoints_stored, "float16", role),
        component("model.knot_values", values_stored, "float16", role),
    ])
    return {"kind": "N4_SPLINE", "residual_mean": residual_mean, "x_mean": x_mean,
            "input_basis": input_basis, "output_basis": output_basis,
            "breakpoints": breakpoints_stored.astype(np.float32),
            "knot_values": values_stored.astype(np.float32),
            "geometry": {"input_rank": 16, "knots": knots, "output_rank": 80}}, components


def interpolate_table(coordinate: np.ndarray, points: np.ndarray,
                      values: np.ndarray) -> np.ndarray:
    right = np.searchsorted(points, coordinate, side="right")
    right = np.clip(right, 1, len(points) - 1)
    left = right - 1
    denominator = np.maximum(points[right] - points[left], 1e-8)
    fraction = ((coordinate - points[left]) / denominator).astype(np.float32)
    return values[left] * (1 - fraction[:, None]) + values[right] * fraction[:, None]


def predict_n4(model: dict[str, Any], x: np.ndarray, base: np.ndarray) -> np.ndarray:
    z_value = (x - model["x_mean"][None, :]) @ model["input_basis"].T
    if model["kind"] == "N4_AFFINE":
        design = np.concatenate((z_value, np.ones((z_value.shape[0], 1), dtype=np.float32)),
                                axis=1)
        coefficient = design @ model["affine"]
    else:
        coefficient = np.zeros((x.shape[0], model["output_basis"].shape[0]), dtype=np.float32)
        for direction in range(z_value.shape[1]):
            coefficient += interpolate_table(
                z_value[:, direction], model["breakpoints"][direction],
                model["knot_values"][direction],
            )
    delta = model["residual_mean"][None, :] + coefficient @ model["output_basis"]
    return causal.bf16(base + delta)


def fit_row(
    row: dict[str, Any], x: np.ndarray, base: np.ndarray, target: np.ndarray,
    routed: np.ndarray, steps: int, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kind = row["kind"]
    if kind in {"N1_GATED", "N1_AFFINE", "N6_SWIGLU", "N6_LINEAR"}:
        return physicalize_torch(train_torch_model(kind, x, base, target, routed, steps, seed))
    if kind == "N2_K8":
        return fit_n2(x, base, target, 8, seed)
    if kind == "N2_K1":
        return fit_n2(x, base, target, 1, seed)
    if kind == "N4_SPLINE":
        return fit_n4(x, base, target, True, seed)
    if kind == "N4_AFFINE":
        return fit_n4(x, base, target, False, seed)
    raise GravityError(f"unknown row kind: {kind}")


def predict_row(model: dict[str, Any], x: np.ndarray, base: np.ndarray) -> np.ndarray:
    if model["kind"].startswith("N1") or model["kind"].startswith("N6"):
        return torch_predict(model, x, base)
    if model["kind"].startswith("N2"):
        return predict_n2(model, x, base)
    if model["kind"].startswith("N4"):
        return predict_n4(model, x, base)
    raise GravityError(f"unknown decoded row kind: {model['kind']}")


def model_components_from_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    header, components = f1.read_payload(path)
    values = {value["name"]: value for value in components}
    kind = str(header["kind"])
    if kind.startswith("N1") or kind.startswith("N6"):
        state = {}
        for name in header["state_names"]:
            stem = f"model.{name.removesuffix('.weight')}"
            state[name] = decode_component(values[f"{stem}.q"]) * \
                decode_component(values[f"{stem}.scale"])[:, None]
        model = {"kind": kind, "x_mean": decode_component(values["model.input_mean"]),
                 "output_mean": decode_component(values["model.output_mean"]),
                 "state": state, "geometry": header["geometry"], "loss_trace": []}
    elif kind == "N2_K1":
        model = {"kind": kind,
                 "residual_mean": decode_component(values["model.residual_mean"]),
                 "geometry": header["geometry"]}
    elif kind == "N2_K8":
        model = {"kind": kind,
                 "residual_mean": decode_component(values["model.residual_mean"]),
                 "x_mean": decode_component(values["model.input_mean"]),
                 "input_basis": decode_component(values["model.input_basis.q"]) *
                 decode_component(values["model.input_basis.scale"])[:, None],
                 "output_basis": decode_component(values["model.output_basis.q"]) *
                 decode_component(values["model.output_basis.scale"])[:, None],
                 "activation_centers": decode_component(values["model.activation_centers"]),
                 "output_codebooks": decode_component(values["model.output_codebooks"]),
                 "geometry": header["geometry"]}
    else:
        model = {"kind": kind,
                 "residual_mean": decode_component(values["model.residual_mean"]),
                 "x_mean": decode_component(values["model.input_mean"]),
                 "input_basis": decode_component(values["model.input_basis.q"]) *
                 decode_component(values["model.input_basis.scale"])[:, None],
                 "output_basis": decode_component(values["model.output_basis.q"]) *
                 decode_component(values["model.output_basis.scale"])[:, None],
                 "geometry": header["geometry"]}
        if kind == "N4_AFFINE":
            model["affine"] = decode_component(values["model.affine"])
        elif kind == "N4_SPLINE":
            model["breakpoints"] = decode_component(values["model.breakpoints"])
            model["knot_values"] = decode_component(values["model.knot_values"])
        else:
            raise GravityError(f"unknown physical payload kind: {kind}")
    return header, model


def write_candidate_payload(
    path: Path, row: dict[str, Any], model: dict[str, Any],
    module_components: list[dict[str, Any]], base_components: list[dict[str, Any]],
    parent_hash: str, capture_hash: str,
) -> dict[str, Any]:
    self_contained_base = not row["kind"].startswith("N6")
    components = (base_components + module_components if self_contained_base
                  else module_components)
    metadata = {
        "schema": f"{SCHEMA}.physical_payload", "candidate": row["name"],
        "family": row["family"], "kind": row["kind"], "ablation_of": row["ablation_of"],
        "revision": f1.REVISION, "layer": 1, "sentinel_expert": SENTINEL,
        "logical_weights_represented": LOGICAL_WEIGHTS,
        "complete_ceiling_bytes": COMPLETE_CEILING_BYTES,
        "base_payload_sha256": parent_hash if self_contained_base else None,
        "base_representation_reused": self_contained_base,
        "teacher_access_at_inference": False, "contextual_capture_sha256": capture_hash,
        "geometry": model["geometry"],
        "state_names": sorted(model.get("state", {})),
        "inference_contract": (
            "decode installed components; execute candidate function using activation only"
        ),
    }
    payload = f1.write_payload(path, metadata, components)
    complete_bpw = payload["bytes"] * 8 / LOGICAL_WEIGHTS
    if payload["bytes"] > COMPLETE_CEILING_BYTES or complete_bpw > COMPLETE_CEILING_BPW:
        raise GravityError(
            f"{row['name']} exceeds physical ceiling: {payload['bytes']} bytes / {complete_bpw} BPW"
        )
    return {**payload, "complete_bpw": complete_bpw,
            "logical_weights_represented": LOGICAL_WEIGHTS,
            "complete_ceiling_bytes": COMPLETE_CEILING_BYTES,
            "all_payload_bytes_counted": True}


def row_relative_l2(reference_value: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return causal.row_metrics(reference_value, candidate)["relative_l2"]


def masked_quality(reference_value: np.ndarray, candidate: np.ndarray,
                   mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"tokens": 0, "status": "NO_SELECTED_TOKENS"}
    return {"tokens": int(np.sum(mask)), **f1.quality(reference_value[mask], candidate[mask])}


def paired(reference_value: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
           mask: np.ndarray, seed: int) -> dict[str, Any]:
    if not np.any(mask):
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    improvement = (row_relative_l2(reference_value[mask], baseline[mask]) -
                   row_relative_l2(reference_value[mask], candidate[mask]))
    return bracket.paired_interval(improvement, seed)


def candidate_hidden(
    teacher_hidden: np.ndarray, native_output: np.ndarray, candidate_output: np.ndarray,
    indices: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    delta = np.zeros_like(teacher_hidden, dtype=np.float32)
    tokens, slots = np.where(indices == SENTINEL)
    if tokens.size:
        delta[tokens] += ((candidate_output[tokens] - native_output[tokens]) *
                          weights[tokens, slots, None])
    return causal.bf16(teacher_hidden + delta)


def cv_metrics(
    row: dict[str, Any], fit: dict[str, np.ndarray], seam: dict[str, np.ndarray],
    steps: int, seed: int,
) -> dict[str, Any]:
    groups = seam["fit_group_id"]
    routed = np.any(seam["fit_route_indices"] == SENTINEL, axis=1)
    prediction = np.zeros_like(seam["fit_native_output"])
    fold_records = []
    for fold in sorted(int(value) for value in np.unique(groups)):
        train = groups != fold
        heldout = groups == fold
        model, _ = fit_row(row, fit["x_l1"][train], seam["fit_base_output"][train],
                           seam["fit_native_output"][train], routed[train], steps,
                           seed + fold * 1009)
        prediction[heldout] = predict_row(
            model, fit["x_l1"][heldout], seam["fit_base_output"][heldout],
        )
        fold_records.append({
            "fold": fold, "tokens": int(np.sum(heldout)),
            "routed_tokens": int(np.sum(routed & heldout)),
            "candidate": masked_quality(seam["fit_native_output"], prediction, heldout),
            "paired_vs_parent": paired(
                seam["fit_native_output"], fit["parent_output_l1"], prediction,
                heldout, seed + fold,
            ),
        })
        del model
        gc.collect()
    routed_improvement = paired(
        seam["fit_native_output"], fit["parent_output_l1"], prediction, routed, seed + 90_000,
    )
    return {"grouping": "LEAVE_ONE_CONTEXT_SEGMENT_OUT_LR11", "folds": fold_records,
            "all_tokens": f1.quality(seam["fit_native_output"], prediction),
            "routed_tokens": masked_quality(seam["fit_native_output"], prediction, routed),
            "routed_paired_vs_parent": routed_improvement,
            "prediction_sha256": output_sha256(prediction)}


def domain_intervals(
    reference_value: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    routed: np.ndarray, domain_ids: np.ndarray, domain_names: dict[str, str], seed: int,
) -> dict[str, Any]:
    result = {}
    for domain_id in sorted(int(value) for value in np.unique(domain_ids)):
        mask = routed & (domain_ids == domain_id)
        result[domain_names[str(domain_id)]] = paired(
            reference_value, baseline, candidate, mask, seed + domain_id,
        )
    return result


def score_metrics(
    row: dict[str, Any], model: dict[str, Any], fit: dict[str, np.ndarray],
    score: dict[str, np.ndarray], seam: dict[str, np.ndarray], receipt: dict[str, Any],
    old: dict[str, np.ndarray], old_parent: np.ndarray, seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    candidate = predict_row(model, score["x_l1"], seam["score_base_output"])
    routed = np.any(seam["score_route_indices"] == SENTINEL, axis=1)
    candidate_state = candidate_hidden(
        score["teacher_hidden_l1"], seam["score_native_output"], candidate,
        seam["score_route_indices"], seam["score_route_weights"],
    )
    low_threshold = float(np.quantile(seam["score_route_margin"], 0.25))
    low_margin = routed & (seam["score_route_margin"] <= low_threshold)
    old_candidate = predict_row(model, old["score_x"], old["score_base_output"])
    output_improvement = paired(
        seam["score_native_output"], score["parent_output_l1"], candidate, routed, seed,
    )
    hidden_improvement = paired(
        score["teacher_hidden_l1"], score["parent_hidden_l1"], candidate_state,
        routed, seed + 1,
    )
    parent_quality = masked_quality(
        seam["score_native_output"], score["parent_output_l1"], routed,
    )
    candidate_quality = masked_quality(seam["score_native_output"], candidate, routed)
    rescue_fraction = (
        output_improvement["mean"] / (parent_quality.get("relative_l2", 0) + 1e-30)
    )
    domain_names = receipt["inputs"]["score"]["split"]["domain_names"]
    return {
        "frozen_split": "LR12_CHANGED_ORDER_AND_SUFFIX_NO_REFIT",
        "all_tokens": {"parent": f1.quality(seam["score_native_output"],
                                                     score["parent_output_l1"]),
                       "candidate": f1.quality(seam["score_native_output"], candidate)},
        "routed_tokens": {"parent": parent_quality, "candidate": candidate_quality,
                          "paired_relative_l2_improvement": output_improvement,
                          "relative_l2_rescue_fraction": rescue_fraction,
                          "cosine_gain": (candidate_quality.get("cosine_mean", 0) -
                                          parent_quality.get("cosine_mean", 0))},
        "immediate_residual_hidden": {
            "parent": masked_quality(score["teacher_hidden_l1"],
                                     score["parent_hidden_l1"], routed),
            "candidate": masked_quality(score["teacher_hidden_l1"], candidate_state, routed),
            "paired_relative_l2_improvement": hidden_improvement,
        },
        "low_margin_routed": {
            "threshold": low_threshold, "parent": masked_quality(
                seam["score_native_output"], score["parent_output_l1"], low_margin),
            "candidate": masked_quality(seam["score_native_output"], candidate, low_margin),
            "paired_relative_l2_improvement": paired(
                seam["score_native_output"], score["parent_output_l1"], candidate,
                low_margin, seed + 2,
            ),
        },
        "domains": domain_intervals(
            seam["score_native_output"], score["parent_output_l1"], candidate, routed,
            seam["score_domain_id"], domain_names, seed + 100,
        ),
        "isolated_regression": {
            "tokens": int(old["score_x"].shape[0]),
            "parent": f1.quality(old["score_teacher_output"], old_parent),
            "candidate": f1.quality(old["score_teacher_output"], old_candidate),
        },
        "candidate_output_sha256": output_sha256(candidate),
        "candidate_hidden_sha256": output_sha256(candidate_state),
    }, candidate


def admission(metrics: dict[str, Any], cv: dict[str, Any], payload: dict[str, Any],
              beats_ablation: bool) -> tuple[bool, list[str]]:
    reasons = []
    routed = metrics["routed_tokens"]
    improvement = routed["paired_relative_l2_improvement"]
    hidden = metrics["immediate_residual_hidden"]["paired_relative_l2_improvement"]
    isolated = metrics["isolated_regression"]
    low = metrics["low_margin_routed"]["paired_relative_l2_improvement"]
    if payload["bytes"] > COMPLETE_CEILING_BYTES:
        reasons.append("PHYSICAL_CEILING_EXCEEDED")
    if improvement["ci95_low"] <= 0:
        reasons.append("ROUTED_OUTPUT_CI_NOT_POSITIVE")
    if not (routed["relative_l2_rescue_fraction"] >= 0.05 or routed["cosine_gain"] >= 0.02):
        reasons.append("ROUTED_OUTPUT_GAIN_NOT_PRACTICAL")
    if hidden["ci95_low"] <= 0:
        reasons.append("IMMEDIATE_HIDDEN_CI_NOT_POSITIVE")
    if cv["routed_paired_vs_parent"]["mean"] <= 0:
        reasons.append("GROUPED_CV_NOT_POSITIVE")
    if isolated["candidate"]["cosine_mean"] < isolated["parent"]["cosine_mean"] - 0.005:
        reasons.append("ISOLATED_COSINE_REGRESSION")
    if isolated["candidate"]["cosine_p10"] < isolated["parent"]["cosine_p10"] - 0.005:
        reasons.append("ISOLATED_TAIL_REGRESSION")
    if low["n"] and low["ci95_high"] < 0:
        reasons.append("LOW_MARGIN_SIGNIFICANT_HARM")
    if any(value["n"] and value["ci95_high"] < 0 for value in metrics["domains"].values()):
        reasons.append("DOMAIN_SIGNIFICANT_HARM")
    if not beats_ablation:
        reasons.append("DOES_NOT_BEAT_MANDATORY_ABLATION")
    return not reasons, reasons


ROWS = (
    {"name": "N1_GATED_RESIDUAL_R48_H96_O80", "family": "N1", "kind": "N1_GATED",
     "ablation_of": None},
    {"name": "N1_AFFINE_R68_ABLATION", "family": "N1", "kind": "N1_AFFINE",
     "ablation_of": "N1_GATED_RESIDUAL_R48_H96_O80"},
    {"name": "N2_FUNCTIONAL_CODEBOOK_K8_I32_O80", "family": "N2", "kind": "N2_K8",
     "ablation_of": None},
    {"name": "N2_CONSTANT_CODEBOOK_K1_ABLATION", "family": "N2", "kind": "N2_K1",
     "ablation_of": "N2_FUNCTIONAL_CODEBOOK_K8_I32_O80"},
    {"name": "N4_QUANTILE_SPLINE_I16_K16_O80", "family": "N4", "kind": "N4_SPLINE",
     "ablation_of": None},
    {"name": "N4_AFFINE_I16_O80_ABLATION", "family": "N4", "kind": "N4_AFFINE",
     "ablation_of": "N4_QUANTILE_SPLINE_I16_K16_O80"},
    {"name": "N6_NATIVE_SWIGLU_W244_INT8", "family": "N6", "kind": "N6_SWIGLU",
     "ablation_of": None},
    {"name": "N6_LINEAR_NATIVE_R366_ABLATION", "family": "N6", "kind": "N6_LINEAR",
     "ablation_of": "N6_NATIVE_SWIGLU_W244_INT8"},
)


def valid_cached_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = f1.read_json(path)
    payload = result.get("physical_payload", {})
    payload_path = Path(payload.get("path", ""))
    if (result.get("status") == "PASS" and verify_seal(result) and payload_path.is_file() and
            payload.get("sha256") == f1.sha256_file(payload_path) and
            payload.get("bytes") == payload_path.stat().st_size):
        return result
    return None


def row_steps(row: dict[str, Any], n1_steps: int, n6_steps: int, cv: bool) -> int:
    base = n6_steps if row["kind"].startswith("N6") else n1_steps
    return max(1, base // 2) if cv else base


def run_row(
    repo: Path, row: dict[str, Any], output_dir: Path, receipt: dict[str, Any],
    seam: dict[str, np.ndarray], fit: dict[str, np.ndarray], score: dict[str, np.ndarray],
    old: dict[str, np.ndarray], old_parent: np.ndarray, base_components: list[dict[str, Any]],
    parent_hash: str, seed: int, n1_steps: int, n6_steps: int,
) -> dict[str, Any]:
    result_path = output_dir / f"{row['name']}_RESULT.json"
    cached = valid_cached_row(result_path)
    if cached is not None:
        chapter.append_ledger({"event": "CACHE_REUSE", "at": f1.now(),
                               "experiment_id": row["name"],
                               "evidence_seal_sha256": cached["seal_sha256"]})
        return cached
    hypothesis = {
        "N1": "A gated residual generator captures contextual correction unavailable to a linear map.",
        "N2": "Activation-selected functional codebooks beat a constant output prototype.",
        "N4": "Piecewise activation maps beat an affine map of the same functional seam.",
        "N6": "A native compact SwiGLU student beats a byte-comparable linear native student.",
    }[row["family"]]
    before, started = begin_lane(repo, row["name"], hypothesis)
    routed_fit = np.any(seam["fit_route_indices"] == SENTINEL, axis=1)
    cv = cv_metrics(row, fit, seam, row_steps(row, n1_steps, n6_steps, True), seed)
    model, module_components = fit_row(
        row, fit["x_l1"], seam["fit_base_output"], seam["fit_native_output"],
        routed_fit, row_steps(row, n1_steps, n6_steps, False), seed + 500_000,
    )
    payload_path = output_dir / f"{row['name']}.k26f1"
    payload = write_candidate_payload(
        payload_path, row, model, module_components, base_components,
        parent_hash, receipt["capture"]["sha256"],
    )
    _, decoded = model_components_from_payload(payload_path)
    repeat_a = predict_row(decoded, score["x_l1"][:32], seam["score_base_output"][:32])
    _, decoded_repeat = model_components_from_payload(payload_path)
    repeat_b = predict_row(decoded_repeat, score["x_l1"][:32], seam["score_base_output"][:32])
    if not np.array_equal(repeat_a, repeat_b):
        raise GravityError(f"{row['name']} deterministic decode/execute check failed")
    metrics, candidate = score_metrics(
        row, decoded, fit, score, seam, receipt, old, old_parent, seed + 800_000,
    )
    result = f1.seal({
        "schema": f"{SCHEMA}.candidate_result", "status": "PASS",
        "sealed_at": f1.now(), "experiment_id": row["name"],
        "family": row["family"], "kind": row["kind"],
        "ablation_of": row["ablation_of"], "hypothesis": hypothesis,
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "split_law": {"fit_and_grouped_cv": "LR11_ONLY",
                      "frozen_score": "LR12_ONCE_NO_REFIT",
                      "fit_score_overlap": 0},
        "reuse": {"parent_forwards": 0, "teacher_captures": 0,
                  "contextual_capture_sha256": receipt["capture"]["sha256"]},
        "physical_payload": payload,
        "f0": {"deterministic_decode": True,
               "decode_probe_sha256": output_sha256(repeat_a),
               "teacher_access_at_inference": False,
               "within_0_98_bpw": payload["bytes"] <= COMPLETE_CEILING_BYTES,
               "module_component_bytes": sum(len(value["data"]) for value in module_components),
               "component_count": payload["component_count"]},
        "training": {"seed": seed, "steps": row_steps(row, n1_steps, n6_steps, False),
                     "geometry": model["geometry"],
                     "loss_trace": model.get("loss_trace", [])},
        "grouped_cv": cv, "frozen_score_metrics": metrics,
        "candidate_output_sha256": output_sha256(candidate),
        "claim_boundary": (
            "F0/F1 LAYER1 EXPERT0 CONTEXTUAL FUNCTION; not all-expert or end-to-end capability"
        ),
    })
    f1.atomic_json(result_path, result)
    finish_lane(
        repo, row["name"], hypothesis, "F0_F1_ROW_SEALED",
        {"complete_bpw": payload["complete_bpw"],
         "routed": metrics["routed_tokens"], "ablation_of": row["ablation_of"]},
        result["seal_sha256"], before, started,
        next_experiment="CONTINUE_N1_N2_N4_N6_TOURNAMENT",
        physical_bytes=payload["bytes"], complete_bpw=payload["complete_bpw"],
        notify=(f"[Kimi Gravity] {row['name']} sealed\n"
                f"BPW: {payload['complete_bpw']:.6f}\n"
                f"routed rescue: {metrics['routed_tokens']['relative_l2_rescue_fraction']:.4f}\n"
                "decision: compare with mandatory ablation"),
    )
    return result


def compare_ablation(candidate: dict[str, Any], ablation: dict[str, Any], seed: int) -> dict[str, Any]:
    candidate_metric = candidate["frozen_score_metrics"]["routed_tokens"]
    ablation_metric = ablation["frozen_score_metrics"]["routed_tokens"]
    return {
        "candidate_relative_l2": candidate_metric["candidate"]["relative_l2"],
        "ablation_relative_l2": ablation_metric["candidate"]["relative_l2"],
        "candidate_beats_ablation": (
            candidate_metric["candidate"]["relative_l2"] <
            ablation_metric["candidate"]["relative_l2"]
        ),
        "seed": seed,
    }


def valid_cached_tournament(repo: Path) -> dict[str, Any] | None:
    path = repo / TOURNAMENT_JSON
    if not path.exists():
        return None
    value = f1.read_json(path)
    if value.get("status") != "PASS" or not verify_seal(value):
        return None
    for row in value.get("rows", []):
        payload = row.get("physical_payload", {})
        payload_path = Path(payload.get("path", ""))
        if not payload_path.is_file() or f1.sha256_file(payload_path) != payload.get("sha256"):
            return None
    return value


def tournament_stage(
    repo: Path, source: Path, output_dir: Path, seed: int,
    n1_steps: int, n6_steps: int,
) -> dict[str, Any]:
    cached = valid_cached_tournament(repo)
    if cached is not None:
        chapter.append_ledger({"event": "CACHE_REUSE", "at": f1.now(),
                               "experiment_id": "NG_N1_N2_N4_N6_TOURNAMENT",
                               "evidence_seal_sha256": cached["seal_sha256"]})
        return cached
    receipt, seam = load_seam(output_dir)
    lr11_artifact = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_F2.json")
    lr12_artifact = f1.read_json(repo / "KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json")
    fit = load_original_capture(LR11_CAPTURE, lr11_artifact["capture"]["sha256"])
    score = load_original_capture(LR12_CAPTURE, lr12_artifact["capture"]["sha256"])
    old_receipt = f1.read_json(OLD_CAPTURE_RECEIPT)
    old = load_original_capture(OLD_CAPTURE, old_receipt["capture_sha256"])
    parent_result = f1.read_json(PARENT_RESULT)
    if f1.sha256_file(PARENT_PAYLOAD) != parent_result["payload"]["sha256"]:
        raise GravityError("parent payload changed after contextual capture")
    _, base_components = f1.decode_base_weights(PARENT_PAYLOAD)
    old_base, old_parent = causal.compact_expert_output(PARENT_PAYLOAD, old["score_x"])
    old["score_base_output"] = old_base
    results = {}
    for index, row in enumerate(ROWS):
        results[row["name"]] = run_row(
            repo, row, output_dir, receipt, seam, fit, score, old, old_parent,
            base_components, parent_result["payload"]["sha256"],
            seed + index * 10_000, n1_steps, n6_steps,
        )
    family_results = {}
    promoted = []
    for family in ("N1", "N2", "N4", "N6"):
        candidate_row = next(row for row in ROWS if row["family"] == family and
                             row["ablation_of"] is None)
        ablation_row = next(row for row in ROWS if row["family"] == family and
                            row["ablation_of"] is not None)
        candidate = results[candidate_row["name"]]
        ablation = results[ablation_row["name"]]
        comparison = compare_ablation(candidate, ablation, seed + len(family))
        admitted, reasons = admission(
            candidate["frozen_score_metrics"], candidate["grouped_cv"],
            candidate["physical_payload"], comparison["candidate_beats_ablation"],
        )
        family_results[family] = {
            "candidate": candidate_row["name"], "ablation": ablation_row["name"],
            "ablation_comparison": comparison,
            "decision": "PROMOTE_TO_NEW_HELDOUT_F2" if admitted else "RETIRE_AT_CONTEXTUAL_F1",
            "reasons": reasons,
        }
        if admitted:
            promoted.append(candidate)
    if promoted:
        best = max(promoted, key=lambda value: (
            value["frozen_score_metrics"]["routed_tokens"][
                "paired_relative_l2_improvement"]["mean"],
            -value["physical_payload"]["complete_bpw"],
        ))
        decision = "PROMOTE_STRONGEST_NONLINEAR_ROW_TO_NEW_HELDOUT_F2"
        next_experiment = f"{best['experiment_id']}_NEW_CONTEXT_HELDOUT_F2"
        best_summary = {"candidate": best["experiment_id"],
                        "complete_bpw": best["physical_payload"]["complete_bpw"],
                        "payload_sha256": best["physical_payload"]["sha256"]}
    else:
        decision = "RETIRE_N1_N2_N4_N6_CONTEXTUAL_ROWS"
        next_experiment = "N3_ASYMMETRIC_STRUCTURAL_ALLOCATION_OR_REGION_CLOSURE"
        best_summary = {"candidate": PARENT_NAME,
                        "complete_bpw": parent_result["physical_budget"]["actual_complete_bpw"],
                        "payload_sha256": parent_result["payload"]["sha256"]}
    artifact = f1.seal({
        "schema": f"{SCHEMA}.tournament", "status": "PASS", "sealed_at": f1.now(),
        "experiment_id": "NG_N1_N2_N4_N6_TOURNAMENT",
        "source": reference.source_identity(source),
        "contextual_capture_seal_sha256": receipt["seal_sha256"],
        "split_law": {"fit_and_grouped_cv": "LR11_ONLY",
                      "frozen_score": "LR12_ONCE_PER_SERIALIZED_ROW",
                      "new_heldout_required_before_F2_claim": True},
        "physical_law": {"logical_weights_represented": LOGICAL_WEIGHTS,
                         "complete_ceiling_bpw": COMPLETE_CEILING_BPW,
                         "complete_ceiling_bytes": COMPLETE_CEILING_BYTES,
                         "all_components_headers_and_tables_billed": True},
        "rows": [{"candidate": result["experiment_id"], "family": result["family"],
                  "ablation_of": result["ablation_of"],
                  "physical_payload": result["physical_payload"],
                  "f0": result["f0"], "grouped_cv": result["grouped_cv"],
                  "frozen_score_metrics": result["frozen_score_metrics"],
                  "seal_sha256": result["seal_sha256"]}
                 for result in results.values()],
        "family_decisions": family_results, "decision": decision,
        "best": best_summary, "next_experiment": next_experiment,
        "claim_boundary": (
            "CONTEXTUAL F0/F1 EXPERT0 TOURNAMENT; N3/N5 and new-context F2 remain separate"
        ),
    })
    chapter.mirror_json(TOURNAMENT_JSON, artifact)
    hypothesis = "One nonlinear/native family survives contextual F1 and its mandatory ablation."
    before = resource_snapshot()
    started = time.time()
    finish_lane(
        repo, "NG_N1_N2_N4_N6_TOURNAMENT", hypothesis, decision,
        {"families": family_results, "best": best_summary}, artifact["seal_sha256"],
        before, started, next_experiment=next_experiment,
        notify=("[Kimi Gravity] N1/N2/N4/N6 tournament sealed\n"
                f"decision: {decision}\n"
                f"best: {best_summary['candidate']} / {best_summary['complete_bpw']:.6f} BPW\n"
                f"next: {next_experiment}"),
    )
    prior = read_status(repo)
    chapter.write_status({
        **prior, "current_best_candidate": best_summary["candidate"],
        "current_best_bpw": best_summary["complete_bpw"],
        "f2_promotable": bool(promoted), "next_experiment": next_experiment,
        "latest_result": {"experiment_id": artifact["experiment_id"],
                          "decision": decision, "best": best_summary,
                          "evidence_seal_sha256": artifact["seal_sha256"]},
    })
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "tournament"))
    parser.add_argument("--repo", type=Path, default=chapter.REPO)
    parser.add_argument("--source", type=Path, default=chapter.legacy.SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=26072160)
    parser.add_argument("--n1-steps", type=int, default=160)
    parser.add_argument("--n6-steps", type=int, default=120)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    source = args.source.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    try:
        if args.stage == "capture":
            result = capture_stage(repo, source, output_dir)
        else:
            result = tournament_stage(
                repo, source, output_dir, args.seed, args.n1_steps, args.n6_steps,
            )
        print(json.dumps({"status": result["status"],
                          "experiment_id": result["experiment_id"],
                          "decision": result.get("decision"),
                          "seal_sha256": result["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "stage": args.stage,
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
