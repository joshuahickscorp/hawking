#!/usr/bin/env python3.12
"""Cached/offline manager for the bounded Kimi Gravity M1--M7 one-offs.

This program deliberately has no MLX dependency.  It reuses sealed LR11/LR12
captures for the parts of M1 and M7 that are answerable without another model
forward, validates serialized M5 exports when a nonlinear tournament provides
them, and writes explicit prerequisite records for M2/M3/M4/M6.

M1's payloads are oracle records, not deployable inference artifacts: score-set
correction coefficients and any oracle-selected token IDs are billed exactly.
The resulting bytes therefore give a reproducible physical realization of the
tested oracle bound while making the teacher-at-inference violation explicit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterable

import numpy as np


LOGICAL_WEIGHTS = 44_040_192
COMPLETE_CEILING_BYTES = 5_394_923
CURRENT_PARENT_BYTES = 5_001_815
CURRENT_PARENT_BPW = CURRENT_PARENT_BYTES * 8 / LOGICAL_WEIGHTS
ORACLE_MAGIC = b"K26ORCL1"
GATE_MAGIC = b"K26GATE1"
SCHEMA_PREFIX = "hawking.kimi_k26.gravity"

M1_ARTIFACT = "KIMI_K26_GRAVITY_M1_ORACLE_BANDWIDTH.json"
M2_ARTIFACT = "KIMI_K26_GRAVITY_M2_CONDITIONAL_GATE.json"
M5_ARTIFACT = "KIMI_K26_GRAVITY_M5_RATE_STRESS.json"
M7_ARTIFACT = "KIMI_K26_GRAVITY_M7_ORACLE_GAP.json"
HOOKS_ARTIFACT = "KIMI_K26_GRAVITY_ONEOFF_HOOKS.json"
INDEX_ARTIFACT = "KIMI_K26_GRAVITY_ONEOFFS_INDEX.json"

LR11_ARTIFACT = "KIMI_K26_LONG_RUN_UPSTREAM_F2.json"
LR12_ARTIFACT = "KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json"

TOURNAMENT_DEFAULTS = (
    "KIMI_K26_CONTEXTUAL_SEAM_TOURNAMENT.json",
    "KIMI_K26_GRAVITY_CONTEXTUAL_SEAMS.json",
    "KIMI_K26_FINAL_NONLINEAR_TOURNAMENT.json",
    "KIMI_K26_FINAL_CHAPTER_TOURNAMENT.json",
    "KIMI_K26_FINAL_BYTE_AUCTION.json",
)

M5_TARGETS = {
    "0.75": {"target_bpw": 0.75, "ceiling_bytes": 4_128_768},
    "0.50": {"target_bpw": 0.50, "ceiling_bytes": 2_752_512},
    "0.33": {"target_bpw": 0.33, "ceiling_bytes": 1_816_657},
}


class OneoffError(RuntimeError):
    """Fail-closed input, serialization, or evidence error."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def verify_seal(value: dict[str, Any], *, label: str) -> str:
    expected = value.get("seal_sha256")
    if not isinstance(expected, str):
        raise OneoffError(f"{label} has no evidence seal")
    actual = seal(value)["seal_sha256"]
    if actual != expected:
        raise OneoffError(f"{label} seal mismatch: {actual} != {expected}")
    return actual


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OneoffError(f"required JSON absent or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise OneoffError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_sha256: str | None, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise OneoffError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise OneoffError(f"{label} hash mismatch: {actual} != {expected_sha256}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted(set(int(item.strip()) for item in value.split(",") if item.strip())))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(sorted(set(float(item.strip()) for item in value.split(",") if item.strip())))
    if not result or any(not 0 < item <= 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated fractions in (0,1]")
    return result


def artifact_capture(repo: Path, name: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    artifact_path = repo / name
    artifact = read_json(artifact_path)
    seal_sha = verify_seal(artifact, label=name)
    capture = artifact.get("capture")
    if not isinstance(capture, dict):
        raise OneoffError(f"{name} has no capture receipt")
    path_value = capture.get("path")
    if not isinstance(path_value, str):
        raise OneoffError(f"{name} capture has no path")
    path = Path(path_value).expanduser().resolve()
    receipt = validate_file(path, capture.get("sha256"), label=f"{name} capture")
    return artifact, path, {
        **receipt, "artifact": name, "artifact_seal_sha256": seal_sha,
    }


def input_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def resume_artifact(path: Path, fingerprint: str, *, force: bool) -> dict[str, Any] | None:
    if force or not path.is_file():
        return None
    existing = read_json(path)
    verify_seal(existing, label=path.name)
    if existing.get("run_fingerprint_sha256") != fingerprint:
        return None
    return existing


def discover_tournament(repo: Path, explicit: Path | None) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    status_path = repo / "KIMI_K26_FINAL_CHAPTER_STATUS.json"
    if status_path.is_file():
        status = read_json(status_path)
        for key in ("tournament_artifact", "contextual_seam_artifact", "byte_auction_artifact"):
            value = status.get(key)
            if isinstance(value, str):
                candidates.append(Path(value).expanduser())
    candidates.extend(repo / name for name in TOURNAMENT_DEFAULTS)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (repo / candidate)
        resolved = resolved.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            continue
        value = read_json(resolved)
        try:
            seal_sha = verify_seal(value, label=resolved.name)
        except OneoffError as exc:
            if explicit is not None and resolved == explicit.expanduser().resolve():
                raise
            continue
        return resolved, value, {
            "status": "PRESENT_AND_SEALED", "path": str(resolved),
            "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved),
            "seal_sha256": seal_sha,
        }
    return None, None, {"status": "ABSENT", "searched": [str(path) for path in candidates]}


def nested_export(tournament: dict[str, Any] | None, key: str) -> Any:
    if tournament is None:
        return None
    paths = (
        ("oneoff_exports", key),
        ("oneoffs", key),
        (key,),
        (key.lower(),),
    )
    for path in paths:
        value: Any = tournament
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            return value
    return None


def row_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(
        np.asarray(reference, dtype=np.float32) - np.asarray(candidate, dtype=np.float32),
        axis=1,
    )
    denominator = np.linalg.norm(np.asarray(reference, dtype=np.float32), axis=1) + 1e-30
    return (numerator / denominator).astype(np.float64)


def quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    cosine = np.sum(ref * cand, axis=1) / (
        np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1) + 1e-30
    )
    return {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_p10": float(np.percentile(cosine, 10)),
        "cosine_min": float(np.min(cosine)),
        "relative_l2": float(np.linalg.norm(ref - cand) / (np.linalg.norm(ref) + 1e-30)),
        "norm_ratio": float(np.linalg.norm(cand) / (np.linalg.norm(ref) + 1e-30)),
    }


def wald_interval(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(data))
    sem = float(np.std(data, ddof=1) / math.sqrt(data.size)) if data.size > 1 else 0.0
    return {
        "mean": mean, "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem, "n": int(data.size),
        "method": "PAIRED_WALD_SCREEN",
    }


def bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> dict[str, float | int | str]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        raise OneoffError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    chunk = min(128, draws)
    for offset in range(0, draws, chunk):
        count = min(chunk, draws - offset)
        indices = rng.integers(0, data.size, size=(count, data.size))
        means[offset:offset + count] = data[indices].mean(axis=1)
    return {
        "mean": float(np.mean(data)), "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)), "n": int(data.size),
        "draws": draws, "method": "PAIRED_TOKEN_BOOTSTRAP",
    }


def randomized_basis(delta: np.ndarray, rank: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(delta, dtype=np.float32)
    actual_rank = min(rank, matrix.shape[0], matrix.shape[1])
    oversample = min(8, max(0, min(matrix.shape) - actual_rank))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], actual_rank + oversample), dtype=np.float32)
    sketch = matrix @ omega
    q, _ = np.linalg.qr(sketch, mode="reduced")
    reduced = q.T @ matrix
    _, singular, right = np.linalg.svd(reduced, full_matrices=False)
    basis = np.asarray(right[:actual_rank], dtype=np.float32)
    coefficients = matrix @ basis.T
    captured = float(
        np.sum(coefficients.astype(np.float64) ** 2) /
        (np.sum(matrix.astype(np.float64) ** 2) + 1e-30)
    )
    return basis, {
        "method": "DETERMINISTIC_RANDOMIZED_RANGE_SVD", "seed": seed,
        "requested_rank": rank, "actual_rank": actual_rank, "oversample": oversample,
        "fit_energy_captured": captured,
        "singular_values": [float(value) for value in singular[:actual_rank]],
    }


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    raw = np.asarray(values, dtype=np.uint32).reshape(-1)
    if raw.size and int(raw.max()) >= 1 << bits:
        raise OneoffError("quantized value exceeds its packed bit width")
    expanded = ((raw[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(expanded.reshape(-1), bitorder="little").tobytes()


def quantized_block(
    value: np.ndarray, bits: int, *, scale_axis: int, name: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.asarray(value, dtype=np.float32)
    if bits == 16:
        encoded = np.asarray(matrix, dtype="<f2")
        return encoded.astype(np.float32), [{
            "name": f"{name}_values", "encoding": "FLOAT16_LE",
            "shape": list(matrix.shape), "count": int(matrix.size),
            "payload": encoded.tobytes(order="C"),
        }]
    if bits not in {2, 4, 8}:
        raise OneoffError(f"unsupported oracle precision: {bits}")
    qmax = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(matrix), axis=scale_axis, keepdims=True)
    scale = np.where(maximum > 0, maximum / qmax, 1.0).astype(np.float32)
    signed = np.clip(np.rint(matrix / scale), -qmax, qmax).astype(np.int16)
    unsigned = (signed.astype(np.int32) + qmax).astype(np.uint32)
    packed = pack_unsigned(unsigned, bits)
    scales = np.asarray(np.squeeze(scale, axis=scale_axis), dtype="<f4")
    reconstructed = signed.astype(np.float32) * scale
    return reconstructed, [
        {
            "name": f"{name}_values", "encoding": f"SIGNED_SYMMETRIC_{bits}BIT_LSB",
            "shape": list(matrix.shape), "count": int(matrix.size), "payload": packed,
        },
        {
            "name": f"{name}_scales", "encoding": "FLOAT32_LE",
            "shape": list(scales.shape), "count": int(scales.size),
            "payload": scales.tobytes(order="C"),
        },
    ]


def oracle_container(metadata: dict[str, Any], blocks: Iterable[dict[str, Any]]) -> tuple[bytes, dict[str, Any]]:
    materialized = list(blocks)
    descriptions = []
    for block in materialized:
        payload = block["payload"]
        descriptions.append({
            key: value for key, value in block.items() if key != "payload"
        } | {"nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    header_value = {
        "schema": f"{SCHEMA_PREFIX}.oracle_payload.v1", "metadata": metadata,
        "components": descriptions,
    }
    header = canonical(header_value)
    payload = ORACLE_MAGIC + struct.pack("<Q", len(header)) + header
    payload += b"".join(block["payload"] for block in materialized)
    receipt = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "header_bytes": len(ORACLE_MAGIC) + 8 + len(header),
        "component_bytes": sum(len(block["payload"]) for block in materialized),
        "components": descriptions,
    }
    return payload, receipt


def gate_container(metadata: dict[str, Any], blocks: Iterable[dict[str, Any]]) -> tuple[bytes, dict[str, Any]]:
    materialized = list(blocks)
    descriptions = []
    for block in materialized:
        block_payload = block["payload"]
        descriptions.append({
            key: value for key, value in block.items() if key != "payload"
        } | {
            "nbytes": len(block_payload),
            "sha256": hashlib.sha256(block_payload).hexdigest(),
        })
    header_value = {
        "schema": f"{SCHEMA_PREFIX}.conditional_gate_payload.v1",
        "metadata": metadata, "components": descriptions,
    }
    header = canonical(header_value)
    payload = GATE_MAGIC + struct.pack("<Q", len(header)) + header
    payload += b"".join(block["payload"] for block in materialized)
    return payload, {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "header_bytes": len(GATE_MAGIC) + 8 + len(header),
        "component_bytes": sum(len(block["payload"]) for block in materialized),
        "components": descriptions,
    }


def select_indices(error: np.ndarray, fraction: float, policy: str) -> np.ndarray:
    count = max(1, min(error.size, int(math.ceil(error.size * fraction))))
    if policy == "ORACLE_TOP_ERROR":
        selected = np.argsort(-error, kind="stable")[:count]
        return np.sort(selected.astype(np.uint32))
    if policy == "PERIODIC":
        selected = np.floor(np.arange(count, dtype=np.float64) * error.size / count)
        return selected.astype(np.uint32)
    raise OneoffError(f"unknown selection policy: {policy}")


def build_oracle_row(
    *, layer: int, rank: int, bits: int, fraction: float, policy: str,
    basis: np.ndarray, score_delta: np.ndarray, teacher: np.ndarray,
    parent: np.ndarray, baseline_rows: np.ndarray, capture_sha256: str,
    fit_capture_sha256: str,
) -> tuple[dict[str, Any], bytes, np.ndarray]:
    selected = select_indices(baseline_rows, fraction, policy)
    basis_slice = np.asarray(basis[:rank], dtype=np.float32)
    basis_dequantized, basis_blocks = quantized_block(
        basis_slice, bits, scale_axis=1, name="direction_basis",
    )
    coefficients = score_delta[selected.astype(np.int64)] @ basis_slice.T
    coefficient_dequantized, coefficient_blocks = quantized_block(
        coefficients, bits, scale_axis=0, name="oracle_coefficients",
    )
    correction = coefficient_dequantized @ basis_dequantized
    residual = score_delta[selected.astype(np.int64)] - correction
    candidate_rows = baseline_rows.copy()
    candidate_rows[selected] = (
        np.linalg.norm(residual, axis=1) /
        (np.linalg.norm(teacher[selected], axis=1) + 1e-30)
    )
    improvement = baseline_rows - candidate_rows
    candidate_selected = parent[selected] + correction
    baseline_cosine = np.sum(teacher * parent, axis=1) / (
        np.linalg.norm(teacher, axis=1) * np.linalg.norm(parent, axis=1) + 1e-30
    )
    selected_cosine = np.sum(teacher[selected] * candidate_selected, axis=1) / (
        np.linalg.norm(teacher[selected], axis=1) *
        np.linalg.norm(candidate_selected, axis=1) + 1e-30
    )
    candidate_cosine = baseline_cosine.copy()
    candidate_cosine[selected] = selected_cosine
    blocks = [*basis_blocks, *coefficient_blocks]
    selector_encoding = (
        "IMPLICIT_ALL" if selected.size == baseline_rows.size else
        ("IMPLICIT_PERIODIC" if policy == "PERIODIC" else "EXPLICIT_UINT32")
    )
    if policy == "ORACLE_TOP_ERROR" and selected.size != baseline_rows.size:
        selector = np.asarray(selected, dtype="<u4")
        blocks.append({
            "name": "score_token_indices", "encoding": "UINT32_LE",
            "shape": [int(selector.size)], "count": int(selector.size),
            "payload": selector.tobytes(order="C"),
        })
    metadata = {
        "claim_boundary": "NONDEPLOYABLE_TEACHER_HIDDEN_ORACLE_BOUND",
        "teacher_access_at_inference": True, "layer": layer, "rank": rank,
        "precision_bits": bits, "requested_token_fraction": fraction,
        "selected_tokens": int(selected.size), "score_tokens": int(baseline_rows.size),
        "selector_encoding": selector_encoding, "selection_policy": policy,
        "fit_capture_sha256": fit_capture_sha256,
        "score_capture_sha256": capture_sha256,
    }
    payload, receipt = oracle_container(metadata, blocks)
    baseline_mean = float(np.mean(baseline_rows))
    candidate_mean = float(np.mean(candidate_rows))
    interval = wald_interval(improvement)
    row = {
        **metadata,
        "actual_token_fraction": float(selected.size / baseline_rows.size),
        "update_frequency_tokens_per_correction": float(baseline_rows.size / selected.size),
        "baseline_row_relative_l2_mean": baseline_mean,
        "candidate_row_relative_l2_mean": candidate_mean,
        "boundary_rescue_fraction": float(1.0 - candidate_mean / (baseline_mean + 1e-30)),
        "paired_row_relative_l2_improvement": interval,
        "candidate_row_cosine_mean": float(np.mean(candidate_cosine)),
        "serialized_oracle_payload": receipt,
        "incremental_complete_physical_bytes": CURRENT_PARENT_BYTES + receipt["bytes"],
        "incremental_complete_bpw": (
            (CURRENT_PARENT_BYTES + receipt["bytes"]) * 8 / LOGICAL_WEIGHTS
        ),
        "within_incremental_0_98_bpw": (
            CURRENT_PARENT_BYTES + receipt["bytes"] <= COMPLETE_CEILING_BYTES
        ),
    }
    return row, payload, improvement


def pareto_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    ordered = sorted(
        rows, key=lambda row: (
            row["serialized_oracle_payload"]["bytes"],
            -row["boundary_rescue_fraction"], row["layer"],
        ),
    )
    frontier = []
    best = -float("inf")
    for row in ordered:
        rescue = float(row["boundary_rescue_fraction"])
        if rescue > best + 1e-9:
            frontier.append(row)
            best = rescue
    if len(frontier) <= limit:
        return frontier
    positions = np.linspace(0, len(frontier) - 1, num=limit, dtype=np.int64)
    return [frontier[int(index)] for index in sorted(set(positions.tolist()))]


def run_m1(args: argparse.Namespace, tournament_receipt: dict[str, Any]) -> dict[str, Any]:
    repo = args.repo
    lr11, lr11_capture, lr11_receipt = artifact_capture(repo, LR11_ARTIFACT)
    lr12, lr12_capture, lr12_receipt = artifact_capture(repo, LR12_ARTIFACT)
    source11 = lr11.get("source", {})
    source12 = lr12.get("source", {})
    if source11.get("revision") != source12.get("revision"):
        raise OneoffError("LR11/LR12 source revisions differ")
    token11 = lr11.get("tokenization", {}).get("token_id_sha256")
    token12 = lr12.get("tokenization", {}).get("token_id_sha256")
    if not token11 or not token12 or token11 == token12:
        raise OneoffError("LR11/LR12 context membership is absent or identical")
    run_inputs = {
        "stage": "M1", "lr11": lr11_receipt, "lr12": lr12_receipt,
        "lr11_context_sha256": token11, "lr12_context_sha256": token12,
        "ranks": args.ranks, "bits": args.bits, "fractions": args.fractions,
        "selection_policies": args.selection_policies, "seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "tournament": tournament_receipt,
    }
    fingerprint = input_fingerprint(run_inputs)
    output_path = repo / M1_ARTIFACT
    resumed = resume_artifact(output_path, fingerprint, force=args.force)
    if resumed is not None:
        return resumed
    started_at = now()
    started_wall = time.time()
    all_rows: list[dict[str, Any]] = []
    layer_state: dict[int, dict[str, Any]] = {}
    for layer, teacher_key, parent_key in (
        (1, "teacher_hidden_l1", "parent_hidden_l1"),
        (2, "teacher_hidden_l2", "parent_hidden_l2"),
    ):
        with np.load(lr11_capture, allow_pickle=False) as fit_capture:
            fit_teacher = np.asarray(fit_capture[teacher_key], dtype=np.float32)
            fit_parent = np.asarray(fit_capture[parent_key], dtype=np.float32)
        with np.load(lr12_capture, allow_pickle=False) as score_capture:
            score_teacher = np.asarray(score_capture[teacher_key], dtype=np.float32)
            score_parent = np.asarray(score_capture[parent_key], dtype=np.float32)
        if fit_teacher.shape != fit_parent.shape or score_teacher.shape != score_parent.shape:
            raise OneoffError(f"layer {layer} teacher/parent capture shape mismatch")
        fit_delta = fit_teacher - fit_parent
        score_delta = score_teacher - score_parent
        baseline_rows = row_relative_l2(score_teacher, score_parent)
        maximum_rank = min(max(args.ranks), *fit_delta.shape)
        basis, basis_info = randomized_basis(fit_delta, maximum_rank, args.seed + layer)
        layer_rows: list[dict[str, Any]] = []
        for rank in args.ranks:
            if rank > basis.shape[0]:
                continue
            for bits in args.bits:
                for policy in args.selection_policies:
                    for fraction in args.fractions:
                        row, _, _ = build_oracle_row(
                            layer=layer, rank=rank, bits=bits, fraction=fraction,
                            policy=policy, basis=basis, score_delta=score_delta,
                            teacher=score_teacher, parent=score_parent,
                            baseline_rows=baseline_rows,
                            capture_sha256=lr12_receipt["sha256"],
                            fit_capture_sha256=lr11_receipt["sha256"],
                        )
                        layer_rows.append(row)
        all_rows.extend(layer_rows)
        layer_state[layer] = {
            "basis": basis, "basis_info": basis_info, "score_delta": score_delta,
            "teacher": score_teacher, "parent": score_parent,
            "baseline_rows": baseline_rows,
            "baseline_quality": quality(score_teacher, score_parent),
            "fit_tokens": int(fit_teacher.shape[0]), "score_tokens": int(score_teacher.shape[0]),
        }
        del fit_teacher, fit_parent, fit_delta
        gc.collect()
    per_layer_limit = max(1, int(math.ceil(args.forward_queue_limit / 2)))
    frontier = []
    for layer in (1, 2):
        frontier.extend(pareto_rows(
            [row for row in all_rows if row["layer"] == layer],
            limit=per_layer_limit,
        ))
    queue = []
    qualifying = []
    payload_dir = args.payload_dir / "M1"
    for index, screened in enumerate(frontier):
        state = layer_state[int(screened["layer"])]
        row, payload, improvement = build_oracle_row(
            layer=int(screened["layer"]), rank=int(screened["rank"]),
            bits=int(screened["precision_bits"]),
            fraction=float(screened["requested_token_fraction"]),
            policy=str(screened["selection_policy"]), basis=state["basis"],
            score_delta=state["score_delta"], teacher=state["teacher"],
            parent=state["parent"], baseline_rows=state["baseline_rows"],
            capture_sha256=lr12_receipt["sha256"],
            fit_capture_sha256=lr11_receipt["sha256"],
        )
        interval = bootstrap_interval(
            improvement, args.seed + 10_000 + index, args.bootstrap_draws,
        )
        row["paired_row_relative_l2_improvement"] = interval
        baseline_mean = float(row["baseline_row_relative_l2_mean"])
        row["boundary_rescue_ci95"] = {
            "mean": float(interval["mean"]) / (baseline_mean + 1e-30),
            "ci95_low": float(interval["ci95_low"]) / (baseline_mean + 1e-30),
            "ci95_high": float(interval["ci95_high"]) / (baseline_mean + 1e-30),
            "method": interval["method"], "n": interval["n"],
        }
        payload_name = (
            f"M1_L{row['layer']}_R{row['rank']}_B{row['precision_bits']}_"
            f"F{row['selected_tokens']}_{row['selection_policy']}.k26oracle"
        )
        payload_path = payload_dir / payload_name
        atomic_bytes(payload_path, payload)
        live_receipt = validate_file(
            payload_path, row["serialized_oracle_payload"]["sha256"],
            label=payload_name,
        )
        row["serialized_oracle_payload"] = {
            **row["serialized_oracle_payload"], **live_receipt,
        }
        row["requires_real_forward"] = True
        row["real_forward_scope"] = (
            "CACHED_LAYER1_STATE_THROUGH_LAYERS2_AND3" if row["layer"] == 1 else
            "CACHED_LAYER2_STATE_THROUGH_LAYER3"
        )
        row["boundary_qualifies_for_forward"] = bool(
            row["boundary_rescue_ci95"]["mean"] >= 0.90 and
            row["boundary_rescue_ci95"]["ci95_low"] >= 0.80
        )
        queue.append(row)
        if row["boundary_qualifies_for_forward"]:
            qualifying.append(row)
    for state in layer_state.values():
        state.pop("basis", None)
        state.pop("score_delta", None)
        state.pop("teacher", None)
        state.pop("parent", None)
        state.pop("baseline_rows", None)
    artifact = seal({
        "schema": f"{SCHEMA_PREFIX}.m1_oracle_bandwidth.v1", "status": "PASS",
        "experiment_id": "M1_TEACHER_HIDDEN_REPAIR_BANDWIDTH_CACHED_STAGE",
        "claim_boundary": "CACHED_INJECTION_BOUNDARY_ONLY_NOT_F2",
        "started_at": started_at, "ended_at": now(),
        "duration_seconds": time.time() - started_wall,
        "run_fingerprint_sha256": fingerprint, "inputs": run_inputs,
        "disjointness": {
            "fit": "LR11", "score": "LR12", "fit_context_sha256": token11,
            "score_context_sha256": token12, "different_context_hashes": True,
            "fit_seed": lr11.get("config", {}).get("seed"),
            "score_seed": lr12.get("config", {}).get("seed"),
            "caveat": (
                "LR12 is structurally disjoint from LR11 but its prior aggregate result is known; "
                "it is score evidence, not a pristine promotion held-out set."
            ),
        },
        "physical_rate_law": {
            "logical_weights": LOGICAL_WEIGHTS,
            "current_parent_bytes": CURRENT_PARENT_BYTES,
            "current_parent_bpw": CURRENT_PARENT_BPW,
            "complete_ceiling_bytes": COMPLETE_CEILING_BYTES,
            "complete_ceiling_bpw": COMPLETE_CEILING_BYTES * 8 / LOGICAL_WEIGHTS,
            "all_oracle_basis_coefficients_scales_selectors_and_headers_billed": True,
        },
        "layer_baselines": layer_state,
        "screening": {
            "rows": all_rows, "screen_interval": "PAIRED_WALD",
            "screening_rows_are_not_forward_eligible_without_bootstrap": True,
        },
        "downstream_forward_queue": queue,
        "terminal_criteria": {
            "boundary_forward_admission": (
                "rescue_mean>=0.90 AND paired_bootstrap_rescue_ci95_low>=0.80"
            ),
            "downstream_minimum": (
                "layer3 rescue_mean>=0.90 AND rescue_ci95_low>=0.80 AND "
                "route_matches>=baseline"
            ),
            "saturation": "two successive rank/precision doublings add <0.01 rescue",
        },
        "decision": (
            "ADVANCE_QUALIFYING_ORACLE_ROWS_TO_ONE_CACHED_STATE_FORWARD"
            if qualifying else "NO_CACHED_BOUNDARY_ROW_JUSTIFIES_REAL_FORWARD"
        ),
        "qualifying_rows": len(qualifying),
        "requires_real_model_forward_for_f2": True,
        "deployable_candidate": False,
        "oracle_warning": (
            "Score correction coefficients and ORACLE_TOP_ERROR selectors use teacher state. "
            "Payload BPW is an exact oracle accounting, never an inference candidate."
        ),
    })
    atomic_json(output_path, artifact)
    return artifact


def hashed_projection(value: np.ndarray, bins: int, seed: int) -> np.ndarray:
    """Stateless signed feature hash; only bins/seed must be encoded at inference."""
    matrix = np.asarray(value, dtype=np.float32)
    channels = np.arange(matrix.shape[1], dtype=np.uint64)
    mixed = channels * np.uint64(0x9E3779B185EBCA87) + np.uint64(seed & 0xFFFFFFFF)
    bucket_ids = np.asarray(mixed % np.uint64(bins), dtype=np.int64)
    signs = np.where((mixed >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
    projected = np.empty((matrix.shape[0], bins), dtype=np.float32)
    for bucket in range(bins):
        selected = np.flatnonzero(bucket_ids == bucket)
        if selected.size:
            projected[:, bucket] = (
                matrix[:, selected] @ signs[selected] / math.sqrt(float(selected.size))
            )
        else:
            projected[:, bucket] = 0.0
    return projected


def summary_features(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    absolute = np.abs(matrix)
    return np.stack((
        np.mean(matrix, axis=1), np.std(matrix, axis=1),
        np.sqrt(np.mean(matrix * matrix, axis=1)),
        np.mean(absolute, axis=1), np.max(absolute, axis=1),
        np.min(matrix, axis=1), np.max(matrix, axis=1),
    ), axis=1).astype(np.float32)


def conditional_gate_features(
    x_l1: np.ndarray, parent_output_l1: np.ndarray, bins: int, seed: int,
) -> np.ndarray:
    x = np.asarray(x_l1, dtype=np.float32)
    output = np.asarray(parent_output_l1, dtype=np.float32)
    if x.shape != output.shape:
        raise OneoffError("M2 x_l1/parent_output_l1 shape mismatch")
    x_norm = np.linalg.norm(x, axis=1)
    output_norm = np.linalg.norm(output, axis=1)
    cosine = np.sum(x * output, axis=1) / (x_norm * output_norm + 1e-30)
    ratio = output_norm / (x_norm + 1e-30)
    return np.concatenate((
        hashed_projection(x, bins, seed),
        hashed_projection(output, bins, seed ^ 0x5A17),
        summary_features(x), summary_features(output),
        cosine[:, None].astype(np.float32), ratio[:, None].astype(np.float32),
    ), axis=1).astype(np.float32)


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float64)


def fit_logistic_gate(
    features: np.ndarray, labels: np.ndarray, *, steps: int, l2: float,
) -> dict[str, np.ndarray | float | int]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    weight = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    positives = max(1.0, float(np.sum(y)))
    negatives = max(1.0, float(y.size - np.sum(y)))
    sample_weight = np.where(y > 0.5, negatives / positives, 1.0)
    sample_weight /= np.mean(sample_weight)
    denominator = float(np.sum(sample_weight))
    for step in range(steps):
        probability = sigmoid(z @ weight + bias)
        residual = (probability - y) * sample_weight
        learning_rate = 0.15 / math.sqrt(1.0 + step / 50.0)
        gradient = z.T @ residual / denominator + l2 * weight
        bias_gradient = float(np.sum(residual) / denominator)
        weight -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
    probability = sigmoid(z @ weight + bias)
    loss = -float(np.mean(
        y * np.log(probability + 1e-12) +
        (1.0 - y) * np.log(1.0 - probability + 1e-12)
    ))
    return {
        "mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
        "weight": weight.astype(np.float32), "bias": float(bias),
        "steps": steps, "l2": l2, "train_log_loss": loss,
    }


def predict_logistic_gate(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=np.float32) - np.asarray(model["mean"], dtype=np.float32)
    ) / np.asarray(model["scale"], dtype=np.float32)
    return sigmoid(
        standardized @ np.asarray(model["weight"], dtype=np.float32) +
        float(model["bias"])
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = y[order]
    positives = int(np.sum(ranked))
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / positives)


def binary_metrics(labels: np.ndarray, triggered: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=bool)
    prediction = np.asarray(triggered, dtype=bool)
    tp = int(np.sum(y & prediction))
    fp = int(np.sum(~y & prediction))
    tn = int(np.sum(~y & ~prediction))
    fn = int(np.sum(y & ~prediction))
    return {
        "true_positive": tp, "false_positive": fp, "true_negative": tn,
        "false_negative": fn, "trigger_rate": float(np.mean(prediction)),
        "event_prevalence": float(np.mean(y)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
    }


def run_m2(args: argparse.Namespace, tournament_receipt: dict[str, Any]) -> dict[str, Any]:
    lr11, lr11_capture, lr11_receipt = artifact_capture(args.repo, LR11_ARTIFACT)
    lr12, lr12_capture, lr12_receipt = artifact_capture(args.repo, LR12_ARTIFACT)
    context11 = lr11.get("tokenization", {}).get("token_id_sha256")
    context12 = lr12.get("tokenization", {}).get("token_id_sha256")
    if not context11 or not context12 or context11 == context12:
        raise OneoffError("M2 requires distinct LR11/LR12 context memberships")
    run_inputs = {
        "stage": "M2", "lr11": lr11_receipt, "lr12": lr12_receipt,
        "fit_context_sha256": context11, "score_context_sha256": context12,
        "feature_hash_bins": args.gate_hash_bins, "seed": args.seed,
        "steps": args.gate_steps, "l2": args.gate_l2,
        "trigger_rates": args.gate_trigger_rates,
        "tournament": tournament_receipt,
    }
    fingerprint = input_fingerprint(run_inputs)
    output_path = args.repo / M2_ARTIFACT
    resumed = resume_artifact(output_path, fingerprint, force=args.force)
    if resumed is not None:
        return resumed
    started_at = now()
    started_wall = time.time()
    with np.load(lr11_capture, allow_pickle=False) as capture:
        fit_x = np.asarray(capture["x_l1"], dtype=np.float32)
        fit_output = np.asarray(capture["parent_output_l1"], dtype=np.float32)
        fit_teacher_l2 = np.asarray(capture["teacher_hidden_l2"], dtype=np.float32)
        fit_parent_l2 = np.asarray(capture["parent_hidden_l2"], dtype=np.float32)
    fit_error = row_relative_l2(fit_teacher_l2, fit_parent_l2)
    fit_threshold = float(np.quantile(fit_error, 0.90))
    fit_labels = fit_error >= fit_threshold
    fit_features = conditional_gate_features(
        fit_x, fit_output, args.gate_hash_bins, args.seed,
    )
    del fit_x, fit_output, fit_teacher_l2, fit_parent_l2
    gc.collect()
    rng = np.random.default_rng(args.seed)
    fold_ids = np.arange(fit_features.shape[0], dtype=np.int32) % 4
    rng.shuffle(fold_ids)
    oof_probability = np.empty(fit_features.shape[0], dtype=np.float64)
    fold_losses = []
    for fold in range(4):
        train = fold_ids != fold
        score = ~train
        fold_model = fit_logistic_gate(
            fit_features[train], fit_labels[train], steps=args.gate_steps,
            l2=args.gate_l2,
        )
        oof_probability[score] = predict_logistic_gate(fold_model, fit_features[score])
        fold_losses.append(float(fold_model["train_log_loss"]))
    thresholds = {
        str(rate): float(np.quantile(oof_probability, 1.0 - rate))
        for rate in args.gate_trigger_rates
    }
    full_model = fit_logistic_gate(
        fit_features, fit_labels, steps=args.gate_steps, l2=args.gate_l2,
    )
    with np.load(lr12_capture, allow_pickle=False) as capture:
        score_x = np.asarray(capture["x_l1"], dtype=np.float32)
        score_output = np.asarray(capture["parent_output_l1"], dtype=np.float32)
        score_teacher_l2 = np.asarray(capture["teacher_hidden_l2"], dtype=np.float32)
        score_parent_l2 = np.asarray(capture["parent_hidden_l2"], dtype=np.float32)
    score_error = row_relative_l2(score_teacher_l2, score_parent_l2)
    score_threshold = float(np.quantile(score_error, 0.90))
    score_labels = score_error >= score_threshold
    score_features = conditional_gate_features(
        score_x, score_output, args.gate_hash_bins, args.seed,
    )
    del score_x, score_output, score_teacher_l2, score_parent_l2
    gc.collect()
    combined_weight = np.concatenate((
        np.asarray(full_model["weight"], dtype=np.float32),
        np.asarray([full_model["bias"]], dtype=np.float32),
    ))[None, :]
    dequantized, weight_blocks = quantized_block(
        combined_weight, 8, scale_axis=1, name="logistic_weight_and_bias",
    )
    physical_model = {
        "mean": np.asarray(full_model["mean"], dtype=np.float32),
        "scale": np.asarray(full_model["scale"], dtype=np.float32),
        "weight": dequantized[0, :-1], "bias": float(dequantized[0, -1]),
    }
    score_probability = predict_logistic_gate(physical_model, score_features)
    threshold_array = np.asarray(
        [thresholds[str(rate)] for rate in args.gate_trigger_rates], dtype="<f4",
    )
    mean_array = np.asarray(full_model["mean"], dtype="<f4")
    scale_array = np.asarray(full_model["scale"], dtype="<f4")
    blocks = [*weight_blocks, {
        "name": "feature_mean", "encoding": "FLOAT32_LE",
        "shape": list(mean_array.shape), "count": int(mean_array.size),
        "payload": mean_array.tobytes(order="C"),
    }, {
        "name": "feature_scale", "encoding": "FLOAT32_LE",
        "shape": list(scale_array.shape), "count": int(scale_array.size),
        "payload": scale_array.tobytes(order="C"),
    }, {
        "name": "trigger_thresholds", "encoding": "FLOAT32_LE",
        "shape": list(threshold_array.shape), "count": int(threshold_array.size),
        "payload": threshold_array.tobytes(order="C"),
    }]
    metadata = {
        "claim_boundary": "TRIGGER_GATE_ONLY_NO_HIGH_PRECISION_RESIDUAL_MODULE",
        "teacher_access_at_inference": False,
        "feature_policy": "STATELESS_SIGNED_HASH_PLUS_SUMMARY",
        "feature_hash_bins": args.gate_hash_bins, "feature_hash_seed": args.seed,
        "fit_event": "LR11_LAYER2_PARENT_ERROR_TOP_DECILE",
        "trigger_rates": list(args.gate_trigger_rates),
        "fit_capture_sha256": lr11_receipt["sha256"],
        "score_capture_sha256": lr12_receipt["sha256"],
    }
    payload, payload_receipt = gate_container(metadata, blocks)
    payload_path = args.payload_dir / "M2" / "M2_CONDITIONAL_TRIGGER_GATE.k26gate"
    atomic_bytes(payload_path, payload)
    live_receipt = validate_file(
        payload_path, payload_receipt["sha256"], label="M2 conditional gate payload",
    )
    payload_receipt = {**payload_receipt, **live_receipt}
    rate_metrics = {}
    skillful = []
    prevalence = float(np.mean(score_labels))
    for rate in args.gate_trigger_rates:
        threshold = thresholds[str(rate)]
        metrics = binary_metrics(score_labels, score_probability >= threshold)
        metrics["threshold_selected_from_lr11_oof"] = threshold
        metrics["precision_lift_over_prevalence"] = float(
            metrics["precision"] / (prevalence + 1e-30)
        )
        rate_metrics[str(rate)] = metrics
        if metrics["precision"] > prevalence and metrics["false_negative_rate"] <= 0.50:
            skillful.append(rate)
    artifact = seal({
        "schema": f"{SCHEMA_PREFIX}.m2_conditional_gate.v1", "status": "PASS",
        "experiment_id": "M2_CONDITIONAL_HIGH_PRECISION_GATE_CACHED_STAGE",
        "started_at": started_at, "ended_at": now(),
        "duration_seconds": time.time() - started_wall,
        "run_fingerprint_sha256": fingerprint, "inputs": run_inputs,
        "disjointness": {
            "fit_cv": "LR11_FOUR_FOLD", "score": "LR12",
            "fit_context_sha256": context11, "score_context_sha256": context12,
            "score_used_for_threshold_selection": False,
        },
        "event_definition": {
            "fit_threshold": fit_threshold, "score_threshold": score_threshold,
            "quantile": 0.90, "score_prevalence": prevalence,
        },
        "fit": {
            "tokens": int(fit_features.shape[0]), "features": int(fit_features.shape[1]),
            "four_fold_train_log_losses": fold_losses,
            "oof_average_precision": average_precision(fit_labels, oof_probability),
        },
        "score": {
            "tokens": int(score_features.shape[0]),
            "average_precision": average_precision(score_labels, score_probability),
            "trigger_rows": rate_metrics,
        },
        "serialized_gate_payload": payload_receipt,
        "projected_parent_plus_gate_bytes": CURRENT_PARENT_BYTES + payload_receipt["bytes"],
        "projected_parent_plus_gate_bpw": (
            (CURRENT_PARENT_BYTES + payload_receipt["bytes"]) * 8 / LOGICAL_WEIGHTS
        ),
        "within_0_98_before_high_precision_module": (
            CURRENT_PARENT_BYTES + payload_receipt["bytes"] <= COMPLETE_CEILING_BYTES
        ),
        "skillful_trigger_rates": skillful,
        "decision": (
            "GATE_HAS_HELDOUT_SKILL_WAITING_PHYSICAL_HIGH_PRECISION_MODULE"
            if skillful else "TERMINAL_RETIRE_CONDITIONAL_GATE_NO_HELDOUT_SKILL"
        ),
        "terminal_criteria": {
            "gate_reject": "precision<=event prevalence OR top-decile false-negative rate>0.50",
            "physical_reject": (
                "static/worst-case BPW>0.98 OR paired pre-router improvement CI95 low<=0"
            ),
        },
        "requires_real_model_forward": False,
        "requires_physical_high_precision_module_for_deployable_M2": True,
        "deployable_candidate": False,
        "caveat": (
            "This stage tests whether a teacher-free gate predicts damage. It does not claim F2 "
            "rescue because no high-precision residual representation is installed. The cached "
            "LR11/LR12 export lacks layer-1 route identity, so this is a conservative activation/"
            "compact-output gate; a physical M2 export must add and bill expert/layer routing features."
        ),
    })
    atomic_json(output_path, artifact)
    return artifact


def expected_m5_export() -> dict[str, Any]:
    return {
        "oneoff_exports": {
            "M5": {
                "rows": [{
                    "name": "candidate name", "target_bpw": "0.75|0.50|0.33",
                    "complete_physical_bytes": "integer",
                    "payload": {"path": "complete serialized candidate", "bytes": "integer", "sha256": "hex"},
                    "score": {
                        "parent": {"cosine_mean": "number"},
                        "candidate": {"cosine_mean": "number"},
                        "paired_row_relative_l2_improvement": {
                            "mean": "number", "ci95_low": "number",
                            "ci95_high": "number", "n": "integer",
                        },
                    },
                }],
            },
        },
    }


def normalized_target(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    nearest = min(M5_TARGETS, key=lambda key: abs(float(key) - numeric))
    return nearest if abs(float(nearest) - numeric) <= 0.03 else None


def m5_score(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("score", "f1_score", "heldout_score"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return None


def run_m5(
    args: argparse.Namespace, tournament: dict[str, Any] | None,
    tournament_receipt: dict[str, Any],
) -> dict[str, Any]:
    export = nested_export(tournament, "M5")
    run_inputs = {
        "stage": "M5", "tournament": tournament_receipt,
        "targets": M5_TARGETS, "export_present": export is not None,
    }
    fingerprint = input_fingerprint(run_inputs)
    output_path = args.repo / M5_ARTIFACT
    resumed = resume_artifact(output_path, fingerprint, force=args.force)
    if resumed is not None:
        return resumed
    started_at = now()
    rows_value = export.get("rows") if isinstance(export, dict) else None
    if not isinstance(rows_value, list):
        artifact = seal({
            "schema": f"{SCHEMA_PREFIX}.m5_rate_stress.v1", "status": "WAITING_PREREQUISITE",
            "experiment_id": "M5_AGGRESSIVE_RATE_STRESS_LADDER_CACHED_STAGE",
            "started_at": started_at, "ended_at": now(),
            "run_fingerprint_sha256": fingerprint, "inputs": run_inputs,
            "decision": "WAITING_FOR_THREE_EXACT_SERIALIZED_TOURNAMENT_RATE_ROWS",
            "missing": ["0.75", "0.50", "0.33"],
            "expected_export": expected_m5_export(),
            "terminal_reject_criteria": (
                "Per rung: retire at F1 when paired score improvement CI95 low<=0, "
                "candidate cosine<=same-split parent, or serialized bytes exceed rung ceiling."
            ),
            "real_model_forward_required": False,
        })
        atomic_json(output_path, artifact)
        return artifact
    validated = []
    seen: set[str] = set()
    for raw in rows_value:
        if not isinstance(raw, dict):
            raise OneoffError("M5 row is not an object")
        target = normalized_target(raw.get("target_bpw"))
        if target is None or target in seen:
            raise OneoffError("M5 has an invalid or duplicate target rung")
        seen.add(target)
        target_info = M5_TARGETS[target]
        complete_bytes = raw.get("complete_physical_bytes")
        payload = raw.get("payload")
        if not isinstance(complete_bytes, int) or not isinstance(payload, dict):
            raise OneoffError(f"M5 {target} lacks exact complete bytes/payload")
        payload_path = payload.get("path")
        if not isinstance(payload_path, str):
            raise OneoffError(f"M5 {target} payload path absent")
        receipt = validate_file(
            Path(payload_path).expanduser().resolve(), payload.get("sha256"),
            label=f"M5 {target} payload",
        )
        if receipt["bytes"] != payload.get("bytes") or receipt["bytes"] != complete_bytes:
            raise OneoffError(f"M5 {target} payload is not the complete serialized candidate")
        actual_bpw = complete_bytes * 8 / LOGICAL_WEIGHTS
        score = m5_score(raw)
        interval = score.get("paired_row_relative_l2_improvement") if score else None
        parent = score.get("parent", {}) if score else {}
        candidate = score.get("candidate", {}) if score else {}
        metrics_complete = bool(
            isinstance(interval, dict) and isinstance(interval.get("ci95_low"), (int, float)) and
            isinstance(parent.get("cosine_mean"), (int, float)) and
            isinstance(candidate.get("cosine_mean"), (int, float))
        )
        within_rung = complete_bytes <= target_info["ceiling_bytes"]
        wins = bool(
            metrics_complete and within_rung and interval["ci95_low"] > 0 and
            candidate["cosine_mean"] > parent["cosine_mean"]
        )
        validated.append({
            "name": raw.get("name"), "target": target, **target_info,
            "complete_physical_bytes": complete_bytes, "actual_complete_bpw": actual_bpw,
            "payload": receipt, "score": score, "metrics_complete": metrics_complete,
            "within_rung_ceiling": within_rung,
            "decision": (
                "ESCALATE_OUT_OF_ONEOFF_SCOPE_TO_F2" if wins else
                "TERMINAL_RETIRE_RATE_RUNG_F1"
            ),
        })
    missing = sorted(set(M5_TARGETS) - seen)
    artifact = seal({
        "schema": f"{SCHEMA_PREFIX}.m5_rate_stress.v1",
        "status": "PASS" if not missing else "WAITING_PREREQUISITE",
        "experiment_id": "M5_AGGRESSIVE_RATE_STRESS_LADDER_CACHED_STAGE",
        "started_at": started_at, "ended_at": now(),
        "run_fingerprint_sha256": fingerprint, "inputs": run_inputs,
        "rows": validated, "missing": missing,
        "decision": (
            "M5_THREE_RUNG_F0_F1_COMPLETE" if not missing else
            "WAITING_FOR_MISSING_EXACT_SERIALIZED_RATE_ROWS"
        ),
        "f2_policy": (
            "M5 ends at F1. Any lower-rate row satisfying the promotion gate becomes a "
            "regular candidate and must receive held-out F2 plus replication."
        ),
        "real_model_forward_required": False,
    })
    atomic_json(output_path, artifact)
    return artifact


def best_nonlinear_export(tournament: dict[str, Any] | None) -> dict[str, Any] | None:
    export = nested_export(tournament, "M7")
    if isinstance(export, dict) and isinstance(export.get("best_nonlinear"), dict):
        return export["best_nonlinear"]
    if isinstance(export, dict) and "capture" in export:
        return export
    return None


def run_m7(
    args: argparse.Namespace, tournament: dict[str, Any] | None,
    tournament_receipt: dict[str, Any],
) -> dict[str, Any]:
    lr12, capture_path, capture_receipt = artifact_capture(args.repo, LR12_ARTIFACT)
    context_sha = lr12.get("tokenization", {}).get("token_id_sha256")
    nonlinear = best_nonlinear_export(tournament)
    run_inputs = {
        "stage": "M7", "lr12": capture_receipt,
        "score_context_sha256": context_sha, "tournament": tournament_receipt,
        "nonlinear_export": nonlinear,
        "bootstrap_draws": args.bootstrap_draws, "seed": args.seed,
    }
    fingerprint = input_fingerprint(run_inputs)
    output_path = args.repo / M7_ARTIFACT
    resumed = resume_artifact(output_path, fingerprint, force=args.force)
    if resumed is not None:
        return resumed
    started_at = now()
    with np.load(capture_path, allow_pickle=False) as capture:
        teacher = np.asarray(capture["teacher_hidden_l2"], dtype=np.float32)
        baseline = np.asarray(capture["parent_hidden_l2"], dtype=np.float32)
        linear = np.asarray(capture["candidate_hidden_l2"], dtype=np.float32)
    baseline_error = row_relative_l2(teacher, baseline)
    linear_error = row_relative_l2(teacher, linear)
    nonzero = baseline_error > 1e-12
    linear_explained = np.zeros_like(baseline_error)
    linear_explained[nonzero] = 1.0 - linear_error[nonzero] / baseline_error[nonzero]
    variants: dict[str, Any] = {
        "CURRENT_090859_BASELINE": {
            "quality": quality(teacher, baseline), "row_relative_l2_mean": float(np.mean(baseline_error)),
            "oracle_rescue_explained_fraction": 0.0,
        },
        "BEST_LINEAR_RETIRED_LR12": {
            "quality": quality(teacher, linear), "row_relative_l2_mean": float(np.mean(linear_error)),
            "oracle_rescue_explained_fraction": float(np.mean(linear_explained[nonzero])),
            "explained_fraction_ci95": bootstrap_interval(
                linear_explained[nonzero], args.seed + 70, args.bootstrap_draws,
            ),
        },
        "TEACHER_HIDDEN_ORACLE": {
            "quality": quality(teacher, teacher), "row_relative_l2_mean": 0.0,
            "oracle_rescue_explained_fraction": 1.0,
            "teacher_access_at_inference": True, "deployable": False,
        },
    }
    nonlinear_status: dict[str, Any]
    if nonlinear is None:
        nonlinear_status = {
            "status": "WAITING_PREREQUISITE",
            "missing": "sealed oneoff_exports.M7.best_nonlinear common-score capture",
            "required_fields": {
                "name": "candidate", "score_context_sha256": context_sha,
                "capture": {"path": "npz", "sha256": "hex", "array_key": "candidate hidden"},
                "complete_physical_bytes": "integer", "complete_bpw": "number",
                "attribution": "REPRESENTATION|OPTIMIZATION|BYTE_BUDGET|MIXED_UNRESOLVED",
            },
        }
    else:
        if nonlinear.get("score_context_sha256") != context_sha:
            raise OneoffError("M7 nonlinear candidate is not on the LR12 common score membership")
        receipt_value = nonlinear.get("capture")
        if not isinstance(receipt_value, dict) or not isinstance(receipt_value.get("path"), str):
            raise OneoffError("M7 nonlinear common-score capture receipt absent")
        candidate_capture_path = Path(receipt_value["path"]).expanduser().resolve()
        candidate_receipt = validate_file(
            candidate_capture_path, receipt_value.get("sha256"), label="M7 nonlinear capture",
        )
        array_key = receipt_value.get("array_key")
        if not isinstance(array_key, str):
            raise OneoffError("M7 nonlinear array_key absent")
        with np.load(candidate_capture_path, allow_pickle=False) as capture:
            candidate_hidden = np.asarray(capture[array_key], dtype=np.float32)
        if candidate_hidden.shape != teacher.shape:
            raise OneoffError("M7 nonlinear candidate has a different common-score shape")
        candidate_error = row_relative_l2(teacher, candidate_hidden)
        explained = np.zeros_like(baseline_error)
        explained[nonzero] = 1.0 - candidate_error[nonzero] / baseline_error[nonzero]
        complete_bytes = nonlinear.get("complete_physical_bytes")
        if not isinstance(complete_bytes, int):
            raise OneoffError("M7 nonlinear candidate lacks complete physical bytes")
        variants["BEST_NONLINEAR_PHYSICAL"] = {
            "name": nonlinear.get("name"), "quality": quality(teacher, candidate_hidden),
            "row_relative_l2_mean": float(np.mean(candidate_error)),
            "oracle_rescue_explained_fraction": float(np.mean(explained[nonzero])),
            "explained_fraction_ci95": bootstrap_interval(
                explained[nonzero], args.seed + 71, args.bootstrap_draws,
            ),
            "unexplained_oracle_gap": float(1.0 - np.mean(explained[nonzero])),
            "complete_physical_bytes": complete_bytes,
            "complete_bpw": complete_bytes * 8 / LOGICAL_WEIGHTS,
            "within_0_98_bpw": complete_bytes <= COMPLETE_CEILING_BYTES,
            "capture": candidate_receipt,
        }
        nonlinear_status = {
            "status": "COMMON_SCORE_VARIANT_PRESENT",
            "attribution": nonlinear.get("attribution", "MIXED_UNRESOLVED"),
            "attribution_evidence": nonlinear.get("attribution_evidence"),
        }
    complete = nonlinear_status["status"] == "COMMON_SCORE_VARIANT_PRESENT"
    artifact = seal({
        "schema": f"{SCHEMA_PREFIX}.m7_oracle_gap.v1",
        "status": "PASS" if complete else "PARTIAL_WAITING_PREREQUISITE",
        "experiment_id": "M7_ORACLE_TO_PHYSICAL_GAP_CACHED_STAGE",
        "started_at": started_at, "ended_at": now(),
        "run_fingerprint_sha256": fingerprint, "inputs": run_inputs,
        "common_score": {
            "name": "LR12", "tokens": int(teacher.shape[0]),
            "context_sha256": context_sha, "capture": capture_receipt,
        },
        "definition": (
            "explained_fraction=(E_baseline-E_variant)/(E_baseline-E_oracle), "
            "computed per token before averaging; E_oracle=0 at hidden restoration boundary"
        ),
        "variants": variants, "best_nonlinear": nonlinear_status,
        "decision": (
            "M7_COMMON_SCORE_GAP_COMPLETE" if complete else
            "WAITING_BEST_NONLINEAR_PHYSICAL_ON_LR12_COMMON_SCORE"
        ),
        "terminal_attribution_criteria": {
            "REPRESENTATION_LIMITED": "converged fit and score both plateau with residual error",
            "OPTIMIZATION_LIMITED": "seed/fit instability or failed convergence",
            "BYTE_BUDGET_LIMITED": "same architecture succeeds over budget and degrades monotonically under the ceiling",
            "fallback": "MIXED_UNRESOLVED; never infer a cause without its named control",
        },
        "requires_real_model_forward": False,
    })
    atomic_json(output_path, artifact)
    return artifact


def hook_specs() -> dict[str, Any]:
    common = {
        "F0_reject": (
            "nondeterministic decode OR nonfinite output OR unbilled component OR "
            f"complete_physical_bytes>{COMPLETE_CEILING_BYTES}"
        ),
        "F1_reject": (
            "paired pre-router row-relative-L2 improvement CI95 low<=0 OR "
            "candidate cosine<=same-split parent OR any domain/low-margin stratum CI95 high<0"
        ),
        "F2_reject": (
            "paired hidden improvement CI95 low<=0 OR route_matches<parent OR "
            "layer3 relative-L2>parent"
        ),
        "replication": "new prompt construction, context membership hash, and seed without refit",
    }
    return {
        "M2": {
            "name": "CONDITIONAL_HIGH_PRECISION_ISLANDS",
            "prerequisites": [
                "serialized conditional residual/high-precision module",
                "inference-available gate and fully encoded threshold policy",
                "LR11 fit/CV and LR12 score trigger predictions",
            ],
            "required_export": "oneoff_exports.M2",
            "extra_reject": (
                "static or worst-case BPW>0.98 OR gate precision<=event prevalence OR "
                "top-error-decile false-negative rate>0.50"
            ),
            "real_forward": "only one frozen winner through cached layers 2-3",
            "criteria": common,
        },
        "M3": {
            "name": "NATIVE_STUDENTIZATION",
            "prerequisites": [
                "exact native expert targets for LR11/LR12 cached x_l1",
                "serialized teacher-free functional student",
                "byte-matched parent comparison",
            ],
            "required_export": "oneoff_exports.M3",
            "extra_reject": "fit-only gain, score CI crossing zero, or more bytes than comparator",
            "real_forward": "expert-only target capture; layers 2-3 only after frozen F1 win",
            "criteria": common,
        },
        "M4": {
            "name": "CROSS_LAYER_STATE_ANCHOR",
            "prerequisites": [
                "one physical compact representation installable at layers 2,31,60",
                "teacher and compact trajectories on identical contexts",
            ],
            "required_export": "oneoff_exports.M4",
            "terminal_prerequisite_reject": (
                "PREREQUISITE_ABSENT when no coherent multi-layer compact candidate exists; "
                "a synthetic sentinel perturbation is not admissible evidence"
            ),
            "real_forward": "required at early/middle/late sentinels when prerequisite is met",
            "criteria": common,
        },
        "M6": {
            "name": "DOCTOR_REMOVAL_NATIVE_INVERSION",
            "prerequisites": [
                "current base+Doctor row",
                "same base+native module capped at 974848 bytes",
                "same base+native module at full legal ceiling",
            ],
            "required_export": "oneoff_exports.M6",
            "extra_reject": (
                "native superiority requires a replicated byte-matched win; a CI crossing zero "
                "means no demonstrated advantage, not Doctor optimality"
            ),
            "real_forward": "cached F1 first; only one native winner through layers 2-3",
            "criteria": common,
        },
    }


def run_hooks(
    args: argparse.Namespace, tournament: dict[str, Any] | None,
    tournament_receipt: dict[str, Any],
) -> dict[str, Any]:
    specs = hook_specs()
    statuses = {}
    for key, spec in specs.items():
        export = nested_export(tournament, key)
        statuses[key] = {
            "status": "PREREQUISITE_EXPORT_PRESENT" if export is not None else "WAITING_PREREQUISITE",
            "required_export": spec["required_export"],
            "export_seal_inherited_from_tournament": tournament_receipt.get("seal_sha256"),
        }
    run_inputs = {"stage": "HOOKS", "tournament": tournament_receipt, "statuses": statuses}
    fingerprint = input_fingerprint(run_inputs)
    output_path = args.repo / HOOKS_ARTIFACT
    resumed = resume_artifact(output_path, fingerprint, force=args.force)
    if resumed is not None:
        return resumed
    artifact = seal({
        "schema": f"{SCHEMA_PREFIX}.oneoff_hooks.v1", "status": "PASS",
        "generated_at": now(), "run_fingerprint_sha256": fingerprint,
        "inputs": run_inputs, "hooks": specs, "prerequisite_status": statuses,
        "claim_boundary": (
            "Hooks are preregistered terminal rules, not evidence that the corresponding "
            "one-off ran. A hook closes only when its sealed export is adjudicated."
        ),
    })
    atomic_json(output_path, artifact)
    return artifact


def run_index(args: argparse.Namespace, stage_artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entries = {}
    for stage, artifact in stage_artifacts.items():
        entries[stage] = {
            "artifact": {
                "path": str(args.repo / {
                    "M1": M1_ARTIFACT, "M2": M2_ARTIFACT,
                    "M5": M5_ARTIFACT, "M7": M7_ARTIFACT,
                    "HOOKS": HOOKS_ARTIFACT,
                }[stage]),
                "seal_sha256": artifact["seal_sha256"],
            },
            "status": artifact.get("status"), "decision": artifact.get("decision"),
        }
    artifact = seal({
        "schema": f"{SCHEMA_PREFIX}.oneoffs_index.v1", "status": "PASS",
        "generated_at": now(), "stages": entries,
        "completion_boundary": (
            "M1 is incomplete until its admitted oracle rows receive the named cached-state "
            "downstream forward; M2 gate skill is not rescue without a physical high-precision "
            "module; M5 requires all three serialized rungs; M7 requires the best nonlinear "
            "physical candidate on the common score membership."
        ),
    })
    atomic_json(args.repo / INDEX_ARTIFACT, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage", choices=("hooks", "m1", "m2", "m5", "m7", "index", "all"),
        default="all",
    )
    parser.add_argument("--tournament-artifact", type=Path)
    parser.add_argument(
        "--payload-dir", type=Path,
        default=(Path.home() / "Library/Application Support/Hawking/KimiK26/"
                 "final_chapter/oneoffs"),
    )
    parser.add_argument("--ranks", type=parse_ints, default=(4, 16, 32, 64))
    parser.add_argument("--bits", type=parse_ints, default=(4, 8, 16))
    parser.add_argument(
        "--fractions", type=parse_floats,
        default=(1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0),
    )
    parser.add_argument(
        "--selection-policies", type=lambda value: tuple(
            item.strip().upper() for item in value.split(",") if item.strip()
        ), default=("ORACLE_TOP_ERROR", "PERIODIC"),
    )
    parser.add_argument("--seed", type=int, default=26072121)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--forward-queue-limit", type=int, default=12)
    parser.add_argument("--gate-hash-bins", type=int, default=64)
    parser.add_argument("--gate-steps", type=int, default=300)
    parser.add_argument("--gate-l2", type=float, default=1e-3)
    parser.add_argument(
        "--gate-trigger-rates", type=parse_floats,
        default=(0.01, 0.02, 0.05, 0.10, 0.20),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.repo = args.repo.expanduser().resolve(strict=True)
        args.payload_dir = args.payload_dir.expanduser().resolve()
        if any(bits not in {2, 4, 8, 16} for bits in args.bits):
            raise OneoffError("--bits must contain only 2,4,8,16")
        if args.bootstrap_draws < 200:
            raise OneoffError("--bootstrap-draws must be at least 200")
        if args.forward_queue_limit < 1:
            raise OneoffError("--forward-queue-limit must be positive")
        if args.gate_hash_bins < 8 or args.gate_hash_bins > 512:
            raise OneoffError("--gate-hash-bins must be in [8,512]")
        if args.gate_steps < 20 or args.gate_steps > 5000:
            raise OneoffError("--gate-steps must be in [20,5000]")
        if args.gate_l2 < 0:
            raise OneoffError("--gate-l2 must be nonnegative")
        allowed_policies = {"ORACLE_TOP_ERROR", "PERIODIC"}
        if not args.selection_policies or not set(args.selection_policies) <= allowed_policies:
            raise OneoffError(
                "--selection-policies must use ORACLE_TOP_ERROR and/or PERIODIC"
            )
        tournament_path, tournament, tournament_receipt = discover_tournament(
            args.repo, args.tournament_artifact,
        )
        if tournament_path is not None:
            tournament_receipt["path"] = str(tournament_path)
        stages: dict[str, dict[str, Any]] = {}
        requested = args.stage
        if requested in {"hooks", "all"}:
            stages["HOOKS"] = run_hooks(args, tournament, tournament_receipt)
        if requested in {"m1", "all"}:
            stages["M1"] = run_m1(args, tournament_receipt)
        if requested in {"m2", "all"}:
            stages["M2"] = run_m2(args, tournament_receipt)
        if requested in {"m5", "all"}:
            stages["M5"] = run_m5(args, tournament, tournament_receipt)
        if requested in {"m7", "all"}:
            stages["M7"] = run_m7(args, tournament, tournament_receipt)
        if requested == "index":
            for stage, name in (
                ("HOOKS", HOOKS_ARTIFACT), ("M1", M1_ARTIFACT), ("M2", M2_ARTIFACT),
                ("M5", M5_ARTIFACT), ("M7", M7_ARTIFACT),
            ):
                path = args.repo / name
                artifact = read_json(path)
                verify_seal(artifact, label=name)
                stages[stage] = artifact
        if requested in {"all", "index"}:
            index = run_index(args, stages)
            stages["INDEX"] = index
        print(json.dumps({
            "status": "PASS", "requested_stage": requested,
            "stages": {
                key: {"status": value.get("status"), "decision": value.get("decision"),
                      "seal_sha256": value.get("seal_sha256")}
                for key, value in stages.items()
            },
        }, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
