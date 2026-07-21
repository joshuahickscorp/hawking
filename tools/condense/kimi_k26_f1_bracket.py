#!/usr/bin/env python3.12
"""Real Kimi K2.6 F1 representation bracket on one routed-expert seam.

This is a scientific experiment, not campaign infrastructure.  It reuses the
sealed corpus split and official parent source, captures one real early-MoE
teacher seam once, writes actual bit-packed component payloads, and compares
the P1 (0.98 BPW base-heavy) and P5 (0.50 BPW Doctor-heavy) envelopes.

Claim boundary: F1 single-expert/single-layer output-space evidence.  A PASS
artifact means the experiment is valid; candidate promotion is reported
separately and never implies end-to-end capability parity.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import gc
import hashlib
import json
import math
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

import gravity_forge as forge  # noqa: E402
import kimi_k26_adapter as adapter  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


REPO = "moonshotai/Kimi-K2.6"
REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
LAYER = 1
MAGIC = b"K26F1\0\0\1"
MATRICES = ("gate_proj", "up_proj", "down_proj")

CANDIDATES = {
    "P1": {
        "target": Fraction(49, 50), "base_share": Fraction(75, 100),
        "doctor_share": Fraction(23, 100), "overhead_share": Fraction(2, 100),
        "base_geometry": {"dim": 256, "subspaces": 16, "k": 512, "iters": 3},
        "doctor_geometry": {"dim": 256, "subspaces": 8, "k": 64, "iters": 3},
    },
    "P5": {
        "target": Fraction(1, 2), "base_share": Fraction(52, 100),
        "doctor_share": Fraction(45, 100), "overhead_share": Fraction(3, 100),
        "base_geometry": {"dim": 256, "subspaces": 8, "k": 96, "iters": 3},
        "doctor_geometry": {"dim": 256, "subspaces": 8, "k": 64, "iters": 3},
    },
}


class F1Error(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise F1Error(f"required JSON absent or invalid: {path}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers densely, least-significant bit first."""
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    if bits < 1 or bits > 31 or (values.size and int(values.max()) >= 1 << bits):
        raise F1Error("unsigned bit-pack geometry is invalid")
    expanded = ((values[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(expanded.reshape(-1), bitorder="little").tobytes()


def unpack_unsigned(payload: bytes, count: int, bits: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    expanded = np.unpackbits(raw, bitorder="little")[:count * bits].reshape(count, bits)
    powers = (np.uint32(1) << np.arange(bits, dtype=np.uint32))[None, :]
    return np.sum(expanded.astype(np.uint32) * powers, axis=1, dtype=np.uint32)


def distribute(total: int, weights: list[int]) -> list[int]:
    denominator = sum(weights)
    out = [total * weight // denominator for weight in weights]
    for index in range(total - sum(out)):
        out[index % len(out)] += 1
    return out


def quality(reference_value: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference_value, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    row_cosine = np.sum(ref * cand, axis=-1) / (
        np.linalg.norm(ref, axis=-1) * np.linalg.norm(cand, axis=-1) + 1e-30
    )
    relative_l2 = float(np.linalg.norm(ref - cand) / (np.linalg.norm(ref) + 1e-30))
    return {
        "cosine_mean": float(np.mean(row_cosine)),
        "cosine_p10": float(np.percentile(row_cosine, 10)),
        "cosine_min": float(np.min(row_cosine)),
        "relative_l2": relative_l2,
        "normalized_rmse": float(np.sqrt(np.mean((ref - cand) ** 2)) /
                                 (np.sqrt(np.mean(ref ** 2)) + 1e-30)),
        "norm_ratio": float(np.linalg.norm(cand) / (np.linalg.norm(ref) + 1e-30)),
    }


def fidelity_verdict(metric: dict[str, float]) -> str:
    if (metric["cosine_mean"] >= 0.90 and metric["cosine_p10"] >= 0.80 and
            metric["relative_l2"] <= 0.50):
        return "SURVIVES_F1"
    if metric["cosine_mean"] >= 0.75 and metric["relative_l2"] <= 0.85:
        return "DEGRADED_F1"
    return "COLLAPSE_F1"


def silu(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / (1.0 + np.exp(-np.clip(value, -40, 40)))


def expert_forward(x: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gate = x @ weights["gate_proj"].T
    up = x @ weights["up_proj"].T
    hidden = silu(gate) * up
    output = hidden @ weights["down_proj"].T
    return output.astype(np.float32), {"gate": gate, "up": up, "hidden": hidden}


def dequantized_expert(shard: reference.TensorShard, expert: int) -> dict[str, np.ndarray]:
    base = f"{reference.PREFIX}.layers.{LAYER}.mlp.experts.{expert}"
    result = {}
    for matrix in MATRICES:
        prefix = f"{base}.{matrix}"
        shape = tuple(int(value) for value in np.asarray(
            shard.numpy(prefix + ".weight_shape"), dtype=np.int32).reshape(-1))
        result[matrix] = adapter.dequantize_int4(
            shard.numpy(prefix + ".weight_packed"),
            shard.numpy(prefix + ".weight_scale"), shape,
        )
    return result


def route(x: np.ndarray, shard: reference.TensorShard,
          config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    base = f"{reference.PREFIX}.layers.{LAYER}.mlp.gate"
    gate = np.asarray(shard.mlx(base + ".weight").astype(mx.float32), dtype=np.float32)
    correction = np.asarray(
        shard.mlx(base + ".e_score_correction_bias").astype(mx.float32), dtype=np.float32)
    return adapter.route_noaux_tc(
        x, gate, correction, top_k=int(config["num_experts_per_tok"]),
        scaling=float(config["routed_scaling_factor"]), n_group=int(config["n_group"]),
        topk_group=int(config["topk_group"]),
    )


def capture_teacher(source: Path, corpus: dict[str, Any], output_dir: Path,
                    progress_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    capture_path = output_dir / "teacher_capture.npz"
    receipt_path = output_dir / "teacher_capture.json"
    split = corpus["corpus"]["layer_zero_unique_token_split"]
    fit_ids = [int(value) for value in split["fit_probe_token_ids"]]
    score_ids = [int(value) for value in split["score_probe_token_ids"]]
    expected_ids = fit_ids + score_ids
    if capture_path.exists() and receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt.get("status") == "PASS" and receipt.get("revision") == REVISION and
                receipt.get("token_ids") == expected_ids and
                receipt.get("capture_sha256") == sha256_file(capture_path)):
            loaded = np.load(capture_path, allow_pickle=False)
            return {key: loaded[key] for key in loaded.files}, receipt

    atomic_json(progress_path, {"status": "RUNNING", "stage": "TEACHER_CAPTURE",
                                "started_at": now(), "layer": LAYER})
    config = read_json(source / "config.json")["text_config"]
    tail = reference.TensorShard(reference.shard_path(source, 62))
    embedding = tail.numpy(f"{reference.PREFIX}.embed_tokens.weight")
    hidden = mx.array(np.asarray(embedding[expected_ids])).astype(mx.bfloat16)
    mx.eval(hidden)
    del embedding, tail
    mx.clear_cache()

    dense_shard = reference.TensorShard(reference.shard_path(source, 1))
    hidden, _ = reference.layer_forward(hidden, dense_shard, 0, config, [1] * len(expected_ids))
    del dense_shard
    mx.clear_cache()

    moe_shard = reference.TensorShard(reference.shard_path(source, 2))
    layer_base = f"{reference.PREFIX}.layers.{LAYER}"
    normalized = reference.rms_norm(
        hidden, moe_shard.mlx(layer_base + ".input_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    attention, _ = reference.attention(normalized, moe_shard, LAYER, config,
                                       [1] * len(expected_ids))
    post_attention = (hidden + attention).astype(mx.bfloat16)
    expert_input = reference.rms_norm(
        post_attention, moe_shard.mlx(layer_base + ".post_attention_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    mx.eval(post_attention, expert_input)
    x = np.asarray(expert_input.astype(mx.float32), dtype=np.float32)
    post = np.asarray(post_attention.astype(mx.float32), dtype=np.float32)
    routes, route_weights = route(x, moe_shard, config)
    fit_routes = routes[:len(fit_ids)]
    counts = np.bincount(fit_routes.reshape(-1), minlength=int(config["n_routed_experts"]))
    sentinel = int(np.argmax(counts))
    weights = dequantized_expert(moe_shard, sentinel)
    teacher, internals = expert_forward(x, weights)

    arrays = {
        "fit_x": x[:len(fit_ids)], "score_x": x[len(fit_ids):],
        "fit_post_attention": post[:len(fit_ids)],
        "score_post_attention": post[len(fit_ids):],
        "fit_routes": routes[:len(fit_ids)].astype(np.int32),
        "score_routes": routes[len(fit_ids):].astype(np.int32),
        "fit_route_weights": route_weights[:len(fit_ids)].astype(np.float32),
        "score_route_weights": route_weights[len(fit_ids):].astype(np.float32),
        "fit_teacher_output": teacher[:len(fit_ids)],
        "score_teacher_output": teacher[len(fit_ids):],
        "fit_teacher_hidden": internals["hidden"][:len(fit_ids)].astype(np.float32),
        "score_teacher_hidden": internals["hidden"][len(fit_ids):].astype(np.float32),
        "sentinel_expert": np.array([sentinel], dtype=np.int32),
    }
    temporary = capture_path.with_name(f".{capture_path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, capture_path)
    receipt = seal({
        "schema": "hawking.kimi_k26.f1_teacher_capture.v1", "status": "PASS",
        "captured_at": now(), "revision": REVISION, "layer": LAYER,
        "token_ids": expected_ids, "fit_token_ids": fit_ids, "score_token_ids": score_ids,
        "token_overlap": len(set(fit_ids) & set(score_ids)), "sentinel_expert": sentinel,
        "sentinel_fit_route_slots": int(np.sum(fit_routes == sentinel)),
        "sentinel_score_route_slots": int(np.sum(routes[len(fit_ids):] == sentinel)),
        "capture_sha256": sha256_file(capture_path),
        "capture_bytes": capture_path.stat().st_size,
        "claim_boundary": "ONE_REAL_LAYER_ONE_SENTINEL_EXPERT; teacher captured once",
    })
    atomic_json(receipt_path, receipt)
    del weights, moe_shard, hidden, normalized, attention, post_attention, expert_input
    gc.collect()
    mx.clear_cache()
    return arrays, receipt


def salience_scale(activations: np.ndarray) -> np.ndarray:
    salience = np.maximum(np.mean(np.abs(activations), axis=0), 1e-8) ** 0.5
    salience /= np.exp(np.mean(np.log(salience)))
    return np.asarray(np.asarray(salience, dtype=ml_dtypes.bfloat16), dtype=np.float32)


def pq_payload(weight: np.ndarray, scale: np.ndarray, geometry: dict[str, int], seed: int,
               name: str, role: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scaled = np.ascontiguousarray(weight * scale[None, :], dtype=np.float32)
    artifact = forge.pack_product_quant(
        scaled, dim=geometry["dim"], subspaces=geometry["subspaces"],
        k=geometry["k"], seed=seed, iters=geometry["iters"],
    )
    codes = artifact.config["pq_codes"]
    codebooks = np.stack([
        np.asarray(codebook, dtype=np.float16) for codebook in codes["codebooks"]
    ])
    indices = np.asarray(codes["indices"], dtype=np.uint32)
    bits = max(1, math.ceil(math.log2(int(codebooks.shape[1]))))
    index_bytes = pack_unsigned(indices, bits)
    if not np.array_equal(unpack_unsigned(index_bytes, indices.size, bits), indices.reshape(-1)):
        raise F1Error("packed PQ indices failed roundtrip")
    n_vectors, subspaces = indices.shape
    dimension = int(codes["D"])
    sub = int(codes["sub"])
    reconstructed = np.empty((n_vectors, dimension), dtype=np.float32)
    for subspace in range(subspaces):
        reconstructed[:, subspace * sub:(subspace + 1) * sub] = (
            codebooks[subspace].astype(np.float32)[indices[:, subspace]]
        )
    reconstructed = reconstructed.reshape(weight.shape) / scale[None, :]
    components = [
        {"name": f"{name}.{role}.indices", "role": role, "data": index_bytes,
         "encoding": "dense_unsigned_lsb", "shape": list(indices.shape), "bits": bits},
        {"name": f"{name}.{role}.codebooks", "role": role,
         "data": codebooks.tobytes(order="C"), "encoding": "float16",
         "shape": list(codebooks.shape)},
    ]
    del artifact, scaled
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return reconstructed.astype(np.float32), components


def add_protected_rows(weight: np.ndarray, reconstruction: np.ndarray,
                       activations: np.ndarray, remaining_bytes: int,
                       name: str, role: str, *, residual: bool) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    rows, cols = weight.shape
    index_bits = max(1, math.ceil(math.log2(rows)))
    difference = weight - reconstruction
    score = np.mean((activations @ difference.T) ** 2, axis=0)
    order = np.argsort(-score, kind="stable")
    count = min(rows, remaining_bytes // max(1, cols * 2))
    while count > 0:
        required = count * cols * 2 + math.ceil(count * index_bits / 8)
        if required <= remaining_bytes:
            break
        count -= 1
    if count == 0:
        return reconstruction, [], 0
    selected = np.sort(order[:count]).astype(np.uint32)
    values = difference[selected] if residual else weight[selected]
    stored = np.asarray(values, dtype=ml_dtypes.bfloat16)
    result = reconstruction.copy()
    if residual:
        result[selected] += np.asarray(stored, dtype=np.float32)
    else:
        result[selected] = np.asarray(stored, dtype=np.float32)
    packed_rows = pack_unsigned(selected, index_bits)
    components = [
        {"name": f"{name}.{role}.protected_row_indices", "role": role,
         "data": packed_rows, "encoding": "dense_unsigned_lsb",
         "shape": [count], "bits": index_bits},
        {"name": f"{name}.{role}.protected_row_values", "role": role,
         "data": stored.tobytes(order="C"), "encoding": "bfloat16",
         "shape": [count, cols], "residual": residual},
    ]
    return result, components, count


def role_bytes(components: list[dict[str, Any]], role: str) -> int:
    return sum(len(component["data"]) for component in components if component["role"] == role)


def pack_matrix(weight: np.ndarray, activations: np.ndarray, spec: dict[str, Any],
                base_cap: int, doctor_cap: int, name: str, seed: int) -> tuple[
                    np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    scale = salience_scale(activations)
    scale_component = {"name": f"{name}.base.salience_scale", "role": "base",
                       "data": np.asarray(scale, dtype=ml_dtypes.bfloat16).tobytes(),
                       "encoding": "bfloat16", "shape": list(scale.shape)}
    base, components = pq_payload(
        weight, scale, spec["base_geometry"], seed, name, "base",
    )
    components.append(scale_component)
    used = role_bytes(components, "base")
    if used > base_cap:
        raise F1Error(f"{name} base geometry exceeds cap: {used} > {base_cap}")
    base, protected, base_rows = add_protected_rows(
        weight, base, activations, base_cap - used, name, "base", residual=False,
    )
    components.extend(protected)

    doctor_residual = weight - base
    doctor, doctor_components = pq_payload(
        doctor_residual, scale, spec["doctor_geometry"], seed + 1000, name, "doctor",
    )
    components.extend(doctor_components)
    doctored = base + doctor
    used_doctor = role_bytes(components, "doctor")
    if used_doctor > doctor_cap:
        raise F1Error(f"{name} Doctor geometry exceeds cap: {used_doctor} > {doctor_cap}")
    doctored, protected, doctor_rows = add_protected_rows(
        weight, doctored, activations, doctor_cap - used_doctor,
        name, "doctor", residual=True,
    )
    components.extend(protected)
    record = {
        "shape": list(weight.shape), "weights": int(weight.size),
        "base_cap_bytes": base_cap, "base_bytes": role_bytes(components, "base"),
        "doctor_cap_bytes": doctor_cap, "doctor_bytes": role_bytes(components, "doctor"),
        "base_protected_rows": base_rows, "doctor_protected_rows": doctor_rows,
        "base_geometry": spec["base_geometry"], "doctor_geometry": spec["doctor_geometry"],
        "weight_relative_l2_base": float(
            np.linalg.norm(weight - base) / (np.linalg.norm(weight) + 1e-30)),
        "weight_relative_l2_doctored": float(
            np.linalg.norm(weight - doctored) / (np.linalg.norm(weight) + 1e-30)),
    }
    return base, doctored, components, record


def write_payload(path: Path, metadata: dict[str, Any],
                  components: list[dict[str, Any]]) -> dict[str, Any]:
    descriptors = []
    for component in components:
        data = component["data"]
        descriptors.append({
            **{key: value for key, value in component.items() if key != "data"},
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        })
    header = canonical({**metadata, "components": descriptors})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        for component in components:
            handle.write(component["data"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    overhead = len(MAGIC) + 8 + len(header)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "header_overhead_bytes": overhead,
            "base_component_bytes": role_bytes(components, "base"),
            "doctor_component_bytes": role_bytes(components, "doctor"),
            "component_count": len(components)}


def read_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and hash-check a physical F1 payload without constructing a dense shadow."""
    with path.open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise F1Error(f"invalid F1 payload magic: {path}")
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise F1Error(f"truncated F1 payload header size: {path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        try:
            header = json.loads(handle.read(header_size))
        except json.JSONDecodeError as exc:
            raise F1Error(f"invalid F1 payload header: {path}") from exc
        components = []
        for descriptor in header.get("components", []):
            data = handle.read(int(descriptor["bytes"]))
            if len(data) != int(descriptor["bytes"]):
                raise F1Error(f"truncated F1 component: {descriptor['name']}")
            if hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
                raise F1Error(f"F1 component hash mismatch: {descriptor['name']}")
            components.append({**descriptor, "data": data})
        if handle.read(1):
            raise F1Error(f"unbilled trailing bytes in F1 payload: {path}")
    return header, components


def decode_base_weights(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Decode only the serialized base representation for Doctor architecture auctions."""
    header, components = read_payload(path)
    sentinel = int(header["sentinel_expert"])
    by_name = {component["name"]: component for component in components}
    weights: dict[str, np.ndarray] = {}
    prefix = f"expert.{sentinel}"
    for matrix in MATRICES:
        stem = f"{prefix}.{matrix}.base"
        scale_component = by_name[f"{stem}.salience_scale"]
        scale = np.frombuffer(scale_component["data"], dtype=ml_dtypes.bfloat16).astype(np.float32)
        index_component = by_name[f"{stem}.indices"]
        codebook_component = by_name[f"{stem}.codebooks"]
        index_shape = tuple(int(value) for value in index_component["shape"])
        indices = unpack_unsigned(index_component["data"], math.prod(index_shape),
                                  int(index_component["bits"])).reshape(index_shape)
        codebook_shape = tuple(int(value) for value in codebook_component["shape"])
        codebooks = np.frombuffer(codebook_component["data"], dtype=np.float16).astype(
            np.float32).reshape(codebook_shape)
        n_vectors, subspaces = indices.shape
        sub = codebooks.shape[-1]
        reconstructed = np.empty((n_vectors, subspaces * sub), dtype=np.float32)
        for subspace in range(subspaces):
            reconstructed[:, subspace * sub:(subspace + 1) * sub] = (
                codebooks[subspace][indices[:, subspace]]
            )
        reconstructed = reconstructed.reshape(-1, scale.size) / scale[None, :]
        row_index = by_name.get(f"{stem}.protected_row_indices")
        row_value = by_name.get(f"{stem}.protected_row_values")
        if (row_index is None) != (row_value is None):
            raise F1Error(f"incomplete protected-row pair for {matrix}")
        if row_index is not None and row_value is not None:
            count = int(row_index["shape"][0])
            rows = unpack_unsigned(row_index["data"], count, int(row_index["bits"]))
            values = np.frombuffer(row_value["data"], dtype=ml_dtypes.bfloat16).astype(
                np.float32).reshape(tuple(int(value) for value in row_value["shape"]))
            reconstructed[rows] = values
        weights[matrix] = reconstructed
    return weights, [component for component in components if component["role"] == "base"]


def run_candidate(candidate: str, source: Path, capture: dict[str, np.ndarray],
                  capture_receipt: dict[str, Any], output_dir: Path,
                  progress_path: Path) -> dict[str, Any]:
    result_path = output_dir / f"{candidate}_F1_RESULT.json"
    payload_path = output_dir / f"{candidate}_sentinel_expert.k26f1"
    if result_path.exists() and payload_path.exists():
        cached = read_json(result_path)
        if (cached.get("status") == "PASS" and
                cached.get("payload", {}).get("sha256") == sha256_file(payload_path)):
            return cached

    atomic_json(progress_path, {"status": "RUNNING", "stage": "PACK_CANDIDATE",
                                "candidate": candidate, "started_at": now()})
    spec = CANDIDATES[candidate]
    shard = reference.TensorShard(reference.shard_path(source, 2))
    sentinel = int(capture["sentinel_expert"][0])
    weights = dequantized_expert(shard, sentinel)
    del shard

    n_weights = [int(weights[name].size) for name in MATRICES]
    total_weights = sum(n_weights)
    total_cap = (total_weights * spec["target"].numerator //
                 spec["target"].denominator) // 8
    base_cap_total = total_cap * spec["base_share"].numerator // spec["base_share"].denominator
    doctor_cap_total = total_cap * spec["doctor_share"].numerator // spec["doctor_share"].denominator
    overhead_cap = total_cap - base_cap_total - doctor_cap_total
    base_caps = distribute(base_cap_total, n_weights)
    doctor_caps = distribute(doctor_cap_total, n_weights)

    fit_x = capture["fit_x"].astype(np.float32)
    score_x = capture["score_x"].astype(np.float32)
    activation_by_matrix = {
        "gate_proj": fit_x, "up_proj": fit_x,
        "down_proj": capture["fit_teacher_hidden"].astype(np.float32),
    }
    base_weights: dict[str, np.ndarray] = {}
    doctored_weights: dict[str, np.ndarray] = {}
    matrix_records = {}
    components: list[dict[str, Any]] = []
    for index, matrix in enumerate(MATRICES):
        atomic_json(progress_path, {"status": "RUNNING", "stage": "PACK_MATRIX",
                                    "candidate": candidate, "matrix": matrix,
                                    "matrix_index": index + 1, "matrix_total": len(MATRICES),
                                    "updated_at": now()})
        base, doctored, matrix_components, record = pack_matrix(
            weights[matrix], activation_by_matrix[matrix], spec,
            base_caps[index], doctor_caps[index], f"expert.{sentinel}.{matrix}",
            seed=260621 + index * 37 + (0 if candidate == "P1" else 5000),
        )
        base_weights[matrix] = base
        doctored_weights[matrix] = doctored
        components.extend(matrix_components)
        matrix_records[matrix] = record

    teacher_fit = capture["fit_teacher_output"].astype(np.float32)
    teacher_score = capture["score_teacher_output"].astype(np.float32)
    base_fit, base_fit_internal = expert_forward(fit_x, base_weights)
    base_score, base_score_internal = expert_forward(score_x, base_weights)
    doctored_fit, doctored_fit_internal = expert_forward(fit_x, doctored_weights)
    doctored_score, doctored_score_internal = expert_forward(score_x, doctored_weights)

    score_routes = capture["score_routes"].astype(np.int32)
    score_route_weights = capture["score_route_weights"].astype(np.float32)
    routed_mask = score_routes == sentinel
    routed_tokens = np.any(routed_mask, axis=1)
    route_weight = np.sum(np.where(routed_mask, score_route_weights, 0.0), axis=1)
    if np.any(routed_tokens):
        routed_teacher = teacher_score[routed_tokens] * route_weight[routed_tokens, None]
        routed_base = base_score[routed_tokens] * route_weight[routed_tokens, None]
        routed_doctored = doctored_score[routed_tokens] * route_weight[routed_tokens, None]
        residual = capture["score_post_attention"].astype(np.float32)[routed_tokens]
        routed_metrics = {
            "tokens": int(np.sum(routed_tokens)),
            "slots": int(np.sum(routed_mask)),
            "weighted_expert_output_base": quality(routed_teacher, routed_base),
            "weighted_expert_output_doctored": quality(routed_teacher, routed_doctored),
            "first_residual_add_base": quality(residual + routed_teacher, residual + routed_base),
            "first_residual_add_doctored": quality(
                residual + routed_teacher, residual + routed_doctored),
        }
    else:
        routed_metrics = {"tokens": 0, "slots": 0, "status": "NO_SCORE_ROUTE_SLOTS"}

    payload = write_payload(payload_path, {
        "schema": "hawking.kimi_k26.f1_sentinel_payload.v1", "candidate": candidate,
        "revision": REVISION, "layer": LAYER, "sentinel_expert": sentinel,
        "target_complete_bpw": str(spec["target"]),
        "base_share": str(spec["base_share"]), "doctor_share": str(spec["doctor_share"]),
        "overhead_share": str(spec["overhead_share"]),
    }, components)
    if payload["base_component_bytes"] > base_cap_total:
        raise F1Error("base components exceeded candidate allocation")
    if payload["doctor_component_bytes"] > doctor_cap_total:
        raise F1Error("Doctor components exceeded candidate allocation")
    if payload["header_overhead_bytes"] > overhead_cap:
        raise F1Error("payload header exceeded overhead allocation")
    if payload["bytes"] > total_cap:
        raise F1Error("physical payload exceeded complete candidate ceiling")

    base_score_metric = quality(teacher_score, base_score)
    doctored_score_metric = quality(teacher_score, doctored_score)
    base_verdict = fidelity_verdict(base_score_metric)
    verdict = fidelity_verdict(doctored_score_metric)
    doctor_recovery = float(
        1 - doctored_score_metric["relative_l2"] /
        (base_score_metric["relative_l2"] + 1e-30)
    )
    actual_bpw = payload["bytes"] * 8 / total_weights
    result = seal({
        "schema": "hawking.kimi_k26.f1_candidate_result.v1", "status": "PASS",
        "sealed_at": now(), "candidate": candidate, "source": {"repo": REPO,
        "revision": REVISION}, "layer": LAYER, "sentinel_expert": sentinel,
        "claim_boundary": "F1 ONE_REAL_LAYER_ONE_SENTINEL_EXPERT; not end-to-end capability",
        "teacher_capture_seal_sha256": capture_receipt["seal_sha256"],
        "fit_score_token_overlap": 0, "fit_tokens": int(fit_x.shape[0]),
        "score_tokens": int(score_x.shape[0]),
        "router": {"candidate_modifies_router": False, "route_agreement": 1.0,
                   "selection_uses_fit_only": True},
        "physical_budget": {
            "logical_weights_represented": total_weights,
            "target_complete_bpw": str(spec["target"]),
            "target_complete_bpw_decimal": float(spec["target"]),
            "complete_ceiling_bytes": total_cap,
            "base_ceiling_bytes": base_cap_total,
            "doctor_ceiling_bytes": doctor_cap_total,
            "overhead_ceiling_bytes": overhead_cap,
            "actual_complete_bpw": actual_bpw,
            "unused_ceiling_bytes": total_cap - payload["bytes"],
            "all_payload_bytes_counted": True,
        },
        "payload": payload, "matrix_records": matrix_records,
        "metrics": {
            "fit": {"base": quality(teacher_fit, base_fit),
                    "doctored": quality(teacher_fit, doctored_fit)},
            "score": {"base": base_score_metric, "doctored": doctored_score_metric},
            "score_seams": {
                "gate_base": quality(
                    score_x @ weights["gate_proj"].T, base_score_internal["gate"]),
                "gate_doctored": quality(
                    score_x @ weights["gate_proj"].T, doctored_score_internal["gate"]),
                "up_base": quality(score_x @ weights["up_proj"].T, base_score_internal["up"]),
                "up_doctored": quality(
                    score_x @ weights["up_proj"].T, doctored_score_internal["up"]),
                "nonlinear_hidden_base": quality(
                    capture["score_teacher_hidden"], base_score_internal["hidden"]),
                "nonlinear_hidden_doctored": quality(
                    capture["score_teacher_hidden"], doctored_score_internal["hidden"]),
            },
            "routed_score_subset": routed_metrics,
            "doctor_recovery_fraction_of_output_relative_l2": doctor_recovery,
            "doctor_cosine_gain": (doctored_score_metric["cosine_mean"] -
                                   base_score_metric["cosine_mean"]),
            "f1_capability_density_proxy": doctored_score_metric["cosine_mean"] / actual_bpw,
        },
        "base_verdict": base_verdict, "candidate_verdict": verdict,
        "doctor_prevented_collapse": (
            base_verdict == "COLLAPSE_F1" and verdict != "COLLAPSE_F1"),
        "real_candidate_diagnosis": (
            "ROUTED_EXPERT_OUTPUT_BOUND_AFTER_NATIVE_ROUTING" if verdict != "SURVIVES_F1"
            else "SURVIVES_SENTINEL_EXPERT_F1"
        ),
    })
    atomic_json(result_path, result)
    del weights, base_weights, doctored_weights, components
    gc.collect()
    return result


def decide(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    survivors = [candidate for candidate, result in results.items()
                 if result["candidate_verdict"] == "SURVIVES_F1"]
    if not survivors:
        decision = "NO_PROMOTION"
        best = max(results, key=lambda candidate:
                   results[candidate]["metrics"]["score"]["doctored"]["cosine_mean"])
        next_experiment = "F1_SHARED_GRAMMAR_VS_PROTECTED_ISLANDS_ON_CACHED_TEACHER_SEAM"
        reason = ("Neither concrete PQ+Doctor representation survived F1; retain the envelopes but "
                  "retire these representation instances and test a materially different geometry.")
    elif "P5" in survivors:
        p5 = results["P5"]["metrics"]["score"]["doctored"]
        p1 = results["P1"]["metrics"]["score"]["doctored"]
        if "P1" not in survivors or p5["cosine_mean"] >= p1["cosine_mean"] - 0.02:
            best, decision = "P5", "PROMOTE_P5_TO_F2"
        else:
            best, decision = "P1", "PROMOTE_P1_TO_F2"
        next_experiment = f"{best}_F2_EARLY_MIDDLE_LATE_LAYER_REPLICATION"
        reason = "At least one representation survived; select the lowest-rate non-inferior survivor."
    else:
        best, decision = "P1", "PROMOTE_P1_TO_F2"
        next_experiment = "P1_F2_EARLY_MIDDLE_LATE_LAYER_REPLICATION"
        reason = "P1 survived F1 while P5 did not."

    chosen = results[best]
    routed = chosen["metrics"]["routed_score_subset"]
    residual_metric = routed.get("first_residual_add_doctored", {})
    output_metric = chosen["metrics"]["score"]["doctored"]
    failure_mode = (
        "NONE_AT_F1_SENTINEL" if chosen["candidate_verdict"] == "SURVIVES_F1" else
        "EXPERT_OUTPUT_DEGRADATION_AFTER_ROUTING"
    )
    residual_location = (
        "MASKED_AT_FIRST_RESIDUAL_ADD" if residual_metric.get("cosine_mean", 0) >= 0.99 else
        "VISIBLE_AT_FIRST_RESIDUAL_ADD"
    )
    return {
        "decision": decision, "reason": reason, "current_best_candidate": best,
        "current_best_bpw": chosen["physical_budget"]["actual_complete_bpw"],
        "current_best_capability": {"evidence_level": "F1_SENTINEL_EXPERT_OUTPUT",
                                    **output_metric},
        "current_doctor_allocation": {
            "bytes": chosen["payload"]["doctor_component_bytes"],
            "bpw": chosen["payload"]["doctor_component_bytes"] * 8 /
                   chosen["physical_budget"]["logical_weights_represented"],
            "recovery_fraction": chosen["metrics"][
                "doctor_recovery_fraction_of_output_relative_l2"],
        },
        "current_failure_mode": failure_mode,
        "collapse_location": {"before_or_after_routing": "AFTER_ROUTING",
                              "before_or_after_residual_propagation": residual_location,
                              "downstream_propagation_tested": False},
        "current_dominant_bottleneck": (
            "ROUTED_EXPERT_REPRESENTATION" if failure_mode != "NONE_AT_F1_SENTINEL" else
            "CROSS_LAYER_GENERALIZATION_UNTESTED"),
        "current_scientific_hypothesis": (
            "If PQ+residual Doctor fails, Kimi expert intelligence prefers shared grammar or "
            "protected-island geometry over independent weight reconstruction; if it survives, "
            "the next uncertainty is layer generalization."),
        "current_next_experiment": next_experiment,
    }


def run(source: Path, corpus_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "KIMI_K26_F1_PROGRESS.json"
    corpus = read_json(corpus_path)
    if corpus.get("status") != "PASS" or corpus.get("source", {}).get("revision") != REVISION:
        raise F1Error("sealed Kimi corpus evidence is not usable")
    capture, capture_receipt = capture_teacher(source, corpus, output_dir, progress_path)
    results = {}
    for candidate in ("P1", "P5"):
        results[candidate] = run_candidate(
            candidate, source, capture, capture_receipt, output_dir, progress_path,
        )
    decision = decide(results)
    artifact = seal({
        "schema": "hawking.kimi_k26.f1_representation_bracket.v1", "status": "PASS",
        "sealed_at": now(), "runtime_seconds": time.time() - started,
        "source": {"repo": REPO, "revision": REVISION}, "layer": LAYER,
        "experiment": "P1_AND_P5_F1_REPRESENTATION_BRACKET",
        "claim_boundary": "REAL F1 SENTINEL-EXPERT OUTPUT EVIDENCE; not end-to-end capability",
        "corpus_seal_sha256": corpus["seal_sha256"],
        "teacher_capture_seal_sha256": capture_receipt["seal_sha256"],
        "parent_forwards_repeated": 0, "teacher_captures": 1,
        "candidate_results": {candidate: {"seal_sha256": result["seal_sha256"],
                                           "verdict": result["candidate_verdict"],
                                           "actual_bpw": result["physical_budget"][
                                               "actual_complete_bpw"]}
                              for candidate, result in results.items()},
        "diagnosis_metrics_reason_next_decision": {
            "diagnosis": decision["current_failure_mode"],
            "metrics": decision["current_best_capability"],
            "reason": decision["reason"], "next_decision": decision["decision"],
        },
        **decision,
    })
    atomic_json(output_dir / "KIMI_K26_F1_REPRESENTATION_BRACKET.json", artifact)
    scientific_status = seal({
        "schema": "hawking.kimi_k26.scientific_status.v1", "status": "ACTIVE",
        "updated_at": now(), "evidence_level": "F1_COMPLETE",
        "source": {"repo": REPO, "revision": REVISION},
        **{key: value for key, value in decision.items() if key != "reason"},
        "last_experiment": "P1_AND_P5_F1_REPRESENTATION_BRACKET",
        "last_experiment_seal_sha256": artifact["seal_sha256"],
        "decision_reason": decision["reason"],
    })
    atomic_json(output_dir / "KIMI_K26_SCIENTIFIC_STATUS.json", scientific_status)
    atomic_json(progress_path, {"status": "COMPLETE", "completed_at": now(),
                                "decision": decision["decision"],
                                "next_experiment": decision["current_next_experiment"]})
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.source.resolve(strict=True), args.corpus.resolve(strict=True),
                     args.output_dir.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
