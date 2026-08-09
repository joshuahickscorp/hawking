"""Detached, physical Qwen30 Gravity discovery worker for Ascension.

This is intentionally a *worker*, not another controller.  It operates on the
locally present official Qwen3-Coder-30B BF16 shards, first hashing every shard
to bind the source body and then writing real, packed low-rank/quantized
candidate artifacts for selected MoE tensors.  It never emits a protected V3
receipt, promotes a candidate, starts a full-model runtime, or claims a
complete-model BPW/TG result.

The worker is restart-safe: source hashes and every candidate measurement are
durable, so a launchd restart resumes at the next unmeasured candidate instead
of repeating finished disk-heavy work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import struct
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map
from lab.receipts import seal


SCHEMA = "hawking.ascension.qwen30_physical_campaign.v1"
SOURCE_AUDIT_SCHEMA = "hawking.ascension.qwen30_source_body_audit_candidate.v1"
CANDIDATE_SCHEMA = "hawking.ascension.qwen30_low_rank_gravity_candidate.v1"
MAGIC = b"HAWKQ30G\0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30"
SOURCE_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
GROUP_SIZE = 64


@dataclass(frozen=True)
class CandidateSpec:
    rank: int
    bits: int

    @property
    def identifier(self) -> str:
        return f"r{self.rank}_b{self.bits}"


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(rank=48, bits=4),
    CandidateSpec(rank=64, bits=4),
    CandidateSpec(rank=96, bits=3),
    CandidateSpec(rank=128, bits=3),
)
LAYER_SCHEDULE = (0, 12, 24, 36, 47)
EXPERT_SCHEDULE = (0, 1, 7, 63, 127)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# The first rung is deliberately capped.  It is enough to expose a broad
# source/representation frontier but prevents an unbounded queue of component
# experiments from postponing the one complete-artifact path.
SCOUT_CANDIDATE_BUDGET = 150
FRONTIER_SURVIVOR_COUNT = 12


class PhysicalCampaignError(RuntimeError):
    """The worker cannot safely continue a physical candidate run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _sha256_file(path: Path, progress: callable) -> tuple[str, int]:
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
            progress(observed)
    return digest.hexdigest(), observed


def _shards(model_dir: Path) -> tuple[dict[str, str], ...]:
    index_path = model_dir / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(dict(index["weight_map"]).values()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PhysicalCampaignError(f"cannot load source shard index: {exc}") from exc
    rows: list[dict[str, str]] = []
    for name in names:
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            raise PhysicalCampaignError(f"invalid source shard name: {name!r}")
        if not (model_dir / name).is_file():
            raise PhysicalCampaignError(f"missing source shard: {model_dir / name}")
        rows.append({"name": name})
    return tuple(rows)


def _target_tensors(weight_map: Mapping[str, str]) -> tuple[str, ...]:
    names: list[str] = []
    for layer in LAYER_SCHEDULE:
        for expert in EXPERT_SCHEDULE:
            for projection in PROJECTIONS:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                if name in weight_map:
                    names.append(name)
    if not names:
        raise PhysicalCampaignError("no expected Qwen30 MoE tensors were found in the source index")
    return tuple(names)


def _seed_for(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _low_rank_factors(weights: np.ndarray, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute a deterministic randomized low-rank factorisation of real BF16 weights."""

    matrix = np.ascontiguousarray(weights, dtype=np.float32)
    rows, columns = matrix.shape
    if rank <= 0 or rank > min(rows, columns):
        raise PhysicalCampaignError(f"rank {rank} is invalid for shape {matrix.shape}")
    oversample = min(12, min(rows, columns) - rank)
    generator = np.random.default_rng(seed)
    probe = generator.standard_normal((columns, rank + oversample), dtype=np.float32)
    basis, _ = np.linalg.qr(matrix @ probe, mode="reduced")
    small = basis.T @ matrix
    left, singular, right = np.linalg.svd(small, full_matrices=False)
    left = basis @ left[:, :rank]
    # Fold singular values into U so the packed artifact has exactly two factors.
    return np.ascontiguousarray(left * singular[:rank], dtype=np.float32), np.ascontiguousarray(
        right[:rank, :], dtype=np.float32
    )


def _pack_factor(values: np.ndarray, bits: int) -> tuple[dict[str, Any], bytes, np.ndarray]:
    """Bit-pack a factor with billed FP16 group scales and reconstruct it."""

    if bits < 2 or bits > 8:
        raise PhysicalCampaignError("factor quantization bits must be in [2, 8]")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / GROUP_SIZE)
    padded = np.pad(flat, (0, groups * GROUP_SIZE - flat.size))
    grouped = padded.reshape(groups, GROUP_SIZE)
    max_code = (1 << (bits - 1)) - 1
    # Keep the stored FP16 scale exactly as the artifact will decode it, but
    # never use an FP16-underflowed zero as a divisor while producing codes.
    # An all-zero group has zero codes regardless of the finite normalizer;
    # underflowed near-zero groups are deliberately reconstructed at the
    # stored precision and their reconstruction error is measured below.
    scales_f32 = np.max(np.abs(grouped), axis=1) / max_code
    scales = scales_f32.astype(np.float16)
    code_normalizer = np.where(scales_f32 > 0.0, scales_f32, 1.0).astype(np.float32)
    codes = np.rint(grouped / code_normalizer[:, None]).clip(-max_code, max_code).astype(np.int16)
    unsigned = (codes.reshape(-1) + max_code).astype(np.uint8)
    bit_matrix = ((unsigned[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    packed = np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()
    unpacked_bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[: unsigned.size * bits]
    decoded = (unpacked_bits.reshape(-1, bits) * (1 << np.arange(bits, dtype=np.uint8))).sum(axis=1)
    restored = (decoded.astype(np.int16) - max_code).astype(np.float32)
    restored = (restored.reshape(groups, GROUP_SIZE) * scales.astype(np.float32)[:, None]).reshape(-1)[: flat.size]
    metadata = {
        "shape": list(values.shape),
        "elements": int(flat.size),
        "bits": bits,
        "group_size": GROUP_SIZE,
        "groups": groups,
        "code_bytes": len(packed),
        "scale_bytes": int(scales.nbytes),
    }
    return metadata, packed + scales.tobytes(), restored.reshape(values.shape)


def _candidate_artifact(
    *,
    tensor_name: str,
    source_sha256: str,
    source_shape: tuple[int, ...],
    spec: CandidateSpec,
    left: np.ndarray,
    right: np.ndarray,
    path: Path,
) -> tuple[int, np.ndarray]:
    left_meta, left_payload, left_restored = _pack_factor(left, spec.bits)
    right_meta, right_payload, right_restored = _pack_factor(right, spec.bits)
    header = {
        "schema": "hawking.ascension.qwen30_packed_low_rank_artifact.v1",
        "tensor_name": tensor_name,
        "source_sha256": source_sha256,
        "source_shape": list(source_shape),
        "representation": "randomized_low_rank_plus_group_quantized_factors",
        "rank": spec.rank,
        "factor_bits": spec.bits,
        "left": left_meta,
        "right": right_meta,
        "claim_boundary": {
            "not_a_full_model_artifact": True,
            "not_a_native_metal_execution_plan": True,
            "not_a_capability_or_tg_qualification": True,
        },
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + left_payload + right_payload
    _atomic_bytes(path, payload)
    return len(payload), left_restored @ right_restored


def _quality(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    source = np.ascontiguousarray(reference, dtype=np.float32).reshape(-1)
    restored = np.ascontiguousarray(reconstruction, dtype=np.float32).reshape(-1)
    delta = restored - source
    denom = max(float(np.linalg.norm(source)), 1e-12)
    return {
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "cosine": float(np.dot(source, restored) / max(float(np.linalg.norm(source) * np.linalg.norm(restored)), 1e-12)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _teacher_train_low_rank(
    weights: np.ndarray, left: np.ndarray, right: np.ndarray, *, steps: int = 16
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run bounded teacher-distillation on an actual BF16 tensor.

    This is deliberately a component-level Doctor experiment.  The source
    tensor is the teacher and the packed low-rank factors are the student;
    neither the loss nor a Metal component result is allowed to stand in for a
    full-model capability result.
    """

    try:
        import torch
    except ImportError:
        return left, right, {"status": "UNAVAILABLE", "reason": "torch_not_installed"}
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    try:
        torch.manual_seed(_seed_for(f"doctor:{weights.shape}:{left.shape[1]}"))
        target = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)).to(device)
        first = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(left)).to(device))
        second = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(right)).to(device))
        optimizer = torch.optim.AdamW((first, second), lr=7e-4, weight_decay=1e-6)
        with torch.no_grad():
            initial_loss = float(torch.mean(torch.square(first @ second - target)).item())
        losses: list[float] = []
        started = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.square(first @ second - target))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        if device.type == "mps":
            torch.mps.synchronize()
        return (
            first.detach().to("cpu").float().numpy(),
            second.detach().to("cpu").float().numpy(),
            {
                "status": "RAN",
                "kind": "teacher_distilled_low_rank_doctor",
                "device": device.type,
                "steps": steps,
                "initial_mse": initial_loss,
                "final_mse": losses[-1] if losses else initial_loss,
                "elapsed_seconds": time.perf_counter() - started,
                "claim_boundary": "component-only teacher distillation; not full-model QAT or capability training",
            },
        )
    except Exception as exc:
        return left, right, {"status": "FAILED", "device": device.type, "reason": type(exc).__name__}


def _metal_probe(weights: np.ndarray) -> dict[str, Any]:
    """Run a small real MPS matmul if the installed Torch exposes Metal."""

    try:
        import torch
    except ImportError:
        return {"status": "UNAVAILABLE", "reason": "torch_not_installed"}
    if not torch.backends.mps.is_available():
        return {"status": "UNAVAILABLE", "reason": "mps_not_available"}
    try:
        device = torch.device("mps")
        matrix = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float16)).to(device)
        vector = torch.ones((1, matrix.shape[1]), dtype=torch.float16, device=device)
        for _ in range(3):
            _ = vector @ matrix.T
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            _ = vector @ matrix.T
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "status": "RAN",
            "device": "mps",
            "iterations": 20,
            "elapsed_seconds": elapsed,
            "matmul_per_second": 20.0 / max(elapsed, 1e-12),
            "shape": list(weights.shape),
            "claim_boundary": "component-only Metal smoke; not a complete-token TG measurement",
        }
    except Exception as exc:
        return {"status": "FAILED", "reason": type(exc).__name__}


class Qwen30PhysicalCampaign:
    def __init__(self, *, model_dir: Path, root: Path) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.root = root.expanduser().resolve()
        self.status_path = self.root / "QWEN30_REAL_CAMPAIGN_STATUS.json"
        self.source_audit_path = self.root / "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
        self.candidate_dir = self.root / "candidates"
        self.artifact_dir = self.root / "artifacts"
        self.frontier_path = self.root / "QWEN30_GRAVITY_FRONTIER.json"
        self.gravity_gene_pool_path = self.root.parent / "qwen-family" / "QWEN_GRAVITY_GENE_POOL.json"
        self.kernel_gene_pool_path = self.root.parent / "qwen-family" / "QWEN_KERNEL_GENE_POOL.json"
        self._stopping = False

    def _state(self) -> dict[str, Any]:
        existing = _read_json(self.status_path) or {}
        return {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(existing.get("heartbeat", 0)) + 1,
            "model": {
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                "local_path": str(self.model_dir),
            },
            "source_audit": existing.get("source_audit", {}),
            "experiment": existing.get("experiment", {"completed_candidates": 0}),
            "resource_owner": existing.get("resource_owner", {}),
            "claim_boundary": {
                "all_outputs_are_candidate_research_evidence": True,
                "does_not_certify_qwen30_manager": True,
                "does_not_claim_full_model_complete_bpw": True,
                "does_not_claim_tg3_or_hcli_qualification": True,
                "does_not_select_a_tournament_winner": True,
            },
        }

    def _publish(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.status_path, state)

    def _source_audit(self, state: dict[str, Any]) -> None:
        prior = state.get("source_audit") if isinstance(state.get("source_audit"), Mapping) else {}
        completed = prior.get("shards") if isinstance(prior.get("shards"), Mapping) else {}
        rows = _shards(self.model_dir)
        for row in rows:
            name = row["name"]
            source = self.model_dir / name
            known = completed.get(name) if isinstance(completed, Mapping) else None
            if isinstance(known, Mapping) and known.get("bytes") == source.stat().st_size and isinstance(known.get("sha256"), str):
                continue
            state["phase"] = "SOURCE_AUDIT_RUNNING"
            state["resource_owner"] = {"disk_read": "QWEN30_SOURCE_BODY_AUDIT", "cpu": "sha256"}
            state["source_audit"] = {
                "status": "RUNNING",
                "current_shard": name,
                "shards": dict(completed),
                "source_body_complete": False,
            }
            self._publish(state)

            def progress(observed: int) -> None:
                state["heartbeat"] = int(state["heartbeat"]) + 1
                state["source_audit"] = {
                    "status": "RUNNING",
                    "current_shard": name,
                    "current_shard_bytes_hashed": observed,
                    "current_shard_total_bytes": source.stat().st_size,
                    "shards": dict(completed),
                    "source_body_complete": False,
                }
                self._publish(state)

            digest, observed = _sha256_file(source, progress)
            completed = dict(completed)
            completed[name] = {"bytes": observed, "sha256": digest}
            state["heartbeat"] = int(state["heartbeat"]) + 1
        total = sum(int(item["bytes"]) for item in completed.values() if isinstance(item, Mapping))
        audit = seal(
            {
                "schema": SOURCE_AUDIT_SCHEMA,
                "status": "CANDIDATE_SOURCE_BODY_VERIFIED",
                "recorded_at": _utc_now(),
                "source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION,
                    "model_dir": str(self.model_dir),
                    "shards": completed,
                    "total_bytes": total,
                    "shard_count": len(completed),
                },
                "claim_boundary": {
                    "source_audit_is_not_protected_manager_source_authority": True,
                    "does_not_certify_density_tg_or_capability": True,
                    "does_not_expose_credentials": True,
                },
            }
        )
        _atomic_json(self.source_audit_path, audit)
        state["source_audit"] = {
            "status": "COMPLETE",
            "source_body_complete": True,
            "shards": completed,
            "total_bytes": total,
            "audit_path": str(self.source_audit_path),
            "audit_seal_sha256": audit["seal_sha256"],
        }
        state["resource_owner"] = {"disk_read": "RELEASED", "cpu": "READY_FOR_GRAVITY_CANDIDATE"}
        self._publish(state)

    def _completed_candidate_ids(self) -> set[str]:
        return {path.stem for path in self.candidate_dir.glob("*.json") if _read_json(path) is not None}

    def _publish_frontier_and_gene_pools(self, state: dict[str, Any]) -> None:
        """Successive-halve measured candidates without deleting any evidence.

        This selection is a research scheduler, not a promotion.  Every
        discarded candidate remains sealed on disk; it is simply parked so the
        complete-artifact lane can work from a bounded current frontier.
        """

        rows: list[dict[str, Any]] = []
        for path in sorted(self.candidate_dir.glob("*.json")):
            document = _read_json(path)
            if document is None:
                continue
            measurement = document.get("measurement") if isinstance(document.get("measurement"), Mapping) else {}
            representation = document.get("representation") if isinstance(document.get("representation"), Mapping) else {}
            cosine = float(measurement.get("cosine", -1.0))
            relative_l2 = float(measurement.get("relative_l2", float("inf")))
            bpw = float(representation.get("component_complete_bpw", float("inf")))
            if not (math.isfinite(cosine) and math.isfinite(relative_l2) and math.isfinite(bpw)):
                continue
            # Fidelity dominates; BPW breaks otherwise close candidates.  This
            # is deterministic and deliberately not a capability score.
            score = cosine - relative_l2 - 0.01 * bpw
            rows.append(
                {
                    "candidate_id": document.get("candidate_id"),
                    "candidate_result_path": str(path),
                    "candidate_seal_sha256": document.get("seal_sha256"),
                    "score": score,
                    "relative_l2": relative_l2,
                    "cosine": cosine,
                    "component_complete_bpw": bpw,
                    "rank": representation.get("rank"),
                    "factor_bits": representation.get("factor_bits"),
                    "tensor_name": (document.get("source") or {}).get("tensor_name"),
                    "metal_component_probe": document.get("metal_component_probe"),
                }
            )
        rows.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
        survivors = rows[:FRONTIER_SURVIVOR_COUNT]
        parked = rows[FRONTIER_SURVIVOR_COUNT:]
        frontier = seal(
            {
                "schema": "hawking.ascension.qwen30_gravity_successive_halving_frontier.v1",
                "status": "COMPONENT_FRONTIER_SELECTED_NOT_FULL_ARTIFACT",
                "recorded_at": _utc_now(),
                "source_body_audit_path": str(self.source_audit_path),
                "scout_candidate_budget": SCOUT_CANDIDATE_BUDGET,
                "completed_candidates": len(rows),
                "survivor_count": len(survivors),
                "survivors": survivors,
                "parked_candidate_count": len(parked),
                "parked_candidate_ids": [row["candidate_id"] for row in parked],
                "claim_boundary": {
                    "preserves_all_loser_evidence_without_deletion": True,
                    "selection_is_not_full_model_artifact_or_complete_bpw": True,
                    "selection_is_not_native_runtime_tg3_or_manager_promotion": True,
                },
            }
        )
        _atomic_json(self.frontier_path, frontier)
        gravity_gene_pool = seal(
            {
                "schema": "hawking.ascension.qwen_gravity_gene_pool.v1",
                "status": "QWEN30_PRIORS_AVAILABLE_FOR_QWEN80_RESEARCH",
                "recorded_at": _utc_now(),
                "source_frontier_seal_sha256": frontier["seal_sha256"],
                "transfer_scope": [
                    "low_rank_geometry", "factor_bit_width", "group_scale_layout",
                    "component_fidelity", "teacher_distillation_response",
                ],
                "priors": survivors,
                "claim_boundary": {
                    "priors_are_not_cross_model_qualification": True,
                    "qwen80_hybrid_state_and_512_expert_differences_require_own_evidence": True,
                },
            }
        )
        _atomic_json(self.gravity_gene_pool_path, gravity_gene_pool)
        kernel_rows = [
            {
                "candidate_id": row["candidate_id"],
                "metal_component_probe": row["metal_component_probe"],
            }
            for row in survivors
        ]
        route_probe = _read_json(self.root.parent / "kernel" / "QWEN_DUAL_ROUTE_METAL_COMPONENT_PROBE.json")
        gqa_probe = _read_json(self.root.parent / "kernel" / "QWEN30_GQA_METAL_COMPONENT_PROBE.json")
        kernel_gene_pool = seal(
            {
                "schema": "hawking.ascension.qwen_kernel_gene_pool.v1",
                "status": "QWEN30_PRIORS_AVAILABLE_FOR_QWEN80_RESEARCH",
                "recorded_at": _utc_now(),
                "source_frontier_seal_sha256": frontier["seal_sha256"],
                "transfer_scope": [
                    "router_topk", "packed_decode_geometry", "expert_projection_components",
                    "qwen30_gqa_attention_geometry", "mps_metal_component_measurements",
                    "command_buffer_dispatch_shape",
                ],
                "qwen30_survivor_component_probes": kernel_rows,
                "direct_router_probe": route_probe,
                "direct_gqa_attention_probe": gqa_probe,
                "claim_boundary": {
                    "component_kernel_priors_are_not_full_token_tps": True,
                    "qwen80_deltanet_and_hybrid_attention_need_distinct_kernel_evidence": True,
                },
            }
        )
        _atomic_json(self.kernel_gene_pool_path, kernel_gene_pool)
        state["experiment"] = {
            "status": "SUCCESSIVE_HALVING_COMPLETE",
            "completed_candidates": len(rows),
            "survivors": len(survivors),
            "frontier_path": str(self.frontier_path),
            "gravity_gene_pool_path": str(self.gravity_gene_pool_path),
            "kernel_gene_pool_path": str(self.kernel_gene_pool_path),
            "next_required_lane": "QWEN30_COMPLETE_NATIVE_ARTIFACT",
        }
        state["phase"] = "SUCCESSIVE_HALVING_COMPLETE_AWAITING_COMPLETE_ARTIFACT_LANE"
        state["resource_owner"] = {"cpu": "RELEASED_TO_COMPLETE_RUNTIME_LANE", "gpu": "RELEASED_TO_TG3_LANE"}
        self._publish(state)

    def _run_candidate(
        self, *, state: dict[str, Any], tensor_name: str, spec: CandidateSpec, source_sha256: str
    ) -> None:
        candidate_id = f"{tensor_name.replace('.', '_')}__{spec.identifier}"
        result_path = self.candidate_dir / f"{candidate_id}.json"
        artifact_path = self.artifact_dir / f"{candidate_id}.gravity"
        state["phase"] = "GRAVITY_CANDIDATE_RUNNING"
        state["resource_owner"] = {
            "cpu": "QWEN30_RANDOMIZED_LOW_RANK_FACTORIZATION",
            "gpu": "QWEN30_COMPONENT_METAL_SMOKE_AFTER_FACTORISATION",
            "disk_write": "PACKED_CANDIDATE_ARTIFACT",
        }
        state["experiment"] = {
            "status": "RUNNING",
            "current_candidate": candidate_id,
            "tensor": tensor_name,
            "rank": spec.rank,
            "bits": spec.bits,
            "completed_candidates": len(self._completed_candidate_ids()),
        }
        self._publish(state)
        weights = load_tensor(self.model_dir, load_weight_map(self.model_dir), tensor_name)
        started = time.perf_counter()
        left, right = _low_rank_factors(weights, spec.rank, _seed_for(candidate_id))
        trained_left, trained_right, training = _teacher_train_low_rank(weights, left, right)
        artifact_bytes, reconstruction = _candidate_artifact(
            tensor_name=tensor_name,
            source_sha256=source_sha256,
            source_shape=tuple(weights.shape),
            spec=spec,
            left=trained_left,
            right=trained_right,
            path=artifact_path,
        )
        metrics = _quality(weights, reconstruction)
        metal = _metal_probe(weights)
        candidate = seal(
            {
                "schema": CANDIDATE_SCHEMA,
                "status": "CANDIDATE_RESEARCH_ONLY",
                "recorded_at": _utc_now(),
                "candidate_id": candidate_id,
                "source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION,
                    "source_body_audit_seal_sha256": source_sha256,
                    "tensor_name": tensor_name,
                    "shape": list(weights.shape),
                },
                "representation": {
                    "kind": "randomized_low_rank_plus_group_quantized_factors",
                    "rank": spec.rank,
                    "factor_bits": spec.bits,
                    "group_size": GROUP_SIZE,
                    "artifact_path": str(artifact_path),
                    "artifact_bytes": artifact_bytes,
                    "component_complete_bpw": artifact_bytes * 8.0 / weights.size,
                },
                "measurement": metrics,
                "teacher_training": training,
                "metal_component_probe": metal,
                "elapsed_seconds": time.perf_counter() - started,
                "claim_boundary": {
                    "component_measurement_only": True,
                    "not_a_full_model_artifact": True,
                    "not_a_complete_model_bpw_result": True,
                    "not_a_native_direct_execution_plan": True,
                    "not_a_tg3_or_hcli_result": True,
                    "not_a_protected_manager_receipt": True,
                },
            }
        )
        _atomic_json(result_path, candidate)
        state["heartbeat"] = int(state["heartbeat"]) + 1
        state["experiment"] = {
            "status": "RUNNING",
            "last_completed_candidate": candidate_id,
            "completed_candidates": len(self._completed_candidate_ids()),
            "candidate_result_path": str(result_path),
            "candidate_seal_sha256": candidate["seal_sha256"],
        }
        self._publish(state)

    def run_cycle(self, *, max_candidates: int) -> dict[str, Any]:
        if not self.model_dir.is_dir():
            raise PhysicalCampaignError(f"model directory does not exist: {self.model_dir}")
        state = self._state()
        self._source_audit(state)
        audit = _read_json(self.source_audit_path)
        if audit is None or not isinstance(audit.get("seal_sha256"), str):
            raise PhysicalCampaignError("source audit did not produce a sealed candidate record")
        weight_map = load_weight_map(self.model_dir)
        completed = self._completed_candidate_ids()
        if len(completed) >= SCOUT_CANDIDATE_BUDGET:
            self._publish_frontier_and_gene_pools(state)
            return _read_json(self.status_path) or state
        performed = 0
        for tensor_name in _target_tensors(weight_map):
            for spec in CANDIDATE_SPECS:
                candidate_id = f"{tensor_name.replace('.', '_')}__{spec.identifier}"
                if candidate_id in completed:
                    continue
                self._run_candidate(
                    state=state,
                    tensor_name=tensor_name,
                    spec=spec,
                    source_sha256=audit["seal_sha256"],
                )
                performed += 1
                if len(self._completed_candidate_ids()) >= SCOUT_CANDIDATE_BUDGET:
                    self._publish_frontier_and_gene_pools(state)
                    return _read_json(self.status_path) or state
                if performed >= max_candidates or self._stopping:
                    return _read_json(self.status_path) or state
        state["phase"] = "FIRST_CANDIDATE_SWEEP_COMPLETE"
        state["resource_owner"] = {"cpu": "IDLE_AFTER_SEALED_SWEEP", "gpu": "IDLE_NO_NATIVE_FULL_MODEL_CANDIDATE"}
        state["experiment"] = {
            "status": "FIRST_SWEEP_COMPLETE",
            "completed_candidates": len(self._completed_candidate_ids()),
            "next_automatic_transition": "wait_for_controller review of real candidate evidence; never auto-promote",
        }
        self._publish(state)
        return state

    def watch(self, *, max_candidates_per_cycle: int, idle_seconds: float) -> int:
        if max_candidates_per_cycle <= 0 or idle_seconds <= 0:
            raise PhysicalCampaignError("watch limits must be positive")

        def request_stop(_signal: int, _frame: Any) -> None:
            self._stopping = True

        previous_term = signal.signal(signal.SIGTERM, request_stop)
        previous_int = signal.signal(signal.SIGINT, request_stop)
        try:
            while not self._stopping:
                self.run_cycle(max_candidates=max_candidates_per_cycle)
                if self._stopping:
                    break
                time.sleep(idle_seconds)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    once = sub.add_parser("once", help="complete source audit and measure bounded candidates")
    once.add_argument("--max-candidates", type=int, default=1)
    watch = sub.add_parser("watch", help="run restart-safe physical candidate work indefinitely")
    watch.add_argument("--max-candidates-per-cycle", type=int, default=1)
    watch.add_argument("--idle-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = Qwen30PhysicalCampaign(model_dir=args.model_dir, root=args.root)
    if args.command == "once":
        if args.max_candidates <= 0:
            raise SystemExit("--max-candidates must be positive")
        print(json.dumps(campaign.run_cycle(max_candidates=args.max_candidates), indent=2, sort_keys=True))
        return 0
    if args.command == "watch":
        return campaign.watch(
            max_candidates_per_cycle=args.max_candidates_per_cycle,
            idle_seconds=args.idle_seconds,
        )
    raise AssertionError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
