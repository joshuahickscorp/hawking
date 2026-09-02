"""RIVAL_CODEC_SCREEN — score every named rival family on the real L4 teacher surface.

Exactly one codec has been screened against the 1024-row source-BF16 layer-4
MLP-input corpus, and it failed structurally. This module fits the rival
families already named by expert_bank_school / flash_schools on that same
split, under that same coherence contract, so the numbers are comparable.

It first reproduces shared_input_latent_plus_expert_local_output_readout at
ranks 4/8/16/32/64 against the committed screen receipt. A harness that cannot
reproduce the known result does not publish rival numbers.

This is fit quality on a bounded teacher surface. It is not coherence of a
runtime, not capability, not physical EBPW, and no family "wins" by passing
the screen — it only earns the right to be measured further.

Refuses: hardware measurement, GPU lease, scoring an underdetermined fit,
defaulting a missing corpus into a table, labelling a miss as a pass.

Cannot establish: whole-model function, router stability, generated-token
capability, serialized bytes, or that a family which fails this organ is
dead on another.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future._common import REPO, git, sha256_file, write_receipt

RECEIPT = "RIVAL_CODEC_SCREEN.json"
SCHEMA = "hawking.future.rival_codec_screen.v1"
RECORDED_BY = "tools/future/rival_codec_screen.py"
VERSION = 1

SCREEN_REL = "receipts/future/evidence/FLASH_META_COHERENCE_SCREEN_L4_REAL1024.json"
TEACHER_REL = "receipts/future/evidence/FLASH_META_TEACHER_L4_REAL1024.json"
STATE_REL = "receipts/future/evidence/FLASH_META_TEACHER_L4_MLP_INPUT_REAL1024.f32"
SCREEN_TOOL_REL = "tools/flash_meta_coherence_screen.py"
EBS_REL = "tools/future/expert_bank_school.py"
SCHOOLS_REL = "tools/future/flash_schools.py"
REPLAN_REL = "tools/future/flash_meta_replan.py"

MODEL = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
DEFAULT_MODEL = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
TENSOR_GATE_UP = "model.language_model.layers.4.mlp.experts.gate_up_proj"
TENSOR_SHARED_GATE = "model.language_model.layers.4.mlp.shared_expert.gate_proj.weight"
TENSOR_SHARED_UP = "model.language_model.layers.4.mlp.shared_expert.up_proj.weight"

# Recovered from tools/flash_meta_coherence_screen.py; used only when the
# committed screen receipt does not itself name the sweep / geometry.
RECOVERED_DEFAULT_RANKS = (4, 8, 16, 32, 64)
RECOVERED_HIDDEN = 2560
RECOVERED_Q4_GROUP = 64
RECOVERED_MIN_TEACHER_ROWS = 256
REFERENCE_KIND = "shared_input_latent_plus_expert_local_output_readout"
REFERENCE_ORGAN = "layer_4.routed_experts.gate_up_proj"

# Stated harness tolerances. The load-bearing comparison is held-out error
# and cosine (the contract quantities). Fit-set error is reported but a
# float32 lstsq of 819 x rank against 314 experts moves ~6e-4 across NumPy
# builds; that is not a different split. A wrong holdout or a different
# family misses by 0.05–0.5, well outside these windows.
HELDOUT_ABS_TOL = 2e-4
FIT_ABS_TOL = 2e-3
BPW_ABS_TOL = 1e-12
Q4_ABS_TOL = 5e-4
# Back-compat alias used by tests that check the module's stated window.
REFERENCE_ABS_TOL = HELDOUT_ABS_TOL
CLUSTER_K = 8
KMEANS_SEED = 0

# Campaign Q4 scale overhead: group-64 BF16 scale -> +0.25 bpw (flash_schools
# Q4_NUM/Q4_DEN = 4.25/16 of BF16). Same identity, not a new budget.
Q4_SCALE_BITS_PER_GROUP = 16


class UnderdeterminedFitError(ValueError):
    """n_fit is below the fitted dimension. Scoring this would reopen NS-014."""


class AbsentInputError(ValueError):
    """A required corpus / specimen / contract input is missing."""


# ---------------------------------------------------------------------------
# Recovered primitives from tools/flash_meta_coherence_screen.py
# Accounting and the holdout must match that file, not a cousin.
# ---------------------------------------------------------------------------


def heldout_split(rows: int) -> tuple[np.ndarray, np.ndarray]:
    if rows < 2:
        raise ValueError("at least two teacher rows are required for a holdout")
    heldout = np.zeros(rows, dtype=bool)
    heldout[np.arange(rows) % 5 == 0] = True
    if not heldout.any():
        heldout[-1] = True
    if heldout.all():
        heldout[-1] = False
    return ~heldout, heldout


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64, copy=False)
    b = right.reshape(-1).astype(np.float64, copy=False)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return float(np.dot(a, b) / denominator)


def relative_fro(pred: np.ndarray, teacher: np.ndarray) -> float:
    p = pred.astype(np.float64, copy=False)
    t = teacher.astype(np.float64, copy=False)
    return float(np.linalg.norm(p - t) / max(float(np.linalg.norm(t)), 1e-30))


def symmetric_group_q4(weights: np.ndarray, group: int = 64) -> np.ndarray:
    experts, output, input_width = weights.shape
    if input_width % group:
        raise ValueError(f"Q4 group {group} does not divide input width {input_width}")
    grouped = weights.reshape(experts, output, input_width // group, group)
    scale = np.maximum(np.max(np.abs(grouped), axis=3, keepdims=True) / 7.0, 1e-30)
    code = np.clip(np.rint(grouped / scale), -8, 7).astype(np.int8)
    return (code.astype(np.float32) * scale).reshape(weights.shape)


def symmetric_group_qn(weights: np.ndarray, bits: int, group: int = 64) -> np.ndarray:
    """Same group-wise symmetric quant as the reference Q4, at `bits`."""
    if bits == 4:
        return symmetric_group_q4(weights, group=group)
    if bits < 2 or bits > 8:
        raise ValueError(f"quant bits {bits} is outside the Q2–Q8 control range")
    experts, output, input_width = weights.shape
    if input_width % group:
        raise ValueError(f"Q{bits} group {group} does not divide input width {input_width}")
    half = 1 << (bits - 1)
    qmax = half - 1
    qmin = -half
    grouped = weights.reshape(experts, output, input_width // group, group)
    scale = np.maximum(
        np.max(np.abs(grouped), axis=3, keepdims=True) / max(float(qmax), 1.0),
        1e-30,
    )
    code = np.clip(np.rint(grouped / scale), qmin, qmax)
    return (code.astype(np.float32) * scale).reshape(weights.shape)


def projected_outputs(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ni,eoi->eno", states, weights, optimize=True)


def quant_factor_bytes(n_params: int, bits: int, group: int = 64) -> int:
    """Packed codes plus one BF16 scale per group. Same byte domain as the screen."""
    if n_params < 1 or bits < 1 or group < 1 or n_params % group:
        raise ValueError("quant factor-byte identity is not defined for this geometry")
    n_groups = n_params // group
    return (n_params * bits + 7) // 8 + n_groups * 2


def diagnostic_bpw(factor_bytes: int, source_parameters: int) -> float:
    """Identical identity to the reference screen: factor_bytes * 8 / n_params."""
    return float(factor_bytes) * 8.0 / max(int(source_parameters), 1)


def fit_shared_latent_program(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    teacher: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recovered from tools/flash_meta_coherence_screen.py::fit_shared_latent_program.

    `teacher` may be supplied so a multi-rank sweep does not re-project W x.
    Algebra is the same either way. n_fit < rank is refused, not scored.
    """
    n_fit = int(fit_rows.sum())
    n_held = int(heldout_rows.sum())
    if rank < 1:
        raise ValueError(f"rank {rank} is not available")
    if n_fit < rank:
        raise UnderdeterminedFitError(
            f"n_fit={n_fit} < rank={rank}: NS-014, the score is not the codec's score"
        )
    if rank > states.shape[1]:
        raise ValueError(f"rank {rank} exceeds input width {states.shape[1]}")
    if teacher is None:
        teacher = projected_outputs(states, weights)
    fit_states = states[fit_rows]
    _, _, vh = np.linalg.svd(fit_states, full_matrices=False)
    basis = vh[:rank].T.astype(np.float32, copy=False)
    fit_features = fit_states @ basis
    eval_features = states[heldout_rows] @ basis
    readouts: list[np.ndarray] = []
    fit_prediction = np.empty((weights.shape[0], n_fit, weights.shape[1]), dtype=np.float32)
    heldout_prediction = np.empty(
        (weights.shape[0], n_held, weights.shape[1]), dtype=np.float32
    )
    for expert in range(weights.shape[0]):
        readout, _, _, _ = np.linalg.lstsq(
            fit_features,
            teacher[expert, fit_rows],
            rcond=None,
        )
        readout = readout.astype(np.float32, copy=False)
        readouts.append(readout)
        fit_prediction[expert] = fit_features @ readout
        heldout_prediction[expert] = eval_features @ readout
    readout_stack = np.stack(readouts, axis=0)
    fit_teacher = teacher[:, fit_rows]
    heldout_teacher = teacher[:, heldout_rows]
    fit_error = relative_fro(fit_prediction, fit_teacher)
    heldout_error = relative_fro(heldout_prediction, heldout_teacher)
    source_parameters = int(weights.size)
    factor_bytes = int((basis.size + readout_stack.size) * 2)
    heldout_cosine = cosine(heldout_prediction, heldout_teacher)
    return {
        "rank": rank,
        "fit_rows": n_fit,
        "heldout_rows": n_held,
        "fit_relative_fro_error": fit_error,
        "heldout_relative_fro_error": heldout_error,
        "heldout_cosine": heldout_cosine,
        "diagnostic_factor_bytes": factor_bytes,
        "diagnostic_factor_equivalent_bpw": diagnostic_bpw(factor_bytes, source_parameters),
        "selected_dense_source_bytes": source_parameters * 2,
        "selected_dense_loaded_f32_bytes": int(weights.nbytes),
        "fitted_dimension": rank,
        "scored": True,
    }


# ---------------------------------------------------------------------------
# Evidence discovery. Sparse checkout is not proof of absence.
# ---------------------------------------------------------------------------


def checkout_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    blob = git("worktree", "list", "--porcelain")
    for line in blob.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.split(" ", 1)[1]))
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def find_path(rel: str) -> Path | None:
    for root in checkout_roots():
        path = Path(root) / rel
        if path.is_file():
            return path
    return None


def load_json_path(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_named(rel: str) -> tuple[dict[str, Any] | None, str | None]:
    path = find_path(rel)
    if path is not None:
        try:
            return load_json_path(path), str(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None, str(path)
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return None, None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(doc, dict):
        return None, None
    return doc, f"HEAD:{rel}"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _pyf(value: float) -> float:
    return float(value)


# ---------------------------------------------------------------------------
# Contract. Read from the committed screen, never rewritten here.
# ---------------------------------------------------------------------------


def contract_from_screen(screen: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(screen, Mapping):
        return {
            "ok": False,
            "reason": "screen receipt absent; contract is not defaulted here",
            "min_heldout_cosine": None,
            "max_heldout_relative_fro_error": None,
            "must_beat_per_expert_q4": None,
        }
    block = screen.get("coherence_contract")
    if not isinstance(block, Mapping):
        return {
            "ok": False,
            "reason": "screen.coherence_contract absent; contract is not defaulted here",
            "min_heldout_cosine": None,
            "max_heldout_relative_fro_error": None,
            "must_beat_per_expert_q4": None,
        }
    cosine_g = _as_float(block.get("min_heldout_cosine"))
    err = _as_float(block.get("max_heldout_relative_fro_error"))
    beat = _as_bool(block.get("must_beat_per_expert_q4"))
    missing = [
        name
        for name, value in (
            ("min_heldout_cosine", cosine_g),
            ("max_heldout_relative_fro_error", err),
            ("must_beat_per_expert_q4", beat),
        )
        if value is None
    ]
    if missing:
        return {
            "ok": False,
            "reason": "screen.coherence_contract missing " + ",".join(missing),
            "min_heldout_cosine": cosine_g,
            "max_heldout_relative_fro_error": err,
            "must_beat_per_expert_q4": beat,
        }
    return {
        "ok": True,
        "reason": "read from screen.coherence_contract; not redefined",
        "source": "coherence_contract",
        "min_heldout_cosine": cosine_g,
        "max_heldout_relative_fro_error": err,
        "must_beat_per_expert_q4": beat,
        "fit_holdout_required": _as_bool(block.get("fit_holdout_required")),
        "raw_keys": sorted(block.keys()),
    }


def ranks_from_screen(screen: Mapping[str, Any] | None) -> tuple[int, ...]:
    if not isinstance(screen, Mapping):
        return RECOVERED_DEFAULT_RANKS
    rows = ((screen.get("surface") or {}).get("rows")) or []
    ranks = [_as_int(r.get("rank")) for r in rows if isinstance(r, Mapping)]
    got = tuple(r for r in ranks if r is not None)
    return got if got else RECOVERED_DEFAULT_RANKS


def committed_reference_rows(screen: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(screen, Mapping):
        return []
    rows = ((screen.get("surface") or {}).get("rows")) or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rank = _as_int(row.get("rank"))
        if rank is None:
            continue
        out.append(
            {
                "rank": rank,
                "fit_relative_fro_error": _as_float(row.get("fit_relative_fro_error")),
                "heldout_relative_fro_error": _as_float(row.get("heldout_relative_fro_error")),
                "heldout_cosine": _as_float(row.get("heldout_cosine")),
                "per_expert_q4_heldout_relative_fro_error": _as_float(
                    row.get("per_expert_q4_heldout_relative_fro_error")
                ),
                "beats_per_expert_q4_on_heldout": _as_bool(
                    row.get("beats_per_expert_q4_on_heldout")
                ),
                "diagnostic_factor_equivalent_bpw": _as_float(
                    row.get("diagnostic_factor_equivalent_bpw")
                ),
                "diagnostic_factor_bytes": _as_int(row.get("diagnostic_factor_bytes")),
                "surface_gate_pass": _as_bool(row.get("surface_gate_pass")),
                "first_surface_failure": row.get("first_surface_failure"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Gate. Failure strings and inequalities copied from the reference screen.
# ---------------------------------------------------------------------------


def apply_surface_gates(
    *,
    heldout_error: float,
    heldout_cosine: float,
    q4_error: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign pass/fail from the screen's contract. Never invent a pass."""
    if not contract.get("ok"):
        return {
            "surface_gate_pass": False,
            "surface_failure_gates": ["coherence_contract_unavailable"],
            "first_surface_failure": "coherence_contract_unavailable",
            "beats_per_expert_q4_on_heldout": bool(heldout_error < q4_error),
            "labelled": "REFUSED_NO_CONTRACT",
        }
    max_err = float(contract["max_heldout_relative_fro_error"])
    min_cos = float(contract["min_heldout_cosine"])
    must_beat = bool(contract["must_beat_per_expert_q4"])
    failure_gates: list[str] = []
    if heldout_error > max_err:
        failure_gates.append("held-out function error")
    if heldout_cosine < min_cos:
        failure_gates.append("held-out function cosine")
    beats = bool(heldout_error < q4_error)
    if must_beat and heldout_error >= q4_error:
        failure_gates.append("does not beat per-expert Q4")
    gate_pass = (
        heldout_error <= max_err
        and heldout_cosine >= min_cos
        and (beats if must_beat else True)
    )
    if gate_pass and failure_gates:
        # A validator that can pass with a populated failure list will drift.
        gate_pass = False
    labelled = "SCORED_PASSED_CONTRACT" if gate_pass else "SCORED_FAILED_CONTRACT"
    return {
        "surface_gate_pass": bool(gate_pass),
        "surface_failure_gates": failure_gates,
        "first_surface_failure": failure_gates[0] if failure_gates else None,
        "beats_per_expert_q4_on_heldout": beats,
        "labelled": labelled,
    }


def attach_gates(
    row: dict[str, Any],
    *,
    q4_error: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not row.get("scored"):
        row.setdefault("surface_gate_pass", False)
        row.setdefault("labelled", row.get("status") or "UNSCORED")
        if row.get("surface_gate_pass") is True:
            row["surface_gate_pass"] = False
            row["labelled"] = "REFUSED_UNSCORED_NOT_A_PASS"
        return row
    gates = apply_surface_gates(
        heldout_error=float(row["heldout_relative_fro_error"]),
        heldout_cosine=float(row["heldout_cosine"]),
        q4_error=q4_error,
        contract=contract,
    )
    row.update(gates)
    row["per_expert_q4_heldout_relative_fro_error"] = _pyf(q4_error)
    row["status"] = gates["labelled"]
    return row


def refused_row(
    *,
    family: str,
    reason: str,
    status: str,
    rank: int | None = None,
    fitted_dimension: int | None = None,
    n_fit: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "status": status,
        "scored": False,
        "surface_gate_pass": False,
        "labelled": status,
        "reason": reason,
        "rank": rank,
        "fitted_dimension": fitted_dimension,
        "n_fit": n_fit,
        "heldout_relative_fro_error": None,
        "heldout_cosine": None,
        "beats_per_expert_q4_on_heldout": None,
        "diagnostic_factor_equivalent_bpw": None,
        "first_surface_failure": status,
        "surface_failure_gates": [status],
    }
    if extra:
        row.update(dict(extra))
    return row


# ---------------------------------------------------------------------------
# Specimen I/O. Only the shards named by the index for the tensors we need.
# ---------------------------------------------------------------------------


def locate_tensor(
    root: Path, tensor_name: str
) -> tuple[Path, int, int, tuple[int, ...], str]:
    index = load_json_path(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or tensor_name not in weight_map:
        raise AbsentInputError(f"source tensor not in index: {tensor_name}")
    shard = root / str(weight_map[tensor_name])
    if not shard.is_file():
        raise AbsentInputError(f"source shard not found: {shard}")
    with shard.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    metadata = header.get(tensor_name)
    if not isinstance(metadata, Mapping) or "data_offsets" not in metadata:
        raise AbsentInputError(f"source tensor missing from shard header: {tensor_name}")
    begin, end = metadata["data_offsets"]
    return (
        shard,
        8 + header_length + int(begin),
        int(end - begin),
        tuple(int(item) for item in metadata["shape"]),
        str(metadata["dtype"]),
    )


def _bf16_bytes_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32, copy=True)
    values = (words << 16).view("<f4")
    return np.ascontiguousarray(values.reshape(shape))


def read_bf16_experts(
    location: tuple[Path, int, int, tuple[int, ...], str], experts: Sequence[int]
) -> np.ndarray:
    shard, offset, total_bytes, shape, dtype = location
    if dtype != "BF16" or len(shape) != 3:
        raise ValueError(f"unsupported expert tensor {shape} {dtype}")
    if max(experts) >= shape[0] or min(experts) < 0:
        raise ValueError("expert index outside the source tensor")
    # Sequential full-tensor read then gather: 314 random seeks on corpdrive
    # were the 156s path; the accounting (BF16->f32, expert order) is identical.
    with shard.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(total_bytes)
    if len(raw) != total_bytes:
        raise IOError(f"short expert tensor read: {shard}")
    stacked = _bf16_bytes_to_f32(raw, shape)
    return np.ascontiguousarray(stacked[list(experts)])


def read_bf16_matrix(
    location: tuple[Path, int, int, tuple[int, ...], str],
) -> np.ndarray:
    shard, offset, total_bytes, shape, dtype = location
    if dtype != "BF16":
        raise ValueError(f"unsupported dense tensor {shape} {dtype}")
    with shard.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(total_bytes)
    if len(raw) != total_bytes:
        raise IOError(f"short dense tensor read: {shard}")
    return _bf16_bytes_to_f32(raw, shape)


def load_teacher_states(path: Path, *, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Dedup-by-hash before any holdout, matching the reference loader."""
    if not path.is_file():
        raise AbsentInputError(f"teacher state not found: {path}")
    values = np.fromfile(path, dtype="<f4")
    if values.size == 0 or values.size % width:
        raise ValueError(f"teacher state has invalid width: {path} ({values.size})")
    rows = values.reshape(-1, width)
    if not np.isfinite(rows).all():
        raise ValueError(f"teacher state contains non-finite values: {path}")
    unique_rows: list[np.ndarray] = []
    row_hashes: set[str] = set()
    for row in rows:
        digest = hashlib.sha256(row.astype("<f4", copy=False).tobytes()).hexdigest()
        if digest not in row_hashes:
            row_hashes.add(digest)
            unique_rows.append(row.copy())
    states = np.stack(unique_rows, axis=0).astype(np.float32, copy=False)
    return states, {
        "path": str(path),
        "sha256": sha256_file(path),
        "raw_rows": int(rows.shape[0]),
        "rows": int(states.shape[0]),
        "unique_row_hashes": len(row_hashes),
        "width": int(states.shape[1]),
    }


# ---------------------------------------------------------------------------
# Weight-space factor helpers. Gram eigh, not a new codec.
# ---------------------------------------------------------------------------


def _top_eigh(gram: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal columns for the top-`rank` eigenspace of a PSD gram."""
    rank = int(rank)
    n = int(gram.shape[0])
    if rank < 1 or rank > n:
        raise ValueError(f"rank {rank} is not available for gram {n}")
    evals, evecs = np.linalg.eigh(gram.astype(np.float64, copy=False))
    return np.ascontiguousarray(evecs[:, -rank:].astype(np.float32, copy=False))


# One bank's worth of right eigenvectors, held so a rank sweep pays for them once.
# Deliberately a single entry: the sweep asks for ranks 4, 8, 16, 32, 64 of the SAME
# stacked bank, and the gram and its eigendecomposition do not depend on the rank --
# only the trailing slice does. Holding a strong reference to the bank is what makes
# `id()` a sound key: an array that is still alive cannot have its address reused.
_RIGHT_EIGVECS: tuple[int, np.ndarray, np.ndarray] | None = None


def _bank_right_eigvecs(weights: np.ndarray) -> np.ndarray:
    """Ascending-eigenvalue eigenvectors of the bank's right gram."""
    global _RIGHT_EIGVECS
    if _RIGHT_EIGVECS is not None:
        key, held, evecs = _RIGHT_EIGVECS
        if key == id(weights) and held is weights:
            return evecs
    flat = weights.reshape(-1, weights.shape[2])
    gram = flat.T.astype(np.float64, copy=False) @ flat.astype(np.float64, copy=False)
    _evals, evecs = np.linalg.eigh(gram)
    _RIGHT_EIGVECS = (id(weights), weights, evecs)
    return evecs


def right_basis(weights: np.ndarray, rank: int) -> np.ndarray:
    """Shared right factor V[r, d_in] of the stacked bank (rows orthonormal)."""
    e, _o, d_in = weights.shape
    max_rank = min(d_in, e * weights.shape[1], rank)
    if max_rank < rank:
        raise ValueError(f"right rank {rank} exceeds stacked bank rank {max_rank}")
    if rank < 1 or rank > d_in:
        raise ValueError(f"rank {rank} is not available for gram {d_in}")
    evecs = _bank_right_eigvecs(weights)
    v_cols = np.ascontiguousarray(evecs[:, -rank:].astype(np.float32, copy=False))
    return np.ascontiguousarray(v_cols.T)


def left_basis(weights: np.ndarray, rank: int) -> np.ndarray:
    """Shared left factor U[d_out, r] of the stacked bank (columns orthonormal)."""
    _e, d_out, d_in = weights.shape
    max_rank = min(d_out, weights.shape[0] * d_in, rank)
    if max_rank < rank:
        raise ValueError(f"left rank {rank} exceeds stacked bank rank {max_rank}")
    # W_e @ W_e^T summed over experts: contract the input axis, not the output axis.
    gram = np.zeros((d_out, d_out), dtype=np.float64)
    for expert_w in weights:
        wide = expert_w.astype(np.float64, copy=False)
        gram += wide @ wide.T
    return _top_eigh(gram, rank)


def residual_factors(residual: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """R ≈ U @ V with U[d_out, r], V[r, d_in]. Gram path; rank of zeros is refused."""
    d_out, d_in = residual.shape
    max_rank = min(rank, d_out, d_in)
    if max_rank < 1:
        raise ValueError("residual has no available rank")
    gram = residual.astype(np.float64, copy=False) @ residual.T.astype(np.float64, copy=False)
    u = _top_eigh(gram, max_rank)  # [d_out, r]
    # V = U^T R, then absorb so y = U (V x) with U orthonormal.
    v = u.T @ residual  # [r, d_in]
    return u.astype(np.float32, copy=False), np.ascontiguousarray(v.astype(np.float32, copy=False))


def _eigh_workers() -> int:
    """How many experts to factor at once.

    Each expert's gram and eigendecomposition is independent of every other's,
    and `np.linalg.eigh` releases the GIL, so this is real parallelism rather
    than a scheduling trick. Measured on this host at d_out=1408: 6.41s serial,
    1.26s across 8 threads (5.07x), and every returned array bit-identical to
    the serial result. 14 threads bought 5.32x, so 8 is the knee -- and staying
    at 8 keeps a worker from oversubscribing the box when the suite runs this
    under xdist. RCS_EIGH_WORKERS=1 restores the serial path.
    """
    override = os.environ.get("RCS_EIGH_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, min(8, os.cpu_count() or 1))


def residual_factors_batch(residuals: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Batch of R_e ≈ U_e @ V_e. Per-expert GEMM; a 3-way einsum of this size can explode."""
    _e, d_out, d_in = residuals.shape
    max_rank = min(rank, d_out, d_in)
    if max_rank < 1:
        raise ValueError("residual has no available rank")

    def factor(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wide = residual.astype(np.float64, copy=False)
        gram = wide @ wide.T
        u = _top_eigh(gram, max_rank)
        v = u.T @ residual
        return (
            u.astype(np.float32, copy=False),
            np.ascontiguousarray(v.astype(np.float32, copy=False)),
        )

    workers = min(_eigh_workers(), max(1, int(residuals.shape[0])))
    if workers == 1:
        pairs = [factor(residual) for residual in residuals]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(factor, residuals))
    u_rows = [u for u, _ in pairs]
    v_rows = [v for _, v in pairs]
    return np.stack(u_rows, axis=0), np.stack(v_rows, axis=0)


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 25) -> np.ndarray:
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    choice = rng.choice(n, size=k, replace=False)
    centers = x[choice].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1).astype(np.int32)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
    for j in range(k):
        if not (labels == j).any():
            largest = max(range(k), key=lambda c: int((labels == c).sum()))
            steal = np.where(labels == largest)[0]
            labels[int(steal[0])] = j
    return labels


def route_cluster_labels(
    fit_route_ids: Sequence[Sequence[int]],
    expert_index: Mapping[int, int],
    *,
    k: int,
    seed: int = KMEANS_SEED,
) -> np.ndarray:
    """Cluster experts by co-occurrence on the FIT split only (no holdout leak)."""
    e = len(expert_index)
    cooccur = np.zeros((e, e), dtype=np.float64)
    for routes in fit_route_ids:
        ids = [expert_index[r] for r in routes if r in expert_index]
        if len(ids) < 2:
            continue
        for a in ids:
            cooccur[a, ids] += 1.0
    norms = np.linalg.norm(cooccur, axis=1, keepdims=True)
    features = cooccur / np.maximum(norms, 1e-30)
    rng = np.random.default_rng(seed)
    return _kmeans(features, k, rng)


# ---------------------------------------------------------------------------
# Family fits. Weight-space families are determined by W; activation-space
# lstsq is determined by (X, Y) and is refused when n_fit < fitted dim.
# ---------------------------------------------------------------------------


def _score_pred(
    *,
    family: str,
    pred_held: np.ndarray,
    pred_fit: np.ndarray | None,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    factor_bytes: int,
    source_parameters: int,
    rank: int | None,
    fitted_dimension: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    held_t = teacher[:, heldout_rows]
    row: dict[str, Any] = {
        "family": family,
        "scored": True,
        "rank": rank,
        "fitted_dimension": int(fitted_dimension),
        "fit_rows": int(fit_rows.sum()),
        "heldout_rows": int(heldout_rows.sum()),
        "heldout_relative_fro_error": relative_fro(pred_held, held_t),
        "heldout_cosine": cosine(pred_held, held_t),
        "diagnostic_factor_bytes": int(factor_bytes),
        "diagnostic_factor_equivalent_bpw": diagnostic_bpw(factor_bytes, source_parameters),
        "selected_dense_source_bytes": int(source_parameters) * 2,
    }
    if pred_fit is not None:
        row["fit_relative_fro_error"] = relative_fro(pred_fit, teacher[:, fit_rows])
    if extra:
        row.update(dict(extra))
    return row


def fit_quant_control(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    bits: int,
    family: str,
    group: int = RECOVERED_Q4_GROUP,
) -> dict[str, Any]:
    quantized = symmetric_group_qn(weights, bits, group=group)
    pred_held = projected_outputs(states[heldout_rows], quantized)
    pred_fit = projected_outputs(states[fit_rows], quantized)
    n_params = int(weights.size)
    factor_bytes = quant_factor_bytes(n_params, bits, group=group)
    return _score_pred(
        family=family,
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=n_params,
        rank=None,
        fitted_dimension=0,
        extra={"bits": bits, "group": group, "quant_control": True},
    )


def fit_common_right(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    v = right_basis(weights, rank)  # [r, d_in]
    cores = np.matmul(weights, v.T)  # [E, o, r]
    z_held = states[heldout_rows] @ v.T
    z_fit = states[fit_rows] @ v.T
    pred_held = np.matmul(z_held[None, :, :], cores.transpose(0, 2, 1))
    pred_fit = np.matmul(z_fit[None, :, :], cores.transpose(0, 2, 1))
    factor_bytes = int((v.size + cores.size) * 2)
    return _score_pred(
        family="common_right_subspace_plus_expert_local_core",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(weights.size),
        rank=rank,
        fitted_dimension=rank,
        extra={
            "mechanism_source": f"{EBS_REL}#STORE-COMMON-RIGHT-SUBSPACE",
            "algebra": "y_e = C_e @ (V @ x)",
        },
    )


def fit_common_left(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    u = left_basis(weights, rank)  # [o, r]
    cores = np.matmul(u.T[None, :, :], weights)  # [E, r, i]
    z_held = np.matmul(cores, states[heldout_rows].T).transpose(0, 2, 1)
    z_fit = np.matmul(cores, states[fit_rows].T).transpose(0, 2, 1)
    pred_held = np.matmul(u[None, :, :], z_held.transpose(0, 2, 1)).transpose(0, 2, 1)
    pred_fit = np.matmul(u[None, :, :], z_fit.transpose(0, 2, 1)).transpose(0, 2, 1)
    factor_bytes = int((u.size + cores.size) * 2)
    return _score_pred(
        family="common_left_subspace_plus_expert_local_core",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(weights.size),
        rank=rank,
        fitted_dimension=rank,
        extra={
            "mechanism_source": f"{EBS_REL}#STORE-COMMON-LEFT-SUBSPACE",
            "algebra": "y_e = U @ (C_e @ x)",
        },
    )


def fit_small_core(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    v = right_basis(weights, rank)
    mid = np.matmul(weights, v.T)  # [E, o, r]
    gram = np.zeros((mid.shape[1], mid.shape[1]), dtype=np.float64)
    for slice_e in mid:
        wide = slice_e.astype(np.float64, copy=False)
        gram += wide @ wide.T
    u = _top_eigh(gram, rank)  # [o, r]
    cores = np.matmul(u.T[None, :, :], mid)  # [E, r, r]
    z_held = states[heldout_rows] @ v.T
    z_fit = states[fit_rows] @ v.T
    lat_held = np.matmul(cores, z_held.T).transpose(0, 2, 1)
    lat_fit = np.matmul(cores, z_fit.T).transpose(0, 2, 1)
    pred_held = np.matmul(u[None, :, :], lat_held.transpose(0, 2, 1)).transpose(0, 2, 1)
    pred_fit = np.matmul(u[None, :, :], lat_fit.transpose(0, 2, 1)).transpose(0, 2, 1)
    factor_bytes = int((u.size + v.size + cores.size) * 2)
    return _score_pred(
        family="expert_local_small_core_plus_shared_decoder",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(weights.size),
        rank=rank,
        fitted_dimension=rank,
        extra={
            "mechanism_source": f"{EBS_REL}#STORE-EXPERT-SMALL-CORE",
            "algebra": "y_e = U @ C_e @ (V @ x)",
        },
    )


def fit_clustered(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    rank: int,
    labels: np.ndarray,
) -> dict[str, Any]:
    e, d_out, d_in = weights.shape
    k = int(labels.max()) + 1 if labels.size else 0
    if labels.shape[0] != e or k < 1:
        raise ValueError("cluster labels do not cover the expert axis")
    pred_held = np.empty((e, int(heldout_rows.sum()), d_out), dtype=np.float32)
    pred_fit = np.empty((e, int(fit_rows.sum()), d_out), dtype=np.float32)
    factor_elems = 0
    used_k = 0
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            continue
        used_k += 1
        sub = weights[members]
        max_r = min(rank, d_in, int(members.size) * d_out)
        if max_r < 1:
            raise ValueError(f"cluster {cid} has no available rank")
        v = right_basis(sub, max_r)
        cores = np.matmul(sub, v.T)
        z_held = states[heldout_rows] @ v.T
        z_fit = states[fit_rows] @ v.T
        pred_held[members] = np.matmul(z_held[None, :, :], cores.transpose(0, 2, 1))
        pred_fit[members] = np.matmul(z_fit[None, :, :], cores.transpose(0, 2, 1))
        factor_elems += int(v.size + cores.size)
    factor_bytes = factor_elems * 2
    return _score_pred(
        family="clustered_subspaces_route_conditioned",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(weights.size),
        rank=rank,
        fitted_dimension=rank,
        extra={
            "mechanism_source": f"{EBS_REL}#STORE-CLUSTERED-SUBSPACES",
            "cluster_k": k,
            "occupied_clusters": used_k,
            "cluster_seed": KMEANS_SEED,
            "algebra": "y_e = C_e @ (V_{cluster(e)} @ x)",
        },
    )


def fit_dictionary_sparse_residual(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    """Shared right dictionary, per-expert column-sparse codes, skinny residual."""
    s = max(rank // 4, 1)
    r_res = max(rank // 4, 1)
    v = right_basis(weights, rank)  # [K, d_in]
    codes = np.matmul(weights, v.T)  # [E, o, K]
    sparse = np.zeros_like(codes)
    kept = 0
    for expert in range(weights.shape[0]):
        energy = np.linalg.norm(codes[expert], axis=0)
        order = np.argsort(energy)[::-1]
        take = order[:s]
        sparse[expert][:, take] = codes[expert][:, take]
        kept += int(take.size)
    z_held = states[heldout_rows] @ v.T
    z_fit = states[fit_rows] @ v.T
    pred_held = np.matmul(z_held[None, :, :], sparse.transpose(0, 2, 1))
    pred_fit = np.matmul(z_fit[None, :, :], sparse.transpose(0, 2, 1))
    recon = np.matmul(sparse, v)
    u_stack, v_stack = residual_factors_batch(weights - recon, r_res)
    z_res_held = np.matmul(v_stack, states[heldout_rows].T).transpose(0, 2, 1)
    z_res_fit = np.matmul(v_stack, states[fit_rows].T).transpose(0, 2, 1)
    pred_held = pred_held + np.matmul(u_stack, z_res_held.transpose(0, 2, 1)).transpose(0, 2, 1)
    pred_fit = pred_fit + np.matmul(u_stack, z_res_fit.transpose(0, 2, 1)).transpose(0, 2, 1)
    # Sparse codes priced as s dense columns per expert, not the zeroed rest.
    sparse_elems = int(weights.shape[0] * weights.shape[1] * s)
    factor_bytes = int((v.size + sparse_elems + u_stack.size + v_stack.size) * 2)
    return _score_pred(
        family="dictionary_plus_per_expert_sparse_residual",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(weights.size),
        rank=rank,
        fitted_dimension=rank,
        extra={
            "mechanism_source": f"{EBS_REL}#STORE-DICTIONARY-FAMILIES",
            "dictionary_atoms": rank,
            "sparse_columns_per_expert": s,
            "residual_rank": r_res,
            "algebra": "y_e = C_e_sparse @ (V @ x) + U_e @ (V_e @ x)",
        },
    )


def fit_sparse_residual_backbone(
    states: np.ndarray,
    weights: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
    *,
    rank: int,
    backbone: np.ndarray,
) -> dict[str, Any]:
    """Architecture shared-expert as named shared_op; not the mean expert (dead)."""
    if backbone.shape != weights.shape[1:]:
        raise ValueError(
            f"shared-expert backbone shape {backbone.shape} does not match "
            f"expert gate_up {weights.shape[1:]}"
        )
    shared_held = states[heldout_rows] @ backbone.T
    shared_fit = states[fit_rows] @ backbone.T
    u_stack, v_stack = residual_factors_batch(weights - backbone[None, :, :], rank)
    z_res_held = np.matmul(v_stack, states[heldout_rows].T).transpose(0, 2, 1)
    z_res_fit = np.matmul(v_stack, states[fit_rows].T).transpose(0, 2, 1)
    pred_held = shared_held[None, :, :] + np.matmul(u_stack, z_res_held.transpose(0, 2, 1)).transpose(0, 2, 1)
    pred_fit = shared_fit[None, :, :] + np.matmul(u_stack, z_res_fit.transpose(0, 2, 1)).transpose(0, 2, 1)
    factor_bytes = int((backbone.size + u_stack.size + v_stack.size) * 2)
    return _score_pred(
        family="sparse_residual_on_cheap_backbone",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(weights.size),
        rank=rank,
        fitted_dimension=rank,
        extra={
            "mechanism_source": f"{EBS_REL}#STORE-CONDITIONAL-RESIDUAL",
            "shared_op": "architecture shared_expert.gate_proj ⊕ up_proj; not the mean expert",
            "algebra": "y_e = W_shared @ x + U_e @ (V_e @ x)",
        },
    )


def attempt_full_dim_activation_map(
    states: np.ndarray,
    teacher: np.ndarray,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
) -> dict[str, Any]:
    """The NS-014 control: a full-width X→Y map fitted on captured rows.

    On this corpus n_fit=819 < d_in=2560, so the attempt must refuse. A
    cousin that first reduces X to rank r (the reference family) is a
    different fitted dimension and is not this refusal.
    """
    n_fit = int(fit_rows.sum())
    d_in = int(states.shape[1])
    if n_fit < d_in:
        return refused_row(
            family="full_dim_activation_map_negative_control",
            reason=(
                f"n_fit={n_fit} < d_in={d_in}: full-width activation map is "
                "underdetermined (NS-014 / NNS-007). Not scored."
            ),
            status="REFUSED_UNDERDETERMINED",
            rank=d_in,
            fitted_dimension=d_in,
            n_fit=n_fit,
        )
    # Reachable in unit tests with a skinny input.
    x_fit = states[fit_rows]
    pred_held = np.empty(
        (teacher.shape[0], int(heldout_rows.sum()), teacher.shape[2]), dtype=np.float32
    )
    pred_fit = np.empty(
        (teacher.shape[0], n_fit, teacher.shape[2]), dtype=np.float32
    )
    for expert in range(teacher.shape[0]):
        map_e, _, _, _ = np.linalg.lstsq(x_fit, teacher[expert, fit_rows], rcond=None)
        map_e = map_e.astype(np.float32, copy=False)
        pred_fit[expert] = x_fit @ map_e
        pred_held[expert] = states[heldout_rows] @ map_e
    factor_bytes = int(teacher.shape[0] * d_in * teacher.shape[2] * 2)
    return _score_pred(
        family="full_dim_activation_map_negative_control",
        pred_held=pred_held,
        pred_fit=pred_fit,
        teacher=teacher,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
        factor_bytes=factor_bytes,
        source_parameters=int(teacher.shape[0] * teacher.shape[2] * d_in),
        rank=d_in,
        fitted_dimension=d_in,
    )


# ---------------------------------------------------------------------------
# Harness judgement. Mismatch voids every rival number.
# ---------------------------------------------------------------------------


def _delta(got: float | None, expected: float | None) -> float | None:
    if got is None or expected is None:
        return None
    return abs(float(got) - float(expected))


def judge_harness(
    *,
    reproduced: Sequence[Mapping[str, Any]],
    committed: Sequence[Mapping[str, Any]],
    q4_error: float | None,
    expected_q4: float | None,
    heldout_abs_tol: float = HELDOUT_ABS_TOL,
    fit_abs_tol: float = FIT_ABS_TOL,
    bpw_abs_tol: float = BPW_ABS_TOL,
    q4_abs_tol: float = Q4_ABS_TOL,
) -> dict[str, Any]:
    if not committed:
        return {
            "ok": False,
            "reason": "committed reference rows absent; harness cannot be validated",
            "rivals_publishable": False,
            "q4_ok": False,
            "reference_ok": False,
        }
    if q4_error is None or expected_q4 is None:
        return {
            "ok": False,
            "reason": "Q4 control or committed Q4 number absent; harness is void",
            "rivals_publishable": False,
            "q4_ok": False,
            "reference_ok": False,
        }
    q4_delta = abs(float(q4_error) - float(expected_q4))
    q4_ok = q4_delta <= q4_abs_tol
    if len(reproduced) != len(committed):
        return {
            "ok": False,
            "reason": (
                f"reproduced {len(reproduced)} reference ranks, committed "
                f"{len(committed)}; harness is void"
            ),
            "rivals_publishable": False,
            "q4_ok": q4_ok,
            "q4_delta": q4_delta,
            "reference_ok": False,
        }
    per: list[dict[str, Any]] = []
    worst_held = 0.0
    worst_fit = 0.0
    worst_bpw = 0.0
    for got, exp in zip(reproduced, committed):
        if got.get("rank") != exp.get("rank"):
            return {
                "ok": False,
                "reason": f"rank order mismatch: got {got.get('rank')} committed {exp.get('rank')}",
                "rivals_publishable": False,
                "q4_ok": q4_ok,
                "reference_ok": False,
            }
        d_held_err = _delta(got.get("heldout_relative_fro_error"), exp.get("heldout_relative_fro_error"))
        d_held_cos = _delta(got.get("heldout_cosine"), exp.get("heldout_cosine"))
        d_fit = _delta(got.get("fit_relative_fro_error"), exp.get("fit_relative_fro_error"))
        d_bpw = _delta(got.get("diagnostic_factor_equivalent_bpw"), exp.get("diagnostic_factor_equivalent_bpw"))
        row_held = max(v for v in (d_held_err, d_held_cos) if v is not None) if (
            d_held_err is not None or d_held_cos is not None
        ) else float("inf")
        worst_held = max(worst_held, row_held)
        if d_fit is not None:
            worst_fit = max(worst_fit, d_fit)
        if d_bpw is not None:
            worst_bpw = max(worst_bpw, d_bpw)
        per.append(
            {
                "rank": got.get("rank"),
                "deltas": {
                    "heldout_relative_fro_error": d_held_err,
                    "heldout_cosine": d_held_cos,
                    "fit_relative_fro_error": d_fit,
                    "diagnostic_factor_equivalent_bpw": d_bpw,
                },
                "worst_heldout_abs": row_held,
            }
        )
    held_ok = worst_held <= heldout_abs_tol
    fit_ok = worst_fit <= fit_abs_tol
    bpw_ok = worst_bpw <= bpw_abs_tol
    reference_ok = held_ok and fit_ok and bpw_ok
    ok = q4_ok and reference_ok
    if not q4_ok:
        reason = (
            f"per-expert Q4 held-out error {q4_error} is {q4_delta} from committed "
            f"{expected_q4} (tol {q4_abs_tol}); the harness is wrong and rivals are void"
        )
    elif not held_ok:
        reason = (
            f"reference family held-out worst abs delta {worst_held} exceeds "
            f"{heldout_abs_tol}; rivals are void"
        )
    elif not bpw_ok:
        reason = (
            f"reference family factor-equivalent bpw delta {worst_bpw} exceeds "
            f"{bpw_abs_tol}; the byte identity drifted, rivals are void"
        )
    elif not fit_ok:
        reason = (
            f"reference family fit-set error delta {worst_fit} exceeds {fit_abs_tol}; "
            "rivals are void"
        )
    else:
        reason = (
            "Q4 control and reference family match the committed screen inside "
            f"held-out abs {heldout_abs_tol}, fit abs {fit_abs_tol}, Q4 abs {q4_abs_tol}"
        )
    return {
        "ok": ok,
        "reason": reason,
        "rivals_publishable": ok,
        "q4_ok": q4_ok,
        "q4_delta": q4_delta,
        "q4_abs_tol": q4_abs_tol,
        "expected_q4": expected_q4,
        "measured_q4": q4_error,
        "reference_ok": reference_ok,
        "heldout_ok": held_ok,
        "fit_ok": fit_ok,
        "bpw_ok": bpw_ok,
        "reference_worst_heldout_abs": worst_held,
        "reference_worst_fit_abs": worst_fit,
        "heldout_abs_tol": heldout_abs_tol,
        "fit_abs_tol": fit_abs_tol,
        "bpw_abs_tol": bpw_abs_tol,
        "per_rank": per,
    }


def none_mislabelled(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Every surface_gate_pass True must actually meet the contract inequalities."""
    if not contract.get("ok"):
        return {
            "ok": False,
            "reason": "contract unavailable; cannot certify labelling",
            "violations": ["no_contract"],
        }
    max_err = float(contract["max_heldout_relative_fro_error"])
    min_cos = float(contract["min_heldout_cosine"])
    must_beat = bool(contract["must_beat_per_expert_q4"])
    violations: list[str] = []
    n_pass = 0
    n_scored = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        flag = row.get("surface_gate_pass")
        if flag is True:
            n_pass += 1
            err = _as_float(row.get("heldout_relative_fro_error"))
            cos = _as_float(row.get("heldout_cosine"))
            q4 = _as_float(row.get("per_expert_q4_heldout_relative_fro_error"))
            beats = _as_bool(row.get("beats_per_expert_q4_on_heldout"))
            fam = str(row.get("family") or row.get("rank") or "?")
            if err is None or cos is None:
                violations.append(f"{fam}: pass flag with missing held-out numbers")
                continue
            if err > max_err:
                violations.append(f"{fam}: pass flag with held-out error {err} > {max_err}")
            if cos < min_cos:
                violations.append(f"{fam}: pass flag with cosine {cos} < {min_cos}")
            if must_beat and (q4 is None or beats is not True or err >= q4):
                violations.append(f"{fam}: pass flag without beating Q4")
            if row.get("scored") is False:
                violations.append(f"{fam}: pass flag on an unscored / refused row")
        if row.get("scored") is True:
            n_scored += 1
            if flag is True and row.get("labelled") != "SCORED_PASSED_CONTRACT":
                violations.append(f"{row.get('family')}: pass without SCORED_PASSED_CONTRACT")
            if flag is not True and row.get("labelled") == "SCORED_PASSED_CONTRACT":
                violations.append(f"{row.get('family')}: SCORED_PASSED_CONTRACT without pass flag")
    return {
        "ok": not violations,
        "reason": "no pass flag on a row that misses the contract" if not violations else "mislabelled pass",
        "violations": violations,
        "n_pass": n_pass,
        "n_scored": n_scored,
    }


# ---------------------------------------------------------------------------
# Screen orchestration
# ---------------------------------------------------------------------------


def _family_block(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("scored")]
    passed = [r for r in scored if r.get("surface_gate_pass") is True]
    beats = [r for r in scored if r.get("beats_per_expert_q4_on_heldout") is True]
    return {
        "family": family,
        "n_rows": len(rows),
        "n_scored": len(scored),
        "n_passed_contract": len(passed),
        "n_beats_q4": len(beats),
        "any_pass": bool(passed),
        "rows": rows,
        "wins_the_screen": False,  # a pass earns further measurement, not a win
    }


def screen_on_arrays(
    *,
    states: np.ndarray,
    weights: np.ndarray,
    contract: Mapping[str, Any],
    ranks: Sequence[int],
    committed: Sequence[Mapping[str, Any]] | None = None,
    fit_route_ids: Sequence[Sequence[int]] | None = None,
    expert_ids: Sequence[int] | None = None,
    backbone: np.ndarray | None = None,
    q4_group: int = RECOVERED_Q4_GROUP,
) -> dict[str, Any]:
    """Fit every named family on arrays already in memory. No specimen I/O."""
    if not contract.get("ok"):
        return {
            "status": "REFUSED_NO_CONTRACT",
            "reason": contract.get("reason"),
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "flat_rows": [],
            "rivals_published": False,
        }
    n_rows = int(states.shape[0])
    if n_rows < 2:
        return {
            "status": "REFUSED_INSUFFICIENT_TEACHER_COVERAGE",
            "reason": "need at least two teacher rows for a holdout",
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "flat_rows": [],
            "rivals_published": False,
        }
    fit_rows, heldout_rows = heldout_split(n_rows)
    n_fit = int(fit_rows.sum())
    teacher = projected_outputs(states, weights)
    source_parameters = int(weights.size)

    q4_row = fit_quant_control(
        states, weights, teacher, fit_rows, heldout_rows, bits=4, family="per_expert_q4_control",
        group=q4_group,
    )
    q4_error = float(q4_row["heldout_relative_fro_error"])
    attach_gates(q4_row, q4_error=q4_error, contract=contract)

    # Reference family first. Failure to match committed numbers voids rivals.
    reference_rows: list[dict[str, Any]] = []
    for rank in ranks:
        try:
            got = fit_shared_latent_program(
                states,
                weights,
                rank=int(rank),
                fit_rows=fit_rows,
                heldout_rows=heldout_rows,
                teacher=teacher,
            )
        except UnderdeterminedFitError as exc:
            got = refused_row(
                family=REFERENCE_KIND,
                reason=str(exc),
                status="REFUSED_UNDERDETERMINED",
                rank=int(rank),
                fitted_dimension=int(rank),
                n_fit=n_fit,
            )
        else:
            got["family"] = REFERENCE_KIND
            attach_gates(got, q4_error=q4_error, contract=contract)
        reference_rows.append(got)

    expected_q4 = None
    if committed:
        expected_q4 = committed[0].get("per_expert_q4_heldout_relative_fro_error")
    harness = judge_harness(
        reproduced=reference_rows,
        committed=list(committed or []),
        q4_error=q4_error,
        expected_q4=_as_float(expected_q4) if expected_q4 is not None else q4_error,
        # When no committed table is supplied (unit tests), compare Q4 to itself
        # so the harness is about the reference algebra, not a missing receipt.
    )
    if not committed:
        # Unit-test path: Q4 matches itself; reference is not checked against a receipt.
        harness = {
            "ok": all(r.get("scored") for r in reference_rows),
            "reason": "no committed screen rows; harness self-check only, not a published validation",
            "rivals_publishable": True,
            "q4_ok": True,
            "q4_delta": 0.0,
            "measured_q4": q4_error,
            "expected_q4": q4_error,
            "reference_ok": True,
            "published_validation": False,
        }

    under_full = attempt_full_dim_activation_map(states, teacher, fit_rows, heldout_rows)
    if under_full.get("scored"):
        attach_gates(under_full, q4_error=q4_error, contract=contract)

    families: list[dict[str, Any]] = [
        _family_block("per_expert_q4_control", [q4_row]),
        _family_block(REFERENCE_KIND, reference_rows),
    ]
    flat: list[dict[str, Any]] = [q4_row, *reference_rows, under_full]

    if not harness.get("rivals_publishable"):
        return {
            "status": "HARNESS_INVALID_RIVALS_VOID",
            "reason": harness.get("reason"),
            "harness": harness,
            "q4_error": q4_error,
            "n_fit": n_fit,
            "n_heldout": int(heldout_rows.sum()),
            "families": families,
            "flat_rows": flat,
            "underdetermined_control": under_full,
            "rivals_published": False,
        }

    q3_row = fit_quant_control(
        states, weights, teacher, fit_rows, heldout_rows, bits=3, family="per_expert_q3_control",
        group=q4_group,
    )
    attach_gates(q3_row, q4_error=q4_error, contract=contract)
    q2_row = fit_quant_control(
        states, weights, teacher, fit_rows, heldout_rows, bits=2, family="per_expert_q2_control",
        group=q4_group,
    )
    attach_gates(q2_row, q4_error=q4_error, contract=contract)
    families.append(_family_block("per_expert_q3_control", [q3_row]))
    families.append(_family_block("per_expert_q2_control", [q2_row]))
    flat.extend([q3_row, q2_row])

    def _rank_family(name: str, fn, **kwargs: Any) -> None:
        rows: list[dict[str, Any]] = []
        for rank in ranks:
            try:
                row = fn(
                    states, weights, teacher, fit_rows, heldout_rows, rank=int(rank), **kwargs
                )
            except (ValueError, UnderdeterminedFitError) as exc:
                row = refused_row(
                    family=name,
                    reason=str(exc),
                    status=(
                        "REFUSED_UNDERDETERMINED"
                        if isinstance(exc, UnderdeterminedFitError)
                        else "REFUSED_UNAVAILABLE_RANK"
                    ),
                    rank=int(rank),
                    fitted_dimension=int(rank),
                    n_fit=n_fit,
                )
            else:
                attach_gates(row, q4_error=q4_error, contract=contract)
            rows.append(row)
        families.append(_family_block(name, rows))
        flat.extend(rows)

    _rank_family("common_left_subspace_plus_expert_local_core", fit_common_left)
    _rank_family("common_right_subspace_plus_expert_local_core", fit_common_right)

    labels = None
    if fit_route_ids is not None and expert_ids is not None:
        index = {int(e): i for i, e in enumerate(expert_ids)}
        fit_routes = [
            routes
            for keep, routes in zip(fit_rows.tolist(), fit_route_ids)
            if keep
        ]
        labels = route_cluster_labels(fit_routes, index, k=min(CLUSTER_K, weights.shape[0]))
        _rank_family(
            "clustered_subspaces_route_conditioned",
            fit_clustered,
            labels=labels,
        )
    else:
        families.append(
            _family_block(
                "clustered_subspaces_route_conditioned",
                [
                    refused_row(
                        family="clustered_subspaces_route_conditioned",
                        reason="route ids for the fit split were not supplied; clustering is not invented",
                        status="REFUSED_ABSENT_INPUT",
                    )
                ],
            )
        )
        flat.append(families[-1]["rows"][0])

    _rank_family("dictionary_plus_per_expert_sparse_residual", fit_dictionary_sparse_residual)
    _rank_family("expert_local_small_core_plus_shared_decoder", fit_small_core)

    if backbone is None:
        bb_rows = [
            refused_row(
                family="sparse_residual_on_cheap_backbone",
                reason=(
                    "architecture shared-expert backbone absent; refusing rather than "
                    "substituting the mean expert (SCAR-RAW-GLOBAL-SIMILARITY)"
                ),
                status="REFUSED_ABSENT_INPUT",
                rank=int(r),
                fitted_dimension=int(r),
                n_fit=n_fit,
            )
            for r in ranks
        ]
        families.append(_family_block("sparse_residual_on_cheap_backbone", bb_rows))
        flat.extend(bb_rows)
    else:
        _rank_family(
            "sparse_residual_on_cheap_backbone",
            fit_sparse_residual_backbone,
            backbone=backbone,
        )

    labelling = none_mislabelled(flat, contract)
    any_pass = any(r.get("surface_gate_pass") is True for r in flat if r.get("scored"))
    return {
        "status": "RIVAL_FAMILIES_SCORED_OFFLINE_META_SURFACE",
        "reason": (
            "families scored on the same split and contract as the committed L4 screen; "
            "a pass earns further measurement, not a win"
        ),
        "harness": harness,
        "q4_error": q4_error,
        "n_fit": n_fit,
        "n_heldout": int(heldout_rows.sum()),
        "n_teacher_rows": n_rows,
        "n_experts": int(weights.shape[0]),
        "source_parameters": source_parameters,
        "families": families,
        "flat_rows": flat,
        "underdetermined_control": under_full,
        "labelling": labelling,
        "any_family_passed_contract": any_pass,
        "rivals_published": True,
        "cluster_k": CLUSTER_K if labels is not None else None,
        "occupied_cluster_ids": (
            sorted({int(x) for x in labels.tolist()}) if labels is not None else None
        ),
    }


def load_inputs() -> dict[str, Any]:
    screen, screen_loc = load_named(SCREEN_REL)
    teacher, teacher_loc = load_named(TEACHER_REL)
    state_path = find_path(STATE_REL)
    return {
        "screen": screen,
        "screen_loc": screen_loc,
        "teacher": teacher,
        "teacher_loc": teacher_loc,
        "state_path": state_path,
    }


def _route_ids_from_teacher(teacher: Mapping[str, Any]) -> list[list[int]] | None:
    rows = teacher.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    out: list[list[int]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        ids = row.get("route_ids")
        if not isinstance(ids, list) or not ids:
            return None
        if any(isinstance(v, bool) or not isinstance(v, int) for v in ids):
            return None
        out.append([int(v) for v in ids])
    return out


def _shared_backbone(root: Path, expert_shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        gate = read_bf16_matrix(locate_tensor(root, TENSOR_SHARED_GATE))
        up = read_bf16_matrix(locate_tensor(root, TENSOR_SHARED_UP))
    except (AbsentInputError, ValueError, OSError):
        return None
    if gate.ndim != 2 or up.ndim != 2:
        return None
    stacked = np.concatenate([gate, up], axis=0)
    if tuple(stacked.shape) != tuple(expert_shape):
        return None
    return np.ascontiguousarray(stacked.astype(np.float32, copy=False))


def run_screen() -> dict[str, Any]:
    inputs = load_inputs()
    screen = inputs["screen"]
    teacher = inputs["teacher"]
    state_path = inputs["state_path"]
    cited = {
        "screen": {"path": inputs["screen_loc"], "present": isinstance(screen, dict)},
        "teacher": {"path": inputs["teacher_loc"], "present": isinstance(teacher, dict)},
        "state": {
            "path": str(state_path) if state_path is not None else None,
            "present": state_path is not None and state_path.is_file(),
        },
    }
    contract = contract_from_screen(screen)
    if not contract["ok"]:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": contract["reason"],
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    if not isinstance(screen, dict):
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": f"{SCREEN_REL} not reachable on disk or HEAD",
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    if state_path is None:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": f"{STATE_REL} not reachable; the f32 teacher state is not inferred",
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    width = _as_int((screen.get("teacher_trace") or {}).get("width")) or RECOVERED_HIDDEN
    try:
        states, state_meta = load_teacher_states(state_path, width=width)
    except (AbsentInputError, ValueError) as exc:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": str(exc),
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    cited["state"]["sha256"] = state_meta["sha256"]
    cited["state"]["rows"] = state_meta["rows"]
    expected_sha = (screen.get("teacher_trace") or {}).get("files")
    bind_sha = None
    if isinstance(expected_sha, list) and expected_sha and isinstance(expected_sha[0], Mapping):
        bind_sha = expected_sha[0].get("sha256")
    cap = (screen.get("teacher_trace") or {}).get("capture_binding") or {}
    if bind_sha is None and isinstance(cap, Mapping):
        bind_sha = cap.get("state_sha256")
    if isinstance(teacher, dict):
        trace = teacher.get("teacher_trace") or {}
        if bind_sha is None:
            bind_sha = trace.get("state_sha256")
        if teacher.get("schema") != "hawking.flash.meta_teacher_trace.v1":
            return {
                "status": "REFUSED_ABSENT_INPUT",
                "reason": "teacher capture schema is not hawking.flash.meta_teacher_trace.v1",
                "cited_inputs": cited,
                "contract": contract,
                "harness": {"ok": False, "rivals_publishable": False},
                "families": [],
                "rivals_published": False,
            }
        if teacher.get("status") != "CAPTURED_SOURCE_MLP_INPUT_NOT_CAPABILITY_PROVEN":
            return {
                "status": "REFUSED_ABSENT_INPUT",
                "reason": "teacher capture is not a source MLP-input capture",
                "cited_inputs": cited,
                "contract": contract,
                "harness": {"ok": False, "rivals_publishable": False},
                "families": [],
                "rivals_published": False,
            }
    if bind_sha and bind_sha != state_meta["sha256"]:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": (
                f"teacher state sha256 {state_meta['sha256']} does not match the "
                f"bound capture {bind_sha}"
            ),
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    if state_meta["rows"] < RECOVERED_MIN_TEACHER_ROWS:
        return {
            "status": "REFUSED_INSUFFICIENT_TEACHER_COVERAGE",
            "reason": (
                f"{state_meta['rows']} unique teacher rows < {RECOVERED_MIN_TEACHER_ROWS} "
                "(reference screen default min_teacher_rows)"
            ),
            "cited_inputs": cited,
            "contract": contract,
            "state": state_meta,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }

    experts = list((screen.get("source") or {}).get("route_ids") or [])
    if not experts:
        union = cap.get("route_union") if isinstance(cap, Mapping) else None
        if isinstance(union, list) and union:
            experts = [int(v) for v in union]
    if not experts:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": "screen.source.route_ids absent; the 314-expert union is not guessed",
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }

    model_root = Path(str((screen.get("source") or {}).get("root") or DEFAULT_MODEL))
    if not model_root.is_dir():
        if DEFAULT_MODEL.is_dir():
            model_root = DEFAULT_MODEL
        else:
            return {
                "status": "REFUSED_ABSENT_INPUT",
                "reason": f"specimen root not on disk: {model_root}",
                "cited_inputs": cited,
                "contract": contract,
                "harness": {"ok": False, "rivals_publishable": False},
                "families": [],
                "rivals_published": False,
            }
    expected_index = (screen.get("source") or {}).get("index_sha256")
    index_path = model_root / "model.safetensors.index.json"
    if not index_path.is_file():
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": f"specimen index not on disk: {index_path}",
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    index_sha = sha256_file(index_path)
    if isinstance(expected_index, str) and expected_index and expected_index != index_sha:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": (
                f"specimen index sha256 {index_sha} does not match the screen's "
                f"{expected_index}"
            ),
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }

    try:
        loc = locate_tensor(model_root, TENSOR_GATE_UP)
        weights = read_bf16_experts(loc, experts)
    except (AbsentInputError, ValueError, OSError) as exc:
        return {
            "status": "REFUSED_ABSENT_INPUT",
            "reason": str(exc),
            "cited_inputs": cited,
            "contract": contract,
            "harness": {"ok": False, "rivals_publishable": False},
            "families": [],
            "rivals_published": False,
        }
    backbone = _shared_backbone(model_root, tuple(weights.shape[1:]))
    fit_route_ids = _route_ids_from_teacher(teacher) if isinstance(teacher, dict) else None
    if fit_route_ids is not None and len(fit_route_ids) != states.shape[0]:
        # Dedup dropped rows; refuse clustering rather than realign by guess.
        fit_route_ids = None

    committed = committed_reference_rows(screen)
    ranks = ranks_from_screen(screen)
    result = screen_on_arrays(
        states=states,
        weights=weights,
        contract=contract,
        ranks=ranks,
        committed=committed,
        fit_route_ids=fit_route_ids,
        expert_ids=experts,
        backbone=backbone,
    )
    result["cited_inputs"] = cited
    result["contract"] = contract
    result["state"] = state_meta
    result["specimen"] = {
        "root": str(model_root),
        "index_sha256": index_sha,
        "tensor": TENSOR_GATE_UP,
        "n_experts": len(experts),
        "weight_shape": [int(x) for x in weights.shape],
        "shared_expert_backbone_present": backbone is not None,
    }
    result["ranks"] = list(ranks)
    return result


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, str]]:
    return [
        {
            "path": SCREEN_TOOL_REL,
            "role": (
                "reference screen: holdout % 5, Q4 group-64, relative Frobenius, "
                "cosine, factor-byte bpw identity, surface-gate strings"
            ),
        },
        {
            "path": SCREEN_REL,
            "role": "committed L4 REAL1024 numbers the harness must reproduce first",
        },
        {
            "path": TEACHER_REL,
            "role": "1024 unique source-BF16 layer-4 mlp_input rows + per-row routes",
        },
        {
            "path": STATE_REL,
            "role": "raw f32 teacher state the screen fitted",
        },
        {
            "path": EBS_REL,
            "role": "named rival mechanisms (left/right, clustered, dictionary, sandwich, residual)",
        },
        {
            "path": SCHOOLS_REL,
            "role": "independent Q4 control pricing (4.25 bpw group-64) and program-family names",
        },
        {
            "path": REPLAN_REL,
            "role": "contract-from-screen, NS-014 underdetermination, 'a pass is not a win'",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "No sidecar module had screened any rival family on the real 1024-row L4 teacher surface.",
        "Harness self-check: reproduce the committed shared-latent ranks and Q4 control before publishing rivals.",
        "NS-014 applied as a refusal (full-width activation map, rank > n_fit), not a starved score.",
        "Pass labelling is certified against the screen's own inequalities so a miss cannot be named a pass.",
    ]


def negative_findings() -> list[str]:
    return [
        "Fit quality on one organ, one surface, one split. Not coherence, not capability, not physical EBPW.",
        "A family that beats Q4 on this surface still has to meet cosine and error gates; beating Q4 is not a win.",
        "Cluster K is 8 with a fixed seed; other K were not swept.",
        "Dictionary sparsity and residual rank are rank//4; other sparsities were not swept.",
        "down_proj, other layers, and router/hidden/terminal-logit surfaces were not fitted.",
        "orchestration.BINDINGS is outside this lane's WRITE list; the frontier below is declared, not wired.",
    ]


def build() -> Path:
    result = run_screen()
    families = result.get("families") or []
    labelling = result.get("labelling") or {}
    harness = result.get("harness") or {}
    any_pass = bool(result.get("any_family_passed_contract"))
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Screen rival codec families on the real L4 1024-row teacher corpus "
            "under the committed coherence contract, after reproducing the one "
            "family that has already been measured."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "claim_boundary": (
            "Static sidecar artifact. Fit quality on a bounded teacher hidden-state "
            "surface. No hardware measurement. No physical EBPW. Passing this screen "
            "does not prove a functional representation, serialized bytes, route "
            "stability, capability, residency, GPU latency, or TPS. No family wins "
            "by passing — it only earns the right to be measured further."
        ),
        "cited_inputs": result.get("cited_inputs"),
        "contract": result.get("contract"),
        "harness": harness,
        "specimen": result.get("specimen"),
        "state": result.get("state"),
        "ranks": result.get("ranks"),
        "q4_error": result.get("q4_error"),
        "n_fit": result.get("n_fit"),
        "n_heldout": result.get("n_heldout"),
        "rivals_published": bool(result.get("rivals_published")),
        "any_family_passed_contract": any_pass,
        "a_pass_is_not_a_win": True,
        "labelling": labelling,
        "underdetermined_control": result.get("underdetermined_control"),
        "families": families,
        "family_ids": [f.get("family") for f in families],
        "promotion_allowed": False,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": {
            "entry_point": "tools.future.rival_codec_screen.build()",
            "workunit": (
                "one CPU_ANALYSIS unit; reproduce the committed L4 REAL1024 "
                "shared-latent screen, then fit named rival families on the same "
                "split; no GPU, no lease"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
            "fails_closed": (
                "absent corpus/specimen/contract is a recorded refusal; "
                "n_fit < fitted dimension is REFUSED_UNDERDETERMINED not scored; "
                "a harness that misses the committed numbers voids every rival; "
                "surface_gate_pass is true only when the screen's own inequalities hold"
            ),
            "discoverable": True,
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
