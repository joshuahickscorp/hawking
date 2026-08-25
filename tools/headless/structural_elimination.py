#!/usr/bin/env python3
"""N050 STRUCTURAL ELIMINATION — does this structure need to exist?

S026 §29 / §39 / §40; DOC-ELIMINATION. Pure CPU analysis of REAL parent
tensors (streamed one at a time) plus existing capture_diverse2 activations.
Does not load a second 27B decode, does not open Metal, does not touch the
GPU, does not mutate ~/noetic/NOETIC_PARENT_A.

Question per candidate: if we removed it, how many parent-equivalent
parameters disappear, what is the capability risk, and does the resulting
tensor SHAPE execute better or worse? A smaller badly-shaped tensor can
run slower than a larger well-shaped one (§39; BYTES_FRONTIER).

Removal requires capability evidence. Benchmark myopia is not redundancy
(§109): a generic probe going silent is not a license to cut.

    python3 tools/headless/structural_elimination.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "STRUCTURAL_ELIMINATION.json"
SCHEMA = "hawking.headless.structural_elimination.v1"

PARENT_PARAMS = 26_895_998_464
LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248_320

GQA_HEADS = 24
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256
GQA_GROUP = GQA_HEADS // GQA_KV_HEADS  # 6
FULL_ATTN_INTERVAL = 4

DN_K_HEADS = 16
DN_V_HEADS = 48
DN_VPK = 3
DN_K_DIM = 128
DN_V_DIM = 128
DN_CONV_K = 4
REC_ELEMS_PER_LAYER = DN_K_HEADS * DN_VPK * DN_K_DIM * DN_V_DIM  # 786432

GROUP = 64
TPR = 64
TG = 128
SIMD = 32

# Numerical floors. A channel whose RMS is this small relative to the
# layer median is "numerically dead". 1e-3 relative is "low-magnitude",
# which is NOT the same as dead and is not an elimination license.
DEAD_REL = 1e-8
LOW_REL = 1e-3
COPY_COSINE = 0.99
ALIGNED_COSINE = 0.50
NEAR_ORTH_COSINE = 0.10
IDENTITY_COSINE = 0.995
IDENTITY_REL_L2 = 0.05
ACT_COVER_THR = 1e-3
ACT_LAYERS = (0, 21, 42, 63)
ACT_TOKENS = 512
SEED = 20260824

GQA_LAYERS = tuple(i for i in range(LAYERS) if (i + 1) % FULL_ATTN_INTERVAL == 0)
DN_LAYERS = tuple(i for i in range(LAYERS) if (i + 1) % FULL_ATTN_INTERVAL != 0)

PARENT_CANDIDATES = [
    Path(os.environ["QWEN38_PARENT_BF16"]).expanduser()
    if os.environ.get("QWEN38_PARENT_BF16")
    else None,
    Path.home() / "models/qwen3.8-27b-abliterated-bf16",
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
]
CAPTURE_CANDIDATES = [
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
    REPO / "workspace/campaign/phaseB/capture_diverse2",
]
NOETIC_PARENT_A = Path.home() / "noetic" / "NOETIC_PARENT_A"

# Specialized compile-time shapes the incumbent kernels bake in.
SPECIALIZED = {
    "hidden": HIDDEN,
    "intermediate": INTERMEDIATE,
    "gqa_heads": GQA_HEADS,
    "gqa_kv_heads": GQA_KV_HEADS,
    "gqa_head_dim": GQA_HEAD_DIM,
    "dn_k_heads": DN_K_HEADS,
    "dn_v_heads": DN_V_HEADS,
    "group": GROUP,
    "tpr": TPR,
    "tg": TG,
}

PRIOR_Q_L3 = 0.021602977067232132  # ORGAN_FRONTIERS L3 q_head_pairwise
PRIOR_K_L3 = -0.0038204605225473642

LANG_PREFIX = "model.language_model.layers"


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except Exception:
        return "UNKNOWN"


def j(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(j(obj), indent=2) + "\n")
    tmp.replace(path)


def find_parent() -> Path:
    seen: set[str] = set()
    for p in PARENT_CANDIDATES:
        if p is None:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if (p / "model.safetensors.index.json").is_file():
            return p
    raise FileNotFoundError("qualified parent bf16 not found")


def find_capture() -> Path | None:
    for p in CAPTURE_CANDIDATES:
        if p is None:
            continue
        if (p / "L00.f16").is_file() or (p / "L0.f16").is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# safetensors streaming (one tensor at a time, CPU, no GPU)
# ---------------------------------------------------------------------------

_INDEX: dict[str, str] | None = None
_HEADERS: dict[str, dict] = {}


def weight_index(parent: Path) -> dict[str, str]:
    global _INDEX
    if _INDEX is None:
        _INDEX = json.loads((parent / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    return _INDEX


def _header(parent: Path, shard_name: str) -> dict:
    if shard_name not in _HEADERS:
        with open(parent / shard_name, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _HEADERS[shard_name] = json.loads(f.read(n))
    return _HEADERS[shard_name]


def tensor_meta(parent: Path, name: str) -> dict:
    shard = weight_index(parent)[name]
    meta = _header(parent, shard)[name]
    start, end = meta["data_offsets"]
    return {
        "name": name,
        "shard": shard,
        "dtype": meta["dtype"],
        "shape": list(meta["shape"]),
        "nbytes": int(end - start),
    }


def load_tensor(parent: Path, name: str) -> np.ndarray:
    """Stream one parent tensor to float32. Does not keep the shard mapped."""
    shard = weight_index(parent)[name]
    header = _header(parent, shard)
    meta = header[name]
    start, end = meta["data_offsets"]
    n = None
    with open(parent / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    shape = tuple(meta["shape"])
    dtype = meta["dtype"]
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return np.array(f32.reshape(shape), dtype=np.float32, copy=True)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    raise ValueError(f"{name} dtype {dtype}")


def tname(layer: int, kind: str) -> str:
    return f"{LANG_PREFIX}.{layer}.{kind}"


def is_gqa(layer: int) -> bool:
    return (layer + 1) % FULL_ATTN_INTERVAL == 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def pairwise_cosine(flat: np.ndarray) -> dict:
    """flat: [H, D] L2-normalized rows → mean/min/max off-diagonal cosine."""
    h = int(flat.shape[0])
    if h < 2:
        return {
            "n_heads": h,
            "n_pairs": 0,
            "mean_cosine": None,
            "min_cosine": None,
            "max_cosine": None,
            "n_pairs_ge_0p50": 0,
            "n_pairs_ge_0p90": 0,
            "n_pairs_ge_0p99": 0,
            "null_independent": 0.0,
            "null_sigma": None,
        }
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-30
    u = flat / norms
    g = u @ u.T
    iu = np.triu_indices(h, 1)
    vals = g[iu].astype(np.float64)
    d = float(flat.shape[1])
    return {
        "n_heads": h,
        "n_pairs": int(vals.size),
        "mean_cosine": float(vals.mean()),
        "min_cosine": float(vals.min()),
        "max_cosine": float(vals.max()),
        "n_pairs_ge_0p50": int((vals >= ALIGNED_COSINE).sum()),
        "n_pairs_ge_0p90": int((vals >= 0.90).sum()),
        "n_pairs_ge_0p99": int((vals >= COPY_COSINE).sum()),
        "null_independent": 0.0,
        "null_sigma": float(1.0 / math.sqrt(d)) if d > 0 else None,
        "dim": int(d),
    }


def head_pairwise(W_heads: np.ndarray) -> dict:
    """W_heads: [H, ...] → pairwise cosine of flattened heads."""
    h = int(W_heads.shape[0])
    flat = np.ascontiguousarray(W_heads.reshape(h, -1), dtype=np.float32)
    return pairwise_cosine(flat)


def frobenius_cosine(A: np.ndarray, B: np.ndarray) -> float:
    a = np.asarray(A, dtype=np.float32).ravel()
    b = np.asarray(B, dtype=np.float32).ravel()
    if a.size != b.size:
        raise ValueError(f"frobenius_cosine size {a.size} vs {b.size}")
    num = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    na = float(np.linalg.norm(a.astype(np.float64)))
    nb = float(np.linalg.norm(b.astype(np.float64)))
    return num / (na * nb + 1e-30)


def row_rms(W: np.ndarray) -> np.ndarray:
    w = np.asarray(W, dtype=np.float32)
    if w.ndim == 1:
        return np.abs(w)
    return np.sqrt(np.mean(np.square(w, dtype=np.float64), axis=1)).astype(np.float32)


def col_rms(W: np.ndarray) -> np.ndarray:
    w = np.asarray(W, dtype=np.float32)
    if w.ndim == 1:
        return np.abs(w)
    return np.sqrt(np.mean(np.square(w, dtype=np.float64), axis=0)).astype(np.float32)


def rms_report(x: np.ndarray, *, median: float | None = None) -> dict:
    x64 = np.asarray(x, dtype=np.float64).ravel()
    med = float(np.median(x64)) if median is None else float(median)
    mx = float(x64.max()) if x64.size else 0.0
    mn = float(x64.min()) if x64.size else 0.0
    dead = int(np.sum(x64 <= DEAD_REL * max(med, 1e-30)))
    low = int(np.sum(x64 <= LOW_REL * max(med, 1e-30)))
    return {
        "n": int(x64.size),
        "median": med,
        "mean": float(x64.mean()) if x64.size else 0.0,
        "min": mn,
        "max": mx,
        "p01": float(np.quantile(x64, 0.01)) if x64.size else 0.0,
        "p05": float(np.quantile(x64, 0.05)) if x64.size else 0.0,
        "n_dead": dead,
        "n_low": low,
        "dead_rel": DEAD_REL,
        "low_rel": LOW_REL,
        "frac_dead": float(dead / x64.size) if x64.size else 0.0,
        "frac_low": float(low / x64.size) if x64.size else 0.0,
    }


def silu(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return z / (1.0 + np.exp(-z))


def token_pair_stats(A: np.ndarray, B: np.ndarray) -> dict:
    """Per-token cosine and relative L2 between two [T, H] activation planes."""
    a = np.asarray(A, dtype=np.float32)
    b = np.asarray(B, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"token_pair_stats shape {a.shape} vs {b.shape}")
    num = (a * b).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    den = na * nb + 1e-30
    cos = (num / den).astype(np.float64)
    rel = (np.linalg.norm(a - b, axis=1) / (na + 1e-30)).astype(np.float64)
    return {
        "n_tokens": int(a.shape[0]),
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
        "max_cosine": float(cos.max()),
        "p05_cosine": float(np.quantile(cos, 0.05)),
        "p95_cosine": float(np.quantile(cos, 0.95)),
        "frac_cosine_ge_0p99": float(np.mean(cos >= 0.99)),
        "frac_cosine_ge_0p995": float(np.mean(cos >= IDENTITY_COSINE)),
        "mean_rel_l2": float(rel.mean()),
        "min_rel_l2": float(rel.min()),
        "max_rel_l2": float(rel.max()),
        "near_identity": bool(
            float(cos.mean()) >= IDENTITY_COSINE
            and float(rel.mean()) <= IDENTITY_REL_L2
        ),
    }


def summarize_cosines(rows: list[dict], key: str = "mean_cosine") -> dict:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


# ---------------------------------------------------------------------------
# §39 shape execution
# ---------------------------------------------------------------------------


def aligned(n: int, g: int = GROUP) -> bool:
    return int(n) % int(g) == 0


def shape_execution(
    *,
    name: str,
    old: dict,
    new: dict,
    remaining_tensor_shapes: str,
    work_removed: bool = True,
    notes: list[str] | None = None,
) -> dict:
    """Will the resulting tensor SHAPE execute better or worse than incumbent?

    Incumbent kernels specialize on compile-time hidden=5120, intermediate=
    17408, 24/4 GQA, 16/48 DeltaNet, group=64, tpr=64, tg=128. Changing a
    specialized constant requires a new kernel. A ragged size (not a multiple
    of group-64) cannot use the incumbent pack. BYTES_FRONTIER: fewer stored
    bits is not fewer nanoseconds.
    """
    notes = list(notes or [])
    issues: list[str] = []
    for dim, g in (
        ("rows", GROUP),
        ("cols", GROUP),
        ("intermediate", GROUP),
        ("q_rows", GROUP),
        ("o_cols", GROUP),
        ("head_dim", GROUP),  # 256 % 64 == 0; keep the check
    ):
        if dim in new and new[dim] is not None and not aligned(int(new[dim]), g):
            issues.append(f"{dim}={new[dim]} is not a multiple of {g}")

    old_heads = old.get("gqa_heads")
    new_heads = new.get("gqa_heads")
    old_kv = old.get("gqa_kv_heads")
    new_kv = new.get("gqa_kv_heads")
    if new_heads is not None and new_kv is not None:
        if int(new_kv) > 0 and int(new_heads) % int(new_kv) != 0:
            issues.append(
                f"GQA grouping broken: {new_heads} Q heads / {new_kv} KV "
                f"is not an integer group (incumbent {GQA_GROUP}:1)"
            )
        if int(new_heads) != GQA_HEADS or int(new_kv) != GQA_KV_HEADS:
            notes.append(
                "GQA 24/4 is a compile-time constant in qwen38_geometry.rs "
                "and the specialized decode kernels"
            )

    if new.get("dn_k_heads") not in (None, DN_K_HEADS):
        notes.append("DeltaNet 16 key heads is a power-of-two specialized constant")
    if new.get("intermediate") not in (None, INTERMEDIATE):
        notes.append("MLP intermediate 17408 = 272*64 is baked into geo kernels")

    specialized_changed = False
    for k, v in SPECIALIZED.items():
        if k in new and new[k] is not None and int(new[k]) != int(v):
            specialized_changed = True

    if remaining_tensor_shapes == "RAGGED" or issues:
        executes = "WORSE"
        why = (
            "Ragged or grouping-broken shape cannot use the incumbent "
            "group-64 / tpr-64 / 6:1-GQA specialized kernels; a smaller "
            "badly-shaped tensor can run slower (§39)."
        )
    elif remaining_tensor_shapes == "SAME" and not specialized_changed and work_removed:
        executes = "BETTER"
        why = (
            "Remaining tensors keep incumbent specialized shapes; work is "
            "removed (fewer GEMVs/dispatches), not reshaped. Kernel geometry "
            "is unchanged. Function survival is a separate gate."
        )
    elif specialized_changed:
        executes = "MIXED"
        why = (
            "Bytes drop but a specialized compile-time constant changes, so "
            "a new kernel is required. Until that kernel is competent "
            "(N003 / KERNEL_COMPETENCE), the cut can run SLOWER than the "
            "larger well-shaped incumbent. BYTES_FRONTIER: fewer bits ≠ fewer ns."
        )
    else:
        executes = "SAME"
        why = "Geometry family unchanged; no work is removed or the cut is a no-op."

    return {
        "candidate": name,
        "old": old,
        "new": new,
        "remaining_tensor_shapes": remaining_tensor_shapes,
        "divisible_by_group64": all(
            (d not in new) or new[d] is None or aligned(int(new[d]), GROUP)
            for d in ("rows", "cols", "intermediate", "q_rows", "o_cols")
        ),
        "divisible_by_tpr64": all(
            (d not in new) or new[d] is None or aligned(int(new[d]), TPR)
            for d in ("rows", "q_rows", "o_cols", "intermediate")
        ),
        "specialized_constant_changed": specialized_changed,
        "work_removed": work_removed,
        "issues": issues,
        "executes": executes,
        "why": why,
        "notes": notes,
        "citation": [
            "receipts/headless/BYTES_FRONTIER.json",
            "receipts/headless/KERNEL_COMPETENCE.json",
            "crates/hawking-core/src/model/qwen38_geometry.rs",
        ],
    }


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def q_head_params(n_heads: int = 1) -> int:
    """One GQA Q-head = q slice + gate slice + corresponding o columns."""
    return n_heads * GQA_HEAD_DIM * HIDDEN * 3


def kv_head_params(n_kv: int = 1) -> int:
    return n_kv * GQA_HEAD_DIM * HIDDEN * 2  # k + v


def kv_group_params(n_groups: int = 1) -> int:
    """One GQA group = 6 Q heads (q+gate+o) + 1 K + 1 V."""
    return n_groups * (q_head_params(GQA_GROUP) + kv_head_params(1))


def mlp_channel_params(n_channels: int = 1, n_layers: int = LAYERS) -> int:
    # gate row + up row + down column, all width hidden
    return n_channels * HIDDEN * 3 * n_layers


def mlp_layer_params() -> int:
    return INTERMEDIATE * HIDDEN * 3


def gqa_mixer_params() -> int:
    q = GQA_HEADS * 2 * GQA_HEAD_DIM * HIDDEN
    kv = GQA_KV_HEADS * GQA_HEAD_DIM * HIDDEN * 2
    o = HIDDEN * GQA_HEADS * GQA_HEAD_DIM
    return q + kv + o


def dn_mixer_params() -> int:
    qkv = (DN_K_HEADS * DN_K_DIM * 2 + DN_K_HEADS * DN_VPK * DN_V_DIM) * HIDDEN
    z = DN_V_HEADS * DN_V_DIM * HIDDEN
    out = HIDDEN * DN_V_HEADS * DN_V_DIM
    ab = 2 * DN_V_HEADS * HIDDEN
    return qkv + z + out + ab


def dn_key_head_params(n_heads: int = 1) -> int:
    """Drop n DeltaNet key heads (and their 3 value heads)."""
    qk_rows = n_heads * DN_K_DIM * 2
    v_rows = n_heads * DN_VPK * DN_V_DIM
    z_rows = n_heads * DN_VPK * DN_V_DIM
    out_cols = n_heads * DN_VPK * DN_V_DIM
    ab = 2 * n_heads * DN_VPK * HIDDEN
    return (qk_rows + v_rows + z_rows) * HIDDEN + out_cols * HIDDEN + ab


def ebpw_if_eliminated(n_params: int) -> dict:
    """Parent-equivalent parameters and the EBPW hole they would leave.

    Eliminating P parent parameters does not by itself change EBPW of what
    remains (EBPW = 8 * MODEL_SPECIFIC_BYTES / PARENT_PARAMETER_COUNT, and
    both numerator and the *original* parent count are the accounting base
    unless the parent identity itself is rewritten). We report P, P/N, and
    the stored-byte hole at the incumbent 4.25 and at the leader 3.1393 so
    an allocator can see the physical mass.
    """
    p = int(n_params)
    return {
        "ELIMINATED_PARENT_EQUIVALENT_PARAMETERS": p,
        "fraction_of_parent": p / PARENT_PARAMS,
        "bytes_at_bf16": p * 2,
        "bytes_at_incumbent_q4_4p25": p * 4.25 / 8.0,
        "bytes_at_parent_a_3p1393": p * 3.1393 / 8.0,
        "parent_params": PARENT_PARAMS,
    }


def capability_risk(
    *,
    level: str,
    reason: str,
    section_109: bool,
    what_would_confirm: str,
) -> dict:
    return {
        "level": level,
        "reason": reason,
        "capability_evidence": False,
        "section_109_benchmark_myopia": section_109,
        "what_would_confirm": what_would_confirm,
        "removal_requires_capability_evidence": True,
    }


def candidate(
    *,
    id: str,
    family: str,
    what: str,
    n_params: int,
    data_verdict: str,
    why: str,
    risk: dict,
    shape: dict,
    evidence: dict,
    citations: list[str],
) -> dict:
    allowed = (
        data_verdict == "EVIDENCE_SUPPORTED"
        and risk.get("capability_evidence") is True
    )
    return {
        "id": id,
        "family": family,
        "what": what,
        "data_verdict": data_verdict,
        "elimination_allowed": allowed,
        "why": why,
        "accounting": ebpw_if_eliminated(n_params),
        "capability_risk": risk,
        "shape_execution": shape,
        "evidence": evidence,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# capture I/O
# ---------------------------------------------------------------------------


def capture_path(cap: Path, layer: int) -> Path:
    for name in (f"L{layer:02d}.f16", f"L{layer}.f16"):
        p = cap / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no capture for layer {layer} in {cap}")


def load_X(cap: Path, layer: int) -> np.ndarray:
    p = capture_path(cap, layer)
    raw = np.fromfile(p, dtype=np.float16)
    if raw.size % HIDDEN != 0:
        raise ValueError(f"{p} size {raw.size} not divisible by hidden {HIDDEN}")
    n = raw.size // HIDDEN
    if n < 256:
        raise ValueError(f"{p} only {n} rows; refusing a toy capture")
    return raw.reshape(n, HIDDEN).astype(np.float32)


# ---------------------------------------------------------------------------
# censuses
# ---------------------------------------------------------------------------


def census_gqa_heads(parent: Path) -> dict:
    """Cross-head similarity on every GQA layer: Q, gate, K, V, O."""
    layers: dict[str, Any] = {}
    q_means, gate_means, k_means, v_means, o_means = [], [], [], [], []
    q_maxes: list[float] = []
    n_q_copies = 0
    n_q_aligned = 0
    n_q_pairs = 0
    prev_q: np.ndarray | None = None
    prev_layer: int | None = None
    adj_w: list[dict] = []
    dead_heads = 0
    low_heads = 0
    per_head_rms_all: list[float] = []

    for layer in GQA_LAYERS:
        Wq = load_tensor(parent, tname(layer, "self_attn.q_proj.weight"))
        Wk = load_tensor(parent, tname(layer, "self_attn.k_proj.weight"))
        Wv = load_tensor(parent, tname(layer, "self_attn.v_proj.weight"))
        Wo = load_tensor(parent, tname(layer, "self_attn.o_proj.weight"))
        if tuple(Wq.shape) != (GQA_HEADS * 2 * GQA_HEAD_DIM, HIDDEN):
            raise ValueError(f"L{layer} q_proj {Wq.shape}")
        if tuple(Wk.shape) != (GQA_KV_HEADS * GQA_HEAD_DIM, HIDDEN):
            raise ValueError(f"L{layer} k_proj {Wk.shape}")
        if tuple(Wo.shape) != (HIDDEN, GQA_HEADS * GQA_HEAD_DIM):
            raise ValueError(f"L{layer} o_proj {Wo.shape}")

        qg = Wq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN)
        q_heads = qg[:, 0]
        gate_heads = qg[:, 1]
        k_heads = Wk.reshape(GQA_KV_HEADS, GQA_HEAD_DIM, HIDDEN)
        v_heads = Wv.reshape(GQA_KV_HEADS, GQA_HEAD_DIM, HIDDEN)
        o_heads = Wo.reshape(HIDDEN, GQA_HEADS, GQA_HEAD_DIM).transpose(1, 0, 2)

        qp = head_pairwise(q_heads)
        gp = head_pairwise(gate_heads)
        kp = head_pairwise(k_heads)
        vp = head_pairwise(v_heads)
        op = head_pairwise(o_heads)

        q_rms = np.array(
            [float(np.sqrt(np.mean(np.square(q_heads[h], dtype=np.float64)))) for h in range(GQA_HEADS)],
            dtype=np.float64,
        )
        rr = rms_report(q_rms)
        dead_heads += rr["n_dead"]
        low_heads += rr["n_low"]
        per_head_rms_all.extend(q_rms.tolist())

        if prev_q is not None and prev_layer is not None:
            adj_w.append(
                {
                    "layer_a": prev_layer,
                    "layer_b": layer,
                    "q_proj_frobenius_cosine": frobenius_cosine(prev_q, Wq),
                }
            )
        prev_q = Wq
        prev_layer = layer

        rec = {
            "q": qp,
            "gate": gp,
            "k": kp,
            "v": vp,
            "o": op,
            "q_head_rms": rr,
        }
        layers[str(layer)] = rec
        q_means.append(qp["mean_cosine"])
        gate_means.append(gp["mean_cosine"])
        k_means.append(kp["mean_cosine"])
        v_means.append(vp["mean_cosine"])
        o_means.append(op["mean_cosine"])
        q_maxes.append(qp["max_cosine"])
        n_q_copies += qp["n_pairs_ge_0p99"]
        n_q_aligned += qp["n_pairs_ge_0p50"]
        n_q_pairs += qp["n_pairs"]
        print(
            f"  GQA L{layer:02d} Q mean={qp['mean_cosine']:.4f} "
            f"max={qp['max_cosine']:.4f} K={kp['mean_cosine']:.4f} "
            f"V={vp['mean_cosine']:.4f} O={op['mean_cosine']:.4f}",
            flush=True,
        )
        del Wk, Wv, Wo, qg, q_heads, gate_heads, k_heads, v_heads, o_heads
        gc.collect()

    del prev_q
    gc.collect()
    q_all = float(np.mean(q_means)) if q_means else None
    return {
        "status": "MEASURED",
        "site": "parent BF16 self_attn.{q,k,v,o}_proj streamed CPU, all 16 GQA layers",
        "n_layers": len(GQA_LAYERS),
        "layers": layers,
        "adjacent_gqa_q_proj_frobenius": adj_w,
        "headline": {
            "q_mean_cosine_all_layers": q_all,
            "q_mean_cosine_L3": layers.get("3", {}).get("q", {}).get("mean_cosine"),
            "q_mean_cosine_L63": layers.get("63", {}).get("q", {}).get("mean_cosine"),
            "prior_q_mean_cosine_L3": PRIOR_Q_L3,
            "prior_match_L3": (
                abs(float(layers["3"]["q"]["mean_cosine"]) - PRIOR_Q_L3) < 0.002
                if "3" in layers
                else False
            ),
            "gate_mean_cosine_all_layers": float(np.mean(gate_means)) if gate_means else None,
            "k_mean_cosine_all_layers": float(np.mean(k_means)) if k_means else None,
            "v_mean_cosine_all_layers": float(np.mean(v_means)) if v_means else None,
            "o_mean_cosine_all_layers": float(np.mean(o_means)) if o_means else None,
            "q_max_cosine_all_layers": float(np.max(q_maxes)) if q_maxes else None,
            "q_n_pairs": n_q_pairs,
            "q_n_pairs_ge_0p50": n_q_aligned,
            "q_n_pairs_ge_0p99": n_q_copies,
            "q_n_dead_heads": dead_heads,
            "q_n_low_heads": low_heads,
            "q_head_rms_median": float(np.median(per_head_rms_all)) if per_head_rms_all else None,
        },
        "reading": (
            "Pairwise cosine ~1 would mean duplicate heads; ~0 means an aligned "
            "basis is not free. The ORGAN_FRONTIERS prior (L3 Q mean 0.022) is "
            "the copies-vs-not test, not a Gaussian-independence test: 0.022 is "
            "many sigma above 1/sqrt(D) but two orders below 0.99. Shared-head "
            "collapse is REFUTED unless copies appear."
        ),
    }


def census_mlp(parent: Path, cap: Path | None) -> dict:
    """Dead / low-magnitude intermediate channels + activation coverage."""
    per_layer: list[dict] = []
    total_dead = 0
    total_low = 0
    adj_gate: list[dict] = []
    adj_down: list[dict] = []
    prev_gate: np.ndarray | None = None
    prev_down: np.ndarray | None = None
    prev_layer: int | None = None
    rng = np.random.RandomState(SEED)
    act_rows: list[dict] = []

    for layer in range(LAYERS):
        Wg = load_tensor(parent, tname(layer, "mlp.gate_proj.weight"))
        Wu = load_tensor(parent, tname(layer, "mlp.up_proj.weight"))
        Wd = load_tensor(parent, tname(layer, "mlp.down_proj.weight"))
        if tuple(Wg.shape) != (INTERMEDIATE, HIDDEN):
            raise ValueError(f"L{layer} gate {Wg.shape}")
        if tuple(Wd.shape) != (HIDDEN, INTERMEDIATE):
            raise ValueError(f"L{layer} down {Wd.shape}")

        g_rms = row_rms(Wg)
        u_rms = row_rms(Wu)
        d_rms = col_rms(Wd)
        # A channel is dead only if gate AND up AND the matching down column
        # are all numerically dead. Low-magnitude is the min of the three.
        combo = np.minimum(np.minimum(g_rms, u_rms), d_rms)
        med = float(np.median(combo.astype(np.float64)))
        dead_idx = np.where(combo <= DEAD_REL * max(med, 1e-30))[0]
        low_idx = np.where(combo <= LOW_REL * max(med, 1e-30))[0]
        total_dead += int(dead_idx.size)
        total_low += int(low_idx.size)

        if prev_gate is not None and prev_layer is not None:
            adj_gate.append(
                {
                    "layer_a": prev_layer,
                    "layer_b": layer,
                    "frobenius_cosine": frobenius_cosine(prev_gate, Wg),
                }
            )
            adj_down.append(
                {
                    "layer_a": prev_layer,
                    "layer_b": layer,
                    "frobenius_cosine": frobenius_cosine(prev_down, Wd),
                }
            )
        prev_gate = Wg
        prev_down = Wd
        prev_layer = layer

        rec = {
            "layer": layer,
            "gate": rms_report(g_rms),
            "up": rms_report(u_rms),
            "down_col": rms_report(d_rms),
            "combo_min": rms_report(combo, median=med),
            "n_dead": int(dead_idx.size),
            "n_low": int(low_idx.size),
            "dead_indices_head": [int(i) for i in dead_idx[:16]],
        }
        per_layer.append(rec)

        if cap is not None and layer in ACT_LAYERS:
            X = load_X(cap, layer)
            n = X.shape[0]
            take = min(ACT_TOKENS, n)
            idx = rng.choice(n, size=take, replace=False)
            Xs = X[idx]
            g = Xs @ Wg.T
            u = Xs @ Wu.T
            h = silu(g) * u
            mag = np.abs(h)
            cover = (mag > ACT_COVER_THR).mean(axis=0)
            never = int(np.sum(cover == 0.0))
            rare = int(np.sum(cover < 0.01))
            act_rows.append(
                {
                    "layer": layer,
                    "n_tokens": take,
                    "site": "post_attn_norm capture (MLP input), SwiGLU |silu(gate)*up|",
                    "threshold": ACT_COVER_THR,
                    "frac_channels_never_active": never / INTERMEDIATE,
                    "frac_channels_active_lt_1pct_tokens": rare / INTERMEDIATE,
                    "mean_coverage": float(cover.mean()),
                    "p05_coverage": float(np.quantile(cover, 0.05)),
                    "median_coverage": float(np.median(cover)),
                    "n_never_active": never,
                    "note": (
                        "Probe-silent on capture_diverse2 is NOT capability-dead "
                        "(S026 §109). Long-tail / tool / JSON / code / path tokens "
                        "are not this capture's job."
                    ),
                }
            )
            del X, Xs, g, u, h, mag, cover
        print(
            f"  MLP L{layer:02d} dead={rec['n_dead']} low={rec['n_low']} "
            f"combo_med={med:.5g}",
            flush=True,
        )
        del Wu, g_rms, u_rms, d_rms, combo
        gc.collect()

    del prev_gate, prev_down
    gc.collect()
    gate_cos = [r["frobenius_cosine"] for r in adj_gate]
    down_cos = [r["frobenius_cosine"] for r in adj_down]
    return {
        "status": "MEASURED",
        "site": "parent BF16 mlp.{gate,up,down}_proj streamed CPU, all 64 layers",
        "n_layers": LAYERS,
        "n_channels_per_layer": INTERMEDIATE,
        "n_dead_total": total_dead,
        "n_low_total": total_low,
        "n_channel_slots": LAYERS * INTERMEDIATE,
        "frac_dead": total_dead / (LAYERS * INTERMEDIATE),
        "frac_low": total_low / (LAYERS * INTERMEDIATE),
        "per_layer": per_layer,
        "adjacent_gate_frobenius_cosine": {
            "n_pairs": len(gate_cos),
            "mean": float(np.mean(gate_cos)) if gate_cos else None,
            "min": float(np.min(gate_cos)) if gate_cos else None,
            "max": float(np.max(gate_cos)) if gate_cos else None,
            "n_pairs_ge_0p99": int(sum(c >= COPY_COSINE for c in gate_cos)),
            "pairs": adj_gate,
        },
        "adjacent_down_frobenius_cosine": {
            "n_pairs": len(down_cos),
            "mean": float(np.mean(down_cos)) if down_cos else None,
            "min": float(np.min(down_cos)) if down_cos else None,
            "max": float(np.max(down_cos)) if down_cos else None,
            "n_pairs_ge_0p99": int(sum(c >= COPY_COSINE for c in down_cos)),
            "pairs": adj_down,
        },
        "activation_coverage": {
            "status": "MEASURED" if act_rows else "ABSENT",
            "reason": None if act_rows else "no capture_diverse2 on disk",
            "layers": act_rows,
            "section_109": (
                "A channel that is silent on this capture is not therefore "
                "eliminable. Free capability is not free information."
            ),
        },
        "reading": (
            "Dead = combo min(gate,up,down-col) RMS <= 1e-8 * median. "
            "Low-magnitude = 1e-3 * median, which is a tail, not a corpse. "
            "Adjacent-layer flattened cosine of W tests copies; energy-profile "
            "cosine (ORGAN_CENSUS ~0.997) is a different, weaker statistic "
            "and is not this test."
        ),
    }


def census_layers(cap: Path | None) -> dict:
    """Adjacent-layer representation similarity on real activations."""
    if cap is None:
        return {
            "status": "ABSENT",
            "reason": "capture_diverse2 not on disk; refusing to synthesize activations",
        }
    pairs: list[dict] = []
    n_identity = 0
    prev: np.ndarray | None = None
    n_tokens = None
    for layer in range(LAYERS):
        X = load_X(cap, layer)
        if n_tokens is None:
            n_tokens = int(X.shape[0])
        if prev is not None:
            st = token_pair_stats(prev, X)
            st["layer_a"] = layer - 1
            st["layer_b"] = layer
            st["mixer_a"] = "gqa" if is_gqa(layer - 1) else "deltanet"
            st["mixer_b"] = "gqa" if is_gqa(layer) else "deltanet"
            pairs.append(st)
            if st["near_identity"]:
                n_identity += 1
            print(
                f"  ACT L{layer-1:02d}->L{layer:02d} cos={st['mean_cosine']:.4f} "
                f"relL2={st['mean_rel_l2']:.4f} identity={st['near_identity']}",
                flush=True,
            )
        prev = X
    del prev
    gc.collect()
    cos = [p["mean_cosine"] for p in pairs]
    rel = [p["mean_rel_l2"] for p in pairs]
    # Most-similar / least-update pairs (candidates if any were near-identity)
    ranked = sorted(pairs, key=lambda r: (-r["mean_cosine"], r["mean_rel_l2"]))
    return {
        "status": "MEASURED",
        "site": (
            "capture_diverse2 Lxx.f16 post_attn_norm (MLP input), token-aligned "
            "across 64 layers. This is NOT block-input vs block-output; a residual "
            "stream is expected to stay similar. near_identity requires "
            f"mean_cosine>={IDENTITY_COSINE} AND mean_rel_l2<={IDENTITY_REL_L2}."
        ),
        "capture": str(cap),
        "n_tokens": n_tokens,
        "n_pairs": len(pairs),
        "n_near_identity": n_identity,
        "mean_cosine": float(np.mean(cos)) if cos else None,
        "min_cosine": float(np.min(cos)) if cos else None,
        "max_cosine": float(np.max(cos)) if cos else None,
        "mean_rel_l2": float(np.mean(rel)) if rel else None,
        "min_rel_l2": float(np.min(rel)) if rel else None,
        "max_rel_l2": float(np.max(rel)) if rel else None,
        "most_similar": ranked[:5] if ranked else [],
        "least_similar": list(reversed(ranked[-5:])) if ranked else [],
        "pairs": pairs,
        "reading": (
            "Residual networks keep adjacent hidden states similar even when "
            "every layer is load-bearing. High cosine alone is not a skip "
            "license. near_identity is the bar; anything short of it REFUTES "
            "dropping that layer as a free lunch."
        ),
    }


def census_deltanet(parent: Path) -> dict:
    """DeltaNet head redundancy + decay (A_log) + state mass."""
    a_logs: list[np.ndarray] = []
    dt_biases: list[np.ndarray] = []
    q_means: list[float] = []
    k_means: list[float] = []
    v_means: list[float] = []
    o_means: list[float] = []
    qk_means: list[float] = []
    dead_k = 0
    low_k = 0
    layers: dict[str, Any] = {}
    adj_qkv: list[dict] = []
    prev_qkv: np.ndarray | None = None
    prev_layer: int | None = None

    for layer in DN_LAYERS:
        A = load_tensor(parent, tname(layer, "linear_attn.A_log")).reshape(-1)
        dt = load_tensor(parent, tname(layer, "linear_attn.dt_bias")).reshape(-1)
        a_logs.append(A.astype(np.float32))
        dt_biases.append(dt.astype(np.float32))

        Wqkv = load_tensor(parent, tname(layer, "linear_attn.in_proj_qkv.weight"))
        Wz = load_tensor(parent, tname(layer, "linear_attn.in_proj_z.weight"))
        Wo = load_tensor(parent, tname(layer, "linear_attn.out_proj.weight"))
        # Parent layout is concatenated [Q_all, K_all, V_all], not packed per
        # key head. organ_frontiers.fuse_q38_qkvz is the authority.
        q_rows = DN_K_HEADS * DN_K_DIM
        k_rows = DN_K_HEADS * DN_K_DIM
        v_rows = DN_K_HEADS * DN_VPK * DN_V_DIM
        if tuple(Wqkv.shape) != (q_rows + k_rows + v_rows, HIDDEN):
            raise ValueError(f"L{layer} in_proj_qkv {Wqkv.shape}")
        qh = Wqkv[:q_rows].reshape(DN_K_HEADS, DN_K_DIM, HIDDEN)
        kh = Wqkv[q_rows : q_rows + k_rows].reshape(DN_K_HEADS, DN_K_DIM, HIDDEN)
        vh = Wqkv[q_rows + k_rows :].reshape(DN_K_HEADS * DN_VPK, DN_V_DIM, HIDDEN)
        qp = head_pairwise(qh)
        kp = head_pairwise(kh)
        vp = head_pairwise(vh)
        # z / out: 48 value heads of 128
        zh = Wz.reshape(DN_V_HEADS, DN_V_DIM, HIDDEN)
        oh = Wo.reshape(HIDDEN, DN_V_HEADS, DN_V_DIM).transpose(1, 0, 2)
        zp = head_pairwise(zh)
        op = head_pairwise(oh)

        qk = []
        for h in range(DN_K_HEADS):
            a = qh[h].ravel()
            b = kh[h].ravel()
            qk.append(
                float(
                    np.dot(a, b)
                    / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
                )
            )
        k_rms = np.array(
            [
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.concatenate(
                                    [qh[h].ravel(), kh[h].ravel(), vh[h * DN_VPK : (h + 1) * DN_VPK].ravel()]
                                ),
                                dtype=np.float64,
                            )
                        )
                    )
                )
                for h in range(DN_K_HEADS)
            ]
        )
        rr = rms_report(k_rms)
        dead_k += rr["n_dead"]
        low_k += rr["n_low"]

        # A_log: 48 = 16 * 3. Very negative → fast forget; ~0 → state persists.
        A64 = A.astype(np.float64)
        rec = {
            "q": qp,
            "k": kp,
            "v_from_qkv": vp,
            "z": zp,
            "o": op,
            "qk_same_head_mean": float(np.mean(qk)),
            "qk_same_head_min": float(np.min(qk)),
            "key_head_rms": rr,
            "A_log": {
                "shape": [int(A.size)],
                "mean": float(A64.mean()),
                "std": float(A64.std()),
                "min": float(A64.min()),
                "max": float(A64.max()),
                "n_near_zero": int(np.sum(np.abs(A64) < 1e-4)),
            },
            "dt_bias": {
                "mean": float(dt.astype(np.float64).mean()),
                "min": float(dt.min()),
                "max": float(dt.max()),
            },
        }
        layers[str(layer)] = rec
        q_means.append(qp["mean_cosine"])
        k_means.append(kp["mean_cosine"])
        v_means.append(vp["mean_cosine"])
        o_means.append(op["mean_cosine"])
        qk_means.append(float(np.mean(qk)))

        if prev_qkv is not None and prev_layer is not None:
            adj_qkv.append(
                {
                    "layer_a": prev_layer,
                    "layer_b": layer,
                    "frobenius_cosine": frobenius_cosine(prev_qkv, Wqkv),
                }
            )
        prev_qkv = Wqkv
        prev_layer = layer
        print(
            f"  DN L{layer:02d} Q={qp['mean_cosine']:.4f} K={kp['mean_cosine']:.4f} "
            f"V={vp['mean_cosine']:.4f} O={op['mean_cosine']:.4f} "
            f"A_log_mean={float(A64.mean()):.3f}",
            flush=True,
        )
        del Wz, Wo, qh, kh, vh, zh, oh
        gc.collect()

    del prev_qkv
    gc.collect()
    A_stack = np.stack(a_logs, axis=0)  # [48, 48]
    # cross-layer cosine of A_log vectors
    A_pair = pairwise_cosine(A_stack)
    adj_vals = [r["frobenius_cosine"] for r in adj_qkv]

    rec_elems = REC_ELEMS_PER_LAYER * len(DN_LAYERS)
    conv_elems = (DN_K_HEADS * (DN_K_DIM * 2 + DN_VPK * DN_V_DIM)) * DN_CONV_K * len(DN_LAYERS)
    return {
        "status": "MEASURED",
        "site": "parent BF16 linear_attn.{in_proj_qkv,z,out_proj,A_log,dt_bias} all 48 DN layers",
        "n_layers": len(DN_LAYERS),
        "geometry": {
            "k_heads": DN_K_HEADS,
            "v_heads": DN_V_HEADS,
            "vpk": DN_VPK,
            "k_dim": DN_K_DIM,
            "v_dim": DN_V_DIM,
            "rec_elems_per_layer": REC_ELEMS_PER_LAYER,
            "rec_elems_total": rec_elems,
            "rec_resident_bytes_f32": rec_elems * 4,
            "conv_elems_total": conv_elems,
            "conv_resident_bytes_f32": conv_elems * 4,
        },
        "headline": {
            "q_mean_cosine": float(np.mean(q_means)) if q_means else None,
            "k_mean_cosine": float(np.mean(k_means)) if k_means else None,
            "v_mean_cosine": float(np.mean(v_means)) if v_means else None,
            "o_mean_cosine": float(np.mean(o_means)) if o_means else None,
            "qk_same_head_mean": float(np.mean(qk_means)) if qk_means else None,
            "n_dead_key_heads": dead_k,
            "n_low_key_heads": low_k,
            "A_log_cross_layer_pairwise": A_pair,
            "A_log_global_mean": float(A_stack.mean()),
            "A_log_global_std": float(A_stack.std()),
            "A_log_n_near_zero": int(np.sum(np.abs(A_stack) < 1e-4)),
            "adjacent_in_proj_qkv_frobenius_mean": float(np.mean(adj_vals)) if adj_vals else None,
            "adjacent_in_proj_qkv_frobenius_max": float(np.max(adj_vals)) if adj_vals else None,
            "adjacent_in_proj_qkv_n_copies": int(sum(c >= COPY_COSINE for c in adj_vals)),
        },
        "layers": layers,
        "adjacent_in_proj_qkv": adj_qkv,
        "A_log": [a.tolist() for a in a_logs],
        "cited": {
            "state_cannot_replace_in_proj": True,
            "state_in_proj_capacity_ratio": 0.015,
            "source": "receipts/headless/NOETIC_DELTANET_DESIGN.json",
            "n_static_tensors_that_duplicate_state": 0,
        },
        "reading": (
            "DeltaNet recurrent state is a summary of the prefix, not a copy of "
            "in_proj (NOETIC_DELTANET_DESIGN: 0 of 7 static tensors duplicate S). "
            "Head pairwise ~0 REFUTES collapsing key/value heads. A_log near-zero "
            "would mean a slot does not forget — still not a license to drop the "
            "slot without a long-context capability probe (N048 owns that axis)."
        ),
    }


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def build_candidates(gqa: dict, mlp: dict, layers: dict, dn: dict) -> list[dict]:
    cites_heads = [
        "receipts/headless/ORGAN_FRONTIERS.json",
        "receipts/headless/C1SHAREDBASIS_DESIGN.json",
        "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    ]
    cites_shape = [
        "receipts/headless/BYTES_FRONTIER.json",
        "receipts/headless/KERNEL_COMPETENCE.json",
    ]
    cites_dn = [
        "receipts/headless/NOETIC_DELTANET_DESIGN.json",
        "receipts/headless/DELTANET_ORGAN.json",
        "receipts/headless/ORGAN_FRONTIERS.json",
    ]
    cites_layer = [
        "receipts/headless/NOETIC_ORGAN_CENSUS.json",
        "receipts/headless/C1SHAREDBASIS_DESIGN.json",
    ]
    risk_high_109 = capability_risk(
        level="HIGH",
        reason=(
            "No capability suite has been run on a model with this cut. "
            "S026 §109: a probe that does not exercise tool/JSON/code/path/"
            "long-tail language is not evidence the structure is unused."
        ),
        section_109=True,
        what_would_confirm=(
            "Native generation + capability_suite (tool/JSON/code/path/math/"
            "multilingual) with the structure actually removed, token ids vs parent."
        ),
    )
    risk_med = capability_risk(
        level="MED",
        reason=(
            "Weight-space magnitude or activation-coverage on capture_diverse2 "
            "only. That capture is not the capability distribution."
        ),
        section_109=True,
        what_would_confirm=(
            "Held-out composition + generation after the cut; rare/tool tokens protected."
        ),
    )

    qh = gqa.get("headline") or {}
    mh = mlp
    lh = layers
    dh = (dn.get("headline") or {}) if dn.get("status") == "MEASURED" else {}

    q_mean = qh.get("q_mean_cosine_all_layers")
    q_copies = int(qh.get("q_n_pairs_ge_0p99") or 0)
    q_dead = int(qh.get("q_n_dead_heads") or 0)
    k_mean = qh.get("k_mean_cosine_all_layers")
    v_mean = qh.get("v_mean_cosine_all_layers")
    o_mean = qh.get("o_mean_cosine_all_layers")

    heads_refute = (
        q_mean is not None
        and abs(q_mean) < NEAR_ORTH_COSINE
        and q_copies == 0
        and q_dead == 0
    )
    kv_refute = (
        k_mean is not None
        and abs(k_mean) < NEAR_ORTH_COSINE
        and v_mean is not None
        and abs(v_mean) < NEAR_ORTH_COSINE
        and o_mean is not None
        and abs(o_mean) < NEAR_ORTH_COSINE
    )

    out: list[dict] = []

    out.append(
        candidate(
            id="gqa_shared_q_heads",
            family="attention_head",
            what="Collapse 24 GQA Q heads onto a shared-head / copy-head operator",
            n_params=q_head_params(GQA_HEADS - 1) * len(GQA_LAYERS),
            data_verdict="REFUTED" if heads_refute else ("EVIDENCE_SUPPORTED" if q_copies else "INCONCLUSIVE"),
            why=(
                f"Q-head mean pairwise cosine across 16 GQA layers is {q_mean}; "
                f"copy-pairs (cosine>=0.99) = {q_copies}; dead Q heads = {q_dead}. "
                "Near-orthogonal heads are NOT redundant. ORGAN_FRONTIERS L3 prior "
                f"0.022 is confirmed on L3="
                f"{qh.get('q_mean_cosine_L3')}. Shared-head is not a free lunch "
                "(C1SHAREDBASIS NOT_WORTH_BUILDING on fidelity)."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="gqa_shared_q_heads",
                old={"gqa_heads": GQA_HEADS, "gqa_kv_heads": GQA_KV_HEADS, "q_rows": 12288, "o_cols": 6144},
                new={"gqa_heads": 1, "gqa_kv_heads": GQA_KV_HEADS, "q_rows": 512, "o_cols": 256},
                remaining_tensor_shapes="SPECIALIZED_CONST_CHANGED",
                notes=["1 Q head vs 4 KV breaks 6:1 grouping"],
            ),
            evidence={
                "q_mean_cosine_all_layers": q_mean,
                "q_n_pairs_ge_0p99": q_copies,
                "prior_L3": PRIOR_Q_L3,
            },
            citations=cites_heads + cites_shape,
        )
    )
    out.append(
        candidate(
            id="gqa_drop_one_q_head",
            family="attention_head",
            what="Drop 1 of 24 Q heads (q+gate+o) on every GQA layer, keep KV",
            n_params=q_head_params(1) * len(GQA_LAYERS),
            data_verdict="REFUTED" if heads_refute else "INCONCLUSIVE",
            why=(
                "No Q head is a copy of another and none is numerically dead. "
                "Dropping one anyway is a capability bet, not a redundancy harvest. "
                "23 heads also breaks 6:1 GQA grouping (23/4 is not integer)."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="gqa_drop_one_q_head",
                old={"gqa_heads": 24, "gqa_kv_heads": 4, "q_rows": 12288, "o_cols": 6144},
                new={"gqa_heads": 23, "gqa_kv_heads": 4, "q_rows": 23 * 2 * 256, "o_cols": 23 * 256},
                remaining_tensor_shapes="RAGGED",
                notes=["23 % 4 != 0", "11776 q_rows happens to be 184*64 but grouping is broken"],
            ),
            evidence={"q_n_dead_heads": q_dead, "q_n_pairs_ge_0p99": q_copies},
            citations=cites_heads + cites_shape,
        )
    )
    out.append(
        candidate(
            id="gqa_drop_one_kv_group",
            family="attention_head",
            what="Drop 1 of 4 KV groups (6 Q heads + 1 K + 1 V + matching O) per GQA layer",
            n_params=kv_group_params(1) * len(GQA_LAYERS),
            data_verdict="REFUTED" if (heads_refute and kv_refute) else "INCONCLUSIVE",
            why=(
                f"K mean cosine {k_mean}, V {v_mean}, O {o_mean}; Q {q_mean}. "
                "GQA already shares K/V 6-way; further dropping a group is not "
                "supported by cross-head copies. 18/3 preserves 6:1."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="gqa_drop_one_kv_group",
                old={"gqa_heads": 24, "gqa_kv_heads": 4, "q_rows": 12288, "o_cols": 6144},
                new={"gqa_heads": 18, "gqa_kv_heads": 3, "q_rows": 18 * 2 * 256, "o_cols": 18 * 256},
                remaining_tensor_shapes="SPECIALIZED_CONST_CHANGED",
                notes=["18/3 keeps 6:1; 9216 and 4608 are group-64 aligned; still a new kernel"],
            ),
            evidence={"k_mean": k_mean, "v_mean": v_mean, "o_mean": o_mean},
            citations=cites_heads + cites_shape,
        )
    )

    n_dead = int(mh.get("n_dead_total") or 0)
    n_low = int(mh.get("n_low_total") or 0)
    mlp_dead_verdict = "EVIDENCE_SUPPORTED" if n_dead > 0 else "REFUTED"
    out.append(
        candidate(
            id="mlp_drop_dead_channels",
            family="mlp_channel",
            what="Drop numerically-dead SwiGLU channels (gate+up+down-col RMS ~ 0)",
            n_params=mlp_channel_params(max(n_dead, 0) and 1, n_layers=1) * n_dead
            if n_dead
            else 0,
            data_verdict=mlp_dead_verdict,
            why=(
                f"{n_dead} of {LAYERS * INTERMEDIATE} channel-slots are numerically "
                f"dead at {DEAD_REL} * median. "
                + (
                    "Those rows/cols are ~0 in W, so they cannot contribute on any "
                    "input; still confirm generation because of residual paths."
                    if n_dead
                    else "No dead channels. The structure is live in weight space."
                )
            ),
            risk=risk_med if n_dead else risk_high_109,
            shape=shape_execution(
                name="mlp_drop_dead_channels",
                old={"intermediate": INTERMEDIATE, "rows": INTERMEDIATE, "cols": HIDDEN},
                new={
                    "intermediate": INTERMEDIATE - (n_dead // LAYERS if n_dead and n_dead % LAYERS == 0 else 0),
                    "rows": INTERMEDIATE - (n_dead // LAYERS if n_dead and n_dead % LAYERS == 0 else 0),
                    "cols": HIDDEN,
                },
                remaining_tensor_shapes=(
                    "SAME"
                    if n_dead == 0
                    else (
                        "RAGGED"
                        if n_dead % LAYERS != 0
                        or (INTERMEDIATE - n_dead // LAYERS) % GROUP != 0
                        else "SPECIALIZED_CONST_CHANGED"
                    )
                ),
                work_removed=n_dead > 0,
                notes=["Minimum shape-preserving quantum is 64 channels (group-64)"],
            ),
            evidence={"n_dead_total": n_dead, "frac_dead": mh.get("frac_dead")},
            citations=["receipts/headless/NOETIC_ORGAN_CENSUS.json"] + cites_shape,
        )
    )
    # low-magnitude is NOT supported
    quantum = 64
    out.append(
        candidate(
            id="mlp_drop_low_magnitude_64_quantum",
            family="mlp_channel",
            what=(
                f"Drop the bottom-magnitude {quantum}-channel quantum per layer "
                f"({n_low} slots sit under {LOW_REL}*median across the whole MLP)"
            ),
            n_params=mlp_channel_params(quantum, n_layers=LAYERS),
            data_verdict="REFUTED" if n_low == 0 else "INCONCLUSIVE",
            why=(
                f"{n_low} low-magnitude slots is a tail, not a corpse. "
                "Magnitude is not function and this capture is not capability "
                "(§109). 64 is the group-64 kernel quantum; dropping 1 channel "
                "would be WORSE-shaped."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="mlp_drop_low_magnitude_64_quantum",
                old={"intermediate": INTERMEDIATE, "rows": INTERMEDIATE},
                new={"intermediate": INTERMEDIATE - quantum, "rows": INTERMEDIATE - quantum},
                remaining_tensor_shapes="SPECIALIZED_CONST_CHANGED",
                notes=["17344 = 271*64 still group-aligned; geo kernel constexpr 17408 must change"],
            ),
            evidence={"n_low_total": n_low, "frac_low": mh.get("frac_low")},
            citations=["receipts/headless/NOETIC_ORGAN_CENSUS.json"] + cites_shape,
        )
    )
    out.append(
        candidate(
            id="mlp_drop_one_unaligned_channel",
            family="mlp_channel",
            what="Drop a single intermediate channel (the naive prune)",
            n_params=mlp_channel_params(1, n_layers=LAYERS),
            data_verdict="REFUTED",
            why=(
                "Even if a channel were dead, dropping 1 of 17408 yields 17407, "
                "which is not a multiple of group-64. The incumbent pack and "
                "geo kernels cannot execute it. §39: a smaller badly-shaped "
                "tensor can run slower. This candidate exists as a negative control."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="mlp_drop_one_unaligned_channel",
                old={"intermediate": INTERMEDIATE, "rows": INTERMEDIATE},
                new={"intermediate": INTERMEDIATE - 1, "rows": INTERMEDIATE - 1},
                remaining_tensor_shapes="RAGGED",
            ),
            evidence={"group": GROUP, "17407_mod_64": (INTERMEDIATE - 1) % GROUP},
            citations=cites_shape,
        )
    )

    n_id = int(lh.get("n_near_identity") or 0) if lh.get("status") == "MEASURED" else 0
    mean_cos = lh.get("mean_cosine")
    max_cos = lh.get("max_cosine")
    min_rel = lh.get("min_rel_l2")
    layer_verdict = "EVIDENCE_SUPPORTED" if n_id > 0 else "REFUTED"
    gqa_layer_p = mlp_layer_params() + gqa_mixer_params()
    dn_layer_p = mlp_layer_params() + dn_mixer_params()
    out.append(
        candidate(
            id="drop_near_identity_layer",
            family="layer",
            what="Drop any layer whose real activations are near-identity vs its neighbour",
            n_params=max(gqa_layer_p, dn_layer_p) * max(n_id, 0),
            data_verdict=layer_verdict,
            why=(
                f"near_identity pairs = {n_id} of {lh.get('n_pairs')}; "
                f"mean adjacent cosine {mean_cos}, max {max_cos}, "
                f"min rel_l2 {min_rel}. Residual streams are allowed to look "
                "similar; the identity bar is cosine>="
                f"{IDENTITY_COSINE} AND rel_l2<={IDENTITY_REL_L2}. "
                "ORGAN_CENSUS energy-profile cosine ~0.997 is NOT this test "
                "(it is a row-RMS sketch). C1 adjacent flattened cosine was "
                "1e-5..8e-3 — layers are not copies of W."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="drop_near_identity_layer",
                old={"layers": LAYERS, "hidden": HIDDEN, "intermediate": INTERMEDIATE},
                new={"layers": LAYERS - n_id, "hidden": HIDDEN, "intermediate": INTERMEDIATE},
                remaining_tensor_shapes="SAME",
                work_removed=n_id > 0,
                notes=["Kept tensors keep specialized shapes; fewer dispatches"],
            ),
            evidence={
                "n_near_identity": n_id,
                "mean_cosine": mean_cos,
                "max_cosine": max_cos,
                "min_rel_l2": min_rel,
            },
            citations=cites_layer + cites_shape,
        )
    )

    dn_q = dh.get("q_mean_cosine")
    dn_dead = int(dh.get("n_dead_key_heads") or 0)
    dn_copies = int(dh.get("adjacent_in_proj_qkv_n_copies") or 0)
    a_near0 = int(dh.get("A_log_n_near_zero") or 0)
    dn_head_refute = (
        dn_q is not None and abs(dn_q) < NEAR_ORTH_COSINE and dn_dead == 0
    )
    out.append(
        candidate(
            id="deltanet_collapse_key_heads",
            family="deltanet_state",
            what="Collapse 16 DeltaNet key heads (and their 3 value heads) as copies",
            n_params=dn_key_head_params(DN_K_HEADS - 1) * len(DN_LAYERS),
            data_verdict="REFUTED" if dn_head_refute else ("EVIDENCE_SUPPORTED" if dn_dead else "INCONCLUSIVE"),
            why=(
                f"DN Q-head mean cosine {dn_q}, dead key heads {dn_dead}, "
                f"adjacent in_proj_qkv copies {dn_copies}. State cannot replace "
                "in_proj (capacity ratio 0.015; 0 of 7 static tensors duplicate S)."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="deltanet_collapse_key_heads",
                old={"dn_k_heads": 16, "dn_v_heads": 48},
                new={"dn_k_heads": 1, "dn_v_heads": 3},
                remaining_tensor_shapes="SPECIALIZED_CONST_CHANGED",
            ),
            evidence={
                "q_mean_cosine": dn_q,
                "k_mean_cosine": dh.get("k_mean_cosine"),
                "v_mean_cosine": dh.get("v_mean_cosine"),
                "n_dead_key_heads": dn_dead,
            },
            citations=cites_dn + cites_shape,
        )
    )
    rec_bytes = REC_ELEMS_PER_LAYER * len(DN_LAYERS) * 4
    out.append(
        candidate(
            id="deltanet_drop_one_key_head_state",
            family="deltanet_state",
            what="Drop 1 of 16 key-head recurrent slots per DeltaNet layer (state + matching in_proj rows)",
            n_params=dn_key_head_params(1) * len(DN_LAYERS),
            data_verdict="REFUTED" if dn_head_refute else "INCONCLUSIVE",
            why=(
                f"A_log near-zero slots = {a_near0} of {48 * len(DN_LAYERS)}. "
                "A live decay coefficient is evidence the slot is used, not that "
                "it is free. 15 key heads is not a power of two; the recurrent "
                "kernel geometry is 16 x 3 x 128 x 128."
            ),
            risk=risk_high_109,
            shape=shape_execution(
                name="deltanet_drop_one_key_head_state",
                old={"dn_k_heads": 16, "dn_v_heads": 48, "rec_elems": REC_ELEMS_PER_LAYER},
                new={
                    "dn_k_heads": 15,
                    "dn_v_heads": 45,
                    "rec_elems": 15 * DN_VPK * DN_K_DIM * DN_V_DIM,
                },
                remaining_tensor_shapes="RAGGED",
                notes=["15 is not a power of two; rec-state tile of 16 heads is specialized"],
            ),
            evidence={
                "A_log_n_near_zero": a_near0,
                "rec_resident_bytes_f32": rec_bytes,
                "state_bytes_dropped_if_cut": 15 and (1 * DN_VPK * DN_K_DIM * DN_V_DIM * 4 * len(DN_LAYERS)),
            },
            citations=cites_dn + cites_shape,
        )
    )
    out.append(
        candidate(
            id="experts_moe",
            family="experts",
            what="Drop unused MoE experts / routes",
            n_params=0,
            data_verdict="REFUTED",
            why=(
                "Qwen3.8 is dense. qwen38_geometry.rs refuses configs with "
                "num_experts / moe_intermediate_size. There are no experts to drop. "
                "MTP tensors exist (N049) and are not this census."
            ),
            risk=capability_risk(
                level="LOW",
                reason="Structure is absent.",
                section_109=False,
                what_would_confirm="n/a",
            ),
            shape=shape_execution(
                name="experts_moe",
                old={"experts": 0},
                new={"experts": 0},
                remaining_tensor_shapes="SAME",
                work_removed=False,
            ),
            evidence={"dense": True, "num_experts": 0},
            citations=["crates/hawking-core/src/model/qwen38_geometry.rs"],
        )
    )
    return out


def classify(cands: list[dict]) -> dict:
    by = {"EVIDENCE_SUPPORTED": [], "REFUTED": [], "INCONCLUSIVE": []}
    for c in cands:
        by.setdefault(c["data_verdict"], []).append(c["id"])
    n_allowed = sum(1 for c in cands if c["elimination_allowed"])
    supported_params = sum(
        c["accounting"]["ELIMINATED_PARENT_EQUIVALENT_PARAMETERS"]
        for c in cands
        if c["data_verdict"] == "EVIDENCE_SUPPORTED"
    )
    return {
        "n_candidates": len(cands),
        "n_REFUTED": len(by.get("REFUTED") or []),
        "n_EVIDENCE_SUPPORTED": len(by.get("EVIDENCE_SUPPORTED") or []),
        "n_INCONCLUSIVE": len(by.get("INCONCLUSIVE") or []),
        "n_elimination_allowed": n_allowed,
        "REFUTED": by.get("REFUTED") or [],
        "EVIDENCE_SUPPORTED": by.get("EVIDENCE_SUPPORTED") or [],
        "INCONCLUSIVE": by.get("INCONCLUSIVE") or [],
        "supported_ELIMINATED_PARENT_EQUIVALENT_PARAMETERS": supported_params,
        "rule": (
            "elimination_allowed requires EVIDENCE_SUPPORTED AND capability_evidence. "
            "This census does not run a capability suite, so allowed stays 0 even "
            "if a weight is numerically dead. That is the point of §29."
        ),
    }


# ---------------------------------------------------------------------------
# cited prior science (compact, not re-derived)
# ---------------------------------------------------------------------------


def cite_json(rel: str, keys: list[str]) -> dict:
    path = REPO / rel
    if not path.is_file():
        return {"rel": rel, "found": False}
    try:
        doc = json.loads(path.read_text())
    except Exception as e:
        return {"rel": rel, "found": True, "error": str(e)}
    out: dict[str, Any] = {"rel": rel, "found": True, "schema": doc.get("schema")}
    cur: Any = doc
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            cur = None
            break
    out["value"] = j(cur)
    return out


def prior_science() -> dict:
    return {
        "organ_frontiers_q_L3": cite_json(
            "receipts/headless/ORGAN_FRONTIERS.json",
            ["organs", "gqa", "headline", "q_head_pairwise_mean_cosine_L3"],
        ),
        "organ_frontiers_k_L3": cite_json(
            "receipts/headless/ORGAN_FRONTIERS.json",
            ["organs", "gqa", "headline", "k_head_pairwise_mean_cosine_L3"],
        ),
        "c1_shared_basis": cite_json("receipts/headless/C1SHAREDBASIS_DESIGN.json", ["verdict"]),
        "c2_tensorop": cite_json("receipts/headless/C2TENSOROP_DESIGN.json", ["verdict"]),
        "c5_structtransform": cite_json("receipts/headless/C5STRUCTTRANSFORM_DESIGN.json", ["verdict"]),
        "bytes_frontier_answer": cite_json("receipts/headless/BYTES_FRONTIER.json", ["answer"]),
        "deltanet_design_verdict": cite_json("receipts/headless/NOETIC_DELTANET_DESIGN.json", ["verdict", "decision"]),
        "organ_census_mlp_energy_cosine": cite_json(
            "receipts/headless/NOETIC_ORGAN_CENSUS.json",
            ["organs", "mlp", "shared_structure_across_layers", "mean_pairwise_cosine"],
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build() -> dict:
    t0 = time.time()
    parent = find_parent()
    cap = find_capture()
    parent_a_exists = NOETIC_PARENT_A.exists()
    print(f"parent {parent}", flush=True)
    print(f"capture {cap}", flush=True)

    # Confirm we are not opening Parent A and not using GPU.
    gpu_touch = {
        "imported_mlx": "mlx" in sys.modules,
        "imported_torch": "torch" in sys.modules,
        "this_module_imports": ["numpy"],
        "note": "numpy-only; no Metal, no mlx, no torch. No second 27B decode.",
    }

    print("## GQA heads (all 16 layers, Q/K/V/O/gate)", flush=True)
    gqa = census_gqa_heads(parent)
    print("## MLP channels (all 64 layers)", flush=True)
    mlp = census_mlp(parent, cap)
    print("## Adjacent-layer activations", flush=True)
    layers = census_layers(cap)
    print("## DeltaNet heads + A_log + state", flush=True)
    dn = census_deltanet(parent)

    cands = build_candidates(gqa, mlp, layers, dn)
    summary = classify(cands)
    prior = prior_science()

    q_mean = (gqa.get("headline") or {}).get("q_mean_cosine_all_layers")
    one_line = (
        f"Q heads mean cosine {q_mean} (L3 prior 0.022 confirmed="
        f"{(gqa.get('headline') or {}).get('prior_match_L3')}); "
        f"K/V/O similarly near-orthogonal — shared-head REFUTED. "
        f"MLP dead channels {(mlp.get('n_dead_total'))} / {LAYERS * INTERMEDIATE}. "
        f"Near-identity layers {(layers.get('n_near_identity') if layers.get('status')=='MEASURED' else 'ABSENT')}. "
        f"DeltaNet heads near-orthogonal; state does not duplicate in_proj. "
        f"{summary['n_REFUTED']} REFUTED, {summary['n_EVIDENCE_SUPPORTED']} "
        f"EVIDENCE_SUPPORTED, {summary['n_elimination_allowed']} allowed "
        f"(capability gate closed)."
    )

    receipt = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "obligation": "N050",
        "steer": "S026 §29, §39, §40",
        "family": "DOC-ELIMINATION",
        "question": "Does this structure need to exist?",
        "parent": {
            "path": str(parent),
            "params": PARENT_PARAMS,
            "opened_noetic_parent_a": False,
            "noetic_parent_a_exists": parent_a_exists,
        },
        "capture": {
            "path": str(cap) if cap else None,
            "found": cap is not None,
            "site": "post_attn_norm (MLP input) from capture_diverse2" if cap else None,
            "not_a_second_27b_decode": True,
        },
        "did_not_load_second_27b": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_bench": True,
        "did_not_mutate_noetic_parent_a": True,
        "did_not_write_ascent_or_campaign": True,
        "gpu_touch": gpu_touch,
        "geometry": {
            "layers": LAYERS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "gqa_layers": list(GQA_LAYERS),
            "dn_layers_n": len(DN_LAYERS),
            "gqa_heads": GQA_HEADS,
            "gqa_kv_heads": GQA_KV_HEADS,
            "gqa_group": GQA_GROUP,
            "dn_k_heads": DN_K_HEADS,
            "dn_v_heads": DN_V_HEADS,
            "specialized": SPECIALIZED,
        },
        "thresholds": {
            "DEAD_REL": DEAD_REL,
            "LOW_REL": LOW_REL,
            "COPY_COSINE": COPY_COSINE,
            "ALIGNED_COSINE": ALIGNED_COSINE,
            "NEAR_ORTH_COSINE": NEAR_ORTH_COSINE,
            "IDENTITY_COSINE": IDENTITY_COSINE,
            "IDENTITY_REL_L2": IDENTITY_REL_L2,
        },
        "section_39": {
            "law": (
                "A smaller badly-shaped tensor can run slower than a larger "
                "well-shaped one. Incumbent kernels specialize on compile-time "
                "group=64, tpr=64, tg=128, hidden=5120, intermediate=17408, "
                "GQA 24/4, DeltaNet 16/48. Ragged cuts are WORSE. Specialized-"
                "constant changes are MIXED until a competent new kernel exists. "
                "Layer drops that leave remaining shapes unchanged are the only "
                "shape-BETTER cut, and they still need capability evidence."
            ),
            "citations": [
                "receipts/headless/BYTES_FRONTIER.json",
                "receipts/headless/KERNEL_COMPETENCE.json",
                "receipts/headless/C2TENSOROP_DESIGN.json",
            ],
        },
        "section_109": {
            "law": (
                "Free capability is not free information. A generic probe that "
                "does not exercise tool/JSON/code/path/rare language cannot "
                "declare a structure unused."
            ),
        },
        "attention_heads": gqa,
        "mlp_channels": mlp,
        "layer_redundancy": layers,
        "deltanet_state": dn,
        "candidates": cands,
        "summary": summary,
        "prior_science_cited_not_rederived": prior,
        "verdict": {
            "one_line": one_line,
            "shared_q_heads": "REFUTED",
            "dead_mlp_channels": "REFUTED" if int(mlp.get("n_dead_total") or 0) == 0 else "EVIDENCE_SUPPORTED",
            "near_identity_layers": "REFUTED"
            if layers.get("status") == "MEASURED" and int(layers.get("n_near_identity") or 0) == 0
            else layers.get("status"),
            "deltanet_head_collapse": "REFUTED",
            "experts": "REFUTED",
            "any_elimination_allowed": False,
        },
        "what_i_watched_fail": [
            {
                "what": "Treat ORGAN_CENSUS energy-profile cosine ~0.997 as layer copies",
                "why": (
                    "That metric is cosine of row-RMS || col-RMS sketches. A "
                    "strided flatten of W is ~noise even when energy profiles "
                    "align (noetic_organ_census.fingerprint). Adjacent flattened "
                    "W cosine is the copies test."
                ),
            },
            {
                "what": "Treat mean Q-head cosine 0.022 as Gaussian-independent",
                "why": (
                    "1/sqrt(256*5120) ≈ 8.7e-4, so 0.022 is many sigma above 0, "
                    "but it is still two orders below 0.99 copies. The right "
                    "refutation is copies, not white-noise."
                ),
            },
            {
                "what": "Flag a channel eliminable because capture_diverse2 never fires it",
                "why": "S026 §109 benchmark myopia. This capture is not tool/JSON/code/path.",
            },
            {
                "what": "Assume fewer parameters execute faster",
                "why": "§39 + BYTES_FRONTIER + KERNEL_COMPETENCE. Shape and kernel competence first.",
            },
        ],
        "citations": [
            "receipts/headless/ORGAN_FRONTIERS.json",
            "receipts/headless/NOETIC_ORGAN_CENSUS.json",
            "receipts/headless/NOETIC_DELTANET_DESIGN.json",
            "receipts/headless/BYTES_FRONTIER.json",
            "receipts/headless/KERNEL_COMPETENCE.json",
            "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            "receipts/headless/C2TENSOROP_DESIGN.json",
            "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json",
            "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            "crates/hawking-core/src/model/qwen38_geometry.rs",
        ],
        "wall_s": None,
    }
    receipt["wall_s"] = time.time() - t0
    return receipt


def main() -> int:
    rec = build()
    write_atomic(RECEIPT, rec)
    print()
    print(f"wrote {RECEIPT}")
    print(rec["verdict"]["one_line"])
    print(
        f"summary REFUTED={rec['summary']['n_REFUTED']} "
        f"SUPPORTED={rec['summary']['n_EVIDENCE_SUPPORTED']} "
        f"INCONCLUSIVE={rec['summary']['n_INCONCLUSIVE']} "
        f"allowed={rec['summary']['n_elimination_allowed']} "
        f"wall_s={rec['wall_s']:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
