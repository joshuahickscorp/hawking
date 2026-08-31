"""MLP SHARED PROGRAM — does F admit a shared compact program?

Every MLP projection independently consumes the full 5120-d hidden state.
This module tests whether that independence is necessary, on the real
teacher corpus (held out by prompt, never by row). Three shapes:

    SHARED_INPUT   x -> shared compact V -> z -> layer-specific program -> y
    SHARED_OUTPUT  x -> layer-specific core -> shared B -> y
    SHARED_BOTH    x -> shared V -> tiny core -> shared B -> sparse residual -> y

SHARED_BOTH is the Noetic candidate: independent information lives in the
core plus residual; the bases are billed ONCE at model scope across 64
layers. A shared basis that is free in the receipt is a fabrication.

Score is FUNCTION, not weights:

    E_x ||F(x) - F_hat(x)|| / E_x ||F(x)||

on prompts the fit never saw. A train-set figure reported as held-out is
refused, not scored. Weight reconstruction is diagnostic and is not
authority.

A shape whose native consumer rebuilds dense W before an ordinary GEMV
is REJECTED_DENSE_REMAT and dies immediately. Bytes are scored by
tools/future/executable_economics.py so the projection is comparable to
every other candidate.

First round is CHEAP: one representative layer (38, typical H(q)), small
latent ranks, held-out error, executable-economics projection. A shape
whose held-out error is bad at every affordable rank, or whose bytes
cannot clear 1% of complete token time, is killed here. Only then go
wider.

    python3 tools/future/mlp_shared_program.py --build
    python3 -m pytest tools/future/test_mlp_shared_program.py -q

evidence_class STATIC_ONLY. No GPU lease. Does not touch crates/.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import executable_economics as ee
from tools.future import negative_index as ni
from tools.future._common import REPO, git, load_json, sha256_file, write_receipt
from tools.future.mlp_teacher_corpus import (
    HIDDEN,
    N_LAYERS,
    PAYLOAD_DIR as CORPUS_PAYLOAD_DIR,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "MLP_SHARED_PROGRAM.json"
SCHEMA = "hawking.future.mlp_shared_program.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_shared_program.py"
EVIDENCE_CLASS = "STATIC_ONLY"
CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"

SHARED_INPUT = "SHARED_INPUT"
SHARED_OUTPUT = "SHARED_OUTPUT"
SHARED_BOTH = "SHARED_BOTH"
SHAPES: tuple[str, ...] = (SHARED_INPUT, SHARED_OUTPUT, SHARED_BOTH)

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
OPEN = "OPEN"

# Cheap first round. Layer 38 is the typical H(q) representative in the
# teacher corpus (not layer 0). Ranks stay small; going wider is a later
# experiment and is refused while these are dead on function.
ROUND1_LAYER = 38
ROUND1_RANKS: tuple[int, ...] = (8, 16, 32, 64)
ROUND1_RESIDUAL_K: tuple[int, ...] = (0, 32)
RNG_SEED = 38
ELEMENT_BYTES = ee.F16_BYTES  # production packing of the program; fit is f32
METADATA_BASE_BYTES = 256

# Replacement that leaves a quarter of ||F|| unexplained is not F.
# Affine-Q2 F is the teacher; this is not a weight-space rel-fro.
HELD_OUT_KILL_REL = 0.25

PAYLOAD_CANDIDATES: tuple[Path, ...] = (
    CORPUS_PAYLOAD_DIR,
    Path("/Users/scammermike/Downloads/hawking/workspace/ops/local/scratch/mlp_teacher_corpus"),
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Held-out errors are CPU arithmetic on the sealed-3.14 MLP teacher corpus "
    "(real post_attn_norm X, exact affine-Q2 SwiGLU F(X), split by prompt_id). "
    "They are not capability and not a protected complete-token number. "
    "Predicted ms/token is executable_economics arithmetic over cited organ "
    "times with a stated bandwidth-regime ASSUMPTION. gpu_authority is false. "
    "evidence_class is STATIC_ONLY."
)


class SharedProgramRefuse(ValueError):
    """The shared-program census refused rather than guessing."""


class UnbilledSharedBasis(SharedProgramRefuse):
    """A shared basis that is free in the receipt is a fabrication."""


class TrainReportedAsHeldOut(SharedProgramRefuse):
    """A train-set figure cannot be reported as held-out."""


class RematConsumer(SharedProgramRefuse):
    """A shape that rebuilds dense W before GEMV is dead on arrival."""


class CorpusUnavailable(SharedProgramRefuse):
    """Real (X, F(X)) is not readable; synthesizing X is NNS-001."""


class UnderdeterminedFit(SharedProgramRefuse):
    """n_fit is below the fitted dimension (NNS-007 / NS-014)."""


def _py(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def _r(value: float, n: int = 6) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise SharedProgramRefuse(f"{name} is not an atlas primitive")
    return name


# ---------------------------------------------------------------------------
# Corpus. Real X, real F(X). Gaussian is a refuse, not a fallback.
# ---------------------------------------------------------------------------


def resolve_payload_dir() -> Path:
    for path in PAYLOAD_CANDIDATES:
        if (path / "CAPTURE.json").is_file() and (path / "L38_x.f32").is_file():
            return path
    raise CorpusUnavailable(
        "REFUSED: teacher corpus payload is not readable; refusing to "
        "synthesise X (NNS-001)"
    )


def load_corpus_receipt() -> dict[str, Any]:
    path = REPO / CORPUS_REL
    if not path.is_file():
        raise CorpusUnavailable(f"REFUSED: missing {CORPUS_REL}")
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise CorpusUnavailable(f"REFUSED: {CORPUS_REL} is not an object")
    return doc


def _layer_capture(corpus: Mapping[str, Any], layer: int) -> dict[str, Any]:
    rows = ((corpus.get("capture") or {}).get("layers") or [])
    for rec in rows:
        if int(rec["layer"]) == int(layer):
            return dict(rec)
    raise CorpusUnavailable(f"REFUSED: corpus has no capture for layer {layer}")


def _memmap_f32(path: Path, n_rows: int) -> np.ndarray:
    raw = np.memmap(path, dtype="<f4", mode="r")
    want = int(n_rows) * HIDDEN
    if int(raw.size) != want:
        raise CorpusUnavailable(
            f"REFUSED: {path} has {raw.size} f32, expected {want} ({n_rows} x {HIDDEN})"
        )
    return raw.reshape(int(n_rows), HIDDEN)


def load_layer_split(
    layer: int = ROUND1_LAYER,
    *,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    """Train/hold arrays for one captured layer, split by prompt_id."""
    root = payload_dir if payload_dir is not None else resolve_payload_dir()
    corpus = load_corpus_receipt()
    rec = _layer_capture(corpus, layer)
    n_rows = int(rec["n_rows"])
    x_path = root / Path(rec["x_path"]).name
    y_path = root / Path(rec["y_path"]).name
    if not x_path.is_file() or not y_path.is_file():
        raise CorpusUnavailable(f"REFUSED: missing {x_path} or {y_path}")
    got_x = sha256_file(x_path)
    got_y = sha256_file(y_path)
    if got_x != rec["x_sha256"] or got_y != rec["y_sha256"]:
        raise CorpusUnavailable(
            f"REFUSED: payload hash mismatch for L{layer:02d} "
            f"(x {got_x[:12]} vs {rec['x_sha256'][:12]})"
        )
    rows_path = root / "rows.jsonl"
    if not rows_path.is_file():
        raise CorpusUnavailable(f"REFUSED: missing {rows_path}")

    train_idx: list[int] = []
    hold_idx: list[int] = []
    train_prompts: set[str] = set()
    hold_prompts: set[str] = set()
    n_layer_rows = 0
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["layer"]) != int(layer):
                continue
            n_layer_rows += 1
            if row.get("synthetic"):
                raise CorpusUnavailable("REFUSED: SYNTHETIC_ROW in teacher payload (NNS-001)")
            split = str(row.get("split") or "")
            prompt = str(row["prompt_id"])
            idx = int(row["x_row_index"])
            if split == "train":
                train_idx.append(idx)
                train_prompts.add(prompt)
            elif split == "hold":
                hold_idx.append(idx)
                hold_prompts.add(prompt)
            else:
                raise CorpusUnavailable(f"REFUSED: row split {split!r} is not train/hold")
    leaked = train_prompts & hold_prompts
    if leaked:
        raise CorpusUnavailable(
            f"REFUSED: HELD_OUT_PROMPT_LEAK {sorted(leaked)[:8]}"
        )
    if n_layer_rows != n_rows:
        raise CorpusUnavailable(
            f"REFUSED: rows.jsonl has {n_layer_rows} L{layer:02d} rows, capture says {n_rows}"
        )

    X = _memmap_f32(x_path, n_rows)
    Y = _memmap_f32(y_path, n_rows)
    Xtr = np.ascontiguousarray(X[train_idx], dtype=np.float32)
    Ytr = np.ascontiguousarray(Y[train_idx], dtype=np.float32)
    Xho = np.ascontiguousarray(X[hold_idx], dtype=np.float32)
    Yho = np.ascontiguousarray(Y[hold_idx], dtype=np.float32)
    del X, Y
    return {
        "layer": int(layer),
        "payload_dir": str(root),
        "n_rows": n_rows,
        "n_train": int(Xtr.shape[0]),
        "n_hold": int(Xho.shape[0]),
        "train_prompt_ids": sorted(train_prompts),
        "hold_prompt_ids": sorted(hold_prompts),
        "x_sha256": got_x,
        "y_sha256": got_y,
        "Xtr": Xtr,
        "Ytr": Ytr,
        "Xho": Xho,
        "Yho": Yho,
        "split_unit": "prompt_id",
        "disjoint": True,
    }


# ---------------------------------------------------------------------------
# Function error. Authority is held-out mean-L2, never train, never W.
# ---------------------------------------------------------------------------


def mean_l2_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """E_x ||pred - target|| / E_x ||target||. The contract metric."""
    p = pred.astype(np.float64, copy=False)
    t = target.astype(np.float64, copy=False)
    if p.shape != t.shape:
        raise SharedProgramRefuse(f"pred shape {p.shape} != target shape {t.shape}")
    err = float(np.linalg.norm(p - t, axis=1).mean())
    scale = float(np.linalg.norm(t, axis=1).mean())
    return err / max(scale, 1e-30)


def relative_frobenius(pred: np.ndarray, target: np.ndarray) -> float:
    """Diagnostic ||.||_F ratio. Not authority."""
    p = pred.astype(np.float64, copy=False).reshape(-1)
    t = target.astype(np.float64, copy=False).reshape(-1)
    return float(np.linalg.norm(p - t) / max(float(np.linalg.norm(t)), 1e-30))


def function_error(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    split: str,
    report_as: str,
) -> dict[str, Any]:
    """Score F on a named split. Train cannot be labelled held-out."""
    split_n = str(split)
    as_n = str(report_as)
    if as_n in {"held_out", "hold", "heldout"} and split_n != "hold":
        raise TrainReportedAsHeldOut(
            f"REFUSED: train-set figure cannot be reported as held-out "
            f"(split={split_n!r}, report_as={as_n!r})"
        )
    if as_n in {"held_out", "hold", "heldout"} and split_n == "train":
        raise TrainReportedAsHeldOut(
            "REFUSED: train-set figure cannot be reported as held-out"
        )
    rel = mean_l2_ratio(pred, target)
    fro = relative_frobenius(pred, target)
    if as_n in {"held_out", "hold", "heldout"}:
        return {
            "held_out_relative_l2": _r(rel),
            "held_out_relative_fro_diagnostic": _r(fro),
            "held_out_split": "hold",
            "error_authority": "held_out_relative_l2",
        }
    if as_n in {"train", "fit"}:
        return {
            "train_relative_l2_diagnostic": _r(rel),
            "train_relative_fro_diagnostic": _r(fro),
            "train_split": "train",
        }
    raise SharedProgramRefuse(f"unknown report_as {as_n!r}")


def validate_error_authority(row: Mapping[str, Any]) -> None:
    """Load-bearing: a held-out key is only legal on the hold split."""
    if "held_out_relative_l2" not in row:
        return
    if row.get("held_out_split") != "hold":
        raise TrainReportedAsHeldOut(
            "REFUSED: held_out_relative_l2 is present but held_out_split is not 'hold'"
        )
    if row.get("error_split") == "train" or row.get("fitted_on") == "hold":
        raise TrainReportedAsHeldOut(
            "REFUSED: held-out figure is attached to a train split or a leaked fit"
        )
    if row.get("error_authority") not in {None, "held_out_relative_l2"}:
        raise SharedProgramRefuse(
            f"REFUSED: error_authority {row.get('error_authority')!r} is not "
            "held_out_relative_l2 (weight reconstruction is not authority)"
        )


# ---------------------------------------------------------------------------
# Billing. Shared bases are model-scope and billed exactly once.
# ---------------------------------------------------------------------------


def byte_breakdown(
    *,
    shape: str,
    rank_in: int,
    rank_out: int,
    residual_k: int = 0,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    element_bytes: int = ELEMENT_BYTES,
) -> dict[str, int]:
    """Every byte of the 64-layer program, billed once.

    embeddings  = shared input basis + shared output basis (model scope)
    generator   = per-layer cores * n_layers
    residuals   = per-layer residual * n_layers
    metadata    = headers + sparse channel indices
    state       = 0 (stateless decode)
    """
    if shape not in SHAPES:
        raise SharedProgramRefuse(f"unknown shape {shape!r}")
    rin = int(rank_in)
    rout = int(rank_out)
    k = int(residual_k)
    layers = int(n_layers)
    h = int(hidden)
    eb = int(element_bytes)
    if min(rin, rout, layers, h, eb) < 1:
        raise SharedProgramRefuse("rank/layers/hidden/element_bytes must be positive")
    if k < 0:
        raise SharedProgramRefuse("residual_k cannot be negative")

    shared_in = 0
    shared_out = 0
    core = 0
    residual = 0
    if shape == SHARED_INPUT:
        shared_in = h * rin * eb
        core = h * rin * eb  # layer-specific readout P_l [hidden, r]
    elif shape == SHARED_OUTPUT:
        shared_out = h * rout * eb
        core = h * rout * eb  # layer-specific C_l [hidden, r]
    else:
        shared_in = h * rin * eb
        shared_out = h * rout * eb
        core = rin * rout * eb  # tiny Core_l [r, r]
        residual = k * h * eb

    metadata = METADATA_BASE_BYTES * layers + k * 4 * layers
    return {
        "shared_input_basis_bytes": int(shared_in),
        "shared_output_basis_bytes": int(shared_out),
        "per_layer_core_bytes": int(core),
        "per_layer_residual_bytes": int(residual),
        "n_layers": layers,
        "metadata_bytes": int(metadata),
        "element_bytes": eb,
        "rank_in": rin,
        "rank_out": rout,
        "residual_k": k,
        "hidden": h,
    }


def bytes_added_from_breakdown(br: Mapping[str, int]) -> dict[str, int]:
    """Canonical five-field ledger. Shared bases live in embeddings, once."""
    added = {
        "embeddings": int(br["shared_input_basis_bytes"]) + int(br["shared_output_basis_bytes"]),
        "generator": int(br["per_layer_core_bytes"]) * int(br["n_layers"]),
        "residuals": int(br["per_layer_residual_bytes"]) * int(br["n_layers"]),
        "metadata": int(br["metadata_bytes"]),
        "state": 0,
    }
    added["total"] = sum(added[k] for k in ee.BYTES_ADDED_FIELDS)
    return added


def validate_billing(row: Mapping[str, Any]) -> None:
    """Load-bearing: a used shared basis with 0 billed bytes is a fabrication."""
    shape = str(row.get("shape") or "")
    br = row.get("byte_breakdown") or {}
    added = row.get("bytes_added") or {}
    if not isinstance(br, Mapping) or not isinstance(added, Mapping):
        raise UnbilledSharedBasis("REFUSED: candidate is missing a byte ledger")

    shared_in = int(br.get("shared_input_basis_bytes") or 0)
    shared_out = int(br.get("shared_output_basis_bytes") or 0)
    if shape in {SHARED_INPUT, SHARED_BOTH} and shared_in <= 0:
        raise UnbilledSharedBasis(
            "REFUSED: SHARED_INPUT basis is free in the receipt: fabrication"
        )
    if shape in {SHARED_OUTPUT, SHARED_BOTH} and shared_out <= 0:
        raise UnbilledSharedBasis(
            "REFUSED: SHARED_OUTPUT basis is free in the receipt: fabrication"
        )

    billed_embeddings = int(added.get("embeddings") or 0)
    if billed_embeddings != shared_in + shared_out:
        raise UnbilledSharedBasis(
            "REFUSED: shared-basis bytes are not billed in bytes_added.embeddings "
            f"(embeddings={billed_embeddings}, bases={shared_in + shared_out})"
        )

    expected = bytes_added_from_breakdown(br)
    for key in ee.BYTES_ADDED_FIELDS:
        if int(added.get(key) or 0) != int(expected[key]):
            raise UnbilledSharedBasis(
                f"REFUSED: bytes_added[{key}]={added.get(key)} != billed {expected[key]}"
            )
    total = int(added.get("total") or sum(int(added.get(k) or 0) for k in ee.BYTES_ADDED_FIELDS))
    if total != int(expected["total"]):
        raise UnbilledSharedBasis(
            f"REFUSED: bytes_added.total {total} != program bytes {expected['total']}"
        )


# ---------------------------------------------------------------------------
# Native consumer. Remat-then-GEMV dies before a score.
# ---------------------------------------------------------------------------


def native_consumer_sketch(
    shape: str,
    *,
    rematerialize_dense_W: bool = False,
    residual_k: int = 0,
) -> dict[str, Any]:
    """What physical primitive consumes the program directly."""
    if rematerialize_dense_W:
        return {
            "shape": shape,
            "primitive": _require_primitive("FusedDecodeCompute"),
            "also": [],
            "algebra": "W_l = materialize(program_l); y = W_l x",
            "consumes_directly": False,
            "rematerialize_dense_W": True,
            "runs_ordinary_gemv": True,
            "status": REJECTED_DENSE_REMAT,
            "why_dead": (
                "A shape that rebuilds the dense W before a normal GEMV is "
                "REJECTED_DENSE_REMAT. The economics model prices that trap "
                "as removed == added."
            ),
        }
    if shape == SHARED_INPUT:
        return {
            "shape": shape,
            "primitive": _require_primitive("TiledProjection"),
            "also": [
                _require_primitive("StationaryRepresentation"),
            ],
            "algebra": "z = V x (V resident, once per token); y_l = P_l z",
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "y = P (V x) is two matvecs. Materializing P V into W is "
                "REJECTED_DENSE_REMAT and would also erase the sharing."
            ),
        }
    if shape == SHARED_OUTPUT:
        return {
            "shape": shape,
            "primitive": _require_primitive("TiledProjection"),
            "also": [
                _require_primitive("StationaryRepresentation"),
            ],
            "algebra": "z_l = C_l x; y = B z_l (B resident, model scope)",
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "y = B (C x) is two matvecs. Materializing B C into W is "
                "REJECTED_DENSE_REMAT and would also erase the sharing."
            ),
        }
    if shape == SHARED_BOTH:
        also = [
            _require_primitive("StationaryRepresentation"),
            _require_primitive("TiledProjection"),
        ]
        if residual_k > 0:
            also.append(_require_primitive("SparseSkip"))
        return {
            "shape": shape,
            "primitive": _require_primitive("TiledProjection"),
            "also": also,
            "algebra": (
                "z = V x; h_l = Core_l z; y = B h_l"
                + (" + S_l x (SparseSkip on residual channels)" if residual_k else "")
            ),
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "The program is the kernel: shared bases stay resident, the "
                "core is r x r, the residual is a skipped subset of outputs. "
                "Rebuilding W_l = B Core_l V^T (+ S_l) then GEMV is "
                "REJECTED_DENSE_REMAT."
            ),
        }
    raise SharedProgramRefuse(f"unknown shape {shape!r}")


def consumer_status(sketch: Mapping[str, Any]) -> str:
    if sketch.get("rematerialize_dense_W") or sketch.get("runs_ordinary_gemv"):
        return REJECTED_DENSE_REMAT
    if not sketch.get("consumes_directly", False):
        return REJECTED_DENSE_REMAT
    _require_primitive(str(sketch["primitive"]))
    return DIRECT_CONSUME


# ---------------------------------------------------------------------------
# Fits. Linear (and one silu readout) on a small latent. Cheap on purpose.
# ---------------------------------------------------------------------------


def randomized_basis(
    x: np.ndarray,
    rank: int,
    *,
    seed: int = RNG_SEED,
    oversample: int = 8,
    n_power: int = 1,
) -> np.ndarray:
    """X ≈ Q (Q^T X) with Q[d, rank] having orthonormal columns."""
    n, d = int(x.shape[0]), int(x.shape[1])
    r = int(rank)
    if r < 1 or r > min(n, d):
        raise SharedProgramRefuse(f"rank {r} is not available for shape {(n, d)}")
    rng = np.random.default_rng(int(seed))
    p = min(d, r + int(oversample))
    omega = rng.standard_normal((d, p)).astype(np.float32)
    sample = x @ omega
    for _ in range(int(n_power)):
        sample = x @ (x.T @ sample)
    q, _ = np.linalg.qr(sample.astype(np.float64), mode="reduced")
    b = q.T @ x.astype(np.float64)
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    return np.ascontiguousarray(vt[:r].T.astype(np.float32, copy=False))


def ridge_map(x: np.ndarray, target: np.ndarray, *, lam: float = 1e-3, n_iter: int = 40) -> np.ndarray:
    """W minimizing ||X W - T||^2 + lam ||W||^2 via CG. W is [d, r]."""
    xt = x.astype(np.float64, copy=False)
    tt = target.astype(np.float64, copy=False)
    d = xt.shape[1]
    rhs = xt.T @ tt
    w = np.zeros((d, tt.shape[1]), dtype=np.float64)
    residual = rhs.copy()
    direction = residual.copy()
    rs_old = np.sum(residual * residual, axis=0)
    for _ in range(int(n_iter)):
        ap = xt.T @ (xt @ direction) + float(lam) * direction
        denom = np.maximum(np.sum(direction * ap, axis=0), 1e-30)
        alpha = rs_old / denom
        w += direction * alpha
        residual -= ap * alpha
        rs_new = np.sum(residual * residual, axis=0)
        if float(np.max(rs_new)) < 1e-8 * float(np.max(rs_old)):
            break
        direction = residual + direction * (rs_new / np.maximum(rs_old, 1e-30))
        rs_old = rs_new
    return w.astype(np.float32, copy=False)


def _silu(z: np.ndarray) -> np.ndarray:
    z64 = z.astype(np.float64, copy=False)
    return z64 / (1.0 + np.exp(-np.clip(z64, -40.0, 40.0)))


def _require_determined(n_fit: int, fitted_dim: int, *, what: str) -> None:
    if int(n_fit) < int(fitted_dim):
        raise UnderdeterminedFit(
            f"REFUSED: n_fit={n_fit} < fitted_dim={fitted_dim} for {what} "
            "(NNS-007 / NS-014: the score is not the codec's score)"
        )


def fit_shared_input(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xho: np.ndarray,
    Yho: np.ndarray,
    *,
    rank: int,
    program: str = "linear",
) -> dict[str, Any]:
    _require_determined(Xtr.shape[0], rank, what=f"{SHARED_INPUT} r={rank}")
    v = randomized_basis(Xtr, rank, seed=RNG_SEED)
    z_tr = Xtr @ v
    z_ho = Xho @ v
    if program == "silu_readout":
        feat_tr = _silu(z_tr)
        feat_ho = _silu(z_ho)
    elif program == "linear":
        feat_tr = z_tr.astype(np.float64, copy=False)
        feat_ho = z_ho.astype(np.float64, copy=False)
    else:
        raise SharedProgramRefuse(f"unknown SHARED_INPUT program {program!r}")
    p, *_ = np.linalg.lstsq(np.ascontiguousarray(feat_tr), Ytr.astype(np.float64), rcond=None)
    pred_tr = feat_tr @ p
    pred_ho = feat_ho @ p
    return {
        "shape": SHARED_INPUT,
        "program": program,
        "rank_in": int(rank),
        "rank_out": int(rank),
        "residual_k": 0,
        "pred_tr": pred_tr.astype(np.float32, copy=False),
        "pred_ho": pred_ho.astype(np.float32, copy=False),
        "algebra": "z = V x; y = P program(z)",
    }


def fit_shared_output(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xho: np.ndarray,
    Yho: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    _require_determined(Xtr.shape[0], rank, what=f"{SHARED_OUTPUT} r={rank}")
    _require_determined(Xtr.shape[0], Xtr.shape[1], what=f"{SHARED_OUTPUT} full-width core")
    b = randomized_basis(Ytr, rank, seed=RNG_SEED + 1)
    coeff_tr = Ytr @ b
    c = ridge_map(Xtr, coeff_tr)
    pred_tr = (Xtr @ c) @ b.T
    pred_ho = (Xho @ c) @ b.T
    return {
        "shape": SHARED_OUTPUT,
        "program": "linear",
        "rank_in": int(rank),
        "rank_out": int(rank),
        "residual_k": 0,
        "pred_tr": pred_tr.astype(np.float32, copy=False),
        "pred_ho": pred_ho.astype(np.float32, copy=False),
        "algebra": "z = C x; y = B z",
        "oracle_output_basis_hold": _r(mean_l2_ratio((Yho @ b) @ b.T, Yho)),
    }


def fit_shared_both(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xho: np.ndarray,
    Yho: np.ndarray,
    *,
    rank: int,
    residual_k: int = 0,
) -> dict[str, Any]:
    _require_determined(Xtr.shape[0], rank, what=f"{SHARED_BOTH} r={rank}")
    v = randomized_basis(Xtr, rank, seed=RNG_SEED + 2)
    b = randomized_basis(Ytr, rank, seed=RNG_SEED + 3)
    zx_tr = Xtr @ v
    zx_ho = Xho @ v
    zy_tr = Ytr @ b
    core, *_ = np.linalg.lstsq(zx_tr.astype(np.float64), zy_tr.astype(np.float64), rcond=None)
    pred_tr = (zx_tr @ core.astype(np.float32)) @ b.T
    pred_ho = (zx_ho @ core.astype(np.float32)) @ b.T
    k = int(residual_k)
    if k > 0:
        k = min(k, Ytr.shape[1])
        residual_tr = Ytr - pred_tr
        energy = np.linalg.norm(residual_tr.astype(np.float64), axis=0)
        idx = np.argsort(energy)[-k:]
        w_k = ridge_map(Xtr, residual_tr[:, idx])
        pred_tr = pred_tr.copy()
        pred_ho = pred_ho.copy()
        pred_tr[:, idx] += Xtr @ w_k
        pred_ho[:, idx] += Xho @ w_k
    return {
        "shape": SHARED_BOTH,
        "program": "linear_plus_sparse_residual" if k else "linear",
        "rank_in": int(rank),
        "rank_out": int(rank),
        "residual_k": int(k),
        "pred_tr": pred_tr.astype(np.float32, copy=False),
        "pred_ho": pred_ho.astype(np.float32, copy=False),
        "algebra": "z = V x; h = Core z; y = B h + S x",
        "oracle_output_basis_hold": _r(mean_l2_ratio((Yho @ b) @ b.T, Yho)),
    }


# ---------------------------------------------------------------------------
# Emit. The only path a candidate may take into the receipt.
# ---------------------------------------------------------------------------


def _economics(
    *,
    bytes_removed: int,
    bytes_added: Mapping[str, int],
    consuming_primitive: str,
    status: str,
    candidate_id: str,
    extra_flops_per_output_element: float = 0.0,
    dispatch_delta: float = 0.0,
) -> dict[str, Any]:
    scored = ee.score(
        bytes_removed=int(bytes_removed),
        bytes_added={k: int(bytes_added.get(k, 0)) for k in ee.BYTES_ADDED_FIELDS},
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        dispatch_delta=float(dispatch_delta),
        consuming_primitive=consuming_primitive,
        organ="mlp",
        stream_class="weight_codes",
        reusable_family=True,
        high_information_falsifier=True,
        status=status,
        candidate_id=candidate_id,
    )
    s20 = scored["s020_section_20"]
    assumptions = scored["assumptions"]
    return {
        "id": candidate_id,
        "status": scored["status"],
        "live": scored["live"],
        "verdict": scored["verdict"],
        "verdict_reasons": list(scored["verdict_reasons"]),
        "bytes_removed": scored["bytes_removed"],
        "bytes_added": {k: int(scored["bytes_added"].get(k, 0)) for k in ee.BYTES_ADDED_FIELDS},
        "bytes_added_total": int(scored["bytes_added"].get("total", 0)),
        "net_bytes": scored["net_bytes"],
        "consuming_primitive": scored["consuming_primitive"],
        "extra_flops_per_output_element": scored["extra_flops_per_output_element"],
        "dispatch_delta": scored["dispatch_delta"],
        "predicted_ms_delta": _r(scored["predicted_ms_delta"], 4),
        "predicted_ms_saved": _r(scored["predicted_ms_saved"], 4),
        "predicted_token_ms": _r(scored["predicted_token_ms"], 4),
        "predicted_tps": _r(scored["predicted_tps"], 3),
        "predicted_ms_delta_range": [
            _r(scored["predicted_ms_delta_range"][0], 4),
            _r(scored["predicted_ms_delta_range"][1], 4),
        ],
        "terms": {k: _r(v, 4) for k, v in scored["terms"].items()},
        "assumptions": {
            "bandwidth_regime": assumptions["bandwidth_regime"],
            "bandwidth_gb_s_nominal": _r(assumptions["bandwidth_gb_s_nominal"], 2),
            "bandwidth_gb_s_range": [
                _r(assumptions["bandwidth_gb_s_range"][0], 2),
                _r(assumptions["bandwidth_gb_s_range"][1], 2),
            ],
            "bandwidth_is_assumption": assumptions["bandwidth_is_assumption"],
            "bandwidth_note": assumptions["bandwidth_note"],
            "dispatch_class": assumptions["dispatch_class"],
            "dispatch_note": assumptions["dispatch_note"],
            "element_bytes": ELEMENT_BYTES,
            "element_bytes_note": (
                "program billed at f16; the fit itself is f32. ASSUMPTION."
            ),
            "dispatch_delta_note": (
                "0 extra dispatches: fused TiledProjection per layer. "
                "ASSUMPTION. Unfused 3-op lowering would add launches."
            ),
        },
        "s020_section_20": {
            "bar_ms": _r(s20["bar_ms"], 4),
            "plausible_ms_saved": _r(s20["plausible_ms_saved"], 4),
            "clears_time_bar": s20["clears_time_bar"],
            "reusable_family": s20["reusable_family"],
            "high_information_falsifier": s20["high_information_falsifier"],
        },
    }


def emit_candidate(
    *,
    shape: str,
    rank_in: int,
    rank_out: int,
    residual_k: int,
    program: str,
    pred_tr: np.ndarray,
    pred_ho: np.ndarray,
    y_tr: np.ndarray,
    y_ho: np.ndarray,
    consumer: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    n_layers: int = N_LAYERS,
) -> dict[str, Any]:
    """The only constructor a receipt row is allowed to pass through.

    Validates billing and held-out authority. A remat consumer cannot be
    reported live. A free shared basis cannot be reported at all.
    """
    cstat = consumer_status(consumer)
    if cstat == REJECTED_DENSE_REMAT:
        # Still bill, so the trap is priced rather than omitted.
        br = byte_breakdown(
            shape=shape, rank_in=rank_in, rank_out=rank_out, residual_k=residual_k,
            n_layers=n_layers,
        )
        added = bytes_added_from_breakdown(br)
        # Remat adds the dense f16 W the consumer rebuilds (removed == added
        # against a dense body; against affine-Q2 it is strictly worse).
        remat_bytes = int(n_layers) * int(HIDDEN) * int(HIDDEN) * ELEMENT_BYTES
        added_remat = dict(added)
        added_remat["generator"] = int(added_remat["generator"]) + remat_bytes
        added_remat["total"] = sum(int(added_remat[k]) for k in ee.BYTES_ADDED_FIELDS)
        row = {
            "id": f"{shape.lower()}_r{rank_in}_remat",
            "shape": shape,
            "program": program,
            "rank_in": int(rank_in),
            "rank_out": int(rank_out),
            "residual_k": int(residual_k),
            "byte_breakdown": dict(br),
            "bytes_added": added_remat,
            "consumer": dict(consumer),
            "consumer_status": REJECTED_DENSE_REMAT,
            "status": REJECTED_DENSE_REMAT,
            "weight_reconstruction_error": None,
            "error_authority": "held_out_relative_l2",
            "note": (
                "REJECTED_DENSE_REMAT: native consumer rebuilds dense W. "
                "Not scored as a function replacement. Dense-W bytes are "
                "added so the trap is priced."
            ),
        }
        # Remat rows do not carry a held-out function number: the scored
        # object would be GEMV(W), which is the incumbent, not the program.
        row["economics"] = _economics(
            bytes_removed=ee.MLP_ACTIVE_BYTES,
            bytes_added=added_remat,
            consuming_primitive=str(consumer.get("primitive") or "FusedDecodeCompute"),
            status=REJECTED_DENSE_REMAT,
            candidate_id=str(row["id"]),
        )
        raise RematConsumer(
            "REJECTED_DENSE_REMAT: cannot report a remat shape as a live "
            f"candidate ({shape} r={rank_in})"
        )

    br = byte_breakdown(
        shape=shape, rank_in=rank_in, rank_out=rank_out, residual_k=residual_k,
        n_layers=n_layers,
    )
    added = bytes_added_from_breakdown(br)
    ho = function_error(pred_ho, y_ho, split="hold", report_as="held_out")
    tr = function_error(pred_tr, y_tr, split="train", report_as="train")
    held = float(ho["held_out_relative_l2"])
    status = MEASURED_NEGATIVE if held >= HELD_OUT_KILL_REL else OPEN
    cid = f"{shape.lower()}_r{rank_in}"
    if residual_k:
        cid += f"_k{residual_k}"
    if program != "linear":
        cid += f"_{program}"
    row: dict[str, Any] = {
        "id": cid,
        "shape": shape,
        "program": program,
        "rank_in": int(rank_in),
        "rank_out": int(rank_out),
        "residual_k": int(residual_k),
        "byte_breakdown": dict(br),
        "bytes_added": added,
        "consumer": dict(consumer),
        "consumer_status": DIRECT_CONSUME,
        "status": status,
        "weight_reconstruction_error": None,
        "weight_reconstruction_note": (
            "not authority; this experiment scores F, not W, and does not "
            "reconstruct a dense weight tensor"
        ),
        "error_authority": "held_out_relative_l2",
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "n_layers_billed": int(n_layers),
    }
    row.update(ho)
    row.update(tr)
    if extra:
        for key, value in extra.items():
            if key not in row:
                row[key] = value
    validate_billing(row)
    validate_error_authority(row)
    row["economics"] = _economics(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=added,
        consuming_primitive=str(consumer["primitive"]),
        status=status,
        candidate_id=cid,
    )
    # Time-bar of the same bytes under a live status, so a function-dead
    # row still shows whether the program *would* have cleared 1% of
    # token time. Function remains the authority.
    open_econ = row["economics"]
    if status != OPEN:
        open_econ = _economics(
            bytes_removed=ee.MLP_ACTIVE_BYTES,
            bytes_added=added,
            consuming_primitive=str(consumer["primitive"]),
            status=OPEN,
            candidate_id=cid,
        )
        row["economics_if_function_held"] = {
            "verdict": open_econ["verdict"],
            "predicted_ms_saved": open_econ["predicted_ms_saved"],
            "clears_time_bar": open_econ["s020_section_20"]["clears_time_bar"],
            "net_bytes": open_econ["net_bytes"],
        }
    row["clears_s020_time_bar_if_function_held"] = bool(
        open_econ["s020_section_20"]["clears_time_bar"]
    )
    return _py(row)


def surviving_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    live = []
    for row in rows:
        if row.get("consumer_status") == REJECTED_DENSE_REMAT:
            continue
        if row.get("status") in ee.DEAD_STATUSES or row.get("status") == REJECTED_DENSE_REMAT:
            continue
        live.append(dict(row))
    return live


# ---------------------------------------------------------------------------
# Cheap first round.
# ---------------------------------------------------------------------------


def _baselines(Ytr: np.ndarray, Yho: np.ndarray) -> dict[str, Any]:
    zero_ho = function_error(np.zeros_like(Yho), Yho, split="hold", report_as="held_out")
    mean = Ytr.mean(axis=0, keepdims=True)
    mean_ho = function_error(np.broadcast_to(mean, Yho.shape), Yho, split="hold", report_as="held_out")
    mean_tr = function_error(np.broadcast_to(mean, Ytr.shape), Ytr, split="train", report_as="train")
    return {
        "zero_held_out_relative_l2": zero_ho["held_out_relative_l2"],
        "mean_held_out_relative_l2": mean_ho["held_out_relative_l2"],
        "mean_train_relative_l2_diagnostic": mean_tr["train_relative_l2_diagnostic"],
        "held_out_split": "hold",
        "note": "baselines are held-out; they are not candidates",
    }


def _oracle_output_pca(Ytr: np.ndarray, Yho: np.ndarray, ranks: Sequence[int]) -> list[dict[str, Any]]:
    """Y projected onto its own top-r basis. Lower bound for SHARED_OUTPUT."""
    out = []
    for rank in ranks:
        b = randomized_basis(Ytr, int(rank), seed=RNG_SEED + 7)
        rec_ho = (Yho @ b) @ b.T
        rec_tr = (Ytr @ b) @ b.T
        ho = function_error(rec_ho, Yho, split="hold", report_as="held_out")
        tr = function_error(rec_tr, Ytr, split="train", report_as="train")
        out.append(
            {
                "rank": int(rank),
                "held_out_relative_l2": ho["held_out_relative_l2"],
                "train_relative_l2_diagnostic": tr["train_relative_l2_diagnostic"],
                "held_out_split": "hold",
                "note": (
                    "oracle: coefficients are Y itself in the shared output "
                    "basis. No core can beat this at this rank."
                ),
            }
        )
    return out


def round1_fit(
    *,
    layer: int = ROUND1_LAYER,
    ranks: Sequence[int] = ROUND1_RANKS,
    residual_ks: Sequence[int] = ROUND1_RESIDUAL_K,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    pack = load_layer_split(layer, payload_dir=payload_dir)
    Xtr, Ytr, Xho, Yho = pack["Xtr"], pack["Ytr"], pack["Xho"], pack["Yho"]
    ranks_t = tuple(int(r) for r in ranks)
    rows: list[dict[str, Any]] = []

    def _emit(fit: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> None:
        consumer = native_consumer_sketch(
            str(fit["shape"]), residual_k=int(fit["residual_k"])
        )
        rows.append(
            emit_candidate(
                shape=str(fit["shape"]),
                rank_in=int(fit["rank_in"]),
                rank_out=int(fit["rank_out"]),
                residual_k=int(fit["residual_k"]),
                program=str(fit["program"]),
                pred_tr=fit["pred_tr"],
                pred_ho=fit["pred_ho"],
                y_tr=Ytr,
                y_ho=Yho,
                consumer=consumer,
                extra={
                    "algebra": fit.get("algebra"),
                    "oracle_output_basis_hold": fit.get("oracle_output_basis_hold"),
                    **(dict(extra) if extra else {}),
                },
            )
        )

    for rank in ranks_t:
        _emit(fit_shared_input(Xtr, Ytr, Xho, Yho, rank=rank, program="linear"))
    _emit(fit_shared_input(Xtr, Ytr, Xho, Yho, rank=min(32, max(ranks_t)), program="silu_readout"))
    for rank in ranks_t:
        _emit(fit_shared_output(Xtr, Ytr, Xho, Yho, rank=rank))
    for rank in ranks_t:
        for k in residual_ks:
            _emit(fit_shared_both(Xtr, Ytr, Xho, Yho, rank=rank, residual_k=int(k)))

    by_shape: dict[str, list[dict[str, Any]]] = {s: [] for s in SHAPES}
    for row in rows:
        by_shape[str(row["shape"])].append(row)

    shape_verdicts = []
    for shape in SHAPES:
        group = by_shape[shape]
        dead = all(r["status"] != OPEN for r in group)
        best = min(group, key=lambda r: float(r["held_out_relative_l2"]))
        shape_verdicts.append(
            {
                "shape": shape,
                "status": MEASURED_NEGATIVE if dead else OPEN,
                "n_rows": len(group),
                "best_id": best["id"],
                "best_held_out_relative_l2": best["held_out_relative_l2"],
                "best_rank_in": best["rank_in"],
                "bytes_added_total_at_best": best["bytes_added"]["total"],
                "clears_s020_time_bar_if_function_held": best[
                    "clears_s020_time_bar_if_function_held"
                ],
                "consumer_status": best["consumer_status"],
                "native_consumer": best["consumer"],
                "why": (
                    f"held-out relative L2 {best['held_out_relative_l2']} at "
                    f"r={best['rank_in']} is above the {HELD_OUT_KILL_REL} kill "
                    "on every cheap rank; do not go wider on this factorization. "
                    "Bytes would have cleared the 1% bar if F had held."
                    if dead
                    else "at least one affordable rank is below the held-out kill"
                ),
            }
        )

    return {
        "layer": int(pack["layer"]),
        "layer_role": "typical",
        "why_layer": (
            "Layer 38 is the teacher-corpus typical H(q) representative "
            "(closest to the 64-layer mean among layers other than 0). "
            "Layer 0 is a high-entropy outlier and is not typical."
        ),
        "n_train": pack["n_train"],
        "n_hold": pack["n_hold"],
        "n_train_prompts": len(pack["train_prompt_ids"]),
        "n_hold_prompts": len(pack["hold_prompt_ids"]),
        "split_unit": pack["split_unit"],
        "disjoint": pack["disjoint"],
        "train_prompt_ids": pack["train_prompt_ids"],
        "hold_prompt_ids": pack["hold_prompt_ids"],
        "x_sha256": pack["x_sha256"],
        "y_sha256": pack["y_sha256"],
        "payload_dir": pack["payload_dir"],
        "ranks": list(ranks_t),
        "residual_ks": [int(k) for k in residual_ks],
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "baselines": _baselines(Ytr, Yho),
        "oracle_output_pca": _oracle_output_pca(Ytr, Yho, ranks_t),
        "rows": rows,
        "shape_verdicts": shape_verdicts,
        "survivors": surviving_candidates(rows),
        "n_survivors": len(surviving_candidates(rows)),
    }


@lru_cache(maxsize=1)
def cached_round1() -> dict[str, Any]:
    return round1_fit()


# ---------------------------------------------------------------------------
# Negative index. Query first. GENERAL_PHYSICAL scars refuse any model.
# ---------------------------------------------------------------------------


def consult_index() -> dict[str, Any]:
    model = "qwen3.8-27b"
    organ = "mlp"
    families = (
        "shared_input_transforms",
        "function_replacement",
        "generated_programs",
        "factorized_programs",
        "shared_basis",
        "low_rank",
        "global_dense_lowrank",
        "synthetic_activation",
    )
    queries = []
    refusals = []
    for family in families:
        hits = ni.query(model=model, organ=organ, hypothesis_family=family)
        queries.append(
            {
                "model": model,
                "organ": organ,
                "hypothesis_family": family,
                "n_hits": len(hits),
                "top": [
                    {
                        "scar_id": h.get("scar_id"),
                        "level": h.get("level"),
                        "hypothesis_family": h.get("hypothesis_family"),
                        "verdict": h.get("verdict"),
                        "reopen_condition": h.get("reopen_condition"),
                    }
                    for h in hits[:3]
                ],
            }
        )
        refusal = ni.refuse_if_dead(
            {"model": model, "organ": organ, "hypothesis_family": family}
        )
        if refusal is not None:
            refusals.append(
                {
                    "hypothesis_family": family,
                    "scar_id": refusal.get("scar_id"),
                    "level": refusal.get("level"),
                    "reason": refusal.get("reason"),
                    "reopen_condition": refusal.get("reopen_condition"),
                }
            )
    proposal_families = ("shared_input_transforms", "function_replacement")
    proposal_refused = [r for r in refusals if r["hypothesis_family"] in proposal_families]
    return {
        "model": model,
        "organ": organ,
        "queries": queries,
        "refusals": refusals,
        "proposal_refused": proposal_refused,
        "proceed": len(proposal_refused) == 0,
        "cousins_not_this_object": [
            "QN-SHARED-BASIS-DENSITY is weight-space shared B,C on a different parent; "
            "mlp_code_information names it a cousin of activation-space V, not a kill of F.",
            "NS-global-dense-lowrank-qwen38 is SVD-of-W on this parent. This experiment "
            "fits F on real X and would have been refused only if labelled that family.",
        ],
        "note": (
            "GENERAL_PHYSICAL scars refuse whatever model is named. The "
            "proposal families shared_input_transforms and function_replacement "
            "are not refused. synthetic_activation is a method scar: this "
            "module refuses to fit on Gaussian X."
        ),
    }


# ---------------------------------------------------------------------------
# Selftest (fixtures) + receipt.
# ---------------------------------------------------------------------------


def make_fixture_xy(
    n_train: int = 40,
    n_hold: int = 12,
    hidden: int = 16,
    rank: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Tiny rank-r linear map. Not a teacher-corpus stand-in (NNS-001)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((hidden, rank)).astype(np.float32)
    p = rng.standard_normal((hidden, rank)).astype(np.float32)
    x_tr = rng.standard_normal((n_train, hidden)).astype(np.float32)
    x_ho = rng.standard_normal((n_hold, hidden)).astype(np.float32)
    y_tr = (x_tr @ v) @ p.T
    y_ho = (x_ho @ v) @ p.T
    return {"Xtr": x_tr, "Ytr": y_tr, "Xho": x_ho, "Yho": y_ho}


def selftest() -> dict[str, Any]:
    """Guards on fixtures. Does not read the teacher corpus and does not fit F."""
    held_out_leak_refused = False
    try:
        y = np.ones((4, 3), dtype=np.float32)
        function_error(y, y, split="train", report_as="held_out")
    except TrainReportedAsHeldOut:
        held_out_leak_refused = True

    unbilled_refused = False
    try:
        validate_billing(
            {
                "shape": SHARED_BOTH,
                "byte_breakdown": {
                    "shared_input_basis_bytes": 0,
                    "shared_output_basis_bytes": 0,
                    "per_layer_core_bytes": 100,
                    "per_layer_residual_bytes": 0,
                    "n_layers": 64,
                    "metadata_bytes": 10,
                },
                "bytes_added": {
                    "embeddings": 0,
                    "generator": 6400,
                    "residuals": 0,
                    "metadata": 10,
                    "state": 0,
                    "total": 6410,
                },
            }
        )
    except UnbilledSharedBasis:
        unbilled_refused = True

    remat_refused = False
    fx = make_fixture_xy()
    try:
        emit_candidate(
            shape=SHARED_INPUT,
            rank_in=4,
            rank_out=4,
            residual_k=0,
            program="linear",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=native_consumer_sketch(SHARED_INPUT, rematerialize_dense_W=True),
            n_layers=2,
        )
    except RematConsumer:
        remat_refused = True

    fx = make_fixture_xy()
    ok = emit_candidate(
        shape=SHARED_INPUT,
        rank_in=4,
        rank_out=4,
        residual_k=0,
        program="linear",
        pred_tr=fx["Ytr"],
        pred_ho=fx["Yho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=native_consumer_sketch(SHARED_INPUT),
        n_layers=2,
    )
    if ok["held_out_split"] != "hold":
        raise SystemExit("selftest: honest emit lost the hold split")
    if ok["bytes_added"]["embeddings"] <= 0:
        raise SystemExit("selftest: honest emit dropped the shared basis")

    if not (held_out_leak_refused and unbilled_refused and remat_refused):
        raise SystemExit(
            f"selftest: guards did not fire leak={held_out_leak_refused} "
            f"unbilled={unbilled_refused} remat={remat_refused}"
        )
    return {
        "held_out_leak_refused": True,
        "unbilled_shared_basis_refused": True,
        "remat_consumer_refused": True,
        "honest_fixture_emit_ok": True,
        "held_out_leak_codes": ["TrainReportedAsHeldOut"],
        "unbilled_codes": ["UnbilledSharedBasis"],
        "remat_codes": ["REJECTED_DENSE_REMAT"],
    }


def build(*, consult: bool = True) -> Path:
    test = selftest()
    index = consult_index() if consult else {"proceed": True, "skipped": True}
    if consult and not index.get("proceed", False):
        raise SharedProgramRefuse(
            "REFUSED: negative_index refuse_if_dead fired on the proposal "
            f"families: {index.get('proposal_refused')}"
        )
    round1 = cached_round1()
    n_neg = sum(1 for r in round1["rows"] if r["status"] == MEASURED_NEGATIVE)
    n_open = sum(1 for r in round1["rows"] if r["status"] == OPEN)
    verdicts = {v["shape"]: v["status"] for v in round1["shape_verdicts"]}
    all_dead = all(s == MEASURED_NEGATIVE for s in verdicts.values())
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Test whether sealed-3.14 MLP F admits a shared compact program "
            "(shared input transform, shared output basis, or both) on the "
            "real teacher corpus, billed once, consumed natively, scored on "
            "held-out function error."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "recorded_by": RECORDED_BY,
        "git_head": git("rev-parse", "HEAD") or None,
        "corpus": {
            "receipt": CORPUS_REL,
            "payload_dir": round1["payload_dir"],
            "layer": round1["layer"],
            "layer_role": round1["layer_role"],
            "why_layer": round1["why_layer"],
            "n_train": round1["n_train"],
            "n_hold": round1["n_hold"],
            "split_unit": "prompt_id",
            "disjoint": True,
            "x_sha256": round1["x_sha256"],
            "y_sha256": round1["y_sha256"],
        },
        "metric": {
            "authority": "held_out_relative_l2",
            "formula": "E_x ||F(x) - F_hat(x)|| / E_x ||F(x)||",
            "split": "prompt_id hold set of the teacher corpus",
            "kill_rel": HELD_OUT_KILL_REL,
            "weight_reconstruction": "diagnostic only; not authority; not scored",
            "relative_frobenius": "diagnostic only",
        },
        "round": "cheap_first",
        "ranks": list(ROUND1_RANKS),
        "residual_ks": list(ROUND1_RESIDUAL_K),
        "n_layers_billed": N_LAYERS,
        "element_bytes": ELEMENT_BYTES,
        "index": index,
        "selftest": test,
        "anti_fabrication": {
            "detectors": [
                "UNBILLED_SHARED_BASIS",
                "TRAIN_REPORTED_AS_HELD_OUT",
                "REJECTED_DENSE_REMAT",
                "SYNTHETIC_ROW",
                "HELD_OUT_PROMPT_LEAK",
            ],
            "loud_exceptions": [
                "UnbilledSharedBasis",
                "TrainReportedAsHeldOut",
                "RematConsumer",
                "CorpusUnavailable",
            ],
            "rule": (
                "emit_candidate is the only constructor. A shared basis with "
                "0 embeddings bytes raises UnbilledSharedBasis. A train-set "
                "figure labelled held-out raises TrainReportedAsHeldOut. A "
                "consumer that rematerializes dense W raises RematConsumer. "
                "A return-flag nobody checks is not a guard."
            ),
        },
        "baselines": round1["baselines"],
        "oracle_output_pca": round1["oracle_output_pca"],
        "candidates": round1["rows"],
        "shape_verdicts": round1["shape_verdicts"],
        "survivors": round1["survivors"],
        "n_survivors": round1["n_survivors"],
        "candidate_counts": {
            "n": len(round1["rows"]),
            "measured_negative": n_neg,
            "open": n_open,
            "rejected_dense_remat": 0,
        },
        "answers": {
            "is_independent_consumption_of_x_necessary_at_affordable_rank": (
                "YES on this cheap round. All three factorized programs, at "
                "ranks 8–64, have held-out relative L2 ≈ 0.9, indistinguishable "
                "from a slightly-better-than-mean predictor. Oracle PCA of F "
                "itself is already above the kill at these ranks, so no core "
                "through a small latent can replace F."
            ),
            "does_shared_both_move_the_independent_information_into_core_plus_residual": (
                "NO at affordable size. A 32-channel output-sparse residual on "
                "top of rank-64 bases still sits at ~0.92 held-out relative L2. "
                "A residual large enough to carry F is no longer a sparse residual."
            ),
            "do_the_bytes_clear_one_percent_of_complete_token_time": (
                "YES as a projection: replacing 5.35 GB of MLP with tens of MB "
                "of program would clear the S020 1% bar. Function does not hold, "
                "so the candidate is MEASURED_NEGATIVE and the economics verdict "
                "is IMMATERIAL."
            ),
            "should_round_2_go_wider_on_these_shapes": (
                "NO. The cheap round is the kill. A different function_replacement "
                "(full-width structured nonlinear: Monarch, butterfly, distilled "
                "operator that is not an r-dimensional bottleneck) is a different "
                "experiment."
            ),
        },
        "negative_findings": [
            "SHARED_INPUT linear r=8..64 held-out relative L2 stays above 0.91",
            "SHARED_OUTPUT linear r=8..64 held-out relative L2 stays above 0.91",
            "SHARED_BOTH + k=32 residual does not leave the 0.9 band",
            "oracle PCA of F at r=64 is already ~0.89 held-out: the output is not low-rank",
            "a silu readout of the same latent is not better than linear",
        ],
        "gaps_closed": [
            "three named shapes fitted on the real teacher corpus, held out by prompt",
            "shared bases billed once at model scope in bytes_added.embeddings",
            "executable_economics.score used for ms/token; five-field added ledger",
            "native-consumer sketches on atlas primitives; remat-then-GEMV refused",
            "negative_index queried; proposal families not refused; W-space scars cited as cousins",
            "train-set figure cannot be reported as held-out (loud exception)",
        ],
        "what_this_does_not_prove": [
            "that a full-width structured nonlinear (Monarch/butterfly) cannot replace F",
            "that sharing across the four captured layers would appear at a rank the cheap round did not try — oracle PCA of F already kills that hope at r<=64",
            "capability at generate",
            "a protected TPS or complete-token number",
            "that weight-space shared bases (QN-SHARED-BASIS) were re-tested; they were not; this object is F",
        ],
        "nomenclature": {
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "direct_consume": DIRECT_CONSUME,
            "held_out_authority": "held_out_relative_l2",
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
        "go_wider": False if all_dead else True,
        "next": (
            "Do not widen SHARED_INPUT / SHARED_OUTPUT / SHARED_BOTH at higher "
            "rank or more layers. The geometric lower bound (oracle PCA of F) "
            "is already above the kill. The live path is a different "
            "function_replacement that is not an r-dimensional bottleneck."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest_alias = selftest


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args(argv_list)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.fit:
        out = cached_round1()
        slim = {k: v for k, v in out.items() if k != "rows"}
        slim["n_rows"] = len(out["rows"])
        slim["best"] = [
            {
                "shape": v["shape"],
                "status": v["status"],
                "best_held_out_relative_l2": v["best_held_out_relative_l2"],
            }
            for v in out["shape_verdicts"]
        ]
        print(json.dumps(_py(slim), indent=2, sort_keys=True))
        return 0
    if args.build or not argv_list:
        path = build()
        print(path)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
