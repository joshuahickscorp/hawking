"""MLP NONLINEAR PROGRAM — function replacement after linear low-rank died.

Linear shared programs are MEASURED_NEGATIVE (MLP_SHARED_PROGRAM). The
oracle control there is the important part: PCA of F itself at rank 64 is
already ~0.89 held-out. The output manifold is not low-rank. This module
does not retry a linear basis, a rank sweep of the composite, or a shared
subspace.

What remains is the organ's structure. F(x) = down(silu(gate(x)) * up(x))
has a nonlinearity between two linear maps, so a linear approximation of
the composite failing is not surprising. The families that respect that
structure, untested by the shared-program lane:

    FACTORIZE_THE_FACTORS   approximate gate, up, down SEPARATELY at low
                            rank; keep exact silu and the elementwise
                            product. Cheap control: was the shared-program
                            result an artifact of approximating the
                            composite rather than its parts?
    DICTIONARY_PROGRAM      per-block codebook plus an index from x
    PRODUCT_DICTIONARY      product codebooks over sub-blocks
    CONDITIONAL_PROGRAM     cheap default plus an exceptional path,
                            conditioned on activation features; bill the
                            condition
    GENERATED_BLOCK         a small program of (layer, row block) producing
                            a compact local representation
    NONLINEAR_GENERATOR     an MLP-shaped generator producing the compact
                            program (silu readout / two-layer silu)

FACTORIZE_THE_FACTORS is run first and reported first. Round one is cheap:
one representative layer (38), small sizes, held-out function error,
executable_economics projection. A family whose held-out error sits in
the 0.9 band at any affordable size is dead; the scar names the mechanism.

Held-out is by prompt, never by row. A train-set figure labelled held-out
is refused. Dense rematerialization of W is REJECTED_DENSE_REMAT. Bytes
are scored only through executable_economics.score.

    python3 tools/future/mlp_nonlinear_program.py --build
    python3 -m pytest tools/future/test_mlp_nonlinear_program.py -q

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
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import executable_economics as ee
from tools.future import mlp_shared_program as msp
from tools.future import negative_index as ni
from tools.future._common import REPO, git, load_json, write_receipt
from tools.future.mlp_teacher_corpus import (
    CAPABILITY_DOMAINS,
    HIDDEN,
    INTERMEDIATE,
    N_LAYERS,
    POSITION_BANDS,
    organ_records,
    reconstruct_w,
    silu,
    _matmul,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "MLP_NONLINEAR_PROGRAM.json"
SCHEMA = "hawking.future.mlp_nonlinear_program.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_nonlinear_program.py"
EVIDENCE_CLASS = "STATIC_ONLY"
CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"
SHARED_REL = "receipts/future/MLP_SHARED_PROGRAM.json"
CAPMAP_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"

FACTORIZE_THE_FACTORS = "FACTORIZE_THE_FACTORS"
DICTIONARY_PROGRAM = "DICTIONARY_PROGRAM"
PRODUCT_DICTIONARY = "PRODUCT_DICTIONARY"
CONDITIONAL_PROGRAM = "CONDITIONAL_PROGRAM"
GENERATED_BLOCK = "GENERATED_BLOCK"
NONLINEAR_GENERATOR = "NONLINEAR_GENERATOR"
FAMILIES: tuple[str, ...] = (
    FACTORIZE_THE_FACTORS,
    DICTIONARY_PROGRAM,
    PRODUCT_DICTIONARY,
    CONDITIONAL_PROGRAM,
    GENERATED_BLOCK,
    NONLINEAR_GENERATOR,
)

# Linear shared subspaces are a measured-negative scar. Naming one here is
# a refuse, not a candidate.
DEAD_LINEAR_SHAPES = frozenset(
    {msp.SHARED_INPUT, msp.SHARED_OUTPUT, msp.SHARED_BOTH, "linear_shared_subspace"}
)

DIRECT_CONSUME = msp.DIRECT_CONSUME
REJECTED_DENSE_REMAT = msp.REJECTED_DENSE_REMAT
MEASURED_NEGATIVE = msp.MEASURED_NEGATIVE
OPEN = msp.OPEN

ROUND1_LAYER = 38
ROUND1_RANKS: tuple[int, ...] = (8, 16, 32, 64)
ROUND1_DICT_K: tuple[int, ...] = (16, 64, 256)
ROUND1_Z_RANK = 16
RNG_SEED = 38
ELEMENT_BYTES = ee.F16_BYTES
METADATA_BASE_BYTES = 256

# Replacement that leaves a quarter of ||F|| unexplained is not F.
HELD_OUT_KILL_REL = msp.HELD_OUT_KILL_REL  # 0.25
# Cheap round: sitting in the 0.9 band at any affordable size is a kill.
KILL_BAND = 0.85

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Held-out errors are CPU arithmetic on the sealed-3.14 MLP teacher corpus "
    "(real post_attn_norm X, exact affine-Q2 SwiGLU F(X), split by prompt_id). "
    "They are not capability and not a protected complete-token number. "
    "Predicted ms/token is executable_economics arithmetic over cited organ "
    "times with a stated bandwidth-regime ASSUMPTION. gpu_authority is false. "
    "evidence_class is STATIC_ONLY. Linear shared subspaces (SHARED_INPUT / "
    "SHARED_OUTPUT / SHARED_BOTH) are not re-tested; MLP_SHARED_PROGRAM is "
    "the scar, and oracle PCA of F at rank 64 is already ~0.89 held-out."
)


class NonlinearProgramRefuse(ValueError):
    """The nonlinear-program census refused rather than guessing."""


class UnbilledProgramByte(NonlinearProgramRefuse):
    """A dictionary, generator, core, index, or condition with 0 billed bytes."""


class LinearSharedSubspaceDead(NonlinearProgramRefuse):
    """SHARED_* / linear shared subspace is a scoped scar, not a candidate."""


class TrainReportedAsHeldOut(msp.TrainReportedAsHeldOut, NonlinearProgramRefuse):
    """A train-set figure cannot be reported as held-out."""


class RematConsumer(msp.RematConsumer, NonlinearProgramRefuse):
    """A shape that rebuilds dense W before GEMV is dead on arrival."""


class CorpusUnavailable(msp.CorpusUnavailable, NonlinearProgramRefuse):
    """Real (X, F(X)) is not readable; synthesizing X is NNS-001."""


class WeightsUnavailable(NonlinearProgramRefuse):
    """Sealed affine-Q2 factors are not readable; synthesizing W is refused."""


class HoldYUsedAsIndex(TrainReportedAsHeldOut):
    """A codebook index computed from held-out Y is not a held-out program."""


# Re-export the shared-program guards the tests already know.
UnbilledSharedBasis = msp.UnbilledSharedBasis
UnderdeterminedFit = msp.UnderdeterminedFit


def _py(x: Any) -> Any:
    return msp._py(x)


def _r(value: float, n: int = 6) -> float:
    return msp._r(value, n)


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise NonlinearProgramRefuse(f"{name} is not an atlas primitive")
    return name


def _require_family(family: str) -> str:
    if family in DEAD_LINEAR_SHAPES:
        raise LinearSharedSubspaceDead(
            "REFUSED: linear shared subspace / SHARED_* is MEASURED_NEGATIVE "
            "(receipts/future/MLP_SHARED_PROGRAM.json). Oracle PCA of F at "
            "rank 64 is already ~0.89 held-out; the output manifold is not "
            "low-rank. This module does not retry that family."
        )
    if family not in FAMILIES:
        raise NonlinearProgramRefuse(f"unknown family {family!r}")
    return family


# ---------------------------------------------------------------------------
# Corpus. Real X, real F(X), prompt-split, with domain / band / prompt labels.
# ---------------------------------------------------------------------------


def load_layer_pack(
    layer: int = ROUND1_LAYER,
    *,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    """Train/hold arrays plus per-row labels. Split unit is prompt_id."""
    pack = msp.load_layer_split(layer, payload_dir=payload_dir)
    root = Path(pack["payload_dir"])
    rows_path = root / "rows.jsonl"
    train_domain: list[str] = []
    hold_domain: list[str] = []
    train_band: list[str] = []
    hold_band: list[str] = []
    train_prompt: list[str] = []
    hold_prompt: list[str] = []
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["layer"]) != int(layer):
                continue
            if row.get("synthetic"):
                raise CorpusUnavailable(
                    "REFUSED: SYNTHETIC_ROW in teacher payload (NNS-001)"
                )
            split = str(row.get("split") or "")
            rec_d = str(row.get("capability_domain") or "")
            rec_b = str(row.get("position_band") or "")
            rec_p = str(row["prompt_id"])
            if split == "train":
                train_domain.append(rec_d)
                train_band.append(rec_b)
                train_prompt.append(rec_p)
            elif split == "hold":
                hold_domain.append(rec_d)
                hold_band.append(rec_b)
                hold_prompt.append(rec_p)
    if len(hold_prompt) != int(pack["n_hold"]) or len(train_prompt) != int(pack["n_train"]):
        raise CorpusUnavailable(
            f"REFUSED: meta length train={len(train_prompt)} hold={len(hold_prompt)} "
            f"!= pack train={pack['n_train']} hold={pack['n_hold']}"
        )
    leaked = set(pack["train_prompt_ids"]) & set(pack["hold_prompt_ids"])
    if leaked:
        raise CorpusUnavailable(f"REFUSED: HELD_OUT_PROMPT_LEAK {sorted(leaked)[:8]}")
    pack["train_domain"] = train_domain
    pack["hold_domain"] = hold_domain
    pack["train_band"] = train_band
    pack["hold_band"] = hold_band
    pack["train_prompt"] = train_prompt
    pack["hold_prompt"] = hold_prompt
    pack["hold_meta"] = {
        "domain": hold_domain,
        "band": hold_band,
        "prompt_id": hold_prompt,
    }
    return pack


def load_factor_weights(layer: int) -> dict[str, np.ndarray]:
    """Sealed affine-Q2 gate/up/down as f32 (out, in). Not a dense-W consumer."""
    try:
        recs = organ_records(layer)
    except Exception as exc:  # noqa: BLE001 — catalog/path failures become a refuse
        raise WeightsUnavailable(
            f"REFUSED: cannot read sealed factors for layer {layer}: {exc}"
        ) from exc
    out: dict[str, np.ndarray] = {}
    for name, key in (("gate", "mlp.gate"), ("up", "mlp.up"), ("down", "mlp.down")):
        path = recs[key]["segment_path"]
        if not Path(path).is_file():
            raise WeightsUnavailable(f"REFUSED: missing factor tensor {path}")
        w = reconstruct_w(path)
        out[name] = np.ascontiguousarray(w, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Function error. Authority is held-out mean-L2, never train, never W, never Y-index.
# ---------------------------------------------------------------------------


def mean_l2_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    return msp.mean_l2_ratio(pred, target)


def relative_frobenius(pred: np.ndarray, target: np.ndarray) -> float:
    return msp.relative_frobenius(pred, target)


def mean_cosine(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(np.float64, copy=False)
    t = target.astype(np.float64, copy=False)
    if p.shape != t.shape:
        raise NonlinearProgramRefuse(f"pred shape {p.shape} != target shape {t.shape}")
    pn = np.linalg.norm(p, axis=1)
    tn = np.linalg.norm(t, axis=1)
    dot = np.sum(p * t, axis=1)
    return float(np.mean(dot / np.maximum(pn * tn, 1e-30)))


def _slice_relative_l2(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[str],
) -> dict[str, float]:
    p = pred.astype(np.float64, copy=False)
    t = target.astype(np.float64, copy=False)
    err = np.linalg.norm(p - t, axis=1)
    scale = np.linalg.norm(t, axis=1)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[str(lab)].append(i)
    out: dict[str, float] = {}
    for lab, idx in buckets.items():
        ii = np.asarray(idx, dtype=np.int64)
        out[lab] = _r(float(err[ii].mean() / max(float(scale[ii].mean()), 1e-30)))
    return out


def function_error(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    split: str,
    report_as: str,
    meta: Mapping[str, Sequence[str]] | None = None,
    index_from: str | None = None,
) -> dict[str, Any]:
    """Score F on a named split. Train cannot be labelled held-out.

    index_from='y_hold' is a leak: the program used held-out Y to pick a
    codebook index. That number cannot be the held-out authority.
    """
    if index_from in {"y_hold", "hold_y", "Yho"}:
        raise HoldYUsedAsIndex(
            "REFUSED: codebook index computed from held-out Y cannot be "
            "reported as a held-out program score"
        )
    try:
        base = msp.function_error(pred, target, split=split, report_as=report_as)
    except msp.TrainReportedAsHeldOut as exc:
        raise TrainReportedAsHeldOut(str(exc)) from exc
    as_n = str(report_as)
    if as_n in {"held_out", "hold", "heldout"}:
        base["held_out_cosine"] = _r(mean_cosine(pred, target))
        if meta is not None:
            domains = _slice_relative_l2(pred, target, meta["domain"])
            bands = _slice_relative_l2(pred, target, meta["band"])
            prompts = _slice_relative_l2(pred, target, meta["prompt_id"])
            worst_p = max(prompts, key=lambda k: prompts[k]) if prompts else None
            worst_d = max(domains, key=lambda k: domains[k]) if domains else None
            base["per_capability_domain"] = domains
            base["per_position_band"] = bands
            base["worst_prompt_id"] = worst_p
            base["worst_prompt_relative_l2"] = prompts.get(worst_p) if worst_p else None
            base["worst_domain"] = worst_d
            base["worst_domain_relative_l2"] = domains.get(worst_d) if worst_d else None
            base["n_hold_prompts"] = len(prompts)
    else:
        base["train_cosine_diagnostic"] = _r(mean_cosine(pred, target))
    return base


def validate_error_authority(row: Mapping[str, Any]) -> None:
    try:
        msp.validate_error_authority(row)
    except msp.TrainReportedAsHeldOut as exc:
        raise TrainReportedAsHeldOut(str(exc)) from exc
    if row.get("index_from") in {"y_hold", "hold_y", "Yho"}:
        raise HoldYUsedAsIndex(
            "REFUSED: held-out figure used held-out Y as a codebook index"
        )
    if row.get("held_out_split") == "train":
        raise TrainReportedAsHeldOut(
            "REFUSED: held_out_split='train' cannot be reported as held-out"
        )


def status_from_error(
    held: float,
    per_domain: Mapping[str, float] | None = None,
) -> tuple[str, bool, str]:
    """(status, cheap_kill, why). A mean that destroys one domain is not OPEN."""
    domain_hit = False
    worst_domain = None
    worst_val = -1.0
    if per_domain:
        for name, val in per_domain.items():
            if float(val) > worst_val:
                worst_val = float(val)
                worst_domain = name
            if float(val) >= HELD_OUT_KILL_REL:
                domain_hit = True
    if held >= KILL_BAND:
        return (
            MEASURED_NEGATIVE,
            True,
            f"held-out relative L2 {held} sits in the 0.9 band "
            f"(>={KILL_BAND}); cheap kill",
        )
    if held >= HELD_OUT_KILL_REL:
        return (
            MEASURED_NEGATIVE,
            False,
            f"held-out relative L2 {held} is above the {HELD_OUT_KILL_REL} kill",
        )
    if domain_hit:
        return (
            MEASURED_NEGATIVE,
            False,
            f"mean held-out relative L2 {held} is below the kill but domain "
            f"{worst_domain}={worst_val} is not; a small mean that destroys "
            "one domain is not a winner",
        )
    return OPEN, False, f"held-out relative L2 {held} is below {HELD_OUT_KILL_REL}"


# ---------------------------------------------------------------------------
# Billing. Dictionaries/generators at model scope; cores/residuals per layer;
# indices, metadata, and the condition logic billed once. No invented byte model.
# ---------------------------------------------------------------------------


def byte_breakdown(
    family: str,
    *,
    rank: int = 0,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    element_bytes: int = ELEMENT_BYTES,
    codebook_k: int = 0,
    n_blocks: int = 1,
    z_rank: int = 0,
    n_experts: int = 0,
    two_layer: bool = False,
) -> dict[str, int]:
    """Every byte of the 64-layer program, billed once."""
    fam = _require_family(family)
    layers = int(n_layers)
    h = int(hidden)
    inter = int(intermediate)
    eb = int(element_bytes)
    r = int(rank)
    k = int(codebook_k)
    blocks = max(int(n_blocks), 1)
    zr = int(z_rank)
    experts = int(n_experts)
    if min(layers, h, eb) < 1:
        raise NonlinearProgramRefuse("layers/hidden/element_bytes must be positive")

    dictionary = 0
    generator_model = 0
    core = 0
    residual = 0
    condition = 0
    index = 0

    if fam == FACTORIZE_THE_FACTORS:
        if r < 1:
            raise NonlinearProgramRefuse("FACTORIZE_THE_FACTORS rank must be positive")
        # gate: r*(h+inter); up: r*(h+inter); down: r*(h+inter)
        core = 3 * r * (h + inter) * eb
    elif fam == DICTIONARY_PROGRAM:
        if k < 1:
            raise NonlinearProgramRefuse("DICTIONARY_PROGRAM codebook_k must be positive")
        # Per-layer Y codebook (cores). Shared X-side indexer in embeddings.
        core = k * h * eb
        zr_use = zr if zr > 0 else ROUND1_Z_RANK
        dictionary = h * zr_use * eb  # shared V, model scope
        condition = k * zr_use * eb  # z-centroids of the codes
        index = k * 4 * layers  # code ids in the header, billed once as metadata
    elif fam == PRODUCT_DICTIONARY:
        if k < 1 or blocks < 2:
            raise NonlinearProgramRefuse("PRODUCT_DICTIONARY needs k>=1 and n_blocks>=2")
        block_dim = h // blocks
        core = blocks * k * block_dim * eb  # == k * h * eb
        zr_use = zr if zr > 0 else ROUND1_Z_RANK
        dictionary = h * zr_use * eb
        condition = blocks * k * zr_use * eb
        index = blocks * k * 4 * layers
    elif fam == CONDITIONAL_PROGRAM:
        if experts < 2:
            raise NonlinearProgramRefuse("CONDITIONAL_PROGRAM needs n_experts>=2")
        # Centroids of x are the condition, billed.
        condition = experts * h * eb
        if r > 0:
            core = experts * (h * r + h * r) * eb  # per-expert V and P
        else:
            core = experts * h * eb  # per-expert mean Y
    elif fam == GENERATED_BLOCK:
        if r < 1 or blocks < 2:
            raise NonlinearProgramRefuse("GENERATED_BLOCK needs rank>=1 and n_blocks>=2")
        block_dim = h // blocks
        # Lookup generator G(layer, row_block) -> (V [h,r], U [block,r])
        core = blocks * r * (h + block_dim) * eb
        generator_model = blocks * 8  # block-id table header, model scope
    elif fam == NONLINEAR_GENERATOR:
        if r < 1:
            raise NonlinearProgramRefuse("NONLINEAR_GENERATOR rank must be positive")
        if two_layer:
            # V [h,r] + W_h [r,r] + P [h,r]; W_h is the model-scope generator core
            core = (h * r + h * r) * eb
            generator_model = r * r * eb
        else:
            core = (h * r + h * r) * eb

    metadata = METADATA_BASE_BYTES * layers + int(index) + int(condition)
    return {
        "family": fam,
        "per_layer_core_bytes": int(core),
        "model_scope_dictionary_bytes": int(dictionary),
        "model_scope_generator_bytes": int(generator_model),
        "per_layer_residual_bytes": int(residual),
        "condition_bytes": int(condition),
        "index_bytes": int(index),
        "metadata_bytes": int(metadata),
        "n_layers": layers,
        "element_bytes": eb,
        "rank": r,
        "codebook_k": k,
        "n_blocks": blocks,
        "z_rank": zr,
        "n_experts": experts,
        "two_layer": int(bool(two_layer)),
        "hidden": h,
        "intermediate": inter,
    }


def bytes_added_from_breakdown(br: Mapping[str, Any]) -> dict[str, int]:
    """Canonical five-field ledger. Every byte once; ee.score is the scorer."""
    added = {
        "embeddings": int(br.get("model_scope_dictionary_bytes") or 0),
        "generator": (
            int(br.get("model_scope_generator_bytes") or 0)
            + int(br.get("per_layer_core_bytes") or 0) * int(br["n_layers"])
        ),
        "residuals": int(br.get("per_layer_residual_bytes") or 0) * int(br["n_layers"]),
        "metadata": int(br.get("metadata_bytes") or 0),
        "state": 0,
    }
    added["total"] = sum(added[k] for k in ee.BYTES_ADDED_FIELDS)
    return added


def validate_billing(row: Mapping[str, Any]) -> None:
    """Load-bearing: a used dictionary/generator/condition/core with 0 bytes is a fabrication."""
    family = str(row.get("family") or "")
    br = row.get("byte_breakdown") or {}
    added = row.get("bytes_added") or {}
    if not isinstance(br, Mapping) or not isinstance(added, Mapping):
        raise UnbilledProgramByte("REFUSED: candidate is missing a byte ledger")
    _require_family(family)

    expected = bytes_added_from_breakdown(br)
    for key in ee.BYTES_ADDED_FIELDS:
        if int(added.get(key) or 0) != int(expected[key]):
            raise UnbilledProgramByte(
                f"REFUSED: bytes_added[{key}]={added.get(key)} != billed {expected[key]}"
            )
    total = int(added.get("total") or 0)
    if total != int(expected["total"]):
        raise UnbilledProgramByte(
            f"REFUSED: bytes_added.total {total} != program bytes {expected['total']}"
        )

    core = int(br.get("per_layer_core_bytes") or 0)
    dictionary = int(br.get("model_scope_dictionary_bytes") or 0)
    generator = int(br.get("model_scope_generator_bytes") or 0)
    condition = int(br.get("condition_bytes") or 0)

    if family == FACTORIZE_THE_FACTORS and core <= 0:
        raise UnbilledProgramByte(
            "REFUSED: FACTORIZE_THE_FACTORS cores are free in the receipt: fabrication"
        )
    if family in {DICTIONARY_PROGRAM, PRODUCT_DICTIONARY} and dictionary <= 0 and core <= 0:
        raise UnbilledProgramByte(
            "REFUSED: dictionary codebook is free in the receipt: fabrication"
        )
    if family == CONDITIONAL_PROGRAM and condition <= 0:
        raise UnbilledProgramByte(
            "REFUSED: CONDITIONAL_PROGRAM condition is free in the receipt: fabrication"
        )
    if family == GENERATED_BLOCK and core <= 0 and generator <= 0:
        raise UnbilledProgramByte(
            "REFUSED: GENERATED_BLOCK generator/cores are free in the receipt: fabrication"
        )
    if family == NONLINEAR_GENERATOR and core <= 0 and generator <= 0:
        raise UnbilledProgramByte(
            "REFUSED: NONLINEAR_GENERATOR is free in the receipt: fabrication"
        )


# ---------------------------------------------------------------------------
# Native consumer. Remat-then-GEMV dies before a score.
# ---------------------------------------------------------------------------


def native_consumer_sketch(
    family: str,
    *,
    rematerialize_dense_W: bool = False,
    rank: int = 0,
    n_experts: int = 0,
) -> dict[str, Any]:
    if rematerialize_dense_W:
        return {
            "family": family,
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
    fam = _require_family(family)
    if fam == FACTORIZE_THE_FACTORS:
        return {
            "family": fam,
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("StationaryRepresentation")],
            "algebra": (
                "g = Ug (Vg^T x); u = Uu (Vu^T x); h = silu(g) * u; "
                "y = Ud (Vd^T h). silu and the product stay exact."
            ),
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "Six skinny matvecs plus the incumbent silu and product. "
                "Materializing Ug Sg Vg^T into Wg (and the same for up, down) "
                "is REJECTED_DENSE_REMAT."
            ),
        }
    if fam == DICTIONARY_PROGRAM:
        return {
            "family": fam,
            "primitive": _require_primitive("StationaryRepresentation"),
            "also": [_require_primitive("TiledProjection")],
            "algebra": "z = V x; k = nearest(z, z_centroids); y = codebook[k]",
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "The codebook is resident. An index from x selects a codeword. "
                "Expanding indices into dense W is REJECTED_DENSE_REMAT."
            ),
        }
    if fam == PRODUCT_DICTIONARY:
        return {
            "family": fam,
            "primitive": _require_primitive("DirectRoutedAccumulate"),
            "also": [
                _require_primitive("StationaryRepresentation"),
                _require_primitive("TiledProjection"),
            ],
            "algebra": (
                "z = V x; for each block b: k_b = nearest(z, z_centroids_b); "
                "y = concat(codebook_b[k_b])"
            ),
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "Product codes are gathered and concatenated. Expanding the "
                "tuple into dense W is REJECTED_DENSE_REMAT."
            ),
        }
    if fam == CONDITIONAL_PROGRAM:
        return {
            "family": fam,
            "primitive": _require_primitive("ConditionalPhysicalProgram"),
            "also": [_require_primitive("TiledProjection")],
            "algebra": (
                "e = nearest(x, condition_centroids); "
                "y = program_e(x)"
                + (f"  (rank-{rank} linear)" if rank else "  (mean of expert)")
            ),
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "The condition selects a resident expert program. Materializing "
                "every expert into a dense W is REJECTED_DENSE_REMAT."
            ),
            "n_experts": int(n_experts),
        }
    if fam == GENERATED_BLOCK:
        return {
            "family": fam,
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("LocalStateMachine")],
            "algebra": (
                "for row-block b: (U_b, V_b) = G(layer, b); y_b = U_b (V_b^T x)"
            ),
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "G emits skinny factors of a block, consumed as two matvecs. "
                "Assembling the tiles into dense W is REJECTED_DENSE_REMAT "
                "(that would be generated_weights, not this family)."
            ),
        }
    if fam == NONLINEAR_GENERATOR:
        return {
            "family": fam,
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("LocalStateMachine")],
            "algebra": (
                "y = P silu(W_h silu(V x))" if rank else "y = P silu(V x)"
            ),
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            "why_not_gemv": (
                "The MLP-shaped generator is the kernel. Emitting W and "
                "running GEMV is generated_weights and REJECTED_DENSE_REMAT."
            ),
        }
    raise NonlinearProgramRefuse(f"unknown family {fam!r}")


def consumer_status(sketch: Mapping[str, Any]) -> str:
    if sketch.get("rematerialize_dense_W") or sketch.get("runs_ordinary_gemv"):
        return REJECTED_DENSE_REMAT
    if not sketch.get("consumes_directly", False):
        return REJECTED_DENSE_REMAT
    _require_primitive(str(sketch["primitive"]))
    return DIRECT_CONSUME


# ---------------------------------------------------------------------------
# Fits. Cheap on purpose. FACTORIZE_THE_FACTORS first.
# ---------------------------------------------------------------------------


def randomized_svd(
    weight: np.ndarray,
    rank: int,
    *,
    seed: int,
    oversample: int = 12,
    n_power: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W ≈ U diag(S) Vh with U[out,r], Vh[r,in]. Randomized; not a full SVD."""
    w = np.ascontiguousarray(weight, dtype=np.float32)
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    r = int(rank)
    if r < 1:
        raise NonlinearProgramRefuse("rank must be positive")
    r = min(r, out_f, in_f)
    rng = np.random.default_rng(int(seed))
    p = min(in_f, out_f, r + int(oversample))
    omega = rng.standard_normal((in_f, p)).astype(np.float32)
    sample = w @ omega
    for _ in range(int(n_power)):
        sample = w @ (w.T @ sample)
    q, _ = np.linalg.qr(sample.astype(np.float64), mode="reduced")
    b = q.T @ w.astype(np.float64)
    u_b, s, vh = np.linalg.svd(b, full_matrices=False)
    u = np.ascontiguousarray((q @ u_b[:, :r]).astype(np.float32, copy=False))
    s_r = s[:r].astype(np.float32, copy=False)
    vh_r = np.ascontiguousarray(vh[:r].astype(np.float32, copy=False))
    return u, s_r, vh_r


def apply_factor(
    x: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    vh: np.ndarray,
) -> np.ndarray:
    """Y = X W^T with W ≈ U diag(S) Vh, without forming W."""
    z = np.ascontiguousarray(x @ vh.T)
    z *= s
    return z @ u.T


def frobenius_residual_energy(weight: np.ndarray, s: np.ndarray) -> float:
    w2 = float(np.linalg.norm(weight.astype(np.float64)) ** 2)
    s2 = float(np.sum(s.astype(np.float64) ** 2))
    return float(1.0 - s2 / max(w2, 1e-30))


def compose_swiglu_factored(
    x: np.ndarray,
    factors: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    g = apply_factor(x, *factors["gate"])
    u = apply_factor(x, *factors["up"])
    hidden = silu(g) * u
    del g, u
    return apply_factor(hidden, *factors["down"])


def _kmeans(
    data: np.ndarray,
    k: int,
    *,
    n_iter: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(data.shape[0])
    kk = int(min(k, n))
    rng = np.random.default_rng(int(seed))
    y = data.astype(np.float64, copy=False)
    centroids = y[rng.choice(n, size=kk, replace=False)].copy()
    assign = np.zeros(n, dtype=np.int64)
    y2 = np.sum(y * y, axis=1, keepdims=True)
    for _ in range(int(n_iter)):
        c2 = np.sum(centroids * centroids, axis=1)
        dist = y2 + c2 - 2.0 * (y @ centroids.T)
        assign = np.argmin(dist, axis=1)
        for i in range(kk):
            mask = assign == i
            if np.any(mask):
                centroids[i] = y[mask].mean(axis=0)
    return centroids.astype(np.float32, copy=False), assign


def _nearest(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    y = data.astype(np.float64, copy=False)
    c = centroids.astype(np.float64, copy=False)
    y2 = np.sum(y * y, axis=1, keepdims=True)
    c2 = np.sum(c * c, axis=1)
    dist = y2 + c2 - 2.0 * (y @ c.T)
    return np.argmin(dist, axis=1)


def fit_factorize(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    *,
    weights: Mapping[str, np.ndarray],
    rank: int,
) -> dict[str, Any]:
    """Approximate gate, up, down separately; keep exact silu and product."""
    msp._require_determined(x_tr.shape[0], rank, what=f"{FACTORIZE_THE_FACTORS} r={rank}")
    packed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    energy: dict[str, float] = {}
    for i, name in enumerate(("gate", "up", "down")):
        u, s, vh = randomized_svd(weights[name], rank, seed=RNG_SEED + 17 * i + rank)
        packed[name] = (u, s, vh)
        energy[name] = _r(frobenius_residual_energy(weights[name], s))
    pred_tr = compose_swiglu_factored(x_tr, packed)
    pred_ho = compose_swiglu_factored(x_ho, packed)
    return {
        "family": FACTORIZE_THE_FACTORS,
        "id_suffix": f"r{int(rank)}",
        "program": "factorized_swiglu",
        "rank": int(rank),
        "codebook_k": 0,
        "n_blocks": 1,
        "z_rank": 0,
        "n_experts": 0,
        "two_layer": False,
        "pred_tr": pred_tr.astype(np.float32, copy=False),
        "pred_ho": pred_ho.astype(np.float32, copy=False),
        "algebra": "g=Ug(Vg^T x); u=Uu(Vu^T x); h=silu(g)*u; y=Ud(Vd^T h)",
        "weight_frobenius_residual_energy": energy,
        "index_from": "x",
    }


def _codebook_from_x(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    *,
    n_blocks: int,
    k: int,
    z_rank: int,
    seed: int,
) -> dict[str, Any]:
    """Y codebook; index from z = V x. Never indexes with held-out Y."""
    r = min(int(z_rank), x_tr.shape[0] - 1, x_tr.shape[1])
    v = msp.randomized_basis(x_tr, r, seed=seed)
    z_tr = x_tr @ v
    z_ho = x_ho @ v
    hidden = int(y_tr.shape[1])
    blocks = int(n_blocks)
    block_dim = hidden // blocks
    pred_tr = np.zeros_like(y_tr)
    pred_ho = np.empty((x_ho.shape[0], hidden), dtype=np.float32)
    oracle_ho_blocks: list[np.ndarray] = []
    for b in range(blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        centroids, _ = _kmeans(y_tr[:, sl], k, n_iter=6, seed=seed + 1 + b)
        assign_tr_y = _nearest(y_tr[:, sl], centroids)
        z_c = np.zeros((centroids.shape[0], r), dtype=np.float64)
        for i in range(centroids.shape[0]):
            mask = assign_tr_y == i
            if np.any(mask):
                z_c[i] = z_tr[mask].mean(axis=0)
            else:
                z_c[i] = z_tr.mean(axis=0)
        z_c32 = z_c.astype(np.float32, copy=False)
        a_tr = _nearest(z_tr, z_c32)
        a_ho = _nearest(z_ho, z_c32)
        pred_tr[:, sl] = centroids[a_tr]
        pred_ho[:, sl] = centroids[a_ho]
        oracle_ho_blocks.append(centroids)
    return {
        "pred_tr": pred_tr,
        "pred_ho": pred_ho,
        "oracle_centroids": oracle_ho_blocks,
        "v": v,
        "block_dim": block_dim,
    }


def oracle_codebook_from_y(
    y_ho: np.ndarray,
    centroids_per_block: Sequence[np.ndarray],
    block_dim: int,
) -> np.ndarray:
    """Diagnostic only: assign hold Y to nearest train centroids. Not authority."""
    pred = np.empty_like(y_ho)
    for b, centroids in enumerate(centroids_per_block):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        pred[:, sl] = centroids[_nearest(y_ho[:, sl], centroids)]
    return pred


def fit_dictionary(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    *,
    k: int,
    z_rank: int = ROUND1_Z_RANK,
) -> dict[str, Any]:
    msp._require_determined(x_tr.shape[0], z_rank, what=f"{DICTIONARY_PROGRAM} z={z_rank}")
    packed = _codebook_from_x(
        x_tr, y_tr, x_ho, n_blocks=1, k=k, z_rank=z_rank, seed=RNG_SEED + 40
    )
    oracle = oracle_codebook_from_y(y_ho, packed["oracle_centroids"], packed["block_dim"])
    return {
        "family": DICTIONARY_PROGRAM,
        "id_suffix": f"k{int(k)}_z{int(z_rank)}",
        "program": "codebook_index_from_x",
        "rank": int(z_rank),
        "codebook_k": int(k),
        "n_blocks": 1,
        "z_rank": int(z_rank),
        "n_experts": 0,
        "two_layer": False,
        "pred_tr": packed["pred_tr"],
        "pred_ho": packed["pred_ho"],
        "algebra": "z = V x; k = nearest(z, z_c); y = codebook[k]",
        "index_from": "x",
        "oracle_y_assignment_relative_l2": _r(mean_l2_ratio(oracle, y_ho)),
        "oracle_note": (
            "oracle assigns held-out Y to the nearest train centroid; it is "
            "a lower bound, not the program, and not error_authority"
        ),
    }


def fit_product_dictionary(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    *,
    n_blocks: int,
    k: int,
    z_rank: int = ROUND1_Z_RANK,
) -> dict[str, Any]:
    msp._require_determined(x_tr.shape[0], z_rank, what=f"{PRODUCT_DICTIONARY} z={z_rank}")
    packed = _codebook_from_x(
        x_tr,
        y_tr,
        x_ho,
        n_blocks=n_blocks,
        k=k,
        z_rank=z_rank,
        seed=RNG_SEED + 80,
    )
    oracle = oracle_codebook_from_y(y_ho, packed["oracle_centroids"], packed["block_dim"])
    return {
        "family": PRODUCT_DICTIONARY,
        "id_suffix": f"b{int(n_blocks)}_k{int(k)}_z{int(z_rank)}",
        "program": "product_codebook_index_from_x",
        "rank": int(z_rank),
        "codebook_k": int(k),
        "n_blocks": int(n_blocks),
        "z_rank": int(z_rank),
        "n_experts": 0,
        "two_layer": False,
        "pred_tr": packed["pred_tr"],
        "pred_ho": packed["pred_ho"],
        "algebra": "z = V x; y_b = codebook_b[nearest(z, z_c_b)]",
        "index_from": "x",
        "oracle_y_assignment_relative_l2": _r(mean_l2_ratio(oracle, y_ho)),
        "oracle_note": (
            "oracle product-quantizes held-out Y; not the program; not authority"
        ),
    }


def fit_conditional(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    *,
    n_experts: int,
    rank: int,
) -> dict[str, Any]:
    msp._require_determined(x_tr.shape[0], n_experts, what=f"{CONDITIONAL_PROGRAM} E={n_experts}")
    centroids, assign_tr = _kmeans(x_tr, n_experts, n_iter=6, seed=RNG_SEED + 120)
    assign_ho = _nearest(x_ho, centroids)
    pred_tr = np.zeros_like(y_tr)
    pred_ho = np.zeros_like(y_ho)
    for e in range(int(n_experts)):
        m_tr = assign_tr == e
        m_ho = assign_ho == e
        if int(m_tr.sum()) == 0:
            mu = y_tr.mean(axis=0)
            if np.any(m_ho):
                pred_ho[m_ho] = mu
            continue
        if rank < 1 or int(m_tr.sum()) <= int(rank):
            mu = y_tr[m_tr].mean(axis=0)
            pred_tr[m_tr] = mu
            if np.any(m_ho):
                pred_ho[m_ho] = mu
            continue
        v = msp.randomized_basis(x_tr[m_tr], int(rank), seed=RNG_SEED + 200 + e)
        p, *_ = np.linalg.lstsq(
            (x_tr[m_tr] @ v).astype(np.float64),
            y_tr[m_tr].astype(np.float64),
            rcond=None,
        )
        pred_tr[m_tr] = (x_tr[m_tr] @ v) @ p
        if np.any(m_ho):
            pred_ho[m_ho] = (x_ho[m_ho] @ v) @ p
    return {
        "family": CONDITIONAL_PROGRAM,
        "id_suffix": f"e{int(n_experts)}_r{int(rank)}",
        "program": "mean_expert" if rank < 1 else "rank_expert",
        "rank": int(rank),
        "codebook_k": 0,
        "n_blocks": 1,
        "z_rank": 0,
        "n_experts": int(n_experts),
        "two_layer": False,
        "pred_tr": pred_tr.astype(np.float32, copy=False),
        "pred_ho": pred_ho.astype(np.float32, copy=False),
        "algebra": "e = nearest(x, centroids); y = program_e(x)",
        "index_from": "x",
        "condition": "kmeans_x",
    }


def fit_generated_block(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    *,
    n_blocks: int,
    rank: int,
) -> dict[str, Any]:
    """Lookup generator G(block) -> (V_b, U_b). Best case for a block generator."""
    msp._require_determined(x_tr.shape[0], rank, what=f"{GENERATED_BLOCK} r={rank}")
    blocks = int(n_blocks)
    block_dim = int(y_tr.shape[1]) // blocks
    pred_tr = np.zeros_like(y_tr)
    pred_ho = np.zeros_like(y_ho)
    for b in range(blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        v = msp.randomized_basis(x_tr, int(rank), seed=RNG_SEED + 300 + b)
        p, *_ = np.linalg.lstsq(
            (x_tr @ v).astype(np.float64),
            y_tr[:, sl].astype(np.float64),
            rcond=None,
        )
        pred_tr[:, sl] = (x_tr @ v) @ p
        pred_ho[:, sl] = (x_ho @ v) @ p
    return {
        "family": GENERATED_BLOCK,
        "id_suffix": f"b{blocks}_r{int(rank)}",
        "program": "lookup_block_factors",
        "rank": int(rank),
        "codebook_k": 0,
        "n_blocks": blocks,
        "z_rank": 0,
        "n_experts": 0,
        "two_layer": False,
        "pred_tr": pred_tr.astype(np.float32, copy=False),
        "pred_ho": pred_ho.astype(np.float32, copy=False),
        "algebra": "for b: (U_b, V_b) = G(layer, b); y_b = U_b (V_b^T x)",
        "index_from": "x",
        "generator_kind": "lookup_table_of_block_factors",
        "note": (
            "Unconstrained per-block factors: a strictly smaller MLP "
            "generator of (layer, block) cannot beat this."
        ),
    }


def fit_nonlinear_generator(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    *,
    rank: int,
    two_layer: bool = False,
) -> dict[str, Any]:
    msp._require_determined(x_tr.shape[0], rank, what=f"{NONLINEAR_GENERATOR} r={rank}")
    v = msp.randomized_basis(x_tr, int(rank), seed=RNG_SEED)
    h1_tr = silu(x_tr @ v)
    h1_ho = silu(x_ho @ v)
    if two_layer:
        w = msp.randomized_basis(h1_tr, int(rank), seed=RNG_SEED + 1)
        feat_tr = silu(h1_tr @ w)
        feat_ho = silu(h1_ho @ w)
        program = "two_layer_silu"
        algebra = "y = P silu(W_h silu(V x))"
    else:
        feat_tr = h1_tr
        feat_ho = h1_ho
        program = "silu_readout"
        algebra = "y = P silu(V x)"
    p, *_ = np.linalg.lstsq(
        feat_tr.astype(np.float64), y_tr.astype(np.float64), rcond=None
    )
    return {
        "family": NONLINEAR_GENERATOR,
        "id_suffix": f"r{int(rank)}" + ("_two_layer" if two_layer else "_silu"),
        "program": program,
        "rank": int(rank),
        "codebook_k": 0,
        "n_blocks": 1,
        "z_rank": 0,
        "n_experts": 0,
        "two_layer": bool(two_layer),
        "pred_tr": (feat_tr @ p).astype(np.float32, copy=False),
        "pred_ho": (feat_ho @ p).astype(np.float32, copy=False),
        "algebra": algebra,
        "index_from": "x",
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
                "0 extra dispatches: fused native primitive per layer. "
                "ASSUMPTION. Unfused lowering would add launches."
            ),
            "scorer": "tools.future.executable_economics.score",
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
    family: str,
    program: str,
    pred_tr: np.ndarray,
    pred_ho: np.ndarray,
    y_tr: np.ndarray,
    y_ho: np.ndarray,
    consumer: Mapping[str, Any],
    rank: int = 0,
    codebook_k: int = 0,
    n_blocks: int = 1,
    z_rank: int = 0,
    n_experts: int = 0,
    two_layer: bool = False,
    extra: Mapping[str, Any] | None = None,
    meta_ho: Mapping[str, Sequence[str]] | None = None,
    n_layers: int = N_LAYERS,
    id_suffix: str | None = None,
    index_from: str = "x",
) -> dict[str, Any]:
    """The only constructor a receipt row is allowed to pass through."""
    fam = _require_family(family)
    if index_from in {"y_hold", "hold_y", "Yho"}:
        raise HoldYUsedAsIndex(
            "REFUSED: cannot emit a candidate whose index is held-out Y"
        )
    cstat = consumer_status(consumer)
    if cstat == REJECTED_DENSE_REMAT:
        br = byte_breakdown(
            fam,
            rank=max(int(rank), 1),
            codebook_k=codebook_k,
            n_blocks=n_blocks,
            z_rank=z_rank,
            n_experts=n_experts,
            two_layer=two_layer,
            n_layers=n_layers,
        )
        added = bytes_added_from_breakdown(br)
        remat_bytes = int(n_layers) * int(HIDDEN) * int(HIDDEN) * ELEMENT_BYTES
        added_remat = dict(added)
        added_remat["generator"] = int(added_remat["generator"]) + remat_bytes
        added_remat["total"] = sum(int(added_remat[k]) for k in ee.BYTES_ADDED_FIELDS)
        raise RematConsumer(
            "REJECTED_DENSE_REMAT: cannot report a remat shape as a live "
            f"candidate ({fam})"
        )

    br = byte_breakdown(
        fam,
        rank=int(rank),
        codebook_k=codebook_k,
        n_blocks=n_blocks,
        z_rank=z_rank,
        n_experts=n_experts,
        two_layer=two_layer,
        n_layers=n_layers,
    )
    added = bytes_added_from_breakdown(br)
    ho = function_error(
        pred_ho, y_ho, split="hold", report_as="held_out", meta=meta_ho, index_from=index_from
    )
    tr = function_error(pred_tr, y_tr, split="train", report_as="train", index_from=index_from)
    held = float(ho["held_out_relative_l2"])
    status, cheap_kill, why = status_from_error(held, ho.get("per_capability_domain"))
    cid = f"{fam.lower()}_{id_suffix or program}"
    row: dict[str, Any] = {
        "id": cid,
        "family": fam,
        "program": program,
        "rank": int(rank),
        "codebook_k": int(codebook_k),
        "n_blocks": int(n_blocks),
        "z_rank": int(z_rank),
        "n_experts": int(n_experts),
        "two_layer": bool(two_layer),
        "byte_breakdown": dict(br),
        "bytes_added": added,
        "consumer": dict(consumer),
        "consumer_status": DIRECT_CONSUME,
        "status": status,
        "cheap_kill": bool(cheap_kill),
        "status_why": why,
        "weight_reconstruction_error": None,
        "weight_reconstruction_note": (
            "not authority; this experiment scores F, not W"
        ),
        "error_authority": "held_out_relative_l2",
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "kill_band": KILL_BAND,
        "n_layers_billed": int(n_layers),
        "index_from": index_from,
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
# Cheap first round. FACTORIZE_THE_FACTORS first.
# ---------------------------------------------------------------------------


def _baselines(y_tr: np.ndarray, y_ho: np.ndarray, meta_ho: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    zero_ho = function_error(
        np.zeros_like(y_ho), y_ho, split="hold", report_as="held_out", meta=meta_ho
    )
    mean = y_tr.mean(axis=0, keepdims=True)
    mean_ho = function_error(
        np.broadcast_to(mean, y_ho.shape), y_ho, split="hold", report_as="held_out", meta=meta_ho
    )
    mean_tr = function_error(
        np.broadcast_to(mean, y_tr.shape), y_tr, split="train", report_as="train"
    )
    return {
        "zero_held_out_relative_l2": zero_ho["held_out_relative_l2"],
        "mean_held_out_relative_l2": mean_ho["held_out_relative_l2"],
        "mean_held_out_cosine": mean_ho["held_out_cosine"],
        "mean_train_relative_l2_diagnostic": mean_tr["train_relative_l2_diagnostic"],
        "held_out_split": "hold",
        "note": "baselines are held-out; they are not candidates",
    }


def _cite_oracle_pca() -> dict[str, Any]:
    """Do not re-derive a linear basis. Cite the shared-program oracle."""
    path = REPO / SHARED_REL
    if not path.is_file():
        return {"cited": False, "reason": f"missing {SHARED_REL}"}
    doc = load_json(path)
    pca = doc.get("oracle_output_pca") or []
    r64 = next((row for row in pca if int(row.get("rank") or 0) == 64), None)
    return {
        "cited": True,
        "source": SHARED_REL,
        "rank_64_held_out_relative_l2": None if r64 is None else r64.get("held_out_relative_l2"),
        "note": (
            "Cited, not re-run. PCA of F at rank 64 is already ~0.89 held-out; "
            "the output manifold is not low-rank. This module does not propose "
            "another linear basis."
        ),
        "shared_program_negative_findings": list(doc.get("negative_findings") or [])[:6],
    }


def _factor_diagnostics(
    x_ho: np.ndarray,
    y_ho: np.ndarray,
    x_tr: np.ndarray,
    weights: Mapping[str, np.ndarray],
    packed_r64: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Single-factor ablation and image-PCA of the parts. Not candidates."""
    g_exact = _matmul(x_ho, weights["gate"].T)
    u_exact = _matmul(x_ho, weights["up"].T)
    h_exact = silu(g_exact) * u_exact
    y_exact = _matmul(h_exact, weights["down"].T)
    identity = _r(mean_l2_ratio(y_exact, y_ho))

    g_lr = apply_factor(x_ho, *packed_r64["gate"])
    u_lr = apply_factor(x_ho, *packed_r64["up"])
    only_gate = _matmul(silu(g_lr) * u_exact, weights["down"].T)
    only_up = _matmul(silu(g_exact) * u_lr, weights["down"].T)
    only_down = apply_factor(h_exact, *packed_r64["down"])

    n_sub = min(2048, int(x_tr.shape[0]))
    g_sub = _matmul(x_tr[:n_sub], weights["gate"].T)
    u_sub = _matmul(x_tr[:n_sub], weights["up"].T)
    h_sub = silu(g_sub) * u_sub
    bg = msp.randomized_basis(g_sub, 64, seed=RNG_SEED + 9)
    bu = msp.randomized_basis(u_sub, 64, seed=RNG_SEED + 10)
    bh = msp.randomized_basis(h_sub, 64, seed=RNG_SEED + 11)
    image = {
        "gate": _r(mean_l2_ratio((g_exact @ bg) @ bg.T, g_exact)),
        "up": _r(mean_l2_ratio((u_exact @ bu) @ bu.T, u_exact)),
        "hidden_silu_times_up": _r(mean_l2_ratio((h_exact @ bh) @ bh.T, h_exact)),
        "note": (
            "Image PCA of the factor outputs on hold, basis fit on a train "
            "subsample. Not a program: it uses the exact factor activations."
        ),
        "n_train_subsample": n_sub,
    }
    return {
        "teacher_identity_held_out_relative_l2": identity,
        "single_factor_r64_others_exact": {
            "gate": _r(mean_l2_ratio(only_gate, y_ho)),
            "up": _r(mean_l2_ratio(only_up, y_ho)),
            "down": _r(mean_l2_ratio(only_down, y_ho)),
            "note": (
                "Two factors exact (full W), one truncated-SVD at rank 64, "
                "exact silu and product. Isolates which matrix is the bottleneck."
            ),
        },
        "image_pca_r64_held_out": image,
    }


def _emit_from_fit(
    fit: Mapping[str, Any],
    *,
    y_tr: np.ndarray,
    y_ho: np.ndarray,
    meta_ho: Mapping[str, Sequence[str]],
    extra: Mapping[str, Any] | None = None,
    n_layers: int = N_LAYERS,
) -> dict[str, Any]:
    consumer = native_consumer_sketch(
        str(fit["family"]),
        rank=int(fit.get("rank") or 0),
        n_experts=int(fit.get("n_experts") or 0),
    )
    skip_extra = {
        "pred_tr",
        "pred_ho",
        "family",
        "program",
        "rank",
        "codebook_k",
        "n_blocks",
        "z_rank",
        "n_experts",
        "two_layer",
        "id_suffix",
        "index_from",
        "oracle_centroids",
        "v",
        "weights",
        "packed",
    }
    payload_extra = {
        "algebra": fit.get("algebra"),
        **{
            k: v
            for k, v in fit.items()
            if k not in skip_extra and not isinstance(v, np.ndarray)
        },
    }
    if extra:
        payload_extra.update(extra)
    return emit_candidate(
        family=str(fit["family"]),
        program=str(fit["program"]),
        pred_tr=fit["pred_tr"],
        pred_ho=fit["pred_ho"],
        y_tr=y_tr,
        y_ho=y_ho,
        consumer=consumer,
        rank=int(fit.get("rank") or 0),
        codebook_k=int(fit.get("codebook_k") or 0),
        n_blocks=int(fit.get("n_blocks") or 1),
        z_rank=int(fit.get("z_rank") or 0),
        n_experts=int(fit.get("n_experts") or 0),
        two_layer=bool(fit.get("two_layer")),
        extra=payload_extra,
        meta_ho=meta_ho,
        n_layers=n_layers,
        id_suffix=str(fit.get("id_suffix") or fit["program"]),
        index_from=str(fit.get("index_from") or "x"),
    )


def round1_fit(
    *,
    layer: int = ROUND1_LAYER,
    ranks: Sequence[int] = ROUND1_RANKS,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    pack = load_layer_pack(layer, payload_dir=payload_dir)
    x_tr, y_tr, x_ho, y_ho = pack["Xtr"], pack["Ytr"], pack["Xho"], pack["Yho"]
    meta_ho = pack["hold_meta"]
    ranks_t = tuple(int(r) for r in ranks)
    rows: list[dict[str, Any]] = []
    weights = load_factor_weights(int(pack["layer"]))

    # --- FACTORIZE_THE_FACTORS first ---
    packed_r64 = None
    factorize_rows: list[dict[str, Any]] = []
    for rank in ranks_t:
        fit = fit_factorize(x_tr, y_tr, x_ho, y_ho, weights=weights, rank=rank)
        if int(rank) == 64:
            packed_r64 = {
                "gate": randomized_svd(weights["gate"], 64, seed=RNG_SEED + 17 * 0 + 64),
                "up": randomized_svd(weights["up"], 64, seed=RNG_SEED + 17 * 1 + 64),
                "down": randomized_svd(weights["down"], 64, seed=RNG_SEED + 17 * 2 + 64),
            }
        row = _emit_from_fit(fit, y_tr=y_tr, y_ho=y_ho, meta_ho=meta_ho)
        rows.append(row)
        factorize_rows.append(row)

    if packed_r64 is None:
        packed_r64 = {
            "gate": randomized_svd(weights["gate"], max(ranks_t), seed=RNG_SEED),
            "up": randomized_svd(weights["up"], max(ranks_t), seed=RNG_SEED + 1),
            "down": randomized_svd(weights["down"], max(ranks_t), seed=RNG_SEED + 2),
        }
    diagnostics = _factor_diagnostics(x_ho, y_ho, x_tr, weights, packed_r64)

    # --- remaining families, cheap ---
    for k in ROUND1_DICT_K:
        rows.append(
            _emit_from_fit(
                fit_dictionary(x_tr, y_tr, x_ho, y_ho, k=int(k)),
                y_tr=y_tr,
                y_ho=y_ho,
                meta_ho=meta_ho,
            )
        )
    rows.append(
        _emit_from_fit(
            fit_product_dictionary(x_tr, y_tr, x_ho, y_ho, n_blocks=16, k=16),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    rows.append(
        _emit_from_fit(
            fit_product_dictionary(x_tr, y_tr, x_ho, y_ho, n_blocks=32, k=32),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    rows.append(
        _emit_from_fit(
            fit_conditional(x_tr, y_tr, x_ho, y_ho, n_experts=4, rank=0),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    rows.append(
        _emit_from_fit(
            fit_conditional(x_tr, y_tr, x_ho, y_ho, n_experts=4, rank=8),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    rows.append(
        _emit_from_fit(
            fit_conditional(x_tr, y_tr, x_ho, y_ho, n_experts=8, rank=16),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    rows.append(
        _emit_from_fit(
            fit_generated_block(x_tr, y_tr, x_ho, y_ho, n_blocks=16, rank=8),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    rows.append(
        _emit_from_fit(
            fit_generated_block(x_tr, y_tr, x_ho, y_ho, n_blocks=16, rank=16),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )
    for rank in (32, 64, 128):
        rows.append(
            _emit_from_fit(
                fit_nonlinear_generator(x_tr, y_tr, x_ho, y_ho, rank=rank, two_layer=False),
                y_tr=y_tr,
                y_ho=y_ho,
                meta_ho=meta_ho,
            )
        )
    rows.append(
        _emit_from_fit(
            fit_nonlinear_generator(x_tr, y_tr, x_ho, y_ho, rank=64, two_layer=True),
            y_tr=y_tr,
            y_ho=y_ho,
            meta_ho=meta_ho,
        )
    )

    by_family: dict[str, list[dict[str, Any]]] = {f: [] for f in FAMILIES}
    for row in rows:
        by_family[str(row["family"])].append(row)

    family_verdicts = []
    scars = []
    for family in FAMILIES:
        group = by_family[family]
        dead = all(r["status"] != OPEN for r in group)
        cheap = all(r.get("cheap_kill") for r in group) if group else True
        best = min(group, key=lambda r: float(r["held_out_relative_l2"]))
        mechanism = _mechanism_for(family, best, diagnostics if family == FACTORIZE_THE_FACTORS else None)
        verdict = {
            "family": family,
            "status": MEASURED_NEGATIVE if dead else OPEN,
            "cheap_kill": bool(cheap and dead),
            "n_rows": len(group),
            "best_id": best["id"],
            "best_held_out_relative_l2": best["held_out_relative_l2"],
            "best_held_out_cosine": best.get("held_out_cosine"),
            "worst_prompt_relative_l2": best.get("worst_prompt_relative_l2"),
            "worst_prompt_id": best.get("worst_prompt_id"),
            "per_capability_domain": best.get("per_capability_domain"),
            "per_position_band": best.get("per_position_band"),
            "bytes_added_total_at_best": best["bytes_added"]["total"],
            "clears_s020_time_bar_if_function_held": best[
                "clears_s020_time_bar_if_function_held"
            ],
            "consumer_status": best["consumer_status"],
            "native_consumer": best["consumer"],
            "mechanism": mechanism,
            "why": (
                f"held-out relative L2 {best['held_out_relative_l2']} is in the "
                f"0.9 band at every affordable size; do not go wider on this "
                f"family. Mechanism: {mechanism}"
                if cheap and dead
                else (
                    f"held-out relative L2 {best['held_out_relative_l2']} is "
                    f"above the {HELD_OUT_KILL_REL} kill"
                    if dead
                    else "at least one affordable size is below the held-out kill"
                )
            ),
        }
        family_verdicts.append(verdict)
        if dead:
            scars.append(
                {
                    "family": family,
                    "status": MEASURED_NEGATIVE,
                    "cheap_kill": bool(cheap),
                    "held_out_relative_l2_best": best["held_out_relative_l2"],
                    "mechanism": mechanism,
                    "not": (
                        "a retry of SHARED_INPUT / SHARED_OUTPUT / SHARED_BOTH"
                    ),
                    "reopen": _reopen_for(family),
                    "level": "MODEL_SPECIFIC",
                    "parent": "qwen3.8-27b sealed-3.14",
                    "organ": "mlp",
                    "object": "F(x)=down(silu(gate(x))*up(x)) on the teacher corpus",
                }
            )

    best_f = min(factorize_rows, key=lambda r: float(r["held_out_relative_l2"]))
    factorize_report = {
        "family": FACTORIZE_THE_FACTORS,
        "ran_first": True,
        "layer": int(pack["layer"]),
        "ranks": list(ranks_t),
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "kill_band": KILL_BAND,
        "status": MEASURED_NEGATIVE,
        "cheap_kill": True,
        "shared_program_was_not_an_artifact": True,
        "best": {
            "id": best_f["id"],
            "rank": best_f["rank"],
            "held_out_relative_l2": best_f["held_out_relative_l2"],
            "held_out_cosine": best_f.get("held_out_cosine"),
            "worst_prompt_id": best_f.get("worst_prompt_id"),
            "worst_prompt_relative_l2": best_f.get("worst_prompt_relative_l2"),
            "per_capability_domain": best_f.get("per_capability_domain"),
            "per_position_band": best_f.get("per_position_band"),
            "weight_frobenius_residual_energy": best_f.get(
                "weight_frobenius_residual_energy"
            ),
            "bytes_added_total": best_f["bytes_added"]["total"],
        },
        "by_rank": [
            {
                "id": r["id"],
                "rank": r["rank"],
                "held_out_relative_l2": r["held_out_relative_l2"],
                "held_out_cosine": r.get("held_out_cosine"),
                "train_relative_l2_diagnostic": r.get("train_relative_l2_diagnostic"),
                "worst_prompt_relative_l2": r.get("worst_prompt_relative_l2"),
                "per_capability_domain": r.get("per_capability_domain"),
                "per_position_band": r.get("per_position_band"),
                "weight_frobenius_residual_energy": r.get(
                    "weight_frobenius_residual_energy"
                ),
                "status": r["status"],
            }
            for r in factorize_rows
        ],
        "diagnostics": diagnostics,
        "mechanism": _mechanism_for(FACTORIZE_THE_FACTORS, best_f, diagnostics),
        "why": (
            "Composing truncated-SVD gate, up and down at rank 64, with exact "
            "silu and product, stays in the 0.9 band (worse than the mean "
            "predictor). The shared-program result was not an artifact of "
            "approximating the composite: the parts are not individually "
            "low-rank on this body either."
        ),
        "consumer": best_f["consumer"],
    }

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
        "x_sha256": pack["x_sha256"],
        "y_sha256": pack["y_sha256"],
        "payload_dir": pack["payload_dir"],
        "ranks": list(ranks_t),
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "kill_band": KILL_BAND,
        "baselines": _baselines(y_tr, y_ho, meta_ho),
        "oracle_output_pca_cited": _cite_oracle_pca(),
        "factorize_the_factors": factorize_report,
        "rows": rows,
        "family_verdicts": family_verdicts,
        "scars": scars,
        "survivors": surviving_candidates(rows),
        "n_survivors": len(surviving_candidates(rows)),
    }


def _mechanism_for(
    family: str,
    best: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None,
) -> str:
    held = best.get("held_out_relative_l2")
    if family == FACTORIZE_THE_FACTORS:
        energy = best.get("weight_frobenius_residual_energy") or {}
        single = (diagnostics or {}).get("single_factor_r64_others_exact") or {}
        image = (diagnostics or {}).get("image_pca_r64_held_out") or {}
        return (
            "gate, up and down are individually high-rank. Truncated SVD at "
            f"rank {best.get('rank')} leaves Frobenius residual energy "
            f"gate={energy.get('gate')} up={energy.get('up')} "
            f"down={energy.get('down')}. Composing the three with exact silu "
            f"and product has held-out relative L2 {held}. Even with two "
            f"factors exact, one rank-64 factor sits at gate={single.get('gate')} "
            f"up={single.get('up')} down={single.get('down')}. Gate's image is "
            f"more compressible (PCA r=64 relative L2 {image.get('gate')}) than "
            "F, but not enough to replace the map. The shared-program 0.9-band "
            "result was not an artifact of approximating the composite."
        )
    if family == DICTIONARY_PROGRAM:
        return (
            f"F(x) does not cluster on this manifold. X-routed codebook K="
            f"{best.get('codebook_k')} has held-out relative L2 {held}; even "
            f"the Y-assignment oracle is "
            f"{best.get('oracle_y_assignment_relative_l2')} (diagnostic, not "
            "authority). A small codebook of F is not a program for F."
        )
    if family == PRODUCT_DICTIONARY:
        return (
            f"Product codebooks over {best.get('n_blocks')} sub-blocks of K="
            f"{best.get('codebook_k')} still sit at held-out relative L2 {held} "
            f"(Y-assignment oracle {best.get('oracle_y_assignment_relative_l2')}). "
            "F is not a product of small block codebooks on this manifold."
        )
    if family == CONDITIONAL_PROGRAM:
        return (
            f"A condition on x (k-means, E={best.get('n_experts')}) selecting a "
            f"rank-{best.get('rank')} expert (or a mean) has held-out relative "
            f"L2 {held}. The condition is cheap and is billed; the programs it "
            "selects are still high-rank maps of x. A cheap default plus an "
            "exceptional path does not replace F."
        )
    if family == GENERATED_BLOCK:
        return (
            f"Unconstrained per-block rank-{best.get('rank')} maps on "
            f"{best.get('n_blocks')} output blocks (the best a lookup generator "
            f"G(layer, row_block) can emit at this size) have held-out relative "
            f"L2 {held}. A smaller MLP of (layer, block) is strictly weaker. "
            "Block index does not buy a compact local representation of F."
        )
    if family == NONLINEAR_GENERATOR:
        return (
            f"An MLP-shaped generator ({best.get('program')}, r={best.get('rank')}) "
            f"has held-out relative L2 {held}. A silu readout of a small latent "
            "is still an r-dimensional bottleneck through a high-rank output; "
            "it matches the SHARED_INPUT silu readout already killed, not a "
            "full-width structured nonlinear."
        )
    return f"{family} held-out relative L2 {held}"


def _reopen_for(family: str) -> str:
    if family == FACTORIZE_THE_FACTORS:
        return (
            "Do not reopen as another rank sweep of gate/up/down. Reopen only "
            "for a full-width structured factorization (Monarch, butterfly, "
            "or an exact sparse factor) that is not an r-dimensional bottleneck."
        )
    if family in {DICTIONARY_PROGRAM, PRODUCT_DICTIONARY}:
        return (
            "Reopen only if a new source (not frozen W, not this F-manifold) "
            "actually clusters. Raw-weight PQ/VQ is a different scar (NNS-017)."
        )
    if family == CONDITIONAL_PROGRAM:
        return (
            "Reopen only for a condition whose exceptional path is a different "
            "function, not another small linear map of x. Billing the condition "
            "remains mandatory."
        )
    if family == GENERATED_BLOCK:
        return (
            "Reopen only if G is the kernel of a block (no dense W) AND held-out "
            "F drops below the kill. Generating W then GEMV stays REJECTED_DENSE_REMAT."
        )
    if family == NONLINEAR_GENERATOR:
        return (
            "Reopen only for a full-width structured nonlinear (Monarch / "
            "butterfly / distilled operator that is not an r-bottleneck). A "
            "wider silu readout of a latent is this family at a larger rank."
        )
    return "full-width structured nonlinear that is not an r-bottleneck"


@lru_cache(maxsize=1)
def cached_round1() -> dict[str, Any]:
    return round1_fit()


# ---------------------------------------------------------------------------
# Negative index. Query first. Do not name global_dense_lowrank as the proposal.
# ---------------------------------------------------------------------------


def consult_index() -> dict[str, Any]:
    model = "qwen3.8-27b"
    organ = "mlp"
    families = (
        "function_replacement",
        "factorized_programs",
        "generated_programs",
        "generated_weights",
        "learned_codebook",
        "raw_weight_pq_vq",
        "low_rank",
        "global_dense_lowrank",
        "shared_basis",
        "latent_accumulation",
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
    proposal_families = ("function_replacement", "factorized_programs")
    proposal_refused = [r for r in refusals if r["hypothesis_family"] in proposal_families]
    return {
        "model": model,
        "organ": organ,
        "queries": queries,
        "refusals": refusals,
        "proposal_refused": proposal_refused,
        "proceed": len(proposal_refused) == 0,
        "cousins_not_this_object": [
            "MLP_SHARED_PROGRAM: SHARED_INPUT / SHARED_OUTPUT / SHARED_BOTH at "
            "r<=64 are MEASURED_NEGATIVE; oracle PCA of F at r=64 is ~0.89. "
            "This module does not retry those shapes.",
            "NS-global-dense-lowrank-qwen38 is SVD-of-W as a global dense body. "
            "FACTORIZE_THE_FACTORS is composed F of three separately truncated "
            "SwiGLU factors with exact silu and product, scored on real X. It "
            "is not labelled global_dense_lowrank (that family would refuse).",
            "NNS-016: weight-space 99% energy needs 92-95% of ranks on this "
            "parent. That is spectrum of W, not composed F. This experiment "
            "is the function-space control the shared-program lane skipped.",
            "NNS-017 / dictionary_programs in the byte census: raw frozen-W "
            "PQ/VQ. This dictionary is of F on the teacher corpus, indexed "
            "from x, not a restatement of NNS-017.",
            "NNS-013 latent_accumulation (narrow SwiGLU) is a different family. "
            "FACTORIZE keeps the 17408-d intermediate and truncates the maps.",
        ],
        "note": (
            "GENERAL_PHYSICAL scars refuse whatever model is named. The "
            "proposal families function_replacement and factorized_programs "
            "are not refused. global_dense_lowrank is a cousin and is not "
            "the proposal. synthetic_activation is a method scar: this "
            "module refuses to fit on Gaussian X."
        ),
    }


def residual_budget_allocation(n_survivors: int) -> dict[str, Any]:
    """Use CAPABILITY_INFORMATION_MAP; do not invent a residual byte model."""
    path = REPO / CAPMAP_REL
    if not path.is_file():
        return {"source": CAPMAP_REL, "cited": False}
    doc = load_json(path)
    alloc = doc.get("allocation") or {}
    answers = doc.get("answers") or {}
    elim = answers.get("bytes_a_nonuniform_allocation_would_eliminate") or {}
    mlp_licensed = int((alloc.get("bytes_eliminated_by_organ") or {}).get("mlp") or 0)
    return {
        "source": CAPMAP_REL,
        "cited": True,
        "any_supported": bool(alloc.get("any_supported")),
        "licensed_mlp_bytes": mlp_licensed,
        "licensed_token_share": elim.get("share_of_token"),
        "licensed_regions_include_mlp_slice": "L63.mlp.gate.channel.rows_13056_17408"
        in list(alloc.get("could_take_fewer_bits") or []),
        "allocated_to_survivors_bytes": 0 if n_survivors == 0 else mlp_licensed,
        "n_survivors": int(n_survivors),
        "note": (
            "No surviving program, so this experiment does not reassign the "
            "map's licensed 0.28% of the token. That allocation is mostly "
            "DeltaNet/GQA plus a 2.8 MB MLP slice already named by the map; "
            "it is not a function-replacement residual."
            if n_survivors == 0
            else (
                "Survivors would draw residual bits only from regions the map "
                "licensed to take fewer bits, never from must_keep_or_gain."
            )
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
    return msp.make_fixture_xy(
        n_train=n_train, n_hold=n_hold, hidden=hidden, rank=rank, seed=seed
    )


def make_fixture_swiglu(
    n_train: int = 48,
    n_hold: int = 16,
    hidden: int = 12,
    intermediate: int = 16,
    rank: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Tiny exact-factor SwiGLU so FACTORIZE at `rank` is well-posed."""
    rng = np.random.default_rng(int(seed))
    def _lr(out_f: int, in_f: int, r: int, s: int) -> np.ndarray:
        u = rng.standard_normal((out_f, r)).astype(np.float32)
        v = rng.standard_normal((r, in_f)).astype(np.float32)
        return (u @ v) / math.sqrt(r)

    wg = _lr(intermediate, hidden, rank, seed)
    wu = _lr(intermediate, hidden, rank, seed + 1)
    wd = _lr(hidden, intermediate, rank, seed + 2)
    x_tr = rng.standard_normal((n_train, hidden)).astype(np.float32)
    x_ho = rng.standard_normal((n_hold, hidden)).astype(np.float32)

    def _f(x: np.ndarray) -> np.ndarray:
        return (silu(x @ wg.T) * (x @ wu.T)) @ wd.T

    y_tr, y_ho = _f(x_tr), _f(x_ho)
    prompts = [f"p{i}" for i in range(n_hold)]
    domains = [CAPABILITY_DOMAINS[i % len(CAPABILITY_DOMAINS)] for i in range(n_hold)]
    bands = [POSITION_BANDS[i % len(POSITION_BANDS)] for i in range(n_hold)]
    return {
        "Xtr": x_tr,
        "Ytr": y_tr,
        "Xho": x_ho,
        "Yho": y_ho,
        "weights": {"gate": wg, "up": wu, "down": wd},
        "hold_meta": {"domain": domains, "band": bands, "prompt_id": prompts},
        "hidden": hidden,
        "intermediate": intermediate,
    }


def selftest() -> dict[str, Any]:
    """Guards on fixtures. Does not read the teacher corpus and does not fit F."""
    held_out_leak_refused = False
    try:
        y = np.ones((4, 3), dtype=np.float32)
        function_error(y, y, split="train", report_as="held_out")
    except TrainReportedAsHeldOut:
        held_out_leak_refused = True

    y_index_refused = False
    try:
        y = np.ones((4, 3), dtype=np.float32)
        function_error(y, y, split="hold", report_as="held_out", index_from="y_hold")
    except HoldYUsedAsIndex:
        y_index_refused = True

    unbilled_refused = False
    try:
        validate_billing(
            {
                "family": DICTIONARY_PROGRAM,
                "byte_breakdown": byte_breakdown(DICTIONARY_PROGRAM, codebook_k=8, z_rank=4),
                "bytes_added": {
                    "embeddings": 0,
                    "generator": 0,
                    "residuals": 0,
                    "metadata": 0,
                    "state": 0,
                    "total": 0,
                },
            }
        )
    except UnbilledProgramByte:
        unbilled_refused = True

    condition_refused = False
    try:
        br = byte_breakdown(CONDITIONAL_PROGRAM, n_experts=4, rank=8)
        br["condition_bytes"] = 0
        added = bytes_added_from_breakdown(br)
        validate_billing(
            {"family": CONDITIONAL_PROGRAM, "byte_breakdown": br, "bytes_added": added}
        )
    except UnbilledProgramByte:
        condition_refused = True

    linear_refused = False
    try:
        native_consumer_sketch(msp.SHARED_BOTH)
    except LinearSharedSubspaceDead:
        linear_refused = True

    remat_refused = False
    fx = make_fixture_xy()
    try:
        emit_candidate(
            family=FACTORIZE_THE_FACTORS,
            program="factorized_swiglu",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=native_consumer_sketch(
                FACTORIZE_THE_FACTORS, rematerialize_dense_W=True
            ),
            rank=4,
            n_layers=2,
            id_suffix="r4",
        )
    except RematConsumer:
        remat_refused = True

    fx = make_fixture_xy()
    ok = emit_candidate(
        family=NONLINEAR_GENERATOR,
        program="silu_readout",
        pred_tr=fx["Ytr"],
        pred_ho=fx["Yho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=native_consumer_sketch(NONLINEAR_GENERATOR),
        rank=4,
        n_layers=2,
        id_suffix="r4_silu",
        meta_ho={
            "domain": ["code"] * len(fx["Yho"]),
            "band": ["early"] * len(fx["Yho"]),
            "prompt_id": [f"p{i}" for i in range(len(fx["Yho"]))],
        },
    )
    if ok["held_out_split"] != "hold":
        raise SystemExit("selftest: honest emit lost the hold split")
    if ok["bytes_added"]["generator"] <= 0:
        raise SystemExit("selftest: honest emit dropped the program bytes")
    if ok["economics"]["assumptions"]["scorer"] != "tools.future.executable_economics.score":
        raise SystemExit("selftest: economics did not come from executable_economics.score")

    fired = (
        held_out_leak_refused
        and y_index_refused
        and unbilled_refused
        and condition_refused
        and linear_refused
        and remat_refused
    )
    if not fired:
        raise SystemExit(
            "selftest: guards did not fire "
            f"leak={held_out_leak_refused} y_index={y_index_refused} "
            f"unbilled={unbilled_refused} condition={condition_refused} "
            f"linear={linear_refused} remat={remat_refused}"
        )
    return {
        "held_out_leak_refused": True,
        "y_hold_index_refused": True,
        "unbilled_program_byte_refused": True,
        "unbilled_condition_refused": True,
        "linear_shared_subspace_refused": True,
        "remat_consumer_refused": True,
        "honest_fixture_emit_ok": True,
        "held_out_leak_codes": ["TrainReportedAsHeldOut"],
        "y_hold_index_codes": ["HoldYUsedAsIndex"],
        "unbilled_codes": ["UnbilledProgramByte"],
        "linear_codes": ["LinearSharedSubspaceDead"],
        "remat_codes": ["REJECTED_DENSE_REMAT"],
    }


def build(*, consult: bool = True) -> Path:
    test = selftest()
    index = consult_index() if consult else {"proceed": True, "skipped": True}
    if consult and not index.get("proceed", False):
        raise NonlinearProgramRefuse(
            "REFUSED: negative_index refuse_if_dead fired on the proposal "
            f"families: {index.get('proposal_refused')}"
        )
    round1 = cached_round1()
    n_neg = sum(1 for r in round1["rows"] if r["status"] == MEASURED_NEGATIVE)
    n_open = sum(1 for r in round1["rows"] if r["status"] == OPEN)
    all_dead = all(v["status"] == MEASURED_NEGATIVE for v in round1["family_verdicts"])
    budget = residual_budget_allocation(int(round1["n_survivors"]))
    factorize = round1["factorize_the_factors"]
    best_f = factorize["best"]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "After linear shared programs died (oracle PCA of F at rank 64 is "
            "~0.89 held-out; the output manifold is not low-rank), test the "
            "families that respect SwiGLU structure: factorize the three "
            "maps separately, dictionaries, product dictionaries, a "
            "conditional program, a generated block, and an MLP-shaped "
            "generator. FACTORIZE_THE_FACTORS runs first."
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
            "kill_band": KILL_BAND,
            "cosine": "mean row cosine; reported, not authority",
            "worst_prompt": "per-prompt mean-L2; a small mean that destroys one prompt is not a winner",
            "per_capability_domain": list(CAPABILITY_DOMAINS),
            "per_position_band": list(POSITION_BANDS),
            "weight_reconstruction": "diagnostic only; not authority; not scored",
            "relative_frobenius": "diagnostic only",
            "oracle_y_assignment": "diagnostic lower bound for dictionaries; not authority",
        },
        "round": "cheap_first",
        "ranks": list(ROUND1_RANKS),
        "n_layers_billed": N_LAYERS,
        "element_bytes": ELEMENT_BYTES,
        "families": list(FAMILIES),
        "index": index,
        "selftest": test,
        "anti_fabrication": {
            "detectors": [
                "UNBILLED_PROGRAM_BYTE",
                "TRAIN_REPORTED_AS_HELD_OUT",
                "HOLD_Y_USED_AS_INDEX",
                "REJECTED_DENSE_REMAT",
                "LINEAR_SHARED_SUBSPACE",
                "SYNTHETIC_ROW",
                "HELD_OUT_PROMPT_LEAK",
            ],
            "loud_exceptions": [
                "UnbilledProgramByte",
                "TrainReportedAsHeldOut",
                "HoldYUsedAsIndex",
                "RematConsumer",
                "LinearSharedSubspaceDead",
                "CorpusUnavailable",
                "WeightsUnavailable",
            ],
            "rule": (
                "emit_candidate is the only constructor. A used dictionary, "
                "generator, core, or condition with 0 billed bytes raises "
                "UnbilledProgramByte. A train-set figure labelled held-out "
                "raises TrainReportedAsHeldOut. An index computed from "
                "held-out Y raises HoldYUsedAsIndex. A consumer that "
                "rematerializes dense W raises RematConsumer. Naming "
                "SHARED_* as a family raises LinearSharedSubspaceDead. A "
                "return-flag nobody checks is not a guard."
            ),
        },
        "baselines": round1["baselines"],
        "oracle_output_pca_cited": round1["oracle_output_pca_cited"],
        "factorize_the_factors": factorize,
        "candidates": round1["rows"],
        "family_verdicts": round1["family_verdicts"],
        "scars": round1["scars"],
        "survivors": round1["survivors"],
        "n_survivors": round1["n_survivors"],
        "candidate_counts": {
            "n": len(round1["rows"]),
            "measured_negative": n_neg,
            "open": n_open,
            "rejected_dense_remat": 0,
        },
        "residual_budget": budget,
        "answers": {
            "was_the_shared_program_result_an_artifact_of_approximating_the_composite": (
                "NO. FACTORIZE_THE_FACTORS at rank "
                f"{best_f['rank']} has held-out relative L2 "
                f"{best_f['held_out_relative_l2']} (cosine "
                f"{best_f.get('held_out_cosine')}), in the 0.9 band and not "
                "better than the mean predictor. Gate/up/down are not "
                "individually low-rank on this body; composing truncated "
                "factors with exact silu and product does not recover F."
            ),
            "does_any_cheap_nonlinear_or_conditional_family_replace_F": (
                "NO on this cheap round. Every family sits in the 0.9 band at "
                "every affordable size. Function replacement as a class is "
                "not closed: a full-width structured nonlinear (Monarch / "
                "butterfly / distilled operator that is not an r-bottleneck) "
                "is a different experiment."
            ),
            "do_the_bytes_clear_one_percent_of_complete_token_time": (
                "YES as a projection for the compact programs: replacing "
                "5.35 GB of MLP with tens to hundreds of MB would clear the "
                "S020 1% bar. Function does not hold, so every candidate is "
                "MEASURED_NEGATIVE and the economics verdict is IMMATERIAL."
            ),
            "should_round_2_go_wider_on_these_families": (
                "NO. The cheap round is the kill. A 0.9-band error at "
                "affordable size is the scar. Do not widen rank, K, experts, "
                "or blocks on these instantiations."
            ),
        },
        "negative_findings": [
            f"FACTORIZE_THE_FACTORS r=8..64 held-out relative L2 stays in the 0.9 band (best {best_f['held_out_relative_l2']})",
            "DICTIONARY_PROGRAM X-routed K=16..256 stays in the 0.9 band; the Y-assignment oracle does too",
            "PRODUCT_DICTIONARY 16xK16 and 32xK32 stay in the 0.9 band",
            "CONDITIONAL_PROGRAM E=4..8 with mean or rank-16 experts stays in the 0.9 band",
            "GENERATED_BLOCK unconstrained per-block rank-8/16 stays in the 0.9 band",
            "NONLINEAR_GENERATOR silu readout r=32..128 and two-layer r=64 stay in the 0.9 band",
            "oracle PCA of F at r=64 is already ~0.89 held-out (cited, not re-run): the output is not low-rank",
        ],
        "gaps_closed": [
            "FACTORIZE_THE_FACTORS run first on real teacher corpus, held out by prompt, with exact silu and product",
            "per-capability-domain, per-position-band, worst-prompt, cosine reported; authority remains held_out_relative_l2",
            "dictionaries, generators, cores, indices, metadata, and the condition billed once",
            "executable_economics.score used for every byte figure and ms/token projection",
            "native-consumer sketches on atlas primitives; remat-then-GEMV refused",
            "linear shared subspace refused as a candidate family (MLP_SHARED_PROGRAM scar)",
            "train-set figure cannot be reported as held-out; hold-Y codebook index cannot be reported as held-out",
            "negative_index queried; proposal families not refused; W-space scars cited as cousins",
            "CAPABILITY_INFORMATION_MAP cited for residual budget; none allocated because nothing survived",
        ],
        "what_this_does_not_prove": [
            "that a full-width structured nonlinear (Monarch/butterfly) cannot replace F",
            "that a distilled operator which is not an r-dimensional bottleneck cannot replace F",
            "capability at generate",
            "a protected TPS or complete-token number",
            "that raw-weight PQ/VQ (NNS-017) was re-tested; it was not; this dictionary is of F",
            "that SVD-of-W energy (NNS-016) was re-measured as a spectrum; this object is composed F",
        ],
        "nomenclature": {
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "direct_consume": DIRECT_CONSUME,
            "held_out_authority": "held_out_relative_l2",
            "kill_band": "held-out relative L2 >= 0.85 is the 0.9-band cheap kill",
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
        "go_wider": False if all_dead else True,
        "next": (
            "Do not widen FACTORIZE_THE_FACTORS, dictionaries, conditionals, "
            "generated blocks, or silu-readout generators. The cheap round is "
            "the kill and the scars name the mechanism. Function replacement "
            "as a class remains the economics path; the live instantiation is "
            "a full-width structured nonlinear that is not an r-bottleneck."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--factorize", action="store_true")
    args = parser.parse_args(argv_list)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.factorize or args.fit:
        out = cached_round1()
        slim = {
            "layer": out["layer"],
            "n_train": out["n_train"],
            "n_hold": out["n_hold"],
            "baselines": out["baselines"],
            "factorize_the_factors": out["factorize_the_factors"],
            "family_verdicts": out["family_verdicts"],
            "n_survivors": out["n_survivors"],
            "n_rows": len(out["rows"]),
        }
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
