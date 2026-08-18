#!/usr/bin/env python3
"""Qwen3.8-27B activation capture v2 and the adequacy gate it exists to enforce.

N = 23216 tokens from >= 64 sequences, prompt-level 25% hold-out, sites at
every layer, parent BF16 plus an additional Q4 twin, f16 site-split storage.

A fit with n_fit < fit_dim is REFUSED. The gate does not emit a score, a
1.0, a 0.0, or any other foldable default. That is NS-014, named, on this
model: 256/6144 = 0.0417 is worse than the Q80 92/2048 = 0.0449 wreck.

--plan never loads a model and never touches the GPU. --run is for a later
GPU lane. This process must not stop, restart, or talk to the resident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

# ---------------------------------------------------------------------------
# Geometry (crates/hawking-core/src/model/qwen38_geometry.rs)
# ---------------------------------------------------------------------------

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
MIXER_IN = 6144
VOCAB = 248320
FULL_ATTENTION_INTERVAL = 4
GQA_HEADS = 24
GQA_HEAD_DIM = 256
BYTES_F16 = 2
BYTES_F32 = 4
BYTES_BF16_WEIGHT = 2

N_TOKENS = 23216
HOLD_FRAC = 0.25
MIN_EVAL_ROWS = 16
MIN_PROMPTS = 3
MIN_HOLDOUT_ROWS = 4
MIN_ROWS_PER_DIM = 1.0
MIN_SEQUENCES = 64
MIN_PROMPT_TEXTS = 32
MIN_SEQ_LEN = 32
MAX_SEQ_LEN = 2048

# v1 capture, MEASURED. Scale time from this, do not guess a new rate.
V1_N_TOKENS = 256
V1_WALL_S = 14.967979082999591
V1_SCHEMA = "hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1"
V1_STATUS = "CAPTURED_REAL_BF16_POST_NORM_HIDDEN"
V1_SHA256_SELF = "fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512"
V1_REL = Path("workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
V1_ABS = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)

# NS-014 wreck. A 6144-dim fit on 256 rows is worse than this.
NS014_ROWS = 92
NS014_DIM = 2048
NS014_RPD = NS014_ROWS / NS014_DIM  # 0.044921875

SOURCE_ELEMENTS = 26_895_998_464
BF16_WEIGHT_BYTES = SOURCE_ELEMENTS * BYTES_BF16_WEIGHT
# Harvested resident payload (genesis-resident.log) and catalog payload.
RESIDENT_WEIGHT_BYTES = 14_297_675_776
Q4_CATALOG_PAYLOAD_BYTES = 14_297_694_680
COHERENT_VEHICLE_BPW = 4.252735126866492
UNIFIED_MEMORY_BYTES = 96 * 10**9
INCOHERENT_BPW = (2.0856, 1.2910)

SCHEMA_V2 = "hawking.ascension.qwen38_activation_capture.v2"
PARENT_MODEL_ID = "Qwen3.8-27B"
PARENT_REPO = "PocketAiHub/Qwen3.8-27B-Abliterated-MLX"
BASE_REPO = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

CANDIDATE_UNDER_TEST_MARKERS = (
    "mixed-2p0",
    "mixed-sub15",
    "mixed_2p0",
    "mixed_sub15",
    "INCOHERENT",
    "candidate_under_test",
)

Procedure = Literal["eval_weight_only", "fit_from_X", "not_a_measurement"]
VehicleName = Literal["parent", "q4"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_parent_dir() -> Path:
    env = os.environ.get("QWEN38_PARENT_BF16")
    if env:
        return Path(env)
    for cand in (
        Path("/Users/scammermike/Downloads/hawking") / V1_REL.parent / "bf16",
        repo_root() / "workspace/campaign/records/runs/qwen38-27b/bf16",
    ):
        if cand.is_dir():
            return cand
    return Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")


def default_q4_dir() -> Path:
    env = os.environ.get("QWEN38_Q4_VEHICLE")
    if env:
        return Path(env)
    return default_parent_dir().parent / "uniform-q4-v1"


def default_v2_out() -> Path:
    return default_parent_dir().parent / "activation-capture-v2"


def resolve_v1_capture() -> Path | None:
    env = os.environ.get("QWEN38_CAPTURE_V1")
    cands = []
    if env:
        cands.append(Path(env))
    cands.extend(
        [
            V1_ABS,
            repo_root() / V1_REL,
            default_parent_dir().parent / "activation-capture-v1",
        ]
    )
    for cand in cands:
        if (cand / "capture-result.json").is_file():
            return cand
    return None


def is_gqa_layer(layer: int) -> bool:
    if layer < 0 or layer >= LAYERS:
        raise ValueError(f"layer {layer} outside 0..{LAYERS}")
    return (layer + 1) % FULL_ATTENTION_INTERVAL == 0


def mixer_kind(layer: int) -> str:
    return "gqa" if is_gqa_layer(layer) else "delta_net"


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    width: int
    n_layers: int
    consumed_by: str
    fit_dim: int
    required: bool = True

    @property
    def min_n_for_fit(self) -> int:
        """Smallest N such that floor(0.75*N) >= fit_dim."""
        return int(math.ceil(self.fit_dim / (1.0 - HOLD_FRAC)))


SITES: tuple[SiteSpec, ...] = (
    SiteSpec(
        "post_input_norm",
        HIDDEN,
        LAYERS,
        "q/k/v, in_proj_qkv/z/a/b (attention in-dim)",
        HIDDEN,
    ),
    SiteSpec(
        "post_attn_norm",
        HIDDEN,
        LAYERS,
        "gate_proj, up_proj (MLP in-dim after mixer residual)",
        HIDDEN,
    ),
    SiteSpec(
        "post_swiglu",
        INTERMEDIATE,
        LAYERS,
        "down_proj. silu(x@Wg.T)*(x@Wu.T), stored, not reconstructed",
        INTERMEDIATE,
    ),
    SiteSpec(
        "mixer_x",
        MIXER_IN,
        LAYERS,
        "out_proj / o_proj. True recurrent mix or GQA gated softmax mix",
        MIXER_IN,
    ),
    SiteSpec(
        "final_norm",
        HIDDEN,
        1,
        "lm_head. Confirmed model.norm, not L63 post-norm hidden",
        HIDDEN,
    ),
)

SITE_BY_ID = {s.site_id: s for s in SITES}

# Per-token elements if every required site is stored at the same N.
ELEMENTS_PER_TOKEN_ALL_SITES = LAYERS * (HIDDEN + HIDDEN + INTERMEDIATE + MIXER_IN) + HIDDEN
# 2_167_808, matches g1-doctor-recovery.md §5.5


def site_store_n(site: SiteSpec, *, store_mode: str) -> int:
    if store_mode == "full":
        return N_TOKENS
    if store_mode == "census":
        return 2048
    if store_mode != "site-split-n":
        raise ValueError(f"unknown store_mode {store_mode}")
    if site.site_id == "post_swiglu":
        return N_TOKENS
    if site.site_id == "mixer_x":
        return 8192
    return 6827


def site_bytes(site: SiteSpec, n: int, *, dtype_bytes: int = BYTES_F16) -> int:
    return int(n) * int(site.n_layers) * int(site.width) * int(dtype_bytes)


def rows_per_dim(n_rows: int, fit_dim: int) -> float:
    if int(fit_dim) <= 0:
        return float("inf") if int(n_rows) > 0 else 0.0
    return float(n_rows) / float(fit_dim)


# ---------------------------------------------------------------------------
# Adequacy gate — the point of this lane
# ---------------------------------------------------------------------------


class AdequacyRefused(ValueError):
    """A fit or score was refused. Never catch this to invent a number."""


@dataclass(frozen=True)
class AdequacyVerdict:
    status: str
    determination: str
    procedure: str
    n_rows: int
    n_fit: int
    n_hold: int
    n_prompts: int
    fit_dim: int
    rows_per_dim: float
    eval_thin: bool
    emit_score: bool
    reason: str
    score: None = None
    holdout_by_prompt: bool = True
    x_source: str | None = None
    site_id: str | None = None
    ns014_rpd: float = NS014_RPD
    worse_than_ns014: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "determination": self.determination,
            "procedure": self.procedure,
            "n_rows": self.n_rows,
            "n_fit": self.n_fit,
            "n_hold": self.n_hold,
            "n_prompts": self.n_prompts,
            "fit_dim": self.fit_dim,
            "rows_per_dim": self.rows_per_dim,
            "eval_thin": self.eval_thin,
            "emit_score": self.emit_score,
            "score": None,
            "reason": self.reason,
            "holdout_by_prompt": self.holdout_by_prompt,
            "x_source": self.x_source,
            "site_id": self.site_id,
            "ns014_rpd": self.ns014_rpd,
            "worse_than_ns014": self.worse_than_ns014,
        }
        if self.status == "REFUSED":
            payload["verdict"] = "REFUSED"
            payload["emit_score"] = False
            payload["score"] = None
        return payload

    def require(self) -> AdequacyVerdict:
        if self.status == "REFUSED":
            raise AdequacyRefused(self.reason)
        return self


def _refuse(**kwargs: Any) -> AdequacyVerdict:
    rpd = float(kwargs.get("rows_per_dim") or 0.0)
    kwargs.setdefault("status", "REFUSED")
    kwargs.setdefault("determination", "UNDERDETERMINED")
    kwargs.setdefault("emit_score", False)
    kwargs.setdefault("score", None)
    kwargs.setdefault("eval_thin", bool(math.isfinite(rpd) and rpd < MIN_ROWS_PER_DIM))
    kwargs.setdefault("worse_than_ns014", bool(math.isfinite(rpd) and 0.0 < rpd < NS014_RPD))
    kwargs.setdefault("holdout_by_prompt", True)
    return AdequacyVerdict(**kwargs)


def _accept(**kwargs: Any) -> AdequacyVerdict:
    kwargs.setdefault("status", "ACCEPTED")
    kwargs.setdefault("determination", "DETERMINED")
    kwargs.setdefault("emit_score", True)
    kwargs.setdefault("score", None)
    kwargs.setdefault("holdout_by_prompt", True)
    kwargs.setdefault("worse_than_ns014", False)
    return AdequacyVerdict(**kwargs)


def prompt_holdout(
    prompt_n_tokens: Sequence[int],
    *,
    hold_frac: float = HOLD_FRAC,
    min_holdout: int = MIN_HOLDOUT_ROWS,
    designated_hold: Sequence[bool] | None = None,
) -> tuple[list[int], list[int], int, int]:
    """Split by prompt id. Never shuffles rows. Returns fit_ids, hold_ids, n_fit, n_hold."""
    n_prompts = len(prompt_n_tokens)
    sizes = [int(x) for x in prompt_n_tokens]
    n_rows = int(sum(sizes))
    need_hold = max(int(min_holdout), int(math.ceil(float(hold_frac) * n_rows)))
    if designated_hold is not None:
        if len(designated_hold) != n_prompts:
            raise ValueError("designated_hold length must match prompts")
        hold_ids = [i for i, flag in enumerate(designated_hold) if flag]
        fit_ids = [i for i, flag in enumerate(designated_hold) if not flag]
    else:
        # Walk from the end so a stable mix of later prompts is held.
        hold_ids = []
        got = 0
        for i in range(n_prompts - 1, -1, -1):
            if got >= need_hold:
                break
            hold_ids.append(i)
            got += sizes[i]
        hold_ids.reverse()
        hold_set = set(hold_ids)
        fit_ids = [i for i in range(n_prompts) if i not in hold_set]
    n_hold = int(sum(sizes[i] for i in hold_ids))
    n_fit = int(sum(sizes[i] for i in fit_ids))
    return fit_ids, hold_ids, n_fit, n_hold


def classify_x_source(meta: Mapping[str, Any], *, path: str = "") -> tuple[str, list[str]]:
    notes: list[str] = []
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    model_dir = str(source.get("model_dir", "") or meta.get("model_dir", ""))
    vehicle_bpw = meta.get("vehicle_bpw")
    status = str(meta.get("status", ""))
    blob = " ".join([path, model_dir, status, str(meta.get("vehicle", ""))]).lower()
    if vehicle_bpw is not None:
        try:
            if abs(float(vehicle_bpw) - COHERENT_VEHICLE_BPW) < 1e-3:
                return "COHERENT_Q4_VEHICLE", notes
        except (TypeError, ValueError):
            pass
    if "uniform-q4" in blob or "COHERENT_Q4" in status:
        return "COHERENT_Q4_VEHICLE", notes
    if "bf16" in blob or status.startswith("CAPTURED_REAL_BF16"):
        notes.append(
            "X is the BF16 parent, not the 4.2527 vehicle. Legal fit source. "
            "Q4-vehicle capture is additional."
        )
        return "PARENT_BF16_REAL", notes
    return "UNKNOWN", notes


def looks_like_candidate(meta: Mapping[str, Any], *, path: str = "") -> str | None:
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    blob = " ".join(
        [
            path,
            str(source.get("model_dir", "")),
            str(meta.get("x_source", "")),
            str(meta.get("vehicle", "")),
            str(meta.get("status", "")),
            str(meta.get("artifact", "")),
        ]
    ).lower()
    for marker in CANDIDATE_UNDER_TEST_MARKERS:
        if marker.lower() in blob:
            return marker
    bpw = meta.get("vehicle_bpw", meta.get("complete_physical_bpw"))
    if bpw is not None:
        try:
            value = float(bpw)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            for bad in INCOHERENT_BPW:
                if abs(value - float(bad)) < 0.02:
                    return f"vehicle_bpw={value}"
    return None


def is_synthetic_gaussian(x: Any, *, kurtosis_tol: float = 1.25) -> bool:
    """True when X is statistically a standard-ish Gaussian (the known lie)."""
    try:
        import numpy as np
    except ImportError:
        return False
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 64:
        return False
    step = max(1, arr.shape[-1] // 64) if arr.ndim == 2 else 1
    sample = arr[:, ::step] if arr.ndim == 2 else arr
    flat = sample.reshape(-1)
    std = float(flat.std())
    if std <= 1e-12:
        return False
    z = (flat - float(flat.mean())) / std
    kurtosis = float(np.mean(z**4))
    return abs(kurtosis - 3.0) < kurtosis_tol and 0.5 < std < 2.0


def adequacy_gate(
    *,
    n_rows: int,
    fit_dim: int,
    n_prompts: int,
    procedure: Procedure,
    n_fit: int | None = None,
    n_hold: int | None = None,
    holdout_by_prompt: bool = True,
    x_source: str | None = None,
    not_synthetic: bool = True,
    candidate_marker: str | None = None,
    rank_claimed: int | None = None,
    rank_clamped_to_n_fit: bool = False,
    interpolated: bool = False,
    granularity: str = "tensor",
    site_id: str | None = None,
    x_sample: Any = None,
) -> AdequacyVerdict:
    """Refuse a number when the unit is underdetermined.

    Law: if procedure is fit_from_X, n_fit >= fit_dim or the number is REFUSED.
    A refused verdict has emit_score=False and score=None. It never returns
    1.0, 0.0, or any other default a downstream fold could treat as a pass.
    """
    n_rows = int(n_rows)
    fit_dim = int(fit_dim)
    n_prompts = int(n_prompts)
    if n_fit is None:
        n_fit = n_rows - int(round(HOLD_FRAC * n_rows)) if n_hold is None else n_rows - int(n_hold)
    n_fit = int(n_fit)
    if n_hold is None:
        n_hold = n_rows - n_fit
    n_hold = int(n_hold)
    rpd = rows_per_dim(n_fit, fit_dim)
    eval_thin = bool(math.isfinite(rpd) and rpd < MIN_ROWS_PER_DIM)
    base = dict(
        procedure=procedure,
        n_rows=n_rows,
        n_fit=n_fit,
        n_hold=n_hold,
        n_prompts=n_prompts,
        fit_dim=fit_dim,
        rows_per_dim=rpd,
        eval_thin=eval_thin,
        x_source=x_source,
        site_id=site_id,
        holdout_by_prompt=holdout_by_prompt,
    )

    if interpolated:
        return _refuse(
            **base,
            reason="REFUSED: interpolated layer is UNMEASURED, not a determination.",
        )
    if candidate_marker is not None:
        return _refuse(
            **{**base, "x_source": "CANDIDATE_UNDER_TEST",
               "reason": (
                   f"REFUSED: capture is a candidate under test ({candidate_marker}). "
                   "X must come from PARENT_BF16_REAL or COHERENT_Q4_VEHICLE."
               )},
        )
    if x_source not in {None, "PARENT_BF16_REAL", "COHERENT_Q4_VEHICLE"}:
        return _refuse(
            **base,
            reason=f"REFUSED: x_source={x_source!r} is not PARENT_BF16_REAL or COHERENT_Q4_VEHICLE.",
        )
    if not not_synthetic:
        return _refuse(
            **base,
            reason="REFUSED: synthetic or unlabeled capture. Gaussian X wrecked earlier sub-bit work.",
        )
    if x_sample is not None and is_synthetic_gaussian(x_sample):
        return _refuse(
            **{**base, "x_source": "SYNTHETIC_GAUSSIAN",
               "reason": "REFUSED: activations match a Gaussian proxy (kurtosis near 3). Flags do not override."},
        )
    if n_prompts < MIN_PROMPTS:
        return _refuse(
            **base,
            reason=f"REFUSED: {n_prompts} prompts < MIN_PROMPTS={MIN_PROMPTS}.",
        )
    if n_rows < MIN_EVAL_ROWS:
        return _refuse(
            **base,
            reason=f"REFUSED: {n_rows} rows < MIN_EVAL_ROWS={MIN_EVAL_ROWS}.",
        )
    need_hold = max(MIN_HOLDOUT_ROWS, int(math.ceil(HOLD_FRAC * n_rows)))
    if not holdout_by_prompt:
        return _refuse(
            **base,
            reason="REFUSED: holdout is not by prompt. Row shuffle leaks tokens from the same prompt.",
        )
    if n_hold < need_hold:
        return _refuse(
            **base,
            reason=(
                f"REFUSED: n_hold={n_hold} < max({MIN_HOLDOUT_ROWS}, ceil(0.25*{n_rows}))={need_hold}."
            ),
        )
    if n_fit + n_hold != n_rows:
        return _refuse(
            **base,
            reason=f"REFUSED: n_fit ({n_fit}) + n_hold ({n_hold}) != n_rows ({n_rows}).",
        )
    if rank_clamped_to_n_fit:
        return _refuse(
            **base,
            reason=(
                "REFUSED: rank = min(budget, n_fit) is a silent starve (NS-014). "
                "Re-score only when n_fit >= the claimed rank, with rank not clamped."
            ),
        )
    if rank_claimed is not None and n_fit < int(rank_claimed):
        return _refuse(
            **{**base, "fit_dim": int(rank_claimed),
               "rows_per_dim": rows_per_dim(n_fit, int(rank_claimed)),
               "reason": (
                   f"REFUSED: n_fit={n_fit} < claimed rank={int(rank_claimed)}. "
                   "Underdetermined rank-r fit (NS-014)."
               )},
        )
    if procedure == "fit_from_X" and n_fit < fit_dim:
        return _refuse(
            **base,
            reason=(
                f"REFUSED: fit_from_X n_fit={n_fit} < fit_dim={fit_dim} "
                f"(rows_per_dim={rpd:.6f}). "
                f"NS-014 wreck was {NS014_ROWS}/{NS014_DIM}={NS014_RPD:.6f}. "
                "This number is not emitted."
            ),
        )
    if granularity in {"attention_head", "channel", "outlier"} and n_fit < fit_dim:
        return _refuse(
            **base,
            reason=(
                f"REFUSED: {granularity} n_fit={n_fit} < sliced fit_dim={fit_dim}."
            ),
        )
    if procedure == "eval_weight_only":
        reason = None
        if eval_thin:
            reason = (
                f"eval_weight_only with rows_per_dim={rpd:.6f} < 1; flagged eval_thin. "
                "May emit an eval. Must not be the sole input to a bit assignment "
                "on an unswept layer, and is not a fit."
            )
        return _accept(**{**base, "reason": reason or "eval_weight_only determined as an eval"})
    if procedure != "fit_from_X":
        return _refuse(
            **base,
            reason=f"REFUSED: procedure={procedure!r} is not a measurement.",
        )
    return _accept(
        **{**base, "eval_thin": False,
           "reason": f"n_fit={n_fit} >= fit_dim={fit_dim}; prompt-level holdout n_hold={n_hold}."},
    )


def adequacy_from_capture(
    meta: Mapping[str, Any],
    *,
    fit_dim: int,
    procedure: Procedure = "fit_from_X",
    site_id: str | None = None,
    path: str = "",
    designated_hold: Sequence[bool] | None = None,
    rank_claimed: int | None = None,
    rank_clamped_to_n_fit: bool = False,
    interpolated: bool = False,
    granularity: str = "tensor",
    x_sample: Any = None,
) -> AdequacyVerdict:
    """Run the gate on a capture-result.json-shaped mapping (v1 or v2)."""
    prompts = list(meta.get("prompts") or [])
    sizes = [int(p.get("n_tokens") or len(p.get("ids") or ())) for p in prompts]
    if designated_hold is None and prompts and all("split" in p for p in prompts):
        designated_hold = [str(p.get("split")) == "hold" for p in prompts]
    if site_id and isinstance(meta.get("sites"), dict):
        site_meta = meta["sites"].get(site_id) or {}
        if site_meta.get("n_tokens") is not None:
            n_rows = int(site_meta["n_tokens"])
        else:
            n_rows = int(meta.get("n_tokens") or sum(sizes) or 0)
        if site_meta.get("n_fit") is not None and site_meta.get("n_hold") is not None:
            n_fit = int(site_meta["n_fit"])
            n_hold = int(site_meta["n_hold"])
            holdout_by_prompt = bool(site_meta.get("holdout_by_prompt", True))
        else:
            _fit, _hold, n_fit, n_hold = prompt_holdout(sizes, designated_hold=designated_hold)
            holdout_by_prompt = True
    else:
        n_rows = int(meta.get("n_tokens") or sum(sizes) or 0)
        if sizes and sum(sizes) == n_rows:
            _fit, _hold, n_fit, n_hold = prompt_holdout(sizes, designated_hold=designated_hold)
            holdout_by_prompt = True
        else:
            n_fit = int(math.floor((1.0 - HOLD_FRAC) * n_rows))
            n_hold = n_rows - n_fit
            holdout_by_prompt = bool(meta.get("holdout_by_prompt", False))
    marker = looks_like_candidate(meta, path=path)
    x_source, _notes = classify_x_source(meta, path=path)
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    not_synthetic = bool(source.get("not_synthetic", meta.get("not_synthetic", False)))
    return adequacy_gate(
        n_rows=n_rows,
        fit_dim=int(fit_dim),
        n_prompts=len(prompts) if prompts else int(meta.get("n_prompts") or 0),
        procedure=procedure,
        n_fit=n_fit,
        n_hold=n_hold,
        holdout_by_prompt=holdout_by_prompt,
        x_source=x_source,
        not_synthetic=not_synthetic,
        candidate_marker=marker,
        rank_claimed=rank_claimed,
        rank_clamped_to_n_fit=rank_clamped_to_n_fit,
        interpolated=interpolated,
        granularity=granularity,
        site_id=site_id,
        x_sample=x_sample,
    )


def gated_score(
    meta: Mapping[str, Any],
    *,
    fit_dim: int,
    procedure: Procedure = "fit_from_X",
    score_fn: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Never call score_fn on a refused unit. Never invent a default number."""
    verdict = adequacy_from_capture(meta, fit_dim=fit_dim, procedure=procedure, **kwargs)
    out = {"adequacy": verdict.as_dict()}
    if verdict.status == "REFUSED":
        out["score"] = None
        out["verdict"] = "REFUSED"
        return out
    if score_fn is None:
        out["score"] = None
        out["verdict"] = "ACCEPTED"
        return out
    out["score"] = score_fn()
    out["verdict"] = "ACCEPTED"
    return out


# ---------------------------------------------------------------------------
# Prompt corpus — 23216 tokens, >= 64 sequences, designated 25% hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: int
    cls: str
    text: str
    target_n_tokens: int
    split: Literal["fit", "hold"]


@dataclass(frozen=True)
class PromptPlan:
    prompts: tuple[PromptSpec, ...]
    n_tokens: int
    n_sequences: int
    n_fit: int
    n_hold: int
    class_mass: dict[str, int]

    def sizes(self) -> list[int]:
        return [p.target_n_tokens for p in self.prompts]

    def designated_hold(self) -> list[bool]:
        return [p.split == "hold" for p in self.prompts]


_CHAT_PREFIX = (
    "<|im_start|>system\nYou are Qwen, a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
)
_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _wrap(body: str) -> str:
    return f"{_CHAT_PREFIX}{body}{_CHAT_SUFFIX}"


def _repeat(body: str, times: int) -> str:
    return " ".join(f"{body} ({k})" for k in range(times))


def _text_for(cls: str, idx: int) -> str:
    if cls == "prose":
        para = (
            f"Encyclopedia {idx}. Paris is the capital of France. The Seine cuts the city. "
            "The derivative of a square is twice the variable. Gravity pulls mass toward mass. "
            "A hash map stores key-value pairs at an average constant lookup. "
            "The printing press, the steam engine, and the transistor each reset a century."
        )
        return _wrap(_repeat(para, 12))
    if cls == "code":
        para = (
            f"def fibonacci_{idx}(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n"
            "def reverse_string(s):\n    return s[::-1]\n"
            "class HashMap:\n    def __init__(self):\n        self._b = [[] for _ in range(64)]\n"
        )
        return _wrap(_repeat(para, 8))
    if cls == "math":
        para = (
            f"STEM {idx}. d/dx x^2 = 2x. The Laplacian of 1/r vanishes off the origin. "
            "Cauchy-Schwarz: |<u,v>|^2 <= ||u||^2 ||v||^2. A group of order p is cyclic. "
            "The heat equation is u_t = k u_xx. Bayes: P(H|E) = P(E|H)P(H)/P(E)."
        )
        return _wrap(_repeat(para, 10))
    if cls == "instruction":
        para = (
            f"User turn {idx}: write a short greeting to a colleague, then explain gravity "
            "in one sentence, then list three primary colors. "
            "Assistant: hello, gravity is curvature, red green blue. "
            "User: now translate the greeting and add a second turn about the derivative."
        )
        return _wrap(_repeat(para, 8))
    if cls == "long":
        para = (
            f"Long-context recurrence probe {idx}. " + ("The field equations couple geometry to stress-energy. " * 20)
        )
        return _wrap(_repeat(para, 20))
    if cls == "multilingual":
        para = (
            f"Mix {idx}. 首都は東京です. La capital de Francia es París. "
            "Столица Франции — Париж. عاصمة فرنسا باريس. 法国的首都是巴黎. "
            "Die Ableitung von x Quadrat ist 2x. 해시맵은 평균 상수 시간에 조회한다."
        )
        return _wrap(_repeat(para, 10))
    para = (
        f"ADV {idx} ::: 000 111 999 ... !!! ??? ;;; --- +++ *** "
        "3.141592653589793 2.718281828459045 1.41421356237 "
        "(((( )))) ,,,, .... token-knife-edge"
    )
    return _wrap(_repeat(para, 6))


def build_prompt_plan() -> PromptPlan:
    """Exact token budgets. Sum 23216. Hold 5804. Fit 17412. >= 64 sequences.

    Tokenization happens on --run and trims each sequence to target_n_tokens.
    Plan mode does not load a tokenizer.
    """
    # (cls, n_seq, sizes) — sizes sum to the class mass in g1-doctor-recovery §5.3.
    class_sizes: list[tuple[str, list[int]]] = [
        ("prose", [362] * 15 + [374]),  # 15*362+374 = 5804
        ("code", [386] * 11 + [397]),  # 11*386+397 = 4643
        ("math", [348] * 9 + [350]),  # 9*348+350 = 3482
        ("instruction", [348] * 9 + [350]),
        ("long", [774, 774, 774]),  # 2322, each in [512, 2048]
        ("multilingual", [232] * 9 + [234]),  # 9*232+234 = 2322
        ("adversarial", [145] * 7 + [146]),  # 7*145+146 = 1161, each >= 32
    ]
    specs: list[PromptSpec] = []
    pid = 0
    for cls, sizes in class_sizes:
        for i, n in enumerate(sizes):
            if n < MIN_SEQ_LEN or n > MAX_SEQ_LEN:
                raise RuntimeError(f"{cls}[{i}] target {n} outside [{MIN_SEQ_LEN},{MAX_SEQ_LEN}]")
            specs.append(
                PromptSpec(
                    prompt_id=pid,
                    cls=cls,
                    text=_text_for(cls, i),
                    target_n_tokens=int(n),
                    split="fit",
                )
            )
            pid += 1

    # Designated mixed hold set, then +278 moved onto hold prose[0] from fit prose[1]
    # so hold sums to exactly 5804 and fit to 17412.
    hold_pick = {
        # 4 prose + 3 code + 2 math + 2 instruction + 1 long + 2 multi + 2 adv
        0,
        1,
        2,
        3,  # prose
        16,
        17,
        18,  # code (prose used 0..15)
        28,
        29,  # math starts at 16+12=28
        38,
        39,  # instruction starts at 28+10=38
        48,  # long starts at 38+10=48
        51,
        52,  # multilingual starts at 48+3=51
        61,
        62,  # adversarial starts at 51+10=61
    }
    adjusted: list[PromptSpec] = []
    for p in specs:
        split: Literal["fit", "hold"] = "hold" if p.prompt_id in hold_pick else "fit"
        n = p.target_n_tokens
        if p.prompt_id == 0:
            n = 362 + 278  # 640
            split = "hold"
        elif p.prompt_id == 4:
            n = 362 - 278  # 84
            split = "fit"
        adjusted.append(
            PromptSpec(
                prompt_id=p.prompt_id,
                cls=p.cls,
                text=p.text,
                target_n_tokens=n,
                split=split,
            )
        )

    n_tokens = sum(p.target_n_tokens for p in adjusted)
    n_fit = sum(p.target_n_tokens for p in adjusted if p.split == "fit")
    n_hold = sum(p.target_n_tokens for p in adjusted if p.split == "hold")
    if n_tokens != N_TOKENS or n_fit != 17412 or n_hold != 5804:
        raise RuntimeError(f"prompt plan drifted: n={n_tokens} fit={n_fit} hold={n_hold}")
    if len(adjusted) < MIN_SEQUENCES:
        raise RuntimeError(f"{len(adjusted)} sequences < {MIN_SEQUENCES}")
    class_mass = {cls: 0 for cls, _ in class_sizes}
    for p in adjusted:
        class_mass[p.cls] += p.target_n_tokens
    expected = {
        "prose": 5804,
        "code": 4643,
        "math": 3482,
        "instruction": 3482,
        "long": 2322,
        "multilingual": 2322,
        "adversarial": 1161,
    }
    if class_mass != expected:
        raise RuntimeError(f"class mass drifted: {class_mass}")
    return PromptPlan(
        prompts=tuple(adjusted),
        n_tokens=n_tokens,
        n_sequences=len(adjusted),
        n_fit=n_fit,
        n_hold=n_hold,
        class_mass=class_mass,
    )


# ---------------------------------------------------------------------------
# Plan arithmetic (disk, time, memory) — no GPU
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SitePlan:
    site_id: str
    width: int
    n_layers: int
    store_n: int
    fit_dim: int
    min_n_for_fit: int
    bytes_f16: int
    bytes_f32: int
    n_fit_if_hold_25: int
    adequacy: str
    consumed_by: str


@dataclass(frozen=True)
class CapturePlan:
    n_tokens: int
    n_sequences: int
    n_prompts: int
    n_fit: int
    n_hold: int
    hold_frac: float
    sites: tuple[SitePlan, ...]
    store_mode: str
    parent_bytes_f16: int
    parent_bytes_f32: int
    q4_store_mode: str
    q4_bytes_f16: int
    total_bytes_f16_parent_only: int
    total_bytes_f16_both: int
    v1_wall_s: float
    v1_n_tokens: int
    s_per_token_measured: float
    wall_s_linear: float
    wall_s_gqa_adjusted: float
    wall_s_lower: float
    wall_s_upper: float
    wall_s_both_vehicles_linear: float
    bf16_weight_bytes: int
    resident_weight_bytes: int
    peak_process_bytes: int
    peak_if_resident_stays_bytes: int
    one_layer_microbatch_bytes: int
    unified_memory_bytes: int
    parent_dir: str
    q4_dir: str
    out_dir: str
    preflight: dict[str, Any]


def _gqa_adjusted_seconds(plan: PromptPlan, s_per_token: float) -> float:
    """ESTIMATED. GQA is quadratic; 16/64 layers. Scale from the ~60-token v1 mix."""
    v1_len = 60.0
    frac_gqa = 16.0 / 64.0
    total = 0.0
    for p in plan.prompts:
        length = max(float(p.target_n_tokens), 1.0)
        ratio = 0.75 + 0.25 * (length / v1_len)
        total += length * s_per_token * ratio
    # frac_gqa is already baked into 0.75/0.25. Keep the name honest.
    _ = frac_gqa
    return float(total)


def build_capture_plan(
    *,
    store_mode: str = "site-split-n",
    q4_store_mode: str = "census",
    parent_dir: Path | None = None,
    q4_dir: Path | None = None,
    out_dir: Path | None = None,
    include_q4: bool = True,
) -> CapturePlan:
    prompts = build_prompt_plan()
    parent_dir = Path(parent_dir or default_parent_dir())
    q4_dir = Path(q4_dir or default_q4_dir())
    out_dir = Path(out_dir or default_v2_out())
    site_plans: list[SitePlan] = []
    parent_bytes = 0
    parent_bytes_f32 = 0
    for site in SITES:
        n = site_store_n(site, store_mode=store_mode)
        n_fit_site = int(math.floor((1.0 - HOLD_FRAC) * n))
        ok = n_fit_site >= site.fit_dim
        sp = SitePlan(
            site_id=site.site_id,
            width=site.width,
            n_layers=site.n_layers,
            store_n=n,
            fit_dim=site.fit_dim,
            min_n_for_fit=site.min_n_for_fit,
            bytes_f16=site_bytes(site, n, dtype_bytes=BYTES_F16),
            bytes_f32=site_bytes(site, n, dtype_bytes=BYTES_F32),
            n_fit_if_hold_25=n_fit_site,
            adequacy="ACCEPTED" if ok else "REFUSED",
            consumed_by=site.consumed_by,
        )
        site_plans.append(sp)
        parent_bytes += sp.bytes_f16
        parent_bytes_f32 += sp.bytes_f32
    q4_bytes = 0
    if include_q4:
        for site in SITES:
            q4_bytes += site_bytes(site, site_store_n(site, store_mode=q4_store_mode))
    s_per = V1_WALL_S / float(V1_N_TOKENS)
    linear = s_per * float(N_TOKENS)
    gqa = _gqa_adjusted_seconds(prompts, s_per)
    # Write of ~67 GB at 1.5 GB/s is minutes, not the limiter.
    write_s = parent_bytes / 1.5e9
    wall_lower = linear + write_s
    wall_upper = max(gqa * 2.0, 3 * 3600.0)  # long GQA bound: a few hours
    one_layer = MAX_SEQ_LEN * INTERMEDIATE * BYTES_F16
    gqa_scores = MAX_SEQ_LEN * MAX_SEQ_LEN * GQA_HEADS * 4
    peak_process = BF16_WEIGHT_BYTES + one_layer + gqa_scores + 2 * 10**9
    preflight = preflight_capture(
        parent_dir=parent_dir,
        q4_dir=q4_dir,
        out_dir=out_dir,
        need_bytes=parent_bytes + q4_bytes + 10 * 10**9,
        mutate=False,
    )
    return CapturePlan(
        n_tokens=N_TOKENS,
        n_sequences=prompts.n_sequences,
        n_prompts=prompts.n_sequences,
        n_fit=prompts.n_fit,
        n_hold=prompts.n_hold,
        hold_frac=HOLD_FRAC,
        sites=tuple(site_plans),
        store_mode=store_mode,
        parent_bytes_f16=parent_bytes,
        parent_bytes_f32=parent_bytes_f32,
        q4_store_mode=q4_store_mode,
        q4_bytes_f16=q4_bytes,
        total_bytes_f16_parent_only=parent_bytes,
        total_bytes_f16_both=parent_bytes + q4_bytes,
        v1_wall_s=V1_WALL_S,
        v1_n_tokens=V1_N_TOKENS,
        s_per_token_measured=s_per,
        wall_s_linear=linear,
        wall_s_gqa_adjusted=gqa,
        wall_s_lower=wall_lower,
        wall_s_upper=wall_upper,
        wall_s_both_vehicles_linear=2.0 * linear + write_s + q4_bytes / 1.5e9,
        bf16_weight_bytes=BF16_WEIGHT_BYTES,
        resident_weight_bytes=RESIDENT_WEIGHT_BYTES,
        peak_process_bytes=peak_process,
        peak_if_resident_stays_bytes=peak_process + RESIDENT_WEIGHT_BYTES,
        one_layer_microbatch_bytes=one_layer,
        unified_memory_bytes=UNIFIED_MEMORY_BYTES,
        parent_dir=str(parent_dir),
        q4_dir=str(q4_dir),
        out_dir=str(out_dir),
        preflight=preflight,
    )


def _gb(n: int | float) -> str:
    return f"{float(n) / 1e9:.6f}"


def format_plan(plan: CapturePlan) -> str:
    lines: list[str] = []
    a = lines.append
    a("QWEN38 CAPTURE V2 PLAN")
    a("======================")
    a("gpu: NOT TOUCHED")
    a("model_load: NO")
    a("forward: NO")
    a("")
    a(f"n_tokens: {plan.n_tokens}")
    a(f"n_sequences: {plan.n_sequences}")
    a(f"n_prompts: {plan.n_prompts}")
    a(f"hold_frac: {plan.hold_frac}")
    a(f"n_fit: {plan.n_fit}")
    a(f"n_hold: {plan.n_hold}")
    a(f"n_fit >= 17408: {'YES' if plan.n_fit >= INTERMEDIATE else 'NO'}")
    a(f"holdout: prompt-level (not row shuffle)")
    a("")
    a("sites:")
    for s in plan.sites:
        a(
            f"  {s.site_id:16s} width={s.width:<5d} layers={s.n_layers:<2d} "
            f"store_n={s.store_n:<5d} fit_dim={s.fit_dim:<5d} "
            f"n_fit={s.n_fit_if_hold_25:<5d} bytes_f16={s.bytes_f16} "
            f"gb_f16={_gb(s.bytes_f16)} adequacy={s.adequacy}"
        )
        a(f"    consumed_by: {s.consumed_by}")
    a(f"store_mode: {plan.store_mode}")
    a(f"parent_bytes_f16: {plan.parent_bytes_f16}")
    a(f"parent_gb_f16: {_gb(plan.parent_bytes_f16)}")
    a(f"parent_bytes_f32: {plan.parent_bytes_f32}")
    a(f"parent_gb_f32: {_gb(plan.parent_bytes_f32)}")
    a(f"q4_store_mode: {plan.q4_store_mode}")
    a(f"q4_bytes_f16: {plan.q4_bytes_f16}")
    a(f"q4_gb_f16: {_gb(plan.q4_bytes_f16)}")
    a(f"total_bytes_f16_parent_only: {plan.total_bytes_f16_parent_only}")
    a(f"total_gb_f16_parent_only: {_gb(plan.total_bytes_f16_parent_only)}")
    a(f"total_bytes_f16_both: {plan.total_bytes_f16_both}")
    a(f"total_gb_f16_both: {_gb(plan.total_bytes_f16_both)}")
    a("")
    a("vehicles:")
    a(f"  parent: PARENT_BF16_REAL  path={plan.parent_dir}")
    a("          calibration source; must be preserved; refuse mixed-2p0 / mixed-sub15")
    a(f"  q4:     COHERENT_Q4_VEHICLE path={plan.q4_dir}")
    a("          additional, not a replacement; native reader only; expand-to-float REFUSED")
    a("")
    a("time (ESTIMATED from MEASURED 14.967979 s / 256 tokens):")
    a(f"  v1_wall_s: {plan.v1_wall_s}")
    a(f"  v1_n_tokens: {plan.v1_n_tokens}")
    a(f"  s_per_token_measured: {plan.s_per_token_measured}")
    a(f"  wall_s_linear: {plan.wall_s_linear}")
    a(f"  wall_s_gqa_adjusted: {plan.wall_s_gqa_adjusted}")
    a(f"  wall_s_lower: {plan.wall_s_lower}")
    a(f"  wall_s_upper: {plan.wall_s_upper}")
    a(f"  wall_s_both_vehicles_linear: {plan.wall_s_both_vehicles_linear}")
    a("  bound: ~23 min (short prompts) to a few hours (long GQA)")
    a("")
    a("memory (ESTIMATED):")
    a(f"  bf16_weight_bytes: {plan.bf16_weight_bytes}")
    a(f"  bf16_weight_gb: {_gb(plan.bf16_weight_bytes)}")
    a(f"  resident_weight_bytes: {plan.resident_weight_bytes}")
    a(f"  resident_weight_gb: {_gb(plan.resident_weight_bytes)}")
    a(f"  one_layer_microbatch_bytes: {plan.one_layer_microbatch_bytes}")
    a(f"  peak_process_bytes: {plan.peak_process_bytes}")
    a(f"  peak_process_gb: {_gb(plan.peak_process_bytes)}")
    a(f"  peak_if_resident_stays_bytes: {plan.peak_if_resident_stays_bytes}")
    a(f"  peak_if_resident_stays_gb: {_gb(plan.peak_if_resident_stays_bytes)}")
    a(f"  unified_memory_bytes: {plan.unified_memory_bytes}")
    a("")
    a("preflight (GPU lane must check before --run; this lane does not mutate):")
    for key, value in plan.preflight.items():
        a(f"  {key}: {value}")
    a("")
    a("adequacy law: n_fit >= fit_dim or the number is REFUSED")
    a(f"  NS-014 wreck: {NS014_ROWS}/{NS014_DIM} = {NS014_RPD}")
    a(f"  v1 256/6144 = {256 / 6144}")
    a(f"  v1 256/17408 = {256 / 17408}")
    a("  mixer_x is required. A v*silu(z) proxy is DEGENERATE and is not mixer_x.")
    a("")
    a(f"out_dir: {plan.out_dir}")
    return "\n".join(lines) + "\n"


def plan_as_dict(plan: CapturePlan) -> dict[str, Any]:
    d = asdict(plan)
    d["sites"] = [asdict(s) for s in plan.sites]
    return d


# ---------------------------------------------------------------------------
# Preflight — observe only. Never stop the resident.
# ---------------------------------------------------------------------------


def _hw_memsize() -> int | None:
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except Exception:
        return None


def _disk_free(path: Path) -> int | None:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        probe = Path("/")
    try:
        st = os.statvfs(str(probe))
        return int(st.f_bavail) * int(st.f_frsize)
    except OSError:
        return None


def _resident_alive() -> dict[str, Any]:
    info: dict[str, Any] = {
        "checked": True,
        "alive": None,
        "action": "DO_NOT_STOP_RESTART_OR_TALK",
    }
    sock = Path("/Users/scammermike/Downloads/hawking/workspace/ops/genesis-resident.sock")
    alt = repo_root() / "workspace/ops/genesis-resident.sock"
    info["socket_exists"] = sock.exists() or alt.exists()
    try:
        import subprocess

        proc = subprocess.run(
            ["pgrep", "-lf", "genesis-resident"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        info["pgrep_rc"] = proc.returncode
        info["alive"] = proc.returncode == 0 and bool(proc.stdout.strip())
        info["pgrep_n_lines"] = len([ln for ln in proc.stdout.splitlines() if ln.strip()])
    except (OSError, subprocess.TimeoutExpired):
        info["pgrep"] = "unavailable"
    return info


def preflight_capture(
    *,
    parent_dir: Path,
    q4_dir: Path,
    out_dir: Path,
    need_bytes: int,
    mutate: bool,
) -> dict[str, Any]:
    parent_dir = Path(parent_dir)
    q4_dir = Path(q4_dir)
    lock = Path("/tmp/hawking-gpu-lane.lock")
    owner = None
    if lock.is_dir():
        owner_path = lock / "owner"
        if owner_path.is_file():
            owner = owner_path.read_text().strip()
    mem = _hw_memsize()
    disk = _disk_free(out_dir if out_dir.parent.exists() else Path("/Users/scammermike/Downloads"))
    parent_ok = parent_dir.is_dir() and "bf16" in str(parent_dir).lower()
    parent_markers = " ".join(
        [
            str(parent_dir),
            str(q4_dir),
        ]
    ).lower()
    candidate = None
    for marker in CANDIDATE_UNDER_TEST_MARKERS:
        if marker.lower() in parent_markers:
            candidate = marker
            break
    resident = _resident_alive()
    report = {
        "mutate": bool(mutate),
        "parent_dir_exists": parent_dir.is_dir(),
        "parent_looks_bf16": parent_ok,
        "q4_dir_exists": q4_dir.is_dir(),
        "candidate_under_test": candidate,
        "gpu_lock_exists": lock.exists(),
        "gpu_lock_owner": owner,
        "hw_memsize": mem,
        "disk_free_bytes": disk,
        "need_bytes": int(need_bytes),
        "disk_enough": None if disk is None else disk >= int(need_bytes),
        "resident": resident,
        "unified_gb": 96,
        "must_not_stop_resident": True,
        "must_not_expand_q4_to_float": True,
        "advice": (
            "Pause the resident only from the GPU lane if 96 GB cannot hold "
            f"resident ({_gb(RESIDENT_WEIGHT_BYTES)} GB) + BF16 "
            f"({_gb(BF16_WEIGHT_BYTES)} GB) + activations. This tool will not pause it."
        ),
    }
    return report


def assert_run_preflight(pre: Mapping[str, Any]) -> None:
    if pre.get("candidate_under_test"):
        raise AdequacyRefused(
            f"REFUSED: parent path looks like a candidate under test ({pre['candidate_under_test']})."
        )
    if not pre.get("parent_looks_bf16"):
        raise AdequacyRefused("REFUSED: parent is not the BF16 directory. Calibration source must be PARENT_BF16_REAL.")
    if pre.get("gpu_lock_exists") and os.environ.get("QWEN38_CAPTURE_I_HOLD_GPU") != "1":
        raise RuntimeError(
            f"GPU lock held by {pre.get('gpu_lock_owner')!r}. "
            "Do not start a capture under the resident. "
            "Set QWEN38_CAPTURE_I_HOLD_GPU=1 only from the serialized GPU lane."
        )
    if pre.get("disk_enough") is False:
        raise RuntimeError(
            f"disk_free={pre.get('disk_free_bytes')} < need={pre.get('need_bytes')}"
        )


# ---------------------------------------------------------------------------
# Streaming f16 writers — never accumulate a full site
# ---------------------------------------------------------------------------


class SiteLayerWriter:
    def __init__(self, path: Path, width: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.n_rows = 0
        self._sum_abs = 0.0
        self._sum_sq = 0.0
        self._sha = hashlib.sha256()

    def append(self, rows: Any) -> int:
        import numpy as np

        arr = np.ascontiguousarray(rows, dtype=np.float16)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != self.width:
            raise ValueError(f"{self.path} width {arr.shape[-1]} != {self.width}")
        raw = arr.tobytes()
        with self.path.open("ab") as fh:
            fh.write(raw)
        self._sha.update(raw)
        x = np.asarray(arr, dtype=np.float64)
        self._sum_abs += float(np.abs(x).sum())
        self._sum_sq += float(np.square(x).sum())
        self.n_rows += int(arr.shape[0])
        return int(arr.shape[0])

    def close(self) -> dict[str, Any]:
        n = max(self.n_rows * self.width, 1)
        return {
            "n_rows": self.n_rows,
            "path": str(self.path),
            "mean_abs": self._sum_abs / n,
            "rms": math.sqrt(self._sum_sq / n),
            "sha256": self._sha.hexdigest(),
            "dtype": "f16",
            "width": self.width,
        }


class StreamingCapture:
    """One writer per (site, layer). Open/append/close so we never pin a cube."""

    def __init__(self, root: Path, store_n: dict[str, int]):
        self.root = Path(root)
        self.store_n = dict(store_n)
        self.written: dict[tuple[str, int], int] = {}
        self.writers: dict[tuple[str, int], SiteLayerWriter] = {}
        self.closed: dict[tuple[str, int], dict[str, Any]] = {}

    def _key(self, site_id: str, layer: int) -> tuple[str, int]:
        return site_id, int(layer)

    def remaining(self, site_id: str, layer: int) -> int:
        have = self.written.get(self._key(site_id, layer), 0)
        return max(0, int(self.store_n[site_id]) - have)

    def append(self, site_id: str, layer: int, rows: Any) -> int:
        left = self.remaining(site_id, layer)
        if left <= 0:
            return 0
        import numpy as np

        arr = np.ascontiguousarray(rows)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        take = arr[:left]
        key = self._key(site_id, layer)
        writer = self.writers.get(key)
        if writer is None:
            site = SITE_BY_ID[site_id]
            path = self.root / site_id / f"L{int(layer):02d}.f16"
            writer = SiteLayerWriter(path, site.width)
            self.writers[key] = writer
        n = writer.append(take)
        self.written[key] = self.written.get(key, 0) + n
        return n

    def close(self) -> dict[str, Any]:
        sites: dict[str, Any] = {}
        for site in SITES:
            per_layer: dict[str, Any] = {}
            n_tokens = 0
            for layer in range(site.n_layers):
                key = self._key(site.site_id, layer)
                writer = self.writers.get(key)
                if writer is None:
                    continue
                rec = writer.close()
                self.closed[key] = rec
                per_layer[str(layer)] = rec
                n_tokens = rec["n_rows"]
            sites[site.site_id] = {
                "width": site.width,
                "n_layers": site.n_layers,
                "n_tokens": n_tokens,
                "store_n": self.store_n[site.site_id],
                "dtype": "f16",
                "per_layer": per_layer,
            }
        self.writers.clear()
        return sites


# ---------------------------------------------------------------------------
# Receipt (v1-readable + v2 sites)
# ---------------------------------------------------------------------------


def build_receipt(
    *,
    vehicle: str,
    source_dir: Path,
    n_tokens: int,
    prompts: Sequence[Mapping[str, Any]],
    sites: Mapping[str, Any],
    wall_s: float,
    adequacy: Mapping[str, Any],
    mixer_x_kind: Mapping[str, str],
    notes: Sequence[str],
    status: str,
) -> dict[str, Any]:
    hidden_site = sites.get("post_input_norm") or {}
    per_layer = hidden_site.get("per_layer") or {}
    x_source = "PARENT_BF16_REAL" if vehicle == "parent" else "COHERENT_Q4_VEHICLE"
    receipt = {
        "schema": SCHEMA_V2,
        "schema_v1_compat": V1_SCHEMA,
        "status": status,
        "source": {
            "model_dir": str(source_dir),
            "not_synthetic": True,
            "forward": "mlx_lm.qwen3_5_text" if vehicle == "parent" else "hawking_native_q4_no_expand",
            "vehicle": vehicle,
            "x_source": x_source,
            "parent_repo": PARENT_REPO,
            "base_repo": BASE_REPO,
            "base_revision": BASE_REVISION,
            "rmsnorm": "mlx nn.RMSNorm (parent) or native (q4)",
        },
        "n_tokens": int(n_tokens),
        "n_layers": LAYERS,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "mixer_in": MIXER_IN,
        "prompts": list(prompts),
        "per_layer": per_layer,
        "sites": dict(sites),
        "wall_s": float(wall_s),
        "fit_kind": "real_routed_activation_capture",
        "holdout": {
            "kind": "prompt",
            "hold_frac": HOLD_FRAC,
            "note": "split by prompt id; row shuffle is refused",
        },
        "adequacy": dict(adequacy),
        "mixer_x_kind": dict(mixer_x_kind),
        "notes": list(notes),
        "sha256_self": None,
    }
    return receipt


def seal_receipt(receipt: dict[str, Any], root: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    for site_id in sorted((receipt.get("sites") or {}).keys()):
        per_layer = (receipt["sites"][site_id].get("per_layer") or {})
        for layer in sorted(per_layer, key=lambda x: int(x)):
            digest = per_layer[layer].get("sha256")
            if digest:
                hasher.update(digest.encode("ascii"))
    receipt["sha256_self"] = hasher.hexdigest()
    path = Path(root) / "capture-result.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# ---------------------------------------------------------------------------
# Parent BF16 capture via mlx_lm (imported only on --run)
# ---------------------------------------------------------------------------


def _to_numpy_f16(array: Any) -> Any:
    import numpy as np

    try:
        import mlx.core as mx

        mx.eval(array)
    except Exception:
        pass
    arr = np.array(array)
    if arr.ndim == 3:
        # (B, T, C) — capture is token-major, one sequence
        arr = arr.reshape(-1, arr.shape[-1])
    return np.ascontiguousarray(arr, dtype=np.float16)


def _bind_call(module: Any, fn: Callable[..., Any]) -> None:
    module.__call__ = fn  # instance override; mlx has no forward-hook API


def _unwrap_language_model(model: Any) -> Any:
    for attr in ("language_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None and (hasattr(inner, "layers") or hasattr(inner, "model")):
            model = inner
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "layers"):
        return model.model
    return model


def _layers_of(lm: Any) -> list[Any]:
    layers = getattr(lm, "layers", None)
    if layers is None:
        raise RuntimeError("language model has no layers")
    return list(layers)


def install_site_hooks(
    model: Any,
    *,
    on_site: Callable[[str, int, Any], None],
) -> dict[str, str]:
    """Hook real sites. mixer_x is out_proj/o_proj *input*, never v*silu(z)."""
    lm = _unwrap_language_model(model)
    layers = _layers_of(lm)
    if len(layers) != LAYERS:
        raise RuntimeError(f"expected {LAYERS} layers, got {len(layers)}")
    mixer_kinds: dict[str, str] = {}

    def tap_out(mod: Any, site_id: str, layer: int) -> None:
        orig = mod.__call__

        def hooked(*args: Any, **kwargs: Any) -> Any:
            y = orig(*args, **kwargs)
            on_site(site_id, layer, y)
            return y

        _bind_call(mod, hooked)

    def tap_in(mod: Any, site_id: str, layer: int) -> None:
        orig = mod.__call__

        def hooked(x: Any, *args: Any, **kwargs: Any) -> Any:
            on_site(site_id, layer, x)
            return orig(x, *args, **kwargs)

        _bind_call(mod, hooked)

    for layer_idx, layer in enumerate(layers):
        tap_out(layer.input_layernorm, "post_input_norm", layer_idx)
        tap_out(layer.post_attention_layernorm, "post_attn_norm", layer_idx)
        mlp = layer.mlp
        if not hasattr(mlp, "down_proj"):
            raise RuntimeError(f"layer {layer_idx} mlp has no down_proj")
        tap_in(mlp.down_proj, "post_swiglu", layer_idx)
        if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
            tap_in(layer.linear_attn.out_proj, "mixer_x", layer_idx)
            mixer_kinds[str(layer_idx)] = "deltanet_gated_recurrent_normed_out_proj_input"
        elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            tap_in(layer.self_attn.o_proj, "mixer_x", layer_idx)
            mixer_kinds[str(layer_idx)] = "gqa_softmax_gated_repeat_v_o_proj_input"
        else:
            mixer_kinds[str(layer_idx)] = "DEGENERATE_MISSING_HOOK"
    final = getattr(lm, "norm", None)
    if final is None:
        raise RuntimeError("missing language_model.model.norm (final_norm)")
    tap_out(final, "final_norm", 0)
    mixer_kinds["final_norm"] = "language_model.model.norm"
    if any(v.startswith("DEGENERATE") for k, v in mixer_kinds.items() if k.isdigit()):
        raise AdequacyRefused(
            "REFUSED: mixer_x hook missing on at least one layer. "
            "Will not silently store v*silu(z) as mixer_x."
        )
    return mixer_kinds


def _load_mlx(parent_dir: Path) -> tuple[Any, Any]:
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError(
            "mlx_lm is required for --run parent capture and is imported only then"
        ) from exc
    model, tokenizer = load(str(parent_dir))
    return model, tokenizer


def _tokenize(tokenizer: Any, text: str, target_n: int) -> list[int]:
    if hasattr(tokenizer, "encode"):
        ids = list(tokenizer.encode(text))
    else:
        ids = list(tokenizer(text)["input_ids"])
    if len(ids) < MIN_SEQ_LEN:
        # Repeat the body until we can trim to target. Chat wrap stays once.
        body = text
        while len(ids) < max(target_n, MIN_SEQ_LEN) and len(ids) < MAX_SEQ_LEN:
            body = body + "\n" + text
            ids = list(tokenizer.encode(body)) if hasattr(tokenizer, "encode") else list(tokenizer(body)["input_ids"])
    if len(ids) > target_n:
        ids = ids[:target_n]
    if len(ids) < MIN_SEQ_LEN:
        raise RuntimeError(f"prompt tokenized to {len(ids)} < MIN_SEQ_LEN={MIN_SEQ_LEN}")
    return [int(x) for x in ids]


def run_parent_capture(
    *,
    out_dir: Path,
    parent_dir: Path,
    store_mode: str,
    prompt_plan: PromptPlan | None = None,
) -> dict[str, Any]:
    """GPU lane only. Do not call from this builder lane."""
    import numpy as np

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("mlx is required for --run") from exc

    prompt_plan = prompt_plan or build_prompt_plan()
    store_n = {s.site_id: site_store_n(s, store_mode=store_mode) for s in SITES}
    vehicle_root = Path(out_dir) / "parent_bf16"
    vehicle_root.mkdir(parents=True, exist_ok=True)
    stream = StreamingCapture(vehicle_root, store_n)
    fit_left = {s.site_id: int(math.floor((1.0 - HOLD_FRAC) * store_n[s.site_id])) for s in SITES}
    hold_left = {s.site_id: store_n[s.site_id] - fit_left[s.site_id] for s in SITES}

    notes: list[str] = []
    prompt_records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    model, tokenizer = _load_mlx(parent_dir)

    # Prompt-level remaining budget, closed over by the hook.
    current = {"split": "fit", "allow": {s.site_id: 0 for s in SITES}}

    def hooked_site(site_id: str, layer: int, tensor: Any) -> None:
        allow = current["allow"].get(site_id, 0)
        if allow <= 0:
            return
        arr = _to_numpy_f16(tensor)
        take = arr[:allow]
        n = stream.append(site_id, layer, take)
        if n:
            current["allow"][site_id] = allow - n

    mixer_kinds = install_site_hooks(model, on_site=hooked_site)

    for spec in prompt_plan.prompts:
        ids = _tokenize(tokenizer, spec.text, spec.target_n_tokens)
        n_here = len(ids)
        current["split"] = spec.split
        current["allow"] = {}
        for site in SITES:
            budget = fit_left if spec.split == "fit" else hold_left
            take_n = min(n_here, budget[site.site_id])
            current["allow"][site.site_id] = take_n
            budget[site.site_id] -= take_n
        tokens = mx.array([ids], dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        del logits
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        prompt_records.append(
            {
                "prompt": spec.text[:200],
                "cls": spec.cls,
                "n_tokens": n_here,
                "ids": ids,
                "split": spec.split,
                "prompt_id": spec.prompt_id,
            }
        )

    wall_s = time.perf_counter() - t0
    sites = stream.close()
    # Attach split counts actually stored.
    for site in SITES:
        stored = sites[site.site_id].get("n_tokens") or 0
        stored_fit = int(math.floor((1.0 - HOLD_FRAC) * store_n[site.site_id])) - fit_left[site.site_id]
        stored_hold = (store_n[site.site_id] - int(math.floor((1.0 - HOLD_FRAC) * store_n[site.site_id]))) - hold_left[
            site.site_id
        ]
        sites[site.site_id]["n_fit"] = stored_fit
        sites[site.site_id]["n_hold"] = stored_hold
        sites[site.site_id]["holdout_by_prompt"] = True
        verdict = adequacy_gate(
            n_rows=int(stored),
            fit_dim=site.fit_dim,
            n_prompts=len(prompt_records),
            procedure="fit_from_X",
            n_fit=stored_fit,
            n_hold=stored_hold,
            holdout_by_prompt=True,
            x_source="PARENT_BF16_REAL",
            not_synthetic=True,
            site_id=site.site_id,
        )
        sites[site.site_id]["adequacy"] = verdict.as_dict()

    adequacy = {
        site.site_id: sites[site.site_id]["adequacy"] for site in SITES
    }
    any_refused = any(v.get("status") == "REFUSED" for v in adequacy.values())
    status = "CAPTURED_REAL_BF16_MULTI_SITE"
    if any_refused:
        status = "CAPTURED_REAL_BF16_MULTI_SITE_ADEQUACY_REFUSED"
        notes.append("At least one site adequacy is REFUSED. Do not fit those organs.")
    receipt = build_receipt(
        vehicle="parent",
        source_dir=parent_dir,
        n_tokens=int(sum(p["n_tokens"] for p in prompt_records)),
        prompts=prompt_records,
        sites=sites,
        wall_s=wall_s,
        adequacy=adequacy,
        mixer_x_kind=mixer_kinds,
        notes=notes,
        status=status,
    )
    return seal_receipt(receipt, vehicle_root)


def run_q4_capture(**_kwargs: Any) -> dict[str, Any]:
    """Q4 twin is additional. Expand-to-float is REFUSED. Native path only."""
    raise AdequacyRefused(
        "REFUSED: Q4 vehicle capture must use the native Hawking reader "
        "(no expand-to-float). Set HAWKING_Q4_CAPTURE_BIN to a native "
        "capture helper, or run the Q4 stream from the GPU lane's native "
        "binary. This Python harness will not dequant Q4 into mlx."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_capture_plan(
        store_mode=args.store_mode,
        q4_store_mode=args.q4_store_mode,
        parent_dir=Path(args.parent_dir) if args.parent_dir else None,
        q4_dir=Path(args.q4_dir) if args.q4_dir else None,
        out_dir=Path(args.out) if args.out else None,
        include_q4=True,
    )
    sys.stdout.write(format_plan(plan))
    if args.json:
        sys.stdout.write(json.dumps(plan_as_dict(plan), indent=2) + "\n")
    return 0


def cmd_check_adequacy(args: argparse.Namespace) -> int:
    path = Path(args.capture) if args.capture else resolve_v1_capture()
    if path is None:
        print("REFUSED: no capture-result.json found", file=sys.stderr)
        return 2
    meta_path = path / "capture-result.json" if path.is_dir() else path
    meta = json.loads(meta_path.read_text())
    verdict = adequacy_from_capture(
        meta,
        fit_dim=int(args.fit_dim),
        procedure=args.procedure,
        site_id=args.site,
        path=str(meta_path),
        rank_claimed=args.rank,
        rank_clamped_to_n_fit=bool(args.rank_clamped),
    )
    sys.stdout.write(json.dumps(verdict.as_dict(), indent=2) + "\n")
    if verdict.status == "REFUSED":
        sys.stdout.write("VERDICT: REFUSED\n")
        return 1
    sys.stdout.write("VERDICT: ACCEPTED\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    parent_dir = Path(args.parent_dir) if args.parent_dir else default_parent_dir()
    q4_dir = Path(args.q4_dir) if args.q4_dir else default_q4_dir()
    out_dir = Path(args.out) if args.out else default_v2_out()
    vehicles = [v.strip() for v in args.vehicles.split(",") if v.strip()]
    plan = build_capture_plan(
        store_mode=args.store_mode,
        q4_store_mode=args.q4_store_mode,
        parent_dir=parent_dir,
        q4_dir=q4_dir,
        out_dir=out_dir,
        include_q4="q4" in vehicles,
    )
    assert_run_preflight(plan.preflight)
    results: dict[str, Any] = {}
    if "parent" in vehicles:
        results["parent"] = run_parent_capture(
            out_dir=out_dir,
            parent_dir=parent_dir,
            store_mode=args.store_mode,
        )
    if "q4" in vehicles:
        bin_path = os.environ.get("HAWKING_Q4_CAPTURE_BIN")
        if not bin_path:
            results["q4"] = {
                "verdict": "REFUSED",
                "reason": (
                    "REFUSED: no native Q4 capture binary. "
                    "Will not expand-to-float. Parent stream is independent."
                ),
            }
            (out_dir / "q4_vehicle").mkdir(parents=True, exist_ok=True)
            (out_dir / "q4_vehicle" / "capture-result.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA_V2,
                        "status": "REFUSED",
                        "verdict": "REFUSED",
                        "reason": results["q4"]["reason"],
                        "score": None,
                        "emit_score": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            results["q4"] = run_q4_capture()
    sys.stdout.write(json.dumps({"ran": vehicles, "out": str(out_dir), "summary_keys": list(results)}, indent=2) + "\n")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print estimates; no GPU, no model load")
    mode.add_argument("--dry-run", action="store_true", dest="plan", help="alias for --plan")
    mode.add_argument("--run", action="store_true", help="GPU lane only: run the capture")
    mode.add_argument("--check-adequacy", action="store_true", help="run the gate on a capture receipt")
    p.add_argument("--store-mode", default="site-split-n", choices=("site-split-n", "full", "census"))
    p.add_argument("--q4-store-mode", default="census", choices=("site-split-n", "full", "census"))
    p.add_argument("--parent-dir", default=None)
    p.add_argument("--q4-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--vehicles", default="parent", help="comma list: parent,q4")
    p.add_argument("--json", action="store_true", help="also dump plan as JSON")
    p.add_argument("--capture", default=None, help="capture dir or capture-result.json")
    p.add_argument("--fit-dim", type=int, default=MIXER_IN)
    p.add_argument("--procedure", default="fit_from_X", choices=("fit_from_X", "eval_weight_only"))
    p.add_argument("--site", default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--rank-clamped", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.plan:
        return cmd_plan(args)
    if args.check_adequacy:
        return cmd_check_adequacy(args)
    if args.run:
        return cmd_run(args)
    build_arg_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
