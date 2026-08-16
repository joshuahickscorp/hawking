#!/usr/bin/env python3
"""Q80 capability-vs-BPW curve. Owns the BPW half of the fs/weight law.

Measures where output-space organ cosine (D23 bar 0.8604) actually breaks as
complete physical BPW falls. Packed bytes are billed from real codec payloads,
not design formulae. Does not pack five 13 GiB artifacts. Does not write a
Metal kernel. Does not parse capture-result.json.

Capture defect (stated on every point): the bound 25258-token capture is
route-starved (p10=34, p50=258, 221 never-routed pairs, no on-disk
post-SwiGLU X). down_proj X is recomputed as silu(X@G.T)*(X@U.T) from the
same retained router-input rows. An additive extension capture (layers 0-22
at last check) supplies extra packed post-SwiGLU rows when present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
    GROUP_BINARY,
    GROUP_UNIFORM,
    MAGIC_ACT_SVD,
    _activation_weighted_svd_factors,
    _binary_codec,
    _container,
    _decode_activation_weighted_svd_low_rank_codec,
    _decode_uniform_codec,
    _factor_codec,
    _mean_row_cosine,
    _uniform_codec,
)
from lab.operators.dsv4f_residual_composition_oracle import (  # noqa: E402
    residual_identity_layer_cosine,
    residual_identity_product,
)
from lab.operators.hgravs01_adapter import HGRAVS01_SCHEMA  # noqa: E402
from lab.operators.q80_capture_index import inspect_index  # noqa: E402
from lab.operators.q80_mixed_representation_pack import (  # noqa: E402
    CAPTURE_SHA256,
    F_EXPERT,
    F_NONEXPERT,
    SOURCE_ELEMENTS,
    CaptureHiddens,
    decode_binary,
    post_swiglu,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402
from lab.operators.residual_compact_codec import (  # noqa: E402
    decode_residual_compact,
    encode_residual_compact,
)
from lab.receipts import seal  # noqa: E402

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
CAPTURE = (
    MAIN_HAWKING
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-diagnostics/source-bf16-capture-n192-scale64"
)
ARTIFACT_1P44 = (
    MAIN_HAWKING
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-candidates/mixed-1p5-v1"
)
EXT_CAPTURE = Path(
    "/Users/scammermike/.claude-grok/worktrees/q80-capture-coverage-20260816-003209"
    "/workspace/campaign/records/ascension-sandbox/physical/qwen80"
    "/quality-diagnostics/source-bf16-capture-reservoir-n512-ext"
)

SCHEMA = "hawking.ascension.qwen80_subbit_capability_curve.v1"
BAR = 0.8604
N_LAYERS = 48
N_EXPERTS = 512
HIDDEN = 2048
INTERMEDIATE = 512
N_ROUTED_PER_ORGAN = 24_576
ELEMS_PER_EXPERT_ORGAN = INTERMEDIATE * HIDDEN
EXPERT_ELEMS = 3 * N_ROUTED_PER_ORGAN * ELEMS_PER_EXPERT_ORGAN
NONEXPERT_ELEMS = SOURCE_ELEMENTS - EXPERT_ELEMS
HOLD_FRAC = 0.25
MIN_HOLDOUT_ROWS = 4
ROW_CAP = 2048
ROW_SEED = 0xD0C70A
NULL_N = 16
NULL_SEED = 20260816
TARGET_COMPLETE = (1.44, 1.00, 0.80, 0.655, 0.50)
ARTIFACT_OVERHEAD_BYTES = 13_117_492  # catalog+manifest+fit tables from mixed-1p5-v1
ARTIFACT_NONEEXPERT_Q8_BYTES = 2_439_063_174
ARTIFACT_NONEEXPERT_Q8_BPW = 8.250600705299505

CAPTURE_IDENTITY = {
    "path": str(CAPTURE),
    "capture_result_path": str(CAPTURE / "capture-result.json"),
    "sha256": CAPTURE_SHA256,
    "schema": (
        "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1"
    ),
    "status": "Q80_SUBBIT_CAPABILITY_CURVE_FIT",
    "fit_kind": "real_routed_activation_capture",
    "not_synthetic_unit_direction": True,
}

# Increment ratios ||Δ||/||h|| implied by the already-measured 48-layer true
# residual-stream growth in Q80_COHERENCE_LAYER_DRIFT_PROBE.json under the
# orthogonal-increment identity r = sqrt(max(g^2 - 1, 0)). Conservative.
# This is NOT a 4-layer mixed-codec extrapolation.
_TRUE_RESIDUAL_GROWTH = [
    1.0,
    1.7823134977772206,
    1.5504439556908842,
    1.0221650102418662,
    1.037215699492779,
    1.0593669014552198,
    1.0333478464231964,
    0.9891195835054399,
    0.9863234629603908,
    1.0933529939653293,
    1.030502465777895,
    1.0403803576927477,
    0.9866350863795204,
    1.03237365519041,
    1.0597590121864975,
    1.0093174686798088,
    0.9851302814330738,
    1.0093978934645491,
    0.980787288895073,
    0.972613980859252,
    0.9546139383518276,
    1.0463522391143976,
    1.132043260216194,
    1.0177142506110872,
    1.026204600449286,
    0.9933798940791848,
    1.0673496745237827,
    1.032190432525093,
    0.9861084959812567,
    1.0181412491224835,
    0.9945325570024303,
    1.0211297624317996,
    1.0372543111567836,
    1.0139685913008436,
    1.1155244242910738,
    1.1192586100000000,
    1.0612187382541165,
    1.0306838688556963,
    1.0597281976012403,
    1.0952629406089092,
    1.0068430936313575,
    1.0828395303413034,
    1.1403059382423253,
    1.0679807039666458,
    1.1855288433189595,
    1.106407789670522,
    1.5055570825593938,
    0.7305522211126123,
]


def _stable_name_seed(name: str) -> int:
    h = 2166136261
    for ch in name:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h & 0xFFFF)


def complete_bpw(expert_bpw: float, nonexpert_bpw: float) -> float:
    return float(F_EXPERT) * float(expert_bpw) + float(F_NONEXPERT) * float(nonexpert_bpw)


def physical_bpw(nbytes: int, elements: int) -> float:
    return 8.0 * float(nbytes) / float(max(int(elements), 1))


def holdout_split(
    n: int, *, hold_frac: float = HOLD_FRAC, seed: int, min_holdout: int = MIN_HOLDOUT_ROWS
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Seeded permutation split. n < min_holdout keeps every row (no holdout)."""

    n = int(n)
    idx = np.arange(n, dtype=np.int64)
    if n < int(min_holdout):
        return idx, idx, False
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(idx)
    n_hold = max(1, int(round(n * float(hold_frac))))
    n_hold = min(n_hold, n - 1)
    hold = np.sort(perm[:n_hold])
    fit = np.sort(perm[n_hold:])
    return fit, hold, True


def matched_magnitude_null(src: np.ndarray, hat: np.ndarray, *, seed: int) -> np.ndarray:
    err = (np.asarray(hat, dtype=np.float32) - np.asarray(src, dtype=np.float32)).reshape(-1).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(err)
    return (np.asarray(src, dtype=np.float32).reshape(-1) + err).reshape(src.shape)


def output_cosine(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray) -> float:
    y = X @ np.asarray(W, dtype=np.float32).T
    y_hat = X @ np.asarray(W_hat, dtype=np.float32).T
    return float(_mean_row_cosine(y, y_hat))


def output_rel_l2(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray) -> float:
    y = X @ np.asarray(W, dtype=np.float32).T
    y_hat = X @ np.asarray(W_hat, dtype=np.float32).T
    num = float(np.linalg.norm(y - y_hat))
    den = float(np.linalg.norm(y)) + 1e-12
    return num / den


def increment_ratios_from_growth(growth: list[float]) -> list[float]:
    """r = sqrt(max(g^2 - 1, 0)) from successive residual-stream norms."""

    out: list[float] = []
    for g in growth[1:]:
        gg = float(g)
        out.append(float(math.sqrt(max(gg * gg - 1.0, 0.0))))
    return out


def encode_binary(W: np.ndarray, *, group_size: int) -> tuple[bytes, np.ndarray]:
    codec = _binary_codec(W, group_size=int(group_size))
    hat = decode_binary(codec.payload)
    return codec.payload, np.asarray(hat, dtype=np.float32)


def encode_resid(W: np.ndarray, *, outlier_ratio: float) -> tuple[bytes, np.ndarray]:
    codec = encode_residual_compact(
        W,
        outlier_ratio=float(outlier_ratio),
        group_size=GROUP_BINARY,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    hat = decode_residual_compact(codec.payload)
    return codec.payload, np.asarray(hat, dtype=np.float32)


def encode_uniform(W: np.ndarray, *, bits: int) -> tuple[bytes, np.ndarray]:
    codec = _uniform_codec(W, bits=int(bits), group_size=GROUP_UNIFORM)
    hat = _decode_uniform_codec(codec.payload)
    return codec.payload, np.asarray(hat, dtype=np.float32)


def _hgravs_payload_from_factors(
    W: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    bits: int,
    identity: dict[str, Any],
    fit_meta: dict[str, Any],
) -> tuple[bytes, np.ndarray, int]:
    left_body, _, left_meta = _factor_codec(left, bits=int(bits))
    right_body, _, right_meta = _factor_codec(right, bits=int(bits))
    header = {
        "schema": HGRAVS01_SCHEMA,
        "representation": ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
        "shape": [int(W.shape[0]), int(W.shape[1])],
        "matrix_shape": [int(W.shape[0]), int(W.shape[1])],
        "elements": int(W.size),
        "rank": int(left.shape[1]),
        "factor_bits": int(bits),
        "factor_group_size": GROUP_UNIFORM,
        "left": left_meta,
        "right": right_meta,
        "left_body_bytes": len(left_body),
        "right_body_bytes": len(right_body),
        "fit": fit_meta,
        "activation_capture": {
            "path": identity.get("path"),
            "capture_result_path": identity.get("capture_result_path"),
            "sha256": identity.get("sha256"),
            "schema": identity.get("schema"),
            "status": identity.get("status"),
            "fit_kind": identity.get("fit_kind"),
            "not_synthetic_unit_direction": True,
        },
    }
    payload = _container(MAGIC_ACT_SVD, header, left_body + right_body)
    hat = _decode_activation_weighted_svd_low_rank_codec(payload)
    return payload, np.asarray(hat, dtype=np.float32), int(left.shape[1])


def encode_hgravs_ranks(
    W: np.ndarray,
    X_fit: np.ndarray | None,
    ranks: list[int],
    *,
    bits: int,
    identity: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """One Gram/eigh (or one weight-space SVD), then billed payloads per rank.

    Rank is NOT clamped to n_fit_rows. The 512-d / 2048-d Gram is ridge-
    regularized, matching the mixed-1p5 packer. encode_hgravs01's n_fit clamp
    is deliberately not used.
    """

    matrix = np.ascontiguousarray(W, dtype=np.float32)
    uniq = sorted({int(r) for r in ranks if int(r) >= 1})
    out: dict[int, dict[str, Any]] = {}
    if not uniq:
        return out
    max_r = max(uniq)
    if X_fit is not None and int(np.asarray(X_fit).shape[0]) >= 1:
        X = np.ascontiguousarray(X_fit, dtype=np.float32)
        left, right, fit_meta = _activation_weighted_svd_factors(matrix, X, rank=max_r)
        kind = "activation_weighted"
    else:
        actual = min(max_r, matrix.shape[0], matrix.shape[1])
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        left = (u[:, :actual] * s[:actual]).astype(np.float32)
        right = vt[:actual, :].astype(np.float32)
        fit_meta = {
            "fit": "weight_space_truncated_svd",
            "reason": "n_fit_rows==0_or_missing_X",
            "rank": int(actual),
            "n_fit_tokens": 0,
            "requested_rank": int(max_r),
            "rank_clamped_to_n_fit": False,
        }
        kind = "weight_space"
    for rank in uniq:
        r = min(int(rank), int(left.shape[1]))
        payload, hat, achieved = _hgravs_payload_from_factors(
            matrix,
            left[:, :r],
            right[:r, :],
            bits=bits,
            identity=identity,
            fit_meta={**fit_meta, "requested_rank": int(rank), "rank": int(r)},
        )
        out[int(rank)] = {
            "payload_bytes": int(len(payload)),
            "W_hat": hat,
            "achieved_rank": int(achieved),
            "fit_kind": kind,
            "rank_clamped_to_n_fit": False,
            "rank_clamped_to_matrix": bool(r != int(rank)),
        }
    return out


def _score_hat(
    W: np.ndarray,
    hat: np.ndarray,
    X_score: np.ndarray | None,
    *,
    null_n: int,
    seed: int,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "output_cosine": None,
        "output_rel_l2": None,
        "null_cosines": [],
        "null_p05": None,
        "null_p50": None,
        "null_p95": None,
        "surplus_over_null_p95": None,
        "separated_from_null_p95": None,
    }
    if X_score is None or int(X_score.shape[0]) < 1:
        return rec
    cos = output_cosine(W, hat, X_score)
    rel = output_rel_l2(W, hat, X_score)
    nulls: list[float] = []
    for i in range(int(null_n)):
        null = matched_magnitude_null(W, hat, seed=int(seed) + i)
        nulls.append(output_cosine(W, null, X_score))
    arr = np.asarray(nulls, dtype=np.float64)
    p05 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    rec.update(
        {
            "output_cosine": float(cos),
            "output_rel_l2": float(rel),
            "null_cosines": [float(x) for x in nulls],
            "null_p05": p05,
            "null_p50": p50,
            "null_p95": p95,
            "surplus_over_null_p95": float(cos - p95),
            "separated_from_null_p95": bool(cos > p95 + 1e-6),
        }
    )
    return rec


def select_pairs(counts: np.ndarray, *, smoke: bool) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add(layer: int, expert: int, role: str) -> None:
        key = (int(layer), int(expert))
        if key in seen:
            return
        seen.add(key)
        pairs.append(
            {
                "layer": int(layer),
                "expert": int(expert),
                "role": role,
                "n_rows_primary": int(counts[layer, expert]),
            }
        )

    # Screen organs named by the mixed-representation / down-proj receipts.
    for layer, expert, role in (
        (10, 453, "screen_gate_up_busy"),
        (3, 494, "screen_gate_up_busy"),
        (1, 265, "screen_down_busy"),
        (32, 179, "screen_down_busy"),
        (46, 428, "screen_down_busy"),
        (35, 330, "screen_down_busy"),
    ):
        add(layer, expert, role)

    layers = [0, 23, 47] if smoke else list(range(N_LAYERS))
    extra_low = {0, 23, 47} if smoke else {0, 11, 23, 35, 47}
    for layer in layers:
        row = [(int(e), int(counts[layer, e])) for e in range(N_EXPERTS)]
        nz = [(e, n) for e, n in row if n > 0]
        zero = [(e, n) for e, n in row if n == 0]
        if not nz:
            continue
        nz.sort(key=lambda t: (-t[1], t[0]))
        add(layer, nz[0][0], "busiest")
        add(layer, nz[len(nz) // 2][0], "median")
        if layer in extra_low and len(nz) >= 3:
            add(layer, nz[-1][0], "lowest_nonzero")
        if zero and layer in extra_low:
            add(layer, zero[0][0], "never_routed")
    return pairs


def load_extension_swiglu(ext: Path, layer: int, expert: int) -> np.ndarray | None:
    path = ext / "x" / "swiglu_hidden_routed" / f"L{layer:02d}" / f"E{expert:03d}.f32le"
    if not path.is_file():
        return None
    arr = np.fromfile(path, dtype="<f4")
    if arr.size == 0 or arr.size % INTERMEDIATE != 0:
        return None
    return arr.reshape(-1, INTERMEDIATE).astype(np.float32, copy=False)


def subsample_rows(X: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    n = int(X.shape[0])
    if n <= int(max_rows):
        return X
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_rows), replace=False))
    return np.ascontiguousarray(X[idx], dtype=np.float32)


_WORKER_MODEL: Path | None = None
_WORKER_WMAP: dict[str, str] | None = None


def _init_worker(model_dir: str) -> None:
    global _WORKER_MODEL, _WORKER_WMAP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _WORKER_MODEL = Path(model_dir)
    _WORKER_WMAP = load_weight_map(_WORKER_MODEL)


def _eval_one(job: dict[str, Any]) -> dict[str, Any]:
    layer = int(job["layer"])
    expert = int(job["expert"])
    model_dir = _WORKER_MODEL or Path(job["model_dir"])
    wmap = _WORKER_WMAP or load_weight_map(model_dir)
    X_hid = np.load(job["x_path"]) if job.get("x_path") else None
    if X_hid is not None:
        X_hid = np.ascontiguousarray(X_hid, dtype=np.float32)
        if X_hid.ndim != 2 or X_hid.shape[1] != HIDDEN:
            raise RuntimeError(f"L{layer}.E{expert} hidden shape {X_hid.shape}")
    w_gate = np.asarray(
        load_tensor(model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
        dtype=np.float32,
    )
    w_up = np.asarray(
        load_tensor(model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
        dtype=np.float32,
    )
    w_down = np.asarray(
        load_tensor(model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"),
        dtype=np.float32,
    )
    n_primary = 0 if X_hid is None else int(X_hid.shape[0])
    fit_idx = np.zeros((0,), dtype=np.int64)
    hold_idx = np.zeros((0,), dtype=np.int64)
    has_hold = False
    if n_primary > 0:
        fit_idx, hold_idx, has_hold = holdout_split(
            n_primary, seed=ROW_SEED ^ (layer * 1009 + expert)
        )
    X_fit = X_hid[fit_idx] if n_primary else None
    X_hold = X_hid[hold_idx] if n_primary else None
    X_sw_fit = post_swiglu(X_fit, w_gate, w_up) if X_fit is not None else None
    X_sw_hold = post_swiglu(X_hold, w_gate, w_up) if X_hold is not None else None
    n_ext = 0
    ext_path = job.get("ext_swiglu_path")
    if ext_path and Path(ext_path).is_file():
        ext = np.fromfile(ext_path, dtype="<f4")
        if ext.size >= INTERMEDIATE and ext.size % INTERMEDIATE == 0:
            ext = ext.reshape(-1, INTERMEDIATE).astype(np.float32, copy=False)
            n_ext = int(ext.shape[0])
            X_sw_fit = ext if X_sw_fit is None else np.concatenate([X_sw_fit, ext], axis=0)

    n_fit_gate = 0 if X_fit is None else int(X_fit.shape[0])
    n_fit_down = 0 if X_sw_fit is None else int(X_sw_fit.shape[0])
    n_score_gate = 0 if X_hold is None else int(X_hold.shape[0])
    n_score_down = 0 if X_sw_hold is None else int(X_sw_hold.shape[0])

    def pack_codec(
        name: str,
        W: np.ndarray,
        hat: np.ndarray,
        nbytes: int,
        X_score: np.ndarray | None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scored = _score_hat(
            W,
            hat,
            X_score,
            null_n=int(job["null_n"]),
            seed=NULL_SEED + layer * 10_000 + expert * 17 + _stable_name_seed(name),
        )
        row = {
            "codec": name,
            "payload_bytes": int(nbytes),
            "physical_bpw": physical_bpw(nbytes, int(W.size)),
            "clears_bar": (
                None
                if scored["output_cosine"] is None
                else bool(scored["output_cosine"] >= BAR)
            ),
            **scored,
        }
        if extra:
            row.update(extra)
        return row

    organs: dict[str, Any] = {}
    # --- gate ---
    gate_codecs: list[dict[str, Any]] = []
    for gsz, label in ((128, "binary_g128"), (2048, "binary_g2048")):
        payload, hat = encode_binary(w_gate, group_size=gsz)
        gate_codecs.append(pack_codec(label, w_gate, hat, len(payload), X_hold))
    h_gate = encode_hgravs_ranks(
        w_gate, X_fit, [160, 40, 16, 8], bits=3, identity=CAPTURE_IDENTITY
    )
    for rank, rec in h_gate.items():
        gate_codecs.append(
            pack_codec(
                f"hgravs01_r{rank}_b3",
                w_gate,
                rec["W_hat"],
                rec["payload_bytes"],
                X_hold,
                extra={
                    "achieved_rank": rec["achieved_rank"],
                    "fit_kind": rec["fit_kind"],
                    "rank_clamped_to_n_fit": rec["rank_clamped_to_n_fit"],
                },
            )
        )
    organs["gate_proj"] = {
        "W_shape": [int(w_gate.shape[0]), int(w_gate.shape[1])],
        "x_kind": "router_input",
        "x_dim": HIDDEN,
        "n_rows_primary": n_primary,
        "n_fit": n_fit_gate,
        "n_score": n_score_gate,
        "has_holdout": has_hold,
        "underdetermined_vs_dim": bool(n_fit_gate < HIDDEN),
        "codecs": gate_codecs,
    }

    # --- up ---
    up_codecs: list[dict[str, Any]] = []
    payload, hat = encode_binary(w_up, group_size=128)
    up_codecs.append(pack_codec("binary_g128", w_up, hat, len(payload), X_hold))
    for frac, label in ((0.02, "resid_2pct"), (0.005, "resid_0p5pct")):
        payload, hat = encode_resid(w_up, outlier_ratio=frac)
        up_codecs.append(pack_codec(label, w_up, hat, len(payload), X_hold))
    h_up = encode_hgravs_ranks(
        w_up, X_fit, [160, 40, 16, 8], bits=3, identity=CAPTURE_IDENTITY
    )
    for rank, rec in h_up.items():
        up_codecs.append(
            pack_codec(
                f"hgravs01_r{rank}_b3",
                w_up,
                rec["W_hat"],
                rec["payload_bytes"],
                X_hold,
                extra={
                    "achieved_rank": rec["achieved_rank"],
                    "fit_kind": rec["fit_kind"],
                    "rank_clamped_to_n_fit": rec["rank_clamped_to_n_fit"],
                },
            )
        )
    organs["up_proj"] = {
        "W_shape": [int(w_up.shape[0]), int(w_up.shape[1])],
        "x_kind": "router_input",
        "x_dim": HIDDEN,
        "n_rows_primary": n_primary,
        "n_fit": n_fit_gate,
        "n_score": n_score_gate,
        "has_holdout": has_hold,
        "underdetermined_vs_dim": bool(n_fit_gate < HIDDEN),
        "codecs": up_codecs,
    }

    # --- down ---
    down_codecs: list[dict[str, Any]] = []
    payload, hat = encode_binary(w_down, group_size=128)
    down_codecs.append(pack_codec("binary_g128", w_down, hat, len(payload), X_sw_hold))
    h_down = encode_hgravs_ranks(
        w_down, X_sw_fit, [160, 80, 40, 20, 16, 8], bits=3, identity=CAPTURE_IDENTITY
    )
    for rank, rec in h_down.items():
        down_codecs.append(
            pack_codec(
                f"hgravs01_r{rank}_b3",
                w_down,
                rec["W_hat"],
                rec["payload_bytes"],
                X_sw_hold,
                extra={
                    "achieved_rank": rec["achieved_rank"],
                    "fit_kind": rec["fit_kind"],
                    "rank_clamped_to_n_fit": rec["rank_clamped_to_n_fit"],
                },
            )
        )
    organs["down_proj"] = {
        "W_shape": [int(w_down.shape[0]), int(w_down.shape[1])],
        "x_kind": "swiglu_hidden_routed_recomputed_plus_extension",
        "x_dim": INTERMEDIATE,
        "n_rows_primary": n_primary,
        "n_rows_extension_swiglu": n_ext,
        "n_fit": n_fit_down,
        "n_score": n_score_down,
        "has_holdout": has_hold,
        "underdetermined_vs_dim": bool(n_fit_down < INTERMEDIATE),
        "underdetermined_vs_rank160": bool(n_fit_down < 160),
        "codecs": down_codecs,
    }

    return {
        "layer": layer,
        "expert": expert,
        "role": job["role"],
        "n_rows_primary": n_primary,
        "n_rows_extension_swiglu": n_ext,
        "has_holdout": has_hold,
        "organs": organs,
    }


def summarize_codec(rows: list[dict[str, Any]], organ: str, codec: str) -> dict[str, Any]:
    scored: list[float] = []
    rels: list[float] = []
    bytes_list: list[int] = []
    surplus: list[float] = []
    sep = 0
    n_clear = 0
    n_scored = 0
    n_under = 0
    n_fit_list: list[int] = []
    per_layer: dict[int, list[float]] = {}
    for row in rows:
        org = row["organs"][organ]
        n_fit_list.append(int(org["n_fit"]))
        if org.get("underdetermined_vs_dim"):
            n_under += 1
        hit = next((c for c in org["codecs"] if c["codec"] == codec), None)
        if hit is None:
            continue
        bytes_list.append(int(hit["payload_bytes"]))
        if hit["output_cosine"] is None:
            continue
        n_scored += 1
        scored.append(float(hit["output_cosine"]))
        rels.append(float(hit["output_rel_l2"] or 0.0))
        if hit.get("clears_bar"):
            n_clear += 1
        if hit.get("surplus_over_null_p95") is not None:
            surplus.append(float(hit["surplus_over_null_p95"]))
        if hit.get("separated_from_null_p95"):
            sep += 1
        per_layer.setdefault(int(row["layer"]), []).append(float(hit["output_cosine"]))
    layer_mean = {
        str(L): float(np.mean(v)) for L, v in sorted(per_layer.items())
    }
    layer_rel = {}
    for row in rows:
        hit = next((c for c in row["organs"][organ]["codecs"] if c["codec"] == codec), None)
        if hit and hit.get("output_rel_l2") is not None:
            layer_rel.setdefault(int(row["layer"]), []).append(float(hit["output_rel_l2"]))
    layer_rel_mean = {str(L): float(np.mean(v)) for L, v in sorted(layer_rel.items())}
    growth: list[float] = []
    layers_sorted = sorted(int(k) for k in layer_rel_mean)
    for a, b in zip(layers_sorted, layers_sorted[1:]):
        prev = max(layer_rel_mean[str(a)], 1e-12)
        growth.append(layer_rel_mean[str(b)] / prev)
    geo = (
        float(math.exp(sum(math.log(max(g, 1e-12)) for g in growth) / len(growth)))
        if growth
        else None
    )
    return {
        "organ": organ,
        "codec": codec,
        "n_pairs": len(rows),
        "n_scored": n_scored,
        "n_underdetermined_vs_dim": n_under,
        "frac_underdetermined_vs_dim": float(n_under / max(len(rows), 1)),
        "n_fit": {
            "min": int(min(n_fit_list) if n_fit_list else 0),
            "p50": float(np.median(n_fit_list) if n_fit_list else 0.0),
            "mean": float(np.mean(n_fit_list) if n_fit_list else 0.0),
            "max": int(max(n_fit_list) if n_fit_list else 0),
        },
        "mean_payload_bytes": float(np.mean(bytes_list) if bytes_list else 0.0),
        "physical_bpw_from_mean_bytes": physical_bpw(
            int(round(float(np.mean(bytes_list) if bytes_list else 0.0))),
            ELEMS_PER_EXPERT_ORGAN,
        ),
        "mean_output_cosine": float(np.mean(scored) if scored else float("nan")),
        "p10_output_cosine": float(np.percentile(scored, 10) if scored else float("nan")),
        "p50_output_cosine": float(np.percentile(scored, 50) if scored else float("nan")),
        "p90_output_cosine": float(np.percentile(scored, 90) if scored else float("nan")),
        "min_output_cosine": float(min(scored) if scored else float("nan")),
        "max_output_cosine": float(max(scored) if scored else float("nan")),
        "mean_output_rel_l2": float(np.mean(rels) if rels else float("nan")),
        "frac_clears_bar": float(n_clear / max(n_scored, 1)),
        "frac_separated_from_null_p95": float(sep / max(n_scored, 1)),
        "mean_surplus_over_null_p95": float(np.mean(surplus) if surplus else float("nan")),
        "per_layer_mean_cosine": layer_mean,
        "per_layer_mean_rel_l2": layer_rel_mean,
        "per_layer_rel_l2_growth_ratios": growth,
        "per_layer_rel_l2_geo_growth": geo,
        "clears_bar_on_mean": bool(scored and float(np.mean(scored)) >= BAR),
    }


def compose_point(
    summaries: dict[tuple[str, str], dict[str, Any]],
    *,
    name: str,
    gate: str,
    up: str,
    down: str,
    nonexpert_bits: int,
    nonexpert_bytes: int,
    use_artifact_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    g = summaries[("gate_proj", gate)]
    u = summaries[("up_proj", up)]
    d = summaries[("down_proj", down)]
    gate_bytes = float(g["mean_payload_bytes"]) * N_ROUTED_PER_ORGAN
    up_bytes = float(u["mean_payload_bytes"]) * N_ROUTED_PER_ORGAN
    down_bytes = float(d["mean_payload_bytes"]) * N_ROUTED_PER_ORGAN
    expert_bytes = gate_bytes + up_bytes + down_bytes
    total = int(round(expert_bytes + nonexpert_bytes + ARTIFACT_OVERHEAD_BYTES))
    expert_elems = EXPERT_ELEMS
    expert_bpw = physical_bpw(int(round(expert_bytes)), expert_elems)
    ne_bpw = physical_bpw(int(nonexpert_bytes), NONEXPERT_ELEMS)
    complete = physical_bpw(total, SOURCE_ELEMENTS)
    design = complete_bpw(expert_bpw, ne_bpw)
    organ_cos = {
        "gate_proj": g["mean_output_cosine"],
        "up_proj": u["mean_output_cosine"],
        "down_proj": d["mean_output_cosine"],
    }
    mean_cos = float(np.mean(list(organ_cos.values())))
    min_cos = float(min(organ_cos.values()))
    ratios = increment_ratios_from_growth(_TRUE_RESIDUAL_GROWTH)
    # Conservative residual-identity product uses the worst organ cosine
    # (the bottleneck) against the measured 48-layer increment ratios.
    e2e = residual_identity_product(min_cos, ratios) if math.isfinite(min_cos) else None
    layer_cos_by_depth = []
    for L in range(N_LAYERS):
        vals = []
        for org, codec, key in (
            ("gate_proj", gate, "g"),
            ("up_proj", up, "u"),
            ("down_proj", down, "d"),
        ):
            sm = summaries[(org, codec)]["per_layer_mean_cosine"].get(str(L))
            if sm is not None:
                vals.append(float(sm))
        layer_cos_by_depth.append(None if not vals else float(min(vals)))
    if use_artifact_ledger:
        complete = float(use_artifact_ledger["complete_physical_bpw"])
        expert_bpw = float(use_artifact_ledger["expert_physical_bpw"])
        ne_bpw = float(use_artifact_ledger["nonexpert_physical_bpw"])
        total = int(use_artifact_ledger["all_required_weight_artifact_bytes"])
        design = float(use_artifact_ledger["design_identity_complete_bpw"])
        organ_bpw = use_artifact_ledger["organ_breakdown"]
    else:
        organ_bpw = {
            "routed_gate_proj": {
                "bytes": int(round(gate_bytes)),
                "elements": N_ROUTED_PER_ORGAN * ELEMS_PER_EXPERT_ORGAN,
                "tensors": N_ROUTED_PER_ORGAN,
                "physical_bpw": g["physical_bpw_from_mean_bytes"],
            },
            "routed_up_proj": {
                "bytes": int(round(up_bytes)),
                "elements": N_ROUTED_PER_ORGAN * ELEMS_PER_EXPERT_ORGAN,
                "tensors": N_ROUTED_PER_ORGAN,
                "physical_bpw": u["physical_bpw_from_mean_bytes"],
            },
            "routed_down_proj": {
                "bytes": int(round(down_bytes)),
                "elements": N_ROUTED_PER_ORGAN * ELEMS_PER_EXPERT_ORGAN,
                "tensors": N_ROUTED_PER_ORGAN,
                "physical_bpw": d["physical_bpw_from_mean_bytes"],
            },
            "nonexpert": {
                "bytes": int(nonexpert_bytes),
                "elements": NONEXPERT_ELEMS,
                "physical_bpw": ne_bpw,
                "bits": int(nonexpert_bits),
            },
        }
    return {
        "name": name,
        "recipe": {
            "gate_proj": gate,
            "up_proj": up,
            "down_proj": down,
            "nonexpert_bits": int(nonexpert_bits),
        },
        "complete_physical_bpw": float(complete),
        "design_identity_complete_bpw": float(design),
        "expert_physical_bpw": float(expert_bpw),
        "nonexpert_physical_bpw": float(ne_bpw),
        "all_required_weight_artifact_bytes": int(total),
        "bytes_are": (
            "mixed-1p5-v1 on-disk ledger"
            if use_artifact_ledger
            else "mean measured payload bytes x 24576 tensors + 1p44 catalog overhead"
        ),
        "organ_breakdown": organ_bpw,
        "organ_output_cosine": organ_cos,
        "mean_organ_output_cosine": mean_cos,
        "min_organ_output_cosine": min_cos,
        "clears_bar_all_organs": bool(
            all(math.isfinite(v) and v >= BAR for v in organ_cos.values())
        ),
        "clears_bar_on_mean": bool(math.isfinite(mean_cos) and mean_cos >= BAR),
        "frac_clears_bar": {
            "gate_proj": g["frac_clears_bar"],
            "up_proj": u["frac_clears_bar"],
            "down_proj": d["frac_clears_bar"],
        },
        "frac_separated_from_null_p95": {
            "gate_proj": g["frac_separated_from_null_p95"],
            "up_proj": u["frac_separated_from_null_p95"],
            "down_proj": d["frac_separated_from_null_p95"],
        },
        "mean_surplus_over_null_p95": {
            "gate_proj": g["mean_surplus_over_null_p95"],
            "up_proj": u["mean_surplus_over_null_p95"],
            "down_proj": d["mean_surplus_over_null_p95"],
        },
        "rows_per_fit": {
            "gate_proj": g["n_fit"],
            "up_proj": u["n_fit"],
            "down_proj": d["n_fit"],
        },
        "frac_underdetermined": {
            "gate_proj": g["frac_underdetermined_vs_dim"],
            "up_proj": u["frac_underdetermined_vs_dim"],
            "down_proj": d["frac_underdetermined_vs_dim"],
        },
        "per_layer_min_organ_cosine": layer_cos_by_depth,
        "per_layer_rel_l2_geo_growth": {
            "gate_proj": g["per_layer_rel_l2_geo_growth"],
            "up_proj": u["per_layer_rel_l2_geo_growth"],
            "down_proj": d["per_layer_rel_l2_geo_growth"],
        },
        "residual_identity_e2e_from_worst_organ": e2e,
        "residual_identity_note": (
            "Uses measured 48-layer true residual-stream growth converted by "
            "r=sqrt(max(g^2-1,0)) and the residual-identity product at the "
            "worst organ cosine. This is a composition bound, not teacher-forced "
            "residual-stream probe of the mixed hats. q80-coherence-deep owns "
            "that probe for the 1.44 artifact."
        ),
        "capture_defect": {
            "route_starved": True,
            "p10_rows": 34,
            "p50_rows": 258,
            "never_routed_pairs": 221,
            "no_on_disk_post_swiglu_on_bound_capture": True,
            "typical_gate_up_UNDERDETERMINED": True,
            "typical_down_UNDERDETERMINED": True,
            "extension_swiglu_used_when_present": True,
        },
    }


def pick_targets(
    candidates: list[dict[str, Any]], targets: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Nearest capability-best candidate to each requested complete BPW."""

    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for t in targets:
        best = None
        best_key = None
        for c in candidates:
            if c["name"] in used:
                continue
            dist = abs(float(c["complete_physical_bpw"]) - float(t))
            # Prefer closer BPW; break ties by higher min organ cosine.
            key = (dist, -float(c["min_organ_output_cosine"]))
            if best_key is None or key < best_key:
                best = c
                best_key = key
        if best is None:
            continue
        used.add(best["name"])
        picked.append({**best, "requested_complete_bpw": float(t)})
    return picked


def measure_nonexpert_q4(model_dir: Path, wmap: dict[str, str]) -> dict[str, Any]:
    names = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
    ]
    rows = []
    q8_bytes = 0
    q4_bytes = 0
    elems = 0
    for name in names:
        if name not in wmap:
            continue
        W = np.asarray(load_tensor(model_dir, wmap, name), dtype=np.float32)
        p8, _ = encode_uniform(W, bits=8)
        p4, _ = encode_uniform(W, bits=4)
        q8_bytes += len(p8)
        q4_bytes += len(p4)
        elems += int(W.size)
        rows.append(
            {
                "name": name,
                "elements": int(W.size),
                "q8_bytes": len(p8),
                "q4_bytes": len(p4),
                "q8_bpw": physical_bpw(len(p8), int(W.size)),
                "q4_bpw": physical_bpw(len(p4), int(W.size)),
            }
        )
    scale = (q4_bytes / q8_bytes) if q8_bytes else 0.5
    projected = int(round(ARTIFACT_NONEEXPERT_Q8_BYTES * scale))
    return {
        "sample": rows,
        "sample_q4_over_q8": float(scale),
        "projected_q4_bytes": projected,
        "projected_q4_bpw": physical_bpw(projected, NONEXPERT_ELEMS),
        "q8_artifact_bytes": ARTIFACT_NONEEXPERT_Q8_BYTES,
        "q8_artifact_bpw": ARTIFACT_NONEEXPERT_Q8_BPW,
    }


def load_artifact_ledger(path: Path) -> dict[str, Any] | None:
    report = path / "PACK_REPORT.json"
    if not report.is_file():
        return None
    return json.loads(report.read_text())


def collect_x_for_pairs(
    capture: Path, pairs: list[dict[str, Any]]
) -> dict[tuple[int, int], np.ndarray]:
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        collect_expert_activations,
    )

    wanted = {(int(p["layer"]), int(p["expert"])) for p in pairs}
    by_le, _prov = collect_expert_activations(
        capture,
        wanted_keys=wanted,
        max_rows_per_expert=ROW_CAP,
        row_sample_seed=ROW_SEED,
        use_index=True,
        x_kind="router_input",
    )
    return {k: np.asarray(v, dtype=np.float32) for k, v in by_le.items()}


def run_curve(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    capture = Path(args.capture)
    model_dir = Path(args.model_dir)
    ext = Path(args.ext_capture) if args.ext_capture else None
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    status, root, header = inspect_index(capture)
    if status != "ok" or root is None:
        raise SystemExit(f"capture index not ok: {status}")
    caps = CaptureHiddens(capture)
    pairs = select_pairs(caps.counts, smoke=bool(args.smoke))
    print(f"[curve] selected {len(pairs)} (layer,expert) pairs smoke={args.smoke}", flush=True)

    print("[curve] loading router-input X via capture-index.v1", flush=True)
    by_x = collect_x_for_pairs(capture, pairs)
    jobs = []
    for p in pairs:
        key = (int(p["layer"]), int(p["expert"]))
        x = by_x.get(key)
        x_path = None
        if x is not None and x.size:
            x_path = scratch / f"L{p['layer']:02d}_E{p['expert']:03d}.npy"
            np.save(x_path, np.ascontiguousarray(x, dtype=np.float32))
        ext_path = None
        if ext is not None:
            cand = (
                ext
                / "x"
                / "swiglu_hidden_routed"
                / f"L{p['layer']:02d}"
                / f"E{p['expert']:03d}.f32le"
            )
            if cand.is_file():
                ext_path = str(cand)
        jobs.append(
            {
                "layer": p["layer"],
                "expert": p["expert"],
                "role": p["role"],
                "model_dir": str(model_dir),
                "x_path": str(x_path) if x_path else None,
                "ext_swiglu_path": ext_path,
                "null_n": int(args.null_n),
            }
        )

    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    print(f"[curve] encoding {len(jobs)} pairs workers={workers}", flush=True)
    _init_worker(str(model_dir))
    pool = None
    if workers > 1:
        try:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(str(model_dir),),
            )
            print("[curve] process pool up", flush=True)
        except PermissionError:
            pool = ThreadPoolExecutor(max_workers=workers)
            print("[curve] process pool denied; thread pool fallback", flush=True)
    if pool is None:
        for i, job in enumerate(jobs, 1):
            rec = _eval_one(job)
            rows.append(rec)
            print(
                f"[curve] {i}/{len(jobs)} L{rec['layer']}.E{rec['expert']} "
                f"rows={rec['n_rows_primary']}+ext{rec['n_rows_extension_swiglu']}",
                flush=True,
            )
    else:
        try:
            futs = {pool.submit(_eval_one, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                rows.append(rec)
                done += 1
                if done % 4 == 0 or done == len(futs):
                    print(
                        f"[curve] {done}/{len(futs)} L{rec['layer']}.E{rec['expert']}",
                        flush=True,
                    )
        finally:
            pool.shutdown(wait=True)
    rows.sort(key=lambda r: (r["layer"], r["expert"]))

    wmap = load_weight_map(model_dir)
    ne = measure_nonexpert_q4(model_dir, wmap)
    artifact = load_artifact_ledger(Path(args.artifact_1p44))

    gate_codecs = ["binary_g128", "binary_g2048", "hgravs01_r160_b3", "hgravs01_r40_b3", "hgravs01_r16_b3", "hgravs01_r8_b3"]
    up_codecs = ["binary_g128", "resid_2pct", "resid_0p5pct", "hgravs01_r160_b3", "hgravs01_r40_b3", "hgravs01_r16_b3", "hgravs01_r8_b3"]
    down_codecs = ["hgravs01_r160_b3", "hgravs01_r80_b3", "hgravs01_r40_b3", "hgravs01_r20_b3", "hgravs01_r16_b3", "hgravs01_r8_b3", "binary_g128"]
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for org, codecs in (
        ("gate_proj", gate_codecs),
        ("up_proj", up_codecs),
        ("down_proj", down_codecs),
    ):
        for codec in codecs:
            summaries[(org, codec)] = summarize_codec(rows, org, codec)

    ne8_bytes = ARTIFACT_NONEEXPERT_Q8_BYTES
    ne4_bytes = int(ne["projected_q4_bytes"])
    named = [
        ("mixed_1p44_incumbent", "binary_g128", "resid_2pct", "hgravs01_r160_b3", 8),
        ("mixed_crush_down_r20_ne8", "binary_g128", "binary_g128", "hgravs01_r20_b3", 8),
        ("mixed_crush_down_r8_ne4", "binary_g128", "binary_g128", "hgravs01_r8_b3", 4),
        ("all_hgravs_r40_ne4", "hgravs01_r40_b3", "hgravs01_r40_b3", "hgravs01_r40_b3", 4),
        ("all_hgravs_r16_ne4", "hgravs01_r16_b3", "hgravs01_r16_b3", "hgravs01_r16_b3", 4),
        ("all_hgravs_r8_ne4", "hgravs01_r8_b3", "hgravs01_r8_b3", "hgravs01_r8_b3", 4),
        ("binary_floor_down_r8_ne4", "binary_g2048", "binary_g128", "hgravs01_r8_b3", 4),
    ]
    points: list[dict[str, Any]] = []
    for name, g, u, d, neb in named:
        ledger = None
        if name == "mixed_1p44_incumbent" and artifact:
            ledger = {
                "complete_physical_bpw": artifact["complete_physical_bpw"],
                "expert_physical_bpw": artifact["expert_physical_bpw"],
                "nonexpert_physical_bpw": artifact["nonexpert_physical_bpw"],
                "all_required_weight_artifact_bytes": artifact[
                    "all_required_weight_artifact_bytes"
                ],
                "design_identity_complete_bpw": artifact["design_identity_complete_bpw"],
                "organ_breakdown": artifact["organ_breakdown"],
            }
        points.append(
            compose_point(
                summaries,
                name=name,
                gate=g,
                up=u,
                down=d,
                nonexpert_bits=neb,
                nonexpert_bytes=ne8_bytes if neb == 8 else ne4_bytes,
                use_artifact_ledger=ledger,
            )
        )

    # Full grid for the Pareto / nearest-target pick.
    grid: list[dict[str, Any]] = []
    for g in gate_codecs:
        for u in up_codecs:
            for d in down_codecs:
                for neb, nbytes in ((8, ne8_bytes), (4, ne4_bytes)):
                    grid.append(
                        compose_point(
                            summaries,
                            name=f"{g}|{u}|{d}|ne{neb}",
                            gate=g,
                            up=u,
                            down=d,
                            nonexpert_bits=neb,
                            nonexpert_bytes=nbytes,
                        )
                    )
    targets = pick_targets(grid, TARGET_COMPLETE)

    # Identity arithmetic check against the standing law.
    identity = {
        "stated_in_task": "complete = 0.97032*expert + 0.02968*nonexpert",
        "ledger_coefficients": {
            "f_expert": float(F_EXPERT),
            "f_nonexpert": float(F_NONEXPERT),
        },
        "coefficients_moved": False,
        "current_1p44_artifact": {
            "complete_physical_bpw": None if not artifact else artifact["complete_physical_bpw"],
            "expert_physical_bpw": None if not artifact else artifact["expert_physical_bpw"],
            "nonexpert_physical_bpw": None if not artifact else artifact["nonexpert_physical_bpw"],
            "nonexpert_bytes": ARTIFACT_NONEEXPERT_Q8_BYTES,
            "nonexpert_tensors": 663,
            "nonexpert_mass_fraction": float(F_NONEXPERT),
            "nonexpert_contribution_bpw": float(F_NONEXPERT * ARTIFACT_NONEEXPERT_Q8_BPW),
        },
        "drop_nonexpert_8_to_4_saves_complete_bpw": float(
            F_NONEXPERT * (ARTIFACT_NONEEXPERT_Q8_BPW - ne["projected_q4_bpw"])
        ),
        "sub_0_655_requires_expert_bpw": {
            "with_nonexpert_8bit": float((0.6552 - F_NONEXPERT * ARTIFACT_NONEEXPERT_Q8_BPW) / F_EXPERT),
            "with_nonexpert_4bit": float((0.6552 - F_NONEXPERT * ne["projected_q4_bpw"]) / F_EXPERT),
        },
        "binary_family_floor_note": (
            "binary_group cannot go below ~1.0 BPW. Mixed family "
            "(binary gate/up + any-rank down) therefore floors near "
            "complete ≈ 0.78 with 4-bit non-expert. Sub-0.655 complete "
            "requires leaving the binary family on gate and up."
        ),
    }
    if artifact:
        identity["current_1p44_artifact"]["nonexpert_contribution_bpw"] = float(
            F_NONEXPERT * float(artifact["nonexpert_physical_bpw"])
        )

    cliff = None
    # First requested-target point whose mean organ cosine is below the bar,
    # walking high BPW -> low. If even 1.44 is below, the cliff is at or above 1.44.
    ordered = sorted(points, key=lambda p: -p["complete_physical_bpw"])
    for p in ordered:
        if not p["clears_bar_all_organs"]:
            cliff = {
                "sits_at_or_above_complete_bpw": p["complete_physical_bpw"],
                "first_named_point_failing_all_organs_bar": p["name"],
                "min_organ_output_cosine": p["min_organ_output_cosine"],
                "sub_100_fs_reachable_at_preserved_capability": False,
            }
            break
    if cliff is None:
        cliff = {
            "sits_at_or_above_complete_bpw": None,
            "first_named_point_failing_all_organs_bar": None,
            "sub_100_fs_reachable_at_preserved_capability": True,
            "note": "every named point cleared the 0.8604 bar on all three organs",
        }

    receipt = {
        "schema": SCHEMA,
        "lane": "q80-subbit-capability-curve",
        "status": "CURVE_MEASURED",
        "timing_label": "DIRTY_ENGINEERING",
        "bar": BAR,
        "bar_source": (
            "D23 measured residual-identity break-even from Q80's own "
            "25258-token capture (receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json)"
        ),
        "identity_arithmetic": identity,
        "measurement": {
            "n_pairs": len(rows),
            "smoke": bool(args.smoke),
            "null_n": int(args.null_n),
            "hold_frac": HOLD_FRAC,
            "row_cap": ROW_CAP,
            "workers": workers,
            "wall_secs": time.perf_counter() - started,
            "did_not_parse_giant_json": True,
            "did_not_pack_full_artifacts": True,
            "reconstruction_from_physical_bytes": True,
            "rank_not_clamped_to_n_fit": True,
            "capture": str(capture),
            "extension_capture": str(ext) if ext else None,
            "pairs": [
                {
                    "layer": r["layer"],
                    "expert": r["expert"],
                    "role": r["role"],
                    "n_rows_primary": r["n_rows_primary"],
                    "n_rows_extension_swiglu": r["n_rows_extension_swiglu"],
                    "has_holdout": r["has_holdout"],
                }
                for r in rows
            ],
        },
        "nonexpert_q4_probe": ne,
        "codec_summaries": {
            f"{org}.{codec}": summaries[(org, codec)]
            for org, codec in summaries
        },
        "named_points": points,
        "target_nearest": targets,
        "cliff": cliff,
        "generated_text": None,
        "generation_note": (
            "Autoregressive generation is owned by q80-mixed-generate for the "
            "1.44 artifact. Lower-BPW recipes have no packed full artifact and "
            "no decode kernel. Organ cosine + tiled per-layer rel-L2 is the "
            "capability curve this lane can close without minting 13 GiB copies."
        ),
        "claim_boundary": {
            "artifact_packed_only_at_1p44": True,
            "lower_bpw_bytes_from_measured_payloads_scaled": True,
            "coherence_generation_tested": False,
            "teacher_forced_residual_stream_full_depth": False,
            "teacher_forced_organ_output_full_depth": True,
            "null_is_a_distribution": True,
            "did_not_copy_q30_static_le_1p5": True,
            "did_not_use_cross_expert_shared_basis": True,
        },
        "pair_rows": rows,
    }
    return seal(receipt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--capture", type=Path, default=CAPTURE)
    p.add_argument("--ext-capture", type=Path, default=EXT_CAPTURE)
    p.add_argument("--artifact-1p44", type=Path, default=ARTIFACT_1P44)
    p.add_argument(
        "--scratch",
        type=Path,
        default=Path("/tmp/q80-subbit-capability-curve-scratch"),
    )
    p.add_argument(
        "--receipt",
        type=Path,
        default=REPO / "receipts/ascent-2026-08-16/q80-subbit-capability-curve.json",
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--null-n", type=int, default=NULL_N)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_curve(args)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Drop the bulky per-pair null vectors from the on-disk receipt? Keep them:
    # they are the distribution the law asked for, and this is not a 1.38 GB file.
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[receipt] {args.receipt}", flush=True)
    slim = {
        "cliff": receipt.get("cliff"),
        "named_points": [
            {
                "name": p["name"],
                "complete_physical_bpw": p["complete_physical_bpw"],
                "expert_physical_bpw": p["expert_physical_bpw"],
                "organ_output_cosine": p["organ_output_cosine"],
                "clears_bar_all_organs": p["clears_bar_all_organs"],
                "residual_identity_e2e_from_worst_organ": p[
                    "residual_identity_e2e_from_worst_organ"
                ],
            }
            for p in receipt.get("named_points", [])
        ],
        "target_nearest": [
            {
                "requested": p.get("requested_complete_bpw"),
                "name": p["name"],
                "complete_physical_bpw": p["complete_physical_bpw"],
                "min_organ_output_cosine": p["min_organ_output_cosine"],
                "clears_bar_all_organs": p["clears_bar_all_organs"],
            }
            for p in receipt.get("target_nearest", [])
        ],
        "identity": receipt.get("identity_arithmetic"),
        "wall_secs": receipt.get("measurement", {}).get("wall_secs"),
    }
    print(json.dumps(slim, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
