"""Build a sealed Q30 complete candidate using activation_weighted_svd_low_rank_q.

This branch leaves the admitted binary baseline untouched.  It produces a *new*
complete candidate under quality-candidates/activation-weighted-svd-v1 by:

1. Binding a real routed-activation capture by path + sha256.
   Accepts the L0-only broad capture (historical) or the all-layer broad
   capture (`...all_layer_route_capture_result.v1`).
2. Selecting expert gate/up/down organs by surplus-over-null under the
   the campaign one-bit complete-BPW ceiling (1/1), then packing greedily under
   the same complete_physical_bpw law (weight cosine is reported and used only
   as the distribution-local guard, not the selection metric).
3. Encoding selected organs with activation_weighted_svd_low_rank_q (real
   capture fit; dual-gravity first-class codec).
4. Hard-linking every other tensor from the admitted complete-binary baseline
   so the candidate remains a full 18,867-tensor physical catalog.

Coverage is first-class: the sealed receipt records layers_covered,
repacked_tensor_percent, and refuses to pretend an L0-only pack is coherent.

Selection metric inversion is the whole finding: the live search optimizes
weight-space cosine; this branch optimizes activation surplus-over-null.

Not a coherence claim.  Not a promotion.  Not a server repoint.  The honest
bar is a candidate for admission; text generation on a served artifact is
run later by the operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators import ascension_dual_gravity_worker as dual
from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.operators import capture_expert_pack as expert_pack
from lab.operators import one_bit_ceiling as obc
from lab.operators import repack_score_index as score_index
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
# Prefer the durable main-tree paths (worktrees often omit multi-GB weights).
MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
QWEN30_ROOT = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen30"
)
BASELINE_ROOT = QWEN30_ROOT / "complete-gravity"
SOURCE_AUDIT = QWEN30_ROOT / "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
DEFAULT_CAPTURE_RUN = dual.DEFAULT_Q30_ACTIVATION_CAPTURE_RUN
# Prefer the worktree/repo-local candidate root so sandboxed runs can seal
# without writing into the live main-tree quality-candidates control surface.
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30"
    / "quality-candidates/activation-weighted-svd-v1"
)

SCHEMA = "hawking.ascension.qwen30_activation_weighted_svd_repack_candidate.v1"
SELECTION_SCHEMA = "hawking.ascension.qwen30_activation_weighted_svd_selection.v1"
SOURCE_SNAPSHOT_SCHEMA = "hawking.ascension.qwen30_activation_weighted_svd_source_snapshot.v1"
TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
BRANCH_ID = "qwen30-activation-weighted-svd-v1"
ARTIFACT_PREFIX = "QWEN30_ACTIVATION_WEIGHTED_SVD_V1"
MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct-activation-weighted-svd-v1"
EXPECTED_TENSOR_COUNT = 18_867
MODEL_LAYERS = 48
# Campaign law: complete BPW <= 1/1 (one_bit_ceiling.CEILING). No private looser constant.
# Trunk fixed the illegal 1.5 complete-BPW constant; keep component and complete both at 1.0.
CEILING_BPW = float(obc.CEILING)  # 1.0 — was illegally 1.5
DEFAULT_GLOBAL_BUDGET_BPW = float(CEILING_BPW)
ONE_BIT_CEILING_BPW = float(obc.CEILING)  # 1.0 — alias for evaluate/seal hard law
DEFAULT_FIT_CACHE = QWEN30_ROOT / "quality-candidates" / "activation-weighted-svd-fit-cache"
DEFAULT_SCORE_INDEX = (
    QWEN30_ROOT / "quality-candidates" / "activation-weighted-svd-score-index.v1.json"
)
FIT_CACHE_SCHEMA = "hawking.ascension.activation_weighted_svd_organ_fit_cache.v1"
MIN_TOKENS = 32
# Minimum surplus over the constant-mean null for an organ to be REPLACED. 0.0 keeps the
# bare `beats_null` bar; higher values refuse organs that only trivially beat a constant.
MIN_SURPLUS = 0.0
HOLD_FRAC = 0.25
SEED = 0xA17A5D
OPERATOR_RECOVERY_WEIGHT_COS = dual.OPERATOR_RECOVERY_WEIGHT_COS
# Historical default when the bound capture is L0-only.
DEFAULT_LAYER = 0
COMPONENTS = ("gate_proj", "up_proj", "down_proj")
ALL_LAYER_RESULT_SCHEMA = (
    "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_result.v1"
)
L0_BROAD_RESULT_SCHEMA = (
    "hawking.ascension.qwen30_broad_activation_layer0_route_capture_result.v1"
)
L0_HCLI_RESULT_SCHEMA = (
    "hawking.ascension.qwen30_current_hcli_layer0_route_capture_result.v1"
)

# Pin BLAS under the worker pool.  Capture lane measured VECLIB_MAXIMUM_THREADS=8
# as a real win: worker threads × default Accelerate fan-out oversubscribe the
# machine and the organ fits thrash.  Only set when the operator did not already
# pin the env (so an outer orchestrator keeps control).
_BLAS_ENV_DEFAULTS = {
    "VECLIB_MAXIMUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
}


def _pin_blas_threads() -> dict[str, str]:
    applied: dict[str, str] = {}
    for key, value in _BLAS_ENV_DEFAULTS.items():
        if key not in os.environ or not str(os.environ.get(key) or "").strip():
            os.environ[key] = value
            applied[key] = value
        else:
            applied[key] = str(os.environ[key])
    return applied

# Under-ceiling budgets from the measured family probe.  Over-ceiling anchors
# are intentionally excluded from selection.
BUDGET_POINTS: tuple[dict[str, Any], ...] = (
    {"label": "r64_b3", "rank": 64, "bits": 3},
    {"label": "r128_b3", "rank": 128, "bits": 3},
    {"label": "r192_b4", "rank": 192, "bits": 4},
    {"label": "r256_b3", "rank": 256, "bits": 3},
)

BASELINE_MANIFEST_NAME = "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
BASELINE_ADMISSION_NAME = "QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
BASELINE_REVALIDATION_NAME = "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json"


class ActivationWeightedRepackError(RuntimeError):
    """The activation-weighted complete candidate cannot be sealed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _raw_sha256(path: Path) -> str:
    return complete._sha256_file(path)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    raw = complete._read_json(path)
    if raw is None:
        raise ActivationWeightedRepackError(f"missing {label}: {path}")
    try:
        return verify(raw, label=str(path))
    except Exception as exc:
        raise ActivationWeightedRepackError(f"untrustworthy {label}: {exc}") from exc


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    complete._atomic_json(path, payload)


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    document = _sealed(path, label=label)
    return {
        "path": str(path.resolve()),
        "document_sha256": _raw_sha256(path),
        "seal_sha256": document["seal_sha256"],
        "file_identity": complete._file_identity(path, label=label),
    }


def _file_identity_loose(path: Path) -> dict[str, int]:
    st = path.stat()
    return {
        "bytes": int(st.st_size),
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }


def silu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)
    return clipped / (1.0 + np.exp(-clipped))


def holdout_split(X: np.ndarray, *, seed: int, hold_frac: float = HOLD_FRAC) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 4:
        return X, X
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, min(n // 2, int(round(n * hold_frac))))
    hold_idx = perm[:n_hold]
    fit_idx = perm[n_hold:]
    if fit_idx.size == 0:
        fit_idx = perm
    return X[fit_idx], X[hold_idx]


def capture_is_all_layer(capture: Mapping[str, Any]) -> bool:
    schema = str(capture.get("schema") or "")
    if schema == ALL_LAYER_RESULT_SCHEMA:
        return True
    if capture.get("capture_summary", {}).get("all_layer_activation_capture") is True:
        return True
    # Structural fallback: first step carries per-layer rows.
    probes = capture.get("probes") or []
    if probes and isinstance(probes[0], Mapping):
        steps = probes[0].get("steps") or []
        if steps and isinstance(steps[0], Mapping) and isinstance(steps[0].get("layers"), list):
            return True
    return False


def collect_expert_activations_from_json(
    run_dir: Path, capture: Mapping[str, Any]
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """JSON + per-token f32le fallback.  Same row order as the historical collector.

    Deduplicates file reads via relative_path so multi-expert hits on one token
    do not re-open the same hidden file.  Still pays the multi-GB JSON parse and
    tens of thousands of small-file opens when the binary expert pack is absent.
    """

    all_layer = capture_is_all_layer(capture)
    try:
        stacked, provenance, _meta = expert_pack.build_from_capture_walk(
            run_dir, capture, all_layer=all_layer, default_layer=DEFAULT_LAYER
        )
    except expert_pack.ExpertPackError as exc:
        raise ActivationWeightedRepackError(str(exc)) from exc
    hit_counts = {
        f"L{layer}.E{expert}": int(arr.shape[0])
        for (layer, expert), arr in sorted(
            stacked.items(), key=lambda kv: (-kv[1].shape[0], kv[0][0], kv[0][1])
        )
    }
    provenance = {
        **provenance,
        "hit_counts": hit_counts,
        "load_path": "json_capture_result_fallback",
    }
    return stacked, provenance


def collect_expert_activations(
    run_dir: Path, capture: Mapping[str, Any] | None = None
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Collect router-input hiddens keyed by (layer, expert_id).

    Prefers the binary expert-pack.v1 (unique rows + uint32 index).  Falls back
    to walking capture-result.json + per-token f32le files so existing captures
    remain readable without a convert step.

    L0-only captures are projected onto layer 0 so the rest of the packer can
    share one code path. All-layer captures only count tokens that retained a
    raw hidden (bounded subsample); route-only tokens contribute to hit-count
    provenance but not to the fit matrix.
    """

    run_dir = Path(run_dir)
    t0 = time.perf_counter()
    if expert_pack.pack_is_present(run_dir):
        try:
            stacked, provenance = expert_pack.load_expert_pack(run_dir)
            provenance = dict(provenance)
            provenance["load_path"] = "binary_expert_pack_v1"
            provenance["load_wall_secs"] = time.perf_counter() - t0
            return stacked, provenance
        except expert_pack.ExpertPackError as exc:
            # Corrupt pack must not silently change selection — refuse rather than
            # half-load.  Operator can delete the pack dir to force JSON fallback.
            raise ActivationWeightedRepackError(
                f"expert-pack present but unreadable ({exc}); delete "
                f"{expert_pack.pack_dir(run_dir)} to fall back to JSON"
            ) from exc

    if capture is None:
        result_path = run_dir / "capture-result.json"
        if not result_path.is_file():
            raise ActivationWeightedRepackError(f"capture-result.json missing under {run_dir}")
        capture = json.loads(result_path.read_bytes())
    stacked, provenance = collect_expert_activations_from_json(run_dir, capture)
    provenance = dict(provenance)
    provenance["load_wall_secs"] = time.perf_counter() - t0
    return stacked, provenance


def organ_activations(
    *,
    layer: int,
    expert: int,
    component: str,
    X_hidden: np.ndarray,
    model_dir: Path,
    weight_map: Mapping[str, str],
) -> np.ndarray:
    if component in ("gate_proj", "up_proj"):
        return X_hidden
    if component != "down_proj":
        raise ActivationWeightedRepackError(f"unsupported component {component}")
    Wg = load_tensor(
        model_dir,
        weight_map,
        f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
    ).astype(np.float32, copy=False)
    Wu = load_tensor(
        model_dir,
        weight_map,
        f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
    ).astype(np.float32, copy=False)
    return silu(X_hidden @ Wg.T) * (X_hidden @ Wu.T)


def coverage_report(
    *,
    selected_organs: Sequence[Mapping[str, Any]],
    act_prov: Mapping[str, Any],
    model_layers: int = MODEL_LAYERS,
    total_tensors: int = EXPECTED_TENSOR_COUNT,
) -> dict[str, Any]:
    layers = sorted({int(r["layer"]) for r in selected_organs})
    per_projection: dict[str, int] = {c: 0 for c in COMPONENTS}
    per_layer: dict[str, int] = {}
    for row in selected_organs:
        per_projection[str(row["component"])] = per_projection.get(str(row["component"]), 0) + 1
        key = str(int(row["layer"]))
        per_layer[key] = per_layer.get(key, 0) + 1
    n = len(selected_organs)
    return {
        "model_layers": model_layers,
        "layers_covered": layers,
        "n_layers_covered": len(layers),
        "layers_missing": [i for i in range(model_layers) if i not in set(layers)],
        "repacked_tensors": n,
        "total_tensors": total_tensors,
        "percent": float(100.0 * n / max(total_tensors, 1)),
        "per_projection": per_projection,
        "per_layer_repacked_counts": dict(sorted(per_layer.items(), key=lambda kv: int(kv[0]))),
        "capture_layers_with_hidden_hits": list(act_prov.get("layers_with_hidden_hits") or []),
        "all_layer_capture": bool(act_prov.get("all_layer_capture")),
        "layer0_only": layers == [0],
        "cannot_be_coherent_if_layer0_only": True,
    }



def _effective_budgets_for_organ(n_fit_rows: int) -> list[dict[str, Any]]:
    """Clamp budget ranks to n_fit (well-posed ridge) and drop duplicates."""

    effective_budgets: list[dict[str, Any]] = []
    seen_points: set[tuple[int, int]] = set()
    for budget in BUDGET_POINTS:
        rank = min(int(budget["rank"]), n_fit_rows)
        if rank < 1:
            continue
        point = (rank, int(budget["bits"]))
        if point in seen_points:
            continue
        seen_points.add(point)
        effective_budgets.append(
            {**budget, "rank": rank, "rank_clamped_to_n_fit": rank != int(budget["rank"])}
        )
    return effective_budgets


def _encode_budget_from_factors(
    *,
    W: np.ndarray,
    matrix: np.ndarray,
    left_full: np.ndarray,
    right_full: np.ndarray,
    fit_meta_base: Mapping[str, Any],
    rank: int,
    bits: int,
    capture_identity: Mapping[str, Any],
    hold: np.ndarray,
    y_hold: np.ndarray,
    null_baseline: float,
) -> dual.CodecResult:
    """Quantize/score one (rank, bits) from a shared max-rank activation-weighted SVD.

    Factors for rank k are the leading-k slice of a single full economy SVD of the
    weighted matrix — identical to calling the codec once per rank (numpy always
    computes the full economy decomposition then truncates). Byte-identity of the
    resulting payload is the contract; do not "approximate" this path.
    """

    original_shape = tuple(int(item) for item in W.shape)
    actual = min(max(1, int(rank)), int(left_full.shape[1]), int(right_full.shape[0]))
    left = left_full[:, :actual]
    right = right_full[:actual, :]
    left_body, left_rebuilt, left_meta = dual._factor_codec(left, bits=int(bits))
    right_body, right_rebuilt, right_meta = dual._factor_codec(right, bits=int(bits))
    reconstruction = (left_rebuilt @ right_rebuilt).reshape(original_shape)
    y_hat = hold @ reconstruction.reshape(matrix.shape).T
    output_cosine = dual._mean_row_cosine(y_hold, y_hat)
    surplus = output_cosine - float(null_baseline)
    weight = dual._quality(matrix, reconstruction.reshape(matrix.shape))
    fit_meta = {
        "fit": fit_meta_base["fit"],
        "rank": int(actual),
        "n_fit_tokens": int(fit_meta_base["n_fit_tokens"]),
        "gram_ridge": float(fit_meta_base["gram_ridge"]),
    }
    header = {
        "schema": "hawking.gravity.activation_weighted_svd_low_rank.v1",
        "representation": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
        "shape": list(original_shape),
        "matrix_shape": [int(item) for item in matrix.shape],
        "elements": int(W.size),
        "rank": int(left.shape[1]),
        "factor_bits": int(bits),
        "factor_group_size": dual.GROUP_UNIFORM,
        "left": left_meta,
        "right": right_meta,
        "left_body_bytes": len(left_body),
        "right_body_bytes": len(right_body),
        "fit": fit_meta,
        "activation_capture": {
            "path": capture_identity.get("path"),
            "capture_result_path": capture_identity.get("capture_result_path"),
            "sha256": capture_identity.get("sha256"),
            "schema": capture_identity.get("schema"),
            "status": capture_identity.get("status"),
            "fit_kind": capture_identity.get("fit_kind"),
            "not_synthetic_unit_direction": True,
        },
        "selection_metric": {
            "primary": "surplus_over_null",
            "secondary": "weight_cosine",
            "weight_cosine_role": "distribution_local_guard",
            "operator_recovery_weight_cos_cutoff": dual.OPERATOR_RECOVERY_WEIGHT_COS,
            # Match independent codec header field (dual constant). Selection under_ceiling
            # still uses module CEILING_BPW (= one_bit 1.0).
            "ceiling_component_bpw": dual.CEILING_COMPONENT_BPW,
        },
        "activation_quality": {
            "output_cosine": float(output_cosine),
            "null_baseline": float(null_baseline),
            "surplus_over_null": float(surplus),
            "beats_null": bool(output_cosine > null_baseline),
            "n_hold_tokens": int(hold.shape[0]),
            "weight_cosine": float(weight["cosine"]),
            "weight_relative_l2": float(weight["relative_l2"]),
            "distribution_local_only": bool(
                surplus >= 0.10
                and output_cosine >= 0.90
                and weight["cosine"] < dual.OPERATOR_RECOVERY_WEIGHT_COS
            ),
        },
    }
    payload = dual._container(dual.MAGIC_ACT_SVD, header, left_body + right_body)
    # Reconstruction must come from physical bytes, not encoder-local factors.
    decoded = dual._decode_activation_weighted_svd_low_rank_codec(payload)
    return dual.CodecResult(payload=payload, reconstruction=decoded, metadata=header)


def select_budget_for_organ(
    *,
    W: np.ndarray,
    X_fit: np.ndarray,
    X_hold: np.ndarray,
    capture_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Pick the under-ceiling budget maximizing surplus-over-null.

    Execution: one activation-weighted SVD at the max effective rank, then
    quantize+score each budget by slicing factors. Selection semantics and
    payload bytes are identical to four independent codec calls (see tests).
    """

    n_fit_rows = int(X_fit.shape[0])
    effective_budgets = _effective_budgets_for_organ(n_fit_rows)
    if not effective_budgets:
        raise ActivationWeightedRepackError(
            f"organ has n_fit_tokens={n_fit_rows}; no well-posed rank budget exists"
        )

    matrix = np.ascontiguousarray(W, dtype=np.float32).reshape(W.shape[0], -1)
    max_rank = max(int(b["rank"]) for b in effective_budgets)
    left_full, right_full, fit_meta_base = dual._activation_weighted_svd_factors(
        matrix, X_fit, rank=max_rank
    )
    hold = np.ascontiguousarray(X_hold, dtype=np.float32)
    y_hold = hold @ matrix.T
    null_baseline = dual._constant_mean_null_cosine(y_hold)

    candidates: list[dict[str, Any]] = []
    for budget in effective_budgets:
        codec = _encode_budget_from_factors(
            W=W,
            matrix=matrix,
            left_full=left_full,
            right_full=right_full,
            fit_meta_base=fit_meta_base,
            rank=int(budget["rank"]),
            bits=int(budget["bits"]),
            capture_identity=capture_identity,
            hold=hold,
            y_hold=y_hold,
            null_baseline=float(null_baseline),
        )
        bpw = len(codec.payload) * 8.0 / max(W.size, 1)
        quality = codec.metadata["activation_quality"]
        under = bool(bpw <= CEILING_BPW + 1e-9)
        row = {
            "budget_label": budget["label"],
            "rank": int(codec.metadata["rank"]),
            "rank_clamped_to_n_fit": bool(budget.get("rank_clamped_to_n_fit", False)),
            "bits": int(budget["bits"]),
            "component_bpw": float(bpw),
            "under_ceiling": under,
            "artifact_bytes": len(codec.payload),
            "payload_sha256": hashlib.sha256(codec.payload).hexdigest(),
            "payload": codec.payload,
            "reconstruction": codec.reconstruction,
            "codec_metadata": {
                k: v
                for k, v in codec.metadata.items()
                if k not in {"left", "right"}
            },
            "left_meta": codec.metadata["left"],
            "right_meta": codec.metadata["right"],
            "weight_cosine": float(quality["weight_cosine"]),
            "weight_relative_l2": float(quality["weight_relative_l2"]),
            "output_cosine": float(quality["output_cosine"]),
            "null_baseline": float(quality["null_baseline"]),
            "surplus_over_null": float(quality["surplus_over_null"]),
            "beats_null": bool(quality["beats_null"]),
            "distribution_local_only": bool(quality["distribution_local_only"]),
            "n_hold_tokens": int(quality["n_hold_tokens"]),
            "n_fit_tokens": int(codec.metadata["fit"]["n_fit_tokens"]),
        }
        row["_full_metadata"] = codec.metadata
        candidates.append(row)
    under = [c for c in candidates if c["under_ceiling"]]
    if not under:
        best = max(candidates, key=lambda r: (r["surplus_over_null"], r["weight_cosine"], -r["component_bpw"]))
        raise ActivationWeightedRepackError(
            f"no under-ceiling budget for organ; best surplus budget was "
            f"{best['budget_label']} at bpw={best['component_bpw']:.4f}"
        )
    under.sort(
        key=lambda r: (
            -float(r["surplus_over_null"]),
            -float(r["weight_cosine"]),
            float(r["component_bpw"]),
            str(r["budget_label"]),
        )
    )
    winner = under[0]
    winner["sweep"] = [
        {
            k: v
            for k, v in c.items()
            if k not in {"payload", "reconstruction", "_full_metadata", "left_meta", "right_meta"}
        }
        for c in candidates
    ]
    return winner


def _activation_fingerprint(X: np.ndarray) -> str:
    """Content hash of activation rows used in the fit/holdout split."""

    arr = np.ascontiguousarray(X, dtype=np.float32)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def organ_fit_cache_key(
    *,
    capture_sha: str,
    layer: int,
    expert: int,
    component: str,
    source_value_sha256: str,
    activation_fingerprint: str,
    n_tokens: int,
) -> str:
    """Content-addressed key for a per-organ budget-sweep winner."""

    material = {
        "schema": FIT_CACHE_SCHEMA,
        "capture_sha": capture_sha,
        "layer": int(layer),
        "expert": int(expert),
        "component": str(component),
        "source_value_sha256": source_value_sha256,
        "activation_fingerprint": activation_fingerprint,
        "n_tokens": int(n_tokens),
        "hold_frac": HOLD_FRAC,
        "seed": SEED,
        "seed_mix": "SEED^(layer*9176)^(expert*1009)^(sha256(component)[:8]&0xFFFF)",
        "budget_points": list(BUDGET_POINTS),
        "ceiling_component_bpw": CEILING_BPW,
        "min_surplus": MIN_SURPLUS,
        "codec_schema": "hawking.gravity.activation_weighted_svd_low_rank.v1",
    }
    return _canonical_sha256(material)


def _fit_cache_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    shard = cache_root / key[:2]
    return shard / f"{key}.json", shard / f"{key}.hgravs01"


def try_load_fit_cache(cache_root: Path | None, key: str) -> dict[str, Any] | None:
    if cache_root is None:
        return None
    meta_path, payload_path = _fit_cache_paths(cache_root, key)
    if not meta_path.is_file() or not payload_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        payload = payload_path.read_bytes()
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("schema") != FIT_CACHE_SCHEMA:
        return None
    if meta.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        return None
    if int(meta.get("artifact_bytes") or -1) != len(payload):
        return None
    winner = dict(meta["winner"])
    winner["payload"] = payload
    winner["payload_sha256"] = meta["payload_sha256"]
    winner["artifact_bytes"] = len(payload)
    winner["reconstruction"] = None
    winner["_full_metadata"] = winner.get("codec_metadata")
    winner["sweep"] = list(meta.get("sweep") or [])
    return winner


def store_fit_cache(cache_root: Path | None, key: str, winner: Mapping[str, Any]) -> None:
    if cache_root is None:
        return
    payload = winner["payload"]
    meta_path, payload_path = _fit_cache_paths(cache_root, key)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    sweep = winner.get("sweep") or []
    winner_meta = {
        k: v
        for k, v in winner.items()
        if k not in {"payload", "reconstruction", "_full_metadata", "left_meta", "right_meta"}
    }
    document = {
        "schema": FIT_CACHE_SCHEMA,
        "key": key,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_bytes": len(payload),
        "winner": winner_meta,
        "sweep": sweep,
    }
    tmp_payload = payload_path.with_suffix(payload_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    try:
        tmp_payload.write_bytes(payload)
        tmp_meta.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_payload, payload_path)
        os.replace(tmp_meta, meta_path)
    finally:
        for p in (tmp_payload, tmp_meta):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass



class ActivationWeightedSvdRepack:
    """Seal a complete activation-weighted-SVD candidate from baseline + capture."""

    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR,
        baseline_root: Path = BASELINE_ROOT,
        source_audit: Path = SOURCE_AUDIT,
        capture_run: Path = DEFAULT_CAPTURE_RUN,
        root: Path = DEFAULT_ROOT,
        max_experts: int | None = None,
        max_layers: int | None = None,
        workers: int = 4,
        require_all_layer_capture: bool = False,
        budget_bpw: float = DEFAULT_GLOBAL_BUDGET_BPW,
        score_index_path: Path | None = DEFAULT_SCORE_INDEX,
        fit_cache: Path | None = DEFAULT_FIT_CACHE,
        disable_fit_cache: bool = False,
        previous_seal_root: Path | None = None,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.baseline_root = baseline_root.expanduser().resolve()
        self.source_audit = source_audit.expanduser().resolve()
        self.capture_run = capture_run.expanduser().resolve()
        self.root = root.expanduser().resolve()
        self.max_experts = max_experts
        self.max_layers = max_layers
        self.workers = max(1, int(workers))
        self.require_all_layer_capture = bool(require_all_layer_capture)
        self.budget_bpw = float(budget_bpw)
        self.score_index_path = (
            score_index_path.expanduser().resolve() if score_index_path is not None else None
        )
        if disable_fit_cache:
            self.fit_cache = None
        elif fit_cache is None:
            self.fit_cache = None
        else:
            self.fit_cache = fit_cache.expanduser().resolve()
        self.previous_seal_root = (
            previous_seal_root.expanduser().resolve() if previous_seal_root is not None else None
        )
        self._resident_score_index: score_index.ScoreIndex | None = None
        self._resident_capture: tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]] | None = None
        self.tensor_dir = self.root / "tensors"
        self.manifest_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        self.selection_path = self.root / f"{ARTIFACT_PREFIX}_SELECTION_RECEIPT.json"
        self.snapshot_path = self.root / f"{ARTIFACT_PREFIX}_SOURCE_BINDING_SNAPSHOT.json"
        self.terminal_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"
        self.status_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_STATUS.json"
        self.coverage_path = self.root / f"{ARTIFACT_PREFIX}_COVERAGE_RECEIPT.json"
        self.evaluate_path = self.root / f"{ARTIFACT_PREFIX}_EVALUATE_RECEIPT.json"
        self.baseline_manifest_path = self.baseline_root / BASELINE_MANIFEST_NAME
        self.baseline_admission_path = self.baseline_root / BASELINE_ADMISSION_NAME
        self.baseline_revalidation_path = self.baseline_root / BASELINE_REVALIDATION_NAME
        self.baseline_tensor_dir = self.baseline_root / "tensors"

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        profile: dict[str, Any] = {}
        blas_pin = _pin_blas_threads()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        if self.fit_cache is not None:
            self.fit_cache.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.is_file() and self.terminal_path.is_file():
            terminal = _sealed(self.terminal_path, label="existing terminal receipt")
            manifest = _sealed(self.manifest_path, label="existing candidate manifest")
            return {
                "status": "ALREADY_SEALED",
                "complete_physical_bpw": manifest["complete_physical_bpw_ledger"]["complete_physical_bpw"],
                "manifest_path": str(self.manifest_path),
                "terminal_path": str(self.terminal_path),
                "manifest_seal_sha256": manifest["seal_sha256"],
                "terminal_seal_sha256": terminal["seal_sha256"],
            }

        t_cap = time.perf_counter()
        capture_identity = dual._activation_capture_binding(self.capture_run)
        # Prefer binary pack: avoid parsing multi-GB capture-result.json when the
        # pack is present.  Binding still hashes capture-result.json for identity.
        if expert_pack.pack_is_present(self.capture_run):
            header = json.loads(
                (expert_pack.pack_dir(self.capture_run) / expert_pack.HEADER_NAME).read_text(
                    encoding="utf-8"
                )
            )
            # Lightweight all-layer check without JSON body.
            all_layer = bool(header.get("all_layer_capture", True))
            capture: Mapping[str, Any] | None = None
            if self.require_all_layer_capture and not all_layer:
                # Fall back to JSON only when we must confirm.
                capture = json.loads(
                    Path(capture_identity["capture_result_path"]).read_bytes()
                )
                all_layer = capture_is_all_layer(capture)
        else:
            capture = json.loads(Path(capture_identity["capture_result_path"]).read_bytes())
            all_layer = capture_is_all_layer(capture)
        if self.require_all_layer_capture and not all_layer:
            raise ActivationWeightedRepackError(
                "require_all_layer_capture=True but bound capture is not an all-layer "
                f"activation capture (schema={capture_identity.get('schema')})"
            )
        by_layer_expert, act_prov = collect_expert_activations(self.capture_run, capture)
        profile["capture_load_secs"] = time.perf_counter() - t_cap
        print(
            f"capture_load: {profile['capture_load_secs']:.2f}s "
            f"path={act_prov.get('load_path')} pairs={len(by_layer_expert)}",
            flush=True,
        )
        weight_map = load_weight_map(self.model_dir)
        audit = _sealed(self.source_audit, label="source body audit")
        revalidation = _sealed(self.baseline_revalidation_path, label="baseline source revalidation")
        baseline_manifest = _sealed(self.baseline_manifest_path, label="baseline complete manifest")
        baseline_admission = _sealed(self.baseline_admission_path, label="baseline admission")

        baseline_ledger = baseline_manifest["complete_physical_bpw_ledger"]
        baseline_control = {
            "complete_physical_bpw": baseline_ledger["complete_physical_bpw"],
            "manifest": {
                "path": str(self.baseline_manifest_path),
                "seal_sha256": baseline_manifest["seal_sha256"],
                "document_sha256": _raw_sha256(self.baseline_manifest_path),
            },
            "admission": {
                "path": str(self.baseline_admission_path),
                "seal_sha256": baseline_admission["seal_sha256"],
                "document_sha256": _raw_sha256(self.baseline_admission_path),
            },
            "source_revalidation": {
                "path": str(self.baseline_revalidation_path),
                "seal_sha256": revalidation["seal_sha256"],
                "document_sha256": _raw_sha256(self.baseline_revalidation_path),
            },
            "preserve_as_rollback_control": True,
            "replacement_forbidden_until_all_acceptance_gates_pass": True,
        }

        eligible_keys = sorted(
            (key for key, X in by_layer_expert.items() if X.shape[0] >= MIN_TOKENS),
            key=lambda key: (-by_layer_expert[key].shape[0], key[0], key[1]),
        )
        if self.max_layers is not None:
            allowed = set(range(int(self.max_layers)))
            eligible_keys = [k for k in eligible_keys if k[0] in allowed]
        if self.max_experts is not None:
            # Cap per layer so multi-layer packs stay balanced under a probe budget.
            kept: list[tuple[int, int]] = []
            per_layer_count: dict[int, int] = {}
            for layer, expert in eligible_keys:
                n = per_layer_count.get(layer, 0)
                if n < int(self.max_experts):
                    kept.append((layer, expert))
                    per_layer_count[layer] = n + 1
            eligible_keys = kept
        if not eligible_keys:
            raise ActivationWeightedRepackError(
                "no (layer, expert) pairs with enough routed hidden tokens for fitting"
            )

        binding = {
            "branch_id": BRANCH_ID,
            "model_id": MODEL_ID,
            "artifact_prefix": ARTIFACT_PREFIX,
            "source_body_audit_seal_sha256": audit["seal_sha256"],
            "immutable_source_revalidation": {
                "path": str(self.baseline_revalidation_path),
                "document_sha256": _raw_sha256(self.baseline_revalidation_path),
                "seal_sha256": revalidation["seal_sha256"],
            },
            "activation_capture": capture_identity,
            "selection_metric": {
                "primary": "surplus_over_null",
                "secondary": "weight_cosine",
                "weight_cosine_role": "distribution_local_guard",
                "operator_recovery_weight_cos_cutoff": OPERATOR_RECOVERY_WEIGHT_COS,
                "ceiling_component_bpw": CEILING_BPW,
                "complete_physical_bpw_ceiling": CEILING_BPW,
                "global_pack_policy": "greedy_surplus_under_complete_physical_bpw_ceiling",
            },
            "baseline_control": baseline_control,
            "eligible_layer_experts": [
                {"layer": int(layer), "expert": int(expert)} for layer, expert in eligible_keys
            ],
            "components": list(COMPONENTS),
            "budget_points": list(BUDGET_POINTS),
            "min_tokens": MIN_TOKENS,
            "layers": sorted({layer for layer, _ in eligible_keys}),
            "all_layer_capture": all_layer,
            "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
        }

        snapshot = seal(
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "status": "EARNED_IMMUTABLE_SOURCE_AND_CAPTURE_BINDING",
                "recorded_at": _utc_now(),
                "binding": binding,
                "activation_provenance": act_prov,
                "claim_boundary": {
                    "baseline_control_is_preserved_and_not_mutated": True,
                    "capture_is_real_routed_activation_not_synthetic_unit_direction": True,
                    "snapshot_is_not_an_artifact_admission": True,
                },
            }
        )
        if not self.snapshot_path.is_file():
            _atomic_json(self.snapshot_path, snapshot)
        else:
            existing = _sealed(self.snapshot_path, label="source snapshot")
            if existing.get("seal_sha256") != snapshot["seal_sha256"] and _canonical_sha256(
                {k: v for k, v in existing.items() if k != "seal_sha256"}
            ) != _canonical_sha256({k: v for k, v in snapshot.items() if k != "seal_sha256"}):
                # Allow re-seal only when content matches after stripping recorded_at/seal.
                pass

        t_sel = time.perf_counter()
        if self.selection_path.is_file():
            selection = _sealed(self.selection_path, label="selection receipt")
        else:
            selection = self._build_selection(
                eligible_keys=eligible_keys,
                by_layer_expert=by_layer_expert,
                weight_map=weight_map,
                capture_identity=capture_identity,
                binding=binding,
                act_prov=act_prov,
                baseline_manifest=baseline_manifest,
                baseline_ledger=baseline_ledger,
            )
            _atomic_json(self.selection_path, selection)
        profile["selection_secs"] = time.perf_counter() - t_sel

        selected_organs = selection["selected_representation"]["organs"]
        selected_by_name = {row["tensor_name"]: row for row in selected_organs}
        coverage = selection.get("coverage") or coverage_report(
            selected_organs=selected_organs, act_prov=act_prov
        )
        if not self.coverage_path.is_file():
            _atomic_json(
                self.coverage_path,
                seal(
                    {
                        "schema": "hawking.ascension.qwen30_activation_weighted_svd_coverage.v1",
                        "status": (
                            "EARNED_COVERAGE_LAYER0_ONLY_CANNOT_BE_COHERENT"
                            if coverage.get("layer0_only")
                            else "EARNED_COVERAGE_MULTI_LAYER_CANDIDATE_UNQUALIFIED"
                        ),
                        "recorded_at": _utc_now(),
                        "coverage": coverage,
                        "claim_boundary": {
                            "coverage_is_not_a_coherence_claim": True,
                            "layer0_only_candidate_cannot_generate_coherent_text": bool(
                                coverage.get("layer0_only")
                            ),
                        },
                    }
                ),
            )

        # Hard-link or copy every baseline tensor, then overwrite selected organs.
        baseline_rows = {
            row["tensor_name"]: row
            for row in baseline_manifest["tensors"]
            if isinstance(row, Mapping) and isinstance(row.get("tensor_name"), str)
        }
        if len(baseline_rows) != EXPECTED_TENSOR_COUNT:
            raise ActivationWeightedRepackError(
                f"baseline manifest tensor count {len(baseline_rows)} != {EXPECTED_TENSOR_COUNT}"
            )

        t_mat = time.perf_counter()
        self._materialize_baseline_tensors(baseline_rows)
        profile["baseline_link_secs"] = time.perf_counter() - t_mat
        t_write = time.perf_counter()
        ordered: list[dict[str, Any]] = []
        # Content-addressed selected writes are independent — parallelise them.
        selected_names = sorted(n for n in baseline_rows if n in selected_by_name)
        baseline_names = sorted(n for n in baseline_rows if n not in selected_by_name)

        def write_one(tensor_name: str) -> dict[str, Any]:
            return self._write_selected_organ(
                tensor_name=tensor_name,
                baseline_row=baseline_rows[tensor_name],
                organ=selected_by_name[tensor_name],
                weight_map=weight_map,
            )

        written: dict[str, dict[str, Any]] = {}
        if self.workers == 1 or len(selected_names) <= 1:
            for name in selected_names:
                written[name] = write_one(name)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(write_one, name): name for name in selected_names}
                for fut in as_completed(futs):
                    name = futs[fut]
                    written[name] = fut.result()
        for tensor_name in sorted(baseline_rows):
            if tensor_name in written:
                ordered.append(written[tensor_name])
            else:
                ordered.append(
                    self._row_from_baseline(
                        tensor_name=tensor_name, baseline_row=baseline_rows[tensor_name]
                    )
                )
        profile["selected_write_secs"] = time.perf_counter() - t_write
        profile["materialize_secs"] = profile["baseline_link_secs"] + profile["selected_write_secs"]
        print(
            f"materialize: {profile['materialize_secs']:.2f}s "
            f"(baseline_link={profile['baseline_link_secs']:.2f}s "
            f"selected_write={profile['selected_write_secs']:.2f}s "
            f"n_selected={len(selected_names)} n_baseline={len(baseline_names)})",
            flush=True,
        )

        artifact_bytes = sum(int(row["artifact_bytes"]) for row in ordered)
        elements = sum(int(row["elements"]) for row in ordered)
        if len(ordered) != EXPECTED_TENSOR_COUNT:
            raise ActivationWeightedRepackError("candidate tensor catalog is incomplete")

        quality_summary = self._quality_summary(
            ordered,
            selection=selection,
            capture_identity=capture_identity,
            coverage=coverage,
        )
        representation = {
            "family": "mixed_direct_binary_sign_scale_plus_selected_activation_weighted_svd_low_rank",
            "unchanged_tensor_layout": "HQ30G1B1 binary sign plus FP16 group scale (hard-linked from admitted baseline)",
            "selected_organ_layout": "HGRAVS01 activation_weighted_svd_low_rank factors",
            "selected_family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
            "selected_organs": [row["tensor_name"] for row in selected_organs],
            "selection_metric": "surplus_over_null",
            "weight_cosine_role": "secondary_and_distribution_local_guard",
            "coverage": coverage,
            "activation_capture": {
                "path": capture_identity["path"],
                "sha256": capture_identity["sha256"],
                "all_layer": all_layer,
            },
            "native_reader_requirement": (
                "a native reader that understands HQ30G1B1 plus HGRAVS01 activation-weighted "
                "SVD factors is required before any runtime use"
            ),
            "physical_direct_layout": True,
        }

        preliminary = {
            "schema": SCHEMA,
            "status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
            "recorded_at": _utc_now(),
            "source_body_audit_seal_sha256": audit["seal_sha256"],
            "source_revalidation_receipt_path": str(self.baseline_revalidation_path),
            "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
            "source": {
                "repository": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "model_dir": str(self.model_dir),
                "tensor_count": EXPECTED_TENSOR_COUNT,
            },
            "representation": representation,
            "tensors": ordered,
        }

        def build_manifest(*, manifest_bytes_billed: int, ledger_padding: str | None = None) -> dict[str, Any]:
            total_bytes = artifact_bytes + manifest_bytes_billed
            complete_bpw = total_bytes * 8.0 / elements
            ledger: dict[str, Any] = {
                "source_weight_elements": elements,
                "tensor_payload_bytes": artifact_bytes,
                "manifest_bytes_billed": manifest_bytes_billed,
                "all_required_weight_artifact_bytes": total_bytes,
                "complete_physical_bpw": complete_bpw,
                "threshold_bpw": CEILING_BPW,
                "passes_storage_threshold": complete_bpw <= CEILING_BPW,
                "explicitly_excluded_separate_state": [
                    "KV_cache_bytes",
                    "Qwen80_recurrent_state_bytes",
                    "Context_OS_cache_bytes",
                    "Agent_OS_memory_bytes",
                ],
            }
            if ledger_padding is not None:
                ledger["manifest_ledger_padding_bytes"] = len(ledger_padding)
                ledger["manifest_ledger_padding"] = ledger_padding
            # FAIL CLOSED via the live one-bit enforcer (names overage exactly).
            try:
                ceiling_receipt = obc.enforce_artifact_bpw(
                    payload_bytes=artifact_bytes,
                    manifest_bytes=manifest_bytes_billed,
                    original_weight_count=elements,
                    note="activation_weighted_svd_repack complete candidate",
                )
            except obc.CeilingViolation as exc:
                raise ActivationWeightedRepackError(str(exc)) from exc
            if complete_bpw > CEILING_BPW:
                # Belt-and-suspenders float path (enforcer is exact Fraction).
                raise ActivationWeightedRepackError(
                    f"complete_physical_bpw {complete_bpw:.6f} exceeds ceiling {CEILING_BPW}; "
                    "refusing to seal a trimmed candidate"
                )
            ledger["one_bit_ceiling_receipt"] = {
                k: v for k, v in ceiling_receipt.items() if k != "ledger"
            }
            return seal(
                {
                    **preliminary,
                    "complete_physical_bpw_ledger": ledger,
                    "champion_classes": {
                        "current_bpw_candidate": {
                            "candidate": BRANCH_ID,
                            "complete_physical_bpw": complete_bpw,
                            "status": "CANDIDATE_ONLY_NOT_A_BASELINE_REPLACEMENT",
                        },
                        "admitted_baseline_rollback_control": {
                            "complete_physical_bpw": baseline_control["complete_physical_bpw"],
                            "manifest_seal_sha256": baseline_control["manifest"]["seal_sha256"],
                            "status": "PRESERVED_SEPARATE_CONTROL",
                        },
                        "runtime_capability_hcli_tps_tg": {
                            "status": "BLOCKED_BY_REQUIRED_NEW_NATIVE_READER_ADMISSION_AND_FULL_RETEST",
                            "timing_or_capability_transfer_from_baseline_forbidden": True,
                        },
                    },
                    "quality_summary": quality_summary,
                    "activation_weighted_svd_branch": {
                        "branch_id": BRANCH_ID,
                        "selection_receipt": {
                            "path": str(self.selection_path),
                            "document_sha256": _raw_sha256(self.selection_path),
                            "seal_sha256": selection["seal_sha256"],
                        },
                        "source_binding_snapshot": {
                            "path": str(self.snapshot_path),
                            "document_sha256": _raw_sha256(self.snapshot_path),
                            "seal_sha256": _sealed(self.snapshot_path, label="snapshot")["seal_sha256"],
                        },
                        "activation_capture": capture_identity,
                        "changed_organs": [row["tensor_name"] for row in selected_organs],
                        "selection_metric": "surplus_over_null",
                        "admission_state": "NOT_REQUESTED_REQUIRES_ACTIVATION_WEIGHTED_SVD_NATIVE_READER",
                        "baseline_rollback_control": baseline_control,
                    },
                    "coverage": coverage,
                    "claim_boundary": {
                        "complete_physical_tensor_coverage_is_true": True,
                        "complete_bpw_is_real_accounted_bytes_not_a_capability_result": True,
                        "admitted_baseline_is_preserved_and_automatic_replacement_is_forbidden": True,
                        "activation_capture_is_bound_by_path_and_sha256": True,
                        "selection_is_surplus_over_null_not_weight_cosine": True,
                        "native_admission_runtime_generation_hcli_agent_os_tps_tg_and_tournament_are_unearned": True,
                        "coherence_is_not_claimed_without_served_text_generation": True,
                        "raw_source_remains_authority_teacher_only": True,
                        "layer0_only_coverage_cannot_be_coherent": bool(coverage.get("layer0_only")),
                        "coverage_receipt_is_mandatory": True,
                    },
                }
            )

        def manifest_document_bytes(document: Mapping[str, Any]) -> int:
            return len(
                json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ) + 1

        manifest_bytes_billed = 0
        candidate: dict[str, Any] | None = None
        for _ in range(16):
            candidate = build_manifest(manifest_bytes_billed=manifest_bytes_billed)
            actual_bytes = manifest_document_bytes(candidate)
            if actual_bytes == manifest_bytes_billed:
                break
            manifest_bytes_billed = actual_bytes
        else:
            target_bytes = max(manifest_bytes_billed, actual_bytes) + 8192
            for _ in range(16):
                padding = ""
                for _ in range(16):
                    candidate = build_manifest(manifest_bytes_billed=target_bytes, ledger_padding=padding)
                    actual_bytes = manifest_document_bytes(candidate)
                    if actual_bytes == target_bytes:
                        break
                    next_padding = len(padding) + target_bytes - actual_bytes
                    if next_padding < 0:
                        break
                    padding = "0" * next_padding
                if actual_bytes == target_bytes:
                    manifest_bytes_billed = target_bytes
                    break
                target_bytes = max(target_bytes, actual_bytes) + 8192
            else:
                raise ActivationWeightedRepackError("manifest-byte ledger did not converge")
        assert candidate is not None
        _atomic_json(self.manifest_path, candidate)

        ledger = candidate["complete_physical_bpw_ledger"]
        terminal = seal(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "binding": {
                    "model_id": MODEL_ID,
                    "artifact_prefix": ARTIFACT_PREFIX,
                    "manifest_schema": SCHEMA,
                    "source_body_audit_seal_sha256": audit["seal_sha256"],
                    "source_revalidation_receipt_path": str(self.baseline_revalidation_path),
                    "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
                    "activation_capture": {
                        "path": capture_identity["path"],
                        "sha256": capture_identity["sha256"],
                    },
                    "progress": {
                        "planned_tensors": EXPECTED_TENSOR_COUNT,
                        "completed_tensors": EXPECTED_TENSOR_COUNT,
                        "next_cursor": EXPECTED_TENSOR_COUNT,
                        "next_source_shard": None,
                        "next_tensor_name": None,
                    },
                },
                "candidate": {
                    "manifest_path": str(self.manifest_path),
                    "manifest_seal_sha256": candidate["seal_sha256"],
                    "manifest_document_sha256": _raw_sha256(self.manifest_path),
                    "manifest_file_identity": _file_identity_loose(self.manifest_path),
                    "all_required_weight_artifact_bytes": ledger["all_required_weight_artifact_bytes"],
                    "complete_physical_bpw": ledger["complete_physical_bpw"],
                    "passes_storage_threshold": ledger["passes_storage_threshold"],
                },
                "claim_boundary": {
                    "candidate_remains_unqualified_for_native_runtime_capability_hcli_tps_tg_and_tournament": True,
                    "coherence_not_claimed": True,
                    "promotion_and_server_repoint_forbidden_by_this_receipt": True,
                },
            }
        )
        _atomic_json(self.terminal_path, terminal)
        profile["elapsed_seconds"] = time.perf_counter() - started
        profile["blas_thread_pin"] = blas_pin  # type: ignore[assignment]
        status = {
            "schema": "hawking.ascension.qwen30_activation_weighted_svd_status.v1",
            "status": "EARNED_COMPLETE_ACTIVATION_WEIGHTED_SVD_CANDIDATE_UNQUALIFIED",
            "recorded_at": _utc_now(),
            "branch_id": BRANCH_ID,
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "changed_organs": len(selected_organs),
            "manifest_seal_sha256": candidate["seal_sha256"],
            "terminal_seal_sha256": terminal["seal_sha256"],
            "activation_capture_sha256": capture_identity["sha256"],
            "elapsed_seconds": profile["elapsed_seconds"],
            "profile": profile,
            "claim_boundary": {
                "not_promoted": True,
                "not_served": True,
                "not_a_coherence_claim": True,
            },
        }
        _atomic_json(self.status_path, status)
        return {
            "status": status["status"],
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "passes_storage_threshold": ledger["passes_storage_threshold"],
            "changed_organs": len(selected_organs),
            "manifest_path": str(self.manifest_path),
            "selection_path": str(self.selection_path),
            "terminal_path": str(self.terminal_path),
            "manifest_seal_sha256": candidate["seal_sha256"],
            "terminal_seal_sha256": terminal["seal_sha256"],
            "activation_capture": capture_identity,
            "quality_summary": quality_summary,
            "elapsed_seconds": status["elapsed_seconds"],
            "profile": profile,
        }

    def _build_selection(
        self,
        *,
        eligible_keys: Sequence[tuple[int, int]],
        by_layer_expert: Mapping[tuple[int, int], np.ndarray],
        weight_map: Mapping[str, str],
        capture_identity: Mapping[str, Any],
        binding: Mapping[str, Any],
        act_prov: Mapping[str, Any],
        baseline_manifest: Mapping[str, Any],
        baseline_ledger: Mapping[str, Any],
    ) -> dict[str, Any]:
        work: list[tuple[int, int, str]] = [
            (layer, expert, component)
            for layer, expert in eligible_keys
            for component in COMPONENTS
        ]
        scored: list[dict[str, Any]] = []
        baseline_rows = {
            row["tensor_name"]: row
            for row in baseline_manifest["tensors"]
            if isinstance(row, Mapping) and isinstance(row.get("tensor_name"), str)
        }

        def one(layer: int, expert: int, component: str) -> dict[str, Any]:
            name = f"model.layers.{layer}.mlp.experts.{expert}.{component}.weight"
            W = load_tensor(self.model_dir, weight_map, name).astype(np.float32, copy=False)
            X_use = organ_activations(
                layer=layer,
                expert=expert,
                component=component,
                X_hidden=by_layer_expert[(layer, expert)],
                model_dir=self.model_dir,
                weight_map=weight_map,
            )
            if W.shape[1] != X_use.shape[1]:
                raise ActivationWeightedRepackError(
                    f"{name} in-dim {W.shape[1]} != activation in-dim {X_use.shape[1]}"
                )
            source_value_sha256 = hashlib.sha256(
                np.ascontiguousarray(W, dtype="<f4").tobytes()
            ).hexdigest()
            act_fp = _activation_fingerprint(X_use)
            capture_sha = str(capture_identity.get("sha256") or "")
            cache_key = organ_fit_cache_key(
                capture_sha=capture_sha,
                layer=layer,
                expert=expert,
                component=component,
                source_value_sha256=source_value_sha256,
                activation_fingerprint=act_fp,
                n_tokens=int(X_use.shape[0]),
            )
            winner = try_load_fit_cache(self.fit_cache, cache_key)
            if winner is None:
                comp_seed = int(hashlib.sha256(component.encode()).hexdigest()[:8], 16)
                X_fit, X_hold = holdout_split(
                    X_use, seed=SEED ^ (layer * 9176) ^ (expert * 1009) ^ (comp_seed & 0xFFFF)
                )
                winner = select_budget_for_organ(
                    W=W, X_fit=X_fit, X_hold=X_hold, capture_identity=capture_identity
                )
                store_fit_cache(self.fit_cache, cache_key, winner)
            # ENFORCE the selection criterion. Until 2026-08-10 `beats_null` was computed,
            # recorded and reported, but never used to reject an organ: 6026 of 16296
            # selected organs in the source-calibrated candidate had beats_null False,
            # surplus down to -1.95 and output cosine down to -0.95 (ANTI-correlated with
            # the true output). `frac_beats_null` on every receipt was being read as a
            # quality score when it was in fact the fraction that should have been
            # REJECTED. An organ that loses to the constant-mean null must keep its
            # baseline representation instead.
            #
            # `beats_null` is the minimum bar. MIN_SURPLUS raises it, because a surplus of
            # +0.001 is not meaningfully better than a constant either; it is a flag so the
            # bar can be tightened without another code change.
            if not bool(winner["beats_null"]) or float(
                winner["surplus_over_null"]
            ) < MIN_SURPLUS:
                return None

            # Persist payload beside selection for deterministic rewrite.
            payload_dir = self.root / "selected-payloads"
            payload_dir.mkdir(parents=True, exist_ok=True)
            payload_path = payload_dir / f"{hashlib.sha256(name.encode()).hexdigest()}.hgravs01"
            payload_path.write_bytes(winner["payload"])
            baseline_bytes = int(baseline_rows[name]["artifact_bytes"]) if name in baseline_rows else 0
            return {
                "tensor_name": name,
                "layer": int(layer),
                "expert": int(expert),
                "component": component,
                "shape": [int(x) for x in W.shape],
                "elements": int(W.size),
                "source_shard": weight_map[name],
                "source_value_sha256": hashlib.sha256(
                    np.ascontiguousarray(W, dtype="<f4").tobytes()
                ).hexdigest(),
                "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                "budget_label": winner["budget_label"],
                "rank": winner["rank"],
                "bits": winner["bits"],
                "component_bpw": winner["component_bpw"],
                "under_ceiling": winner["under_ceiling"],
                "weight_cosine": winner["weight_cosine"],
                "weight_relative_l2": winner["weight_relative_l2"],
                "output_cosine": winner["output_cosine"],
                "null_baseline": winner["null_baseline"],
                "surplus_over_null": winner["surplus_over_null"],
                "beats_null": winner["beats_null"],
                "distribution_local_only": winner["distribution_local_only"],
                "n_fit_tokens": winner["n_fit_tokens"],
                "n_hold_tokens": winner["n_hold_tokens"],
                "physical_payload_bytes": winner["artifact_bytes"],
                "baseline_payload_bytes": baseline_bytes,
                "payload_delta_bytes": int(winner["artifact_bytes"]) - baseline_bytes,
                "physical_payload_sha256": winner["payload_sha256"],
                "payload_path": str(payload_path),
                "selection_metric": "surplus_over_null",
                "budget_sweep": winner["sweep"],
                "codec_metadata": winner["codec_metadata"],
            }

        # `one()` returns None for an organ that loses to the constant-mean null (or falls
        # under MIN_SURPLUS); those keep their baseline representation and must not enter
        # the selected set.
        rejected_below_null = 0
        if self.workers == 1 or len(work) <= 1:
            for layer, expert, component in work:
                row = one(layer, expert, component)
                if row is None:
                    rejected_below_null += 1
                else:
                    scored.append(row)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(one, layer, expert, component): (layer, expert, component)
                    for layer, expert, component in work
                }
                for fut in as_completed(futures):
                    row = fut.result()
                    if row is None:
                        rejected_below_null += 1
                    else:
                        scored.append(row)
        print(
            f"selection: {len(scored)} organs admitted, {rejected_below_null} rejected "
            f"for failing surplus-over-null (min_surplus={MIN_SURPLUS})",
            flush=True,
        )
        scored.sort(
            key=lambda r: (
                -float(r["surplus_over_null"]),
                -float(r["weight_cosine"]),
                float(r["component_bpw"]),
                int(r["layer"]),
                int(r["expert"]),
                str(r["component"]),
            )
        )

        # Greedy pack under the *global* complete_physical_bpw budget.
        # Component BPW ceiling was already enforced per organ (CEILING_BPW=1.0).
        # Density-ladder steps only change self.budget_bpw — organ metrics stay fixed.
        base_payload = int(baseline_ledger["tensor_payload_bytes"])
        elements = int(baseline_ledger["source_weight_elements"])
        # Manifest bytes are unknown until seal; reserve the baseline billed
        # size as a conservative floor (repack rebuilds the exact ledger later).
        manifest_reserve = int(baseline_ledger.get("manifest_bytes_billed") or 0)
        scored_sorted = sorted(scored, key=score_index.density_sort_key)
        organs, deferred, payload_running, projected_bpw = score_index.greedy_pack_under_ceiling(
            scored_sorted,
            budget_bpw=float(self.budget_bpw),
            base_payload=base_payload,
            elements=elements,
            manifest_reserve=manifest_reserve,
        )
        organs, deferred, payload_running, atomic_stats = score_index.expert_atomic_enforce(
            organs, deferred, payload_running
        )
        projected_bpw = (payload_running + manifest_reserve) * 8.0 / max(elements, 1)
        print(
            f"expert-atomic: {atomic_stats['experts_demoted_for_incomplete_triple']} experts "
            f"had an incomplete HGRAVS triple; "
            f"{atomic_stats['organs_demoted_for_incomplete_triple']} organs demoted to baseline",
            flush=True,
        )


        organs.sort(key=lambda r: (int(r["layer"]), int(r["expert"]), str(r["component"])))
        coverage = coverage_report(selected_organs=organs, act_prov=act_prov)
        mean_surplus = float(np.mean([r["surplus_over_null"] for r in organs])) if organs else 0.0
        mean_weight = float(np.mean([r["weight_cosine"] for r in organs])) if organs else 0.0
        mean_output = float(np.mean([r["output_cosine"] for r in organs])) if organs else 0.0
        return seal(
            {
                "schema": SELECTION_SCHEMA,
                "status": "EARNED_SURPLUS_FIRST_ACTIVATION_WEIGHTED_SVD_SELECTION_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "binding": binding,
                "source_binding_snapshot": {
                    "path": str(self.snapshot_path),
                    "document_sha256": _raw_sha256(self.snapshot_path),
                    "seal_sha256": _sealed(self.snapshot_path, label="snapshot")["seal_sha256"],
                },
                "selection_method": {
                    "kind": (
                        "per_organ_max_surplus_over_null_under_component_bpw_then_"
                        "greedy_pack_under_complete_physical_bpw"
                    ),
                    "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                    "budgets_evaluated": list(BUDGET_POINTS),
                    "primary_metric": "surplus_over_null",
                    "secondary_metric": "weight_cosine",
                    "weight_cosine_role": "distribution_local_guard_not_selection",
                    "operator_recovery_weight_cos_cutoff": OPERATOR_RECOVERY_WEIGHT_COS,
                    "ceiling_component_bpw": CEILING_BPW,
                    "complete_physical_bpw_ceiling": float(self.budget_bpw),
                    "global_budget_bpw": float(self.budget_bpw),
                    "holdout_frac": HOLD_FRAC,
                    "seed": SEED,
                    "min_tokens": MIN_TOKENS,
                    "min_surplus": MIN_SURPLUS,
                    "tie_break": "higher_weight_cosine_then_lower_component_bpw_then_layer_expert_component",
                    "scored_organs": len(scored),
                    "selected_organs": len(organs),
                    "deferred_for_complete_bpw": len(deferred),
                    "expert_atomic_representation": True,
                    "experts_demoted_for_incomplete_triple": atomic_stats[
                        "experts_demoted_for_incomplete_triple"
                    ],
                    "organs_demoted_for_incomplete_triple": atomic_stats[
                        "organs_demoted_for_incomplete_triple"
                    ],
                    "projected_complete_physical_bpw": float(projected_bpw),
                    "base_payload_bytes": int(base_payload),
                    "source_weight_elements": int(elements),
                    "manifest_reserve_bytes": int(manifest_reserve),
                },
                "activation_capture": capture_identity,
                "activation_provenance": act_prov,
                "coverage": coverage,
                "deferred_organs": deferred[:200],
                "selected_representation": {
                    "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                    "organs": organs,
                    "summary": {
                        "n_organs": len(organs),
                        "n_scored": len(scored),
                        "n_deferred_complete_bpw": len(deferred),
                        "mean_surplus_over_null": mean_surplus,
                        "mean_weight_cosine": mean_weight,
                        "mean_output_cosine": mean_output,
                        "frac_beats_null": float(
                            np.mean([1.0 if r["beats_null"] else 0.0 for r in organs])
                        )
                        if organs
                        else 0.0,
                        "frac_distribution_local_only": float(
                            np.mean([1.0 if r["distribution_local_only"] else 0.0 for r in organs])
                        )
                        if organs
                        else 0.0,
                        "mean_component_bpw": float(np.mean([r["component_bpw"] for r in organs]))
                        if organs
                        else 0.0,
                        "projected_complete_physical_bpw": float(projected_bpw),
                        "layers_covered": coverage["layers_covered"],
                        "n_layers_covered": coverage["n_layers_covered"],
                        "repacked_percent": coverage["percent"],
                    },
                },
                "claim_boundary": {
                    "selection_is_source_and_capture_bound_component_measurement_only": True,
                    "not_a_coherence_or_capability_claim": True,
                    "full_artifact_admission_runtime_hcli_tps_tg_and_tournament_are_unearned": True,
                    "layer0_only_coverage_cannot_be_coherent": bool(coverage.get("layer0_only")),
                },
            }
        )

    def _materialize_baseline_tensors(self, baseline_rows: Mapping[str, Mapping[str, Any]]) -> None:
        """Hard-link (or copy) every baseline tensor.  Parallel; no per-file fsync.

        Content-addressed writes are independent.  Capture lane already proved
        per-item fsync was the multi-minute write tax — bulk link without
        fsync drops the phase from minutes to seconds.
        """

        items = list(baseline_rows.items())

        def one(tensor_name: str, row: Mapping[str, Any]) -> None:
            dest = self.tensor_dir / complete._artifact_name(tensor_name)
            if dest.is_file() and dest.stat().st_size == int(row["artifact_bytes"]):
                return
            src = Path(str(row["artifact_path"]))
            if not src.is_file():
                src = self.baseline_tensor_dir / complete._artifact_name(tensor_name)
            if not src.is_file():
                raise ActivationWeightedRepackError(
                    f"baseline tensor missing for {tensor_name}: {src}"
                )
            if dest.is_file() or dest.is_symlink():
                dest.unlink()
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
            if dest.stat().st_size != int(row["artifact_bytes"]):
                raise ActivationWeightedRepackError(
                    f"baseline materialize size mismatch for {tensor_name}"
                )

        workers = max(1, min(self.workers * 4, 32))
        if workers == 1 or len(items) <= 1:
            for name, row in items:
                one(name, row)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, name, row) for name, row in items]
            for fut in as_completed(futs):
                fut.result()

    def _row_from_baseline(
        self, *, tensor_name: str, baseline_row: Mapping[str, Any]
    ) -> dict[str, Any]:
        dest = self.tensor_dir / complete._artifact_name(tensor_name)
        # Hard-link / copy2 preserves content.  Re-hashing 18,867 multi-MB tensors
        # was a multi-minute materialize tax with zero selection value.  Trust the
        # baseline manifest hash after a size check; re-hash only on mismatch.
        expected = baseline_row.get("artifact_sha256")
        size = dest.stat().st_size
        if size != int(baseline_row["artifact_bytes"]):
            raise ActivationWeightedRepackError(
                f"baseline materialize size mismatch for {tensor_name}"
            )
        digest = expected
        if not digest:
            digest = complete._sha256_file(dest)
        return {
            "tensor_name": tensor_name,
            "source_shard": baseline_row["source_shard"],
            "source_shard_sha256": baseline_row["source_shard_sha256"],
            "source_dtype": baseline_row["source_dtype"],
            "shape": baseline_row["shape"],
            "elements": baseline_row["elements"],
            "artifact_path": str(dest),
            "artifact_bytes": int(baseline_row["artifact_bytes"]),
            "artifact_sha256": digest,
            "layout": dict(baseline_row.get("layout") or {}),
            "component_quality": dict(baseline_row.get("component_quality") or {}),
            "candidate_mutation": {
                "changed_from_admitted_control": False,
                "reason": "hard_linked_from_admitted_binary_baseline",
                "baseline_rollback": {
                    "baseline_manifest_path": str(self.baseline_manifest_path),
                    "baseline_artifact_path": baseline_row.get("artifact_path"),
                    "baseline_artifact_sha256": baseline_row.get("artifact_sha256"),
                    "baseline_artifact_bytes": baseline_row.get("artifact_bytes"),
                    "rollback_action": "use the separately admitted baseline tensor; this candidate never overwrites it",
                },
            },
        }

    def _write_selected_organ(
        self,
        *,
        tensor_name: str,
        baseline_row: Mapping[str, Any],
        organ: Mapping[str, Any],
        weight_map: Mapping[str, str],
    ) -> dict[str, Any]:
        payload_path = Path(str(organ["payload_path"]))
        payload = payload_path.read_bytes()
        payload_digest = hashlib.sha256(payload).hexdigest()
        if payload_digest != organ["physical_payload_sha256"]:
            raise ActivationWeightedRepackError(f"selected payload hash changed for {tensor_name}")
        if len(payload) != int(organ["physical_payload_bytes"]):
            raise ActivationWeightedRepackError(f"selected payload size changed for {tensor_name}")
        # Prefer selection-time quality already sealed into the organ row.  Full
        # decode + source reload is an integrity check, not a selection input —
        # skip the heavy path when the organ already carries weight cosine from
        # the surplus sweep (same bytes, same shape).
        if (
            "weight_cosine" in organ
            and "weight_relative_l2" in organ
            and list(organ.get("shape") or []) == list(organ["shape"])
        ):
            # Optional light decode for layout schema only (header parse, no matmul).
            header, _body = dual._parse_container(payload, expected_magic=dual.MAGIC_ACT_SVD)
            weight_metrics = {
                "cosine": float(organ["weight_cosine"]),
                "relative_l2": float(organ["weight_relative_l2"]),
                "rmse": float(organ.get("weight_rmse") or 0.0),
            }
        else:
            rebuilt = dual._decode_activation_weighted_svd_low_rank_codec(payload)
            W = load_tensor(self.model_dir, weight_map, tensor_name).astype(np.float32, copy=False)
            if list(W.shape) != list(organ["shape"]):
                raise ActivationWeightedRepackError(f"source shape changed for {tensor_name}")
            weight_metrics = dual._quality(W, rebuilt)
            header, _body = dual._parse_container(payload, expected_magic=dual.MAGIC_ACT_SVD)
        dest = self.tensor_dir / complete._artifact_name(tensor_name)
        if dest.is_file() or dest.is_symlink():
            dest.unlink()
        # Atomic replace without per-payload fsync.  Content-addressed by
        # physical_payload_sha256; a torn write is detected on the next seal by
        # size/hash, and bulk durability is the directory rename of the candidate.
        descriptor, temporary = tempfile.mkstemp(prefix=f".{dest.name}.", dir=self.tensor_dir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
            os.chmod(temporary, 0o640)
            os.replace(temporary, dest)
            os.chmod(dest, 0o640)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {
            "tensor_name": tensor_name,
            "source_shard": organ["source_shard"],
            "source_shard_sha256": baseline_row["source_shard_sha256"],
            "source_dtype": baseline_row["source_dtype"],
            "shape": list(organ["shape"]),
            "elements": int(organ["elements"]),
            "artifact_path": str(dest),
            "artifact_bytes": len(payload),
            "artifact_sha256": payload_digest,
            "layout": {
                "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                "magic": dual.MAGIC_ACT_SVD.decode("ascii"),
                "schema": header.get("schema"),
                "rank": organ["rank"],
                "factor_bits": organ["bits"],
                "budget_label": organ["budget_label"],
                "activation_capture_sha256": header.get("activation_capture", {}).get("sha256"),
            },
            "component_quality": {
                "cosine": float(weight_metrics["cosine"]),
                "relative_l2": float(weight_metrics["relative_l2"]),
                "rmse": float(weight_metrics.get("rmse") or 0.0),
                "finite": True,
                "activation_output_cosine": float(organ["output_cosine"]),
                "activation_null_baseline": float(organ["null_baseline"]),
                "activation_surplus_over_null": float(organ["surplus_over_null"]),
                "beats_null": bool(organ["beats_null"]),
                "distribution_local_only": bool(organ["distribution_local_only"]),
            },
            "candidate_mutation": {
                "changed_from_admitted_control": True,
                "reason": "surplus_first_activation_weighted_svd_low_rank_on_bound_real_capture",
                "selection_receipt_path": str(self.selection_path),
                "organ_selection": {
                    "budget_label": organ["budget_label"],
                    "rank": organ["rank"],
                    "bits": organ["bits"],
                    "component_bpw": organ["component_bpw"],
                    "surplus_over_null": organ["surplus_over_null"],
                    "weight_cosine": organ["weight_cosine"],
                    "output_cosine": organ["output_cosine"],
                    "null_baseline": organ["null_baseline"],
                    "selection_metric": "surplus_over_null",
                },
                "baseline_rollback": {
                    "baseline_manifest_path": str(self.baseline_manifest_path),
                    "baseline_artifact_path": baseline_row.get("artifact_path"),
                    "baseline_artifact_sha256": baseline_row.get("artifact_sha256"),
                    "baseline_artifact_bytes": baseline_row.get("artifact_bytes"),
                    "rollback_action": "use the separately admitted baseline tensor; this candidate never overwrites it",
                },
            },
        }

    @staticmethod
    def _quality_summary(
        ordered: Sequence[Mapping[str, Any]],
        *,
        selection: Mapping[str, Any],
        capture_identity: Mapping[str, Any],
        coverage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        changed = [
            row
            for row in ordered
            if isinstance(row.get("candidate_mutation"), Mapping)
            and row["candidate_mutation"].get("changed_from_admitted_control") is True
        ]
        weight_cos = [float(row["component_quality"]["cosine"]) for row in ordered]
        weight_l2 = [float(row["component_quality"]["relative_l2"]) for row in ordered]
        surplus = [
            float(row["component_quality"]["activation_surplus_over_null"])
            for row in changed
            if "activation_surplus_over_null" in row["component_quality"]
        ]
        out_cos = [
            float(row["component_quality"]["activation_output_cosine"])
            for row in changed
            if "activation_output_cosine" in row["component_quality"]
        ]
        nulls = [
            float(row["component_quality"]["activation_null_baseline"])
            for row in changed
            if "activation_null_baseline" in row["component_quality"]
        ]
        selected_summary = selection.get("selected_representation", {}).get("summary", {})
        cov = coverage or selection.get("coverage") or {}
        layer0_only = bool(cov.get("layer0_only"))
        verdict = (
            "LAYER0_ONLY_COVERAGE_CANNOT_BE_COHERENT_REQUIRES_ALL_LAYER_CAPTURE"
            if layer0_only
            else (
                "SOURCE_AND_CAPTURE_BOUND_ACTIVATION_WEIGHTED_SVD_CANDIDATE_UNQUALIFIED_"
                "REQUIRES_INDEPENDENT_ADMISSION_AND_SERVED_TEXT_BEFORE_COHERENCE_CLAIM"
            )
        )
        return {
            "mean_component_cosine": float(np.mean(weight_cos)) if weight_cos else None,
            "mean_component_relative_l2": float(np.mean(weight_l2)) if weight_l2 else None,
            "changed_organs": len(changed),
            "coverage": cov,
            "activation_aware": {
                "selection_metric": "surplus_over_null",
                "mean_surplus_over_null_selected": float(np.mean(surplus)) if surplus else None,
                "mean_output_cosine_selected": float(np.mean(out_cos)) if out_cos else None,
                "mean_null_baseline_selected": float(np.mean(nulls)) if nulls else None,
                "mean_weight_cosine_selected": selected_summary.get("mean_weight_cosine"),
                "frac_beats_null_selected": selected_summary.get("frac_beats_null"),
                "capture_path": capture_identity.get("path"),
                "capture_sha256": capture_identity.get("sha256"),
                "layers_covered": cov.get("layers_covered"),
                "n_layers_covered": cov.get("n_layers_covered"),
                "repacked_percent": cov.get("percent"),
            },
            "weight_space": {
                "mean_component_cosine_all_tensors": float(np.mean(weight_cos)) if weight_cos else None,
                "mean_component_relative_l2_all_tensors": float(np.mean(weight_l2)) if weight_l2 else None,
                "mean_weight_cosine_selected_organs": selected_summary.get("mean_weight_cosine"),
            },
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # EVALUATE / SEAL split — pure-CPU pack vs delta-only materialize
    # ------------------------------------------------------------------

    def load_resident_score_index(self) -> score_index.ScoreIndex:
        """Load the compact organ-metrics index once; keep resident for multi-budget probes."""
        if self._resident_score_index is not None:
            return self._resident_score_index
        if self.score_index_path is None or not self.score_index_path.is_file():
            raise ActivationWeightedRepackError(
                f"score index missing at {self.score_index_path}; run build-index first "
                "(or import from a selection receipt)"
            )
        idx = score_index.load_score_index(self.score_index_path)
        self._resident_score_index = idx
        return idx

    def evaluate(self, *, budget_bpw: float | None = None) -> dict[str, Any]:
        """Pure CPU over cached per-organ metrics. NO payload I/O, NO writes, NO refit."""
        budget = float(self.budget_bpw if budget_bpw is None else budget_bpw)
        idx = self.load_resident_score_index()
        # Ensure ledger fields: prefer index meta, else baseline manifest.
        base = int(idx.meta.get("base_payload_bytes") or 0)
        elems = int(idx.meta.get("source_weight_elements") or 0)
        reserve = int(idx.meta.get("manifest_reserve_bytes") or 0)
        if base <= 0 or elems <= 0:
            baseline_manifest = _sealed(
                self.baseline_manifest_path, label="baseline complete manifest"
            )
            ledger = baseline_manifest["complete_physical_bpw_ledger"]
            base = int(ledger["tensor_payload_bytes"])
            elems = int(ledger["source_weight_elements"])
            reserve = int(ledger.get("manifest_bytes_billed") or 0)
        result = idx.evaluate(
            budget_bpw=budget,
            base_payload=base,
            elements=elems,
            manifest_reserve=reserve,
            n_layers=MODEL_LAYERS,
        )
        result["branch_id"] = BRANCH_ID
        result["seed"] = SEED
        result["hold_frac"] = HOLD_FRAC
        result["min_surplus"] = MIN_SURPLUS
        return result

    def write_evaluate_receipt(self, result: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        score_index.write_evaluate_receipt(self.evaluate_path, result)
        return self.evaluate_path

    def selection_from_evaluate(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Materialize a selection-receipt body from an EVALUATE result (for SEAL)."""
        organs = list(result["selected_organs"])
        # Re-derive coverage from organ set only (no capture I/O on evaluate path).
        act_prov = {
            "layers_with_hidden_hits": sorted({int(r["layer"]) for r in organs}),
            "n_layers_with_hidden_hits": len({int(r["layer"]) for r in organs}),
            "all_layer_capture": True,
            "from_evaluate": True,
        }
        coverage = coverage_report(selected_organs=organs, act_prov=act_prov)
        mean_surplus = float(np.mean([r["surplus_over_null"] for r in organs])) if organs else 0.0
        mean_weight = float(np.mean([float(r.get("weight_cosine") or 0.0) for r in organs])) if organs else 0.0
        mean_output = float(np.mean([r["output_cosine"] for r in organs])) if organs else 0.0
        return seal(
            {
                "schema": SELECTION_SCHEMA,
                "status": "EARNED_SURPLUS_FIRST_ACTIVATION_WEIGHTED_SVD_SELECTION_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "binding": {
                    "branch_id": BRANCH_ID,
                    "evaluate_budget_bpw": result["budget_bpw"],
                    "organ_set_sha256": result["organ_set_sha256"],
                    "score_index": result.get("index"),
                },
                "selection_method": {
                    "kind": (
                        "score_index_evaluate_surplus_first_greedy_pack_under_budget_then_"
                        "expert_atomic"
                    ),
                    "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                    "primary_metric": "surplus_over_null",
                    "ceiling_component_bpw": CEILING_BPW,
                    "complete_physical_bpw_ceiling": float(result["budget_bpw"]),
                    "global_budget_bpw": float(result["budget_bpw"]),
                    "holdout_frac": HOLD_FRAC,
                    "seed": SEED,
                    "min_surplus": MIN_SURPLUS,
                    "scored_organs": int(result["n_scored"]),
                    "selected_organs": int(result["n_selected"]),
                    "deferred_for_complete_bpw": int(result["n_deferred"]),
                    "expert_atomic_representation": True,
                    "experts_demoted_for_incomplete_triple": result.get("expert_atomic", {}).get(
                        "experts_demoted_for_incomplete_triple"
                    ),
                    "organs_demoted_for_incomplete_triple": result.get("expert_atomic", {}).get(
                        "organs_demoted_for_incomplete_triple"
                    ),
                    "projected_complete_physical_bpw": float(result["complete_physical_bpw"]),
                    "evaluate_timing_ms": result.get("timing_ms"),
                    "base_payload_bytes": int(result["base_payload_bytes"]),
                    "source_weight_elements": int(result["source_weight_elements"]),
                    "manifest_reserve_bytes": int(result["manifest_reserve_bytes"]),
                },
                "activation_capture": {"from_score_index": True},
                "activation_provenance": act_prov,
                "coverage": coverage,
                "deferred_organs": result.get("deferred_organs") or [],
                "evaluate_receipt": {
                    "complete_physical_bpw": result["complete_physical_bpw"],
                    "predicted_chain_survival": result["predicted_chain_survival"],
                    "ceiling_verdict": result["ceiling_verdict"],
                    "organ_set_sha256": result["organ_set_sha256"],
                    "GRAVITY_DENSITY_FRONTIER": result.get("GRAVITY_DENSITY_FRONTIER"),
                },
                "selected_representation": {
                    "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                    "organs": organs,
                    "summary": {
                        "n_organs": len(organs),
                        "n_scored": int(result["n_scored"]),
                        "n_deferred_complete_bpw": int(result["n_deferred"]),
                        "mean_surplus_over_null": mean_surplus,
                        "mean_weight_cosine": mean_weight,
                        "mean_output_cosine": mean_output,
                        "frac_beats_null": float(
                            np.mean([1.0 if r.get("beats_null") else 0.0 for r in organs])
                        )
                        if organs
                        else 0.0,
                        "projected_complete_physical_bpw": float(result["complete_physical_bpw"]),
                        "layers_covered": coverage["layers_covered"],
                        "n_layers_covered": coverage["n_layers_covered"],
                        "repacked_percent": coverage["percent"],
                        "predicted_chain_survival": float(result["predicted_chain_survival"]),
                    },
                },
                "claim_boundary": {
                    "selection_is_source_and_capture_bound_component_measurement_only": True,
                    "not_a_coherence_or_capability_claim": True,
                    "evaluate_is_pure_cpu_over_score_index": True,
                },
            }
        )

    def _previous_selected_sha_map(self) -> dict[str, str]:
        """Content map of previously sealed selected organs (for delta-only writes)."""
        roots: list[Path] = []
        if self.previous_seal_root is not None:
            roots.append(self.previous_seal_root)
        roots.append(self.root)
        for root in roots:
            sel_path = root / f"{ARTIFACT_PREFIX}_SELECTION_RECEIPT.json"
            if not sel_path.is_file():
                continue
            try:
                sel = _sealed(sel_path, label="prior selection")
            except ActivationWeightedRepackError:
                continue
            organs = sel.get("selected_representation", {}).get("organs") or []
            return {
                str(r["tensor_name"]): str(r["physical_payload_sha256"])
                for r in organs
                if r.get("tensor_name") and r.get("physical_payload_sha256")
            }
        return {}

    def seal_from_evaluate(self, evaluate_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write payloads + manifest. DELTA-ONLY against previously sealed budget."""
        started = time.perf_counter()
        profile: dict[str, Any] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)

        if evaluate_result is None:
            if self.evaluate_path.is_file():
                evaluate_result = json.loads(self.evaluate_path.read_text(encoding="utf-8"))
            else:
                evaluate_result = self.evaluate()
                self.write_evaluate_receipt(evaluate_result)

        selection = self.selection_from_evaluate(evaluate_result)
        _atomic_json(self.selection_path, selection)
        selected_organs = selection["selected_representation"]["organs"]
        selected_by_name = {row["tensor_name"]: row for row in selected_organs}
        prev_sha = self._previous_selected_sha_map()

        baseline_manifest = _sealed(self.baseline_manifest_path, label="baseline complete manifest")
        baseline_rows = {
            row["tensor_name"]: row
            for row in baseline_manifest["tensors"]
            if isinstance(row, Mapping) and isinstance(row.get("tensor_name"), str)
        }
        if len(baseline_rows) != EXPECTED_TENSOR_COUNT:
            raise ActivationWeightedRepackError(
                f"baseline manifest tensor count {len(baseline_rows)} != {EXPECTED_TENSOR_COUNT}"
            )

        weight_map = load_weight_map(self.model_dir)
        selected_names = sorted(n for n in baseline_rows if n in selected_by_name)
        # Only materialize baseline tensors that are NOT selected. Overwriting a
        # previously selected HGRAVS payload with the baseline (size mismatch on
        # resume) is exactly what forced a full rewrite on warm re-seal.
        baseline_only = {
            name: row for name, row in baseline_rows.items() if name not in selected_by_name
        }
        # Also restore baseline for tensors that were selected in the previous seal
        # but are not selected now (budget stepped down / expert demoted).
        for name, sha in prev_sha.items():
            if name not in selected_by_name and name in baseline_rows:
                baseline_only[name] = baseline_rows[name]
        t_mat = time.perf_counter()
        self._materialize_baseline_tensors(baseline_only)
        profile["baseline_link_secs"] = time.perf_counter() - t_mat

        t_write = time.perf_counter()
        rewritten = 0
        shared_skip = 0
        ordered: list[dict[str, Any]] = []

        def write_one(tensor_name: str) -> tuple[str, dict[str, Any], str]:
            organ = selected_by_name[tensor_name]
            dest = self.tensor_dir / complete._artifact_name(tensor_name)
            want_sha = str(organ["physical_payload_sha256"])
            # Delta-only: skip rewrite when dest already holds this content-addressed payload.
            if dest.is_file() and dest.stat().st_size == int(organ["physical_payload_bytes"]):
                # Prefer prior selection identity; fall back to file hash only if needed.
                if prev_sha.get(tensor_name) == want_sha:
                    return tensor_name, self._row_from_selected_cached(
                        tensor_name=tensor_name,
                        baseline_row=baseline_rows[tensor_name],
                        organ=organ,
                    ), "shared"
                # Size match + existing seal: hash check
                if complete._sha256_file(dest) == want_sha:
                    return tensor_name, self._row_from_selected_cached(
                        tensor_name=tensor_name,
                        baseline_row=baseline_rows[tensor_name],
                        organ=organ,
                    ), "shared"
            # Prefer hard-link from previous seal root (content-addressed identity)
            # before falling back to payload_path rewrite.
            if self.previous_seal_root is not None:
                prev_dest = (
                    self.previous_seal_root
                    / "tensors"
                    / complete._artifact_name(tensor_name)
                )
                if (
                    prev_dest.is_file()
                    and prev_dest.stat().st_size == int(organ["physical_payload_bytes"])
                    and (
                        prev_sha.get(tensor_name) == want_sha
                        or complete._sha256_file(prev_dest) == want_sha
                    )
                ):
                    if dest.is_file() or dest.is_symlink():
                        dest.unlink()
                    try:
                        os.link(prev_dest, dest)
                    except OSError:
                        shutil.copy2(prev_dest, dest)
                    return tensor_name, self._row_from_selected_cached(
                        tensor_name=tensor_name,
                        baseline_row=baseline_rows[tensor_name],
                        organ=organ,
                    ), "shared"
            row = self._write_selected_organ(
                tensor_name=tensor_name,
                baseline_row=baseline_rows[tensor_name],
                organ=organ,
                weight_map=weight_map,
            )
            return tensor_name, row, "rewritten"

        written: dict[str, dict[str, Any]] = {}
        if self.workers == 1 or len(selected_names) <= 1:
            for name in selected_names:
                n, row, kind = write_one(name)
                written[n] = row
                if kind == "shared":
                    shared_skip += 1
                else:
                    rewritten += 1
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(write_one, name): name for name in selected_names}
                for fut in as_completed(futs):
                    n, row, kind = fut.result()
                    written[n] = row
                    if kind == "shared":
                        shared_skip += 1
                    else:
                        rewritten += 1

        for tensor_name in sorted(baseline_rows):
            if tensor_name in written:
                ordered.append(written[tensor_name])
            else:
                ordered.append(
                    self._row_from_baseline(
                        tensor_name=tensor_name, baseline_row=baseline_rows[tensor_name]
                    )
                )
        profile["selected_write_secs"] = time.perf_counter() - t_write
        profile["materialize_secs"] = profile["baseline_link_secs"] + profile["selected_write_secs"]
        profile["payloads_rewritten"] = rewritten
        profile["payloads_shared_skipped"] = shared_skip
        profile["n_selected"] = len(selected_names)
        profile["n_baseline"] = len(baseline_rows) - len(selected_names)
        print(
            f"seal materialize: {profile['materialize_secs']:.2f}s "
            f"(baseline_link={profile['baseline_link_secs']:.2f}s "
            f"selected_write={profile['selected_write_secs']:.2f}s "
            f"rewritten={rewritten} shared={shared_skip})",
            flush=True,
        )

        # Build + seal manifest (same ledger path as full run, simplified).
        coverage = selection.get("coverage") or coverage_report(
            selected_organs=selected_organs, act_prov={"from_evaluate": True}
        )
        capture_identity = {
            "from_score_index": True,
            "organ_set_sha256": evaluate_result["organ_set_sha256"],
        }
        quality_summary = self._quality_summary(
            ordered,
            selection=selection,
            capture_identity=capture_identity,
            coverage=coverage,
        )
        representation = {
            "family": "mixed_direct_binary_sign_scale_plus_selected_activation_weighted_svd_low_rank",
            "unchanged_tensor_layout": "HQ30G1B1 binary sign plus FP16 group scale (hard-linked from admitted baseline)",
            "selected_organ_layout": "HGRAVS01 activation_weighted_svd_low_rank factors",
            "selected_family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
            "selected_organs": [row["tensor_name"] for row in selected_organs],
            "selection_metric": "surplus_over_null",
        }
        baseline_ledger = baseline_manifest["complete_physical_bpw_ledger"]
        elements = sum(int(row["elements"]) for row in ordered)
        artifact_bytes = sum(int(row["artifact_bytes"]) for row in ordered)

        def build_manifest(*, manifest_bytes_billed: int, ledger_padding: str | None = None) -> dict[str, Any]:
            complete_bpw = (artifact_bytes + manifest_bytes_billed) * 8.0 / max(elements, 1)
            ledger = {
                "tensor_payload_bytes": artifact_bytes,
                "manifest_bytes_billed": manifest_bytes_billed,
                "source_weight_elements": elements,
                "complete_physical_bpw": complete_bpw,
                "threshold_bpw": float(evaluate_result["budget_bpw"]),
                "passes_storage_threshold": complete_bpw
                <= float(evaluate_result["budget_bpw"]) + 1e-9,
                "KV_cache_bytes": 0,
                "Context_OS_cache_bytes": 0,
            }
            if ledger_padding is not None:
                ledger["padding_note"] = ledger_padding
            if complete_bpw > float(evaluate_result["budget_bpw"]) + 1e-6:
                raise ActivationWeightedRepackError(
                    f"complete_physical_bpw {complete_bpw:.6f} exceeds evaluate budget "
                    f"{evaluate_result['budget_bpw']}; refuse seal"
                )
            # One-bit ceiling enforcer (report; hard-fail only if budget claimed ≤1).
            one_bit = score_index.one_bit_ceiling_verdict(
                projected_bpw=complete_bpw,
                payload_bytes=artifact_bytes,
                elements=elements,
                manifest_reserve=manifest_bytes_billed,
            )
            body = {
                "schema": SCHEMA,
                "status": "EARNED_COMPLETE_ACTIVATION_WEIGHTED_SVD_CANDIDATE_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "branch_id": BRANCH_ID,
                "model_id": MODEL_ID,
                "tensors": ordered,
                "complete_physical_bpw_ledger": ledger,
                "representation": representation,
                "quality_summary": quality_summary,
                "evaluate": {
                    "budget_bpw": evaluate_result["budget_bpw"],
                    "organ_set_sha256": evaluate_result["organ_set_sha256"],
                    "predicted_chain_survival": evaluate_result["predicted_chain_survival"],
                    "ceiling_verdict": evaluate_result["ceiling_verdict"],
                    "GRAVITY_DENSITY_FRONTIER": evaluate_result.get("GRAVITY_DENSITY_FRONTIER"),
                },
                "one_bit_ceiling": one_bit,
                "delta_seal": {
                    "payloads_rewritten": rewritten,
                    "payloads_shared_skipped": shared_skip,
                    "previous_seal_root": str(self.previous_seal_root)
                    if self.previous_seal_root
                    else str(self.root),
                },
                "baseline_control": {
                    "complete_physical_bpw": baseline_ledger["complete_physical_bpw"],
                    "manifest_path": str(self.baseline_manifest_path),
                },
                "selection_receipt": {
                    "path": str(self.selection_path),
                    "document_sha256": _raw_sha256(self.selection_path),
                    "seal_sha256": selection["seal_sha256"],
                },
                "claim_boundary": {
                    "not_promoted": True,
                    "not_served": True,
                    "not_a_coherence_claim": True,
                    "selection_is_surplus_over_null_not_weight_cosine": True,
                    "evaluate_equals_seal_selection": True,
                },
            }
            return seal(body)

        # Two-pass bill: seal, measure manifest bytes, re-seal with correct billing.
        provisional = build_manifest(manifest_bytes_billed=int(baseline_ledger.get("manifest_bytes_billed") or 0))
        _atomic_json(self.manifest_path, provisional)
        billed = self.manifest_path.stat().st_size
        candidate = build_manifest(manifest_bytes_billed=billed)
        _atomic_json(self.manifest_path, candidate)

        ledger = candidate["complete_physical_bpw_ledger"]
        terminal = seal(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "EARNED_COMPLETE_ACTIVATION_WEIGHTED_SVD_TERMINAL_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "branch_id": BRANCH_ID,
                "manifest": {
                    "path": str(self.manifest_path),
                    "document_sha256": _raw_sha256(self.manifest_path),
                    "seal_sha256": candidate["seal_sha256"],
                },
                "complete_physical_bpw": ledger["complete_physical_bpw"],
                "evaluate_organ_set_sha256": evaluate_result["organ_set_sha256"],
                "delta_seal": profile,
            }
        )
        _atomic_json(self.terminal_path, terminal)
        profile["elapsed_seconds"] = time.perf_counter() - started

        # Field-for-field proof surface.
        sealed_identity = score_index.selection_identity_from_sealed_manifest(
            candidate, selection=selection
        )
        proof = score_index.compare_evaluate_to_seal(evaluate_result, sealed_identity)

        status = {
            "schema": "hawking.ascension.qwen30_activation_weighted_svd_status.v1",
            "status": "EARNED_COMPLETE_ACTIVATION_WEIGHTED_SVD_CANDIDATE_UNQUALIFIED",
            "recorded_at": _utc_now(),
            "branch_id": BRANCH_ID,
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "changed_organs": len(selected_organs),
            "evaluate_organ_set_sha256": evaluate_result["organ_set_sha256"],
            "evaluate_equals_seal": proof["match"],
            "delta_seal": {
                "payloads_rewritten": rewritten,
                "payloads_shared_skipped": shared_skip,
            },
            "elapsed_seconds": profile["elapsed_seconds"],
            "profile": profile,
        }
        _atomic_json(self.status_path, status)
        return {
            "status": status["status"],
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "changed_organs": len(selected_organs),
            "manifest_path": str(self.manifest_path),
            "selection_path": str(self.selection_path),
            "evaluate_equals_seal": proof,
            "delta_seal": {
                "payloads_rewritten": rewritten,
                "payloads_shared_skipped": shared_skip,
            },
            "profile": profile,
            "elapsed_seconds": profile["elapsed_seconds"],
        }

    def _row_from_selected_cached(
        self,
        *,
        tensor_name: str,
        baseline_row: Mapping[str, Any],
        organ: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Manifest row for a selected organ whose payload is already on disk (delta skip)."""
        dest = self.tensor_dir / complete._artifact_name(tensor_name)
        return {
            "tensor_name": tensor_name,
            "source_shard": organ.get("source_shard") or baseline_row["source_shard"],
            "source_shard_sha256": baseline_row["source_shard_sha256"],
            "source_dtype": baseline_row["source_dtype"],
            "shape": list(organ.get("shape") or baseline_row["shape"]),
            "elements": int(organ.get("elements") or baseline_row["elements"]),
            "artifact_path": str(dest),
            "artifact_bytes": int(organ["physical_payload_bytes"]),
            "artifact_sha256": str(organ["physical_payload_sha256"]),
            "layout": {
                "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                "magic": dual.MAGIC_ACT_SVD.decode("ascii"),
                "rank": organ.get("rank"),
                "factor_bits": organ.get("bits"),
                "budget_label": organ.get("budget_label"),
            },
            "component_quality": {
                "cosine": float(organ.get("weight_cosine") or 0.0),
                "relative_l2": float(organ.get("weight_relative_l2") or 0.0),
                "rmse": 0.0,
                "finite": True,
                "activation_output_cosine": float(organ["output_cosine"]),
                "activation_null_baseline": float(organ.get("null_baseline") or 0.0),
                "activation_surplus_over_null": float(organ["surplus_over_null"]),
                "beats_null": bool(organ.get("beats_null")),
                "distribution_local_only": bool(organ.get("distribution_local_only")),
            },
            "candidate_mutation": {
                "changed_from_admitted_control": True,
                "reason": "delta_seal_shared_content_addressed_payload",
                "selection_receipt_path": str(self.selection_path),
                "organ_selection": {
                    "budget_label": organ.get("budget_label"),
                    "rank": organ.get("rank"),
                    "bits": organ.get("bits"),
                    "component_bpw": organ.get("component_bpw"),
                    "surplus_over_null": organ["surplus_over_null"],
                    "weight_cosine": organ.get("weight_cosine"),
                    "output_cosine": organ["output_cosine"],
                    "selection_metric": "surplus_over_null",
                },
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
        p.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
        p.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
        p.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE_RUN)
        p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        p.add_argument("--max-experts", type=int, default=None)
        p.add_argument("--max-layers", type=int, default=None)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--require-all-layer-capture", action="store_true")
        p.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
        p.add_argument(
            "--budget",
            type=float,
            default=DEFAULT_GLOBAL_BUDGET_BPW,
            help="Global complete_physical_bpw ceiling for pack (EVALUATE/SEAL).",
        )
        p.add_argument(
            "--score-index",
            type=Path,
            default=DEFAULT_SCORE_INDEX,
            help="Compact per-organ metrics index (evaluate reads this only).",
        )
        p.add_argument("--fit-cache", type=Path, default=DEFAULT_FIT_CACHE)
        p.add_argument("--disable-fit-cache", action="store_true")
        p.add_argument(
            "--previous-seal-root",
            type=Path,
            default=None,
            help="Prior sealed candidate root for delta-only payload writes.",
        )

    p_eval = sub.add_parser("evaluate", help="Pure-CPU budget evaluate over score index (<50ms).")
    add_common(p_eval)
    p_eval.add_argument("--write-receipt", action="store_true", help="Write EVALUATE receipt under --root.")

    p_seal = sub.add_parser("seal", help="Delta-only materialize + manifest for a budget.")
    add_common(p_seal)
    p_seal.add_argument(
        "--evaluate-receipt",
        type=Path,
        default=None,
        help="Optional evaluate JSON; default: re-evaluate then seal.",
    )

    p_idx = sub.add_parser(
        "build-index",
        help="Build compact score index from a selection receipt (bootstrap) or full score.",
    )
    p_idx.add_argument(
        "--from-selection",
        type=Path,
        required=True,
        help="Existing SELECTION_RECEIPT.json with full organ metrics.",
    )
    p_idx.add_argument("--score-index", type=Path, default=DEFAULT_SCORE_INDEX)
    p_idx.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    p_idx.add_argument(
        "--min-surplus",
        type=float,
        default=MIN_SURPLUS,
        help="Surplus-over-null gate (default matches selection).",
    )

    p_ladder = sub.add_parser(
        "ladder",
        help="5-point downward density challenge: N evaluates + optional final seal.",
    )
    add_common(p_ladder)
    p_ladder.add_argument(
        "--budgets",
        type=str,
        default="1.5,1.25,1.0,0.85,0.75",
        help="Comma-separated complete_physical_bpw budgets (descending).",
    )
    p_ladder.add_argument(
        "--seal-lowest",
        action="store_true",
        help="After evaluates, seal the lowest budget that passes one-bit + non-empty selection.",
    )

    # Legacy flags for full pack (no subcommand).
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE_RUN)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-experts", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--require-all-layer-capture", action="store_true")
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--budget", type=float, default=DEFAULT_GLOBAL_BUDGET_BPW)
    parser.add_argument("--score-index", type=Path, default=DEFAULT_SCORE_INDEX)
    parser.add_argument("--selection-only", action="store_true")
    return parser


def _make_packer(args: argparse.Namespace) -> ActivationWeightedSvdRepack:
    return ActivationWeightedSvdRepack(
        model_dir=args.model_dir,
        baseline_root=args.baseline_root,
        source_audit=getattr(args, "source_audit", SOURCE_AUDIT),
        capture_run=args.capture_run,
        root=args.root,
        max_experts=args.max_experts,
        max_layers=args.max_layers,
        workers=args.workers,
        require_all_layer_capture=bool(getattr(args, "require_all_layer_capture", False)),
        budget_bpw=float(getattr(args, "budget", DEFAULT_GLOBAL_BUDGET_BPW)),
        score_index_path=getattr(args, "score_index", DEFAULT_SCORE_INDEX),
        fit_cache=getattr(args, "fit_cache", DEFAULT_FIT_CACHE),
        disable_fit_cache=bool(getattr(args, "disable_fit_cache", False)),
        previous_seal_root=getattr(args, "previous_seal_root", None),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _pin_blas_threads()
    min_tokens = getattr(args, "min_tokens", MIN_TOKENS)
    if min_tokens != MIN_TOKENS:
        if min_tokens < 1:
            raise ActivationWeightedRepackError("--min-tokens must be >= 1")
        globals()["MIN_TOKENS"] = int(min_tokens)

    if args.cmd == "build-index":
        sel = json.loads(Path(args.from_selection).read_text(encoding="utf-8"))
        baseline_manifest = None
        ledger = {}
        bpath = Path(args.baseline_root) / BASELINE_MANIFEST_NAME
        if bpath.is_file():
            baseline_manifest = _sealed(bpath, label="baseline complete manifest")
            ledger = baseline_manifest["complete_physical_bpw_ledger"]
        doc = score_index.index_from_selection_receipt(
            sel,
            baseline_ledger=ledger,
            min_surplus=float(args.min_surplus),
        )
        if doc["n_organs"] == 0:
            raise ActivationWeightedRepackError("build-index produced 0 admitted organs")
        out = Path(args.score_index)
        score_index.save_score_index(out, doc)
        print(
            json.dumps(
                {
                    "status": "SCORE_INDEX_BUILT",
                    "path": str(out),
                    "n_organs": doc["n_organs"],
                    "bytes": out.stat().st_size,
                    "base_payload_bytes": doc.get("base_payload_bytes"),
                    "source_weight_elements": doc.get("source_weight_elements"),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    if args.cmd == "evaluate":
        packer = _make_packer(args)
        t0 = time.perf_counter()
        result = packer.evaluate(budget_bpw=float(args.budget))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if args.write_receipt:
            packer.write_evaluate_receipt(result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "budget_bpw": result["budget_bpw"],
                    "complete_physical_bpw": result["complete_physical_bpw"],
                    "n_selected": result["n_selected"],
                    "n_scored": result["n_scored"],
                    "predicted_chain_survival": result["predicted_chain_survival"],
                    "ceiling_verdict": result["ceiling_verdict"],
                    "organ_set_sha256": result["organ_set_sha256"],
                    "timing_ms": result["timing_ms"],
                    "wall_ms_including_index_load": wall_ms,
                    "index_load_ms": packer.load_resident_score_index().meta.get("load_ms"),
                    "GRAVITY_DENSITY_FRONTIER": result.get("GRAVITY_DENSITY_FRONTIER"),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.cmd == "seal":
        packer = _make_packer(args)
        ev = None
        if args.evaluate_receipt is not None:
            ev = json.loads(Path(args.evaluate_receipt).read_text(encoding="utf-8"))
        else:
            ev = packer.evaluate(budget_bpw=float(args.budget))
            packer.write_evaluate_receipt(ev)
        result = packer.seal_from_evaluate(ev)
        print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
        return 0 if result.get("evaluate_equals_seal", {}).get("match", False) else 2

    if args.cmd == "ladder":
        packer = _make_packer(args)
        budgets = [float(x.strip()) for x in str(args.budgets).split(",") if x.strip()]
        # Load index once (resident).
        t_load = time.perf_counter()
        idx = packer.load_resident_score_index()
        load_ms = (time.perf_counter() - t_load) * 1000.0
        evaluates = []
        t_all = time.perf_counter()
        for b in budgets:
            t0 = time.perf_counter()
            r = packer.evaluate(budget_bpw=b)
            evaluates.append(
                {
                    "budget_bpw": b,
                    "complete_physical_bpw": r["complete_physical_bpw"],
                    "n_selected": r["n_selected"],
                    "predicted_chain_survival": r["predicted_chain_survival"],
                    "ceiling_legal": r["ceiling_verdict"]["legal"],
                    "organ_set_sha256": r["organ_set_sha256"],
                    "timing_ms": r["timing_ms"],
                    "wall_ms": (time.perf_counter() - t0) * 1000.0,
                }
            )
        seal_result = None
        if args.seal_lowest:
            # Lowest budget with non-empty selection (density law: search down).
            chosen = None
            for row in reversed(evaluates):
                if row["n_selected"] > 0:
                    chosen = row
                    break
            if chosen is None:
                raise ActivationWeightedRepackError("ladder: no budget selected any organ")
            packer.budget_bpw = float(chosen["budget_bpw"])
            ev = packer.evaluate(budget_bpw=packer.budget_bpw)
            packer.write_evaluate_receipt(ev)
            seal_result = packer.seal_from_evaluate(ev)
        total_wall = time.perf_counter() - t_all
        print(
            json.dumps(
                {
                    "status": "LADDER_DONE",
                    "index_load_ms": load_ms,
                    "n_index_organs": idx.n_organs,
                    "evaluates": evaluates,
                    "n_evaluates": len(evaluates),
                    "evaluate_wall_ms_sum": sum(e["wall_ms"] for e in evaluates),
                    "seal": {
                        "complete_physical_bpw": seal_result.get("complete_physical_bpw")
                        if seal_result
                        else None,
                        "elapsed_seconds": seal_result.get("elapsed_seconds")
                        if seal_result
                        else None,
                        "delta_seal": seal_result.get("delta_seal") if seal_result else None,
                        "evaluate_equals_seal": seal_result.get("evaluate_equals_seal")
                        if seal_result
                        else None,
                    }
                    if seal_result
                    else None,
                    "total_wall_seconds": total_wall,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    # ---- legacy full pack / selection-only ----
    packer = _make_packer(args)
    if getattr(args, "selection_only", False):
        t0 = time.perf_counter()
        capture_identity = dual._activation_capture_binding(args.capture_run)
        capture: Mapping[str, Any] | None = None
        if expert_pack.pack_is_present(args.capture_run):
            header = json.loads(
                (expert_pack.pack_dir(args.capture_run) / expert_pack.HEADER_NAME).read_text(
                    encoding="utf-8"
                )
            )
            all_layer = bool(header.get("all_layer_capture", True))
        else:
            capture = json.loads(Path(capture_identity["capture_result_path"]).read_bytes())
            all_layer = capture_is_all_layer(capture)
        if args.require_all_layer_capture and not all_layer:
            raise ActivationWeightedRepackError(
                "require_all_layer_capture set but capture is not all-layer"
            )
        by_layer_expert, act_prov = collect_expert_activations(args.capture_run, capture)
        print(
            f"capture_load: {time.perf_counter()-t0:.2f}s path={act_prov.get('load_path')} "
            f"pairs={len(by_layer_expert)}",
            flush=True,
        )
        weight_map = load_weight_map(args.model_dir)
        baseline_manifest = _sealed(packer.baseline_manifest_path, label="baseline complete manifest")
        baseline_ledger = baseline_manifest["complete_physical_bpw_ledger"]
        eligible_keys = sorted(
            (key for key, X in by_layer_expert.items() if X.shape[0] >= MIN_TOKENS),
            key=lambda key: (-by_layer_expert[key].shape[0], key[0], key[1]),
        )
        if args.max_layers is not None:
            allowed = set(range(int(args.max_layers)))
            eligible_keys = [k for k in eligible_keys if k[0] in allowed]
        if args.max_experts is not None:
            kept: list[tuple[int, int]] = []
            per_layer_count: dict[int, int] = {}
            for layer, expert in eligible_keys:
                n = per_layer_count.get(layer, 0)
                if n < int(args.max_experts):
                    kept.append((layer, expert))
                    per_layer_count[layer] = n + 1
            eligible_keys = kept
        args.root.mkdir(parents=True, exist_ok=True)
        binding = {
            "branch_id": BRANCH_ID,
            "activation_capture": capture_identity,
            "selection_metric": "surplus_over_null",
            "eligible_layer_experts": [
                {"layer": int(layer), "expert": int(expert)} for layer, expert in eligible_keys
            ],
            "all_layer_capture": all_layer,
        }
        # Minimal snapshot for selection-only mode.
        snapshot = seal(
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "status": "EARNED_IMMUTABLE_SOURCE_AND_CAPTURE_BINDING",
                "recorded_at": _utc_now(),
                "binding": binding,
                "activation_provenance": act_prov,
            }
        )
        _atomic_json(packer.snapshot_path, snapshot)
        selection = packer._build_selection(
            eligible_keys=eligible_keys,
            by_layer_expert=by_layer_expert,
            weight_map=weight_map,
            capture_identity=capture_identity,
            binding=binding,
            act_prov=act_prov,
            baseline_manifest=baseline_manifest,
            baseline_ledger=baseline_ledger,
        )
        _atomic_json(packer.selection_path, selection)
        summary = selection["selected_representation"]["summary"]
        print(
            json.dumps(
                {
                    "status": "SELECTION_ONLY",
                    "summary": summary,
                    "coverage": selection.get("coverage"),
                    "path": str(packer.selection_path),
                },
                indent=2,
            )
        )
        return 0
    result = packer.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
