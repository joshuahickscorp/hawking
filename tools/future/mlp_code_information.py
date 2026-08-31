"""MLP CODE INFORMATION — how much of the 4.28 GB of 2-bit codes is independent?

Sealed-3.14 MLP active bytes split, already measured, into 4,278,190,080 bytes
of 2-bit codes and 1,069,605,696 bytes of scale/bias/header. The auxiliary 1.07
GB is a different object (nine sharing families already MEASURED_NEGATIVE
there). This module reads the real HGRAVF01 *code* arrays and asks how much of
the 4.28 GB is independent information, beyond uniform quantization.

    python3 tools/future/mlp_code_information.py --build
    python3 -m pytest tools/future/test_mlp_code_information.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import math
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future._common import REPO, git, write_receipt
from tools.future.ebpw_categories import PRODUCTION, judge_dense_rematerialization
from tools.future.mlp_auxiliary_information import (
    AUXILIARY_BYTES_TARGET,
    CODE_BYTES_TARGET,
    MLP_ACTIVE_TARGET,
    CatalogLayoutRefuse,
    _read_u8,
    _unpack_q,
    auxiliary_rows,
    parse_hgrafv01_header,
)
from tools.future.mlp_byte_census import (
    AFFINE_CODE_BITS,
    CatalogAbsent,
    load_geometry,
    load_sealed,
    resolve_artifact_root,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "MLP_CODE_INFORMATION.json"
SCHEMA = "hawking.future.mlp_code_information.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_code_information.py"

INCUMBENT_GROUP = 64
AFFINE_CODEC = 5
DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FALSIFIED = "ALREADY_FALSIFIED"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
OPEN = "OPEN"
UNMEASURED = "UNMEASURED"

# NNS-022 reopen: entropy of the index ≤ 0.9 of uniform. Analog applied to q.
NNS022_REOPEN_UNIFORM_FRAC = 0.9

SAMPLE_LAYERS: tuple[int, ...] = (0, 21, 42, 63)
RNG_SEED = 38
ZLIB_LEVEL = 1
MODE_MASS_SAMPLE = 200_000

# Nine auxiliary sharing families already MEASURED_NEGATIVE on scale/bias.
# This receipt must not restate them as if they were about the code body.
AUXILIARY_DEAD_IDS: tuple[str, ...] = (
    "shared_scale_basis",
    "per_tensor_curve_plus_residual",
    "predict_scale_from_code_stats",
    "low_rank_scale_matrix",
    "parametric_scale_program",
    "tie_bias_to_minus_half_codes",
    "drop_bias",
    "collapse_to_global_scale",
    "cross_layer_scale_delta",
)

# Contract search list, plus the code-stream entropy floor as its own row.
REQUIRED_CANDIDATE_IDS: tuple[str, ...] = (
    "lower_bit_native",
    "heterogeneous_bit_allocation",
    "generated_tensors",
    "generated_programs",
    "shared_code_bases",
    "factorized_programs",
    "dictionary_of_code_blocks",
    "product_codebooks",
    "lowrank_plus_sparse_residual",
    "block_generators",
    "cross_layer_code_prediction",
    "capability_sensitive_literal_islands",
    "shared_input_transforms",
    "latent_routed_accumulation",
    "function_replacement",
    "entropy_coded_code_stream",
)

NOETIC_RELS = (
    "receipts/future/evidence/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
)
QN_REL = "tools/headless/negative_science.py"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Numbers are catalog-header byte counts plus CPU statistics on the real "
    "HGRAVF01 2-bit code payloads of sealed-3.14 (not the f16 scale/bias, not "
    "a model of W). Shannon / conditional entropy / mutual information / zlib "
    "are lossless statements about the stored q ∈ {0,1,2,3} stream. They are "
    "not capability. A candidate that only rematerializes a dense W, or that "
    "expands a compressed code stream back to the incumbent 2-bit buffer "
    "before the kernel, does not eliminate active bytes."
)


class CodeRefuse(ValueError):
    """The code-body census refused rather than guessing."""


class UnreconciledCode(CodeRefuse):
    """2-bit code bytes do not equal the recorded 4,278,190,080."""

    def __init__(self, got: int, want: int = CODE_BYTES_TARGET, *, detail: str = "") -> None:
        self.got = int(got)
        self.want = int(want)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: code bytes {got} != recorded code total {want}{extra}"
        )


# ---------------------------------------------------------------------------
# Accounting. Load-bearing refusal.
# ---------------------------------------------------------------------------


def reconcile_code_bytes(
    code_bytes: int,
    want: int = CODE_BYTES_TARGET,
    *,
    detail: str = "",
) -> int:
    """Refuse unless the 2-bit payload sums to the recorded code total."""
    got = int(code_bytes)
    if got != int(want):
        raise UnreconciledCode(got, int(want), detail=detail)
    return got


def expected_code_bytes(shape: Sequence[int], bits: int = AFFINE_CODE_BITS) -> int:
    n = int(shape[0]) * int(shape[1])
    if n <= 0 or (n * bits) % 8:
        raise CodeRefuse(f"shape {list(shape)} is not a whole number of code bytes")
    return (n * bits) // 8


def code_rows(
    *,
    root: Path | None = None,
    mlp: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-tensor 2-bit code bytes from the real HGRAVF01 headers."""
    rows = list(mlp) if mlp is not None else auxiliary_rows(root=root)
    out: list[dict[str, Any]] = []
    for rec in rows:
        want = expected_code_bytes(rec["shape"])
        got = int(rec["code_bytes"])
        if got != want:
            raise CatalogLayoutRefuse(
                f"{rec['name']}: code_bytes {got} != shape*bits/8 {want}"
            )
        if int(rec["group_size"]) != INCUMBENT_GROUP:
            raise CatalogLayoutRefuse(
                f"{rec['name']}: group_size {rec['group_size']} != {INCUMBENT_GROUP}"
            )
        out.append(dict(rec))
    if not out:
        raise CodeRefuse("catalog holds no MLP affine tensors; refusing")
    return out


def accounting_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum the 2-bit payloads and refuse if they are not 4,278,190,080."""
    if not rows:
        raise CodeRefuse("no code rows; refusing an empty accounting")
    code = sum(int(r["code_bytes"]) for r in rows)
    header = sum(int(r["header_bytes"]) for r in rows)
    scale = sum(int(r["scale_bytes"]) for r in rows)
    bias = sum(int(r["bias_bytes"]) for r in rows)
    stored = sum(int(r["stored_bytes"]) for r in rows)
    n_groups = sum(int(r["groups"]) for r in rows)
    n_params = n_groups * INCUMBENT_GROUP
    reconcile_code_bytes(code, detail="sum of per-tensor HGRAVF01 code_bytes")
    derived = (n_params * AFFINE_CODE_BITS) // 8
    reconcile_code_bytes(derived, detail="n_params * 2 bits / 8")
    if stored != MLP_ACTIVE_TARGET:
        raise UnreconciledCode(
            stored, MLP_ACTIVE_TARGET, detail="MLP stored bytes vs recorded MLP active total"
        )
    if scale + bias + header != AUXILIARY_BYTES_TARGET:
        raise UnreconciledCode(
            scale + bias + header,
            AUXILIARY_BYTES_TARGET,
            detail="auxiliary (not this object) failed to reassemble; catalog layout is broken",
        )
    if header + scale + bias + code != stored:
        raise CatalogLayoutRefuse("parts do not reassemble to stored MLP bytes")
    return {
        "n_tensors": len(rows),
        "n_groups": n_groups,
        "n_parameters": n_params,
        "group_size": INCUMBENT_GROUP,
        "code_bits": AFFINE_CODE_BITS,
        "code_bytes": code,
        "header_bytes": header,
        "scale_bytes": scale,
        "bias_bytes": bias,
        "auxiliary_bytes": scale + bias + header,
        "stored_bytes": stored,
        "code_share_of_mlp": code / MLP_ACTIVE_TARGET,
        "target": CODE_BYTES_TARGET,
        "reconciled": True,
        "incumbent_packing": {
            "family": "affine_q2_group64_ls",
            "code_bits": AFFINE_CODE_BITS,
            "alphabet": [0, 1, 2, 3],
            "reconstruction": "w = float(q) * scale + bias, q unsigned in {0,1,2,3}",
        },
    }


def accounting(*, root: Path | None = None) -> dict[str, Any]:
    rows = code_rows(root=root)
    snap = accounting_from_rows(rows)
    snap["identity"] = _identity(root)
    return snap


def _identity(root: Path | None = None) -> dict[str, Any]:
    sealed = load_sealed()
    artifact = root if root is not None else resolve_artifact_root(sealed)
    geo = load_geometry(artifact)
    return {
        "resident_identity": sealed.get("resident_identity"),
        "artifact_root": str(artifact),
        "catalog": str(artifact / "catalog.hq38m20"),
        "model_id": sealed.get("model_id"),
        "geometry": {
            "hidden_size": geo["hidden_size"],
            "intermediate_size": geo["intermediate_size"],
            "num_hidden_layers": geo["num_hidden_layers"],
        },
    }


# ---------------------------------------------------------------------------
# Information-theoretic primitives on packed 2-bit streams.
# ---------------------------------------------------------------------------


def _shannon(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _mi(joint: np.ndarray) -> float:
    j = np.asarray(joint, dtype=np.float64)
    tot = float(j.sum())
    if tot <= 0:
        return 0.0
    p = j / tot
    pa = p.sum(axis=1, keepdims=True)
    pb = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p > 0) & (pa > 0) & (pb > 0), p / (pa * pb), 1.0)
        val = np.where(p > 0, p * np.log2(ratio), 0.0)
    return float(val.sum())


def _cond_h(joint: np.ndarray) -> float:
    j = np.asarray(joint, dtype=np.float64)
    tot = float(j.sum())
    if tot <= 0:
        return 0.0
    pa = j.sum(axis=1)
    hb = 0.0
    for a in range(j.shape[0]):
        if pa[a] <= 0:
            continue
        pba = j[a] / pa[a]
        pba = pba[pba > 0]
        hb += (pa[a] / tot) * float(-(pba * np.log2(pba)).sum())
    return hb


def _q_and_within_from_byte_hist(bh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold a 256-bin packed-byte hist into q, nibble, and within-byte lag-1 joints."""
    qh = np.zeros(4, dtype=np.int64)
    nh = np.zeros(16, dtype=np.int64)
    within = np.zeros((4, 4), dtype=np.int64)
    for b, c in enumerate(bh.tolist()):
        if not c:
            continue
        qs = (b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3)
        for q in qs:
            qh[q] += c
        nh[b & 15] += c
        nh[b >> 4] += c
        for i in range(3):
            within[qs[i], qs[i + 1]] += c
    return qh, nh, within


def _joint_aligned(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    j = np.zeros((4, 4), dtype=np.int64)
    for s in (0, 2, 4, 6):
        qa = (a >> s) & 3
        qb = (b >> s) & 3
        j += np.bincount(qa.astype(np.int64) * 4 + qb.astype(np.int64), minlength=16).reshape(4, 4)
    return j


def _unique_groups(packed: np.ndarray) -> tuple[int, int]:
    n = packed.size // 16
    h = packed.reshape(n, 16).view(np.dtype((np.void, 16))).ravel()
    return int(np.unique(h).size), n


def _mode_mass_groups(
    packed: np.ndarray, take: int = MODE_MASS_SAMPLE, rng: np.random.Generator | None = None
) -> tuple[float, int, int]:
    n = packed.size // 16
    view = packed.reshape(n, 16)
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
    idx = rng.choice(n, min(int(take), n), replace=False)
    samp = view[idx].view(np.dtype((np.void, 16))).ravel()
    _uniq, counts = np.unique(samp, return_counts=True)
    return float(counts.max()) / float(samp.size), int(counts.size), int(samp.size)


def _summ(xs: Sequence[float]) -> dict[str, Any] | None:
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


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


# ---------------------------------------------------------------------------
# Array measurements. Real 2-bit codes, not a model of them.
# ---------------------------------------------------------------------------


def measure_code_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_layers: Sequence[int] = SAMPLE_LAYERS,
    rng_seed: int = RNG_SEED,
) -> dict[str, Any]:
    """CPU statistics on every MLP 2-bit code payload, plus sampled probes."""
    ordered = sorted(rows, key=lambda r: (int(r["layer"]), str(r["organ"])))
    global_q = np.zeros(4, dtype=np.int64)
    global_b = np.zeros(256, dtype=np.int64)
    global_nibble = np.zeros(16, dtype=np.int64)
    global_within = np.zeros((4, 4), dtype=np.int64)
    organs = {
        name: {"q": np.zeros(4, dtype=np.int64), "Hq": [], "Hb": [], "Hcond": []}
        for name in ("mlp.gate", "mlp.up", "mlp.down")
    }
    per_tensor: list[dict[str, Any]] = []
    gate_up: list[dict[str, float]] = []
    cross_layer: dict[str, dict[str, list[float]]] = {
        name: {"mi": [], "match": [], "indep_match": []}
        for name in ("mlp.gate", "mlp.up", "mlp.down")
    }
    sample_extra: list[dict[str, Any]] = []
    prev: dict[str, Any] = {}
    sample_set = {int(x) for x in sample_layers}
    rng = np.random.default_rng(rng_seed)
    code_bytes_sum = 0

    for rec in ordered:
        path = Path(rec["segment_path"])
        parsed = parse_hgrafv01_header(path)
        code_off = parsed["payload_off"] + parsed["scale_bytes"] + parsed["bias_bytes"]
        packed = _read_u8(path, code_off, int(parsed["code_bytes"]))
        if packed.size != int(rec["code_bytes"]):
            raise CatalogLayoutRefuse(f"{rec['name']}: short code read")
        code_bytes_sum += packed.size
        bh = np.bincount(packed, minlength=256)
        qh, nh, within = _q_and_within_from_byte_hist(bh)
        global_q += qh
        global_b += bh
        global_nibble += nh
        global_within += within
        Hq = _shannon(qh)
        Hb = _shannon(bh)
        Hc = _cond_h(within)
        organ = str(rec["organ"])
        layer = int(rec["layer"])
        organs[organ]["q"] += qh
        organs[organ]["Hq"].append(Hq)
        organs[organ]["Hb"].append(Hb)
        organs[organ]["Hcond"].append(Hc)
        p = qh.astype(np.float64) / qh.sum()
        indep_match = float((p * p).sum())
        per_tensor.append(
            {
                "layer": layer,
                "organ": organ,
                "H_q": Hq,
                "H_byte": Hb,
                "H_q_given_prev_within_byte": Hc,
            }
        )

        if organ == "mlp.up" and prev.get("_layer") == layer and "mlp.gate" in prev:
            j = _joint_aligned(prev["mlp.gate"], packed)
            gate_up.append({"layer": float(layer), "mi": _mi(j), "match": float(np.trace(j) / j.sum())})

        prev_key = f"prev_{organ}"
        if prev_key in prev:
            j = _joint_aligned(prev[prev_key], packed)
            tot = float(j.sum())
            cross_layer[organ]["mi"].append(_mi(j))
            cross_layer[organ]["match"].append(float(np.trace(j) / tot))
            cross_layer[organ]["indep_match"].append(indep_match)

        prev[organ] = packed
        prev[prev_key] = packed
        prev["_layer"] = layer

        if layer in sample_set:
            sample_extra.append(
                _sample_probe(packed, parsed, rec, rng)
            )

    reconcile_code_bytes(code_bytes_sum, detail="sum of bytes actually read from HGRAVF01")
    n_params = int(global_q.sum())
    H_q = _shannon(global_q)
    H_b = _shannon(global_b)
    H_nibble = _shannon(global_nibble)
    H_cond = _cond_h(global_within)
    mi_within = _mi(global_within)
    iid_bytes = (H_q * n_params) / 8.0
    redundant = float(CODE_BYTES_TARGET) - iid_bytes
    p_q = (global_q.astype(np.float64) / n_params).tolist()
    byte_occupied = int(np.count_nonzero(global_b))
    nibble_occupied = int(np.count_nonzero(global_nibble))

    H_l0 = [t["H_q"] for t in per_tensor if t["layer"] == 0]
    H_rest = [t["H_q"] for t in per_tensor if t["layer"] != 0]
    H_all = [t["H_q"] for t in per_tensor]

    organ_out = {}
    for name, acc in organs.items():
        q = acc["q"]
        organ_out[name] = {
            "n_tensors": len(acc["Hq"]),
            "H_q": _summ(acc["Hq"]),
            "H_byte": _summ(acc["Hb"]),
            "H_q_given_prev_within_byte": _summ(acc["Hcond"]),
            "q_hist": q.tolist(),
            "p_q": (q.astype(np.float64) / q.sum()).tolist() if int(q.sum()) else None,
        }

    zlib_ratios = [s["zlib_ratio"] for s in sample_extra]
    unique_fracs = [s["unique_frac"] for s in sample_extra]
    rowH_mins = [s["rowH_min"] for s in sample_extra]
    rowH_frac_lt_15 = [s["rowH_frac_lt_1_5"] for s in sample_extra]
    mode_masses = [s["sampled_mode_mass"] for s in sample_extra]

    return {
        "n_tensors_measured": len(ordered),
        "code_bytes_read": code_bytes_sum,
        "n_parameters": n_params,
        "n_groups": n_params // INCUMBENT_GROUP,
        "alphabet": [0, 1, 2, 3],
        "q_hist": global_q.tolist(),
        "p_q": p_q,
        "H_q_bits": H_q,
        "H_q_over_uniform": H_q / float(AFFINE_CODE_BITS),
        "H_nibble_bits": H_nibble,
        "H_byte_bits": H_b,
        "H_byte_if_iid_q": 4.0 * H_q,
        "byte_occupied_of_256": byte_occupied,
        "nibble_occupied_of_16": nibble_occupied,
        "H_q_given_prev_within_byte": H_cond,
        "mi_within_byte_bits": mi_within,
        "iid_shannon_bytes": iid_bytes,
        "iid_shannon_bytes_rounded": int(round(iid_bytes)),
        "iid_redundant_bytes": redundant,
        "iid_redundant_bytes_rounded": int(round(redundant)),
        "independent_fraction": iid_bytes / float(CODE_BYTES_TARGET),
        "nns022_reopen_uniform_frac": NNS022_REOPEN_UNIFORM_FRAC,
        "nns022_reopen_fires": (H_q / float(AFFINE_CODE_BITS)) <= NNS022_REOPEN_UNIFORM_FRAC,
        "by_organ": organ_out,
        "H_q_all_tensors": _summ(H_all),
        "H_q_layer0": _summ(H_l0),
        "H_q_later": _summ(H_rest),
        "cross_layer": {
            name: {
                "mi_bits": _summ(acc["mi"]),
                "match": _summ(acc["match"]),
                "independent_match": _summ(acc["indep_match"]),
            }
            for name, acc in cross_layer.items()
        },
        "cross_tensor_gate_vs_up": {
            "mi_bits": _summ([x["mi"] for x in gate_up]),
            "match": _summ([x["match"] for x in gate_up]),
            "n_layers": len(gate_up),
            "note": (
                "down has a transposed shape (5120 x 17408 vs 17408 x 5120); "
                "no elementwise code alignment."
            ),
        },
        "sample_layers": list(sample_layers),
        "sample_block_and_row": sample_extra,
        "sample_unique_frac": _summ(unique_fracs),
        "sample_zlib_ratio": _summ(zlib_ratios),
        "sample_rowH_min": _summ(rowH_mins),
        "sample_rowH_frac_lt_1_5": _summ(rowH_frac_lt_15),
        "sample_group_mode_mass": _summ(mode_masses),
        "per_tensor": per_tensor,
        "note": (
            "H(q) is Shannon entropy of the 4-ary alphabet over every MLP code. "
            "H(byte) is entropy of the packed 8-bit words (four consecutive codes). "
            "Conditional entropy is H(q_i | q_{i-1}) inside a packed byte. "
            "Cross-layer / gate-up MI is of aligned 2-bit symbols. zlib is a "
            "stdlib compressor on the packed stream of sampled tensors, not a "
            "production path. Unique groups are exact 16-byte (64-code) patterns."
        ),
    }


def _sample_probe(
    packed: np.ndarray,
    parsed: Mapping[str, Any],
    rec: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    ug, ng = _unique_groups(packed)
    mm, nuniq_s, ns = _mode_mass_groups(packed, rng=rng)
    z = zlib.compress(memoryview(packed), ZLIB_LEVEL)
    rows_n, cols_n = parsed["shape"]
    q = _unpack_q(packed.reshape(ng, 16))
    mat = q.reshape(rows_n, cols_n)
    counts = np.stack([(mat == k).sum(axis=1) for k in range(4)], axis=1).astype(np.float64)
    pr = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rowH = -np.nansum(np.where(pr > 0, pr * np.log2(pr), 0.0), axis=1)
    q3 = (packed[:-1] >> 6) & 3
    q0 = packed[1:] & 3
    across = np.bincount(q3.astype(np.int64) * 4 + q0.astype(np.int64), minlength=16).reshape(4, 4)
    return {
        "layer": int(rec["layer"]),
        "organ": rec["organ"],
        "unique_groups": ug,
        "n_groups": ng,
        "unique_frac": ug / ng,
        "sampled_mode_mass": mm,
        "sampled_unique": nuniq_s,
        "sampled_n": ns,
        "zlib_bytes": len(z),
        "zlib_ratio": len(z) / packed.size,
        "rowH_mean": float(rowH.mean()),
        "rowH_min": float(rowH.min()),
        "rowH_max": float(rowH.max()),
        "rowH_std": float(rowH.std()),
        "rowH_frac_lt_1_8": float((rowH < 1.8).mean()),
        "rowH_frac_lt_1_5": float((rowH < 1.5).mean()),
        "H_q_given_prev_across_byte": _cond_h(across),
        "mi_across_byte_bits": _mi(across),
    }


# ---------------------------------------------------------------------------
# Negative index. Query before proposing.
# ---------------------------------------------------------------------------


def _index_hits(family_slugs: Sequence[str]) -> list[dict[str, Any]]:
    try:
        from tools.future.negative_index import refuse_if_dead
    except Exception as exc:  # pragma: no cover
        return [{"index_error": f"{type(exc).__name__}: {exc}"}]
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slug in family_slugs:
        for organ in ("mlp", "gate", "up", "down"):
            refusal = refuse_if_dead(
                {
                    "model": "qwen3.8-27b",
                    "organ": organ,
                    "hypothesis_family": slug,
                }
            )
            if not refusal:
                continue
            key = str(refusal.get("scar_id") or "")
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "scar_id": refusal.get("scar_id"),
                    "source_path": refusal.get("source_path"),
                    "hypothesis_family": refusal.get("hypothesis_family"),
                    "organ": refusal.get("organ"),
                    "verdict": refusal.get("verdict"),
                    "claim_refuted": refusal.get("claim_refuted"),
                    "reopen_condition": refusal.get("reopen_condition"),
                    "queried_slug": slug,
                    "queried_organ": organ,
                    "applies_to_this_object": False,
                    "why_not_this_object": (
                        "Index family matches a weight-space or codec scar on "
                        "this parent. The object here is the 2-bit affine-Q2 "
                        "code body, not W, not the f16 scale/bias, and not a "
                        "different parent."
                    ),
                }
            )
    return hits


def _nns_cite(nns_id: str) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    src = NOETIC_RELS[0]
    for rel in NOETIC_RELS:
        path = REPO / rel
        text = None
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = git("show", f"HEAD:{rel}")
        if not text:
            continue
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        src = rel
        for item in doc.get("entries") or []:
            if isinstance(item, Mapping) and str(item.get("id") or "") == nns_id:
                entry = dict(item)
                break
        if entry:
            break
    scope = entry.get("scope") if isinstance(entry.get("scope"), dict) else {}
    return {
        "scar_id": nns_id,
        "source_path": src,
        "claim_refuted": str(entry.get("claim_refuted") or entry.get("capability") or ""),
        "reopen_condition": str(entry.get("reopen_condition") or ""),
        "surface": " ".join(
            p for p in (str(scope.get("model") or "").strip(), str(scope.get("organ") or "").strip()) if p
        ) or "as recorded in NOETIC_NEGATIVE_SCIENCE",
        "kind": str(entry.get("kind") or ""),
        "this_specimen": "qwen3.8-27b sealed-3.14 affine-Q2 MLP 2-bit code body",
    }


def _qn_cite(qn_id: str, claim: str, reopen: str) -> dict[str, Any]:
    return {
        "scar_id": qn_id,
        "source_path": QN_REL,
        "claim_refuted": claim,
        "reopen_condition": reopen,
        "surface": "qwen3.8-27b mlp_gate_up+mlp_down (QN catalog; abliterated sibling of this parent)",
        "kind": "MODEL_SPECIFIC",
        "this_specimen": "qwen3.8-27b sealed-3.14 affine-Q2 MLP 2-bit code body",
    }


def _remat_tag(decompresses_w: bool, ordinary: bool) -> str:
    verdict = judge_dense_rematerialization(
        {
            "path_kind": PRODUCTION,
            "dense_rematerialization": decompresses_w,
            "decompresses_to_dense_weight_tensor": decompresses_w,
            "runs_ordinary_kernels": ordinary,
            "consumes_representation_directly": (not decompresses_w),
        }
    )
    if not verdict.ok and decompresses_w:
        return REJECTED_DENSE_REMAT
    if decompresses_w:
        return REJECTED_DENSE_REMAT
    return DIRECT_CONSUME


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise CodeRefuse(f"{name} is not an atlas primitive")
    return name


# ---------------------------------------------------------------------------
# Candidates. Structure of q, not of the f16 aux and not of abstract W.
# ---------------------------------------------------------------------------


def candidates(
    acc: Mapping[str, Any],
    meas: Mapping[str, Any],
    *,
    consult_index: bool = True,
) -> list[dict[str, Any]]:
    code_b = int(acc["code_bytes"])
    n_params = int(acc["n_parameters"])
    n_tensors = int(acc["n_tensors"])
    H_q = float(meas["H_q_bits"])
    H_cond = float(meas["H_q_given_prev_within_byte"])
    H_b = float(meas["H_byte_bits"])
    H_b_iid = float(meas["H_byte_if_iid_q"])
    mi_w = float(meas["mi_within_byte_bits"])
    frac = float(meas["independent_fraction"])
    redundant = int(meas["iid_redundant_bytes_rounded"])
    floor_b = int(meas["iid_shannon_bytes_rounded"])
    Hq_u = float(meas["H_q_over_uniform"])
    reopen = bool(meas["nns022_reopen_fires"])
    p_q = meas.get("p_q")
    uniq = (meas.get("sample_unique_frac") or {}).get("min")
    zlib_r = (meas.get("sample_zlib_ratio") or {}).get("mean")
    row_min = (meas.get("sample_rowH_min") or {}).get("min")
    mode_m = (meas.get("sample_group_mode_mass") or {}).get("max")
    H_all = meas.get("H_q_all_tensors") or {}
    cl = meas.get("cross_layer") or {}
    gu = meas.get("cross_tensor_gate_vs_up") or {}
    cl_mi = {
        k: ((cl.get(k) or {}).get("mi_bits") or {}).get("mean")
        for k in ("mlp.gate", "mlp.up", "mlp.down")
    }
    cl_match = {
        k: ((cl.get(k) or {}).get("match") or {}).get("mean")
        for k in ("mlp.gate", "mlp.up", "mlp.down")
    }
    cl_indep = {
        k: ((cl.get(k) or {}).get("independent_match") or {}).get("mean")
        for k in ("mlp.gate", "mlp.up", "mlp.down")
    }
    gu_mi = (gu.get("mi_bits") or {}).get("mean")
    gu_match = (gu.get("match") or {}).get("mean")
    byte_occ = meas.get("byte_occupied_of_256")
    keep_one_layer = code_b - (code_b // 64)

    rows: list[dict[str, Any]] = [
        {
            "id": "lower_bit_native",
            "name": "lower-bit native representations of the same W",
            "mechanism": (
                "Replace the incumbent 2-bit affine-Q2 codes with a native 1-bit "
                "/ ternary / sign-code of the same weights. Not a new function "
                "for F; a thinner packing of the same q-object."
            ),
            "byte_model": (
                f"active ≈ (bits/8)*{n_params} + group scale/bias. 1-bit codes "
                f"would be {code_b // 2} code bytes vs incumbent {code_b}. "
                f"Lossless 1-bit requires H(q) ≤ 1; measured H(q)={H_q:.4f}."
            ),
            "bytes_eliminated_if_true": code_b // 2,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A native 1-bit GEMV consumes codes in-register (QN-BINARY "
                "kernel was competent). Unpack-to-Q4-then-generic-GEMV is a "
                "different lowering and is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": ALREADY_FALSIFIED,
            "support": "SCAR",
            "cheapest_falsifier": (
                "Already run: uniform Q2 MLP output rel-fro 0.578 vs q3 0.198 "
                "(NNS-029); 1.25-bpw binary body generation-incoherent, 0/4 "
                "healers (QN-BINARY-INJURY). STATIC on these arrays: H(q)="
                f"{H_q:.4f} > 1, so even a lossless 1-bit recoding of the "
                "stored q is information-theoretically impossible. Retry is "
                "not a new experiment."
            ),
            "index_slugs": ["uniform_q2", "binary_quantization", "ternary"],
            "citations": [
                _nns_cite("NNS-029"),
                _qn_cite(
                    "QN-BINARY-INJURY",
                    "the 1.25-bpw binary body is physically fast but generation-injured; 0 of 4 healing candidates reached coherent generation",
                    "a healing scheme that restores coherent generation while the healed body stays faster than q2f_g64",
                ),
            ],
            "measured": {"H_q_bits": H_q, "lossless_1bit_impossible": H_q > 1.0},
        },
        {
            "id": "heterogeneous_bit_allocation",
            "name": "heterogeneous bits across rows / organs / depth of the code body",
            "mechanism": (
                "Spend 2 bits only where the stored q has high entropy; crush "
                "low-entropy rows, organs, or layers to 1 bit. This is a map "
                "over the incumbent code stream, not a new sensitivity map of F."
            ),
            "byte_model": (
                f"sum_rows bits_r * n_cols_r / 8. A win requires a proper "
                f"subset of the {n_tensors} tensors (or their rows) with H(q) "
                "materially below 2. Measured per-tensor H(q) sits in "
                f"[{H_all.get('min')}, {H_all.get('max')}] with std "
                f"{H_all.get('std')}; sampled row-min is {row_min}."
            ),
            "bytes_eliminated_if_true": None,
            "bytes_eliminated_if_true_note": (
                "Exact only after a bit map. No row/tensor on these arrays is "
                "a 1-bit island, so the save from entropy-heterogeneity is ~0."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Each island keeps a native kernel (incumbent already mixes "
                "Q2 and Q4). Unpacking crushed rows to dense W is "
                "REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("ConditionalPhysicalProgram"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: every tensor's H(q) is in 1.850–1.874 bits; "
                f"sampled per-row min is {row_min} and frac of rows with H<1.5 "
                "is 0. There is no low-entropy island in the stored q to crush. "
                "A *capability* sensitivity map of F (requantize W, not recode "
                "q) is a different object; that is OPEN in the byte census and "
                "UNMEASURED here."
            ),
            "index_slugs": ["uniform_subbit_allocation"],
            "measured": {
                "H_q_all_tensors": H_all,
                "sample_rowH_min": meas.get("sample_rowH_min"),
                "sample_rowH_frac_lt_1_5": meas.get("sample_rowH_frac_lt_1_5"),
            },
        },
        {
            "id": "generated_tensors",
            "name": "generated tensors (emit W, then GEMV)",
            "mechanism": (
                "Store a small generator G(θ, layer, organ) that writes W at "
                "use instead of storing the 2-bit codes."
            ),
            "byte_model": (
                f"|θ| + generator program, independent of {n_params}. A win "
                f"requires |θ| << {code_b} AND production that never writes W."
            ),
            "bytes_eliminated_if_true": code_b,
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "The cheap lowering is generate-then-ordinary-GEMV. That is "
                "dense rematerialization of W and is refused as a production "
                "path. A generator that is itself the matvec (no W) is "
                "generated_programs / function_replacement."
            ),
            "physical_primitive": _require_primitive("MoveOrRecompute"),
            "status": REJECTED_DENSE_REMAT,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC: any plan whose native_execution_concept is 'emit W, "
                "then affine2 / generic GEMV' is REJECTED_DENSE_REMAT before a "
                f"fit. Independently, these codes are not a tiny G: H(q)={H_q:.4f}, "
                f"unique 16-byte groups ≥ {uniq} of n_groups, neighbour MI "
                f"{mi_w:.4f} bits. A G that emits this stream is storing it."
            ),
            "index_slugs": ["generated_tied_params"],
            "citations": [_nns_cite("NNS-015")],
            "note": (
                "NNS-015's reopen is a distilled *operator* matching F "
                "(function_replacement), not generated W. generated_tied_params "
                "canonizes to cross_expert_structure (MoE); this body has no "
                "experts — that hit is a cousin."
            ),
        },
        {
            "id": "generated_programs",
            "name": "generated programs (emit q in-register)",
            "mechanism": (
                "A tiny program of group/row/layer index emits q ∈ {0,1,2,3} "
                "in-register. Residual stored only if needed. Distinct from "
                "generated_tensors (which write W)."
            ),
            "byte_model": (
                f"O(degree + n_rows) plus residual. If the residual is the "
                f"code body, zero bytes are eliminated. Incumbent {code_b}."
            ),
            "bytes_eliminated_if_true": code_b,
            "dense_rematerialization": DEPENDS_ON_LOWERING,
            "dense_rematerialization_reason": (
                "In-register evaluation is DIRECT_CONSUME / FusedDecodeCompute. "
                "The cheap lowering — run the program, write the incumbent "
                "2-bit buffer, bind affine2 — eliminates zero active bytes."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: H(q|neighbour)="
                f"{H_cond:.4f} vs H(q)={H_q:.4f} (MI {mi_w:.4f} bits); unique "
                f"16-byte groups ≥ {uniq}; sampled mode mass ≤ {mode_m}. A "
                "smooth / periodic / low-complexity program of index cannot "
                "be this stream. A program that memorizes q is not tiny."
            ),
            "index_slugs": ["generated_tied_params"],
            "measured": {
                "mi_within_byte_bits": mi_w,
                "sample_unique_frac": meas.get("sample_unique_frac"),
                "sample_group_mode_mass": meas.get("sample_group_mode_mass"),
            },
        },
        {
            "id": "shared_code_bases",
            "name": "shared bases across layers / organs of the code body",
            "mechanism": (
                "One (or K) shared code templates B with local coefficients, "
                "so q_{l,o} ≈ B C_{l,o}. Would store B once plus small C "
                "instead of 192 independent 2-bit arrays. This body is dense "
                "SwiGLU, not MoE — 'across experts' does not apply."
            ),
            "byte_model": (
                f"|B|*K + sum |C|. Incumbent {code_b}. A K=1 shared template "
                "is the same claim as 'codes repeat across layers'."
            ),
            "bytes_eliminated_if_true": code_b,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Fused shared-basis matvec is a direct consumer (QN-SHARED-BASIS "
                "kernel was competent). Writing dense W is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: aligned-symbol I(q_l; q_{l+1}) is "
                f"{cl_mi} bits; match rates {cl_match} equal the independent "
                f"baseline {cl_indep}. Gate↔up I(q_g; q_u)={gu_mi} bits, match "
                f"{gu_match}. The codes do not share a basis. QN-SHARED-BASIS-"
                "DENSITY is a cousin on W, not this measurement."
            ),
            "index_slugs": ["shared_basis", "qn_shared_k_hybrid"],
            "citations": [
                _qn_cite(
                    "QN-SHARED-BASIS-DENSITY",
                    "the KERNEL is competent and the byte win does translate to nanoseconds, but no K below ~2.25 bpw composes coherently for the MLP: the local functional probe dies at held-out activation",
                    "a shared-basis point that is coherent at held-out activation AND beats q2f on both density and COMPLETE_TOKEN_NS",
                ),
            ],
            "cousin_not_this_object": True,
            "measured": {
                "cross_layer_mi_bits": cl_mi,
                "cross_layer_match": cl_match,
                "gate_up_mi_bits": gu_mi,
                "gate_up_match": gu_match,
            },
        },
        {
            "id": "factorized_programs",
            "name": "factorized programs (low-rank / Kronecker of the code matrix)",
            "mechanism": (
                "q ≈ UV over the integers / a small alphabet, consumed as two "
                "skinny matvecs. Distinct from W-space SVD (NNS-016)."
            ),
            "byte_model": (
                f"per organ, r*(m+n)*code_bits/8. Gate/up are 17408×5120; a "
                f"rank that preserves H(q)={H_q:.4f} bits/entry is not a save "
                "when neighbour MI is ~0.002 bits."
            ),
            "bytes_eliminated_if_true": None,
            "bytes_eliminated_if_true_note": (
                "Exact only after choosing r. The measured local MI says r that "
                "preserves q is not a byte win."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Two skinny matvecs consume the factors directly. Materializing "
                "U@V into W or into q then running affine2 is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("TiledProjection"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: neighbour MI of q is "
                f"{mi_w:.4f} bits; H(byte)={H_b:.4f} vs 4·H(q)={H_b_iid:.4f}, "
                "so even 4-tuples are almost independent. A factorizable code "
                "matrix would show itself as local dependence. NNS-016 / "
                "QN-LOWRANK-HEALING are cousins on W, not a retry licence."
            ),
            "index_slugs": ["low_rank", "kronecker", "global_dense_lowrank"],
            "citations": [
                _nns_cite("NNS-016"),
                _qn_cite(
                    "QN-LOWRANK-HEALING",
                    "no distributed correction under the 1.0 bpw budget restored held-out activations on real X; even r=256 at 1.035 extra bpw pushed the body to 2.285 > 2.25 with rel_fro 0.4798",
                    "a correction whose extra bpw keeps the body under 2.25 while rel_fro on real held-out X drops below the q2f baseline",
                ),
            ],
            "cousin_not_this_object": True,
            "measured": {"mi_within_byte_bits": mi_w, "H_byte_bits": H_b, "H_byte_if_iid_q": H_b_iid},
        },
        {
            "id": "dictionary_of_code_blocks",
            "name": "dictionary of 16-byte (64-code) groups",
            "mechanism": (
                "Replace each group of 64 2-bit codes with an index into a "
                "codebook of observed 16-byte patterns. Native consume looks "
                "up a codeword; it does not store q."
            ),
            "byte_model": (
                f"n_groups * ceil(log2(K))/8 + K * 16. n_groups="
                f"{n_params // INCUMBENT_GROUP}. A win requires K << n_groups."
            ),
            "bytes_eliminated_if_true": code_b,
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "The cheap post-hoc lowering expands indices into a dense W "
                "(or into the incumbent 2-bit buffer) and runs ordinary GEMV. "
                "A native codebook-in-register kernel is a different lowering "
                "and is still blocked here by uniqueness: K ≈ n_groups."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: unique 16-byte groups / n_groups ≥ "
                f"{uniq} on sampled layers (0, 21, 42, 63) × 3 organs; sampled "
                f"mode mass ≤ {mode_m}. The codebook *is* the array. NNS-017 "
                "is a cousin (PQ of raw frozen W), not this 16-byte q-group object."
            ),
            "index_slugs": ["raw_weight_pq_vq", "post_hoc_frozen_codec", "learned_codebook"],
            "citations": [_nns_cite("NNS-017")],
            "cousin_not_this_object": True,
            "measured": {
                "sample_unique_frac": meas.get("sample_unique_frac"),
                "sample_group_mode_mass": meas.get("sample_group_mode_mass"),
            },
        },
        {
            "id": "product_codebooks",
            "name": "product codebooks of the 2-bit stream",
            "mechanism": (
                "A block of q is a tuple of codebook indices, reconstructed "
                "as a sum / concat of codewords. On this packing the natural "
                "4-code block is the packed byte (256-ary)."
            ),
            "byte_model": (
                f"M * (n_blocks * ceil(log2(K))/8 + K * d/M). For M=1, d=4 "
                f"(the packed byte): K≤256, and measured H(byte)={H_b:.4f} of "
                f"8 with occupancy {byte_occ}/256. Entropy coding of those "
                "indices is entropy_coded_code_stream."
            ),
            "bytes_eliminated_if_true": redundant,
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "Same cheap lowering as dictionary_of_code_blocks: expand "
                "product codes to dense W (or to incumbent q), then ordinary GEMV."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: H(byte)="
                f"{H_b:.4f} vs 4·H(q)={H_b_iid:.4f} (gap {H_b_iid - H_b:.4f} "
                f"bits / 4-tuple); occupancy {byte_occ}/256. Product structure "
                "beyond the 4-level histogram bias is not in the stream. "
                "NNS-017 / NNS-022 are cousins on PQ-of-W / rANS-of-PQ-indices."
            ),
            "index_slugs": ["raw_weight_pq_vq", "entropy_coded_pq"],
            "citations": [_nns_cite("NNS-017"), _nns_cite("NNS-022")],
            "cousin_not_this_object": True,
            "measured": {
                "H_byte_bits": H_b,
                "H_byte_if_iid_q": H_b_iid,
                "byte_occupied_of_256": byte_occ,
            },
        },
        {
            "id": "lowrank_plus_sparse_residual",
            "name": "low-rank plus sparse residual of the code body",
            "mechanism": (
                "Cheap backbone (a low-complexity q, or the incumbent) plus a "
                "sparse residual of groups/rows that restores q. "
                "y = backbone(x) + R(x) if consumed in function space; here "
                "the question is whether q itself is a sparse residual of a "
                "simple pattern."
            ),
            "byte_model": (
                f"backbone_bytes + nnz*(index+value). A win requires the "
                f"residual of a cheap backbone to be sparse. Sampled 16-byte "
                f"mode mass ≤ {mode_m} — there is no dominant template."
            ),
            "bytes_eliminated_if_true": None,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Residual applied in the same kernel is a direct consumer. "
                "Reconstructing W_backbone + W_R as dense W is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: no 16-byte group repeats enough to be a "
                f"backbone (mode mass ≤ {mode_m}, unique frac ≥ {uniq}). "
                "NNS-015 / QN-LOWRANK-HEALING killed the W-space version as a "
                "density win; they are cousins, not this q-residual."
            ),
            "index_slugs": ["low_rank", "residual_codebook"],
            "citations": [_nns_cite("NNS-015"), _nns_cite("NNS-014")],
            "cousin_not_this_object": True,
            "measured": {
                "sample_group_mode_mass": meas.get("sample_group_mode_mass"),
                "sample_unique_frac": meas.get("sample_unique_frac"),
            },
        },
        {
            "id": "block_generators",
            "name": "block generators of 64-code groups",
            "mechanism": (
                "A small family of programs emits each 16-byte group from a "
                "short seed (group index, row index, a few bits). Distinct "
                "from a stored dictionary: the codeword is computed, not looked up."
            ),
            "byte_model": (
                f"n_groups * seed_bits/8 + program. A win requires seed_bits "
                f"<< 128 (the 16-byte group). Unique frac ≥ {uniq} says the "
                "seed would have to be the group."
            ),
            "bytes_eliminated_if_true": code_b,
            "dense_rematerialization": DEPENDS_ON_LOWERING,
            "dense_rematerialization_reason": (
                "In-register generation of a 64-code group is DIRECT_CONSUME. "
                "Writing the incumbent 2-bit buffer then binding affine2 is "
                "not an active-byte win."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: unique 16-byte groups / n_groups ≥ "
                f"{uniq}; zlib ratio ≈ {zlib_r} (no long repeats); neighbour "
                f"MI {mi_w:.4f} bits. A block generator that is not a dictionary "
                "of the groups has nothing to generate from."
            ),
            "index_slugs": ["generated_tied_params"],
            "measured": {
                "sample_unique_frac": meas.get("sample_unique_frac"),
                "sample_zlib_ratio": meas.get("sample_zlib_ratio"),
            },
        },
        {
            "id": "cross_layer_code_prediction",
            "name": "cross-layer prediction of the 2-bit codes",
            "mechanism": (
                "q_l = P(q_{l-1}) + Δ_l with P = identity or a small map. "
                "Store layer-0 codes plus residuals. A win requires "
                "H(Δ) << H(q)."
            ),
            "byte_model": (
                f"q_0 + sum |Δ_l|. Incumbent {code_b}. If Δ ≈ q, zero save. "
                f"Keeping one layer and predicting the rest would claim "
                f"{keep_one_layer} bytes eliminated."
            ),
            "bytes_eliminated_if_true": keep_one_layer,
            "dense_rematerialization": DEPENDS_ON_LOWERING,
            "dense_rematerialization_reason": (
                "Predicting q_l into the incumbent 2-bit buffer then running "
                "affine2 is not an active-byte win. Direct consume would "
                "evaluate P in-register. Predicting W_l is generated_tensors "
                "and REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": MEASURED_NEGATIVE,
            "support": "MEASURED",
            "cheapest_falsifier": (
                "STATIC, run here: I(q_l; q_{l+1}) is "
                f"{cl_mi} bits; match {cl_match} vs independent baseline "
                f"{cl_indep}. Residuals are the parent. The byte-census left "
                "this OPEN on W; the *codes* are now measured. Layer 0 is not "
                "a low-entropy exception in q (H_L0={meas.get('H_q_layer0')})."
            ),
            "index_slugs": ["cross_expert_structure", "cross_layer_weight_delta"],
            "citations": [_nns_cite("NNS-016")],
            "cousin_not_this_object": True,
            "measured": {
                "cross_layer": cl,
                "H_q_layer0": meas.get("H_q_layer0"),
                "H_q_later": meas.get("H_q_later"),
            },
            "note": (
                "cross_expert_structure is a MoE scar. This body has no routed "
                "experts; that hit is a cousin. NNS-016's full-rank depth is "
                "the W-space cousin of this code-space measurement."
            ),
        },
        {
            "id": "capability_sensitive_literal_islands",
            "name": "capability-sensitive literal islands in the code body",
            "mechanism": (
                "Keep a sensitivity-selected subset of rows/layers/organs as "
                "literal 2-bit (or higher) tensors; crush the rest. The island "
                "is the capability, the bulk is packing."
            ),
            "byte_model": (
                f"island_frac * full + (1-island_frac) * bulk. A win requires "
                f"island_frac small. Measured H(q) is homogeneous across all "
                f"{n_tensors} tensors (std {H_all.get('std')}); there is no "
                "entropy-island to keep."
            ),
            "bytes_eliminated_if_true": None,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Island and bulk each have a native kernel. Healing that remats "
                "the binary body to dense W around the island is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("ConditionalPhysicalProgram"),
            "status": ALREADY_FALSIFIED,
            "support": "SCAR",
            "cheapest_falsifier": (
                "Already run as binary-body + high-precision islands "
                "(QN-BINARY-HEALING): injury is broad, not localized; 0/4 "
                "reached coherent generation. STATIC on these arrays: H(q) does "
                "not localize either (per-tensor std "
                f"{H_all.get('std')}, sampled row-min {row_min}). Reopen needs "
                "a sensitivity map of F cheaper than just using the affine-Q2 "
                "body, not another entropy map of q."
            ),
            "index_slugs": ["protected_islands", "qn_binary_healing"],
            "citations": [
                _qn_cite(
                    "QN-BINARY-HEALING",
                    "the injury is broad, not localized: no small protected island cheaply restored it; 0/4 candidates reached coherent generation",
                    "a sensitivity map that localizes the injury to a region small enough that protecting it costs less than the 2.25-bpw q2f body",
                ),
                _qn_cite(
                    "QN-BINARY-INJURY",
                    "the 1.25-bpw binary body is physically fast but generation-injured; 0 of 4 healing candidates reached coherent generation",
                    "a healing scheme that restores coherent generation while the healed body stays faster than q2f_g64",
                ),
            ],
            "measured": {"H_q_all_tensors": H_all, "sample_rowH_min": meas.get("sample_rowH_min")},
        },
        {
            "id": "shared_input_transforms",
            "name": "shared input transforms (activation-space V, local readout)",
            "mechanism": (
                "One shared V on the MLP input, organ- or layer-local readout: "
                "y_{l,o} = W'_{l,o} (V x). This is a property of F, not of q. "
                "The code arrays cannot support or kill it."
            ),
            "byte_model": (
                f"|V| (5120×r) + 64*3 of W' with inner dim r. Byte win iff "
                f"r << 5120 AND the readouts stay cheaper than incumbent "
                f"{code_b} code bytes + aux."
            ),
            "bytes_eliminated_if_true": None,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "y = W'(V x) is two matvecs. Materializing W'V into W is "
                "REJECTED_DENSE_REMAT and would also erase the sharing."
            ),
            "physical_primitive": _require_primitive("TiledProjection"),
            "status": UNMEASURED,
            "support": "UNMEASURED",
            "cheapest_falsifier": (
                "Not a code-stream probe. CHEAP CPU: PCA / ridge V on real "
                "post-norm X pooled across a few layers; reconstruct gate and "
                "up held-out. If the r that meets incumbent rel-fro is ≈ hidden, "
                "there is no shared input to store once. Do not cite QN-SHARED-"
                "BASIS (weight-space B, C) or the Flash L4 rival-codec screen "
                "as a kill of this object."
            ),
            "index_slugs": ["shared_basis"],
            "note": (
                "Attractive in the byte census (OPEN) and still unmeasured on "
                "activations. High entropy of q is consistent with 'W is not "
                "a shared template' and does not speak to a shared V on X. "
                "Index slug shared_basis will fire QN-SHARED-BASIS; that scar "
                "is unconditioned weight-space sharing, a cousin."
            ),
        },
        {
            "id": "latent_routed_accumulation",
            "name": "latent routed accumulation (narrow / routed SwiGLU)",
            "mechanism": (
                "Accumulate the SwiGLU in a latent of width m << 17408, then "
                "expand once, or route a token-dependent subset of the 17408. "
                "F(x) ≈ down_m(silu(gate_m(x)) * up_m(x)). This eliminates "
                "columns of q by not having them, not by coding them."
            ),
            "byte_model": (
                f"64 * (2*m*5120 + 5120*m) * incumbent_bpw/8. NNS-013: matching "
                "q3's held-out needs m ~ 10000–12000, at which bytes approach q3."
            ),
            "bytes_eliminated_if_true": None,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A narrower SwiGLU is a direct kernel. Expanding the latent to "
                "a 17408-wide dense W each token is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("DirectRoutedAccumulate"),
            "status": ALREADY_FALSIFIED,
            "support": "SCAR",
            "cheapest_falsifier": (
                "Already run as G3 / narrow shared grouped SwiGLU (NNS-013 "
                "property kill). The codes being high-entropy does not reopen "
                "it. Reopen is a full-width structured nonlinear "
                "(function_replacement), not a retry of m<17408, and not a "
                "recoding of these q."
            ),
            "index_slugs": [],
            "citations": [_nns_cite("NNS-013"), _nns_cite("NNS-012")],
        },
        {
            "id": "function_replacement",
            "name": "function replacement (stop storing q; represent F)",
            "mechanism": (
                "Stop representing W / q. Represent F_l itself with a cheaper "
                "program: full-width structured nonlinear (Monarch, butterfly), "
                "a distilled small net, a kernel not equal to three affines. "
                "Narrow bottleneck replacement is latent_routed_accumulation (dead)."
            ),
            "byte_model": (
                f"|program_l| * 64, independent of {n_params}. A win is "
                f"|program| << {code_b} at held-out F and at generate."
            ),
            "bytes_eliminated_if_true": code_b,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "The program is the kernel. A replacement that emits W and "
                "runs GEMV is generated_tensors and REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": UNMEASURED,
            "support": "UNMEASURED",
            "cheapest_falsifier": (
                "Not a code-stream probe. Do not retry m<17408 (NNS-013). "
                "CHEAP CPU: one full-width Monarch/butterfly (or a distilled "
                "operator) on a single layer's real post-norm X, held-out "
                "rel-fro vs affine-Q2, with a byte ledger that does not remat "
                "W. High H(q) says the *current packing* of F is close to "
                "incompressible; it does not say F cannot be a cheaper program. "
                "NNS-015's reopen is exactly this probe, and it has not been run."
            ),
            "index_slugs": [],
            "citations": [_nns_cite("NNS-013"), _nns_cite("NNS-012"), _nns_cite("NNS-015")],
            "note": (
                "Attractive, not supported. The measured entropy floor is a "
                "statement about q, not about F. Narrow replacement is "
                "ALREADY_FALSIFIED (latent_routed_accumulation)."
            ),
        },
        {
            "id": "entropy_coded_code_stream",
            "name": "entropy coding of the 2-bit code stream (the lossless floor)",
            "mechanism": (
                "The 4-ary histogram is not uniform "
                f"(p≈{p_q}). Store q with a Shannon / Huffman / rANS code of "
                "that histogram, optionally conditioned on the previous symbol. "
                "Affine2 would consume the entropy-coded stream in-register."
            ),
            "byte_model": (
                f"n_params * H / 8. i.i.d. H={H_q:.6f} bits → {floor_b} bytes "
                f"vs incumbent {code_b}. Markov-1 H={H_cond:.6f} bits, a "
                f"{H_q - H_cond:.4f}-bit improvement. zlib on sampled tensors "
                f"reaches ratio ≈ {zlib_r}."
            ),
            "bytes_eliminated_if_true": redundant,
            "dense_rematerialization": DEPENDS_ON_LOWERING,
            "dense_rematerialization_reason": (
                "A fused rANS/Huffman affine2 is DIRECT_CONSUME / "
                "FusedDecodeCompute and would actually drop active bytes. "
                "The cheap lowering — decode to the incumbent 2-bit buffer, "
                "bind affine2 — eliminates zero active bytes and is not this "
                "candidate."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": OPEN,
            "support": "SHANNON_GAP_MEASURED_KERNEL_UNMEASURED",
            "cheapest_falsifier": (
                "STATIC, already run: the i.i.d. gap is "
                f"{redundant} bytes ({1 - frac:.4%} of the code body), "
                f"H(q)/2={Hq_u:.4f} of uniform. NNS-022 reopen is ≤ "
                f"{NNS022_REOPEN_UNIFORM_FRAC} of uniform; it does not fire "
                f"(fires={reopen}). Neighbour conditioning adds {mi_w:.4f} bits, "
                "not a second lever. Kill as a *4.28 GB* attack: the floor is "
                f"{floor_b} bytes, still 4.00 GB. Kill as a TPS lever without "
                "a native register-decodable path (NNS-022). Remain OPEN only "
                "as a fused-decode micro-lever of ~278 MB, capability UNMEASURED."
            ),
            "index_slugs": ["entropy_coded_pq"],
            "citations": [_nns_cite("NNS-022")],
            "cousin_not_this_object": True,
            "note": (
                "NNS-022 killed rANS on Lloyd-optimal *PQ indices* as an "
                "active-byte/TPS lever, reopen if index entropy ≤ 0.9 of "
                "uniform. These 2-bit affine codes are a different object "
                "(histogram bias from LS-on-roughly-Gaussian W, not a PQ "
                "codebook). Cited as a cousin so the 278 MB Shannon gap is "
                "not laundered as a TPS win, and so a 10–25% claim is not "
                "re-proposed."
            ),
            "measured": {
                "H_q_bits": H_q,
                "H_q_over_uniform": Hq_u,
                "H_q_given_prev_within_byte": H_cond,
                "mi_within_byte_bits": mi_w,
                "iid_redundant_bytes_rounded": redundant,
                "independent_fraction": frac,
                "nns022_reopen_fires": reopen,
                "sample_zlib_ratio": meas.get("sample_zlib_ratio"),
                "p_q": p_q,
            },
        },
    ]

    have = [r["id"] for r in rows]
    if have != list(REQUIRED_CANDIDATE_IDS):
        raise CodeRefuse(f"candidate catalog {have} != required {list(REQUIRED_CANDIDATE_IDS)}")
    overlap = set(have) & set(AUXILIARY_DEAD_IDS)
    if overlap:
        raise CodeRefuse(f"restated auxiliary sharing families: {sorted(overlap)}")

    for row in rows:
        if row["dense_rematerialization"] == REJECTED_DENSE_REMAT:
            tag = _remat_tag(True, True)
            if tag != REJECTED_DENSE_REMAT:
                raise CodeRefuse(f"{row['id']}: expected REJECTED_DENSE_REMAT, got {tag}")
        if row["dense_rematerialization"] != REJECTED_DENSE_REMAT:
            if row.get("physical_primitive") not in ATLAS_PRIMITIVES:
                raise CodeRefuse(f"{row['id']} missing atlas primitive")
        if row["status"] == REJECTED_DENSE_REMAT and row["dense_rematerialization"] != REJECTED_DENSE_REMAT:
            raise CodeRefuse(f"{row['id']}: status REJECTED_DENSE_REMAT but lowering is not")
        row["evidence_class"] = "STATIC_ONLY"
        row["gpu_authority"] = False
        if consult_index:
            slugs = list(row.get("index_slugs") or [])
            row["index_refusals"] = _index_hits(slugs) if slugs else []
        else:
            row["index_refusals"] = []
        if row["status"] not in {ALREADY_FALSIFIED, REJECTED_DENSE_REMAT} and row.get("index_refusals"):
            row["index_hits_are_cousins"] = True
    return rows


def answers(acc: Mapping[str, Any], meas: Mapping[str, Any], cands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cands}
    floor_b = int(meas["iid_shannon_bytes_rounded"])
    redundant = int(meas["iid_redundant_bytes_rounded"])
    frac = float(meas["independent_fraction"])
    return {
        "how_much_of_the_code_body_is_independent": {
            "answer": (
                f"{frac:.4%} of the 4,278,190,080 code bytes is independent "
                f"information at the measured i.i.d. Shannon bound "
                f"({floor_b} bytes). Neighbours, layers, and gate/up/down add "
                "essentially nothing. The leftover "
                f"{redundant} bytes ({1 - frac:.4%}) is the 4-level histogram "
                "bias (p≈0.146/0.354/0.355/0.146), not sharing."
            ),
            "stored_code_bytes": int(acc["code_bytes"]),
            "iid_shannon_bytes_rounded": floor_b,
            "iid_redundant_bytes_rounded": redundant,
            "independent_fraction": frac,
            "H_q_bits": meas.get("H_q_bits"),
            "H_q_given_prev_within_byte": meas.get("H_q_given_prev_within_byte"),
            "mi_within_byte_bits": meas.get("mi_within_byte_bits"),
            "p_q": meas.get("p_q"),
        },
        "is_entropy_coding_a_4gb_lever": {
            "answer": (
                "NO. The lossless floor is still 4.00 GB. The 278 MB i.i.d. "
                "gap is a fused-decode micro-lever (OPEN, kernel UNMEASURED) "
                "and is below NNS-022's 0.9-of-uniform reopen."
            ),
            "status": by_id["entropy_coded_code_stream"]["status"],
            "nns022_reopen_fires": meas.get("nns022_reopen_fires"),
            "bytes_eliminated_if_true": by_id["entropy_coded_code_stream"]["bytes_eliminated_if_true"],
        },
        "do_the_codes_share_across_layers_or_organs": {
            "answer": "NO. Aligned-symbol mutual information is ~1e-5 bits across layers and ~1e-4 bits gate↔up. Match rates equal the independent baseline.",
            "status": by_id["shared_code_bases"]["status"],
            "cross_layer": meas.get("cross_layer"),
            "gate_vs_up": meas.get("cross_tensor_gate_vs_up"),
        },
        "can_a_dictionary_or_block_generator_win": {
            "answer": "NO. 16-byte groups are essentially unique; zlib ratio ≈ 0.95 tracks the Shannon histogram, not long repeats.",
            "dictionary_status": by_id["dictionary_of_code_blocks"]["status"],
            "block_generator_status": by_id["block_generators"]["status"],
            "sample_unique_frac": meas.get("sample_unique_frac"),
            "sample_zlib_ratio": meas.get("sample_zlib_ratio"),
        },
        "can_a_tiny_program_generate_the_codes": {
            "answer": "NO. Neighbour MI is 0.002 bits and groups do not repeat. Generating the residual is storing q.",
            "generated_tensors": by_id["generated_tensors"]["status"],
            "generated_programs": by_id["generated_programs"]["status"],
            "generated_tensors_remat": by_id["generated_tensors"]["dense_rematerialization"],
        },
        "are_some_rows_or_layers_lower_entropy": {
            "answer": "NO at a scale that would change bits. Per-tensor H(q) std is ~0.003 bits; no sampled row has H<1.5; layer 0 is slightly *higher* entropy than later layers.",
            "status": by_id["heterogeneous_bit_allocation"]["status"],
            "H_q_all_tensors": meas.get("H_q_all_tensors"),
            "H_q_layer0": meas.get("H_q_layer0"),
            "H_q_later": meas.get("H_q_later"),
            "sample_rowH_min": meas.get("sample_rowH_min"),
        },
        "what_the_codes_do_not_measure": {
            "answer": (
                "shared_input_transforms and function_replacement are properties "
                "of F / of activations, not of q. They stay UNMEASURED on this "
                "object. Attractive is not supported."
            ),
            "shared_input_transforms": by_id["shared_input_transforms"]["status"],
            "function_replacement": by_id["function_replacement"]["status"],
        },
    }


# ---------------------------------------------------------------------------
# Snapshot / receipt.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _measured() -> tuple[dict[str, Any], dict[str, Any]]:
    """Catalog + code pass. Cached so pytest and --build share one 4.28 GB read."""
    root = resolve_artifact_root()
    rows = code_rows(root=root)
    acc = accounting_from_rows(rows)
    acc["identity"] = _identity(root)
    meas = measure_code_arrays(rows)
    return acc, meas


def snapshot(consult_index: bool = True) -> dict[str, Any]:
    acc, meas = _measured()
    cands = candidates(acc, meas, consult_index=consult_index)
    return {
        "accounting": acc,
        "measurements": meas,
        "candidates": cands,
        "answers": answers(acc, meas, cands),
        "floor": {
            "stored_code_bytes": int(acc["code_bytes"]),
            "iid_shannon_bits_per_code": meas["H_q_bits"],
            "iid_shannon_bytes_rounded": meas["iid_shannon_bytes_rounded"],
            "iid_redundant_bytes_rounded": meas["iid_redundant_bytes_rounded"],
            "independent_fraction": meas["independent_fraction"],
            "markov1_bits_per_code": meas["H_q_given_prev_within_byte"],
            "context_adds_bits": meas["mi_within_byte_bits"],
            "verdict": (
                f"{meas['iid_shannon_bytes_rounded']} of {int(acc['code_bytes'])} "
                "bytes is independent information at the measured i.i.d. Shannon "
                "bound. Neighbours, layers, and gate/up/down add essentially "
                "nothing. The leftover "
                f"{meas['iid_redundant_bytes_rounded']} bytes is the 4-level "
                "histogram bias, not sharing."
            ),
        },
    }


def build(*, consult_index: bool = True) -> Path:
    snap = snapshot(consult_index=consult_index)
    acc = snap["accounting"]
    cands = snap["candidates"]
    n_open = sum(1 for c in cands if c["status"] == OPEN)
    n_meas = sum(1 for c in cands if c["status"] == MEASURED_NEGATIVE)
    n_dead = sum(1 for c in cands if c["status"] == ALREADY_FALSIFIED)
    n_unm = sum(1 for c in cands if c["status"] == UNMEASURED)
    n_remat = sum(1 for c in cands if c["dense_rematerialization"] == REJECTED_DENSE_REMAT)
    n_status_remat = sum(1 for c in cands if c["status"] == REJECTED_DENSE_REMAT)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Measure how much of the 4,278,190,080 MLP 2-bit code bytes of "
            "sealed-3.14 is independent information, on the real HGRAVF01 "
            "code arrays, not in the abstract, and not by restating the "
            "auxiliary scale/bias sharing kills."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "what_this_does_not_prove": [
            "capability of a fused rANS/Huffman affine2 (no generate gate, no kernel)",
            "capability of function_replacement or shared_input_transforms (UNMEASURED on this object)",
            "physical EBPW of a different packing",
            "actual_read_bytes_per_token (cache, contention)",
            "that H(q)=1.87 bits is inaudible at generate if a bit is dropped",
        ],
        "accounting": _py(acc),
        "measurements": _py(snap["measurements"]),
        "floor": _py(snap["floor"]),
        "candidates": _py(cands),
        "answers": _py(snap["answers"]),
        "candidate_counts": {
            "n": len(cands),
            "open": n_open,
            "measured_negative": n_meas,
            "already_falsified": n_dead,
            "unmeasured": n_unm,
            "status_rejected_dense_remat": n_status_remat,
            "rejected_dense_remat": n_remat,
        },
        "open_byte_levers": [
            {
                "id": c["id"],
                "bytes_eliminated_if_true": c.get("bytes_eliminated_if_true"),
                "status": c["status"],
                "support": c.get("support"),
            }
            for c in cands
            if c["status"] == OPEN
        ],
        "unmeasured_attractive": [
            {
                "id": c["id"],
                "status": c["status"],
                "support": c.get("support"),
            }
            for c in cands
            if c["status"] == UNMEASURED
        ],
        "auxiliary_families_not_restated": list(AUXILIARY_DEAD_IDS),
        "recovered_implementation": {
            "catalog_format": "HQ38M20 + HGRAVF01 affine_q2_group64_fp16_scale_bias",
            "artifact_root": acc["identity"]["artifact_root"],
            "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            "reconstruction": "w = float(q)*scale + bias, q unsigned in {0,1,2,3}",
            "object": "2-bit code payload, not f16 scale/bias",
        },
        "gaps_closed": [
            "code bytes re-measured from 192 HGRAVF01 headers and from the bytes actually read, refused unless they sum to 4,278,190,080",
            "entropy, conditional entropy, cross-layer MI, gate-up MI, unique 16-byte groups, per-row H, and zlib taken on the real code arrays",
            "lossless information floor reported as bytes, not as a vibe",
            "nine auxiliary scale/bias sharing families not restated",
            "negative_index queried; W-space scars cited as cousins, not laundered as code-array kills except where the scar is this object",
        ],
        "negative_findings": [
            "4.00 of 4.28 GB of q is independent at the i.i.d. Shannon bound",
            "p(q) ≈ (0.146, 0.354, 0.355, 0.146): a 4-level histogram bias, not uniform and not a shared template",
            "neighbour MI ≈ 0.002 bits; Markov-1 is not a second lever",
            "cross-layer MI ≈ 1e-5 bits; gate↔up MI ≈ 1e-4 bits; match = independent baseline",
            "16-byte groups are essentially unique; there is no group dictionary",
            "per-tensor H(q) std ≈ 0.003 bits; no entropy islands to crush",
            "the only OPEN byte lever on this 4.28 GB is fused entropy coding of the 278 MB histogram gap, kernel UNMEASURED, NNS-022 cousin for TPS",
        ],
        "nomenclature": {
            "already_falsified": ALREADY_FALSIFIED,
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "unmeasured": UNMEASURED,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "direct_consume": DIRECT_CONSUME,
            "depends_on_lowering": DEPENDS_ON_LOWERING,
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--accounting-only", action="store_true")
    args = parser.parse_args(argv_list)
    if args.accounting_only:
        snap = accounting()
        json.dump(
            {
                "code_bytes": snap["code_bytes"],
                "n_parameters": snap["n_parameters"],
                "n_tensors": snap["n_tensors"],
                "reconciled": snap["reconciled"],
            },
            _sys.stdout,
            indent=2,
        )
        _sys.stdout.write("\n")
        return 0
    if args.build or args.selftest or not argv_list:
        out = build()
        print(out)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))
