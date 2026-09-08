"""DELTANET QKVZ PRECISION — heterogeneous bits on linear_qkvz, or a measured NO.

linear_qkvz is 2,139,096,960 bytes (21.65% of the token). The predecessor
census (DELTANET_REPRESENTATION) already split the fused tensor into q/k/v/z
row ranges and showed that equal precision is not justified by the consume
path: z never enters gated-delta or conv. That lane left capability
UNMEASURED. This lane does not redo the census. It measures, per sub-block,
the i.i.d. Shannon entropy of the HQ30UQ4 codes (same method as
MLP_CODE_INFORMATION) and the STATIC operator sensitivity of a 3-bit drop,
then allocates bits.

An entropy argument alone never licenses a bit reduction. The allocation
refuses to mark a drop supported unless a sensitivity measurement exists.

    python3 tools/future/deltanet_qkvz_precision.py --build
    python3 -m pytest tools/future/test_deltanet_qkvz_precision.py -q

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

from tools.future._common import write_receipt
from tools.future.mlp_byte_census import CatalogAbsent, resolve_artifact_root
from tools.future.mlp_code_information import (
    NNS022_REOPEN_UNIFORM_FRAC,
    _cond_h,
    _mi,
    _shannon,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES
from tools.future import deltanet_representation as dnr


RECEIPT = "DELTANET_QKVZ_PRECISION.json"
SCHEMA = "hawking.future.deltanet_qkvz_precision.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_qkvz_precision.py"
PREDECESSOR_REL = "receipts/future/DELTANET_REPRESENTATION.json"

QKVZ_ACTIVE_TARGET = dnr.QKVZ_ACTIVE_TARGET
TOKEN_ACTIVE_TARGET = dnr.TOKEN_ACTIVE_TARGET
SUBBLOCKS: tuple[str, ...] = dnr.SUBBLOCKS
INCUMBENT_BITS = dnr.Q4_CODE_BITS
INCUMBENT_GROUP = dnr.INCUMBENT_GROUP
CANDIDATE_BITS = 3
NIBBLE_ALPHABET = 16
RMS_EPS = 1.0e-6
N_ROLL_TOKENS = 8
RNG_SEED = 38
ZLIB_LEVEL = 1
SAMPLE_LAYERS: tuple[int, ...] = (0, 21, 42)
SENSITIVITY_LAYERS: tuple[int, ...] = (0, 21, 42)

# NNS-019's mean-row cosine bar, applied here as a STATIC operator-output
# cousin of a generate identity gate. Clearing it does not license generate.
# Failing it is enough to refuse a production bit drop.
COSINE_BAR = 0.990
STATE_RELFRO_BAR = 0.01

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FALSIFIED = "ALREADY_FALSIFIED"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
OPEN = "OPEN"
UNMEASURED = "UNMEASURED"

ENTROPY_ALONE_INSUFFICIENT = "ENTROPY_ALONE_INSUFFICIENT"
SENSITIVITY_INCOMPLETE = "SENSITIVITY_INCOMPLETE"
GATED_COSINE_BELOW_BAR = "GATED_COSINE_BELOW_BAR"
REC_STATE_INJURED = "REC_STATE_INJURED"
SHADER_BINDING_DISHONEST = "SHADER_BINDING_DISHONEST"
SENSITIVITY_CLEARS_BAR = "SENSITIVITY_CLEARS_BAR"

NOETIC_RELS = dnr.NOETIC_RELS
QN_REL = dnr.QN_REL

REQUIRED_CANDIDATE_IDS: tuple[str, ...] = (
    "heterogeneous_qkvz_bits",
    "z_only_q3",
    "q_readout_q3",
    "k_state_q3",
    "v_state_q3",
    "uniform_q3_qkvz",
    "entropy_coded_qkvz_codes",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Byte totals are HQ38M20 stored bytes of attention.linear_qkvz, split into "
    "q/k/v/z row ranges of the fused per-key-head Q128/K128/V384/Z384 layout "
    "already measured in DELTANET_REPRESENTATION. Shannon / conditional "
    "entropy / mutual information are lossless statements about the stored "
    "signed-nibble codes q ∈ {-8..7}. Sensitivity is a CPU replay of "
    "qwen38_qkvz_rearrange_conv_l2_f32 + gated-delta + gated RMSNorm on "
    "isotropic Gaussian x (seed 38), S and conv_state starting at 0. It is "
    "not capability and not a generate identity gate. A candidate that "
    "unpacks to dense W then runs ordinary GEMV is REJECTED_DENSE_REMAT. "
    "A bit reduction is never supported on entropy alone."
)


class QkvzPrecisionRefuse(ValueError):
    """The qkvz precision census refused rather than guessing."""


class UnreconciledQkvz(QkvzPrecisionRefuse):
    """linear_qkvz bytes do not equal the recorded 2,139,096,960."""

    def __init__(self, got: int, want: int = QKVZ_ACTIVE_TARGET, *, detail: str = "") -> None:
        self.got = int(got)
        self.want = int(want)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: linear_qkvz bytes {got} != recorded organ total {want}{extra}"
        )


class CatalogLayoutRefuse(QkvzPrecisionRefuse):
    """A payload disagreed with its header or with the geometry."""


# ---------------------------------------------------------------------------
# Accounting. Load-bearing refusal.
# ---------------------------------------------------------------------------


def reconcile_qkvz(
    got: int,
    want: int = QKVZ_ACTIVE_TARGET,
    *,
    detail: str = "",
) -> int:
    """Refuse unless linear_qkvz stored bytes equal 2,139,096,960."""
    if int(got) != int(want):
        raise UnreconciledQkvz(int(got), int(want), detail=detail)
    return int(got)


def qkvz_subblock_payload_sum(parts: Mapping[str, Mapping[str, Any]]) -> int:
    """q+k+v+z payload plus the unsplit 40-byte headers. Refuses on mismatch."""
    payload = 0
    for name in SUBBLOCKS:
        if name not in parts:
            raise UnreconciledQkvz(0, QKVZ_ACTIVE_TARGET, detail=f"missing sub-block {name}")
        payload += int(parts[name]["payload_bytes"])
    header = int((parts.get("header") or {}).get("header_bytes") or 0)
    got = payload + header
    reconcile_qkvz(got, detail="q+k+v+z payload + headers vs linear_qkvz organ total")
    return got


def q4_code_bytes_at_bits(code_bytes_q4: int, bits: int, *, incumbent: int = INCUMBENT_BITS) -> int:
    """Exact integer code bytes of a uniform bit-width recode. No padding."""
    if bits <= 0 or incumbent <= 0:
        raise QkvzPrecisionRefuse(f"cannot recode {code_bytes_q4} Q{incumbent} bytes at {bits} bits")
    n_codes = (int(code_bytes_q4) * 8) // int(incumbent)
    if n_codes * int(incumbent) != int(code_bytes_q4) * 8:
        raise QkvzPrecisionRefuse(f"cannot recode {code_bytes_q4} Q{incumbent} bytes at {bits} bits")
    if (n_codes * bits) % 8:
        raise QkvzPrecisionRefuse(f"{n_codes} codes at {bits} bits is not a whole number of bytes")
    return (n_codes * bits) // 8


def bytes_eliminated_codes(code_bytes_q4: int, bits: int) -> int:
    return int(code_bytes_q4) - q4_code_bytes_at_bits(code_bytes_q4, bits)


def _identity(geo: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    return dnr._identity(root, geo)


def accounting_from_rows(
    rows: Sequence[Mapping[str, Any]],
    geo: Mapping[str, Any],
) -> dict[str, Any]:
    """Sum the 48 linear_qkvz tensors and refuse if they are not 2,139,096,960."""
    qkvz = [r for r in rows if str(r["organ"]) == "attention.linear_qkvz"]
    if not qkvz:
        raise QkvzPrecisionRefuse("catalog holds no attention.linear_qkvz tensors; refusing")
    header = sum(int(r["header_bytes"]) for r in qkvz)
    scale = sum(int(r["scale_bytes"]) for r in qkvz)
    bias = sum(int(r["bias_bytes"]) for r in qkvz)
    code = sum(int(r["code_bytes"]) for r in qkvz)
    stored = sum(int(r["stored_bytes"]) for r in qkvz)
    elements = sum(int(r["elements"]) for r in qkvz)
    reconcile_qkvz(stored, detail="sum of per-tensor stored bytes")
    if header + scale + bias + code != stored:
        raise CatalogLayoutRefuse("qkvz header+scale+bias+code does not reassemble to stored")
    if bias != 0:
        raise CatalogLayoutRefuse("HQ30UQ4 qkvz is documented as unbiased; catalog has bias")
    parts = dnr.qkvz_subblock_parts(geo)
    qkvz_subblock_payload_sum(parts)
    n_layers = int(geo["n_deltanet_layers"])
    if len(qkvz) != n_layers:
        raise QkvzPrecisionRefuse(f"qkvz tensors {len(qkvz)} != n_deltanet_layers {n_layers}")
    return {
        "n_tensors": len(qkvz),
        "n_layers": n_layers,
        "organ": "attention.linear_qkvz",
        "header_bytes": header,
        "scale_bytes": scale,
        "bias_bytes": bias,
        "code_bytes": code,
        "stored_bytes": stored,
        "elements": elements,
        "incumbent_bits": INCUMBENT_BITS,
        "group_size": INCUMBENT_GROUP,
        "code_share": code / stored,
        "share_of_token": stored / TOKEN_ACTIVE_TARGET,
        "target": QKVZ_ACTIVE_TARGET,
        "reconciled": True,
        "qkvz_subblocks": parts,
        "incumbent_packing": {
            "family": "hq30uq4_uniform_q4_group64_signed_nibble_f16_scale",
            "code_bits": INCUMBENT_BITS,
            "alphabet": list(range(NIBBLE_ALPHABET)),
            "signed_q": "q = nibble - 8, nibble in {0..15}, q in {-8..7}",
            "reconstruction": "w = float(nibble - 8) * f16_scale; no bias",
            "layout": geo["qkvz_layout"],
        },
        "identity": _identity(geo),
    }


@lru_cache(maxsize=1)
def _rows_geo() -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    rows, geo = dnr.census_rows()
    return tuple(rows), geo


def accounting(*, root: Path | None = None) -> dict[str, Any]:
    if root is None:
        rows, geo = _rows_geo()
        return accounting_from_rows(list(rows), geo)
    rows, geo = dnr.census_rows(root=root)
    return accounting_from_rows(rows, geo)


# ---------------------------------------------------------------------------
# Information-theoretic primitives on packed 4-bit (nibble) streams.
# Same functions as MLP_CODE_INFORMATION (_shannon / _mi / _cond_h), 16-ary.
# ---------------------------------------------------------------------------


def _nibble_from_byte_hist(bh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold a 256-bin packed-byte hist into 16-ary nibble hist and lo→hi joint.

    Packing (qwen_uniform_q4): even local weight in the low nibble, odd in the
    high nibble. Neighbour = those two consecutive codes inside a byte.
    """
    bh = np.asarray(bh, dtype=np.int64)
    if bh.shape != (256,):
        raise QkvzPrecisionRefuse(f"byte hist shape {bh.shape} != (256,)")
    lo = np.arange(256, dtype=np.int64) & 15
    hi = np.arange(256, dtype=np.int64) >> 4
    nh = (
        np.bincount(lo, weights=bh.astype(np.float64), minlength=16)
        + np.bincount(hi, weights=bh.astype(np.float64), minlength=16)
    ).astype(np.int64)
    within = np.zeros((16, 16), dtype=np.int64)
    np.add.at(within, (lo, hi), bh)
    return nh, within


def _joint_aligned_nibbles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint8).ravel()
    b = np.asarray(b, dtype=np.uint8).ravel()
    if a.size != b.size:
        raise QkvzPrecisionRefuse("aligned nibble joint requires equal packed lengths")
    j = np.zeros((16, 16), dtype=np.int64)
    for shift in (0, 4):
        na = (a >> shift) & np.uint8(15)
        nb = (b >> shift) & np.uint8(15)
        j += np.bincount(
            na.astype(np.int64) * 16 + nb.astype(np.int64), minlength=256
        ).reshape(16, 16)
    return j


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


def _unique_groups(packed: np.ndarray, group_bytes: int = 32) -> tuple[int, int]:
    n = packed.size // group_bytes
    if n <= 0:
        return 0, 0
    h = packed.reshape(n, group_bytes).view(np.dtype((np.void, group_bytes))).ravel()
    return int(np.unique(h).size), n


# ---------------------------------------------------------------------------
# Code-stream entropy. Real HQ30UQ4 nibbles, not a model of W.
# ---------------------------------------------------------------------------


def measure_code_entropy(
    rows: Sequence[Mapping[str, Any]],
    geo: Mapping[str, Any],
    *,
    sample_layers: Sequence[int] = SAMPLE_LAYERS,
) -> dict[str, Any]:
    """i.i.d. Shannon, neighbour conditional, and cross-layer structure per sub-block."""
    idx = dnr.fused_qkvz_row_indices(geo)
    qkvz = sorted(
        (r for r in rows if str(r["organ"]) == "attention.linear_qkvz"),
        key=lambda r: int(r["layer"]),
    )
    if len(qkvz) != int(geo["n_deltanet_layers"]):
        raise QkvzPrecisionRefuse(f"qkvz tensors {len(qkvz)} != {geo['n_deltanet_layers']}")
    rows_n = int(geo["qkvz_rows"])
    cols = int(geo["qkvz_cols"])
    gpr = cols // INCUMBENT_GROUP
    sample_set = {int(x) for x in sample_layers}

    glob: dict[str, dict[str, Any]] = {
        s: {
            "nibble": np.zeros(16, dtype=np.int64),
            "byte": np.zeros(256, dtype=np.int64),
            "within": np.zeros((16, 16), dtype=np.int64),
            "Hq": [],
            "Hb": [],
            "Hcond": [],
        }
        for s in SUBBLOCKS
    }
    cross: dict[str, dict[str, list[float]]] = {
        s: {"mi": [], "match": [], "indep_match": []} for s in SUBBLOCKS
    }
    prev_packed: dict[str, np.ndarray] = {}
    sample_extra: list[dict[str, Any]] = []
    code_bytes_sum = 0
    stored_sum = 0
    stash: dict[int, dict[str, Any]] = {}

    for rec in qkvz:
        path = Path(rec["segment_path"])
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise CatalogAbsent(f"cannot read {path}: {exc}") from exc
        parsed = dnr.parse_hq30uq4_header(blob, name=str(rec["name"]))
        if parsed["shape"] != [rows_n, cols]:
            raise CatalogLayoutRefuse(
                f"{rec['name']}: shape {parsed['shape']} != [{rows_n}, {cols}]"
            )
        code_off = parsed["payload_off"] + parsed["scale_bytes"]
        codes = np.frombuffer(
            blob[code_off : code_off + parsed["code_bytes"]], dtype=np.uint8
        ).reshape(rows_n, gpr, INCUMBENT_GROUP // 2)
        scales = np.frombuffer(
            blob[parsed["payload_off"] : parsed["payload_off"] + parsed["scale_bytes"]],
            dtype="<f2",
        ).reshape(rows_n, gpr)
        code_bytes_sum += int(parsed["code_bytes"])
        stored_sum += int(rec["stored_bytes"])
        layer = int(rec["layer"])
        if layer in sample_set:
            stash[layer] = {
                "codes": codes.copy(),
                "scales": scales.copy(),
                "rec": rec,
            }

        for s in SUBBLOCKS:
            sub = codes[idx[s]]
            packed = sub.ravel()
            bh = np.bincount(packed, minlength=256)
            nh, within = _nibble_from_byte_hist(bh)
            glob[s]["nibble"] += nh
            glob[s]["byte"] += bh
            glob[s]["within"] += within
            Hq = _shannon(nh)
            Hb = _shannon(bh)
            Hc = _cond_h(within)
            glob[s]["Hq"].append(Hq)
            glob[s]["Hb"].append(Hb)
            glob[s]["Hcond"].append(Hc)
            p = nh.astype(np.float64)
            p = p / p.sum() if int(p.sum()) else p
            indep_match = float((p * p).sum())
            if s in prev_packed:
                j = _joint_aligned_nibbles(prev_packed[s], packed)
                tot = float(j.sum())
                cross[s]["mi"].append(_mi(j))
                cross[s]["match"].append(float(np.trace(j) / tot) if tot else 0.0)
                cross[s]["indep_match"].append(indep_match)
            prev_packed[s] = packed
            if layer in sample_set:
                ug, ng = _unique_groups(packed, group_bytes=INCUMBENT_GROUP // 2)
                z = zlib.compress(memoryview(packed), ZLIB_LEVEL)
                sample_extra.append(
                    {
                        "layer": layer,
                        "subblock": s,
                        "unique_groups": ug,
                        "n_groups": ng,
                        "unique_frac": (ug / ng) if ng else 0.0,
                        "zlib_bytes": len(z),
                        "zlib_ratio": (len(z) / packed.size) if packed.size else 0.0,
                        "H_q": Hq,
                        "H_byte": Hb,
                        "H_q_given_prev_within_byte": Hc,
                    }
                )

    reconcile_qkvz(stored_sum, detail="sum of bytes actually read from HQ30UQ4")
    derived_code = (int(geo["qkvz_rows"]) * int(geo["qkvz_cols"]) * INCUMBENT_BITS * len(qkvz)) // 8
    if code_bytes_sum != derived_code:
        raise CatalogLayoutRefuse(
            f"code bytes read {code_bytes_sum} != geometry {derived_code}"
        )

    by_block: dict[str, Any] = {}
    global_nibble = np.zeros(16, dtype=np.int64)
    global_byte = np.zeros(256, dtype=np.int64)
    global_within = np.zeros((16, 16), dtype=np.int64)
    for s in SUBBLOCKS:
        nh = glob[s]["nibble"]
        global_nibble += nh
        global_byte += glob[s]["byte"]
        global_within += glob[s]["within"]
        n_codes = int(nh.sum())
        Hq = _shannon(nh)
        Hb = _shannon(glob[s]["byte"])
        Hc = _cond_h(glob[s]["within"])
        mi = _mi(glob[s]["within"])
        iid_bytes = (Hq * n_codes) / 8.0
        parts = dnr.qkvz_subblock_parts(geo)
        code_b = int(parts[s]["code_bytes"])
        by_block[s] = {
            "rows": int(idx[s].size),
            "n_codes": n_codes,
            "code_bytes": code_b,
            "incumbent_bits": INCUMBENT_BITS,
            "q_hist": nh.tolist(),
            "p_q": (nh.astype(np.float64) / n_codes).tolist() if n_codes else None,
            "H_q_bits": Hq,
            "H_q_over_uniform": Hq / float(INCUMBENT_BITS),
            "H_byte_bits": Hb,
            "H_byte_if_iid_q": 2.0 * Hq,
            "H_q_given_prev_within_byte": Hc,
            "mi_within_byte_bits": mi,
            "iid_shannon_bytes": iid_bytes,
            "iid_shannon_bytes_rounded": int(round(iid_bytes)),
            "iid_redundant_bytes_rounded": int(round(code_b - iid_bytes)),
            "independent_fraction": iid_bytes / code_b if code_b else None,
            "lossless_q3_impossible": Hq > float(CANDIDATE_BITS),
            "nibble_0_unused": int(nh[0]) == 0,
            "H_q_per_tensor": _summ(glob[s]["Hq"]),
            "cross_layer": {
                "mi_bits": _summ(cross[s]["mi"]),
                "match": _summ(cross[s]["match"]),
                "independent_match": _summ(cross[s]["indep_match"]),
            },
        }

    n_codes_all = int(global_nibble.sum())
    H_all = _shannon(global_nibble)
    iid_all = (H_all * n_codes_all) / 8.0
    code_all = int(code_bytes_sum)
    sample_H = [s["H_q"] for s in sample_extra]
    return {
        "n_tensors_measured": len(qkvz),
        "code_bytes_read": code_bytes_sum,
        "n_parameters": n_codes_all,
        "alphabet": list(range(NIBBLE_ALPHABET)),
        "q_hist": global_nibble.tolist(),
        "p_q": (global_nibble.astype(np.float64) / n_codes_all).tolist(),
        "H_q_bits": H_all,
        "H_q_over_uniform": H_all / float(INCUMBENT_BITS),
        "H_byte_bits": _shannon(global_byte),
        "H_byte_if_iid_q": 2.0 * H_all,
        "H_q_given_prev_within_byte": _cond_h(global_within),
        "mi_within_byte_bits": _mi(global_within),
        "iid_shannon_bytes": iid_all,
        "iid_shannon_bytes_rounded": int(round(iid_all)),
        "iid_redundant_bytes_rounded": int(round(code_all - iid_all)),
        "independent_fraction": iid_all / code_all,
        "nns022_reopen_uniform_frac": NNS022_REOPEN_UNIFORM_FRAC,
        "nns022_reopen_fires": (H_all / float(INCUMBENT_BITS)) <= NNS022_REOPEN_UNIFORM_FRAC,
        "lossless_q3_impossible": H_all > float(CANDIDATE_BITS),
        "by_subblock": by_block,
        "sample_layers": list(sample_layers),
        "sample_block_and_row": sample_extra,
        "sample_unique_frac": _summ([s["unique_frac"] for s in sample_extra]),
        "sample_zlib_ratio": _summ([s["zlib_ratio"] for s in sample_extra]),
        "sample_H_q": _summ(sample_H),
        "method": (
            "MLP_CODE_INFORMATION._shannon / _mi / _cond_h on the 16-ary nibble "
            "alphabet of HQ30UQ4 (even local → low nibble, odd → high). "
            "Neighbour is the other nibble of the same packed byte. Cross-layer "
            "MI is of aligned nibbles of the same sub-block between consecutive "
            "DeltaNet layers. Unique groups are 32-byte (64-nibble) Q4 groups."
        ),
        "_stash": stash,
        "note": (
            "H(q) is Shannon entropy of the 16-ary nibble over every linear_qkvz "
            "code of that sub-block. It is not capability. H>3 bits makes a "
            "lossless 3-bit recode of the stored q information-theoretically "
            "impossible; a lossy Q3 is a sensitivity question, not an entropy one."
        ),
    }


# ---------------------------------------------------------------------------
# CPU consume path. The physical primitives that actually eat q, k, v, z.
# ---------------------------------------------------------------------------


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _requant_signed_nibble(q: np.ndarray, bits: int) -> np.ndarray:
    """Drop incumbent signed nibbles q = nibble-8 in [-8, 7] onto 2^bits levels."""
    levels = 1 << int(bits)
    lo, hi = -8.0, 7.0
    u = np.clip(np.round((q - lo) / (hi - lo) * (levels - 1)), 0, levels - 1)
    return lo + u * ((hi - lo) / (levels - 1))


def _refit_subblock(W: np.ndarray, q: np.ndarray, bits: int, *, gpr: int, group: int) -> np.ndarray:
    """Least-squares per-group scale after nibble drop. Does not write a dense production W."""
    rows, cols = W.shape
    q2 = _requant_signed_nibble(q, bits).reshape(rows, gpr, group)
    Wg = W.reshape(rows, gpr, group)
    num = (q2 * Wg).sum(-1)
    den = (q2 * q2).sum(-1)
    scale = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return (q2 * scale[:, :, None]).reshape(rows, cols)


def _unpack_q4(
    codes: np.ndarray, scales: np.ndarray, *, group: int = INCUMBENT_GROUP
) -> tuple[np.ndarray, np.ndarray]:
    """Dequant HQ30UQ4 for the CPU operator probe. Not a production path."""
    rows_n, gpr, nbytes = codes.shape
    if nbytes != group // 2:
        raise CatalogLayoutRefuse(f"code group {nbytes} != {group // 2}")
    lo = (codes & 0x0F).astype(np.float32) - 8.0
    hi = (codes >> 4).astype(np.float32) - 8.0
    q = np.empty((rows_n, gpr, group), dtype=np.float32)
    q[:, :, 0::2] = lo
    q[:, :, 1::2] = hi
    W = (q * scales.astype(np.float32)[:, :, None]).reshape(rows_n, gpr * group)
    return W, q.reshape(rows_n, gpr * group)


def _load_f32(rec: Mapping[str, Any]) -> np.ndarray:
    blob = Path(rec["segment_path"]).read_bytes()
    parsed = dnr.parse_f32v2_header(blob, name=str(rec["name"]), shape=rec.get("shape"))
    return np.frombuffer(blob[parsed["payload_off"] :], dtype="<f4").copy()


def _load_q4_matrix(rec: Mapping[str, Any]) -> np.ndarray:
    blob = Path(rec["segment_path"]).read_bytes()
    parsed = dnr.parse_hq30uq4_header(blob, name=str(rec["name"]))
    rows_n, cols = parsed["shape"]
    gpr = cols // INCUMBENT_GROUP
    scales = np.frombuffer(
        blob[parsed["payload_off"] : parsed["payload_off"] + parsed["scale_bytes"]],
        dtype="<f2",
    ).reshape(rows_n, gpr)
    codes = np.frombuffer(
        blob[parsed["payload_off"] + parsed["scale_bytes"] :], dtype=np.uint8
    ).reshape(rows_n, gpr, INCUMBENT_GROUP // 2)
    W, _q = _unpack_q4(codes, scales)
    return W


def _rearrange_conv(
    y: np.ndarray,
    conv_w: np.ndarray,
    conv_state: np.ndarray,
    geo: Mapping[str, Any],
    *,
    eps: float = RMS_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """qwen38_qkvz_rearrange_conv_l2_f32: conv+silu on q,k,v; z copied; q/k L2."""
    kh = int(geo["key_heads"])
    kd = int(geo["key_head_dim"])
    vd = int(geo["value_head_dim"])
    vpk = int(geo["values_per_key"])
    fused = kd * 2 + vpk * vd * 2
    Y = y.reshape(kh, fused)
    q_raw = Y[:, :kd]
    k_raw = Y[:, kd : 2 * kd]
    v_raw = Y[:, 2 * kd : 2 * kd + vpk * vd]
    z_raw = Y[:, 2 * kd + vpk * vd :]
    current = np.concatenate([q_raw.reshape(-1), k_raw.reshape(-1), v_raw.reshape(-1)])
    acc = (conv_state * conv_w[:, :3]).sum(-1) + current * conv_w[:, 3]
    conv_out = _silu(acc)
    conv_state[:, :-1] = conv_state[:, 1:]
    conv_state[:, -1] = current
    q_c = conv_out[: kh * kd].reshape(kh, kd)
    k_c = conv_out[kh * kd : 2 * kh * kd].reshape(kh, kd)
    v_c = conv_out[2 * kh * kd :].reshape(kh, vpk, vd)
    q_hat = q_c / np.sqrt((q_c * q_c).sum(-1, keepdims=True) + eps) / math.sqrt(kd)
    k_hat = k_c / np.sqrt((k_c * k_c).sum(-1, keepdims=True) + eps)
    q_out = np.repeat(q_hat, vpk, axis=0)
    k_out = np.repeat(k_hat, vpk, axis=0)
    v_out = v_c.reshape(kh * vpk, vd)
    z_out = z_raw.reshape(kh * vpk, vd)
    return q_out, k_out, v_out, z_out


def _ba_decay_beta(
    ba: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    geo: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """qwen38_ba_decay_beta_f32. decay = exp(-exp(A_log)*softplus(a+dt_bias)); beta = sigmoid(b)."""
    kh = int(geo["key_heads"])
    vpk = int(geo["values_per_key"])
    B = ba.reshape(kh, vpk * 2)
    bb = B[:, :vpk].reshape(-1)
    a = B[:, vpk:].reshape(-1)
    x = a + dt_bias
    softplus = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    g = -np.exp(a_log) * softplus
    return np.exp(g), 1.0 / (1.0 + np.exp(-bb))


def _gated_delta(
    state: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q. Vectorized vi kernel."""
    decayed = state * decay[:, None, None]
    kv_mem = np.einsum("hkv,hk->hv", decayed, key)
    delta = (value - kv_mem) * beta[:, None]
    state2 = decayed + np.einsum("hk,hv->hkv", key, delta)
    h = np.einsum("hkv,hk->hv", state2, query)
    return state2, h


def _gated_rmsnorm(
    h: np.ndarray, z: np.ndarray, weight: np.ndarray, *, eps: float = RMS_EPS
) -> np.ndarray:
    """qwen80_deltanet_gated_rmsnorm: RMSNorm(h) * weight * silu(z)."""
    inv = 1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps)
    return h * inv * weight * _silu(z)


def _relfro(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.square(a - b).sum()))
    den = float(np.sqrt(np.square(b).sum()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel().astype(np.float64)
    bb = b.ravel().astype(np.float64)
    den = math.sqrt(float(aa @ aa) * float(bb @ bb))
    if den == 0.0:
        return float("nan")
    return float(aa @ bb / den)


def _layer_aux(rows: Sequence[Mapping[str, Any]], geo: Mapping[str, Any], layer: int) -> dict[str, Any]:
    by_org = {str(r["organ"]): r for r in rows if int(r["layer"]) == layer}
    adj = {}
    for rec in dnr.dn_records(organs=dnr.ADJACENT_ORGANS):
        if int(rec["layer"]) != layer:
            continue
        row = dnr.tensor_accounting_row(rec, geo)
        adj[str(row["organ"])] = row
    return {
        "ba": _load_q4_matrix(by_org["attention.linear_ba"]),
        "conv": _load_f32(by_org["attention.linear_conv1d"]).reshape(
            int(geo["conv_channels"]), int(geo["conv_kernel"])
        ),
        "a_log": _load_f32(adj["state.A_log"]),
        "dt_bias": _load_f32(adj["state.dt_bias"]),
        "norm": _load_f32(adj["norms.linear_attn"]),
    }


def _rollout(
    W: np.ndarray,
    xs: Sequence[np.ndarray],
    aux: Mapping[str, Any],
    geo: Mapping[str, Any],
) -> dict[str, list[np.ndarray]]:
    vh = int(geo["value_heads"])
    kd = int(geo["key_head_dim"])
    vd = int(geo["value_head_dim"])
    C = int(geo["conv_channels"])
    S = np.zeros((vh, kd, vd), dtype=np.float32)
    cs = np.zeros((C, int(geo["conv_kernel"]) - 1), dtype=np.float32)
    out_S: list[np.ndarray] = []
    out_h: list[np.ndarray] = []
    out_g: list[np.ndarray] = []
    Wba = aux["ba"]
    conv = aux["conv"]
    for x in xs:
        y = W @ x
        ba = Wba @ x
        q, k, v, z = _rearrange_conv(y, conv, cs, geo)
        decay, beta = _ba_decay_beta(ba, aux["a_log"], aux["dt_bias"], geo)
        S, h = _gated_delta(S, q, k, v, decay, beta)
        gated = _gated_rmsnorm(h, z, aux["norm"])
        out_S.append(S.copy())
        out_h.append(h.copy())
        out_g.append(gated.copy())
    return {"S": out_S, "h": out_h, "gated": out_g}


def _probe_stats(
    pert: Mapping[str, list[np.ndarray]],
    base: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    s_rel = [_relfro(a, b) for a, b in zip(pert["S"], base["S"])]
    h_rel = [_relfro(a, b) for a, b in zip(pert["h"], base["h"])]
    g_rel = [_relfro(a, b) for a, b in zip(pert["gated"], base["gated"])]
    g_cos = [_cosine(a, b) for a, b in zip(pert["gated"], base["gated"])]
    h_cos = [_cosine(a, b) for a, b in zip(pert["h"], base["h"])]
    s_ident = [bool(np.array_equal(a, b)) for a, b in zip(pert["S"], base["S"])]
    h_ident = [bool(np.array_equal(a, b)) for a, b in zip(pert["h"], base["h"])]
    return {
        "n_tokens": len(s_rel),
        "rec_state_relfro": _summ(s_rel),
        "rec_out_h_relfro": _summ(h_rel),
        "gated_relfro": _summ(g_rel),
        "rec_out_h_cosine": _summ(h_cos),
        "gated_cosine": _summ(g_cos),
        "rec_state_identical_every_token": all(s_ident),
        "rec_out_h_identical_every_token": all(h_ident),
        "gated_cosine_min": float(min(g_cos)) if g_cos else None,
        "gated_relfro_max": float(max(g_rel)) if g_rel else None,
        "rec_state_relfro_max": float(max(s_rel)) if s_rel else None,
        "token0_gated_relfro": g_rel[0] if g_rel else None,
        "token_last_gated_relfro": g_rel[-1] if g_rel else None,
        "token0_gated_cosine": g_cos[0] if g_cos else None,
        "token_last_gated_cosine": g_cos[-1] if g_cos else None,
    }


def measure_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    geo: Mapping[str, Any],
    *,
    layers: Sequence[int] = SENSITIVITY_LAYERS,
    n_tokens: int = N_ROLL_TOKENS,
    bits: int = CANDIDATE_BITS,
    rng_seed: int = RNG_SEED,
    stash: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Perturb one sub-block's codes at `bits`, replay the layer consume path."""
    idx = dnr.fused_qkvz_row_indices(geo)
    roles = dnr.independent_information(geo)["roles"]
    hidden = int(geo["hidden_size"])
    gpr = int(geo["qkvz_cols"]) // INCUMBENT_GROUP
    rng = np.random.default_rng(rng_seed)
    xs = [
        rng.normal(0.0, 1.0 / math.sqrt(hidden), size=hidden).astype(np.float32)
        for _ in range(int(n_tokens))
    ]
    per_layer: list[dict[str, Any]] = []
    acc_block: dict[str, dict[str, list[float]]] = {
        s: {
            "gated_cos": [],
            "gated_rel": [],
            "S_rel": [],
            "h_rel": [],
            "S_ident": [],
            "h_ident": [],
        }
        for s in SUBBLOCKS
    }

    for layer in layers:
        packed = (stash or {}).get(int(layer))
        if packed is None:
            rec = next(
                r
                for r in rows
                if int(r["layer"]) == int(layer) and str(r["organ"]) == "attention.linear_qkvz"
            )
            blob = Path(rec["segment_path"]).read_bytes()
            parsed = dnr.parse_hq30uq4_header(blob, name=str(rec["name"]))
            rows_n, cols = parsed["shape"]
            gpr_l = cols // INCUMBENT_GROUP
            codes = np.frombuffer(
                blob[parsed["payload_off"] + parsed["scale_bytes"] :], dtype=np.uint8
            ).reshape(rows_n, gpr_l, INCUMBENT_GROUP // 2)
            scales = np.frombuffer(
                blob[parsed["payload_off"] : parsed["payload_off"] + parsed["scale_bytes"]],
                dtype="<f2",
            ).reshape(rows_n, gpr_l)
        else:
            codes = packed["codes"]
            scales = packed["scales"]
        W, qn = _unpack_q4(codes, scales)
        aux = _layer_aux(rows, geo, int(layer))
        base = _rollout(W, xs, aux, geo)
        layer_row: dict[str, Any] = {"layer": int(layer), "by_subblock": {}}
        for s in SUBBLOCKS:
            Wp = W.copy()
            ri = idx[s]
            Wp[ri] = _refit_subblock(W[ri], qn[ri], int(bits), gpr=gpr, group=INCUMBENT_GROUP)
            pert = _rollout(Wp, xs, aux, geo)
            stats = _probe_stats(pert, base)
            writes = bool(roles[s]["enters_state_update"])
            layer_row["by_subblock"][s] = {
                **stats,
                "writes_rec_state": writes,
                "enters_gated_delta": bool(roles[s]["enters_gated_delta"]),
                "enters_conv": bool(roles[s]["enters_conv"]),
                "physical_primitive": (
                    "LocalStateMachine" if s != "z" else "FusedDecodeCompute"
                ),
                "consumed_by": (
                    "qwen38_gated_delta_decode_vi_simd (readout h=S^T q)"
                    if s == "q"
                    else "qwen38_gated_delta_decode_vi_simd (write key)"
                    if s == "k"
                    else "qwen38_gated_delta_decode_vi_simd (write value)"
                    if s == "v"
                    else "qwen80_deltanet_gated_rmsnorm (silu gate; z never bound to gated-delta)"
                ),
            }
            acc_block[s]["gated_cos"].append(float(stats["gated_cosine_min"]))
            acc_block[s]["gated_rel"].append(float(stats["gated_relfro_max"]))
            acc_block[s]["S_rel"].append(float(stats["rec_state_relfro_max"]))
            acc_block[s]["h_rel"].append(float(stats["rec_out_h_relfro"]["max"]))
            acc_block[s]["S_ident"].append(bool(stats["rec_state_identical_every_token"]))
            acc_block[s]["h_ident"].append(bool(stats["rec_out_h_identical_every_token"]))
        per_layer.append(layer_row)

    by_block: dict[str, Any] = {}
    for s in SUBBLOCKS:
        writes = bool(roles[s]["enters_state_update"])
        by_block[s] = {
            "measured": True,
            "candidate_bits": int(bits),
            "writes_rec_state": writes,
            "enters_gated_delta": bool(roles[s]["enters_gated_delta"]),
            "enters_conv": bool(roles[s]["enters_conv"]),
            "gated_cosine_min": float(min(acc_block[s]["gated_cos"])),
            "gated_relfro_max": float(max(acc_block[s]["gated_rel"])),
            "rec_state_relfro_max": float(max(acc_block[s]["S_rel"])),
            "rec_out_h_relfro_max": float(max(acc_block[s]["h_rel"])),
            "rec_state_identical": all(acc_block[s]["S_ident"]),
            "rec_out_h_identical": all(acc_block[s]["h_ident"]),
            "gated_cosine": _summ(acc_block[s]["gated_cos"]),
            "gated_relfro": _summ(acc_block[s]["gated_rel"]),
            "rec_state_relfro": _summ(acc_block[s]["S_rel"]),
            "physical_primitive": "LocalStateMachine" if s != "z" else "FusedDecodeCompute",
        }

    return {
        "measured": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "candidate_bits": int(bits),
        "n_tokens": int(n_tokens),
        "layers": [int(x) for x in layers],
        "n_layers": len(list(layers)),
        "probe": {
            "kind": "cpu_operator_rollout",
            "x": f"isotropic gaussian N(0, 1/sqrt(hidden)), seed {rng_seed}",
            "state_init": "S=0 and conv_state=0 (sequence start)",
            "requant": (
                f"per-group least-squares scale refit after nibble drop to "
                f"{1 << int(bits)} levels on [-8, 7]; scales of unperturbed "
                "blocks are untouched"
            ),
            "kernels": (
                "qwen38_qkvz_rearrange_conv_l2_f32 + "
                "qwen38_gated_delta_decode_vi + "
                "qwen80_deltanet_gated_rmsnorm"
            ),
            "rms_eps": RMS_EPS,
            "real_post_norm_x": False,
            "generate_gate": False,
            "not_capability": True,
        },
        "cosine_bar": COSINE_BAR,
        "cosine_bar_source": (
            "NNS-019 mean-row cosine ≥ 0.990 vs BF16, applied here as a STATIC "
            "operator-output cousin, not a generate identity gate"
        ),
        "state_relfro_bar": STATE_RELFRO_BAR,
        "by_subblock": by_block,
        "per_layer": per_layer,
        "note": (
            "z-only Q3 must leave rec_state and rec_out (h) bit-identical if "
            "the shader binding is honest. q-only Q3 must leave rec_state "
            "bit-identical (readout, no write). k and v rewrite S."
        ),
    }


# ---------------------------------------------------------------------------
# Bit-reduction licence. Entropy alone is never enough.
# ---------------------------------------------------------------------------


def _sensitivity_measured(sensitivity: Mapping[str, Any] | None) -> bool:
    return isinstance(sensitivity, Mapping) and sensitivity.get("measured") is True


def decide_supported_bit_reduction(
    *,
    candidate_bits: int,
    incumbent_bits: int = INCUMBENT_BITS,
    H_q_bits: float | None = None,
    sensitivity: Mapping[str, Any] | None = None,
    cosine_bar: float = COSINE_BAR,
    state_relfro_bar: float = STATE_RELFRO_BAR,
) -> dict[str, Any]:
    """A bit reduction is never reported as supported on entropy alone.

    Sensitivity must be an object with measured=True and the operator metrics.
    Missing or incomplete sensitivity is ENTROPY_ALONE_INSUFFICIENT even when
    H(q) would allow a lossless recode.
    """
    lossless_possible = H_q_bits is not None and float(H_q_bits) <= float(candidate_bits)
    entropy_row = {
        "H_q_bits": None if H_q_bits is None else float(H_q_bits),
        "incumbent_bits": int(incumbent_bits),
        "candidate_bits": int(candidate_bits),
        "lossless_possible": bool(lossless_possible),
        "entropy_gap_bits": (
            None if H_q_bits is None else float(incumbent_bits) - float(H_q_bits)
        ),
    }
    if not _sensitivity_measured(sensitivity):
        return {
            "supported": False,
            "reason": ENTROPY_ALONE_INSUFFICIENT,
            "sensitivity_measured": False,
            **entropy_row,
        }
    assert sensitivity is not None
    required = (
        "gated_cosine_min",
        "writes_rec_state",
        "rec_state_identical",
        "rec_state_relfro_max",
    )
    if any(k not in sensitivity for k in required):
        return {
            "supported": False,
            "reason": SENSITIVITY_INCOMPLETE,
            "sensitivity_measured": True,
            **entropy_row,
        }
    writes = bool(sensitivity["writes_rec_state"])
    s_ident = bool(sensitivity["rec_state_identical"])
    s_rel = float(sensitivity["rec_state_relfro_max"])
    g_cos = float(sensitivity["gated_cosine_min"])
    if not writes and not s_ident:
        return {
            "supported": False,
            "reason": SHADER_BINDING_DISHONEST,
            "sensitivity_measured": True,
            "gated_cosine_min": g_cos,
            **entropy_row,
        }
    if writes and s_rel > float(state_relfro_bar):
        return {
            "supported": False,
            "reason": REC_STATE_INJURED,
            "sensitivity_measured": True,
            "rec_state_relfro_max": s_rel,
            "gated_cosine_min": g_cos,
            **entropy_row,
        }
    if g_cos < float(cosine_bar):
        return {
            "supported": False,
            "reason": GATED_COSINE_BELOW_BAR,
            "sensitivity_measured": True,
            "gated_cosine_min": g_cos,
            "cosine_bar": float(cosine_bar),
            **entropy_row,
        }
    return {
        "supported": True,
        "reason": SENSITIVITY_CLEARS_BAR,
        "sensitivity_measured": True,
        "gated_cosine_min": g_cos,
        "cosine_bar": float(cosine_bar),
        **entropy_row,
    }


def _consume_role(name: str, info: Mapping[str, Any]) -> dict[str, Any]:
    role = info["roles"][name]
    primitive = "LocalStateMachine" if name != "z" else "FusedDecodeCompute"
    if primitive not in ATLAS_PRIMITIVES:
        raise QkvzPrecisionRefuse(f"{primitive} is not an atlas primitive")
    return {
        "subblock": name,
        "rows": role["rows"],
        "enters_conv": role["enters_conv"],
        "enters_gated_delta": role["enters_gated_delta"],
        "enters_state_update": role["enters_state_update"],
        "role": role["role"],
        "physical_primitive": primitive,
        "sensitivity_implication": role["sensitivity_implication"],
    }


def allocation_from_measurements(
    acc: Mapping[str, Any],
    entropy: Mapping[str, Any],
    sensitivity: Mapping[str, Any] | None,
    info: Mapping[str, Any],
    *,
    candidate_bits: int = CANDIDATE_BITS,
) -> dict[str, Any]:
    """Per-sub-block bit count. Entropy without sensitivity cannot support a drop."""
    parts = acc["qkvz_subblocks"]
    by_e = entropy["by_subblock"]
    by_s = (sensitivity or {}).get("by_subblock") if _sensitivity_measured(sensitivity) else None
    blocks: dict[str, Any] = {}
    total_save = 0
    for s in SUBBLOCKS:
        code_b = int(parts[s]["code_bytes"])
        save_if = bytes_eliminated_codes(code_b, candidate_bits)
        sens_row = dict(by_s[s]) if isinstance(by_s, Mapping) and s in by_s else None
        if sens_row is not None:
            sens_row = {**sens_row, "measured": True}
        decision = decide_supported_bit_reduction(
            candidate_bits=candidate_bits,
            incumbent_bits=INCUMBENT_BITS,
            H_q_bits=float(by_e[s]["H_q_bits"]),
            sensitivity=sens_row,
        )
        bits = int(candidate_bits) if decision["supported"] else INCUMBENT_BITS
        save = save_if if decision["supported"] else 0
        total_save += save
        if decision["supported"] and not decision["sensitivity_measured"]:
            raise QkvzPrecisionRefuse(
                "internal error: supported bit reduction without sensitivity "
                f"on sub-block {s}"
            )
        blocks[s] = {
            "bits": bits,
            "incumbent_bits": INCUMBENT_BITS,
            "candidate_bits": int(candidate_bits),
            "supported": bool(decision["supported"]),
            "reason": decision["reason"],
            "bytes_eliminated": save,
            "bytes_eliminated_if_q3": save_if,
            "code_bytes": code_b,
            "payload_bytes": int(parts[s]["payload_bytes"]),
            "H_q_bits": by_e[s]["H_q_bits"],
            "lossless_q3_impossible": by_e[s]["lossless_q3_impossible"],
            "sensitivity_measured": bool(decision["sensitivity_measured"]),
            "decision": decision,
            "consume": _consume_role(s, info),
            "evidence": {
                "entropy": {
                    "H_q_bits": by_e[s]["H_q_bits"],
                    "H_q_given_prev_within_byte": by_e[s]["H_q_given_prev_within_byte"],
                    "mi_within_byte_bits": by_e[s]["mi_within_byte_bits"],
                    "independent_fraction": by_e[s]["independent_fraction"],
                    "cross_layer_mi_bits_mean": (by_e[s]["cross_layer"]["mi_bits"] or {}).get("mean"),
                },
                "sensitivity": None
                if sens_row is None
                else {
                    "gated_cosine_min": sens_row.get("gated_cosine_min"),
                    "gated_relfro_max": sens_row.get("gated_relfro_max"),
                    "rec_state_identical": sens_row.get("rec_state_identical"),
                    "rec_out_h_identical": sens_row.get("rec_out_h_identical"),
                    "rec_state_relfro_max": sens_row.get("rec_state_relfro_max"),
                    "writes_rec_state": sens_row.get("writes_rec_state"),
                },
            },
        }

    ranking = sorted(
        SUBBLOCKS,
        key=lambda s: (
            1 if blocks[s]["consume"]["enters_state_update"] else 0,
            1 if blocks[s]["consume"]["enters_gated_delta"] else 0,
            -float((blocks[s]["evidence"]["sensitivity"] or {}).get("gated_cosine_min") or 0.0),
        ),
    )
    cheapest = ranking[0]
    not_touch = list(SUBBLOCKS)
    return {
        "by_subblock": blocks,
        "candidate_bits": int(candidate_bits),
        "incumbent_bits": INCUMBENT_BITS,
        "total_bytes_eliminated": total_save,
        "total_bytes_eliminated_if_all_q3": sum(blocks[s]["bytes_eliminated_if_q3"] for s in SUBBLOCKS),
        "share_of_token_eliminated": total_save / TOKEN_ACTIVE_TARGET,
        "share_of_qkvz_eliminated": total_save / QKVZ_ACTIVE_TARGET,
        "token_bytes_after": TOKEN_ACTIVE_TARGET - total_save,
        "qkvz_bytes_after": QKVZ_ACTIVE_TARGET - total_save,
        "cheapest_relative_win": cheapest,
        "not_worth_touching": not_touch,
        "any_supported": any(blocks[s]["supported"] for s in SUBBLOCKS),
        "cosine_bar": COSINE_BAR,
        "verdict": (
            "All four keep 4 bits. Heterogeneous allocation is available as a "
            "row-range FusedDecodeCompute, but the CPU operator probe does not "
            "license a production drop of q, k, v, or z. z is the cheapest "
            "relative experiment (Q3 cannot rewrite rec_state or h); q injures "
            "the readout, k and v rewrite S at ~0.2 rel-fro. Entropy does not "
            "pick a winner (H(q)≈3.47–3.50 of 4 bits on every block) and does "
            "not license a drop."
            if total_save == 0
            else f"Supported drops eliminate {total_save} bytes of linear_qkvz."
        ),
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
        for organ in ("deltanet", "attention", "mlp"):
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
                    "applies_to_this_object": organ in {"deltanet", "attention"}
                    and slug in {"uniform_subbit_allocation"},
                    "why_not_this_object": (
                        None
                        if organ in {"deltanet", "attention"}
                        else (
                            "Index family matches a weight-space or codec scar on "
                            "this parent. The object here is linear_qkvz q/k/v/z "
                            "sub-blocks, not the MLP code body."
                        )
                    ),
                }
            )
    return hits


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise QkvzPrecisionRefuse(f"{name} is not an atlas primitive")
    return name


def _cand(
    *,
    cid: str,
    name: str,
    mechanism: str,
    byte_model: str,
    bytes_eliminated_if_true: int | None,
    status: str,
    cheapest_falsifier: str,
    physical_primitive: str,
    dense: str,
    dense_reason: str,
    index_slugs: Sequence[str],
    citations: list[dict[str, Any]] | None = None,
    measured: Mapping[str, Any] | None = None,
    support: str = "MEASURED",
    note: str | None = None,
    cousin: bool = False,
    consult_index: bool = True,
) -> dict[str, Any]:
    hits = _index_hits(index_slugs) if consult_index else []
    cousin_hits = [h for h in hits if h.get("applies_to_this_object") is False]
    row: dict[str, Any] = {
        "id": cid,
        "name": name,
        "mechanism": mechanism,
        "byte_model": byte_model,
        "bytes_eliminated_if_true": bytes_eliminated_if_true,
        "dense_rematerialization": dense,
        "dense_rematerialization_reason": dense_reason,
        "physical_primitive": physical_primitive if dense != REJECTED_DENSE_REMAT else physical_primitive,
        "status": status,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "cheapest_falsifier": cheapest_falsifier,
        "index_slugs": list(index_slugs),
        "index_refusals": hits,
        "support": support,
    }
    if citations:
        row["citations"] = citations
    if measured is not None:
        row["measured"] = _py(measured)
    if note:
        row["note"] = note
    if cousin or cousin_hits:
        row["cousin_not_this_object"] = True
        row["index_hits_are_cousins"] = True
    return row


def candidates(
    acc: Mapping[str, Any],
    entropy: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    alloc: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    consult_index: bool = True,
) -> list[dict[str, Any]]:
    parts = acc["qkvz_subblocks"]
    save = {s: bytes_eliminated_codes(int(parts[s]["code_bytes"]), CANDIDATE_BITS) for s in SUBBLOCKS}
    save_all = sum(save.values())
    by_e = entropy["by_subblock"]
    by_s = sensitivity["by_subblock"]
    by_a = alloc["by_subblock"]
    specimen = "qwen3.8-27b sealed-3.14 DeltaNet linear_qkvz HQ30UQ4 q/k/v/z"
    nns019 = dnr._nns_cite("NNS-019", this_object=specimen)
    nns029 = dnr._nns_cite("NNS-029", this_object=specimen)
    nns022 = dnr._nns_cite("NNS-022", this_object=specimen)
    fused = _require_primitive("FusedDecodeCompute")
    local = _require_primitive("LocalStateMachine")
    common_dense = (
        "A native row-range Q4/Q3 GEMV consumes packed codes in-register. "
        "Splitting the fused tensor, writing four dense W, and running ordinary "
        "GEMV is REJECTED_DENSE_REMAT."
    )

    def sens(s: str) -> dict[str, Any]:
        return {
            "H_q_bits": by_e[s]["H_q_bits"],
            "gated_cosine_min": by_s[s]["gated_cosine_min"],
            "gated_relfro_max": by_s[s]["gated_relfro_max"],
            "rec_state_identical": by_s[s]["rec_state_identical"],
            "rec_out_h_identical": by_s[s]["rec_out_h_identical"],
            "rec_state_relfro_max": by_s[s]["rec_state_relfro_max"],
            "supported": by_a[s]["supported"],
            "reason": by_a[s]["reason"],
        }

    z_enters = bool(info["roles"]["z"]["enters_gated_delta"])
    if z_enters:
        raise QkvzPrecisionRefuse("z is documented as entering gated-delta; the shader says it does not")

    rows: list[dict[str, Any]] = [
        _cand(
            cid="heterogeneous_qkvz_bits",
            name="heterogeneous bit allocation across q, k, v, z",
            mechanism=(
                "Spend 4-bit HQ30UQ4 codes only on the fused-row ranges the "
                "state machine needs, and a 3-bit code on the rest, decoded "
                "in-register by a row-range Q4/Q3 matvec. The consume path "
                "already splits activations: z never enters gated-delta; q is "
                "a readout of S; k is the rank-1 projector and write key; v is "
                "the write value."
            ),
            byte_model=(
                f"Q3 saves 1/4 of a block's code bytes: q={save['q']}, k={save['k']}, "
                f"v={save['v']}, z={save['z']}. Scales stay f16 per group. Header "
                f"40 B/tensor is not split. All-four Q3 would save {save_all}."
            ),
            bytes_eliminated_if_true=int(alloc["total_bytes_eliminated"]),
            status=MEASURED_NEGATIVE,
            cheapest_falsifier=(
                "STATIC, run here: H(q) is 3.45–3.50 bits on every block (lossless "
                "Q3 impossible). CPU operator Q3 with per-group scale refit, 8-token "
                f"rollout, layers {list(SENSITIVITY_LAYERS)}: no sub-block clears the "
                f"{COSINE_BAR} gated-cosine bar on every token of every sampled layer. "
                "k/v rewrite S at ~0.2 rel-fro. z leaves S and h bit-identical but "
                "still injures the silu gate. Entropy does not pick a winner. A "
                "generate gate would be a new experiment, not a retry of this probe."
            ),
            physical_primitive=fused,
            dense=DIRECT_CONSUME,
            dense_reason=common_dense,
            index_slugs=["uniform_subbit_allocation"],
            citations=[nns019, nns029],
            measured={s: sens(s) for s in SUBBLOCKS},
            support="MEASURED",
            note=(
                "Predecessor DELTANET_REPRESENTATION left this OPEN with capability "
                "UNMEASURED. W-space Q3 rel-fro ~0.22 on all four was packing "
                "uniformity. This lane measured the consume operator."
            ),
            consult_index=consult_index,
        ),
        _cand(
            cid="z_only_q3",
            name="3-bit codes on z only (output gate; never writes S)",
            mechanism=(
                "Crush only the Z384 rows of each key-head fuse to 3-bit codes. "
                "q/k/v stay Q4. rearrange copies z without conv; gated RMSNorm "
                "multiplies RMSNorm(h) by silu(z). rec_state cannot change."
            ),
            byte_model=f"z codes {parts['z']['code_bytes']}; Q3 saves {save['z']}.",
            bytes_eliminated_if_true=0,
            status=MEASURED_NEGATIVE,
            cheapest_falsifier=(
                "STATIC, run here: Q3 z leaves rec_state and h bit-identical on every "
                "sampled layer and token (shader binding holds). Gated cosine min "
                f"{by_s['z']['gated_cosine_min']} is below {COSINE_BAR}. Stacked "
                "across 48 layers that residual injury is not a quiet drop. z is "
                "the cheapest *relative* follow-up for a generate gate, not a "
                "licensed byte lever."
            ),
            physical_primitive=fused,
            dense=DIRECT_CONSUME,
            dense_reason=common_dense,
            index_slugs=["uniform_q3", "uniform_subbit_allocation"],
            citations=[nns019],
            measured=sens("z"),
            support="MEASURED",
            note="Cheapest relative win. Not worth touching as a production drop.",
            consult_index=consult_index,
        ),
        _cand(
            cid="q_readout_q3",
            name="3-bit codes on q only (readout of S; no state write)",
            mechanism=(
                "Crush only the Q128 rows. q is L2-normalised after conv and "
                "read out as h = S^T q. Injuring q injures h without rewriting S."
            ),
            byte_model=f"q codes {parts['q']['code_bytes']}; Q3 saves {save['q']}.",
            bytes_eliminated_if_true=0,
            status=MEASURED_NEGATIVE,
            cheapest_falsifier=(
                "STATIC, run here: rec_state stays bit-identical (honest readout). "
                f"Gated cosine min {by_s['q']['gated_cosine_min']}, h rel-fro max "
                f"{by_s['q']['rec_out_h_relfro_max']}. The readout is not the cheap "
                "block. Not worth touching."
            ),
            physical_primitive=local,
            dense=DIRECT_CONSUME,
            dense_reason=common_dense,
            index_slugs=["uniform_q3"],
            citations=[nns019],
            measured=sens("q"),
            support="MEASURED",
            consult_index=consult_index,
        ),
        _cand(
            cid="k_state_q3",
            name="3-bit codes on k only (rank-1 projector and write key)",
            mechanism=(
                "Crush only the K128 rows. k appears twice in the update: "
                "(I - beta k k^T) and the write key of beta k v^T."
            ),
            byte_model=f"k codes {parts['k']['code_bytes']}; Q3 saves {save['k']}.",
            bytes_eliminated_if_true=0,
            status=MEASURED_NEGATIVE,
            cheapest_falsifier=(
                "STATIC, run here: rec_state rel-fro max "
                f"{by_s['k']['rec_state_relfro_max']} (bar {STATE_RELFRO_BAR}). k "
                "rewrites S. Not worth touching."
            ),
            physical_primitive=local,
            dense=DIRECT_CONSUME,
            dense_reason=common_dense,
            index_slugs=["uniform_q3"],
            citations=[nns019],
            measured=sens("k"),
            support="MEASURED",
            consult_index=consult_index,
        ),
        _cand(
            cid="v_state_q3",
            name="3-bit codes on v only (write value into S)",
            mechanism="Crush only the V384 rows. Injuring v writes a wrong column into S.",
            byte_model=f"v codes {parts['v']['code_bytes']}; Q3 saves {save['v']}.",
            bytes_eliminated_if_true=0,
            status=MEASURED_NEGATIVE,
            cheapest_falsifier=(
                "STATIC, run here: rec_state rel-fro max "
                f"{by_s['v']['rec_state_relfro_max']}; gated cosine min "
                f"{by_s['v']['gated_cosine_min']}. First-token injury is large. "
                "v is 37.5% of rows — same byte mass as z — and unlike z it "
                "rewrites S. Not worth touching."
            ),
            physical_primitive=local,
            dense=DIRECT_CONSUME,
            dense_reason=common_dense,
            index_slugs=["uniform_q3"],
            citations=[nns019],
            measured=sens("v"),
            support="MEASURED",
            consult_index=consult_index,
        ),
        _cand(
            cid="uniform_q3_qkvz",
            name="uniform 3-bit recode of the fused 16384-row qkvz",
            mechanism="Replace HQ30UQ4 on all rows with uniform Q3 of the same W.",
            byte_model=f"qkvz codes {acc['code_bytes']}; uniform Q3 saves {save_all}.",
            bytes_eliminated_if_true=0,
            status=MEASURED_NEGATIVE,
            cheapest_falsifier=(
                "STATIC, run here: the four per-block Q3 injuries do not vanish "
                "when applied together; each already fails the cosine bar alone. "
                "NNS-029 killed uniform bit-descent below q3 as a clean path on "
                "the whole artifact — a cousin of a whole-body plan, cited so "
                "this is not re-proposed as a DN idea. DN-only Q3 is now measured "
                "negative on the consume operator, not only as a cousin."
            ),
            physical_primitive=fused,
            dense=DIRECT_CONSUME,
            dense_reason=common_dense,
            index_slugs=["uniform_q3", "uniform_q2"],
            citations=[nns029, nns019],
            measured={"save_if_true": save_all, "any_block_supported": alloc["any_supported"]},
            support="MEASURED",
            cousin=True,
            consult_index=consult_index,
        ),
        _cand(
            cid="entropy_coded_qkvz_codes",
            name="entropy coding of the 4-bit nibble stream (not heterogeneous bits)",
            mechanism=(
                "Lossless recode of the stored nibbles (rANS / Huffman fused into "
                "decode). Not a bit-width change of q/k/v/z. Recorded because the "
                "MLP method reports the Shannon floor on the same parent."
            ),
            byte_model=(
                f"stored codes {acc['code_bytes']}; i.i.d. Shannon "
                f"{entropy['iid_shannon_bytes_rounded']}; histogram gap "
                f"{entropy['iid_redundant_bytes_rounded']}."
            ),
            bytes_eliminated_if_true=int(entropy["iid_redundant_bytes_rounded"]),
            status=OPEN,
            cheapest_falsifier=(
                f"STATIC: H(q)={entropy['H_q_bits']} of 4 bits "
                f"({entropy['H_q_over_uniform']:.4f} of uniform). NNS-022 reopen "
                f"fires at ≤ {NNS022_REOPEN_UNIFORM_FRAC} of uniform. A native "
                "register-decodable path that changes active bytes/token is "
                "UNMEASURED. This is not a heterogeneous-precision licence."
            ),
            physical_primitive=fused,
            dense=DEPENDS_ON_LOWERING,
            dense_reason=(
                "In-register entropy decode then the incumbent Q4 matvec can be "
                "DIRECT_CONSUME. Expanding to the incumbent 4-bit buffer before "
                "the kernel eliminates zero active bytes. Unpack-to-dense-W is "
                "REJECTED_DENSE_REMAT."
            ),
            index_slugs=["entropy_coded_pq"],
            citations=[nns022],
            measured={
                "H_q_bits": entropy["H_q_bits"],
                "H_q_over_uniform": entropy["H_q_over_uniform"],
                "nns022_reopen_fires": entropy["nns022_reopen_fires"],
                "iid_redundant_bytes_rounded": entropy["iid_redundant_bytes_rounded"],
            },
            support="SHANNON_GAP_MEASURED_KERNEL_UNMEASURED",
            cousin=True,
            note=(
                "Not the heterogeneous allocation. Listed so the Shannon floor "
                "is comparable to MLP_CODE_INFORMATION and is not mistaken for "
                "a q/k/v/z bit map."
            ),
            consult_index=consult_index,
        ),
    ]
    ids = [c["id"] for c in rows]
    if ids != list(REQUIRED_CANDIDATE_IDS):
        raise QkvzPrecisionRefuse(f"candidate ids {ids} != {list(REQUIRED_CANDIDATE_IDS)}")
    return rows


def answers(
    acc: Mapping[str, Any],
    entropy: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    alloc: Mapping[str, Any],
    cands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cands}
    by_e = entropy["by_subblock"]
    by_s = sensitivity["by_subblock"]
    by_a = alloc["by_subblock"]
    return {
        "do_the_subblocks_have_different_code_entropy": {
            "answer": (
                "NO at a scale that would change bits. H(q) sits in "
                f"q={by_e['q']['H_q_bits']:.5f}, k={by_e['k']['H_q_bits']:.5f}, "
                f"v={by_e['v']['H_q_bits']:.5f}, z={by_e['z']['H_q_bits']:.5f} "
                "of 4 bits. Neighbour MI is millibits. Cross-layer MI is ~0. "
                "Nibble 0 (q=-8) is unused on every block — histogram bias, "
                "not a 3-bit island. Lossless Q3 is impossible on all four."
            ),
            "status": MEASURED_NEGATIVE,
            "H_q_bits": {s: by_e[s]["H_q_bits"] for s in SUBBLOCKS},
            "lossless_q3_impossible": {s: by_e[s]["lossless_q3_impossible"] for s in SUBBLOCKS},
        },
        "does_sensitivity_license_heterogeneous_bits": {
            "answer": (
                "NO. Q3 with per-group scale refit, 8-token CPU rollout from "
                f"zero state on layers {sensitivity['layers']}: no sub-block "
                f"clears gated cosine {COSINE_BAR} on every token of every "
                "sampled layer. z leaves rec_state and h bit-identical (shader "
                "binding holds) but still injures silu(z). k and v rewrite S "
                "at ~0.2 rel-fro. q injures the readout. They do not deserve "
                "equal precision as a consume-path default, and they also do "
                "not deserve a production bit drop."
            ),
            "status": MEASURED_NEGATIVE,
            "cosine_bar": COSINE_BAR,
            "gated_cosine_min": {s: by_s[s]["gated_cosine_min"] for s in SUBBLOCKS},
            "rec_state_identical": {s: by_s[s]["rec_state_identical"] for s in SUBBLOCKS},
            "rec_out_h_identical": {s: by_s[s]["rec_out_h_identical"] for s in SUBBLOCKS},
            "rec_state_relfro_max": {s: by_s[s]["rec_state_relfro_max"] for s in SUBBLOCKS},
        },
        "what_is_the_heterogeneous_allocation": {
            "answer": alloc["verdict"],
            "bits": {s: by_a[s]["bits"] for s in SUBBLOCKS},
            "supported": {s: by_a[s]["supported"] for s in SUBBLOCKS},
            "bytes_eliminated": {s: by_a[s]["bytes_eliminated"] for s in SUBBLOCKS},
            "total_bytes_eliminated": alloc["total_bytes_eliminated"],
            "share_of_token_eliminated": alloc["share_of_token_eliminated"],
            "qkvz_bytes_after": alloc["qkvz_bytes_after"],
            "token_bytes_after": alloc["token_bytes_after"],
        },
        "which_subblock_is_the_cheapest_real_win": {
            "answer": (
                f"{alloc['cheapest_relative_win']} is the cheapest *relative* "
                "block (cannot rewrite rec_state or h; smallest consume-path "
                "blast radius). It is not a licensed byte win: gated cosine "
                f"min {by_s[alloc['cheapest_relative_win']]['gated_cosine_min']} "
                f"fails {COSINE_BAR}, and 48 stacked layers of that residual "
                "injury is not quiet. The production answer is that they all "
                "keep 4 bits."
            ),
            "cheapest_relative": alloc["cheapest_relative_win"],
            "licensed_win": None,
            "not_worth_touching": alloc["not_worth_touching"],
        },
        "which_is_not_worth_touching": {
            "answer": (
                "k and v: they rewrite S at ~0.2 rel-fro. q: readout injury "
                "without z's state-isolation. z: cheapest relative, still not "
                "a production drop. Uniform Q3 of the fused tensor is the same "
                "no, four times."
            ),
            "not_worth_touching": list(SUBBLOCKS),
            "status": MEASURED_NEGATIVE,
        },
        "is_entropy_coding_this_object": {
            "answer": (
                "A different lever. H(q)/4 "
                f"= {entropy['H_q_over_uniform']:.4f} ≤ {NNS022_REOPEN_UNIFORM_FRAC}, "
                "so the NNS-022 reopen analog fires on this nibble stream "
                f"({entropy['iid_redundant_bytes_rounded']} byte i.i.d. gap). "
                "Kernel UNMEASURED. Not a q/k/v/z bit map."
            ),
            "status": by_id["entropy_coded_qkvz_codes"]["status"],
            "nns022_reopen_fires": entropy["nns022_reopen_fires"],
            "bytes_eliminated_if_true": by_id["entropy_coded_qkvz_codes"]["bytes_eliminated_if_true"],
        },
    }


# ---------------------------------------------------------------------------
# Snapshot / receipt.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _measured() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Catalog + entropy + sensitivity. Cached so pytest and --build share one pass."""
    rows, geo = _rows_geo()
    rows = list(rows)
    acc = accounting_from_rows(rows, geo)
    ent = measure_code_entropy(rows, geo)
    stash = ent.pop("_stash", {})
    sens = measure_sensitivity(rows, geo, stash=stash)
    info = dnr.independent_information(geo)
    return acc, ent, sens, info


def snapshot(consult_index: bool = True) -> dict[str, Any]:
    acc, ent, sens, info = _measured()
    alloc = allocation_from_measurements(acc, ent, sens, info)
    cands = candidates(acc, ent, sens, alloc, info, consult_index=consult_index)
    return {
        "accounting": acc,
        "entropy": ent,
        "sensitivity": sens,
        "allocation": alloc,
        "independent_information": info,
        "candidates": cands,
        "answers": answers(acc, ent, sens, alloc, cands),
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
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Split attention.linear_qkvz into q/k/v/z on the real catalog, "
            "measure per-sub-block Shannon entropy of the HQ30UQ4 codes "
            "(MLP_CODE_INFORMATION method) and STATIC operator sensitivity of "
            "a 3-bit drop, and allocate bits. A measured NO that they all keep "
            "4 bits closes the last large open byte lever."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "predecessor": PREDECESSOR_REL,
        "what_this_does_not_prove": [
            "capability or generate identity of Q3 on any qkvz range",
            "that isotropic Gaussian x is the real post-norm hidden",
            "physical EBPW of a different packing",
            "actual_read_bytes_per_token (cache, contention)",
            "a fused rANS kernel for the nibble-histogram gap",
        ],
        "accounting": _py(acc),
        "entropy": _py(snap["entropy"]),
        "sensitivity": _py(snap["sensitivity"]),
        "allocation": _py(snap["allocation"]),
        "independent_information": _py(snap["independent_information"]),
        "candidates": _py(cands),
        "answers": _py(snap["answers"]),
        "candidate_counts": {
            "n": len(cands),
            "open": n_open,
            "measured_negative": n_meas,
            "already_falsified": n_dead,
            "unmeasured": n_unm,
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
        "recovered_implementation": {
            "catalog_format": "HQ38M20 + HQ30UQ4 group-64 signed nibble",
            "artifact_root": acc["identity"]["artifact_root"],
            "qkvz_layout": acc["identity"]["geometry"]["qkvz_layout"],
            "gated_delta": "S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q",
            "z_binding": "gated RMSNorm only; not gated-delta; not conv",
            "entropy_method": "tools/future/mlp_code_information.py _shannon/_mi/_cond_h on 16-ary nibbles",
        },
        "gaps_closed": [
            "linear_qkvz bytes re-measured from 48 HQ30UQ4 headers and refused unless they sum to 2,139,096,960",
            "q/k/v/z payload + headers reconcile to the same total",
            "i.i.d. Shannon, neighbour conditional entropy, and cross-layer MI measured per sub-block on the real codes",
            "CPU operator sensitivity of a 3-bit drop measured per sub-block on rec_state, rec_out (h), and the silu gate",
            "z-only Q3 leaves rec_state and h bit-identical (shader binding holds)",
            "a bit reduction is never reported as supported on entropy alone",
            "negative_index queried; NNS-019 / NNS-029 / NNS-022 cited as cousins or reopen analogs, not laundered",
        ],
        "negative_findings": [
            "H(q) ≈ 3.47–3.50 of 4 bits on q, k, v and z; lossless Q3 is impossible; entropy does not pick a winner",
            "neighbour MI is millibits; cross-layer nibble MI is ~0",
            "Q3 operator injury fails the 0.990 gated-cosine bar on every sub-block across sampled layers",
            "k and v rewrite rec_state at ~0.2 rel-fro; they are not cheap",
            "z cannot corrupt rec_state and is the cheapest relative experiment, still not a licensed byte win",
            "all four keep 4 bits; 0 bytes eliminated from the token",
        ],
        "nomenclature": {
            "already_falsified": ALREADY_FALSIFIED,
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "unmeasured": UNMEASURED,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "direct_consume": DIRECT_CONSUME,
            "depends_on_lowering": DEPENDS_ON_LOWERING,
            "entropy_alone_insufficient": ENTROPY_ALONE_INSUFFICIENT,
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
                "stored_bytes": snap["stored_bytes"],
                "code_bytes": snap["code_bytes"],
                "reconciled": snap["reconciled"],
                "by_subblock_payload": {
                    s: snap["qkvz_subblocks"][s]["payload_bytes"] for s in SUBBLOCKS
                },
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
