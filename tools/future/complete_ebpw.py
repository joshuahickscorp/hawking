#!/usr/bin/env python3
"""COMPLETE EXECUTABLE BPW — bill every part, refuse a guess.

The resident must never compute a bit-per-weight figure itself. It got
17.113e9 x 2.25 / 8 wrong: a rounded element count, the 2.25 coherent-class
floor instead of the packed 2.5, and only the MLP body. This module is the
deterministic calculator. A candidate is its declared parts — per-region
bitwidths, generators, metadata, tables, residuals, runtime auxiliaries —
and every part bills. A generator, a codebook and a lookup table are not
free. Declared parts that do not reconcile against a stated total are
REFUSED. A candidate that stores small but reconstructs the full parent
to execute is flagged DENSE_PARENT_REMATERIALIZATION; the complete
executable BPW is the parent working set, not the flattering stored
number, and it is not a sub-2 executable.

Time is billed at the stream-class rates from executable_economics /
ECONOMICS_CALIBRATION.json: weight_codes 0.547282 ms/GB, broadcast_aux
0.000. A byte cut that saves no time is not a speed lever. Both bytes
and milliseconds are reported.

    python3 tools/future/complete_ebpw.py --build
    python3 -m pytest tools/future/test_complete_ebpw.py -q

This module MEASURES NOTHING. It is STATIC_ONLY arithmetic over MIX_REPORT
and the sealed stream-class calibration. Missing input REFUSES.
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import REPO, write_receipt


RECORDED_BY = "tools/future/complete_ebpw.py"
RECEIPT = "COMPLETE_EBPW.json"
SCHEMA = "hawking.future.complete_ebpw.v1"
VERSION = 1
EVIDENCE_CLASS = "STATIC_ONLY"

MIX_REPORT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A/MIX_REPORT.json")
ECON_REL = "receipts/future/ECONOMICS_CALIBRATION.json"

STREAM_WEIGHT_CODES = "weight_codes"
STREAM_BROADCAST_AUX = "broadcast_aux"
STREAM_ACTIVATION = "activation"
STREAM_CLASS_NAMES = frozenset(
    {STREAM_WEIGHT_CODES, STREAM_BROADCAST_AUX, STREAM_ACTIVATION}
)

DENSE_PARENT_REMATERIALIZATION = "DENSE_PARENT_REMATERIALIZATION"
SUB2_BPW = 2.0
INCUMBENT_BPW_LABEL = 3.1393
INCUMBENT_BPW_TOLERANCE = 0.001
GB = 1e9

# Expected catalog rates. Loaded from the calibration receipt; a mismatch
# REFUSES rather than silently billing a different measurement.
CITED_WEIGHT_CODES_MS_PER_GB = 0.547282
CITED_BROADCAST_AUX_MS_PER_GB = 0.0

PART_CATEGORIES = (
    "regions",
    "generators",
    "metadata",
    "tables",
    "residuals",
    "runtime_auxiliaries",
)

REQUIRED_CANDIDATE_KEYS = (
    "id",
    "parent_params",
    "stated_total_bytes",
    *PART_CATEGORIES,
    "reconstructs_dense_parent",
    "consumes_representation_directly",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. complete_ebpw is arithmetic over a candidate's "
    "declared parts and MIX_REPORT parent_params. ms figures are unique bytes "
    "times the stream-class ms/GB in ECONOMICS_CALIBRATION.json "
    "(weight_codes 0.547282, broadcast_aux 0.000). They are not a protected "
    "measurement and not a promise that a fit will hold capability. "
    "evidence_class is STATIC_ONLY. gpu_authority is false."
)


class CompleteEbpwRefused(RuntimeError):
    """An input is missing or self-inconsistent; guessing would be a fake BPW."""


def _r(value: float, n: int = 6) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def _gb(n_bytes: int | float) -> float:
    return _r(float(n_bytes) / GB, 6)


def _bpw(n_bytes: int | float, n_params: int) -> float:
    if n_params <= 0:
        raise CompleteEbpwRefused("parent_params is not positive; bpw is undefined")
    return float(n_bytes) * 8.0 / float(n_params)


def _need(d: Mapping[str, Any], *keys: str, source: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            raise CompleteEbpwRefused(f"{source} is missing {'.'.join(keys)}")
        cur = cur[k]
    return cur


def _require_int(value: Any, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompleteEbpwRefused(f"{what} must be an int, got {value!r}")
    if value < 0:
        raise CompleteEbpwRefused(f"{what} cannot be negative: {value}")
    return value


def _require_bool(value: Any, *, what: str) -> bool:
    if not isinstance(value, bool):
        raise CompleteEbpwRefused(f"{what} must be a bool, got {value!r}")
    return value


def _load_json(path: Path, *, why: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompleteEbpwRefused(
            f"{path} is not on disk; {why}. A calculator with a missing input "
            "is a guess wearing a receipt"
        )
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise CompleteEbpwRefused(f"{path} is not a JSON object")
    return d


def mix_report() -> dict[str, Any]:
    d = _load_json(
        MIX_REPORT, why="the incumbent artifact report is the payload authority"
    )
    for k in (
        "payload_bytes",
        "parent_params",
        "mlp_elements",
        "affine_bytes",
        "q4_bytes",
        "f32_bytes",
        "storage_bpw",
        "affine_bpw_billing",
    ):
        if k not in d:
            raise CompleteEbpwRefused(f"MIX_REPORT.json is missing {k}")
    payload = _require_int(d["payload_bytes"], what="MIX_REPORT.payload_bytes")
    parts = (
        _require_int(d["affine_bytes"], what="MIX_REPORT.affine_bytes")
        + _require_int(d["q4_bytes"], what="MIX_REPORT.q4_bytes")
        + _require_int(d["f32_bytes"], what="MIX_REPORT.f32_bytes")
    )
    if parts != payload:
        raise CompleteEbpwRefused(
            f"MIX_REPORT parts {parts} != payload_bytes {payload}; the split "
            "does not add and must not be billed"
        )
    return d


def stream_rates() -> dict[str, Any]:
    """Catalog ms/GB by stream class. Missing or mismatched calibration REFUSES."""
    p = REPO / ECON_REL
    d = _load_json(p, why="stream-class ms/GB is the time-cost authority")
    classes = d.get("stream_classes")
    if not isinstance(classes, dict):
        raise CompleteEbpwRefused(f"{ECON_REL} has no stream_classes object")
    out: dict[str, Any] = {}
    for name in (STREAM_WEIGHT_CODES, STREAM_BROADCAST_AUX, STREAM_ACTIVATION):
        row = classes.get(name)
        if not isinstance(row, dict) or "ms_per_gb_saved" not in row:
            raise CompleteEbpwRefused(
                f"{ECON_REL} stream_classes.{name}.ms_per_gb_saved is absent"
            )
        out[name] = {
            "ms_per_gb": float(row["ms_per_gb_saved"]),
            "on_critical_path": bool(row.get("on_critical_path")),
            "source": ECON_REL,
        }
    aux = out[STREAM_BROADCAST_AUX]["ms_per_gb"]
    if aux != CITED_BROADCAST_AUX_MS_PER_GB:
        raise CompleteEbpwRefused(
            f"{ECON_REL} broadcast_aux ms/GB is {aux}, not "
            f"{CITED_BROADCAST_AUX_MS_PER_GB}; the size-not-time reading would "
            "be a different measurement"
        )
    codes = out[STREAM_WEIGHT_CODES]["ms_per_gb"]
    if abs(codes - CITED_WEIGHT_CODES_MS_PER_GB) > 1e-9:
        raise CompleteEbpwRefused(
            f"{ECON_REL} weight_codes ms/GB is {codes}, not "
            f"{CITED_WEIGHT_CODES_MS_PER_GB}; refusing to bill a different rate"
        )
    return out


def packed_bytes(*, elements: int, bitwidth: Any, what: str) -> int:
    """elements x bitwidth / 8, exact. Fractional bits or leftover bits REFUSE.

    The resident's 17.113e9 x 2.25 / 8 is the thing this exists to stop:
    a rounded count times a floor that is not the packing, silently.
    """
    n = _require_int(elements, what=f"{what}.elements")
    if isinstance(bitwidth, bool) or not isinstance(bitwidth, (int, float)):
        raise CompleteEbpwRefused(f"{what}.bitwidth must be a number, got {bitwidth!r}")
    if bitwidth < 0:
        raise CompleteEbpwRefused(f"{what}.bitwidth cannot be negative: {bitwidth}")
    frac = Fraction(bitwidth).limit_denominator(256)
    if abs(float(frac) - float(bitwidth)) > 1e-12:
        raise CompleteEbpwRefused(
            f"{what}.bitwidth {bitwidth} is not a short rational; refusing to "
            "round a bitwidth"
        )
    bits = frac * n
    if bits.denominator != 1:
        raise CompleteEbpwRefused(
            f"{what}: {n} elements x {bitwidth} bpw is {bits} bits, not an "
            "integer; refusing to round a packed body"
        )
    n_bits = int(bits)
    if n_bits % 8 != 0:
        raise CompleteEbpwRefused(
            f"{what}: {n_bits} bits is not divisible by 8; refusing to round "
            "a packed body to a byte"
        )
    return n_bits // 8


def _normalize_part(
    row: Any, *, category: str, index: int
) -> dict[str, Any]:
    loc = f"{category}[{index}]"
    if not isinstance(row, Mapping):
        raise CompleteEbpwRefused(f"{loc} is not an object")
    if "name" not in row:
        raise CompleteEbpwRefused(f"{loc} is missing name")
    name = row["name"]
    if not isinstance(name, str) or not name:
        raise CompleteEbpwRefused(f"{loc}.name must be a non-empty string")
    if "stream_class" not in row:
        raise CompleteEbpwRefused(
            f"{loc} ({name}) is missing stream_class; defaulting to the organ "
            "average is how an aux byte gets billed as time"
        )
    stream = row["stream_class"]
    if stream not in STREAM_CLASS_NAMES:
        raise CompleteEbpwRefused(
            f"{loc} ({name}) has unknown stream_class {stream!r}; known: "
            f"{sorted(STREAM_CLASS_NAMES)}"
        )
    has_bytes = "bytes" in row
    has_elements = "elements" in row
    has_bitwidth = "bitwidth" in row
    if has_elements ^ has_bitwidth:
        raise CompleteEbpwRefused(
            f"{loc} ({name}) has elements or bitwidth but not both"
        )
    if not has_bytes and not has_elements:
        raise CompleteEbpwRefused(
            f"{loc} ({name}) has neither bytes nor elements+bitwidth; "
            "refusing to default a part to zero"
        )
    computed: int | None = None
    if has_elements:
        computed = packed_bytes(
            elements=row["elements"],
            bitwidth=row["bitwidth"],
            what=f"{loc} ({name})",
        )
    if has_bytes:
        declared = _require_int(row["bytes"], what=f"{loc} ({name}).bytes")
        if computed is not None and declared != computed:
            raise CompleteEbpwRefused(
                f"{loc} ({name}) declares bytes={declared} but "
                f"elements x bitwidth / 8 = {computed}; the part does not "
                "reconcile"
            )
        n_bytes = declared
    else:
        assert computed is not None
        n_bytes = computed
    part: dict[str, Any] = {
        "category": category,
        "name": name,
        "bytes": n_bytes,
        "gb": _gb(n_bytes),
        "stream_class": stream,
    }
    if has_elements:
        part["elements"] = _require_int(
            row["elements"], what=f"{loc} ({name}).elements"
        )
        part["bitwidth"] = float(row["bitwidth"])
    return part


def _parts_of(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    missing = [k for k in REQUIRED_CANDIDATE_KEYS if k not in candidate]
    if missing:
        raise CompleteEbpwRefused(
            f"candidate is missing {missing}; refusing to default a missing "
            "input (an omitted generator is not a zero-byte generator)"
        )
    out: list[dict[str, Any]] = []
    for category in PART_CATEGORIES:
        rows = candidate[category]
        if not isinstance(rows, list):
            raise CompleteEbpwRefused(f"{category} must be a list, got {type(rows).__name__}")
        for i, row in enumerate(rows):
            out.append(_normalize_part(row, category=category, index=i))
    return out


def _bill_ms(n_bytes: int, stream: str, rates: Mapping[str, Any]) -> float:
    return _r(_gb(n_bytes) * float(rates[stream]["ms_per_gb"]), 6)


def incumbent_candidate() -> dict[str, Any]:
    """The sealed-3.14 mix, as a candidate, from MIX_REPORT parts only.

    MLP body is billed from affine_bpw_billing (2.0 codes + 0.25 scale +
    0.25 bias). The leftover of affine_bytes is the per-tensor headers.
    q4 attention/DeltaNet and f32 are the MIX_REPORT byte totals; MIX_REPORT
    does not split q4 codes from q4 scale, so q4_bytes is declared as one
    weight_codes part rather than invented as two.
    """
    mix = mix_report()
    billing = mix["affine_bpw_billing"]
    for k in ("codes_bpw", "scale_bpw", "bias_bpw", "total_bpw"):
        if k not in billing:
            raise CompleteEbpwRefused(f"MIX_REPORT.affine_bpw_billing is missing {k}")
    codes_bpw = float(billing["codes_bpw"])
    scale_bpw = float(billing["scale_bpw"])
    bias_bpw = float(billing["bias_bpw"])
    total_bpw = float(billing["total_bpw"])
    if abs((codes_bpw + scale_bpw + bias_bpw) - total_bpw) > 1e-12:
        raise CompleteEbpwRefused(
            f"MIX_REPORT affine_bpw_billing {codes_bpw}+{scale_bpw}+{bias_bpw} "
            f"!= total_bpw {total_bpw}"
        )
    elements = _require_int(mix["mlp_elements"], what="MIX_REPORT.mlp_elements")
    affine = _require_int(mix["affine_bytes"], what="MIX_REPORT.affine_bytes")
    body = packed_bytes(
        elements=elements, bitwidth=total_bpw, what="incumbent.mlp_body"
    )
    headers = affine - body
    if headers < 0:
        raise CompleteEbpwRefused(
            f"MIX_REPORT affine_bytes {affine} is smaller than the "
            f"{total_bpw} bpw body {body}"
        )
    return {
        "id": "incumbent_sealed_3_14",
        "parent_params": _require_int(
            mix["parent_params"], what="MIX_REPORT.parent_params"
        ),
        "stated_total_bytes": _require_int(
            mix["payload_bytes"], what="MIX_REPORT.payload_bytes"
        ),
        "regions": [
            {
                "name": "mlp_codes",
                "elements": elements,
                "bitwidth": codes_bpw,
                "stream_class": STREAM_WEIGHT_CODES,
            },
            {
                "name": "mlp_scale",
                "elements": elements,
                "bitwidth": scale_bpw,
                "stream_class": STREAM_BROADCAST_AUX,
            },
            {
                "name": "mlp_bias",
                "elements": elements,
                "bitwidth": bias_bpw,
                "stream_class": STREAM_BROADCAST_AUX,
            },
            {
                "name": "q4_attention_deltanet",
                "bytes": _require_int(mix["q4_bytes"], what="MIX_REPORT.q4_bytes"),
                "stream_class": STREAM_WEIGHT_CODES,
            },
            {
                "name": "f32",
                "bytes": _require_int(mix["f32_bytes"], what="MIX_REPORT.f32_bytes"),
                "stream_class": STREAM_WEIGHT_CODES,
            },
        ],
        "generators": [],
        "metadata": [
            {
                "name": "mlp_headers",
                "bytes": headers,
                "stream_class": STREAM_BROADCAST_AUX,
            }
        ],
        "tables": [],
        "residuals": [],
        "runtime_auxiliaries": [],
        "reconstructs_dense_parent": False,
        "consumes_representation_directly": True,
        "source": str(MIX_REPORT),
    }


def cost(
    candidate: Mapping[str, Any],
    *,
    versus: Mapping[str, Any] | None = None,
    rates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete executable BPW of one candidate. Nothing is free."""
    if not isinstance(candidate, Mapping):
        raise CompleteEbpwRefused("candidate must be an object")
    parts = _parts_of(candidate)
    parent_params = _require_int(
        candidate["parent_params"], what="candidate.parent_params"
    )
    if parent_params <= 0:
        raise CompleteEbpwRefused("parent_params is not positive; bpw is undefined")
    stated = _require_int(
        candidate["stated_total_bytes"], what="candidate.stated_total_bytes"
    )
    parts_bytes = sum(int(p["bytes"]) for p in parts)
    if parts_bytes != stated:
        raise CompleteEbpwRefused(
            f"declared parts sum to {parts_bytes} bytes but stated_total_bytes "
            f"is {stated}; an unreconciled candidate is refused, not billed"
        )

    remat = _require_bool(
        candidate["reconstructs_dense_parent"],
        what="candidate.reconstructs_dense_parent",
    )
    consumes = _require_bool(
        candidate["consumes_representation_directly"],
        what="candidate.consumes_representation_directly",
    )
    if remat and consumes:
        raise CompleteEbpwRefused(
            "candidate claims reconstructs_dense_parent and "
            "consumes_representation_directly; those cannot both be true"
        )
    if not remat and not consumes:
        raise CompleteEbpwRefused(
            "candidate declares neither reconstructs_dense_parent nor "
            "consumes_representation_directly; refusing to assume a consume path"
        )

    spec = rates if rates is not None else stream_rates()
    billed_parts = []
    by_stream = {
        STREAM_WEIGHT_CODES: {"bytes": 0, "ms": 0.0},
        STREAM_BROADCAST_AUX: {"bytes": 0, "ms": 0.0},
        STREAM_ACTIVATION: {"bytes": 0, "ms": 0.0},
    }
    stored_ms = 0.0
    for p in parts:
        ms = _bill_ms(int(p["bytes"]), p["stream_class"], spec)
        billed = {**p, "ms": ms, "ms_per_gb": spec[p["stream_class"]]["ms_per_gb"]}
        billed_parts.append(billed)
        by_stream[p["stream_class"]]["bytes"] += int(p["bytes"])
        by_stream[p["stream_class"]]["ms"] = _r(
            by_stream[p["stream_class"]]["ms"] + ms, 6
        )
        stored_ms = _r(stored_ms + ms, 6)

    stored_bytes = parts_bytes
    stored_bpw = _bpw(stored_bytes, parent_params)
    flags: list[str] = []
    remat_block: dict[str, Any]

    if remat:
        if "parent_executable_bytes" not in candidate:
            raise CompleteEbpwRefused(
                "reconstructs_dense_parent is true but parent_executable_bytes "
                "is missing; refusing to guess the rematerialized working set"
            )
        if "parent_stream_class" not in candidate:
            raise CompleteEbpwRefused(
                "reconstructs_dense_parent is true but parent_stream_class is "
                "missing; refusing to default the parent stream"
            )
        parent_bytes = _require_int(
            candidate["parent_executable_bytes"],
            what="candidate.parent_executable_bytes",
        )
        parent_stream = candidate["parent_stream_class"]
        if parent_stream not in STREAM_CLASS_NAMES:
            raise CompleteEbpwRefused(
                f"parent_stream_class {parent_stream!r} is unknown; known: "
                f"{sorted(STREAM_CLASS_NAMES)}"
            )
        if parent_bytes <= stored_bytes:
            raise CompleteEbpwRefused(
                f"parent_executable_bytes {parent_bytes} is not larger than "
                f"stored {stored_bytes}; this is not a rematerializing working set"
            )
        executable_bytes = parent_bytes
        executable_ms = _bill_ms(parent_bytes, parent_stream, spec)
        complete_ebpw = _bpw(executable_bytes, parent_params)
        flags.append(DENSE_PARENT_REMATERIALIZATION)
        remat_block = {
            "flagged": True,
            "flag": DENSE_PARENT_REMATERIALIZATION,
            "reason": (
                "stores small but reconstructs the full parent to execute; "
                "complete executable BPW is the parent working set, not the "
                "stored figure, and this is not a sub-2 executable"
            ),
            "stored_bytes": stored_bytes,
            "stored_bpw": stored_bpw,
            "parent_executable_bytes": parent_bytes,
            "parent_stream_class": parent_stream,
            "complete_ebpw": complete_ebpw,
            "is_sub2_executable": False,
        }
        is_sub2 = False
    else:
        executable_bytes = stored_bytes
        executable_ms = stored_ms
        complete_ebpw = stored_bpw
        remat_block = {
            "flagged": False,
            "flag": None,
            "reason": "production consumes the representation directly",
            "stored_bytes": stored_bytes,
            "stored_bpw": stored_bpw,
            "complete_ebpw": complete_ebpw,
            "is_sub2_executable": complete_ebpw < SUB2_BPW,
        }
        is_sub2 = complete_ebpw < SUB2_BPW

    for k, v in by_stream.items():
        v["gb"] = _gb(v["bytes"])
        v["ms"] = _r(v["ms"], 6)
        v["ms_per_gb"] = spec[k]["ms_per_gb"]

    row: dict[str, Any] = {
        "id": candidate["id"],
        "parent_params": parent_params,
        "stated_total_bytes": stated,
        "parts_bytes": parts_bytes,
        "reconciled": True,
        "stored_bytes": stored_bytes,
        "stored_gb": _gb(stored_bytes),
        "stored_bpw": stored_bpw,
        "stored_ms": stored_ms,
        "executable_bytes": executable_bytes,
        "executable_gb": _gb(executable_bytes),
        "complete_ebpw": complete_ebpw,
        "billed_ms": executable_ms,
        "by_stream_class": by_stream,
        "parts": billed_parts,
        "flags": flags,
        "is_sub2_executable": is_sub2,
        "dense_parent_rematerialization": remat_block,
        "reconstructs_dense_parent": remat,
        "consumes_representation_directly": consumes,
        "nothing_is_free": (
            "a generator, a codebook and a lookup table all bill; "
            "omitting a part category is a refusal, not a zero"
        ),
    }
    if versus is not None:
        base = versus if "complete_ebpw" in versus and "billed_ms" in versus else cost(
            versus, rates=spec
        )
        bytes_saved = int(base["executable_bytes"]) - int(executable_bytes)
        ms_saved = _r(float(base["billed_ms"]) - float(executable_ms), 6)
        row["versus"] = {
            "id": base.get("id"),
            "bytes_saved": bytes_saved,
            "gb_saved": _gb(bytes_saved),
            "ms_saved": ms_saved,
            "bpw_delta": float(base["complete_ebpw"]) - complete_ebpw,
            "baseline_executable_bytes": int(base["executable_bytes"]),
            "baseline_billed_ms": float(base["billed_ms"]),
            "baseline_complete_ebpw": float(base["complete_ebpw"]),
        }
    return row


def _selftest() -> dict[str, Any]:
    """Run the acceptance checks. A failing selftest REFUSES the receipt."""
    inc_cand = incumbent_candidate()
    inc = cost(inc_cand)
    bpw = float(inc["complete_ebpw"])
    if abs(bpw - INCUMBENT_BPW_LABEL) > INCUMBENT_BPW_TOLERANCE:
        raise CompleteEbpwRefused(
            f"incumbent complete_ebpw {bpw} is not {INCUMBENT_BPW_LABEL} "
            f"within {INCUMBENT_BPW_TOLERANCE}"
        )

    unreconciled_refused = False
    bad = {
        **inc_cand,
        "id": "unreconciled",
        "stated_total_bytes": int(inc_cand["stated_total_bytes"]) + 1,
    }
    try:
        cost(bad)
    except CompleteEbpwRefused as exc:
        unreconciled_refused = "unreconciled" in str(exc).lower() or "stated_total" in str(exc)

    remat_cand = {
        **inc_cand,
        "id": "tiny_store_dense_parent",
        "stated_total_bytes": 1_000_000,
        "regions": [
            {
                "name": "generator_only_body",
                "bytes": 1_000_000,
                "stream_class": STREAM_WEIGHT_CODES,
            }
        ],
        "generators": [],
        "metadata": [],
        "tables": [],
        "residuals": [],
        "runtime_auxiliaries": [],
        "reconstructs_dense_parent": True,
        "consumes_representation_directly": False,
        "parent_executable_bytes": int(inc_cand["stated_total_bytes"]),
        "parent_stream_class": STREAM_WEIGHT_CODES,
    }
    remat = cost(remat_cand)
    remat_flagged = DENSE_PARENT_REMATERIALIZATION in remat["flags"]
    remat_not_flattering = remat["complete_ebpw"] >= inc["complete_ebpw"] - 1e-12
    remat_not_sub2 = remat["is_sub2_executable"] is False

    aux_cand = {
        **inc_cand,
        "id": "aux_only_header_cut",
        "metadata": [
            {
                "name": "mlp_headers",
                "bytes": 0,
                "stream_class": STREAM_BROADCAST_AUX,
            }
        ],
        "stated_total_bytes": int(inc_cand["stated_total_bytes"])
        - int(inc_cand["metadata"][0]["bytes"]),
    }
    aux = cost(aux_cand, versus=inc)
    aux_zero = float(aux["versus"]["ms_saved"]) == 0.0
    aux_bytes = int(aux["versus"]["bytes_saved"]) > 0

    failed = [
        name
        for name, held in (
            ("unreconciled_refused", unreconciled_refused),
            ("remat_flagged", remat_flagged),
            ("remat_not_flattering", remat_not_flattering),
            ("remat_not_sub2", remat_not_sub2),
            ("aux_zero", aux_zero),
            ("aux_bytes", aux_bytes),
        )
        if not held
    ]
    if failed:
        raise CompleteEbpwRefused(f"complete_ebpw selftest failed: {failed}")
    return {
        "incumbent_bpw": bpw,
        "incumbent_bpw_within_0_001_of_3_1393": True,
        "unreconciled_refused": True,
        "dense_parent_rematerialization_flagged": True,
        "remat_complete_ebpw_is_not_the_stored_figure": True,
        "aux_only_cut_ms_saved": 0.0,
        "aux_only_cut_bytes_saved": int(aux["versus"]["bytes_saved"]),
        "aux_only_cut_ms_saved_is_zero": True,
    }


def build() -> dict[str, Any]:
    mix = mix_report()
    rates = stream_rates()
    inc_cand = incumbent_candidate()
    inc = cost(inc_cand, rates=rates)
    selftest = _selftest()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": EVIDENCE_CLASS,
        "obligation": (
            "complete executable BPW calculator; the resident must not "
            "compute a bit-per-weight figure itself"
        ),
        "question": "What does a candidate representation actually cost, billing everything?",
        "claim_boundary": CLAIM_BOUNDARY,
        "incumbent": {
            "id": inc["id"],
            "source": str(MIX_REPORT),
            "parent_params": inc["parent_params"],
            "payload_bytes": inc["stated_total_bytes"],
            "mlp_elements": int(mix["mlp_elements"]),
            "mlp_body_bpw": float(mix["affine_bpw_billing"]["total_bpw"]),
            "affine_bytes": int(mix["affine_bytes"]),
            "q4_attention_deltanet_bytes": int(mix["q4_bytes"]),
            "f32_bytes": int(mix["f32_bytes"]),
            "mix_storage_bpw": float(mix["storage_bpw"]),
            "complete_ebpw": inc["complete_ebpw"],
            "complete_ebpw_matches_mix": abs(
                inc["complete_ebpw"] - float(mix["storage_bpw"])
            )
            < 1e-12,
            "stored_bytes": inc["stored_bytes"],
            "billed_ms": inc["billed_ms"],
            "by_stream_class": inc["by_stream_class"],
            "parts": inc["parts"],
            "is_sub2_executable": inc["is_sub2_executable"],
            "flags": inc["flags"],
            "why_the_resident_must_not_do_this_itself": (
                "17.113e9 x 2.25 / 8 uses a rounded element count, the 2.25 "
                "coherent-class floor instead of the packed 2.5, and only the "
                "MLP body. The complete incumbent is "
                f"{inc['stated_total_bytes']} payload bytes / "
                f"{inc['parent_params']} params = {inc['complete_ebpw']:.6f} BPW."
            ),
        },
        "stream_rates": {
            name: {
                "ms_per_gb": rates[name]["ms_per_gb"],
                "on_critical_path": rates[name]["on_critical_path"],
                "source": rates[name]["source"],
            }
            for name in (
                STREAM_WEIGHT_CODES,
                STREAM_BROADCAST_AUX,
                STREAM_ACTIVATION,
            )
        },
        "rules": {
            "missing_input": "REFUSE; never default a missing part category to empty-as-zero",
            "unreconciled_stated_total": "REFUSE",
            "generator_codebook_lookup_table": "all bill",
            "dense_parent_rematerialization": (
                "flag DENSE_PARENT_REMATERIALIZATION; complete_ebpw is the "
                "parent working set; is_sub2_executable is false"
            ),
            "broadcast_aux_ms_per_gb": CITED_BROADCAST_AUX_MS_PER_GB,
            "weight_codes_ms_per_gb": CITED_WEIGHT_CODES_MS_PER_GB,
            "sub2_bpw": SUB2_BPW,
        },
        "selftest": selftest,
        "what_is_measured": (
            "MIX_REPORT payload/params/affine/q4/f32 split and affine_bpw_billing; "
            "ECONOMICS_CALIBRATION stream-class ms/GB "
            f"(weight_codes {CITED_WEIGHT_CODES_MS_PER_GB}, broadcast_aux "
            f"{CITED_BROADCAST_AUX_MS_PER_GB})."
        ),
        "what_is_estimated": (
            "ms of a candidate is unique declared bytes times the catalog "
            "stream-class rate, not a complete-token re-measure of a new packing. "
            "q4_bytes is billed as one declared weight_codes part because "
            "MIX_REPORT does not split q4 codes from q4 scale."
        ),
        "load_bearing": [
            {
                "id": "payload_bytes",
                "value": inc["stated_total_bytes"],
                "source": str(MIX_REPORT),
                "role": "incumbent storage payload / stated total",
            },
            {
                "id": "parent_params",
                "value": inc["parent_params"],
                "source": str(MIX_REPORT),
                "role": "bpw denominator",
            },
            {
                "id": "complete_ebpw",
                "value": inc["complete_ebpw"],
                "source": "derived: payload_bytes * 8 / parent_params",
                "role": "incumbent complete executable BPW",
            },
            {
                "id": "weight_codes_ms_per_gb",
                "value": rates[STREAM_WEIGHT_CODES]["ms_per_gb"],
                "source": ECON_REL,
                "role": "binding-stream time cost",
            },
            {
                "id": "broadcast_aux_ms_per_gb",
                "value": rates[STREAM_BROADCAST_AUX]["ms_per_gb"],
                "source": ECON_REL,
                "role": "aux time cost; 0 means SIZE not TIME",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT, doc, RECORDED_BY))
        return 0
    print(
        json.dumps(
            {
                "incumbent_complete_ebpw": doc["incumbent"]["complete_ebpw"],
                "selftest": doc["selftest"],
                "stream_rates": doc["stream_rates"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
