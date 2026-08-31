#!/usr/bin/env python3
"""GEOMETRY IS A TABLE, NOT A CONSTANT.

The campaign already swept geometry on one representative MLP layer and did
not find a single winner: the answer depends on what you hold fixed. Raising
occupancy made the issue-rate TG sweep worse (64->1024: 309/323/320/291/236
GB/s). Accumulator chains spanned 1.062. Working-set spanned 1.078 at occupancy
span 1.0. Stream-count peaked at mid_2_4_32 (526.6 GB/s); merging further hurt.

This sidecar sweeps launch geometry across the resident's REAL HOT DIMENSIONS
and produces a (rows, cols, dtype, organ) -> winning-geometry table THE
COMPILER CONSULTS. A shape absent from the table gets a named fallback
(UNMEASURED_SHAPE), never a silent default to geo_tpr64_tg128.

    python3 tools/future/geometry_table.py --record
    python3 tools/future/geometry_table.py --from receipts/future/_GEOMETRY_TABLE_raw.json --record
    python3 tools/future/geometry_table.py --measure --record
    python3 -m pytest tools/future/test_geometry_table.py -q

evidence_class SELF_MEASURED_DIRTY. Absolute GB/s is measured-under-load.
Does not change the production decode path. Does not edit production shaders.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.future._common import (
    REPO,
    measurement_provenance,
    write_measured_receipt,
)


RECEIPT = REPO / "receipts" / "future" / "GEOMETRY_TABLE.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_GEOMETRY_TABLE_raw.json"
SCHEMA = "hawking.future.geometry_table.v1"
VERSION = 1
RECORDED_BY = "tools/future/geometry_table.py"

UNMEASURED_SHAPE = "UNMEASURED_SHAPE"
STATUS_HIT = "HIT"
STATUS_UNMEASURED = UNMEASURED_SHAPE

PRODUCTION_GEO_ID = "tg128_r2"
PRODUCTION_TG = 128
PRODUCTION_ROWS_PER_TG = 2
PRODUCTION_TPR = 64
PRODUCTION_PACKING_AFFINE = "mlp_2_2_2_32"
PACKING_MID = "mid_2_4_32"

# Cited, not this-run. PATH_TO_71 / ORGAN_BANDWIDTH / TOKEN_NS_OBJECTIVE /
# MLP_STREAM_COUNT / MLP_ISSUE_RATE_LADDER.
TOKEN_MS = 28.722
TOKEN_TPS = 34.82
MLP_MS = 15.541
DELTANET_MS = 8.227
GQA_MS = 2.607
LM_HEAD_MS = 1.358
TOKEN_TPS_ACCEPTED_CONTROL = 35.158
CAPABILITY_FLOOR_OVER_43 = 30
BYTES_PER_TOKEN = 9_878_901_136
SPECIALIZED_COLS = (5120, 17408)

# Closed on one representative MLP layer. Do not re-open.
REFUTED_DISCRIMINATORS = {
    "accumulator_chain": {
        "receipt": "receipts/future/MLP_ISSUE_RATE_LADDER.json",
        "span": 1.062,
        "gb_s": (308.3, 328.5, 325.5, 327.4),
        "verdict": "NOT a dependency chain",
        "do_not_rerun": True,
    },
    "working_set": {
        "receipt": "receipts/future/MLP_ISSUE_RATE_LADDER.json",
        "span": 1.078,
        "occupancy_span": 1.0,
        "gb_s": (308.3, 332.4, 332.2, 332.4),
        "verdict": "NOT register pressure",
        "do_not_rerun": True,
    },
    "stream_merge_beyond_mid_2_4_32": {
        "receipt": "receipts/future/MLP_STREAM_COUNT.json",
        "peak": {"id": "mid_2_4_32", "gb_s": 526.6},
        "hurt": {"pack_6_32": 218.7, "pack_38": 45.6},
        "verdict": "merging further than mid_2_4_32 is measured to hurt",
        "do_not_rerun": True,
    },
}

REFUSED_FAMILIES = frozenset(
    {
        "ilp",
        "accumulator_chain",
        "register_pressure",
        "working_set",
        "pack_6_32",
        "pack_38",
        "pack6",
        "pack38",
    }
)
REFUSED_IDS = frozenset(
    {
        "ilp2",
        "ilp4",
        "ilp8",
        "ws0",
        "ws8",
        "ws16",
        "ws32",
        "pack_6_32",
        "pack_38",
    }
)

# qwen38_geometry.rs ARGMAX_GROUPS comment + KERNEL_GEOMETRY occupancy_class.
# Cited, not a this-run hardware occupancy counter.
GPU_CORES_CITED = 60
GPU_CORES_PROVENANCE = (
    "crates/hawking-core/src/model/qwen38_geometry.rs ARGMAX_GROUPS comment "
    "('240 = four waves over 60 GPU cores'); receipts/future/KERNEL_GEOMETRY.json "
    "occupancy_class.gpu_cores=60. Not a this-run hardware occupancy counter."
)

CLAIM_BOUNDARY = (
    "Sealed-3.14 Qwen3.8-27B mixed catalog (HQ38M20) hot GEMVs only. "
    "SELF_MEASURED_DIRTY. GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime "
    "for an isolated command buffer of one diagnostic kernel. Bytes are the "
    "GPU-resident unique payload (codes+scales[+biases]) of the launched "
    "tensor; bandwidth is those bytes divided by GPU ns (perfect-locality). "
    "Launch-geometry candidates vary threadgroup size and rows-per-threadgroup "
    "with production arithmetic. Stream packing is only production 2+2+2+32 vs "
    "mid_2_4_32 on affine2 organs — merging further is REFUTED by "
    "MLP_STREAM_COUNT and is not re-run. Accumulator-chain and working-set "
    "discriminators are REFUTED (spans 1.062 and 1.078) and are not re-run. "
    "A shape absent from the table returns UNMEASURED_SHAPE; geo_tpr64_tg128 "
    "is the production incumbent, not a silent lookup default. Token-ms "
    "numbers tagged projection are arithmetic over ORGAN_BANDWIDTH organ "
    "times and are not a resident measurement. Capability is not measured "
    "here. Does not change the production decode path. Does not edit "
    "production shaders."
)

S016 = {
    "section": "S016 §10 / TOKEN_NS_OBJECTIVE primary_objective",
    "above_bandwidth": [
        "complete token_ns",
        "accepted TPS",
        "capability",
        "complete executable cost",
    ],
    "primary_objective": "MINIMIZE VERIFIED STEADY-STATE ACCEPTED TOKEN_NS",
    "not_the_objective": [
        "GPU utilization",
        "a geometry that streams more bytes faster but produces slower tokens",
        "any single fixed TPS number",
    ],
    "subject_to": "capability preserved",
    "how_this_table_ranks": (
        "Unique payload bytes are held fixed across launch-geometry candidates, "
        "so GEMV gpu_ns and GB/s agree. A packing that expanded storage "
        "(pack_6_32 / pack_38) is not a candidate. A faster GB/s that is not "
        "bit-identical is recorded, but is not a qualified accepted-TPS win."
    ),
    "cited_control": {
        "token_ms": TOKEN_MS,
        "raw_tps": TOKEN_TPS,
        "accepted_tps_control": TOKEN_TPS_ACCEPTED_CONTROL,
        "capability_floor_over_43": CAPABILITY_FLOOR_OVER_43,
        "bytes_per_token": BYTES_PER_TOKEN,
        "from": (
            "receipts/future/TOKEN_NS_OBJECTIVE.json, ORGAN_BANDWIDTH.json, "
            "PATH_TO_71.json"
        ),
    },
}


# ---------------------------------------------------------------------------
# Resident hot dimensions. Every row is a GEMV the sealed-3.14 resident
# actually launches. Provenance is named; a synthetic grid is refused.
# ---------------------------------------------------------------------------

def _hot(
    organ: str,
    rows: int,
    cols: int,
    dtype: str,
    *,
    family: str,
    count: int,
    catalog_suffix: str,
    layer: int,
    provenance: Sequence[str],
    codec: str,
) -> dict[str, Any]:
    return {
        "organ": organ,
        "rows": rows,
        "cols": cols,
        "dtype": dtype,
        "family": family,
        "count": count,
        "layer_measured": layer,
        "catalog_tensor": (
            "language_model.lm_head.weight"
            if organ == "lm_head"
            else f"language_model.model.layers.{layer}.{catalog_suffix}"
        ),
        "codec": codec,
        "provenance": list(provenance),
    }


_GEO_RS = "crates/hawking-core/src/model/qwen38_geometry.rs"
_KG = "receipts/future/KERNEL_GEOMETRY.json"
_CENSUS = "receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json (cited by qwen38_geometry.rs)"
_CATALOG = "sealed-3.14 catalog.hq38m20 tensor names"

HOT_DIMENSIONS: tuple[dict[str, Any], ...] = (
    _hot(
        "mlp.gate_proj", 17408, 5120, "affine2_q2",
        family="mlp", count=64, catalog_suffix="mlp.gate_proj.weight", layer=0,
        codec="HGRAVF01",
        provenance=(
            f"{_GEO_RS} QWEN38_INTERMEDIATE=17408 x QWEN38_HIDDEN=5120",
            f"{_KG} organs[mlp.gate_proj] rows=17408 cols=5120 kernel_class=affine2 count=64",
            f"{_CATALOG}: language_model.model.layers.{{0..63}}.mlp.gate_proj.weight",
            _CENSUS,
        ),
    ),
    _hot(
        "mlp.up_proj", 17408, 5120, "affine2_q2",
        family="mlp", count=64, catalog_suffix="mlp.up_proj.weight", layer=0,
        codec="HGRAVF01",
        provenance=(
            f"{_GEO_RS} QWEN38_INTERMEDIATE=17408 x QWEN38_HIDDEN=5120",
            f"{_KG} organs[mlp.up_proj] rows=17408 cols=5120 kernel_class=affine2 count=64",
            f"{_CATALOG}: language_model.model.layers.{{0..63}}.mlp.up_proj.weight",
            _CENSUS,
        ),
    ),
    _hot(
        "mlp.down_proj", 5120, 17408, "affine2_q2",
        family="mlp", count=64, catalog_suffix="mlp.down_proj.weight", layer=0,
        codec="HGRAVF01",
        provenance=(
            f"{_GEO_RS} QWEN38_HIDDEN=5120 x QWEN38_INTERMEDIATE=17408 (specialized_cols 17408)",
            f"{_KG} organs[mlp.down_proj] rows=5120 cols=17408 kernel_class=affine2 count=64",
            f"{_CATALOG}: language_model.model.layers.{{0..63}}.mlp.down_proj.weight",
            "MLP_STREAM_COUNT / NR_NX_GENERIC declared specialized_cols=[5120, 17408]",
        ),
    ),
    _hot(
        "linear_attn.in_proj_qkvz", 16384, 5120, "uniform_q4",
        family="deltanet", count=48, catalog_suffix="linear_attn.in_proj_qkvz.weight", layer=0,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_QKVZ_ROWS=16384 x QWEN38_HIDDEN=5120 "
            "(pack-time fuse of in_proj_qkv=10240 + in_proj_z=6144)",
            f"{_KG} organs[linear_attn.in_proj_qkvz] rows=16384 cols=5120 kernel_class=q4 count=48",
            f"{_CATALOG}: language_model.model.layers.<DeltaNet>.linear_attn.in_proj_qkvz.weight",
        ),
    ),
    _hot(
        "linear_attn.in_proj_ba", 96, 5120, "uniform_q4",
        family="deltanet", count=48, catalog_suffix="linear_attn.in_proj_ba.weight", layer=0,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_BA_ROWS=96 x QWEN38_HIDDEN=5120",
            f"{_KG} organs[linear_attn.in_proj_ba] rows=96 cols=5120; occupancy_starved_organs",
            f"{_CATALOG}: language_model.model.layers.<DeltaNet>.linear_attn.in_proj_ba.weight",
        ),
    ),
    _hot(
        "linear_attn.out_proj", 5120, 6144, "uniform_q4",
        family="deltanet", count=48, catalog_suffix="linear_attn.out_proj.weight", layer=0,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_O_PROJ_ROWS=5120 x QWEN38_O_PROJ_COLS=6144 "
            "(DeltaNet out_proj shares the GQA o_proj extent)",
            f"{_KG} organs[linear_attn.out_proj] rows=5120 cols=6144 kernel_class=q4 count=48",
            f"{_CATALOG}: language_model.model.layers.<DeltaNet>.linear_attn.out_proj.weight",
        ),
    ),
    _hot(
        "self_attn.q_proj", 12288, 5120, "uniform_q4",
        family="gqa", count=16, catalog_suffix="self_attn.q_proj.weight", layer=3,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_Q_PROJ_ROWS=12288 x QWEN38_HIDDEN=5120; "
            "GQA iff (layer+1)%4==0 (first GQA layer=3)",
            f"{_KG} organs[self_attn.q_proj] rows=12288 cols=5120 kernel_class=q4 count=16",
            f"{_CATALOG}: language_model.model.layers.<GQA>.self_attn.q_proj.weight",
        ),
    ),
    _hot(
        "self_attn.k_proj", 1024, 5120, "uniform_q4",
        family="gqa", count=16, catalog_suffix="self_attn.k_proj.weight", layer=3,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_KV_PROJ_ROWS=1024 x QWEN38_HIDDEN=5120",
            f"{_KG} organs[self_attn.k_proj] rows=1024 cols=5120 kernel_class=q4 count=16",
            f"{_CATALOG}: language_model.model.layers.<GQA>.self_attn.k_proj.weight",
        ),
    ),
    _hot(
        "self_attn.v_proj", 1024, 5120, "uniform_q4",
        family="gqa", count=16, catalog_suffix="self_attn.v_proj.weight", layer=3,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_KV_PROJ_ROWS=1024 x QWEN38_HIDDEN=5120",
            f"{_KG} organs[self_attn.v_proj] rows=1024 cols=5120 kernel_class=q4 count=16",
            f"{_CATALOG}: language_model.model.layers.<GQA>.self_attn.v_proj.weight",
        ),
    ),
    _hot(
        "self_attn.o_proj", 5120, 6144, "uniform_q4",
        family="gqa", count=16, catalog_suffix="self_attn.o_proj.weight", layer=3,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_O_PROJ_ROWS=5120 x QWEN38_O_PROJ_COLS=6144",
            f"{_KG} organs[self_attn.o_proj] rows=5120 cols=6144 kernel_class=q4 count=16",
            f"{_CATALOG}: language_model.model.layers.<GQA>.self_attn.o_proj.weight",
        ),
    ),
    _hot(
        "lm_head", 248320, 5120, "uniform_q4",
        family="lm_head", count=1, catalog_suffix="lm_head.weight", layer=0,
        codec="HQ30UQ4",
        provenance=(
            f"{_GEO_RS} QWEN38_VOCAB=248320 x QWEN38_HIDDEN=5120; qwen38_lm_head_name()",
            f"{_KG} organs[lm_head] rows=248320 cols=5120 kernel_class=q4 count=1",
            f"{_CATALOG}: language_model.lm_head.weight",
        ),
    ),
)


def shape_key(rows: int, cols: int, dtype: str, organ: str) -> tuple[int, int, str, str]:
    return (int(rows), int(cols), str(dtype), str(organ))


def key_dict(rows: int, cols: int, dtype: str, organ: str) -> dict[str, Any]:
    return {"rows": int(rows), "cols": int(cols), "dtype": str(dtype), "organ": str(organ)}


def key_str(rows: int, cols: int, dtype: str, organ: str) -> str:
    return f"{organ}|{rows}x{cols}|{dtype}"


HOT_BY_KEY: dict[tuple[int, int, str, str], dict[str, Any]] = {
    shape_key(h["rows"], h["cols"], h["dtype"], h["organ"]): h for h in HOT_DIMENSIONS
}


class UnmeasuredShape(LookupError):
    """A shape the table does not contain. Named, never a silent default."""

    fallback = UNMEASURED_SHAPE


class SilentDefaultRefused(ValueError):
    """Raised when a caller asks consult() to invent a geometry for a miss."""


class RefutedDiscriminatorRerun(ValueError):
    """Raised rather than treat a closed discriminator as an open sweep arm."""


class NotAHotDimension(ValueError):
    """Raised rather than put a synthetic grid point in the table."""


class MissingSweep(ValueError):
    """Raised rather than emit a winner without a runner-up or a measurement."""


class EmptyGpuSample(ValueError):
    """Raised rather than divide by a missing GPU timestamp."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def organ_baseline_ms(family: str) -> float:
    return {
        "mlp": MLP_MS,
        "deltanet": DELTANET_MS,
        "gqa": GQA_MS,
        "lm_head": LM_HEAD_MS,
    }[family]


def assert_hot_dimension(rows: int, cols: int, dtype: str, organ: str) -> dict[str, Any]:
    key = shape_key(rows, cols, dtype, organ)
    hit = HOT_BY_KEY.get(key)
    if hit is None:
        raise NotAHotDimension(
            f"{key_str(rows, cols, dtype, organ)} is not a resident hot dimension; "
            "refusing a synthetic grid point. Provenance lives on HOT_DIMENSIONS."
        )
    return hit


def _ident(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("kernel") or "")


def _family(row: Mapping[str, Any]) -> str:
    return str(row.get("family") or "")


def assert_no_refuted_discriminators(points: Iterable[Mapping[str, Any]]) -> None:
    for row in points:
        ident = _ident(row)
        fam = _family(row)
        if ident in REFUSED_IDS or fam in REFUSED_FAMILIES:
            raise RefutedDiscriminatorRerun(
                f"refusing to re-run closed discriminator {ident or fam}: "
                "accumulator-chain span 1.062 and working-set span 1.078 closed "
                "dependency and register pressure; pack_6_32/pack_38 merging "
                "further than mid_2_4_32 is measured to hurt"
            )


def _geo_tuple(row: Mapping[str, Any]) -> tuple[int, int, int]:
    tg = int(row["threads_per_threadgroup"])
    rpt = int(row["rows_per_threadgroup"])
    tpr = int(row.get("threads_per_row") or (tg // max(rpt, 1)))
    return tg, rpt, tpr


def pick_winner_and_runner_up(
    points: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(points) < 2:
        raise MissingSweep(
            f"a table row with no runner-up cannot show whether the win was marginal "
            f"({len(points)} launch-geometry point(s))"
        )
    ranked = sorted(
        points,
        key=lambda p: (
            -float(p["effective_gb_s"]),
            int(p["gpu_ns_median"]),
            _ident(p),
        ),
    )
    winner, runner = ranked[0], ranked[1]
    if float(winner["effective_gb_s"]) <= 0 or int(winner["gpu_ns_median"]) <= 0:
        raise EmptyGpuSample("winner has no positive GPU sample")
    return dict(winner), dict(runner)


def packing_winner(
    points: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not points:
        return None, None
    if len(points) == 1:
        raise MissingSweep(
            "packing was measured on one arm only; a packing row needs a runner-up"
        )
    ranked = sorted(
        points,
        key=lambda p: (-float(p["effective_gb_s"]), _ident(p)),
    )
    return dict(ranked[0]), dict(ranked[1])


def _gb_s_span(points: Sequence[Mapping[str, Any]]) -> float | None:
    vals = [float(p["effective_gb_s"]) for p in points if float(p.get("effective_gb_s") or 0) > 0]
    if len(vals) < 2:
        return None
    lo = min(vals)
    if lo <= 0:
        return None
    return max(vals) / lo


def project_token(
    *,
    family: str,
    production_gb_s: float,
    winner_gb_s: float,
) -> dict[str, Any]:
    baseline = organ_baseline_ms(family)
    if production_gb_s <= 0 or winner_gb_s <= 0:
        raise EmptyGpuSample("projection requires positive GB/s")
    # Unique payload held fixed: organ_ms scales with 1/GB/s. This assumes the
    # organ's GEMVs of this class all move with this shape's ratio — a
    # PROJECTION, not a resident measurement.
    organ_ms = baseline * (production_gb_s / winner_gb_s)
    saved = baseline - organ_ms
    token_ms = TOKEN_MS - saved
    tps = 1000.0 / token_ms if token_ms > 0 else None
    return {
        "kind": "projection",
        "label": (
            "arithmetic over ORGAN_BANDWIDTH organ ms scaled by this-GEMV "
            "GB/s ratio; not a resident complete-token measurement"
        ),
        "baseline_organ_ms": baseline,
        "baseline_token_ms": TOKEN_MS,
        "baseline_raw_tps": TOKEN_TPS,
        "organ_ms_projected": round(organ_ms, 3),
        "token_ms_projected": round(token_ms, 3),
        "raw_tps_projected": None if tps is None else round(tps, 2),
        "delta_organ_ms": round(-saved, 3),
        "note": (
            "S016 §10 ranks complete token_ns / accepted TPS / capability / "
            "complete executable cost above bandwidth. Unique payload bytes "
            "are held fixed, so this projection and GB/s agree on the winner. "
            "Capability is not measured. A non-bit-identical geometry is not "
            "a qualified accepted-TPS win."
        ),
    }


def _point_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    weight_bytes = int(raw["weight_bytes"])
    gpu_ns = int(raw["gpu_ns_median"])
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    tg = int(raw["threads_per_threadgroup"])
    rpt = int(raw["rows_per_threadgroup"])
    tpr = int(raw.get("threads_per_row") or (tg // max(rpt, 1)))
    out = {
        "id": str(raw.get("id") or ""),
        "kernel": raw.get("kernel"),
        "family": str(raw.get("family") or "launch"),
        "threads_per_threadgroup": tg,
        "rows_per_threadgroup": rpt,
        "threads_per_row": tpr,
        "stream_packing": raw.get("stream_packing"),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "gpu_us_median": round(gpu_ns / 1e3, 3),
        "effective_gb_s": round(gb_s, 1),
        "bit_identical_vs_production_geo": raw.get("bit_identical_vs_production_geo"),
        "byte_compare": raw.get("byte_compare"),
        "occupancy": raw.get("occupancy"),
        "note": raw.get("note"),
    }
    return out


def _shape_row_from_raw(raw_shape: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(raw_shape["rows"])
    cols = int(raw_shape["cols"])
    dtype = str(raw_shape["dtype"])
    organ = str(raw_shape["organ"])
    hot = assert_hot_dimension(rows, cols, dtype, organ)

    launch = [_point_view(p) for p in (raw_shape.get("launch") or [])]
    packing = [_point_view(p) for p in (raw_shape.get("packing") or [])]
    assert_no_refuted_discriminators(list(raw_shape.get("launch") or []) + list(raw_shape.get("packing") or []))
    assert_no_refuted_discriminators(launch + packing)

    if not launch:
        raise MissingSweep(f"{organ}: no launch-geometry points")
    winner, runner = pick_winner_and_runner_up(launch)
    pack_w, pack_r = packing_winner(packing) if packing else (None, None)

    if dtype.startswith("affine2") and packing:
        allowed = {PRODUCTION_PACKING_AFFINE, PACKING_MID}
        got = {str(p.get("stream_packing") or p.get("id")) for p in packing}
        if not got <= allowed and not got <= {PRODUCTION_PACKING_AFFINE, PACKING_MID, "2+2+2+32", "2+4+32"}:
            extra = got - allowed
            if extra & {"pack_6_32", "pack_38"}:
                raise RefutedDiscriminatorRerun(
                    f"{organ}: packing arms {sorted(extra)} re-run a merge past mid_2_4_32"
                )

    production = next((p for p in launch if p["id"] == PRODUCTION_GEO_ID), None)
    production_gb = float(production["effective_gb_s"]) if production else float(winner["effective_gb_s"])
    winner_gb = float(winner["effective_gb_s"])
    projection = project_token(
        family=str(hot["family"]),
        production_gb_s=production_gb,
        winner_gb_s=winner_gb,
    )

    packing_id = None
    if pack_w is not None:
        packing_id = pack_w.get("stream_packing") or pack_w.get("id")
    elif not dtype.startswith("affine2"):
        packing_id = "q4_codes_scale_x"

    span = _gb_s_span(launch)
    return {
        "organ": organ,
        "rows": rows,
        "cols": cols,
        "dtype": dtype,
        "family": hot["family"],
        "codec": hot["codec"],
        "count": hot["count"],
        "catalog_tensor": hot["catalog_tensor"],
        "layer_measured": hot["layer_measured"],
        "provenance": list(hot["provenance"]),
        "weight_bytes": int(raw_shape.get("weight_bytes") or winner["weight_bytes"]),
        "winner": {
            "id": winner["id"],
            "threads_per_threadgroup": winner["threads_per_threadgroup"],
            "rows_per_threadgroup": winner["rows_per_threadgroup"],
            "threads_per_row": winner["threads_per_row"],
            "tile": winner["rows_per_threadgroup"],
            "stream_packing": packing_id,
            "effective_gb_s": winner["effective_gb_s"],
            "gpu_ns_median": winner["gpu_ns_median"],
            "kernel": winner.get("kernel"),
            "bit_identical_vs_production_geo": winner.get("bit_identical_vs_production_geo"),
        },
        "runner_up": {
            "id": runner["id"],
            "threads_per_threadgroup": runner["threads_per_threadgroup"],
            "rows_per_threadgroup": runner["rows_per_threadgroup"],
            "threads_per_row": runner["threads_per_row"],
            "tile": runner["rows_per_threadgroup"],
            "effective_gb_s": runner["effective_gb_s"],
            "gpu_ns_median": runner["gpu_ns_median"],
            "kernel": runner.get("kernel"),
            "gb_s_ratio_winner_over_runner": round(
                float(winner["effective_gb_s"]) / float(runner["effective_gb_s"]), 4
            )
            if float(runner["effective_gb_s"]) > 0
            else None,
        },
        "launch": launch,
        "packing": packing,
        "packing_winner": pack_w,
        "packing_runner_up": pack_r,
        "production_geo": production,
        "launch_gb_s_span": None if span is None else round(span, 4),
        "s016_projection": projection,
        "specialized_cols_member": cols in SPECIALIZED_COLS,
    }


def table_is_flat(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signatures: list[tuple[Any, ...]] = []
    for row in rows:
        w = row["winner"]
        signatures.append(
            (
                int(w["threads_per_threadgroup"]),
                int(w["rows_per_threadgroup"]),
                int(w["threads_per_row"]),
            )
        )
    unique = sorted(set(signatures))
    flat = len(unique) <= 1
    return {
        "flat": flat,
        "n_shapes": len(rows),
        "n_distinct_launch_winners": len(unique),
        "distinct_launch_winners": [
            {
                "threads_per_threadgroup": t[0],
                "rows_per_threadgroup": t[1],
                "threads_per_row": t[2],
            }
            for t in unique
        ],
        "reading": (
            "one geometry wins at every real hot dimension: geometry IS a "
            "constant for this resident and the obligation is answered in the negative"
            if flat
            else "winning geometry depends on the hot dimension; the compiler must consult the table"
        ),
    }


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    shapes_raw = raw.get("shapes")
    if not isinstance(shapes_raw, list) or not shapes_raw:
        raise MissingSweep("refusing a table: raw.shapes is empty")
    assert_no_refuted_discriminators(_iter_all_points(raw))
    rows = [_shape_row_from_raw(s) for s in shapes_raw]
    organs = {r["organ"] for r in rows}
    expected = {h["organ"] for h in HOT_DIMENSIONS}
    missing = sorted(expected - organs)
    extra = sorted(organs - expected)
    if extra:
        raise NotAHotDimension(f"raw contains non-hot organs {extra}")
    flat = table_is_flat(rows)
    return {
        "layer_mlp_dn": int(raw.get("layer_mlp_dn", 0)),
        "layer_gqa": int(raw.get("layer_gqa", 3)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "concurrent_load": raw.get("concurrent_load") or {},
        "concurrent_load_end": raw.get("concurrent_load_end") or {},
        "measured_at": raw.get("measured_at"),
        "loadavg": (raw.get("concurrent_load") or {}).get("loadavg"),
        "absolute_gb_s_are_measured_under_load": True,
        "shapes_missing_from_this_run": missing,
        "shapes": rows,
        "flat": flat,
        "gpu_lane_lock_held": bool(raw.get("gpu_lane_lock_held", True)),
        "does_not_edit_production_shaders": True,
        "refused_discriminators": REFUTED_DISCRIMINATORS,
        "specialized_cols": list(SPECIALIZED_COLS),
    }


def _iter_all_points(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for shape in raw.get("shapes") or []:
        out.extend(shape.get("launch") or [])
        out.extend(shape.get("packing") or [])
        for extra_key in ("ilp", "register_pressure", "working_set", "accumulator_chain"):
            extra = shape.get(extra_key) or raw.get(extra_key)
            if extra:
                out.extend(extra if isinstance(extra, list) else [extra])
    for extra_key in ("ilp", "register_pressure", "working_set", "threadgroup"):
        extra = raw.get(extra_key)
        if isinstance(extra, list):
            out.extend(extra)
    return out


def _finding(measurement: Mapping[str, Any]) -> str:
    flat = measurement["flat"]
    rows = measurement["shapes"]
    parts = [
        "GEOMETRY IS A TABLE, NOT A CONSTANT."
        if not flat["flat"]
        else (
            "The table is FLAT: one launch geometry wins at every measured hot "
            "dimension, so geometry IS a constant for this resident and the "
            "obligation is answered in the negative."
        ),
    ]
    parts.append(flat["reading"] + ".")
    for row in rows:
        w, r = row["winner"], row["runner_up"]
        pack = w.get("stream_packing")
        parts.append(
            f"{row['organ']} {row['rows']}x{row['cols']} {row['dtype']}: "
            f"winner {w['id']} tg={w['threads_per_threadgroup']} "
            f"rows_per_tg={w['rows_per_threadgroup']} tpr={w['threads_per_row']} "
            f"{w['effective_gb_s']} GB/s; runner-up {r['id']} {r['effective_gb_s']} GB/s"
            f"{'' if pack is None else f'; packing {pack}'}."
        )
    missing = measurement.get("shapes_missing_from_this_run") or []
    if missing:
        parts.append(
            "Unmeasured hot dimensions remain UNMEASURED_SHAPE (not a silent default): "
            + ", ".join(missing)
            + "."
        )
    parts.append(
        "Consult via geometry_table.consult(rows, cols, dtype, organ); "
        "an absent shape returns UNMEASURED_SHAPE, not geo_tpr64_tg128."
    )
    return " ".join(parts)


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    finding = _finding(measurement)
    table = {
        key_str(r["rows"], r["cols"], r["dtype"], r["organ"]): {
            "key": key_dict(r["rows"], r["cols"], r["dtype"], r["organ"]),
            "winner": r["winner"],
            "runner_up": r["runner_up"],
            "provenance": r["provenance"],
            "family": r["family"],
            "codec": r["codec"],
            "count": r["count"],
            "catalog_tensor": r["catalog_tensor"],
            "weight_bytes": r["weight_bytes"],
            "launch_gb_s_span": r["launch_gb_s_span"],
            "s016_projection": r["s016_projection"],
            "packing_winner": None
            if r.get("packing_winner") is None
            else {
                "id": r["packing_winner"].get("id"),
                "stream_packing": r["packing_winner"].get("stream_packing")
                or r["packing_winner"].get("id"),
                "effective_gb_s": r["packing_winner"].get("effective_gb_s"),
                "gpu_ns_median": r["packing_winner"].get("gpu_ns_median"),
            },
            "packing_runner_up": None
            if r.get("packing_runner_up") is None
            else {
                "id": r["packing_runner_up"].get("id"),
                "stream_packing": r["packing_runner_up"].get("stream_packing")
                or r["packing_runner_up"].get("id"),
                "effective_gb_s": r["packing_runner_up"].get("effective_gb_s"),
                "gpu_ns_median": r["packing_runner_up"].get("gpu_ns_median"),
            },
        }
        for r in measurement["shapes"]
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/geometry_table_sweep.rs; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime); "
            "sealed-3.14 hot GEMVs, production arithmetic on launch geometry, "
            "stripped packing arms 2+2+2+32 vs mid_2_4_32 only"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "obligation": (
            "GEOMETRY IS A TABLE, NOT A CONSTANT. The sweep produces a shape -> "
            "winning-geometry table THE COMPILER CONSULTS for the resident's REAL "
            "HOT DIMENSIONS, rather than one hard-coded winner."
        ),
        "s016": S016,
        "hot_dimensions": list(HOT_DIMENSIONS),
        "specialized_cols": list(SPECIALIZED_COLS),
        "production_incumbent": {
            "id": PRODUCTION_GEO_ID,
            "threads_per_threadgroup": PRODUCTION_TG,
            "rows_per_threadgroup": PRODUCTION_ROWS_PER_TG,
            "threads_per_row": PRODUCTION_TPR,
            "name": "geo_tpr64_tg128",
            "note": (
                "production today is geo_tpr64 with 128 threads. This is the "
                "incumbent, not a lookup result for an unmeasured shape."
            ),
        },
        "refuted_discriminators": REFUTED_DISCRIMINATORS,
        "does_not_edit_production_shaders": True,
        "absolute_gb_s_are_measured_under_load": True,
        "gpu_cores_cited": GPU_CORES_CITED,
        "gpu_cores_provenance": GPU_CORES_PROVENANCE,
        "timing": measurement.get("timing"),
        "warmup": measurement.get("warmup"),
        "reps": measurement.get("reps"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "concurrent_load": measurement.get("concurrent_load"),
        "concurrent_load_end": measurement.get("concurrent_load_end"),
        "shapes": measurement["shapes"],
        "table": table,
        "flat": measurement["flat"],
        "geometry_is_a_constant_for_this_resident": bool(measurement["flat"]["flat"]),
        "shapes_missing_from_this_run": measurement.get("shapes_missing_from_this_run") or [],
        "fallback_for_absent_shape": UNMEASURED_SHAPE,
        "planner_entry": "tools.future.geometry_table.consult",
        "finding": finding,
        "verdict": (
            "GEOMETRY_IS_A_CONSTANT_FOR_THIS_RESIDENT"
            if measurement["flat"]["flat"]
            else "GEOMETRY_IS_A_TABLE"
        ),
    }


def record(measurement: Mapping[str, Any], *, path: Path | None = None) -> Path:
    doc = build(measurement)
    prov = measurement_provenance(
        lock_held=bool(measurement.get("gpu_lane_lock_held", True)),
        loadavg=measurement.get("loadavg"),
        lane="g025-geometry-table",
        measured_at=measurement.get("measured_at"),
        retrofit=measurement.get("measured_at") is None,
    )
    return write_measured_receipt(
        path or RECEIPT,
        doc,
        RECORDED_BY,
        provenance=prov,
    )


def load_table(path: Path | None = None) -> dict[str, Any]:
    target = path or RECEIPT
    doc = json.loads(Path(target).read_text())
    if not isinstance(doc, dict) or "table" not in doc:
        raise MissingSweep(f"{target} is not a geometry table")
    return doc


def consult(
    rows: int,
    cols: int,
    dtype: str,
    organ: str,
    *,
    table: Mapping[str, Any] | None = None,
    default: Any = None,
) -> dict[str, Any]:
    """Lookup the planner calls.

    HIT: the shape is in the table; winner and runner-up are returned.
    UNMEASURED_SHAPE: the shape is absent. geo_tpr64_tg128 is NOT returned.
    Passing `default=` is refused — that is the silent default this API exists
    to make impossible.
    """
    if default is not None:
        raise SilentDefaultRefused(
            "refusing a silent default; an unmeasured shape must be visibly "
            f"unmeasured ({UNMEASURED_SHAPE}), not replaced with {default!r}"
        )
    key = key_str(rows, cols, dtype, organ)
    kd = key_dict(rows, cols, dtype, organ)
    doc = table if table is not None else (load_table() if RECEIPT.is_file() else {"table": {}})
    store = doc.get("table") or {}
    hit = store.get(key)
    if hit is None:
        return {
            "status": STATUS_UNMEASURED,
            "fallback": UNMEASURED_SHAPE,
            "key": kd,
            "winner": None,
            "runner_up": None,
            "geometry": None,
            "why": (
                f"{key} is absent from the geometry table; refusing a silent "
                "default to geo_tpr64_tg128. The production incumbent is cited "
                "on the receipt, not selected for an unmeasured shape."
            ),
            "production_incumbent_cited_not_selected": "geo_tpr64_tg128",
        }
    winner = hit.get("winner")
    return {
        "status": STATUS_HIT,
        "fallback": None,
        "key": kd,
        "winner": winner,
        "runner_up": hit.get("runner_up"),
        "geometry": winner,
        "provenance": hit.get("provenance"),
        "s016_projection": hit.get("s016_projection"),
        "why": f"{key} is a measured hot dimension; using the table winner",
    }


def planner_geometry(
    rows: int,
    cols: int,
    dtype: str,
    organ: str,
    *,
    table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Alias the planner actually calls. Same contract as consult()."""
    return consult(rows, cols, dtype, organ, table=table)


def example_binaries() -> list[Path]:
    names = ("geometry_table_sweep",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "target",
            REPO / "workspace" / "ops" / "build" / "rust",
        ]
    )
    out: list[Path] = []
    for root in roots:
        for profile in ("release-fast", "release"):
            for name in names:
                p = root / profile / "examples" / name
                if p.is_file():
                    out.append(p)
    return out


def run_example(
    artifact_root: Path,
    *,
    warmup: int = 3,
    reps: int = 7,
    out: Path | None = None,
    binary: Path | None = None,
    use_lock: bool = True,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "geometry_table_sweep binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example geometry_table_sweep`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--out",
        str(out),
    ]
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd = ["bash", str(lock), "g025-geometry-table", *inner] if use_lock and lock.is_file() else inner
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} exited {proc.returncode}\nstdout:\n{proc.stdout[-8000:]}\n"
            f"stderr:\n{proc.stderr[-8000:]}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path.home() / "noetic" / "NOETIC_PARENT_A",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--consult", nargs=4, metavar=("ROWS", "COLS", "DTYPE", "ORGAN"))
    args = parser.parse_args(argv)

    if args.consult:
        rows, cols, dtype, organ = args.consult
        decision = consult(int(rows), int(cols), dtype, organ)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["status"] == STATUS_HIT else 2

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
            warmup=args.warmup,
            reps=args.reps,
            out=Path(args.raw_path) if args.raw_path else args.out or RAW_DEFAULT,
        )
    elif args.raw_path:
        raw = load_raw(Path(args.raw_path))

    if raw is None:
        parser.error("need --measure, --from RAW, or --consult")

    measured = measurement_from_raw(raw)
    if args.record:
        path = record(measured, path=args.out)
        print(f"wrote {path}")
    else:
        print(json.dumps(build(measured), indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
