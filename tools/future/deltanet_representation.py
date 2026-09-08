"""DELTANET REPRESENTATION — 2.96 GB, 30% of the token, one fused Q4 family.

Sealed-3.14 DeltaNet is 2,961,659,904 active weight bytes per token, second
only to the MLP, of which linear_qkvz is 2,139,096,960 (21.65% of the token
and larger than GQA + LM head combined). This module re-derives that total
from the HQ38M20 catalog tensor by tensor, splits every byte into codes vs
scale vs header, splits linear_qkvz into q/k/v/z row ranges, and asks what
independent information the recurrent state and those projections need.

    python3 tools/future/deltanet_representation.py --build
    python3 -m pytest tools/future/test_deltanet_representation.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future._common import REPO, git, write_receipt
from tools.future.ebpw_categories import PRODUCTION, judge_dense_rematerialization
from tools.future.mlp_auxiliary_information import parse_catalog_records
from tools.future.mlp_byte_census import (
    CATALOG_NAME,
    CatalogAbsent,
    CensusRefuse,
    classify_tensor,
    load_geometry,
    load_sealed,
    resolve_artifact_root,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "DELTANET_REPRESENTATION.json"
SCHEMA = "hawking.future.deltanet_representation.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_representation.py"

# Reconciliation TARGET. The split is re-measured; this number is the
# refusal bar, copied from the organ census (attention.linear_* four organs).
DELTANET_ACTIVE_TARGET = 2_961_659_904
QKVZ_ACTIVE_TARGET = 2_139_096_960
OUT_ACTIVE_TARGET = 802_162_560
BA_ACTIVE_TARGET = 12_535_680
CONV_ACTIVE_TARGET = 7_864_704
TOKEN_ACTIVE_TARGET = 9_878_901_136

HQ30UQ4_MAGIC = b"HQ30UQ4\0"
HQ30UQ4_HEADER_FIXED = 32
HQ30UQ4_VERSION = 1
INCUMBENT_GROUP = 64
Q4_CODE_BITS = 4
F16_BYTES = 2
F32_BYTES = 4
F32V2_HEADER = 8

DN_ORGANS: tuple[str, ...] = (
    "attention.linear_qkvz",
    "attention.linear_ba",
    "attention.linear_out",
    "attention.linear_conv1d",
)
Q4_ORGANS: tuple[str, ...] = (
    "attention.linear_qkvz",
    "attention.linear_ba",
    "attention.linear_out",
)
F32_ORGANS: tuple[str, ...] = ("attention.linear_conv1d",)
SUBBLOCKS: tuple[str, ...] = ("q", "k", "v", "z")

# Adjacent catalog tensors that ARE DeltaNet parameters and ARE NOT in the
# 2.96 GB organ total (they sit in state.* / norms.*).
ADJACENT_ORGANS: tuple[str, ...] = (
    "state.A_log",
    "state.dt_bias",
    "norms.linear_attn",
)

NOETIC_RELS = (
    "receipts/future/evidence/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
)
QN_REL = "tools/headless/negative_science.py"
BA_DELTA_REL = "receipts/future/BA_DELTA_AB.json"
DISPATCH_MOTIFS_REL = "receipts/future/DISPATCH_MOTIFS.json"

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FALSIFIED = "ALREADY_FALSIFIED"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
OPEN = "OPEN"
EXISTING_LEVER = "EXISTING_LEVER"

SAMPLE_LAYERS: tuple[int, ...] = (0, 5, 21, 32, 42, 61)
RECON_GROUPS = 8192
RNG_SEED = 38

REQUIRED_CANDIDATE_IDS: tuple[str, ...] = (
    "heterogeneous_qkvz_bits",
    "lower_bit_uniform_qkvz",
    "lower_bit_out_proj",
    "gravity_family_on_dn_weights",
    "larger_q4_group",
    "shared_transforms_across_layers",
    "generated_coefficients",
    "factorized_qkvz",
    "conv1d_lower_bit",
    "lower_bit_recurrent_state",
    "structured_transition_state",
    "recurrent_state_replacement",
    "share_or_merge_state_across_depth",
    "direct_state_machine",
    "fused_update_consume",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "The 2,961,659,904 figure is catalog stored bytes of the four DeltaNet "
    "weight organs a decode token reads (linear_qkvz, linear_out, linear_ba, "
    "linear_conv1d). Codes/scale/header splits are HQ30UQ4 and f32v2 headers. "
    "q/k/v/z byte splits are row ranges of the fused QKVZ layout "
    "(per key-head Q128,K128,V384,Z384). Recurrent state and conv_state are "
    "geometry-derived f32 traffic, not HQ38M20, and are not added to the "
    "2.96 GB bar. Weight-space requant rel-fro is incumbent-nibble drop, "
    "not a refit and not capability. A candidate that rematerializes dense W "
    "does not eliminate active bytes."
)


class DeltaNetRefuse(ValueError):
    """The DeltaNet census refused rather than guessing."""


class UnreconciledDeltaNet(DeltaNetRefuse):
    """Per-tensor DeltaNet bytes do not equal the recorded 2.96 GB."""

    def __init__(self, got: int, want: int = DELTANET_ACTIVE_TARGET, *, detail: str = "") -> None:
        self.got = int(got)
        self.want = int(want)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: DeltaNet per-tensor active-byte sum {got} != "
            f"recorded organ total {want}{extra}"
        )


class CatalogLayoutRefuse(DeltaNetRefuse):
    """A payload disagreed with its header or with the geometry."""


# ---------------------------------------------------------------------------
# Geometry. Config + catalog shapes must agree or we refuse.
# ---------------------------------------------------------------------------


def _posint(mapping: Mapping[str, Any], key: str) -> int:
    raw = mapping.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise DeltaNetRefuse(f"specimen config missing positive int {key}")
    return raw


def load_dn_geometry(root: Path | None = None) -> dict[str, Any]:
    """DeltaNet layout from the specimen config, not from a remembered constant."""
    artifact = root if root is not None else resolve_artifact_root()
    geo = load_geometry(artifact)
    cfg_path = artifact / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogAbsent(f"config.json unreadable under {artifact}: {exc}") from exc
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if not isinstance(text, dict):
        raise DeltaNetRefuse("specimen text_config is not an object")
    hidden = int(geo["hidden_size"])
    key_heads = _posint(text, "linear_num_key_heads")
    value_heads = _posint(text, "linear_num_value_heads")
    key_dim = _posint(text, "linear_key_head_dim")
    value_dim = _posint(text, "linear_value_head_dim")
    conv_kernel = _posint(text, "linear_conv_kernel_dim")
    if value_heads % key_heads:
        raise DeltaNetRefuse(
            f"value_heads {value_heads} is not a multiple of key_heads {key_heads}"
        )
    values_per_key = value_heads // key_heads
    q_rows = key_heads * key_dim
    k_rows = key_heads * key_dim
    v_rows = value_heads * value_dim
    z_rows = value_heads * value_dim
    qkvz_rows = q_rows + k_rows + v_rows + z_rows
    ba_rows = key_heads * values_per_key * 2
    conv_channels = q_rows + k_rows + v_rows  # z does not enter conv
    rec_elems = value_heads * key_dim * value_dim
    conv_state_elems = conv_channels * (conv_kernel - 1)
    n_dn = 0
    kinds = geo.get("layer_types") or []
    if isinstance(kinds, list):
        n_dn = sum(1 for k in kinds if k == "linear_attention")
    return {
        "hidden_size": hidden,
        "num_hidden_layers": int(geo["num_hidden_layers"]),
        "key_heads": key_heads,
        "value_heads": value_heads,
        "key_head_dim": key_dim,
        "value_head_dim": value_dim,
        "values_per_key": values_per_key,
        "conv_kernel": conv_kernel,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "v_rows": v_rows,
        "z_rows": z_rows,
        "qkvz_rows": qkvz_rows,
        "qkvz_cols": hidden,
        "ba_rows": ba_rows,
        "ba_cols": hidden,
        "out_rows": hidden,
        "out_cols": v_rows,
        "conv_channels": conv_channels,
        "conv_kernel_size": conv_kernel,
        "recurrent_state_elements_per_layer": rec_elems,
        "conv_state_elements_per_layer": conv_state_elems,
        "n_deltanet_layers": n_dn,
        "fused_rows_per_key": key_dim * 2 + values_per_key * value_dim * 2,
        "qkvz_layout": "per_key_head_Q{kd}_K{kd}_V{vd}_Z{vd}".format(
            kd=key_dim, vd=values_per_key * value_dim
        ),
    }


def fused_qkvz_row_indices(geo: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Row indices of q, k, v, z inside the fused [qkvz_rows, hidden] tensor.

    Pack-time fuse (qwen38_geometry::fuse_in_proj_qkvz) interleaves per key
    head as Q, K, V, Z. These four lists are disjoint and cover 0..qkvz_rows.
    """
    key_heads = int(geo["key_heads"])
    key_dim = int(geo["key_head_dim"])
    value_rows = int(geo["values_per_key"]) * int(geo["value_head_dim"])
    rows_per_key = int(geo["fused_rows_per_key"])
    q: list[int] = []
    k: list[int] = []
    v: list[int] = []
    z: list[int] = []
    for key_head in range(key_heads):
        base = key_head * rows_per_key
        q.extend(range(base, base + key_dim))
        k.extend(range(base + key_dim, base + 2 * key_dim))
        v.extend(range(base + 2 * key_dim, base + 2 * key_dim + value_rows))
        z.extend(range(base + 2 * key_dim + value_rows, base + rows_per_key))
    out = {
        "q": np.asarray(q, dtype=np.int64),
        "k": np.asarray(k, dtype=np.int64),
        "v": np.asarray(v, dtype=np.int64),
        "z": np.asarray(z, dtype=np.int64),
    }
    union = np.concatenate(list(out.values()))
    if int(np.unique(union).size) != int(geo["qkvz_rows"]):
        raise CatalogLayoutRefuse("fused q/k/v/z row indices do not cover qkvz_rows")
    if union.size != int(geo["qkvz_rows"]):
        raise CatalogLayoutRefuse("fused q/k/v/z row indices overlap")
    for name, want in (
        ("q", geo["q_rows"]),
        ("k", geo["k_rows"]),
        ("v", geo["v_rows"]),
        ("z", geo["z_rows"]),
    ):
        if out[name].size != int(want):
            raise CatalogLayoutRefuse(f"{name} rows {out[name].size} != geometry {want}")
    return out


def q4_parts(rows: int, cols: int, *, group: int = INCUMBENT_GROUP, rank: int = 2) -> dict[str, int]:
    """Exact HQ30UQ4 byte parts for a row-major matrix. No padding."""
    if rows <= 0 or cols <= 0 or group <= 0:
        raise DeltaNetRefuse(f"illegal Q4 shape {rows}x{cols} G={group}")
    if cols % group:
        raise DeltaNetRefuse(f"cols {cols} is not a multiple of group {group}")
    elements = rows * cols
    groups = elements // group
    header = HQ30UQ4_HEADER_FIXED + rank * 4
    scale = groups * F16_BYTES
    code = groups * (group // 2)
    return {
        "elements": elements,
        "groups": groups,
        "header_bytes": header,
        "scale_bytes": scale,
        "bias_bytes": 0,
        "code_bytes": code,
        "stored_bytes": header + scale + code,
        "groups_per_row": cols // group,
        "bytes_per_row": (cols // group) * (F16_BYTES + group // 2),
    }


def f32v2_parts(n_elements: int) -> dict[str, int]:
    return {
        "elements": int(n_elements),
        "header_bytes": F32V2_HEADER,
        "scale_bytes": 0,
        "bias_bytes": 0,
        "code_bytes": int(n_elements) * F32_BYTES,  # the values are the payload
        "stored_bytes": F32V2_HEADER + int(n_elements) * F32_BYTES,
    }


def qkvz_subblock_parts(geo: Mapping[str, Any], *, n_layers: int | None = None) -> dict[str, dict[str, int]]:
    """Codes+scales of each fused sub-block. The 40-byte header is not split."""
    n = int(n_layers if n_layers is not None else geo["n_deltanet_layers"])
    gpr = int(geo["qkvz_cols"]) // INCUMBENT_GROUP
    row_bytes = gpr * (F16_BYTES + INCUMBENT_GROUP // 2)
    out: dict[str, dict[str, int]] = {}
    for name, rows in (
        ("q", int(geo["q_rows"])),
        ("k", int(geo["k_rows"])),
        ("v", int(geo["v_rows"])),
        ("z", int(geo["z_rows"])),
    ):
        groups = rows * gpr
        scale = groups * F16_BYTES
        code = groups * (INCUMBENT_GROUP // 2)
        out[name] = {
            "rows": rows,
            "groups": groups * n,
            "scale_bytes": scale * n,
            "code_bytes": code * n,
            "payload_bytes": (scale + code) * n,
            "bytes_per_layer": rows * row_bytes,
            "n_layers": n,
            "row_share_of_qkvz": rows / int(geo["qkvz_rows"]),
        }
    header = (HQ30UQ4_HEADER_FIXED + 8) * n  # rank-2
    payload = sum(p["payload_bytes"] for p in out.values())
    if payload + header != QKVZ_ACTIVE_TARGET:
        raise UnreconciledDeltaNet(
            payload + header,
            QKVZ_ACTIVE_TARGET,
            detail="q+k+v+z payload + headers vs linear_qkvz organ total",
        )
    out["header"] = {
        "rows": 0,
        "groups": 0,
        "scale_bytes": 0,
        "code_bytes": 0,
        "payload_bytes": 0,
        "header_bytes": header,
        "n_layers": n,
    }
    return out


# ---------------------------------------------------------------------------
# Parsers.
# ---------------------------------------------------------------------------


def parse_hq30uq4_header(blob: bytes, *, name: str = "") -> dict[str, Any]:
    if len(blob) < HQ30UQ4_HEADER_FIXED:
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 truncated header")
    if blob[:8] != HQ30UQ4_MAGIC:
        raise CatalogLayoutRefuse(f"{name} magic {blob[:8]!r} is not HQ30UQ4")
    version, group_size = struct.unpack_from("<II", blob, 8)
    rank, reserved = struct.unpack_from("<HH", blob, 16)
    elements = struct.unpack_from("<Q", blob, 20)[0]
    reserved_tail = struct.unpack_from("<I", blob, 28)[0]
    if version != HQ30UQ4_VERSION:
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 version {version} != {HQ30UQ4_VERSION}")
    if group_size not in (64, 128):
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 group_size {group_size} not 64 or 128")
    if reserved != 0 or reserved_tail != 0:
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 reserved fields must be zero")
    if rank <= 0:
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 rank must be positive")
    dim_off = HQ30UQ4_HEADER_FIXED
    dim_bytes = rank * 4
    if len(blob) < dim_off + dim_bytes:
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 truncated dimensions")
    shape = list(struct.unpack_from("<" + "I" * rank, blob, dim_off))
    if any(d <= 0 for d in shape):
        raise CatalogLayoutRefuse(f"{name} HQ30UQ4 dimensions must be positive")
    derived = 1
    for d in shape:
        derived *= d
    if derived != int(elements):
        raise CatalogLayoutRefuse(
            f"{name} HQ30UQ4 elements {elements} != shape product {derived}"
        )
    groups = (int(elements) + group_size - 1) // group_size
    scale_bytes = groups * F16_BYTES
    code_bytes = groups * (group_size // 2)
    header_bytes = dim_off + dim_bytes
    payload_off = header_bytes
    need = payload_off + scale_bytes + code_bytes
    if need != len(blob):
        raise CatalogLayoutRefuse(
            f"{name} HQ30UQ4 file {len(blob)} != header+scale+code {need}"
        )
    return {
        "representation": "hq30uq4_uniform_q4",
        "codec_name": "HQ30UQ4",
        "bits": Q4_CODE_BITS,
        "group_size": int(group_size),
        "rank": int(rank),
        "shape": shape,
        "elements": int(elements),
        "groups": int(groups),
        "header_bytes": int(header_bytes),
        "scale_bytes": int(scale_bytes),
        "bias_bytes": 0,
        "code_bytes": int(code_bytes),
        "payload_off": int(payload_off),
        "reconstruction": "w = float(nibble - 8) * f16_scale; no bias",
        "has_bias": False,
    }


def parse_f32v2_header(blob: bytes, *, name: str = "", shape: Sequence[int] | None = None) -> dict[str, Any]:
    if len(blob) < F32V2_HEADER:
        raise CatalogLayoutRefuse(f"{name} f32v2 truncated")
    n = struct.unpack_from("<Q", blob, 0)[0]
    need = F32V2_HEADER + int(n) * F32_BYTES
    if need != len(blob):
        raise CatalogLayoutRefuse(f"{name} f32v2 file {len(blob)} != 8+4*n {need}")
    if shape is not None:
        product = 1
        for d in shape:
            product *= int(d)
        if product != int(n):
            raise CatalogLayoutRefuse(f"{name} f32v2 numel {n} != shape product {product}")
    return {
        "representation": "f32v2_le",
        "codec_name": "f32v2",
        "bits": 32,
        "group_size": 1,
        "shape": list(shape) if shape is not None else [int(n)],
        "elements": int(n),
        "groups": int(n),
        "header_bytes": F32V2_HEADER,
        "scale_bytes": 0,
        "bias_bytes": 0,
        "code_bytes": int(n) * F32_BYTES,
        "payload_off": F32V2_HEADER,
        "reconstruction": "little-endian f32 values after a u64 count",
        "has_bias": False,
    }


def _read_rel(rel: str) -> tuple[str | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace"), "disk"
        except OSError:
            pass
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, "git:HEAD"
    return None, "missing"


# ---------------------------------------------------------------------------
# Catalog rows.
# ---------------------------------------------------------------------------


def dn_records(
    *,
    root: Path | None = None,
    catalog_records: Sequence[Mapping[str, Any]] | None = None,
    organs: Sequence[str] = DN_ORGANS,
) -> list[dict[str, Any]]:
    artifact = root if root is not None else resolve_artifact_root()
    records = (
        list(catalog_records)
        if catalog_records is not None
        else parse_catalog_records(artifact / CATALOG_NAME)
    )
    want = set(organs)
    out: list[dict[str, Any]] = []
    for rec in records:
        name = str(rec["name"])
        try:
            layer, organ, whole = classify_tensor(name)
        except Exception:
            continue
        if organ not in want:
            continue
        if not whole:
            raise CatalogLayoutRefuse(f"{name} is a partial-tensor organ; DeltaNet weights are whole")
        out.append(
            {
                **dict(rec),
                "layer": layer,
                "organ": organ,
                "segment_path": str(artifact / "segments" / rec["filename"]),
            }
        )
    if not out:
        raise DeltaNetRefuse(f"catalog holds no tensors in {sorted(want)}; refusing")
    return out


def tensor_accounting_row(rec: Mapping[str, Any], geo: Mapping[str, Any]) -> dict[str, Any]:
    """Per-tensor header/scale/bias/code bytes from the real payload."""
    path = Path(rec["segment_path"])
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise CatalogAbsent(f"cannot read {path}: {exc}") from exc
    if int(rec["stored_bytes"]) != len(blob):
        raise CatalogLayoutRefuse(
            f"{rec['name']}: catalog stored {rec['stored_bytes']} != file {len(blob)}"
        )
    organ = str(rec["organ"])
    shape = [int(x) for x in rec["shape"]]
    if organ in Q4_ORGANS:
        parsed = parse_hq30uq4_header(blob, name=str(rec["name"]))
        if parsed["shape"] != shape:
            raise CatalogLayoutRefuse(
                f"{rec['name']}: HQ30UQ4 shape {parsed['shape']} != catalog {shape}"
            )
        expected = q4_parts(shape[0], shape[1] if len(shape) > 1 else 1)
        if expected["stored_bytes"] != len(blob):
            raise CatalogLayoutRefuse(
                f"{rec['name']}: geometry Q4 {expected['stored_bytes']} != file {len(blob)}"
            )
        if organ == "attention.linear_qkvz" and shape != [geo["qkvz_rows"], geo["qkvz_cols"]]:
            raise CatalogLayoutRefuse(
                f"{rec['name']}: qkvz shape {shape} != geometry "
                f"[{geo['qkvz_rows']}, {geo['qkvz_cols']}]"
            )
        if organ == "attention.linear_out" and shape != [geo["out_rows"], geo["out_cols"]]:
            raise CatalogLayoutRefuse(f"{rec['name']}: out_proj shape {shape} != geometry")
        if organ == "attention.linear_ba" and shape != [geo["ba_rows"], geo["ba_cols"]]:
            raise CatalogLayoutRefuse(f"{rec['name']}: ba shape {shape} != geometry")
    elif organ in F32_ORGANS:
        parsed = parse_f32v2_header(blob, name=str(rec["name"]), shape=shape)
        if organ == "attention.linear_conv1d":
            want_n = int(geo["conv_channels"]) * int(geo["conv_kernel"])
            if parsed["elements"] != want_n:
                raise CatalogLayoutRefuse(
                    f"{rec['name']}: conv1d elements {parsed['elements']} != {want_n}"
                )
    else:
        parsed = parse_f32v2_header(blob, name=str(rec["name"]), shape=shape)
    parts = (
        parsed["header_bytes"]
        + parsed["scale_bytes"]
        + parsed["bias_bytes"]
        + parsed["code_bytes"]
    )
    if parts != len(blob):
        raise CatalogLayoutRefuse(
            f"{rec['name']}: header+scale+bias+code {parts} != stored {len(blob)}"
        )
    return {
        "name": rec["name"],
        "layer": rec["layer"],
        "organ": organ,
        "shape": parsed["shape"],
        "elements": parsed["elements"],
        "representation": parsed["representation"],
        "codec_name": parsed["codec_name"],
        "bits": parsed["bits"],
        "group_size": parsed["group_size"],
        "groups": parsed.get("groups"),
        "header_bytes": parsed["header_bytes"],
        "scale_bytes": parsed["scale_bytes"],
        "bias_bytes": parsed["bias_bytes"],
        "code_bytes": parsed["code_bytes"],
        "stored_bytes": len(blob),
        "code_share": parsed["code_bytes"] / len(blob),
        "scale_share": parsed["scale_bytes"] / len(blob),
        "header_share": parsed["header_bytes"] / len(blob),
        "reconstruction": parsed["reconstruction"],
        "segment_path": rec["segment_path"],
        "has_bias": parsed["has_bias"],
    }


def reconcile_deltanet(
    got: int,
    *,
    want: int = DELTANET_ACTIVE_TARGET,
    detail: str = "",
) -> int:
    """Refuse unless the per-tensor sum is the recorded organ total."""
    if int(got) != int(want):
        raise UnreconciledDeltaNet(int(got), int(want), detail=detail)
    return int(got)


def accounting_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    want: int = DELTANET_ACTIVE_TARGET,
) -> dict[str, Any]:
    if not rows:
        raise DeltaNetRefuse("no DeltaNet rows; refusing an empty accounting")
    header = sum(int(r["header_bytes"]) for r in rows)
    scale = sum(int(r["scale_bytes"]) for r in rows)
    bias = sum(int(r["bias_bytes"]) for r in rows)
    code = sum(int(r["code_bytes"]) for r in rows)
    stored = sum(int(r["stored_bytes"]) for r in rows)
    reconcile_deltanet(stored, want=want, detail="sum of per-tensor stored bytes")
    if header + scale + bias + code != stored:
        raise CatalogLayoutRefuse("parts do not reassemble to stored DeltaNet bytes")
    by_organ: dict[str, dict[str, int]] = {}
    for rec in rows:
        slot = by_organ.setdefault(
            str(rec["organ"]),
            {
                "n_tensors": 0,
                "header_bytes": 0,
                "scale_bytes": 0,
                "bias_bytes": 0,
                "code_bytes": 0,
                "stored_bytes": 0,
                "elements": 0,
            },
        )
        slot["n_tensors"] += 1
        slot["header_bytes"] += int(rec["header_bytes"])
        slot["scale_bytes"] += int(rec["scale_bytes"])
        slot["bias_bytes"] += int(rec["bias_bytes"])
        slot["code_bytes"] += int(rec["code_bytes"])
        slot["stored_bytes"] += int(rec["stored_bytes"])
        slot["elements"] += int(rec["elements"])
    for organ, target in (
        ("attention.linear_qkvz", QKVZ_ACTIVE_TARGET),
        ("attention.linear_out", OUT_ACTIVE_TARGET),
        ("attention.linear_ba", BA_ACTIVE_TARGET),
        ("attention.linear_conv1d", CONV_ACTIVE_TARGET),
    ):
        got = int(by_organ[organ]["stored_bytes"])
        if got != target:
            raise UnreconciledDeltaNet(got, target, detail=f"organ {organ}")
    q4_elems = sum(int(r["elements"]) for r in rows if r["organ"] in Q4_ORGANS)
    q4_stored = sum(int(r["stored_bytes"]) for r in rows if r["organ"] in Q4_ORGANS)
    return {
        "n_tensors": len(rows),
        "header_bytes": header,
        "scale_bytes": scale,
        "bias_bytes": bias,
        "code_bytes": code,
        "stored_bytes": stored,
        "auxiliary_bytes": header + scale + bias,
        "code_share": code / stored,
        "scale_share": scale / stored,
        "header_share": header / stored,
        "auxiliary_share": (header + scale + bias) / stored,
        "share_of_token": stored / TOKEN_ACTIVE_TARGET,
        "target": want,
        "reconciled": True,
        "by_organ": {
            name: {
                **slot,
                "share_of_deltanet": slot["stored_bytes"] / stored,
                "share_of_token": slot["stored_bytes"] / TOKEN_ACTIVE_TARGET,
                "derived_bpw": (
                    8.0 * slot["stored_bytes"] / slot["elements"] if slot["elements"] else None
                ),
            }
            for name, slot in by_organ.items()
        },
        "incumbent_packing": {
            "qkvz_ba_out": "hq30uq4_uniform_q4_group64_signed_nibble_f16_scale",
            "conv1d": "f32v2_le",
            "q4_code_bits": Q4_CODE_BITS,
            "q4_group_size": INCUMBENT_GROUP,
            "q4_elements": q4_elems,
            "q4_stored_bytes": q4_stored,
            "q4_derived_bpw": 8.0 * q4_stored / q4_elems,
            "bias": "none on HQ30UQ4; conv1d is unscaled f32",
            "kernel_q4": "qwen_uniform_q4_* / geo_tpr64",
            "kernel_state": "qwen38_gated_delta_decode_vi_simd (FUSE_BA_DELTA folds ba_to_decay)",
        },
    }


def census_rows(*, root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = root if root is not None else resolve_artifact_root()
    geo = load_dn_geometry(artifact)
    recs = dn_records(root=artifact)
    rows = [tensor_accounting_row(r, geo) for r in recs]
    if len(rows) != 4 * int(geo["n_deltanet_layers"]):
        raise DeltaNetRefuse(
            f"expected {4 * geo['n_deltanet_layers']} DeltaNet weight tensors, got {len(rows)}"
        )
    return rows, geo


def accounting(*, root: Path | None = None) -> dict[str, Any]:
    rows, geo = census_rows(root=root)
    snap = accounting_from_rows(rows)
    snap["identity"] = _identity(root, geo)
    snap["qkvz_subblocks"] = qkvz_subblock_parts(geo)
    snap["per_layer"] = _per_layer(rows, geo)
    snap["adjacent_not_in_2gb"] = _adjacent(root, geo)
    snap["geometry_state"] = geometry_state(geo)
    return snap


def _identity(root: Path | None, geo: Mapping[str, Any]) -> dict[str, Any]:
    sealed = load_sealed()
    artifact = root if root is not None else resolve_artifact_root(sealed)
    return {
        "resident_identity": sealed.get("resident_identity"),
        "artifact_root": str(artifact),
        "catalog": str(artifact / CATALOG_NAME),
        "model_id": sealed.get("model_id"),
        "geometry": {
            "hidden_size": geo["hidden_size"],
            "n_deltanet_layers": geo["n_deltanet_layers"],
            "key_heads": geo["key_heads"],
            "value_heads": geo["value_heads"],
            "key_head_dim": geo["key_head_dim"],
            "value_head_dim": geo["value_head_dim"],
            "qkvz_rows": geo["qkvz_rows"],
            "qkvz_layout": geo["qkvz_layout"],
            "conv_channels": geo["conv_channels"],
            "conv_kernel": geo["conv_kernel"],
        },
    }


def _per_layer(rows: Sequence[Mapping[str, Any]], geo: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[str, Any]] = {}
    for rec in rows:
        layer = int(rec["layer"])
        slot = by_layer.setdefault(
            layer,
            {"layer": layer, "kind": "linear_attention", "organs": {}, "stored_bytes": 0},
        )
        slot["organs"][rec["organ"]] = {
            "organ": rec["organ"],
            "stored_bytes": rec["stored_bytes"],
            "header_bytes": rec["header_bytes"],
            "scale_bytes": rec["scale_bytes"],
            "code_bytes": rec["code_bytes"],
            "bits": rec["bits"],
            "representation": rec["representation"],
        }
        slot["stored_bytes"] += int(rec["stored_bytes"])
    out = [by_layer[i] for i in sorted(by_layer)]
    if len(out) != int(geo["n_deltanet_layers"]):
        raise DeltaNetRefuse(f"per-layer count {len(out)} != {geo['n_deltanet_layers']}")
    layer_bytes = {int(r["stored_bytes"]) for r in out}
    if len(layer_bytes) != 1:
        raise CatalogLayoutRefuse(f"DeltaNet layers are not uniform: {sorted(layer_bytes)}")
    return out


def _adjacent(root: Path | None, geo: Mapping[str, Any]) -> dict[str, Any]:
    """A_log, dt_bias, linear_attn.norm: catalogued, not in the 2.96 GB bar."""
    artifact = root if root is not None else resolve_artifact_root()
    recs = dn_records(root=artifact, organs=ADJACENT_ORGANS)
    rows = [tensor_accounting_row(r, geo) for r in recs]
    stored = sum(int(r["stored_bytes"]) for r in rows)
    return {
        "note": (
            "A_log, dt_bias and linear_attn.norm are DeltaNet parameters in "
            "HQ38M20 and ARE counted in the whole-token census under state.* / "
            "norms.*. They are not in the 2,961,659,904 organ bar this module "
            "reconciles. Recorded so the 2.96 GB is not mistaken for 'every "
            "DeltaNet catalog byte'."
        ),
        "stored_bytes": stored,
        "in_2gb_bar": False,
        "by_organ": {
            name: {
                "n_tensors": sum(1 for r in rows if r["organ"] == name),
                "stored_bytes": sum(int(r["stored_bytes"]) for r in rows if r["organ"] == name),
                "representation": next(r["representation"] for r in rows if r["organ"] == name),
                "bits": next(r["bits"] for r in rows if r["organ"] == name),
            }
            for name in ADJACENT_ORGANS
        },
    }


def geometry_state(geo: Mapping[str, Any]) -> dict[str, Any]:
    """Recurrent + conv state. Not catalog. Not in the 2.96 GB bar."""
    n = int(geo["n_deltanet_layers"])
    rec_one = int(geo["recurrent_state_elements_per_layer"]) * F32_BYTES
    conv_one = int(geo["conv_state_elements_per_layer"]) * F32_BYTES
    rec = rec_one * n
    conv = conv_one * n
    return {
        "in_catalog": False,
        "in_2gb_bar": False,
        "dtype": "f32",
        "bits": 32,
        "n_layers": n,
        "recurrent_state": {
            "layout": "value_heads x key_head_dim x value_head_dim",
            "elements_per_layer": int(geo["recurrent_state_elements_per_layer"]),
            "resident_bytes": rec,
            "rw_bytes_per_token": rec * 2,
            "consumed_by": "qwen38_gated_delta_decode_vi_simd: every element is read and written every token",
        },
        "conv_state": {
            "layout": "conv_channels x (kernel-1)",
            "elements_per_layer": int(geo["conv_state_elements_per_layer"]),
            "resident_bytes": conv,
            "rw_bytes_per_token": conv * 2,
            "consumed_by": "qwen38_qkvz_rearrange_conv_l2_f32; z does not enter conv",
        },
        "resident_bytes": rec + conv,
        "rw_bytes_per_token": (rec + conv) * 2,
        "reason": (
            "HQ38M20 describes packed weights only. State bytes are exact from "
            "Qwen38DeltaNetLayout and are reported so a packing attack on W is "
            "not confused with a packing attack on S."
        ),
    }


# ---------------------------------------------------------------------------
# Independent information. Source-true roles, then array measurements.
# ---------------------------------------------------------------------------


def independent_information(geo: Mapping[str, Any]) -> dict[str, Any]:
    """What q, k, v, z, S actually do. From the consume kernels, not from bytes."""
    rec = int(geo["recurrent_state_elements_per_layer"])
    return {
        "operator": (
            "S_t = (I - beta k k^T) (decay * S_{t-1}) + beta k v^T; "
            "h_t = S_t^T q; gated = RMSNorm(h) * silu(z). "
            "decay, beta come from in_proj_ba + A_log + dt_bias."
        ),
        "source": (
            "crates/hawking-core/shaders/qwen38_device_activations.metal "
            "qwen38_gated_delta_decode_vi / qwen38_gated_delta_decode_vi_simd"
        ),
        "state_rank_growth": (
            "The update is rank-1 per token after a scalar decay. S is a "
            f"{geo['key_head_dim']}x{geo['value_head_dim']} matrix per value "
            f"head ({geo['value_heads']} heads). It is not a diagonal SSM. "
            f"{rec} f32 cells per layer are independently addressed."
        ),
        "roles": {
            "q": {
                "rows": geo["q_rows"],
                "enters_conv": True,
                "enters_gated_delta": True,
                "enters_state_update": False,
                "role": "readout: h = S^T q",
                "sensitivity_implication": (
                    "Injuring q injures h without rewriting S. Next token still "
                    "sees an intact S."
                ),
            },
            "k": {
                "rows": geo["k_rows"],
                "enters_conv": True,
                "enters_gated_delta": True,
                "enters_state_update": True,
                "role": "rank-1 projector (I - beta k k^T) AND the write key k v^T",
                "sensitivity_implication": (
                    "k appears twice in the update. Equal W-space distortion "
                    "with q/v/z is not a reason to give k equal bits."
                ),
            },
            "v": {
                "rows": geo["v_rows"],
                "enters_conv": True,
                "enters_gated_delta": True,
                "enters_state_update": True,
                "role": "write value: beta k v^T",
                "sensitivity_implication": "Injuring v writes a wrong column into S.",
            },
            "z": {
                "rows": geo["z_rows"],
                "enters_conv": False,
                "enters_gated_delta": False,
                "enters_state_update": False,
                "role": "output gate after RMSNorm(h); never bound to gated-delta",
                "sensitivity_implication": (
                    "Lowering z bits cannot corrupt rec_state. This is source-"
                    "true, not a measurement. z is 37.5% of linear_qkvz rows."
                ),
            },
        },
        "equal_precision_default": {
            "incumbent_does": "one HQ30UQ4 GEMV over the fused 16384-row tensor",
            "justified_by_consume": False,
            "justified_by_weight_space_relfro": (
                "weight-space Q3/Q2 rel-fro is similar across q/k/v/z; that "
                "is packing uniformity, not operator sensitivity"
            ),
            "heterogeneous_allocation_available": True,
            "why_available": (
                "qwen38_qkvz_rearrange_conv_l2_f32 already splits the GEMV "
                "output into q, k, v, z buffers. A row-range bit-width Q4 "
                "matvec (or a split decode that does not write dense W) can "
                "spend bits differently per range. Four independent tensors "
                "without a fused kernel would add dispatches and is a "
                "different candidate."
            ),
            "physical_primitive": "FusedDecodeCompute",
        },
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def _shannon_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    if p.size == 0:
        return 0.0
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _shannon_f16(arr: np.ndarray) -> tuple[float, int]:
    bits = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype="<u2")
    counts = np.bincount(bits, minlength=65536)
    return _shannon_bits(counts), int(np.count_nonzero(counts))


def _relfro(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.square(a - b).sum()))
    den = float(np.sqrt(np.square(b).sum()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _requant_q(q: np.ndarray, bits: int) -> np.ndarray:
    """Drop incumbent signed nibbles q=nibble-8 in [-8,7] onto 2^bits levels."""
    levels = 1 << int(bits)
    lo, hi = -8.0, 7.0
    u = np.clip(np.round((q - lo) / (hi - lo) * (levels - 1)), 0, levels - 1)
    return lo + u * ((hi - lo) / (levels - 1))


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


def measure_qkvz_subblocks(
    rows: Sequence[Mapping[str, Any]],
    geo: Mapping[str, Any],
    *,
    sample_layers: Sequence[int] = SAMPLE_LAYERS,
    recon_groups: int = RECON_GROUPS,
    rng_seed: int = RNG_SEED,
) -> dict[str, Any]:
    """CPU statistics on fused QKVZ scales (all 48) and sampled nibble codes."""
    idx = fused_qkvz_row_indices(geo)
    qkvz = [r for r in rows if r["organ"] == "attention.linear_qkvz"]
    if len(qkvz) != int(geo["n_deltanet_layers"]):
        raise DeltaNetRefuse(f"qkvz tensors {len(qkvz)} != {geo['n_deltanet_layers']}")
    rows_n = int(geo["qkvz_rows"])
    gpr = int(geo["qkvz_cols"]) // INCUMBENT_GROUP
    acc: dict[str, dict[str, list[Any]]] = {
        s: {"mean_abs": [], "std": [], "shannon": [], "unique": [], "energy": []}
        for s in SUBBLOCKS
    }
    row_means: dict[str, dict[int, np.ndarray]] = {s: {} for s in SUBBLOCKS}
    rng = np.random.default_rng(rng_seed)
    sample_set = {int(x) for x in sample_layers}
    requant: dict[str, dict[str, list[float]]] = {
        s: {"q3": [], "q2": [], "q1": [], "global_scale": []} for s in SUBBLOCKS
    }
    energy_shares: list[dict[str, float]] = []

    for rec in qkvz:
        path = Path(rec["segment_path"])
        blob = path.read_bytes()
        parsed = parse_hq30uq4_header(blob, name=str(rec["name"]))
        scales = np.frombuffer(
            blob[parsed["payload_off"] : parsed["payload_off"] + parsed["scale_bytes"]],
            dtype="<f2",
        )
        S = scales.astype(np.float32).reshape(rows_n, gpr)
        layer = int(rec["layer"])
        e: dict[str, float] = {}
        for s, rows_i in idx.items():
            sub = S[rows_i]
            mag = np.abs(sub)
            acc[s]["mean_abs"].append(float(mag.mean()))
            acc[s]["std"].append(float(sub.std()))
            sh, un = _shannon_f16(scales.reshape(rows_n, gpr)[rows_i])
            acc[s]["shannon"].append(sh)
            acc[s]["unique"].append(un)
            e[s] = float(np.square(sub).sum())
            acc[s]["energy"].append(e[s])
            row_means[s][layer] = sub.mean(axis=1)
        tot = sum(e.values()) or 1.0
        energy_shares.append({s: e[s] / tot for s in SUBBLOCKS} | {"layer": layer})

        if layer in sample_set:
            code_off = parsed["payload_off"] + parsed["scale_bytes"]
            codes = np.frombuffer(
                blob[code_off : code_off + parsed["code_bytes"]], dtype=np.uint8
            ).reshape(rows_n, gpr, INCUMBENT_GROUP // 2)
            for s, rows_i in idx.items():
                n_r = int(rows_i.size)
                take = min(int(recon_groups), n_r * gpr)
                ri = rng.integers(0, n_r, size=take)
                gi = rng.integers(0, gpr, size=take)
                abs_rows = rows_i[ri]
                ss = S[abs_rows, gi]
                packed = codes[abs_rows, gi]
                lo = packed & 0x0F
                hi = packed >> 4
                nib = np.stack([lo, hi], axis=2).reshape(take, INCUMBENT_GROUP)
                q = nib.astype(np.float32) - 8.0
                W = q * ss[:, None]
                for bits, key in ((3, "q3"), (2, "q2"), (1, "q1")):
                    requant[s][key].append(_relfro(_requant_q(q, bits) * ss[:, None], W))
                gscale = np.full_like(ss, float(ss.mean()))
                requant[s]["global_scale"].append(_relfro(q * gscale[:, None], W))

    layers = sorted(row_means["q"])
    scalars = {
        s: np.array([float(row_means[s][L].mean()) for L in layers]) for s in SUBBLOCKS
    }
    cross_block: dict[str, float] = {}
    for a in SUBBLOCKS:
        for b in SUBBLOCKS:
            if a >= b:
                continue
            cross_block[f"{a}_vs_{b}"] = _pearson(scalars[a], scalars[b])
    cross_layer: dict[str, dict[str, float]] = {}
    for s in SUBBLOCKS:
        cors = [
            _pearson(row_means[s][a], row_means[s][b])
            for a, b in zip(layers, layers[1:])
        ]
        cross_layer[s] = {
            "n_pairs": len(cors),
            "mean": float(sum(cors) / len(cors)) if cors else float("nan"),
            "min": float(min(cors)) if cors else float("nan"),
            "max": float(max(cors)) if cors else float("nan"),
        }

    def _avg(vals: Sequence[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else float("nan")

    by_block = {}
    for s in SUBBLOCKS:
        by_block[s] = {
            "n_tensors": len(acc[s]["mean_abs"]),
            "rows": int(idx[s].size),
            "mean_abs_scale": _avg(acc[s]["mean_abs"]),
            "std_scale": _avg(acc[s]["std"]),
            "scale_shannon_bits": _avg(acc[s]["shannon"]),
            "scale_unique_f16_mean": _avg(acc[s]["unique"]),
            "scale_l2_energy_mean": _avg(acc[s]["energy"]),
            "requant_relfro_vs_incumbent_q4": {
                k: _avg(requant[s][k]) for k in ("q3", "q2", "q1", "global_scale")
            },
        }
    row_share = {s: int(idx[s].size) / rows_n for s in SUBBLOCKS}
    mean_energy_share = {
        s: _avg([e[s] for e in energy_shares]) for s in SUBBLOCKS
    }
    return {
        "n_tensors_measured": len(qkvz),
        "sample_layers": list(sample_layers),
        "reconstruction_groups_per_tensor": recon_groups,
        "by_subblock": by_block,
        "row_share": row_share,
        "scale_l2_energy_share_mean": mean_energy_share,
        "cross_block_layer_mean_scale_pearson": cross_block,
        "cross_layer_per_row_mean_scale_pearson": cross_layer,
        "note": (
            "Shannon is over f16 bit patterns of per-group scales. Requant "
            "rel-fro drops incumbent signed nibbles onto 2^b levels with the "
            "same scale; it is not a Q3 refit and not a generate gate. "
            "Energy share tracking row share means the four blocks are packed "
            "similarly, not that they are equally sensitive as operators."
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
        for organ in ("deltanet", "attention"):
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
                }
            )
    return hits


def _nns_cite(nns_id: str, *, this_object: str) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    src = NOETIC_RELS[0]
    for rel in NOETIC_RELS:
        text, _via = _read_rel(rel)
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
            p
            for p in (str(scope.get("model") or "").strip(), str(scope.get("organ") or "").strip())
            if p
        )
        or "as recorded in NOETIC_NEGATIVE_SCIENCE",
        "kind": str(entry.get("kind") or ""),
        "this_specimen": this_object,
    }


def _qn_cite(qn_id: str, claim: str, reopen: str, *, organ: str) -> dict[str, Any]:
    return {
        "scar_id": qn_id,
        "source_path": QN_REL,
        "claim_refuted": claim,
        "reopen_condition": reopen,
        "surface": f"qwen3.8-27b-abliterated {organ}",
        "kind": "MODEL_SPECIFIC",
        "this_specimen": "qwen3.8-27b sealed-3.14 DeltaNet",
    }


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise DeltaNetRefuse(f"{name} is not an atlas primitive")
    return name


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


# ---------------------------------------------------------------------------
# Candidates.
# ---------------------------------------------------------------------------


def candidates(
    acc: Mapping[str, Any],
    meas: Mapping[str, Any],
    info: Mapping[str, Any],
    geo: Mapping[str, Any],
    *,
    consult_index: bool = True,
) -> list[dict[str, Any]]:
    stored = int(acc["stored_bytes"])
    qkvz_b = int(acc["by_organ"]["attention.linear_qkvz"]["stored_bytes"])
    out_b = int(acc["by_organ"]["attention.linear_out"]["stored_bytes"])
    conv_b = int(acc["by_organ"]["attention.linear_conv1d"]["stored_bytes"])
    scale_b = int(acc["scale_bytes"])
    header_b = int(acc["header_bytes"])
    code_b = int(acc["code_bytes"])
    sub = acc["qkvz_subblocks"]
    q_code = int(sub["q"]["code_bytes"])
    k_code = int(sub["k"]["code_bytes"])
    v_code = int(sub["v"]["code_bytes"])
    z_code = int(sub["z"]["code_bytes"])
    q3_save = {
        "q": q_code - (q_code * 3) // 4,
        "k": k_code - (k_code * 3) // 4,
        "v": v_code - (v_code * 3) // 4,
        "z": z_code - (z_code * 3) // 4,
    }
    q4_elems = int(acc["incumbent_packing"]["q4_elements"])
    g64_scale = scale_b
    g128_scale = (q4_elems // 128) * F16_BYTES
    g256_scale = (q4_elems // 256) * F16_BYTES
    g1024_scale = (q4_elems // 1024) * F16_BYTES
    rec_res = int(acc["geometry_state"]["recurrent_state"]["resident_bytes"])
    rec_rw = int(acc["geometry_state"]["recurrent_state"]["rw_bytes_per_token"])
    conv_payload = int(acc["by_organ"]["attention.linear_conv1d"]["code_bytes"])
    by_block = meas.get("by_subblock") or {}
    q3_rel = {s: (by_block.get(s) or {}).get("requant_relfro_vs_incumbent_q4", {}).get("q3") for s in SUBBLOCKS}
    cl = meas.get("cross_layer_per_row_mean_scale_pearson") or {}
    specimen = "qwen3.8-27b sealed-3.14 DeltaNet HQ30UQ4"

    nns019 = _nns_cite("NNS-019", this_object=specimen)
    nns029 = _nns_cite("NNS-029", this_object=specimen)
    nns016 = _nns_cite("NNS-016", this_object=specimen)
    qn_state = _qn_cite(
        "QN-STATE-MERGING",
        "depth-state and KV merging measured negative on this Qwen under the tested conditions",
        "a state topology (recurrent, latent-attention, or a longer-context regime) where merged state preserves capability",
        organ="kv_state+deltanet_state",
    )
    qn_head = _qn_cite(
        "QN-HEAD-REDUNDANCY",
        "Q heads mean cosine 0.0438 and K/V/O similarly near-orthogonal, so there is no shared-head structure to exploit",
        "an organ or model where head cosine similarity is high enough that sharing costs less capability than the bits it saves",
        organ="gqa_attention",
    )
    qn_shared = _qn_cite(
        "QN-SHARED-BASIS-DENSITY",
        "no K below ~2.25 bpw composes coherently for the MLP: the local functional probe dies at held-out activation",
        "a shared-basis point that is coherent at held-out activation AND beats q2f on both density and COMPLETE_TOKEN_NS",
        organ="mlp_gate_up+mlp_down",
    )

    z_enters = bool(info["roles"]["z"]["enters_gated_delta"])
    if z_enters:
        raise DeltaNetRefuse("z is documented as entering gated-delta; the shader says it does not")

    rows: list[dict[str, Any]] = [
        {
            "id": "heterogeneous_qkvz_bits",
            "name": "heterogeneous bit allocation across q, k, v, z",
            "mechanism": (
                "linear_qkvz is one HQ30UQ4 GEMV over a fused 16384-row tensor. "
                "The consume path already splits those rows: z never enters "
                "gated-delta or conv; q is a readout of S; k is the rank-1 "
                "projector and write key; v is the write value. Spend 4-bit "
                "codes only on the ranges the state machine needs, and a "
                "narrower code on the rest, decoded in-register by a "
                "row-range Q4/Q3 matvec. Do not unpack to dense W."
            ),
            "byte_model": (
                f"qkvz codes {q_code + k_code + v_code + z_code} across "
                f"q/k={q_code} each, v/z={v_code} each. Q3 codes save 1/4 of "
                f"the crushed block: q or k {q3_save['q']}, v or z {q3_save['v']}. "
                "Scales stay f16 per group unless a refit drops them too. "
                "Header 40 B/tensor is not split."
            ),
            "bytes_eliminated_if_true": q3_save["z"],
            "bytes_eliminated_breakdown": {
                "q_codes_to_3bit": q3_save["q"],
                "k_codes_to_3bit": q3_save["k"],
                "v_codes_to_3bit": q3_save["v"],
                "z_codes_to_3bit": q3_save["z"],
                "q_and_k_to_3bit": q3_save["q"] + q3_save["k"],
                "note": (
                    "z_codes_to_3bit is the listed bytes_eliminated_if_true "
                    "because z is the only block that cannot corrupt rec_state. "
                    "That is a consume-path ranking, not a generate gate. "
                    "Crushing v instead saves the same number of bytes and "
                    "would rewrite S."
                ),
            },
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A native row-range Q4/Q3 GEMV consumes packed codes. "
                "Splitting the fused tensor, writing four dense W, and running "
                "ordinary GEMV is REJECTED_DENSE_REMAT and also adds launches."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "STATIC, already run: weight-space nibble-drop Q3 rel-fro is "
                f"q={q3_rel['q']}, k={q3_rel['k']}, v={q3_rel['v']}, z={q3_rel['z']} "
                "— packing uniformity, not operator ranking. CHEAP CPU next: "
                "decode one DN layer's real post-norm x, keep three blocks at "
                "Q4, drop one to Q3 (refit scales on that range), report "
                "rec_out rel-fro AND rec_state rel-fro. z-only Q3 must leave "
                "rec_state bit-identical if the shader binding is honest. If "
                "the four injuries are equal on rec_out, heterogeneous bits "
                "die as a byte-weighted idea. Do not unpack to dense W. Do "
                "not skip the rec_state comparison."
            ),
            "index_slugs": ["uniform_subbit_allocation"],
            "measured": {
                "z_enters_gated_delta": False,
                "z_enters_conv": False,
                "weight_space_q3_relfro": q3_rel,
                "row_share": meas.get("row_share"),
                "scale_l2_energy_share_mean": meas.get("scale_l2_energy_share_mean"),
            },
            "note": (
                "The incumbent fused GEMV is an equal-precision default. The "
                "consume kernels are not. W-space similarity is recorded so "
                "nobody claims the arrays 'look different enough' to skip the "
                "function-space probe."
            ),
        },
        {
            "id": "lower_bit_uniform_qkvz",
            "name": "uniform bit-descent of the fused qkvz Q4",
            "mechanism": (
                "Replace HQ30UQ4 group-64 on all 16384 rows with uniform Q3 "
                "or Q2 of the same W, still consumed by a native GEMV."
            ),
            "byte_model": (
                f"qkvz codes {q_code + k_code + v_code + z_code}. Uniform Q3 "
                f"saves 1/4 of those codes ({(q_code + k_code + v_code + z_code) // 4}); "
                f"uniform Q2 saves half. Scales remain unless G changes."
            ),
            "bytes_eliminated_if_true": (q_code + k_code + v_code + z_code) // 4,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "qwen_uniform_q4 already dequants in-register. A Q3 sibling "
                "is the same primitive. Unpack-to-f16-then-GEMV is "
                "REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "NNS-029 killed uniform bit-descent below q3 as a clean path "
                "on the whole Qwen3.8 artifact and uniform Q2 on this MLP. "
                "That is a cousin of a *whole-body* plan, not a measured DN-"
                "only Q3. CHEAP CPU: one-layer Q3 refit of qkvz on real x vs "
                "incumbent Q4 rec_out. If rel-fro sits in the NNS-029 Q2 band "
                "(~0.58 vs q3 0.20 on MLP W), stop before a generate. A retry "
                "of HGRAVB01/R02/S01 on this organ is gravity_family_on_dn_weights, "
                "already falsified."
            ),
            "citations": [nns029],
            "index_slugs": ["uniform_q2", "uniform_q3"],
            "cousin_not_this_object": True,
            "note": (
                "NNS-029 organ is MLP (sparsity) and whole artifact "
                "(bit-descent). Cited so a uniform Q2 of the *body* is not "
                "re-proposed as a DN idea. DN-only native Q3 remains OPEN."
            ),
        },
        {
            "id": "lower_bit_out_proj",
            "name": "uniform bit-descent of linear_out (8.12% of the token)",
            "mechanism": (
                "linear_out is HQ30UQ4 [5120, 6144], consumed after gated "
                "RMSNorm as a single GEMV into residual-add. Same packing as "
                "qkvz, different operator (no recurrent state)."
            ),
            "byte_model": (
                f"out stored {out_b}. Q3 codes save 1/4 of out code bytes "
                f"{int(acc['by_organ']['attention.linear_out']['code_bytes']) // 4}."
            ),
            "bytes_eliminated_if_true": int(acc["by_organ"]["attention.linear_out"]["code_bytes"]) // 4,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": "Native Q3 GEMV. No dense W.",
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "CHEAP CPU: Q3 refit of one out_proj on real gated x vs Q4. "
                "NNS-019 already killed Gravity families on DeltaNet out at "
                "the Q4-equivalent cosine bar; a NEW Q3 family is the reopen, "
                "not HGRAVB01."
            ),
            "citations": [nns019],
            "index_slugs": ["uniform_q3"],
        },
        {
            "id": "gravity_family_on_dn_weights",
            "name": "HGRAVB01 / R02 / S01 (or Q80 mixed bundle) on DeltaNet in/out",
            "mechanism": (
                "Replace HQ30UQ4 qkvz/out with an existing Gravity expert "
                "family scored as if Q4-equivalent."
            ),
            "byte_model": "Whatever that family stores vs 2.94 GB of qkvz+out. Irrelevant: the quality bar already failed.",
            "bytes_eliminated_if_true": 0,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Gravity kernels consume packed codes. The kill is capability, "
                "not remat."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run: NNS-019, scope explicitly 'Q/K/V/O, DeltaNet "
                "in/out'. Reopen is a NEW attention codec family on real BF16 "
                "X, mean-row cosine ≥ 0.990 AND a multi-prompt generate gate. "
                "Not another HGRAVB01/R02/S01 transfer."
            ),
            "citations": [nns019],
            "index_slugs": ["hgravb01", "hgravr02", "hgravs01"],
        },
        {
            "id": "larger_q4_group",
            "name": "larger HQ30UQ4 group size (byte curve exact; capability not)",
            "mechanism": (
                "Auxiliary Q4 bytes are f16 scale per group. Codes are 4 bits "
                "per weight regardless of G. gcd(5120, 6144)=1024, so G in "
                "{128,256,512,1024} tiles qkvz, ba and out. Parser already "
                "admits G=128."
            ),
            "byte_model": (
                f"Q4 elements {q4_elems}. scale(G)=2*{q4_elems}/G. Incumbent "
                f"G=64 → {g64_scale}. G=128 → {g128_scale} (save "
                f"{g64_scale - g128_scale}); G=256 → {g256_scale}; G=1024 → "
                f"{g1024_scale}. Codes {code_b - conv_payload} unchanged. "
                f"Headers {header_b} unchanged."
            ),
            "bytes_eliminated_if_true": g64_scale - g128_scale,
            "bytes_eliminated_breakdown": {
                "G128": g64_scale - g128_scale,
                "G256": g64_scale - g256_scale,
                "G1024": g64_scale - g1024_scale,
            },
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Same Q4 kernel, different G. Not a dense W."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "CHEAP CPU: LS-refit one qkvz and one out_proj at G=128 from "
                "incumbent reconstruction (weaker) or parent bf16 (honest). "
                "If W rel-fro jumps like the nibble-drop Q2 band (~0.55), G "
                "dies without a generate."
            ),
            "index_slugs": ["uniform_q4"],
        },
        {
            "id": "shared_transforms_across_layers",
            "name": "share qkvz / out across the 48 DN layers",
            "mechanism": (
                "One (or K) shared W with per-layer coefficients, or a shared "
                "input transform T then small per-layer maps. Would store T "
                "once instead of 48 copies of 44.6 MB qkvz."
            ),
            "byte_model": (
                f"48 independent qkvz = {qkvz_b}. One copy plus 47 residual "
                f"maps. A win requires the residual to be cheap. Cross-layer "
                "Pearson of per-row mean Q4 scales is ~0, so the cheap "
                "identity residual is the parent."
            ),
            "bytes_eliminated_if_true": qkvz_b - qkvz_b // 48,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "y = C_l (T x) is two GEMVs. Materializing W_l = C_l T then "
                "running incumbent Q4 is REJECTED_DENSE_REMAT and not an "
                "active-byte win."
            ),
            "physical_primitive": _require_primitive("TiledProjection"),
            "status": MEASURED_NEGATIVE,
            "cheapest_falsifier": (
                "STATIC, run here: consecutive-layer Pearson of per-row mean "
                f"qkvz scales is q={ (cl.get('q') or {}).get('mean') }, "
                f"k={(cl.get('k') or {}).get('mean')}, "
                f"v={(cl.get('v') or {}).get('mean')}, "
                f"z={(cl.get('z') or {}).get('mean')}. Unconditioned sharing "
                "of the scale field is not in the arrays. Cousin, not a "
                "launder: QN-SHARED-BASIS-DENSITY killed shared W on this "
                "MLP; QN-HEAD-REDUNDANCY killed shared GQA heads (near-"
                "orthogonal). Neither is a DN-head cosine; do not reopen as "
                "if this measurement were that scar."
            ),
            "citations": [qn_shared, qn_head],
            "index_slugs": ["shared_basis", "head_sharing"],
            "cousin_not_this_object": True,
            "measured": {"cross_layer_per_row_mean_scale_pearson": cl},
        },
        {
            "id": "generated_coefficients",
            "name": "tiny program emits qkvz / out coefficients",
            "mechanism": (
                "Store a generator G(θ, layer, block) that emits W at use. "
                "Elimination of independent storage of 48 fused QKVZ maps."
            ),
            "byte_model": (
                f"|θ| + program, independent of {qkvz_b}. A win requires "
                f"|θ| << {qkvz_b} AND production that never writes W."
            ),
            "bytes_eliminated_if_true": qkvz_b,
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "The cheap lowering is generate-then-ordinary-Q4-GEMV. That "
                "is dense rematerialization of W. A generator that IS the "
                "matvec (no W) is recurrent_state_replacement / function "
                "replacement, a different candidate."
            ),
            "physical_primitive": None,
            "status": OPEN,
            "cheapest_falsifier": (
                "STATIC: any plan whose native_execution_concept is 'emit W, "
                "then qwen_uniform_q4' is REJECTED_DENSE_REMAT before a fit."
            ),
            "index_slugs": ["generated_tied_params"],
        },
        {
            "id": "factorized_qkvz",
            "name": "low-rank / SVD / two skinny matvecs of qkvz",
            "mechanism": (
                "W_qkvz ≈ U V with rank r << min(16384, 5120). y = U (V x). "
                "Same family as NNS-016, different matrix."
            ),
            "byte_model": (
                f"r*(16384+5120)*bytes_per. Incumbent {qkvz_b} at ~4.25 bpw. "
                "A coherent r on this MLP needed 92–95% of ranks (NNS-016); "
                "DN qkvz spectrum is UNMEASURED."
            ),
            "bytes_eliminated_if_true": None,
            "bytes_eliminated_if_true_note": "Exact only after choosing r. Not claimed.",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Two skinny matvecs consume the factors. Materializing U@V "
                "into W then Q4 is REJECTED_DENSE_REMAT."
            ),
            "physical_primitive": _require_primitive("TiledProjection"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "CHEAP CPU: randomized SVD of one reconstructed qkvz "
                "(16384x5120 from HQ30UQ4, ~320 MB f32) report energy vs rank. "
                "If 99% energy needs >80% of ranks, the family dies the same "
                "way NNS-016 died on this parent's MLP. Do not quote the MLP "
                "spectrum as this matrix. Do not write U@V as production W."
            ),
            "citations": [nns016],
            "index_slugs": ["low_rank", "kronecker"],
            "cousin_not_this_object": True,
        },
        {
            "id": "conv1d_lower_bit",
            "name": "store depthwise conv1d as f16 (or Q8) instead of f32",
            "mechanism": (
                "conv1d is f32v2 [10240, 4, 1], consumed inside rearrange. "
                "z does not enter. 7.86 MB of the 2.96 GB."
            ),
            "byte_model": (
                f"payload {conv_payload} f32. f16 payload {conv_payload // 2} "
                f"+ 48*8 headers. Save {conv_payload // 2}."
            ),
            "bytes_eliminated_if_true": conv_payload // 2,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "rearrange already loads conv_w as f32. An f16 load widened "
                "in-register is the same kernel. Expanding f16 to an f32 "
                "buffer is not an active-byte win."
            ),
            "physical_primitive": _require_primitive("FusedDecodeCompute"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "CHEAP CPU: f16 round-trip of one conv1d vs f32 on real qkv "
                "activations; report conv output rel-fro. This cannot move "
                "the token by itself (0.08% of the 2.96 GB)."
            ),
            "index_slugs": [],
        },
        {
            "id": "lower_bit_recurrent_state",
            "name": "store / traffic rec_state at fewer than 32 bits",
            "mechanism": (
                "S is f32, {v_heads} x {kd} x {vd} per layer, read and written "
                "every token by gated-delta. Not a catalog tensor. A fused "
                "kernel that keeps S as f16 (or f8) in UMA, widening in "
                "register, cuts resident and RW bytes. The F4 kernel in tree "
                "is a compute-width change, not a storage change."
            ).format(v_heads=geo["value_heads"], kd=geo["key_head_dim"], vd=geo["value_head_dim"]),
            "byte_model": (
                f"resident {rec_res} f32. f16 resident {rec_res // 2}, save "
                f"{rec_res // 2} resident and {rec_rw // 2} RW/token. Not "
                "counted in the 2.96 GB bar."
            ),
            "bytes_eliminated_if_true": rec_res // 2,
            "bytes_eliminated_are": "geometry_derived_state_not_catalog",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "vi_simd already holds S in registers for the update. An f16 "
                "load is the same LocalStateMachine. Materializing f32 S "
                "from a codec every token is not an active-byte win."
            ),
            "physical_primitive": _require_primitive("LocalStateMachine"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "CHEAP CPU: run the python gated-delta step on one layer with "
                "S rounded to f16 vs f32 across a 128-token prompt; report "
                "h and S rel-fro. GPU A/B of a vi_simd_f16 sibling is a later "
                "lease. STATIC_ONLY: named, not run."
            ),
            "index_slugs": ["resident_state"],
        },
        {
            "id": "structured_transition_state",
            "name": "further-structure S (diagonal, low-rank, DPLR)",
            "mechanism": (
                "The update is already structured: "
                "S := (I - beta k k^T)(decay S) + beta k v^T. Replacing the "
                "128x128 S with a diagonal, a rank-r factor, or a DPLR SSM "
                "changes the operator. It is not a packing of the present S."
            ),
            "byte_model": (
                f"Incumbent S {rec_res} f32. Diagonal would be "
                f"{int(geo['value_heads']) * int(geo['key_head_dim']) * 4 * int(geo['n_deltanet_layers'])} "
                "and is a different mixer."
            ),
            "bytes_eliminated_if_true": rec_res
            - int(geo["value_heads"]) * int(geo["key_head_dim"]) * 4 * int(geo["n_deltanet_layers"]),
            "bytes_eliminated_are": "geometry_derived_state_not_catalog",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A diagonal/low-rank S is consumed by a different state "
                "kernel. Expanding it to a dense 128x128 then running vi_simd "
                "is not an active-byte win."
            ),
            "physical_primitive": _require_primitive("LocalStateMachine"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "STATIC: rank(S) after a long prompt is the cheap probe — if "
                "S is full-rank in practice, a rank-r store is a different "
                "model, not a compression of this one. Do not quote Mamba "
                "as this mixer."
            ),
            "index_slugs": ["state_merging"],
        },
        {
            "id": "recurrent_state_replacement",
            "name": "replace gated-delta with a cheaper state machine",
            "mechanism": (
                "Stop storing 128x128 S per value head. Use a different O(1) "
                "mixer (diagonal SSM, linear attention with a vector state, "
                "or a windowed recompute). This is function replacement of "
                "DeltaNet, not a codec of W_qkvz."
            ),
            "byte_model": (
                f"Would drop S ({rec_res} resident) and possibly reshape "
                f"qkvz ({qkvz_b}) if q/k dims change. Not a packing of the "
                "present 2.96 GB."
            ),
            "bytes_eliminated_if_true": rec_res,
            "bytes_eliminated_are": "geometry_derived_state_not_catalog",
            "dense_rematerialization": DEPENDS_ON_LOWERING,
            "dense_rematerialization_reason": (
                "A native new mixer is DIRECT_CONSUME. A lowering that "
                "emulates DeltaNet by writing dense S from a compressed code "
                "every token eliminates zero active bytes of S."
            ),
            "physical_primitive": _require_primitive("LocalStateMachine"),
            "status": OPEN,
            "capability": "UNMEASURED",
            "cheapest_falsifier": (
                "This is an architecture change. The cheapest honest probe is "
                "not a codec sweep: it is whether a vector-state mixer at "
                "matched qkvz bytes holds greedy identity on this parent. "
                "Do not launder QN-STATE-MERGING (depth merge of the present "
                "S) as this candidate."
            ),
            "index_slugs": ["qn_state_merging"],
        },
        {
            "id": "share_or_merge_state_across_depth",
            "name": "merge / share rec_state or KV across DN layers",
            "mechanism": (
                "One S (or a tied S) for several of the 48 DN layers, or a "
                "shared KV across GQA depth. Cuts resident state, not W."
            ),
            "byte_model": f"resident S {rec_res}. Sharing across 48 layers would drop 47/48 of it.",
            "bytes_eliminated_if_true": rec_res - rec_res // 48,
            "bytes_eliminated_are": "geometry_derived_state_not_catalog",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": "One S buffer, same kernel, different slot.",
            "physical_primitive": _require_primitive("LocalStateMachine"),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run: QN-STATE-MERGING on this parent "
                "(kv_state+deltanet_state), depth-state and KV merging "
                "negative under the tested conditions. Reopen is a different "
                "state topology or a longer-context regime, not a retry of "
                "shared S across these 48 layers."
            ),
            "citations": [qn_state],
            "index_slugs": ["state_merging", "qn_state_merging"],
        },
        {
            "id": "direct_state_machine",
            "name": "keep conv_state and rec_state valid in a persistent region",
            "mechanism": (
                "The DN layer is already a 7-launch sequence with resident S. "
                "A PersistentPhysicalRegion / LocalStateMachine keeps those "
                "buffers bound and occupies the GPU across tokens instead of "
                "re-encoding 337 launches. This removes launches, not catalog "
                "bytes. DISPATCH_MOTIFS already judged YES_STATE_MACHINE."
            ),
            "byte_model": (
                f"catalog bytes unchanged ({stored}). Launch count 337 → 48 "
                "in the motif census if one region per DN layer. Not a 2.96 GB lever."
            ),
            "bytes_eliminated_if_true": 0,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": "No W is written. Occupancy of the existing kernels.",
            "physical_primitive": _require_primitive("LocalStateMachine"),
            "status": OPEN,
            "cheapest_falsifier": (
                "The inner cut already in tree: FUSE_BA_DELTA removes 48 of "
                "the 337. BA_DELTA_AB is the token-identical A/B of that cut "
                "(named, not re-run here; this module is STATIC_ONLY). If 48 "
                "fewer launches do not save on the order of that receipt's "
                "class, a 7-kernel region is the wrong bet."
            ),
            "index_slugs": ["resident_state", "megakernel"],
            "citations": [
                {
                    "scar_id": "dn_layer_state_machine",
                    "source_path": DISPATCH_MOTIFS_REL,
                    "claim_refuted": "",
                    "reopen_condition": "",
                    "surface": "sealed-3.14 337 DN launches",
                    "kind": "EXISTING_JUDGMENT",
                    "this_specimen": specimen,
                }
            ],
        },
        {
            "id": "fused_update_consume",
            "name": "fold ba_to_decay into gated-delta (already in tree)",
            "mechanism": (
                "HAWKING_QWEN38_FUSE_BA_DELTA=1 binds "
                "qwen38_gated_delta_decode_vi_simd_ba, computing decay/beta "
                "in-register from ba + A_log + dt_bias. 48 launches on the "
                "628 graph. Token-identical. Default Off."
            ),
            "byte_model": (
                "Zero catalog bytes. Decay/beta workspace is no longer a "
                "stored round-trip; A_log and dt_bias (200 B each per layer) "
                "are still read."
            ),
            "bytes_eliminated_if_true": 0,
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": "Same arithmetic, one kernel, no W.",
            "physical_primitive": _require_primitive("LocalStateMachine"),
            "status": EXISTING_LEVER,
            "cheapest_falsifier": (
                "Already run: receipts/future/BA_DELTA_AB.json. 628→580 "
                "dispatches, token ids identical, zero fallbacks. STATIC_ONLY "
                "here: cited, not re-measured. Enabling it in the sealed "
                "profile is an ops act, not a new representation."
            ),
            "index_slugs": [],
            "citations": [
                {
                    "scar_id": "BA_DELTA_AB",
                    "source_path": BA_DELTA_REL,
                    "claim_refuted": "",
                    "reopen_condition": "",
                    "surface": "HAWKING_QWEN38_FUSE_BA_DELTA=1 on sealed-3.14",
                    "kind": "EXISTING_LEVER",
                    "this_specimen": specimen,
                }
            ],
        },
    ]

    have = [r["id"] for r in rows]
    if have != list(REQUIRED_CANDIDATE_IDS):
        raise DeltaNetRefuse(f"candidate catalog {have} != required {list(REQUIRED_CANDIDATE_IDS)}")

    for row in rows:
        if row["dense_rematerialization"] == REJECTED_DENSE_REMAT:
            tag = _remat_tag(True, True)
            if tag != REJECTED_DENSE_REMAT:
                raise DeltaNetRefuse(f"{row['id']}: expected REJECTED_DENSE_REMAT, got {tag}")
        if row["dense_rematerialization"] == DIRECT_CONSUME:
            if row.get("physical_primitive") not in ATLAS_PRIMITIVES:
                raise DeltaNetRefuse(f"{row['id']} missing atlas primitive")
        row["evidence_class"] = "STATIC_ONLY"
        row["gpu_authority"] = False
        slugs = list(row.get("index_slugs") or [])
        row["index_refusals"] = _index_hits(slugs) if (consult_index and slugs) else []
        if row["status"] not in {ALREADY_FALSIFIED, EXISTING_LEVER} and row.get("index_refusals"):
            row["index_hits_are_cousins"] = True
    return rows


def answers(
    acc: Mapping[str, Any],
    meas: Mapping[str, Any],
    info: Mapping[str, Any],
    cands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cands}
    return {
        "what_are_the_bytes": {
            "answer": (
                "2,961,659,904 catalog bytes: HQ30UQ4 group-64 on qkvz/ba/out "
                "(4 code bits + f16 scale per 64 weights, no bias) plus f32v2 "
                "depthwise conv1d. Most of it is codes, not headers."
            ),
            "stored_bytes": acc["stored_bytes"],
            "code_bytes": acc["code_bytes"],
            "scale_bytes": acc["scale_bytes"],
            "bias_bytes": acc["bias_bytes"],
            "header_bytes": acc["header_bytes"],
            "reconciled": acc["reconciled"],
        },
        "do_qkvz_subblocks_deserve_equal_precision": {
            "answer": (
                "NO as a default. They feed different operators. z never "
                "enters gated-delta or conv; q is a readout of S; k is the "
                "rank-1 projector and write key; v is the write value. "
                "Weight-space Q3 rel-fro is similar on all four — that is "
                "packing uniformity, not sensitivity. Heterogeneous "
                "allocation is available as a row-range native GEMV because "
                "rearrange already splits the four activations."
            ),
            "available": True,
            "status": by_id["heterogeneous_qkvz_bits"]["status"],
            "z_enters_gated_delta": False,
            "weight_space_q3_relfro": {
                s: (meas.get("by_subblock") or {}).get(s, {}).get("requant_relfro_vs_incumbent_q4", {}).get("q3")
                for s in SUBBLOCKS
            },
            "linear_qkvz_share_of_token": acc["by_organ"]["attention.linear_qkvz"]["share_of_token"],
        },
        "what_does_the_recurrent_state_need": {
            "answer": (
                "A 128x128 f32 matrix per value head, updated by a rank-1 "
                "gated delta every token, O(1) in sequence length. Depth-"
                "merging that S is already falsified (QN-STATE-MERGING). "
                "Lower-bit S, a structured (diagonal/low-rank) S, or a "
                "different mixer are OPEN and are not catalog bytes."
            ),
            "resident_bytes": acc["geometry_state"]["resident_bytes"],
            "rw_bytes_per_token": acc["geometry_state"]["rw_bytes_per_token"],
            "share_or_merge_status": by_id["share_or_merge_state_across_depth"]["status"],
        },
        "can_transforms_be_shared_across_layers": {
            "answer": (
                "Not from the Q4 scale field: consecutive-layer Pearson of "
                "per-row mean scales is ~0. Sharing W itself is unmeasured "
                "on this organ and is a cousin of QN-SHARED-BASIS (MLP) and "
                "QN-HEAD-REDUNDANCY (GQA)."
            ),
            "status": by_id["shared_transforms_across_layers"]["status"],
        },
        "is_there_an_existing_fuse_of_update_and_consume": {
            "answer": (
                "Yes. FUSE_BA_DELTA folds ba_to_decay into gated-delta, 48 "
                "launches, token-identical, zero catalog bytes. Cited from "
                "BA_DELTA_AB, not re-measured."
            ),
            "status": by_id["fused_update_consume"]["status"],
            "bytes_eliminated_if_true": 0,
        },
    }


# ---------------------------------------------------------------------------
# Snapshot / receipt.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _measured() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, geo = census_rows()
    acc = accounting_from_rows(rows)
    acc["identity"] = _identity(None, geo)
    acc["qkvz_subblocks"] = qkvz_subblock_parts(geo)
    acc["per_layer"] = _per_layer(rows, geo)
    acc["adjacent_not_in_2gb"] = _adjacent(None, geo)
    acc["geometry_state"] = geometry_state(geo)
    info = independent_information(geo)
    meas = measure_qkvz_subblocks(rows, geo)
    return acc, meas, info, geo


def snapshot(consult_index: bool = True) -> dict[str, Any]:
    acc, meas, info, geo = _measured()
    cands = candidates(acc, meas, info, geo, consult_index=consult_index)
    return {
        "accounting": acc,
        "measurements": meas,
        "independent_information": info,
        "candidates": cands,
        "answers": answers(acc, meas, info, cands),
        "geometry": geo,
    }


def build(*, consult_index: bool = True) -> Path:
    snap = snapshot(consult_index=consult_index)
    acc = snap["accounting"]
    cands = snap["candidates"]
    n_open = sum(1 for c in cands if c["status"] == OPEN)
    n_meas = sum(1 for c in cands if c["status"] == MEASURED_NEGATIVE)
    n_dead = sum(1 for c in cands if c["status"] == ALREADY_FALSIFIED)
    n_exist = sum(1 for c in cands if c["status"] == EXISTING_LEVER)
    n_remat = sum(1 for c in cands if c["dense_rematerialization"] == REJECTED_DENSE_REMAT)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Exact per-tensor DeltaNet census of sealed-3.14 (codes vs "
            "scale/bias/header), a q/k/v/z split of linear_qkvz, and an "
            "elimination catalog that does not assume those four blocks "
            "deserve equal precision."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "what_this_does_not_prove": [
            "capability of Q3 on any qkvz range (no generate gate)",
            "that weight-space rel-fro ranks gated-delta sensitivity",
            "physical EBPW of a different packing",
            "actual_read_bytes_per_token (cache, contention)",
            "that FUSE_BA_DELTA's TPS delta is a hardware claim of this receipt",
        ],
        "accounting": _py(acc),
        "measurements": _py(snap["measurements"]),
        "independent_information": _py(snap["independent_information"]),
        "candidates": _py(cands),
        "answers": _py(snap["answers"]),
        "candidate_counts": {
            "n": len(cands),
            "open": n_open,
            "measured_negative": n_meas,
            "already_falsified": n_dead,
            "existing_lever": n_exist,
            "rejected_dense_remat": n_remat,
        },
        "open_byte_levers": [
            {
                "id": c["id"],
                "bytes_eliminated_if_true": c.get("bytes_eliminated_if_true"),
                "status": c["status"],
            }
            for c in cands
            if c["status"] == OPEN and c.get("bytes_eliminated_if_true")
        ],
        "recovered_implementation": {
            "catalog_format": "HQ38M20 + HQ30UQ4 (qkvz/ba/out) + f32v2 (conv1d)",
            "artifact_root": acc["identity"]["artifact_root"],
            "qkvz_layout": acc["identity"]["geometry"]["qkvz_layout"],
            "gated_delta": "S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q",
            "z_binding": "gated RMSNorm only; not gated-delta",
        },
        "gaps_closed": [
            "per-tensor DeltaNet bytes summed from HQ30UQ4/f32v2 headers and refused unless they equal 2,961,659,904",
            "codes vs scale vs header split; bias is zero on HQ30UQ4",
            "q/k/v/z row ranges of the fused QKVZ cover 16384 rows and reconcile to linear_qkvz",
            "z is source-true not-in-gated-delta; heterogeneous bits are available as a row-range GEMV",
            "weight-space Q3 rel-fro measured on all four blocks so equal packing is not mistaken for equal sensitivity",
            "geometry rec/conv state reported separately and not added to the 2.96 GB bar",
            "negative_index queried; NNS-019 / QN-STATE-MERGING cited on this organ; MLP/GQA scars marked cousins",
        ],
        "negative_findings": [
            "linear_qkvz is 21.65% of the token and is one fused Q4 GEMV, not four tensors",
            "q/k/v/z W-space Q3 rel-fro is ~0.22 on all four; packing does not pick a winner",
            "cross-layer Pearson of qkvz per-row mean scales is ~0; sharing W from the scale field is measured negative",
            "Gravity families on DeltaNet in/out are NNS-019 dead",
            "depth-merging rec_state is QN-STATE-MERGING dead",
            "FUSE_BA_DELTA already folds update+consume for decay/beta; it eliminates launches, not the 2.96 GB",
            "headers are 6,144 bytes of the 2.96 GB — not the prize",
        ],
        "nomenclature": {
            "already_falsified": ALREADY_FALSIFIED,
            "measured_negative": MEASURED_NEGATIVE,
            "open": OPEN,
            "existing_lever": EXISTING_LEVER,
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
    parser.add_argument("--census-only", action="store_true")
    args = parser.parse_args(argv_list)
    if args.census_only:
        snap = accounting()
        json.dump(
            {
                "stored_bytes": snap["stored_bytes"],
                "code_bytes": snap["code_bytes"],
                "scale_bytes": snap["scale_bytes"],
                "header_bytes": snap["header_bytes"],
                "bias_bytes": snap["bias_bytes"],
                "reconciled": snap["reconciled"],
                "by_organ": {
                    k: v["stored_bytes"] for k, v in snap["by_organ"].items()
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
