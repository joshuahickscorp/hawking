#!/usr/bin/env python3
"""Read-only DeepSeek-V4-Flash tensor schedule derived from the sealed manifest.

This module never writes under a ``.gravity`` artifact, never opens a chunk
body, and never performs a forward.  It streams ``manifest.json`` tensor
descriptors (name/dtype/shape/bytes only) and classifies every tensor into an
organ class so a later capture/runtime lane can size a memory-bounded stream
and an activation-weighted fit without re-deriving the architecture.
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "hawking.dsv4f.tensor_schedule.v1"
EXPECTED_TENSOR_COUNT = 69_187
EXPECTED_TOTAL_TENSOR_BYTES = 159_609_485_896
EXPECTED_CHUNK_SHA256 = (
    "15e00fb1b91ac074b7f24686de4e289f76d66eb1c3fb4ad643de027adc78ca13"
)
EXPECTED_MANIFEST_SEAL_PREFIX = "ba9039bfe71328e2"
PINNED_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
PINNED_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

BASE_LAYER_COUNT = 43
MTP_LAYER = 43
ROUTED_EXPERTS = 256
TOP_K = 6
HIDDEN = 4096
VOCAB = 129_280
Q_LORA = 1024
O_LORA = 1024
O_GROUPS = 8
HEAD_DIM = 512
N_HEADS = 64
MOE_INTER = 2048
HC_MULT = 4
HASH_LAYER_COUNT = 3
TARGET_COMPLETE_BPW = 1.5
EXPERT_BPW_GRID = (1.0, 1.2, 1.3, 1.4, 1.4609)

# Official inference/config.json compress_ratios (43 base + 1 MTP).
COMPRESS_RATIOS: tuple[int, ...] = (
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 0,
)

DTYPE_WIDTH: dict[str, int] = {
    "BF16": 2,
    "F32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
}

# Manifest native_decode_validation buckets mapped onto stored dtypes.
SOURCE_DTYPE_FAMILY: dict[str, str] = {
    "F8_E4M3": "fp8_e4m3",
    "I8": "fp4_e2m1fn_x2_packed_i8",
    "F8_E8M0": "scale_ue8m0",
    "BF16": "bf16",
    "F32": "f32",
    "I64": "i64",
}

ORGAN_CLASSES: tuple[str, ...] = (
    "routed_expert",
    "shared_expert",
    "mla",
    "mhc",
    "indexer_compressor",
    "hash_layers",
    "embeddings",
    "lm_head",
    "norms",
    "router_gate",
    "other",
)

_ARTIFACT_CANDIDATES = (
    Path("workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
    Path(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
        "deepseek-v4/full-43-layer-stream.gravity"
    ),
)

_RE_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_RE_MTP = re.compile(r"^mtp\.(\d+)\.(.+)$")
_RE_ROUTED = re.compile(
    r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_RE_SHARED = re.compile(r"^ffn\.shared_experts\.(w[123])\.(weight|scale)$")
_RE_MLA_PAIR = re.compile(r"^attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.(weight|scale)$")
_RE_MHC = re.compile(r"^hc_(attn|ffn|head)_(fn|base|scale)$")
_RE_INDEXER = re.compile(r"^attn\.indexer\.(.+)$")
_RE_COMPRESSOR = re.compile(r"^attn\.compressor\.(.+)$")


@dataclass(frozen=True)
class TensorRow:
    name: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int


@dataclass
class Classified:
    organ: str
    subrole: str
    layer: int | None
    is_scale: bool
    is_weight: bool
    logical_params: int
    stored_elements: int
    notes: str = ""


@dataclass
class OrganAcc:
    tensor_count: int = 0
    byte_mass: int = 0
    logical_params: int = 0
    stored_elements: int = 0
    scale_bytes: int = 0
    weight_bytes: int = 0
    dtypes: Counter[str] = field(default_factory=Counter)
    subroles: dict[str, dict[str, int]] = field(default_factory=dict)
    names_if_other: list[str] = field(default_factory=list)

    def add(self, row: TensorRow, cls: Classified) -> None:
        self.tensor_count += 1
        self.byte_mass += row.bytes
        self.logical_params += cls.logical_params
        self.stored_elements += cls.stored_elements
        self.dtypes[row.dtype] += 1
        if cls.is_scale:
            self.scale_bytes += row.bytes
        if cls.is_weight:
            self.weight_bytes += row.bytes
        bucket = self.subroles.setdefault(
            cls.subrole,
            {
                "tensor_count": 0,
                "byte_mass": 0,
                "logical_params": 0,
                "stored_elements": 0,
            },
        )
        bucket["tensor_count"] += 1
        bucket["byte_mass"] += row.bytes
        bucket["logical_params"] += cls.logical_params
        bucket["stored_elements"] += cls.stored_elements
        if cls.organ == "other":
            self.names_if_other.append(row.name)


def resolve_artifact_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"no manifest.json under {path}")
        return path.resolve()
    here = Path(__file__).resolve()
    repo_relative = here.parents[2] / _ARTIFACT_CANDIDATES[0]
    for candidate in (repo_relative, *_ARTIFACT_CANDIDATES):
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "sealed DSV4F artifact not found; looked in "
        + ", ".join(str(c) for c in (repo_relative, *_ARTIFACT_CANDIDATES))
    )


def _read_u64(buf: bytes, start: int) -> tuple[int, int]:
    end = start
    n = len(buf)
    while end < n and 48 <= buf[end] <= 57:
        end += 1
    if end == start:
        raise ValueError("expected integer")
    return int(buf[start:end]), end


def _skip_json_value(buf: bytes, pos: int) -> int:
    """Skip one JSON value starting at pos. Used only for the trailing fields."""
    n = len(buf)
    ch = buf[pos]
    if ch == 34:  # "
        pos += 1
        while pos < n:
            if buf[pos] == 92:  # backslash
                pos += 2
                continue
            if buf[pos] == 34:
                return pos + 1
            pos += 1
        raise ValueError("unterminated string")
    if ch == 123:  # {
        depth = 1
        pos += 1
        in_str = False
        while pos < n and depth:
            c = buf[pos]
            if in_str:
                if c == 92:
                    pos += 2
                    continue
                if c == 34:
                    in_str = False
                pos += 1
                continue
            if c == 34:
                in_str = True
                pos += 1
                continue
            if c == 123:
                depth += 1
            elif c == 125:
                depth -= 1
            pos += 1
        return pos
    if ch == 91:  # [
        depth = 1
        pos += 1
        in_str = False
        while pos < n and depth:
            c = buf[pos]
            if in_str:
                if c == 92:
                    pos += 2
                    continue
                if c == 34:
                    in_str = False
                pos += 1
                continue
            if c == 34:
                in_str = True
                pos += 1
                continue
            if c == 91:
                depth += 1
            elif c == 93:
                depth -= 1
            pos += 1
        return pos
    if ch in (45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57):
        if ch == 45:
            pos += 1
        while pos < n and 48 <= buf[pos] <= 57:
            pos += 1
        return pos
    if buf.startswith(b"true", pos):
        return pos + 4
    if buf.startswith(b"false", pos):
        return pos + 5
    if buf.startswith(b"null", pos):
        return pos + 4
    raise ValueError(f"cannot skip JSON at {pos}")


def iter_manifest_tensors(manifest_path: Path) -> Iterator[TensorRow]:
    """Yield name/dtype/shape/bytes for every tensor. Never reads chunk bodies."""

    with manifest_path.open("rb") as handle:
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = mapping.find(b'"tensors":{')
            if start < 0:
                raise ValueError("manifest has no tensors object")
            pos = start + len(b'"tensors":{')
            buf = mapping
            n = len(buf)
            while pos < n:
                if buf[pos] == 125:  # end of tensors
                    return
                if buf[pos] == 44:  # comma between entries
                    pos += 1
                    continue
                if buf[pos] != 34:
                    raise ValueError(f"expected tensor key at {pos}")
                key_end = buf.find(b'"', pos + 1)
                if key_end < 0:
                    raise ValueError("unterminated tensor key")
                name = buf[pos + 1 : key_end].decode("ascii")
                pos = key_end + 1
                if buf[pos] != 58 or buf[pos + 1] != 123:
                    raise ValueError(f"expected object for {name}")
                pos += 2
                fields: dict[str, Any] = {}
                while pos < n and buf[pos] != 125:
                    if buf[pos] == 44:
                        pos += 1
                        continue
                    if buf[pos] != 34:
                        raise ValueError(f"expected field key inside {name}")
                    f_end = buf.find(b'"', pos + 1)
                    field = buf[pos + 1 : f_end].decode("ascii")
                    pos = f_end + 1
                    if buf[pos] != 58:
                        raise ValueError(f"expected colon after {field}")
                    pos += 1
                    if field == "bytes":
                        value, pos = _read_u64(buf, pos)
                        fields["bytes"] = value
                    elif field == "dtype":
                        if buf[pos] != 34:
                            raise ValueError("dtype is not a string")
                        d_end = buf.find(b'"', pos + 1)
                        fields["dtype"] = buf[pos + 1 : d_end].decode("ascii")
                        pos = d_end + 1
                    elif field == "name":
                        if buf[pos] != 34:
                            raise ValueError("name is not a string")
                        n_end = buf.find(b'"', pos + 1)
                        fields["name"] = buf[pos + 1 : n_end].decode("ascii")
                        pos = n_end + 1
                    elif field == "shape":
                        if buf[pos] != 91:
                            raise ValueError("shape is not an array")
                        pos += 1
                        dims: list[int] = []
                        while buf[pos] != 93:
                            if buf[pos] == 44:
                                pos += 1
                                continue
                            dim, pos = _read_u64(buf, pos)
                            dims.append(dim)
                        pos += 1
                        fields["shape"] = tuple(dims)
                    else:
                        pos = _skip_json_value(buf, pos)
                if pos >= n or buf[pos] != 125:
                    raise ValueError(f"unterminated tensor object {name}")
                pos += 1
                dtype = fields.get("dtype")
                shape = fields.get("shape")
                nbytes = fields.get("bytes")
                if not isinstance(dtype, str) or not isinstance(shape, tuple) or not isinstance(nbytes, int):
                    raise ValueError(f"incomplete descriptor for {name}: {fields!r}")
                if fields.get("name") not in (None, name):
                    raise ValueError(f"key/name mismatch {name} vs {fields.get('name')}")
                yield TensorRow(name=name, dtype=dtype, shape=shape, bytes=nbytes)
        finally:
            mapping.close()


def extract_manifest_identity(manifest_path: Path) -> dict[str, Any]:
    """Pull sealed header scalars without parsing the 69k tensor bodies."""

    data = manifest_path.read_bytes()
    identity: dict[str, Any] = {
        "path": str(manifest_path),
        "manifest_bytes": len(data),
    }
    patterns = {
        "schema": rb'"schema":"(hawking\.gravity\.deepseek_v4\.full_stream\.v1)"',
        "seal_sha256": rb'"schema":"hawking\.gravity\.deepseek_v4\.full_stream\.v1","seal_sha256":"([0-9a-f]{64})"',
        "total_tensor_bytes": rb'"total_tensor_bytes":(\d+)',
        "content_addressed_chunk_sha256": rb'"content_addressed_chunk_sha256":"([0-9a-f]{64})"',
        "content_addressed_chunk_count": rb'"content_addressed_chunk_count":(\d+)',
        "tensor_count": rb'"full_model_scope":\{[^}]*"tensor_count":(\d+)',
        "repository": rb'"repository":"(deepseek-ai/DeepSeek-V4-Flash)"',
        "revision": rb'"revision":"([0-9a-f]{40})"',
        "status": rb'"status":"(FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY)"',
    }
    text = data  # already bytes
    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match is None:
            identity[key] = None
            continue
        raw = match.group(1).decode("ascii")
        identity[key] = int(raw) if raw.isdigit() else raw
    # Prefer the full_model_scope tensor_count; fall back to architecture.
    if identity.get("tensor_count") is None:
        match = re.search(rb'"architecture":\{[^}]*"tensor_count":(\d+)', text)
        if match:
            identity["tensor_count"] = int(match.group(1))
    return identity


def stored_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def expected_bytes(dtype: str, shape: tuple[int, ...]) -> int | None:
    width = DTYPE_WIDTH.get(dtype)
    if width is None:
        return None
    return stored_elements(shape) * width


def logical_params_for(row: TensorRow, is_scale: bool) -> int:
    elems = stored_elements(row.shape)
    if is_scale or row.dtype == "F8_E8M0":
        return 0
    if row.dtype == "I64":
        return 0
    if row.dtype == "I8":
        return elems * 2
    if row.dtype in {"F8_E4M3", "BF16", "F32"}:
        return elems
    return 0


def classify_tensor(row: TensorRow) -> Classified:
    """Assign exactly one organ class. Unknown names fall through to other."""

    elems = stored_elements(row.shape)
    name = row.name

    if name == "embed.weight":
        return Classified("embeddings", "embed.weight", None, False, True, elems, elems)
    if name == "head.weight":
        return Classified("lm_head", "head.weight", None, False, True, elems, elems)
    if name == "norm.weight":
        return Classified("norms", "final_norm", None, False, True, elems, elems)
    if name in {"hc_head_fn", "hc_head_base", "hc_head_scale"}:
        sub = name[len("hc_head_") :]
        return Classified(
            "mhc",
            f"hc_head_{sub}",
            None,
            False,
            name.endswith("_fn"),
            elems if row.dtype != "F8_E8M0" else 0,
            elems,
        )

    layer_match = _RE_LAYER.match(name)
    mtp_match = _RE_MTP.match(name)
    if layer_match is not None:
        layer = int(layer_match.group(1))
        rest = layer_match.group(2)
    elif mtp_match is not None:
        layer = BASE_LAYER_COUNT + int(mtp_match.group(1))
        rest = mtp_match.group(2)
    else:
        return Classified(
            "other",
            "unprefixed",
            None,
            False,
            False,
            logical_params_for(row, False),
            elems,
        )

    routed = _RE_ROUTED.match(rest)
    if routed:
        _expert, proj, kind = routed.groups()
        is_scale = kind == "scale"
        return Classified(
            "routed_expert",
            proj,
            layer,
            is_scale,
            not is_scale,
            logical_params_for(row, is_scale),
            elems,
        )

    shared = _RE_SHARED.match(rest)
    if shared:
        proj, kind = shared.groups()
        is_scale = kind == "scale"
        return Classified(
            "shared_expert",
            proj,
            layer,
            is_scale,
            not is_scale,
            logical_params_for(row, is_scale),
            elems,
        )

    if rest == "ffn.gate.tid2eid":
        return Classified("hash_layers", "tid2eid", layer, False, False, 0, elems, "i64 lookup, not a linear")
    if rest == "ffn.gate.weight":
        return Classified("router_gate", "gate.weight", layer, False, True, elems, elems)
    if rest == "ffn.gate.bias":
        return Classified("router_gate", "gate.bias", layer, False, True, elems, elems)

    mla = _RE_MLA_PAIR.match(rest)
    if mla:
        proj, kind = mla.groups()
        is_scale = kind == "scale"
        if proj in {"wq_a", "wq_b"}:
            family = "q_lora"
        elif proj == "wkv":
            family = "kv_lora"
        else:
            family = "o_proj"
        return Classified(
            "mla",
            f"{family}.{proj}.{kind}",
            layer,
            is_scale,
            not is_scale,
            logical_params_for(row, is_scale),
            elems,
        )

    if rest == "attn.attn_sink":
        return Classified("mla", "attn_sink", layer, False, True, elems, elems)

    mhc = _RE_MHC.match(rest)
    if mhc:
        stage, piece = mhc.groups()
        return Classified(
            "mhc",
            f"hc_{stage}_{piece}",
            layer,
            False,
            piece == "fn",
            elems,
            elems,
        )

    indexer = _RE_INDEXER.match(rest)
    if indexer:
        tail = indexer.group(1)
        is_scale = tail.endswith(".scale")
        return Classified(
            "indexer_compressor",
            f"indexer.{tail}",
            layer,
            is_scale,
            (not is_scale) and tail.endswith(".weight"),
            logical_params_for(row, is_scale),
            elems,
        )

    compressor = _RE_COMPRESSOR.match(rest)
    if compressor:
        tail = compressor.group(1)
        is_scale = tail.endswith(".scale")
        return Classified(
            "indexer_compressor",
            f"compressor.{tail}",
            layer,
            is_scale,
            (not is_scale) and tail.endswith(".weight"),
            logical_params_for(row, is_scale),
            elems,
        )

    if rest in {
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        "enorm.weight",
        "hnorm.weight",
        "norm.weight",
    }:
        return Classified("norms", rest, layer, False, True, elems, elems)

    if rest in {"e_proj.weight", "h_proj.weight", "e_proj.scale", "h_proj.scale"}:
        is_scale = rest.endswith(".scale")
        return Classified(
            "other",
            f"mtp.{rest}",
            layer,
            is_scale,
            not is_scale,
            logical_params_for(row, is_scale),
            elems,
            "MTP auxiliary projection",
        )

    return Classified(
        "other",
        rest,
        layer,
        row.dtype == "F8_E8M0",
        False,
        logical_params_for(row, row.dtype == "F8_E8M0"),
        elems,
        "unrecognized suffix",
    )


def compression_mode(layer: int) -> str:
    if layer < 0 or layer >= len(COMPRESS_RATIOS):
        return "unknown"
    ratio = COMPRESS_RATIOS[layer]
    if ratio == 0:
        return "sliding_window_only"
    if ratio == 4:
        return "ratio_4_with_indexer"
    if ratio == 128:
        return "ratio_128"
    return f"unsupported_ratio_{ratio}"


def gate_mode(layer: int) -> str:
    if layer < HASH_LAYER_COUNT:
        return "hash_token_id_to_expert_ids"
    return "learned_scores_with_selection_bias"


def classify_all(rows: Iterator[TensorRow]) -> dict[str, Any]:
    organs = {name: OrganAcc() for name in ORGAN_CLASSES}
    layer_bytes: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "tensor_count": 0,
            "all_bytes": 0,
            "routed_expert_bytes": 0,
            "shared_expert_bytes": 0,
            "mla_bytes": 0,
            "mhc_bytes": 0,
            "indexer_compressor_bytes": 0,
            "norms_bytes": 0,
            "router_gate_bytes": 0,
            "hash_layers_bytes": 0,
            "other_bytes": 0,
            "one_routed_expert_bytes": 0,
        }
    )
    # Per-layer one-expert accumulator: first-seen expert id's bytes, or mean.
    per_layer_expert_bytes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    scope_bytes = {"base": 0, "mtp": 0, "global": 0}
    scope_count = {"base": 0, "mtp": 0, "global": 0}
    dtype_acc: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "byte_mass": 0, "logical_params": 0, "stored_elements": 0}
    )
    undetermined: list[dict[str, Any]] = []
    byte_sum = 0
    count = 0
    shape_mismatch = 0

    for row in rows:
        count += 1
        byte_sum += row.bytes
        cls = classify_tensor(row)
        want = expected_bytes(row.dtype, row.shape)
        if want is None or want != row.bytes:
            undetermined.append(
                {
                    "name": row.name,
                    "dtype": row.dtype,
                    "shape": list(row.shape),
                    "bytes": row.bytes,
                    "expected_bytes": want,
                    "reason": "unknown_dtype" if want is None else "shape_dtype_bytes_mismatch",
                }
            )
            shape_mismatch += 1
        organs[cls.organ].add(row, cls)
        if cls.layer is None:
            scope = "global"
        elif cls.layer >= BASE_LAYER_COUNT:
            scope = "mtp"
        else:
            scope = "base"
        scope_bytes[scope] += row.bytes
        scope_count[scope] += 1
        dacc = dtype_acc[row.dtype]
        dacc["tensor_count"] += 1
        dacc["byte_mass"] += row.bytes
        dacc["logical_params"] += cls.logical_params
        dacc["stored_elements"] += cls.stored_elements
        if cls.layer is not None:
            lb = layer_bytes[cls.layer]
            lb["tensor_count"] += 1
            lb["all_bytes"] += row.bytes
            key = {
                "routed_expert": "routed_expert_bytes",
                "shared_expert": "shared_expert_bytes",
                "mla": "mla_bytes",
                "mhc": "mhc_bytes",
                "indexer_compressor": "indexer_compressor_bytes",
                "norms": "norms_bytes",
                "router_gate": "router_gate_bytes",
                "hash_layers": "hash_layers_bytes",
                "other": "other_bytes",
            }.get(cls.organ)
            if key:
                lb[key] += row.bytes
            routed = _RE_LAYER.match(row.name)
            if routed:
                rest = routed.group(2)
                em = _RE_ROUTED.match(rest)
                if em:
                    per_layer_expert_bytes[cls.layer][int(em.group(1))] += row.bytes

    for layer, experts in per_layer_expert_bytes.items():
        if experts:
            # All 256 experts share geometry; use the exact first-expert mass.
            layer_bytes[layer]["one_routed_expert_bytes"] = experts[min(experts)]
            layer_bytes[layer]["routed_expert_ids"] = len(experts)

    return {
        "organs": organs,
        "layer_bytes": dict(layer_bytes),
        "dtype_acc": dict(dtype_acc),
        "undetermined": undetermined,
        "byte_sum": byte_sum,
        "count": count,
        "shape_mismatch": shape_mismatch,
        "scope_bytes": scope_bytes,
        "scope_count": scope_count,
    }


def _pct(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (100.0 * part) / whole


def _bpw(bits_or_bytes: int, params: int, *, from_bytes: bool = True) -> float | None:
    if params <= 0:
        return None
    bits = bits_or_bytes * 8 if from_bytes else bits_or_bytes
    return bits / params


def required_nonexpert_bpw(f_expert: float, expert_bpw: float, target: float = TARGET_COMPLETE_BPW) -> dict[str, Any]:
    f_ne = 1.0 - f_expert
    if f_ne <= 0:
        return {
            "expert_bpw": expert_bpw,
            "required_nonexpert_bpw": None,
            "complete_bpw_if_nonexpert_zero": f_expert * expert_bpw,
            "feasible": f_expert * expert_bpw <= target,
            "note": "non-expert fraction is zero",
        }
    required = (target - f_expert * expert_bpw) / f_ne
    return {
        "expert_bpw": expert_bpw,
        "required_nonexpert_bpw": required,
        "complete_bpw_at_required": target,
        "feasible": required >= 0,
        "note": (
            "non-experts must be *below* source width"
            if 0 <= required < 16
            else (
                "experts already exhaust the 1.5 complete budget"
                if required < 0
                else "non-experts may stay at or above native 16-bit"
            )
        ),
    }


def build_schedule(
    *,
    artifact_root: Path,
    identity: dict[str, Any],
    classified: dict[str, Any],
) -> dict[str, Any]:
    organs: dict[str, OrganAcc] = classified["organs"]
    total_bytes = classified["byte_sum"]
    total_count = classified["count"]
    total_logical = sum(acc.logical_params for acc in organs.values())
    total_stored = sum(acc.stored_elements for acc in organs.values())
    expert_logical = organs["routed_expert"].logical_params + organs["shared_expert"].logical_params
    nonexpert_logical = total_logical - expert_logical
    f_expert = (expert_logical / total_logical) if total_logical else 0.0
    f_nonexpert = 1.0 - f_expert
    expert_bytes = organs["routed_expert"].byte_mass + organs["shared_expert"].byte_mass
    nonexpert_bytes = total_bytes - expert_bytes

    organ_table = {}
    for name in ORGAN_CLASSES:
        acc = organs[name]
        organ_table[name] = {
            "tensor_count": acc.tensor_count,
            "byte_mass": acc.byte_mass,
            "byte_pct": _pct(acc.byte_mass, total_bytes),
            "logical_params": acc.logical_params,
            "logical_param_pct": _pct(acc.logical_params, total_logical),
            "stored_elements": acc.stored_elements,
            "stored_element_pct": _pct(acc.stored_elements, total_stored),
            "weight_bytes": acc.weight_bytes,
            "scale_bytes": acc.scale_bytes,
            "source_bpw_vs_logical": _bpw(acc.byte_mass, acc.logical_params),
            "source_bpw_vs_stored": _bpw(acc.byte_mass, acc.stored_elements),
            "dtypes": dict(acc.dtypes),
            "subroles": acc.subroles,
            "enumerated_names": list(acc.names_if_other) if name == "other" else None,
        }

    dtype_table = {}
    for dtype, acc in sorted(classified["dtype_acc"].items()):
        dtype_table[dtype] = {
            **acc,
            "family": SOURCE_DTYPE_FAMILY.get(dtype, "unknown"),
            "byte_pct": _pct(acc["byte_mass"], total_bytes),
            "width_bytes": DTYPE_WIDTH.get(dtype),
        }

    envelope_rows = [
        required_nonexpert_bpw(f_expert, expert_bpw) for expert_bpw in EXPERT_BPW_GRID
    ]

    # Per-token served reconstruction (decode, 1 token) from classified masses.
    # Teacher MoE prior used weight-only (no scales) + gate.weight.
    one_layer_ref = 4  # a typical ratio-4 learned layer
    lb4 = classified["layer_bytes"].get(one_layer_ref, {})
    # w1/w3: [2048,2048] I8; w2: [4096,1024] I8. Each stored proj is 4,194,304 bytes.
    one_expert_weight_bytes = 3 * (2048 * 2048)
    one_expert_scale_bytes = 3 * (2048 * 128)
    six_expert_weight_bytes = TOP_K * one_expert_weight_bytes
    shared_weight_bytes = 2 * (2048 * 4096) + (4096 * 2048)  # w1/w3 + w2 F8
    gate_weight_bytes = ROUTED_EXPERTS * HIDDEN * 2
    moe_teacher_prior_style = six_expert_weight_bytes + shared_weight_bytes + gate_weight_bytes
    mla_weight_bytes = (
        (Q_LORA * HIDDEN)  # wq_a
        + (N_HEADS * HEAD_DIM * Q_LORA)  # wq_b
        + (HEAD_DIM * HIDDEN)  # wkv
        + ((N_HEADS * HEAD_DIM // O_GROUPS) * (O_GROUPS * O_LORA))  # wo_a
        + (HIDDEN * (O_GROUPS * O_LORA))  # wo_b
    )
    mla_plus_shared_plus_one_expert = mla_weight_bytes + shared_weight_bytes + one_expert_weight_bytes

    layers_out = []
    peak_full = 0
    peak_streamed = 0
    peak_full_layer = -1
    peak_streamed_layer = -1
    for layer in range(BASE_LAYER_COUNT + 1):
        lb = classified["layer_bytes"].get(layer)
        if lb is None:
            continue
        one = lb.get("one_routed_expert_bytes", 0)
        streamed = (
            lb["all_bytes"]
            - lb["routed_expert_bytes"]
            + TOP_K * one
        )
        attn_phase = (
            lb["mla_bytes"]
            + lb["indexer_compressor_bytes"]
            + lb["mhc_bytes"]  # both stages; attn-only would be ~half
            + lb["norms_bytes"]
            + lb["hash_layers_bytes"]
        )
        # Honest attn-phase excludes ffn-stage mHC/norms/gate; report both.
        ffn_phase = (
            lb["router_gate_bytes"]
            + lb["hash_layers_bytes"]
            + lb["shared_expert_bytes"]
            + TOP_K * one
            + lb["mhc_bytes"]
            + lb["norms_bytes"]
            + lb["other_bytes"]
        )
        fifo_peak = max(attn_phase, ffn_phase)
        if lb["all_bytes"] > peak_full:
            peak_full = lb["all_bytes"]
            peak_full_layer = layer
        if streamed > peak_streamed:
            peak_streamed = streamed
            peak_streamed_layer = layer
        layers_out.append(
            {
                "layer": layer,
                "scope": "mtp_auxiliary" if layer == MTP_LAYER else "base",
                "compression": compression_mode(layer),
                "gate": gate_mode(layer),
                "tensor_count": lb["tensor_count"],
                "full_layer_resident_bytes": lb["all_bytes"],
                "non_expert_bytes": lb["all_bytes"] - lb["routed_expert_bytes"],
                "one_routed_expert_bytes": one,
                "six_routed_expert_bytes": TOP_K * one,
                "shared_expert_bytes": lb["shared_expert_bytes"],
                "mla_bytes": lb["mla_bytes"],
                "indexer_compressor_bytes": lb["indexer_compressor_bytes"],
                "mhc_bytes": lb["mhc_bytes"],
                "router_gate_bytes": lb["router_gate_bytes"],
                "hash_layers_bytes": lb["hash_layers_bytes"],
                "norms_bytes": lb["norms_bytes"],
                "other_bytes": lb["other_bytes"],
                "streamed_decode_peak_bytes": streamed,
                "attn_plus_controls_phase_bytes": attn_phase,
                "ffn_selected_expert_phase_bytes": ffn_phase,
                "stage_fifo_peak_bytes": fifo_peak,
                "resident_together_to_execute": _resident_set(layer),
            }
        )

    x_capture = _x_capture_organs(f_expert=f_expert, organ_table=organ_table)

    coverage = {
        "tensor_count": total_count,
        "expected_tensor_count": EXPECTED_TENSOR_COUNT,
        "class_count_sum": sum(acc.tensor_count for acc in organs.values()),
        "unclassified": 0,
        "byte_mass_sum": total_bytes,
        "expected_total_tensor_bytes": identity.get("total_tensor_bytes")
        or EXPECTED_TOTAL_TENSOR_BYTES,
        "byte_residual": total_bytes
        - int(identity.get("total_tensor_bytes") or EXPECTED_TOTAL_TENSOR_BYTES),
        "logical_params_unpacked": total_logical,
        "stored_elements": total_stored,
        "shape_dtype_undetermined": classified["undetermined"],
        "other_enumerated": organs["other"].names_if_other,
        "scope_tensor_count": classified["scope_count"],
        "scope_byte_mass": classified["scope_bytes"],
        "covers_all_tensors": (
            total_count == EXPECTED_TENSOR_COUNT
            and sum(acc.tensor_count for acc in organs.values()) == total_count
            and total_bytes
            == int(identity.get("total_tensor_bytes") or EXPECTED_TOTAL_TENSOR_BYTES)
        ),
    }

    return {
        "schema": SCHEMA,
        "status": "ANALYSIS_ONLY_NOT_A_RUNTIME",
        "artifact": {
            **identity,
            "artifact_root": str(artifact_root),
            "read_only": True,
            "chunks_opened": 0,
            "expected_chunk_sha256": EXPECTED_CHUNK_SHA256,
            "expected_seal_prefix": EXPECTED_MANIFEST_SEAL_PREFIX,
            "seal_matches_contract_prefix": str(identity.get("seal_sha256") or "").startswith(
                EXPECTED_MANIFEST_SEAL_PREFIX
            ),
            "chunk_sha256_matches_contract": identity.get("content_addressed_chunk_sha256")
            == EXPECTED_CHUNK_SHA256,
        },
        "architecture": {
            "model_type": "deepseek_v4",
            "base_layers": BASE_LAYER_COUNT,
            "mtp_layers": 1,
            "hidden_size": HIDDEN,
            "vocab_size": VOCAB,
            "routed_experts": ROUTED_EXPERTS,
            "shared_experts": 1,
            "top_k": TOP_K,
            "hash_layers": HASH_LAYER_COUNT,
            "q_lora_rank": Q_LORA,
            "o_lora_rank": O_LORA,
            "o_groups": O_GROUPS,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "moe_intermediate": MOE_INTER,
            "hc_mult": HC_MULT,
            "wkv": [HEAD_DIM, HIDDEN],
            "compress_ratios": list(COMPRESS_RATIOS),
            "source_dtypes": "fp8 e4m3 control + fp4 e2m1 packed experts + ue8m0 scales",
            "mtp_name_prefix": "mtp.0.",
            "mtp_is_not_layers_43": True,
            "byte_auction_158p07b_is_stored_elements": True,
        },
        "coverage": coverage,
        "organs": organ_table,
        "dtype_breakdown": dtype_table,
        "bpw_feasibility_envelope": {
            "formula": "complete_bpw = f_expert * expert_bpw + f_nonexpert * nonexpert_bpw",
            "target_complete_bpw": TARGET_COMPLETE_BPW,
            "definitions": {
                "logical_params": (
                    "Unpacked source parameters: I8 FP4-packed weights count as "
                    "2 logical elements per stored byte (native_k = 2 * packed_k). "
                    "F8_E4M3/BF16/F32 weights count prod(shape). UE8M0 scales and "
                    "I64 tid2eid tables are sidecar bytes, not logical params."
                ),
                "f_expert": "logical params of routed_expert + shared_expert weights / all logical params",
                "f_nonexpert": "1 - f_expert",
                "expert_bpw": "bits per logical expert parameter (scales billed separately in source_effective)",
                "stored_element_bpw": "8 * bytes / prod(shape) across the organ; this is the 8.08-class figure",
            },
            "f_expert": f_expert,
            "f_nonexpert": f_nonexpert,
            "expert_logical_params": expert_logical,
            "nonexpert_logical_params": nonexpert_logical,
            "expert_byte_mass": expert_bytes,
            "nonexpert_byte_mass": nonexpert_bytes,
            "source_expert_bpw_logical": _bpw(expert_bytes, expert_logical),
            "source_nonexpert_bpw_logical": _bpw(nonexpert_bytes, nonexpert_logical),
            "source_complete_bpw_logical": _bpw(total_bytes, total_logical),
            "source_complete_bpw_stored_elements": _bpw(total_bytes, total_stored),
            "source_complete_bpw_claimed_158p07b": _bpw(total_bytes, 158_070_000_000),
            "table": envelope_rows,
            "scale_sidecar_note": (
                "Source expert_bytes already include UE8M0 scales. The envelope "
                "table treats expert_bpw as the complete expert rate a child "
                "would bill (weights+scales+codebooks) per logical expert param."
            ),
        },
        "per_layer_streaming": {
            "execution_order": [
                "embed.weight (once; evict after expanding hc_mult copies)",
                "for layer L in 0..42:",
                "  hc_pre attn (hc_attn_fn/base/scale) — input dim 16384",
                "  attn_norm",
                "  MLA: wq_a, q_norm, wq_b, wkv, kv_norm, optional compressor, optional indexer, attn_sink, wo_a, wo_b",
                "  hc_post attn",
                "  hc_pre ffn (hc_ffn_fn/base/scale)",
                "  ffn_norm",
                "  gate.weight (+ bias or tid2eid) then route top-6",
                "  6 routed experts (w1,w3,w2 + scales) streamed after the route is known",
                "  1 shared expert (w1,w3,w2 + scales)",
                "  hc_post ffn",
                "hc_head_fn/base/scale + norm.weight + head.weight",
                "MTP layer 43 is excluded from the base streamed token path",
            ],
            "scheduler_stages": [
                "mhc_attention_control",
                "attention_wq_a",
                "attention_wq_b",
                "attention_wkv",
                "attention_wo_a",
                "attention_wo_b",
                "mhc_ffn_control",
                "routed_expert_wave (top-6 only)",
                "shared_expert_w1",
                "shared_expert_w2",
                "shared_expert_w3",
            ],
            "peak_full_layer_resident_bytes": peak_full,
            "peak_full_layer": peak_full_layer,
            "peak_streamed_decode_bytes": peak_streamed,
            "peak_streamed_decode_layer": peak_streamed_layer,
            "layers": layers_out,
        },
        "activation_x_capture": x_capture,
        "prior_runtime_accounting_check": {
            "claimed": {
                "attention_active_bytes": 144_703_488,
                "moe_teacher_active_bytes": 102_760_448,
                "total_active_bytes": 247_463_936,
                "source": "workspace/campaign/evidence/models/deepseek-v4/DEEPSEEK_V4_RUNTIME_ACCOUNTING.json",
            },
            "reconstructed_from_manifest_geometry": {
                "mla_fp8_weight_bytes": mla_weight_bytes,
                "shared_fp8_weight_bytes": shared_weight_bytes,
                "one_routed_fp4_weight_bytes": one_expert_weight_bytes,
                "six_routed_fp4_weight_bytes": six_expert_weight_bytes,
                "gate_bf16_weight_bytes": gate_weight_bytes,
                "moe_prior_style_six_plus_shared_plus_gate": moe_teacher_prior_style,
                "mla_plus_shared_plus_one_routed": mla_plus_shared_plus_one_expert,
                "moe_matches_prior": moe_teacher_prior_style == 102_760_448,
                "attention_equals_mla_plus_36mib": mla_plus_shared_plus_one_expert
                == 144_703_488,
            },
            "typical_ratio4_layer_from_manifest": {
                "layer": one_layer_ref,
                "full_layer_bytes": lb4.get("all_bytes"),
                "mla_bytes": lb4.get("mla_bytes"),
                "indexer_compressor_bytes": lb4.get("indexer_compressor_bytes"),
                "mhc_bytes": lb4.get("mhc_bytes"),
                "shared_expert_bytes": lb4.get("shared_expert_bytes"),
                "six_routed_bytes": TOP_K * lb4.get("one_routed_expert_bytes", 0),
                "router_gate_bytes": lb4.get("router_gate_bytes"),
            },
            "verdict": (
                "The 145 MB vs 103 MB split is a PER-TOKEN SERVED traffic "
                "figure, not a stored-parameter split. Stored byte mass is "
                "overwhelmingly routed experts. The 102,760,448 MoE number "
                "reproduces exactly as 6*routed_fp4_weights + shared_fp8_weights "
                "+ gate.weight (NO scales). The 144,703,488 attention number "
                "reproduces exactly as MLA_fp8_weights + shared_fp8_weights + "
                "ONE routed_fp4_expert_weights — that is not a clean "
                "attention-only mass. A 1.5 COMPLETE budget still cannot ignore "
                "MLA/mHC/indexer/router: they dominate served bytes/token even "
                "though they are a small fraction of stored mass."
            ),
        },
    }


def _resident_set(layer: int) -> dict[str, Any]:
    mode = compression_mode(layer)
    names = [
        f"layers.{layer}.hc_attn_fn|base|scale",
        f"layers.{layer}.attn_norm.weight",
        f"layers.{layer}.attn.wq_a.weight+scale",
        f"layers.{layer}.attn.q_norm.weight",
        f"layers.{layer}.attn.wq_b.weight+scale",
        f"layers.{layer}.attn.wkv.weight+scale",
        f"layers.{layer}.attn.kv_norm.weight",
        f"layers.{layer}.attn.attn_sink",
        f"layers.{layer}.attn.wo_a.weight+scale",
        f"layers.{layer}.attn.wo_b.weight+scale",
    ]
    if mode != "sliding_window_only":
        names.extend(
            [
                f"layers.{layer}.attn.compressor.ape",
                f"layers.{layer}.attn.compressor.wkv.weight",
                f"layers.{layer}.attn.compressor.wgate.weight",
                f"layers.{layer}.attn.compressor.norm.weight",
            ]
        )
    if mode == "ratio_4_with_indexer":
        names.extend(
            [
                f"layers.{layer}.attn.indexer.wq_b.weight+scale",
                f"layers.{layer}.attn.indexer.weights_proj.weight",
                f"layers.{layer}.attn.indexer.compressor.ape",
                f"layers.{layer}.attn.indexer.compressor.wkv.weight",
                f"layers.{layer}.attn.indexer.compressor.wgate.weight",
                f"layers.{layer}.attn.indexer.compressor.norm.weight",
            ]
        )
    names.extend(
        [
            f"layers.{layer}.hc_ffn_fn|base|scale",
            f"layers.{layer}.ffn_norm.weight",
            f"layers.{layer}.ffn.gate.weight",
            (
                f"layers.{layer}.ffn.gate.tid2eid"
                if layer < HASH_LAYER_COUNT
                else f"layers.{layer}.ffn.gate.bias"
            ),
            f"layers.{layer}.ffn.shared_experts.w{{1,2,3}}.weight+scale",
            f"layers.{layer}.ffn.experts.{{top6}}.w{{1,2,3}}.weight+scale",
        ]
    )
    if layer == MTP_LAYER:
        names = [name.replace(f"layers.{layer}.", "mtp.0.") for name in names]
        names.extend(
            [
                "mtp.0.e_proj.weight+scale",
                "mtp.0.h_proj.weight+scale",
                "mtp.0.enorm.weight",
                "mtp.0.hnorm.weight",
                "mtp.0.norm.weight",
                "mtp.0.hc_head_fn|base|scale",
            ]
        )
    return {
        "must_be_co_resident_weight_names": names,
        "can_evict_unselected_routed_experts": True,
        "persistent_state_not_in_weight_peak": [
            "hc_state [hc_mult=4, hidden=4096]",
            "sliding_window kv_cache [128, 512]",
            "compressed kv_cache [seq/ratio, 512] when ratio>0",
            "indexer kv_cache [seq/4, 128] on ratio-4 layers",
        ],
    }


def _x_capture_organs(*, f_expert: float, organ_table: dict[str, Any]) -> dict[str, Any]:
    """Organs that need retained activation X for an activation-weighted fit."""

    organs = [
        {
            "organ": "routed_expert.w1",
            "input_dim": HIDDEN,
            "output_dim": MOE_INTER,
            "required_for_1_5_complete": True,
            "reason": "Dominant stored mass. w1 is the SwiGLU gate; X is post-ffn_norm hidden.",
        },
        {
            "organ": "routed_expert.w3",
            "input_dim": HIDDEN,
            "output_dim": MOE_INTER,
            "required_for_1_5_complete": True,
            "reason": "Same X as w1 (post-ffn_norm hidden). SwiGLU up-projection.",
        },
        {
            "organ": "routed_expert.w2",
            "input_dim": MOE_INTER,
            "output_dim": HIDDEN,
            "required_for_1_5_complete": True,
            "reason": "Needs the REAL SwiGLU hidden (silu(w1(x))*w3(x)), not the router input.",
        },
        {
            "organ": "shared_expert.w1",
            "input_dim": HIDDEN,
            "output_dim": MOE_INTER,
            "required_for_1_5_complete": True,
            "reason": "Always-on expert; same X as routed w1. Small mass, always in the token path.",
        },
        {
            "organ": "shared_expert.w3",
            "input_dim": HIDDEN,
            "output_dim": MOE_INTER,
            "required_for_1_5_complete": True,
            "reason": "Same X as shared w1.",
        },
        {
            "organ": "shared_expert.w2",
            "input_dim": MOE_INTER,
            "output_dim": HIDDEN,
            "required_for_1_5_complete": True,
            "reason": "Same SwiGLU-output X rule as routed w2.",
        },
        {
            "organ": "mla.wq_a",
            "input_dim": HIDDEN,
            "output_dim": Q_LORA,
            "required_for_1_5_complete": True,
            "reason": "Q-LoRA down. X is post-attn_norm hidden. Required because served attention traffic dominates the token.",
        },
        {
            "organ": "mla.wq_b",
            "input_dim": Q_LORA,
            "output_dim": N_HEADS * HEAD_DIM,
            "required_for_1_5_complete": True,
            "reason": "Q-LoRA up. X is q_norm(wq_a(h)), dim 1024 — a distinct capture matrix.",
        },
        {
            "organ": "mla.wkv",
            "input_dim": HIDDEN,
            "output_dim": HEAD_DIM,
            "required_for_1_5_complete": True,
            "reason": "Compressed KV projection 512x4096. Same X as wq_a (post-attn_norm hidden).",
        },
        {
            "organ": "mla.wo_a",
            "input_dim": N_HEADS * HEAD_DIM // O_GROUPS,
            "output_dim": O_GROUPS * O_LORA,
            "required_for_1_5_complete": True,
            "reason": "Grouped O-LoRA. X is attention output reshaped to 8 groups of 4096.",
        },
        {
            "organ": "mla.wo_b",
            "input_dim": O_GROUPS * O_LORA,
            "output_dim": HIDDEN,
            "required_for_1_5_complete": True,
            "reason": "O-LoRA up. X is wo_a output flattened, dim 8192 — a distinct capture matrix.",
        },
        {
            "organ": "indexer.wq_b",
            "input_dim": Q_LORA,
            "output_dim": 64 * 128,
            "required_for_1_5_complete": True,
            "reason": "Ratio-4 layers only. X is the same qr as wq_a output (dim 1024).",
        },
        {
            "organ": "indexer.weights_proj",
            "input_dim": HIDDEN,
            "output_dim": 64,
            "required_for_1_5_complete": True,
            "reason": "Ratio-4 layers only. X is post-attn_norm hidden.",
        },
        {
            "organ": "compressor.wkv",
            "input_dim": HIDDEN,
            "output_dim": None,
            "output_dim_by_mode": {"ratio_4": 1024, "ratio_128": 512},
            "required_for_1_5_complete": True,
            "reason": "Learned KV compress. X is post-attn_norm hidden. Present on every non-sliding layer.",
        },
        {
            "organ": "compressor.wgate",
            "input_dim": HIDDEN,
            "output_dim": None,
            "output_dim_by_mode": {"ratio_4": 1024, "ratio_128": 512},
            "required_for_1_5_complete": True,
            "reason": "Same X as compressor.wkv.",
        },
        {
            "organ": "indexer.compressor.wkv",
            "input_dim": HIDDEN,
            "output_dim": 256,
            "required_for_1_5_complete": True,
            "reason": "Indexer has its own compressor (overlap, index_head_dim=128 → 256). Same X.",
        },
        {
            "organ": "indexer.compressor.wgate",
            "input_dim": HIDDEN,
            "output_dim": 256,
            "required_for_1_5_complete": True,
            "reason": "Same X as indexer.compressor.wkv.",
        },
        {
            "organ": "router_gate.weight",
            "input_dim": HIDDEN,
            "output_dim": ROUTED_EXPERTS,
            "required_for_1_5_complete": True,
            "reason": "Learned layers 3..42. X is post-ffn_norm hidden. Small mass, quality-critical. Hash layers still have this score matrix but route via tid2eid.",
        },
        {
            "organ": "mhc.hc_attn_fn",
            "input_dim": HC_MULT * HIDDEN,
            "output_dim": (2 + HC_MULT) * HC_MULT,
            "required_for_1_5_complete": False,
            "reason": "F32 24x16384. Cheap in bytes (~1.5 MB). Capture only if an mHC-specific fit is attempted. X is flattened hc_mult hiddens.",
        },
        {
            "organ": "mhc.hc_ffn_fn",
            "input_dim": HC_MULT * HIDDEN,
            "output_dim": (2 + HC_MULT) * HC_MULT,
            "required_for_1_5_complete": False,
            "reason": "Same geometry as hc_attn_fn; different X (post-attention HC state).",
        },
        {
            "organ": "lm_head",
            "input_dim": HIDDEN,
            "output_dim": VOCAB,
            "required_for_1_5_complete": False,
            "reason": "Protect at native BF16. Not an AW-SVD target unless the head is later put in play.",
        },
        {
            "organ": "mtp.e_proj",
            "input_dim": HIDDEN,
            "output_dim": HIDDEN,
            "required_for_1_5_complete": False,
            "reason": "MTP auxiliary only; excluded from the base 43-layer token path.",
        },
        {
            "organ": "mtp.h_proj",
            "input_dim": HIDDEN,
            "output_dim": HIDDEN,
            "required_for_1_5_complete": False,
            "reason": "MTP auxiliary only.",
        },
    ]
    distinct_x = [
        {
            "x_id": "h_post_attn_norm",
            "dim": HIDDEN,
            "feeds": [
                "mla.wq_a",
                "mla.wkv",
                "compressor.wkv",
                "compressor.wgate",
                "indexer.weights_proj",
                "indexer.compressor.wkv",
                "indexer.compressor.wgate",
            ],
            "layers": "all base layers; compressor/indexer only where present",
        },
        {
            "x_id": "q_lora_qr",
            "dim": Q_LORA,
            "feeds": ["mla.wq_b", "indexer.wq_b"],
            "layers": "all base (wq_b); indexer.wq_b on ratio-4 only",
        },
        {
            "x_id": "attn_out_grouped",
            "dim": N_HEADS * HEAD_DIM // O_GROUPS,
            "feeds": ["mla.wo_a"],
            "layers": "all base",
        },
        {
            "x_id": "o_lora",
            "dim": O_GROUPS * O_LORA,
            "feeds": ["mla.wo_b"],
            "layers": "all base",
        },
        {
            "x_id": "h_post_ffn_norm",
            "dim": HIDDEN,
            "feeds": [
                "router_gate.weight",
                "routed_expert.w1",
                "routed_expert.w3",
                "shared_expert.w1",
                "shared_expert.w3",
            ],
            "layers": "all base",
        },
        {
            "x_id": "swiglu_hidden_routed",
            "dim": MOE_INTER,
            "feeds": ["routed_expert.w2"],
            "layers": "all base; route-conditioned, only tokens that hit the expert",
        },
        {
            "x_id": "swiglu_hidden_shared",
            "dim": MOE_INTER,
            "feeds": ["shared_expert.w2"],
            "layers": "all base; every token",
        },
        {
            "x_id": "hc_flat_pre_attn",
            "dim": HC_MULT * HIDDEN,
            "feeds": ["mhc.hc_attn_fn"],
            "layers": "all base; optional",
        },
        {
            "x_id": "hc_flat_pre_ffn",
            "dim": HC_MULT * HIDDEN,
            "feeds": ["mhc.hc_ffn_fn"],
            "layers": "all base; optional",
        },
        {
            "x_id": "h_final",
            "dim": HIDDEN,
            "feeds": ["lm_head"],
            "layers": "after layer 42; optional (protect native)",
        },
    ]
    required = [row for row in organs if row["required_for_1_5_complete"]]
    return {
        "f_expert_context": f_expert,
        "required_count": len(required),
        "organs": organs,
        "distinct_x_matrices_that_size_the_capture": distinct_x,
        "do_not_capture_as_x": [
            "embeddings (lookup)",
            "norms / attn_sink / mHC base+scale / compressor.ape (tiny or not a GEMM)",
            "tid2eid (I64 table)",
            "UE8M0 scales (sidecar of a pair, not an X-fit organ)",
        ],
        "organ_mass_reminder": {
            "routed_expert_byte_pct": organ_table["routed_expert"]["byte_pct"],
            "mla_byte_pct": organ_table["mla"]["byte_pct"],
            "shared_expert_byte_pct": organ_table["shared_expert"]["byte_pct"],
            "indexer_compressor_byte_pct": organ_table["indexer_compressor"]["byte_pct"],
        },
    }


def render_markdown(schedule: dict[str, Any]) -> str:
    organs = schedule["organs"]
    env = schedule["bpw_feasibility_envelope"]
    cov = schedule["coverage"]
    prior = schedule["prior_runtime_accounting_check"]
    stream = schedule["per_layer_streaming"]
    dtypes = schedule["dtype_breakdown"]
    lines: list[str] = []
    a = lines.append
    a("# DSV4F tensor schedule")
    a("")
    a("Read-only analysis of the sealed DeepSeek-V4-Flash 43-layer source stream.")
    a("Masses come from `manifest[\"tensors\"]`. No chunk body was opened.")
    a("")
    a("## Artifact identity")
    a("")
    art = schedule["artifact"]
    a(f"- path: `{art.get('artifact_root')}`")
    a(f"- schema: `{art.get('schema')}`")
    a(f"- status: `{art.get('status')}`")
    a(f"- seal_sha256: `{art.get('seal_sha256')}`")
    a(f"- content_addressed_chunk_sha256: `{art.get('content_addressed_chunk_sha256')}`")
    a(f"- chunks: {art.get('content_addressed_chunk_count')}")
    a(f"- tensor_count: {art.get('tensor_count')}")
    a(f"- total_tensor_bytes: {art.get('total_tensor_bytes')}")
    a(f"- repository: `{art.get('repository')}@{art.get('revision')}`")
    a(
        "- MTP tensors are named `mtp.0.*`, not `layers.43.*`. They classify into "
        "the same organs; only the four MTP-unique projections stay in `other`."
    )
    a(
        "- The BYTE_AUCTION `158.07B logical` figure is **stored elements** "
        f"(`prod(shape)` = {schedule['coverage']['stored_elements']}). "
        "Unpacked FP4 logical params are larger because each I8 holds two e2m1 values."
    )
    a("")
    a("## Coverage")
    a("")
    a(f"- classified tensors: **{cov['class_count_sum']} / {cov['expected_tensor_count']}**")
    a(f"- byte mass sum: **{cov['byte_mass_sum']}**")
    a(f"- byte residual vs manifest total: **{cov['byte_residual']}**")
    a(f"- logical unpacked params: **{cov['logical_params_unpacked']}**")
    a(f"- stored elements (prod of physical shapes): **{cov['stored_elements']}**")
    a(f"- shape/dtype undetermined: **{len(cov['shape_dtype_undetermined'])}**")
    a(f"- other (enumerated): **{len(cov['other_enumerated'])}**")
    a(
        f"- scope counts: base={cov['scope_tensor_count']['base']} "
        f"mtp={cov['scope_tensor_count']['mtp']} "
        f"global={cov['scope_tensor_count']['global']}"
    )
    a(f"- covers_all_tensors: **{cov['covers_all_tensors']}**")
    a("")
    a("## Per-organ mass")
    a("")
    a("| organ | tensors | bytes | % bytes | logical params | % params | source bpw (logical) |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for name in ORGAN_CLASSES:
        row = organs[name]
        bpw = row["source_bpw_vs_logical"]
        bpw_s = f"{bpw:.4f}" if isinstance(bpw, float) else "—"
        a(
            f"| {name} | {row['tensor_count']} | {row['byte_mass']} | "
            f"{row['byte_pct']:.4f} | {row['logical_params']} | "
            f"{row['logical_param_pct']:.4f} | {bpw_s} |"
        )
    a("")
    a("Routed-expert subroles (w1/w3/w2):")
    a("")
    a("| subrole | tensors | bytes | logical params |")
    a("|---|---:|---:|---:|")
    for sub, bucket in organs["routed_expert"]["subroles"].items():
        a(f"| {sub} | {bucket['tensor_count']} | {bucket['byte_mass']} | {bucket['logical_params']} |")
    a("")
    if cov["other_enumerated"]:
        a("### Other (fully enumerated)")
        a("")
        a(
            "These names matched no organ suffix after `layers.L.` / `mtp.N.` stripping. "
            "MTP-unique projections live here on purpose."
        )
        a("")
        for name in cov["other_enumerated"]:
            a(f"- `{name}`")
        a("")
    else:
        a("Other is empty: every tensor matched an organ class.")
        a("")
    a("## Dtype / source precision")
    a("")
    a("| stored dtype | family | tensors | bytes | % bytes | logical params |")
    a("|---|---|---:|---:|---:|---:|")
    for dtype, row in dtypes.items():
        a(
            f"| {dtype} | {row['family']} | {row['tensor_count']} | "
            f"{row['byte_mass']} | {row['byte_pct']:.4f} | {row['logical_params']} |"
        )
    a("")
    a(
        "Experts are already FP4-native (`I8` packed e2m1fn_x2 + `F8_E8M0` scales). "
        "Further 'compression' of experts is a second packing on top of FP4, not "
        "a BF16→low-bit collapse. Control path is FP8 e4m3 + UE8M0. Embeddings, "
        "lm_head, norms, gate scores, and compressor weights are BF16. mHC and "
        "APE tables are F32. Hash `tid2eid` is I64."
    )
    a("")
    a("## BPW feasibility envelope")
    a("")
    a("```")
    a(env["formula"])
    a("```")
    a("")
    a(f"- f_expert = {env['f_expert']:.9f}  ({env['expert_logical_params']} logical params)")
    a(f"- f_nonexpert = {env['f_nonexpert']:.9f}  ({env['nonexpert_logical_params']} logical params)")
    a(f"- source expert bpw (logical, includes scales) = {env['source_expert_bpw_logical']}")
    a(f"- source non-expert bpw (logical) = {env['source_nonexpert_bpw_logical']}")
    a(f"- source complete bpw (logical unpacked) = {env['source_complete_bpw_logical']}")
    a(f"- source complete bpw (stored elements) = {env['source_complete_bpw_stored_elements']}")
    a(f"- source complete bpw vs claimed 158.07B = {env['source_complete_bpw_claimed_158p07b']}")
    a("")
    a("Required non-expert bits to hold `complete_bpw = 1.5`:")
    a("")
    a("| expert_bpw | required_nonexpert_bpw | feasible |")
    a("|---:|---:|---|")
    for row in env["table"]:
        req = row["required_nonexpert_bpw"]
        req_s = f"{req:.6f}" if isinstance(req, float) else "—"
        a(f"| {row['expert_bpw']} | {req_s} | {row['feasible']} |")
    a("")
    a(env["scale_sidecar_note"])
    a("")
    a("## Per-layer streaming")
    a("")
    a("A layer executes in this order (official `inference/model.py` `Block.forward`):")
    a("")
    for step in stream["execution_order"]:
        a(f"- {step}")
    a("")
    a(
        f"Peak **full-layer** resident weights: **{stream['peak_full_layer_resident_bytes']}** "
        f"bytes at layer {stream['peak_full_layer']} (all 256 routed experts present)."
    )
    a(
        f"Peak **streamed-decode** resident weights (keep non-experts + shared + top-6 routed): "
        f"**{stream['peak_streamed_decode_bytes']}** bytes at layer {stream['peak_streamed_decode_layer']}."
    )
    a("")
    a("| L | mode | gate | tensors | full bytes | streamed peak | MLA | indexer/compr | 6 experts | shared |")
    a("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in stream["layers"]:
        if row["layer"] == MTP_LAYER:
            continue
        a(
            f"| {row['layer']} | {row['compression']} | {row['gate']} | "
            f"{row['tensor_count']} | {row['full_layer_resident_bytes']} | "
            f"{row['streamed_decode_peak_bytes']} | {row['mla_bytes']} | "
            f"{row['indexer_compressor_bytes']} | {row['six_routed_expert_bytes']} | "
            f"{row['shared_expert_bytes']} |"
        )
    mtp = next((r for r in stream["layers"] if r["layer"] == MTP_LAYER), None)
    if mtp:
        a("")
        a(
            f"MTP layer 43 is **not** on the base token path "
            f"({mtp['tensor_count']} tensors, {mtp['full_layer_resident_bytes']} bytes, "
            f"{mtp['compression']})."
        )
    a("")
    a("## Activation X capture (sizes the later capture lane)")
    a("")
    a("Organs that **must** retain X for a ≤1.5 complete activation-weighted fit:")
    a("")
    a("| organ | input dim | required |")
    a("|---|---:|---|")
    for row in schedule["activation_x_capture"]["organs"]:
        if not row["required_for_1_5_complete"]:
            continue
        dim = row["input_dim"]
        a(f"| `{row['organ']}` | {dim} | yes |")
    a("")
    a("Distinct X matrices (this is the capture surface):")
    a("")
    for row in schedule["activation_x_capture"]["distinct_x_matrices_that_size_the_capture"]:
        a(f"- `{row['x_id']}` dim={row['dim']} → {', '.join(row['feeds'])}")
    a("")
    a("## Prior 145 MB vs 103 MB")
    a("")
    a(prior["verdict"])
    a("")
    rec = prior["reconstructed_from_manifest_geometry"]
    a(f"- reconstructed MoE teacher-active: {rec['moe_prior_style_six_plus_shared_plus_gate']} (match={rec['moe_matches_prior']})")
    a(f"- reconstructed 'attention' 138 MiB identity: {rec['mla_plus_shared_plus_one_routed']} (equals claimed={rec['attention_equals_mla_plus_36mib']})")
    a(f"- MLA fp8 weights only: {rec['mla_fp8_weight_bytes']}")
    a("")
    a("## Undetermined")
    a("")
    if cov["shape_dtype_undetermined"]:
        for row in cov["shape_dtype_undetermined"]:
            a(f"- `{row['name']}` dtype={row['dtype']} shape={row['shape']} bytes={row['bytes']} ({row['reason']})")
    else:
        a("None. Every tensor has a known dtype, shape, and bytes that reconcile.")
    a("")
    return "\n".join(lines) + "\n"


def default_output_dir(repo_root: Path) -> Path:
    return repo_root / "workspace/campaign/records/runs/deepseek-v4"


def write_reports(schedule: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "DSV4F_TENSOR_SCHEDULE.json"
    md_path = out_dir / "DSV4F_TENSOR_SCHEDULE.md"
    json_path.write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(schedule), encoding="utf-8")
    return json_path, md_path


def analyze(artifact_root: Path | None = None) -> dict[str, Any]:
    root = resolve_artifact_root(artifact_root)
    manifest = root / "manifest.json"
    identity = extract_manifest_identity(manifest)
    classified = classify_all(iter_manifest_tensors(manifest))
    return build_schedule(artifact_root=root, identity=identity, classified=classified)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    schedule = analyze(args.artifact)
    if args.write:
        repo = Path(__file__).resolve().parents[2]
        out_dir = args.out_dir or default_output_dir(repo)
        json_path, md_path = write_reports(schedule, out_dir)
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    cov = schedule["coverage"]
    print(
        f"classified {cov['class_count_sum']}/{cov['expected_tensor_count']} "
        f"bytes={cov['byte_mass_sum']} residual={cov['byte_residual']} "
        f"covers={cov['covers_all_tensors']}"
    )
    return 0 if cov["covers_all_tensors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
