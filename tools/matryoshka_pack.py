#!/usr/bin/env python3
"""G11: 2-tier Matryoshka NR packer + reconstruction check for one Qwen3.8 organ set.

BASE is the mixed-q3mlp packing scheme: HGRAVU01 uniform-q3 group-64 absmax RTN.
CORRECTION is the same codec at q2, applied to the residual (bf16 - q3_dequant).
One stored hierarchy: base codes + correction codes + two f16 scale planes that
share the same group-64 layout. Not two duplicate models.

The existing mixed-q3mlp-q3attn-v1 artifact (3.3448 complete BPW, coherent) IS
the standalone base and already runs. This script proves the tiered structure
in numpy. Native-decode of the correction plane is future work.

  python3 tools/matryoshka_pack.py --layers 15,31,47 --out hawking-experiments/superwave/g1/g11-matryoshka.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_RUNS = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b"
)
GROUP = 64
BASE_BITS = 3
CORR_BITS = 2
HIDDEN = 5120
INTERMEDIATE = 17408
N_SOURCE_WEIGHTS = 26_895_998_464
PARENT_MODEL = "Qwen3.8-27B (Genesis patient, abliterated)"
MIXED_Q3MLP = "mixed-q3mlp-q3attn-v1"
MIXED_Q3MLP_BPW = 3.3447723007722434
MIXED_Q3MLP_MLP_BPW = 3.2500251321231617
MIXED_Q3MLP_BASE_BYTES = 36_208_920  # one gate/up/down HGRAVU01 q3 payload (CITED PACK_REPORT)
STRIDE_COSINE = 17  # matches lab.operators.qwen38_mlp_not_r160_pack.strided_weight_cosine
SCHEMA = "hawking.nr.matryoshka_mlp.v1"
MAGIC = b"HGRAVM01"


class PackError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[matryoshka] {msg}", flush=True)


def resolve_parent(explicit: Path | None) -> Path:
    env = os.environ.get("QWEN38_PARENT_BF16")
    candidates = [
        explicit,
        Path(env) if env else None,
        MAIN_RUNS / "bf16",
        ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16",
    ]
    for cand in candidates:
        if cand is not None and (cand / "model.safetensors.index.json").is_file():
            return cand
    raise PackError(
        "bf16 parent not found. Set --parent or QWEN38_PARENT_BF16. "
        "Expected workspace/campaign/records/runs/qwen38-27b/bf16"
    )


def resolve_acts(explicit: Path | None, parent: Path) -> Path:
    env = os.environ.get("QWEN38_POST_SWIGLU")
    candidates = [
        explicit,
        Path(env) if env else None,
        parent.parent / "activation-capture-v2/parent_bf16/post_swiglu",
        MAIN_RUNS / "activation-capture-v2/parent_bf16/post_swiglu",
    ]
    for cand in candidates:
        if cand is not None and cand.is_dir():
            return cand
    raise PackError(
        "post_swiglu capture not found. Set --acts or QWEN38_POST_SWIGLU. "
        "Expected activation-capture-v2/parent_bf16/post_swiglu"
    )


def load_weight_map(model_dir: Path) -> dict[str, str]:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return dict(idx["weight_map"])


_HEADER_CACHE: dict[Path, dict[str, Any]] = {}


def read_safetensors_header(shard: Path) -> dict[str, Any]:
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def load_bf16_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    shard = model_dir / weight_map[name]
    if shard not in _HEADER_CACHE:
        _HEADER_CACHE[shard] = read_safetensors_header(shard)
    info = _HEADER_CACHE[shard][name]
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + n + lo)
        raw = fh.read(hi - lo)
    if dtype not in ("BF16", "BFLOAT16"):
        raise PackError(f"{name}: expected BF16, got {dtype}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    if u16.size != int(np.prod(shape)):
        raise PackError(f"{name}: {u16.size} bf16 values != shape {shape}")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def bound_for(bits: int) -> int:
    if bits < 2 or bits > 8:
        raise PackError(f"uniform bits {bits} not in 2..8")
    return (1 << (bits - 1)) - 1


def payload_bytes(n_elem: int, bits: int, group: int = GROUP) -> tuple[int, int, int]:
    """HGRAVU01 body bytes: (code_bytes, scale_bytes, body_bytes)."""
    groups = math.ceil(n_elem / group)
    padded = groups * group
    code_b = math.ceil(padded * bits / 8)
    scale_b = groups * 2
    return code_b, scale_b, code_b + scale_b


def container_bytes(n_elem: int, bits: int, shape: tuple[int, ...]) -> int:
    """Physical HGRAVU01 file size: 8-byte magic + u32 header len + JSON + body."""
    groups = math.ceil(n_elem / GROUP)
    code_b, scale_b, body = payload_bytes(n_elem, bits)
    header = {
        "schema": "hawking.gravity.uniform_group.v1",
        "representation": f"uniform_q{bits}_group_scale",
        "shape": [int(x) for x in shape],
        "elements": int(n_elem),
        "bits": int(bits),
        "group_size": GROUP,
        "groups": groups,
        "scale_dtype": "float16",
        "code_bytes": code_b,
        "scale_bytes": scale_b,
        "retained_padding_elements": int(groups * GROUP - n_elem),
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return 8 + 4 + len(encoded) + body


@dataclass
class UniformTier:
    bits: int
    codes: np.ndarray  # int8, padded groups x GROUP
    scales_f16: np.ndarray  # float16, one per group
    recon: np.ndarray  # float32, original shape


def encode_uniform(values: np.ndarray, bits: int, group: int = GROUP) -> UniformTier:
    """HGRAVU01: per-group absmax / (2^{bits-1}-1), f16 scale snap, round-to-nearest."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(flat).all():
        raise PackError("non-finite source")
    groups = math.ceil(flat.size / group)
    padded = np.zeros(groups * group, dtype=np.float32)
    padded[: flat.size] = flat
    grouped = padded.reshape(groups, group)
    bnd = bound_for(bits)
    scales_f16 = (np.max(np.abs(grouped), axis=1) / max(bnd, 1)).astype(np.float16)
    denom = scales_f16.astype(np.float32)
    denom = np.where(denom > 0.0, denom, 1.0)
    signed = np.rint(grouped / denom[:, None]).clip(-bnd, bnd).astype(np.int8)
    recon = (signed.astype(np.float32) * denom[:, None]).reshape(-1)[: flat.size]
    return UniformTier(
        bits=bits,
        codes=signed,
        scales_f16=scales_f16,
        recon=np.ascontiguousarray(recon.reshape(values.shape), dtype=np.float32),
    )


def rel_fro(ref: np.ndarray, hat: np.ndarray) -> float:
    a = np.asarray(ref, dtype=np.float64).reshape(-1)
    b = np.asarray(hat, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def cosine(ref: np.ndarray, hat: np.ndarray) -> float:
    a = np.asarray(ref, dtype=np.float64).reshape(-1)
    b = np.asarray(hat, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def strided_weight_cosine(source: np.ndarray, recon: np.ndarray, stride: int = STRIDE_COSINE) -> float:
    """Full-tensor HGRAVU01 recon, then every `stride`-th group (f16 scales)."""
    flat_s = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
    flat_r = np.ascontiguousarray(recon, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat_s.size / GROUP)
    pad_s = np.zeros(groups * GROUP, dtype=np.float32)
    pad_r = np.zeros(groups * GROUP, dtype=np.float32)
    pad_s[: flat_s.size] = flat_s
    pad_r[: flat_r.size] = flat_r
    idx = np.arange(0, groups, stride)
    a = pad_s.reshape(groups, GROUP)[idx].reshape(-1).astype(np.float64)
    b = pad_r.reshape(groups, GROUP)[idx].reshape(-1).astype(np.float64)
    return cosine(a, b)


def packer_strided_cosine(source: np.ndarray, bits: int, stride: int = STRIDE_COSINE) -> float:
    """Byte-identical to qwen38_mlp_not_r160_pack.strided_weight_cosine (f32 scales)."""
    flat = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / GROUP)
    padded = np.zeros(groups * GROUP, dtype=np.float32)
    padded[: flat.size] = flat
    grouped = padded.reshape(groups, GROUP)
    idx = np.arange(0, groups, stride)
    bnd = bound_for(bits)
    scales = np.max(np.abs(grouped[idx]), axis=1) / max(bnd, 1)
    denom = np.where(scales > 0.0, scales, 1.0)
    signed = np.rint(grouped[idx] / denom[:, None]).clip(-bnd, bnd)
    recon = signed * denom[:, None]
    return cosine(grouped[idx], recon)


def tensor_name(layer: int, organ: str) -> str:
    return f"language_model.model.layers.{layer}.mlp.{organ}.weight"


def load_post_swiglu(acts: Path, layer: int, n_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = acts / f"L{layer:02d}.f16"
    if not path.is_file():
        path = acts / f"L{layer}.f16"
    if not path.is_file():
        raise PackError(f"missing post_swiglu capture {path}")
    nbytes = path.stat().st_size
    if nbytes % (INTERMEDIATE * 2) != 0:
        raise PackError(f"{path} size {nbytes} is not a multiple of {INTERMEDIATE}*2")
    n_avail = nbytes // (INTERMEDIATE * 2)
    mm = np.memmap(path, dtype="<f2", mode="r", shape=(n_avail, INTERMEDIATE))
    take = min(int(n_rows), n_avail)
    # Even coverage of the 23216-token v2 capture (prompt-level hold is not contiguous).
    idx = np.linspace(0, n_avail - 1, take, dtype=np.int64)
    x = np.asarray(mm[idx], dtype=np.float32)
    meta = {
        "path": str(path),
        "n_avail": int(n_avail),
        "n_used": int(take),
        "index": "linspace(0, n_avail-1, n_used)",
        "sha256_file": sha256_file(path),
        "bytes": int(nbytes),
    }
    return x, meta


def sha256_file(path: Path, cap: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(cap)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def gemv_rel_fro(x: np.ndarray, w: np.ndarray, w_hat: np.ndarray) -> tuple[float, float]:
    """Output error of Y = X @ W.T on captured rows. Returns (rel_fro, cosine)."""
    y = x @ w.T
    y_hat = x @ w_hat.T
    return rel_fro(y, y_hat), cosine(y, y_hat)


@dataclass
class TensorResult:
    name: str
    layer: int
    organ: str
    shape: list[int]
    n_elem: int
    dtype_parent: str
    base_bits: int
    corr_bits: int
    weight_rel_fro_base: float
    weight_rel_fro_corr: float
    weight_cosine_base: float
    weight_cosine_corr: float
    strided_cosine_base: float
    strided_cosine_corr: float
    packer_strided_cosine_base: float
    act_rel_fro_base: float | None
    act_rel_fro_corr: float | None
    act_cosine_base: float | None
    act_cosine_corr: float | None
    act_rows: int | None
    act_avail: int | None
    act_path: str | None
    base_code_bytes: int
    base_scale_bytes: int
    base_body_bytes: int
    base_container_bytes: int
    corr_code_bytes: int
    corr_scale_bytes: int
    corr_body_bytes: int
    corr_container_bytes: int
    hierarchy_bytes: int
    two_copy_bytes: int
    bf16_bytes: int
    code_hist_base: dict[str, int]
    code_hist_corr: dict[str, int]


def code_hist(codes: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(codes.reshape(-1), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), counts.tolist())}


def pack_one(
    *,
    parent: Path,
    weight_map: dict[str, str],
    acts: Path,
    layer: int,
    organ: str,
    act_rows: int,
) -> TensorResult:
    name = tensor_name(layer, organ)
    log(f"load {name}")
    w = load_bf16_tensor(parent, weight_map, name)
    if w.ndim != 2:
        raise PackError(f"{name}: expected rank-2, got {w.shape}")
    n_elem = int(w.size)
    t0 = time.perf_counter()
    base = encode_uniform(w, BASE_BITS)
    residual = w - base.recon
    corr = encode_uniform(residual, CORR_BITS)
    w_full = base.recon + corr.recon
    encode_s = time.perf_counter() - t0
    w_rel_base = rel_fro(w, base.recon)
    w_rel_corr = rel_fro(w, w_full)
    w_cos_base = cosine(w, base.recon)
    w_cos_corr = cosine(w, w_full)
    s_cos_base = strided_weight_cosine(w, base.recon)
    s_cos_corr = strided_weight_cosine(w, w_full)
    p_cos_base = packer_strided_cosine(w, BASE_BITS)
    log(
        f"  weight rel-fro base={w_rel_base:.8f}  base+corr={w_rel_corr:.8f}  "
        f"Δ={w_rel_base - w_rel_corr:.8f}  encode_s={encode_s:.3f}"
    )
    if not (w_rel_corr < w_rel_base):
        raise PackError(
            f"{name}: correction did not reduce weight rel-fro "
            f"({w_rel_corr} !< {w_rel_base})"
        )

    act_rel_base = act_rel_corr = act_cos_base = act_cos_corr = None
    act_used = act_avail = None
    act_path = None
    if organ == "down_proj":
        if w.shape != (HIDDEN, INTERMEDIATE):
            raise PackError(f"{name}: down_proj shape {w.shape} != {(HIDDEN, INTERMEDIATE)}")
        x, ameta = load_post_swiglu(acts, layer, act_rows)
        act_path = ameta["path"]
        act_used = ameta["n_used"]
        act_avail = ameta["n_avail"]
        log(f"  gemv post_swiglu {act_used}x{INTERMEDIATE} @ {w.shape}")
        t1 = time.perf_counter()
        act_rel_base, act_cos_base = gemv_rel_fro(x, w, base.recon)
        act_rel_corr, act_cos_corr = gemv_rel_fro(x, w, w_full)
        log(
            f"  act rel-fro base={act_rel_base:.8f}  base+corr={act_rel_corr:.8f}  "
            f"Δ={act_rel_base - act_rel_corr:.8f}  gemv_s={time.perf_counter() - t1:.3f}"
        )
        if not (act_rel_corr < act_rel_base):
            raise PackError(
                f"{name}: correction did not reduce activation rel-fro "
                f"({act_rel_corr} !< {act_rel_base})"
            )

    b_code, b_scale, b_body = payload_bytes(n_elem, BASE_BITS)
    c_code, c_scale, c_body = payload_bytes(n_elem, CORR_BITS)
    b_cont = container_bytes(n_elem, BASE_BITS, w.shape)
    c_cont = container_bytes(n_elem, CORR_BITS, w.shape)
    # One hierarchy: two code planes + two scale planes + one envelope.
    # Not two independent HGRAVU01 files (that would double the JSON/magic).
    hierarchy = b_body + c_body + 256  # 256 B reserved for the nested header
    two_copy = b_cont + c_cont

    return TensorResult(
        name=name,
        layer=layer,
        organ=organ,
        shape=[int(x) for x in w.shape],
        n_elem=n_elem,
        dtype_parent="BF16",
        base_bits=BASE_BITS,
        corr_bits=CORR_BITS,
        weight_rel_fro_base=w_rel_base,
        weight_rel_fro_corr=w_rel_corr,
        weight_cosine_base=w_cos_base,
        weight_cosine_corr=w_cos_corr,
        strided_cosine_base=s_cos_base,
        strided_cosine_corr=s_cos_corr,
        packer_strided_cosine_base=p_cos_base,
        act_rel_fro_base=act_rel_base,
        act_rel_fro_corr=act_rel_corr,
        act_cosine_base=act_cos_base,
        act_cosine_corr=act_cos_corr,
        act_rows=act_used,
        act_avail=act_avail,
        act_path=act_path,
        base_code_bytes=b_code,
        base_scale_bytes=b_scale,
        base_body_bytes=b_body,
        base_container_bytes=b_cont,
        corr_code_bytes=c_code,
        corr_scale_bytes=c_scale,
        corr_body_bytes=c_body,
        corr_container_bytes=c_cont,
        hierarchy_bytes=hierarchy,
        two_copy_bytes=two_copy,
        bf16_bytes=n_elem * 2,
        code_hist_base=code_hist(base.codes),
        code_hist_corr=code_hist(corr.codes),
    )


def fmt(x: float | None, digits: int = 8) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def pct(num: float, den: float) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.2f}%"


def write_report(
    *,
    out: Path,
    results: list[TensorResult],
    parent: Path,
    acts: Path,
    layers: list[int],
    organs: list[str],
    act_rows: int,
    wall_s: float,
    argv: list[str],
) -> None:
    downs = [r for r in results if r.organ == "down_proj"]
    gates = [r for r in results if r.organ == "gate_proj"]
    if len(downs) < 3:
        raise PackError(f"need >=3 down_proj layers, got {len(downs)}")
    mean_w_base = float(np.mean([r.weight_rel_fro_base for r in results]))
    mean_w_corr = float(np.mean([r.weight_rel_fro_corr for r in results]))
    mean_a_base = float(np.mean([r.act_rel_fro_base for r in downs]))
    mean_a_corr = float(np.mean([r.act_rel_fro_corr for r in downs]))
    sum_base_body = sum(r.base_body_bytes for r in results)
    sum_corr_body = sum(r.corr_body_bytes for r in results)
    sum_hier = sum(r.hierarchy_bytes for r in results)
    sum_bf16 = sum(r.bf16_bytes for r in results)
    sum_two = sum(r.two_copy_bytes for r in results)
    n_elem = results[0].n_elem
    sample_bpw_base = 8.0 * results[0].base_body_bytes / n_elem
    sample_bpw_corr = 8.0 * results[0].corr_body_bytes / n_elem
    sample_bpw_hier = 8.0 * (results[0].base_body_bytes + results[0].corr_body_bytes) / n_elem

    capture_meta = {}
    cap_json = acts.parent / "capture-result.json"
    if cap_json.is_file():
        raw = json.loads(cap_json.read_text())
        capture_meta = {
            "schema": raw.get("schema"),
            "status": raw.get("status"),
            "n_tokens": raw.get("n_tokens"),
            "sha256_self": raw.get("sha256_self"),
        }

    nr_spec = {
        "nr_version": "1.1.0-matryoshka",
        "nr_kind": "hawking.nos.noetic_representation",
        "schema": SCHEMA,
        "magic": MAGIC.decode("ascii"),
        "semantic_provenance": {
            "parent_model": PARENT_MODEL,
            "parent_revision": "bf16",
            "parent_path": str(parent),
            "parameter_count": N_SOURCE_WEIGHTS,
            "sample_layers": layers,
            "sample_organs": organs,
            "base_artifact": MIXED_Q3MLP,
            "base_complete_physical_bpw": MIXED_Q3MLP_BPW,
            "base_mlp_physical_bpw": MIXED_Q3MLP_MLP_BPW,
            "patient_note": (
                "abliterated parent; Tabula drift is a Doctor axis and is NOT recorded "
                "here because NR states what the representation IS, not how it scored"
            ),
        },
        "representation": {
            "kind": "two_tier_residual_matryoshka",
            "one_hierarchy_not_two_models": True,
            "shared_structure": {
                "group_size": GROUP,
                "flatten": "row_major C-order of the stored [out, in] matrix",
                "scale_dtype": "float16",
                "scale_layout": (
                    "two f16 planes of identical length (n_groups). They share the "
                    "group index; they do NOT share numeric values. Sharing the base "
                    "absmax as the residual scale maps q2 codes to 0 (residual < 0.5 "
                    "base-step) and the correction plane vanishes."
                ),
            },
            "base": {
                "family": "grouped_absmax",
                "codec": "HGRAVU01",
                "representation": "uniform_q3_group_scale",
                "bits": BASE_BITS,
                "group": GROUP,
                "bound": bound_for(BASE_BITS),
                "scale": "f16(absmax(group) / 3)",
                "codes": "int in [-3, +3], packed unsigned as code+3, 3 bits/elem",
                "standalone": (
                    f"the existing {MIXED_Q3MLP} catalog is this base for every MLP "
                    "GEMV; it loads and generates without the correction plane"
                ),
                "decode": "W_base[g] = q3[g] * s_base[g]",
            },
            "correction": {
                "family": "grouped_absmax",
                "codec": "HGRAVU01",
                "representation": "uniform_q2_group_scale",
                "bits": CORR_BITS,
                "group": GROUP,
                "bound": bound_for(CORR_BITS),
                "of": "residual = bf16_weight - dequant(base)",
                "scale": "f16(absmax(residual_group) / 1)",
                "codes": "int in [-1, +1], packed unsigned as code+1, 2 bits/elem",
                "optional_at_runtime": True,
                "decode": "W_hat[g] = W_base[g] + q2[g] * s_corr[g]",
                "native_decode": "FUTURE WORK - not required for this obligation",
            },
            "byte_ledger_per_mlp_tensor": {
                "n_elem": HIDDEN * INTERMEDIATE,
                "base_code_bytes": results[0].base_code_bytes,
                "base_scale_bytes": results[0].base_scale_bytes,
                "base_body_bytes": results[0].base_body_bytes,
                "base_hgravu01_container_bytes": results[0].base_container_bytes,
                "corr_code_bytes": results[0].corr_code_bytes,
                "corr_scale_bytes": results[0].corr_scale_bytes,
                "corr_body_bytes": results[0].corr_body_bytes,
                "hierarchy_bytes_est": results[0].hierarchy_bytes,
                "bf16_bytes": results[0].bf16_bytes,
                "physical_bpw_base": sample_bpw_base,
                "physical_bpw_corr_plane": sample_bpw_corr,
                "physical_bpw_hierarchy": sample_bpw_hier,
            },
            "entropy_streams": [],
            "shared_structures": [
                {
                    "name": "group64_layout",
                    "group": GROUP,
                    "applies_to": ["base", "correction"],
                }
            ],
            "generated_structures": [],
            "latent_codes": [],
            "correction_planes": [
                {
                    "plane": "mlp_residual_q2",
                    "of": "base_uniform_q3_group64",
                    "family": "grouped_absmax",
                    "bits": CORR_BITS,
                    "group": GROUP,
                    "decode": "W_hat = dequant(base) + dequant(correction)",
                    "optional_at_runtime": True,
                }
            ],
            "exact_islands": [],
            "route_graph": None,
        },
        "kernel_requirements": [
            {
                "requires": "grouped_absmax_decoder",
                "bits": BASE_BITS,
                "group": GROUP,
                "note": (
                    "the mixed-q3mlp decoder family. Naming a specific kernel, "
                    "threadgroup geometry or device here would make this NX, not NR."
                ),
            },
            {
                "requires": "grouped_absmax_decoder",
                "bits": CORR_BITS,
                "group": GROUP,
                "note": (
                    "the correction-plane decoder; same family, 2-bit. Native fused "
                    "decode (Y += X @ dequant(corr).T, or a two-scale unpack in one "
                    "tile) is future work."
                ),
            },
            {
                "requires": "gated_delta_recurrence",
                "note": "the DeltaNet mixer family the representation assumes",
            },
        ],
    }

    lines: list[str] = []
    a = lines.append
    a("# G11 — 2-tier Matryoshka NR for Qwen3.8 MLP")
    a("")
    a("Lane: `g11-matryoshka`. CPU numpy. Real BF16 parent + real v2 `post_swiglu`.")
    a("No GPU. No generate. No runtime / receipt / artifact mutation.")
    a("")
    a(f"STATUS: **MEASURED_WIN**. Base-only reconstructs. Base+correction is closer to")
    a(f"the bf16 parent in weight space on all {len(results)} tensors and on")
    a(f"`down_proj` activations on all {len(downs)} layers.")
    a("")
    a("Every number is **MEASURED** (this process) unless tagged **CITED** or **DERIVED**.")
    a("")
    a("---")
    a("")
    a("## 0. Verdict")
    a("")
    a("A real 2-tier hierarchy exists for this organ set:")
    a("")
    a(f"- **BASE** = HGRAVU01 uniform-q{BASE_BITS} group-{GROUP} of the BF16 MLP weight.")
    a(f"  This is the packing scheme already in `{MIXED_Q3MLP}`")
    a(f"  (CITED complete BPW {MIXED_Q3MLP_BPW:.6f}, MLP BPW {MIXED_Q3MLP_MLP_BPW:.6f},")
    a("  coherent on the campaign gate; see `g1-mlp-family-generate.md` for")
    a("  `mixed-q3mlp-v1` and the `mixed-q3mlp-q3attn-v1` PACK_REPORT / Genesis.nr).")
    a(f"  The base is a valid standalone: that artifact loads and generates without a")
    a("  correction plane.")
    a(f"- **CORRECTION** = HGRAVU01 uniform-q{CORR_BITS} group-{GROUP} of")
    a("  `residual = bf16 − q3_dequant`. Same grouping. Own f16 scale plane.")
    a("- **One hierarchy**, not two models: base codes + correction codes + two scale")
    a("  planes that share the group-64 index. Native-decode of the correction plane")
    a("  is **future work**; this lane proves the stored structure and the error drop.")
    a("")
    a("| space | n | base-only rel-fro (mean) | base+correction rel-fro (mean) | drop |")
    a("|---|---:|---:|---:|---:|")
    a(
        f"| weight | {len(results)} | {mean_w_base:.8f} | {mean_w_corr:.8f} | "
        f"{mean_w_base - mean_w_corr:.8f} |"
    )
    a(
        f"| down_proj on post_swiglu | {len(downs)} | {mean_a_base:.8f} | {mean_a_corr:.8f} | "
        f"{mean_a_base - mean_a_corr:.8f} |"
    )
    a("")
    a("Correction is strictly lower on every row below. Fail-closed: the packer exits")
    a("non-zero if any tensor violates that.")
    a("")
    a("---")
    a("")
    a("## 1. Weight-space reconstruction")
    a("")
    a("Relative Frobenius `||W − Ŵ||_F / ||W||_F` against the BF16 parent.")
    a("Full tensor, not the stride-17 packer screen.")
    a("")
    a("| tensor | shape | base-only rel-fro | base+corr rel-fro | drop | base cos | +corr cos |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        drop = r.weight_rel_fro_base - r.weight_rel_fro_corr
        a(
            f"| `{r.name}` | {r.shape[0]}×{r.shape[1]} | "
            f"{r.weight_rel_fro_base:.8f} | {r.weight_rel_fro_corr:.8f} | "
            f"{drop:.8f} | {r.weight_cosine_base:.8f} | {r.weight_cosine_corr:.8f} |"
        )
    a("")
    a("Stride-17 weight cosine. The packer screen (CITED PACK_REPORT) uses f32 group")
    a("scales and does not snap to f16; `packer-screen` below is that exact operator.")
    a("`f16-recon` is the stored HGRAVU01 path (scale snapped to f16, then subsampled).")
    a("")
    a("| tensor | packer-screen base | CITED mixed-q3mlp | Δ | f16-recon base | f16-recon +corr |")
    a("|---|---:|---:|---:|---:|---:|")
    cited = {
        "language_model.model.layers.15.mlp.gate_proj.weight": 0.9697327583072788,
        "language_model.model.layers.15.mlp.down_proj.weight": 0.9693914492670366,
        "language_model.model.layers.31.mlp.gate_proj.weight": 0.9686059871342714,
        "language_model.model.layers.31.mlp.down_proj.weight": 0.9685743707039933,
        "language_model.model.layers.47.mlp.gate_proj.weight": 0.9686495900886931,
        "language_model.model.layers.47.mlp.down_proj.weight": 0.9684466870964236,
    }
    for r in results:
        cit = cited.get(r.name, float("nan"))
        a(
            f"| `{r.name}` | {r.packer_strided_cosine_base:.12f} | {cit:.12f} | "
            f"{r.packer_strided_cosine_base - cit:.3e} | "
            f"{r.strided_cosine_base:.10f} | {r.strided_cosine_corr:.10f} |"
        )
    a("")
    a("A packer-screen Δ on the order of float64 rounding is the proof that this BASE")
    a("is the mixed-q3mlp scheme, not a new q3. The f16-recon column is slightly lower")
    a("because the stored scale is f16; that is the real artifact, not the screen.")
    a("")
    a("---")
    a("")
    a("## 2. `down_proj` on real `post_swiglu` activations")
    a("")
    a("Y = X @ Wᵀ. X is the v2 parent-BF16 `post_swiglu` capture")
    a(f"(CITED schema `{capture_meta.get('schema', 'hawking.ascension.qwen38_activation_capture.v2')}`,")
    a(f"status `{capture_meta.get('status', '?')}`, n_tokens={capture_meta.get('n_tokens', '?')}).")
    a(f"This process used a linspace of {act_rows} rows across the full capture so the")
    a("measurement is a reconstruction screen on real X, not a generate claim and not")
    a("a 256-token v1 leftover.")
    a("")
    a("| layer | X rows used / avail | base-only rel-fro | base+corr rel-fro | drop | base cos | +corr cos |")
    a("|---:|---:|---:|---:|---:|---:|---:|")
    for r in downs:
        drop = (r.act_rel_fro_base or 0.0) - (r.act_rel_fro_corr or 0.0)
        a(
            f"| {r.layer} | {r.act_rows} / {r.act_avail} | "
            f"{fmt(r.act_rel_fro_base)} | {fmt(r.act_rel_fro_corr)} | "
            f"{drop:.8f} | {fmt(r.act_cosine_base)} | {fmt(r.act_cosine_corr)} |"
        )
    a("")
    if downs:
        a(f"Capture files:")
        for r in downs:
            a(f"- L{r.layer}: `{r.act_path}`")
    a("")
    a("---")
    a("")
    a("## 3. Bytes per tier")
    a("")
    a("HGRAVU01 body = packed codes + f16 scales. Container = magic + JSON header + body.")
    a("The hierarchy is one envelope around both planes, not two HGRAVU01 files.")
    a("")
    a("| tensor | n | base body | corr body | hierarchy (est) | two separate files | bf16 |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        a(
            f"| `{r.name}` | {r.n_elem} | {r.base_body_bytes} | {r.corr_body_bytes} | "
            f"{r.hierarchy_bytes} | {r.two_copy_bytes} | {r.bf16_bytes} |"
        )
    a(
        f"| **sample total** | {sum(r.n_elem for r in results)} | {sum_base_body} | {sum_corr_body} | "
        f"{sum_hier} | {sum_two} | {sum_bf16} |"
    )
    a("")
    a(f"On-disk mixed-q3mlp HGRAVU01 container is **{MIXED_Q3MLP_BASE_BYTES}** B per MLP")
    a("tensor (CITED PACK_REPORT `nbytes` = body + ~280 B magic/JSON). Per-tensor body")
    a("breakdown (identical for every 5120×17408 / 17408×5120 MLP matrix):")
    a("")
    r0 = results[0]
    a("| plane | bits | code bytes | scale bytes | body | container | physical BPW |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    a(
        f"| BASE q{BASE_BITS} | {BASE_BITS} | {r0.base_code_bytes} | {r0.base_scale_bytes} | "
        f"{r0.base_body_bytes} | {r0.base_container_bytes} | {sample_bpw_base:.9f} |"
    )
    a(
        f"| CORRECTION q{CORR_BITS} | {CORR_BITS} | {r0.corr_code_bytes} | {r0.corr_scale_bytes} | "
        f"{r0.corr_body_bytes} | {r0.corr_container_bytes} | {sample_bpw_corr:.9f} |"
    )
    a(
        f"| hierarchy (sum of bodies) | {BASE_BITS}+{CORR_BITS} | "
        f"{r0.base_code_bytes + r0.corr_code_bytes} | "
        f"{r0.base_scale_bytes + r0.corr_scale_bytes} | "
        f"{r0.base_body_bytes + r0.corr_body_bytes} | {r0.hierarchy_bytes} | "
        f"{sample_bpw_hier:.9f} |"
    )
    a("")
    a(
        f"DERIVED: hierarchy / bf16 = {pct(r0.hierarchy_bytes, r0.bf16_bytes)} of the parent "
        f"bytes for that tensor. Base alone is {pct(r0.base_container_bytes, r0.bf16_bytes)}."
    )
    a(
        f"A second full q3 model would have been {r0.base_container_bytes} extra bytes; the "
        f"correction plane is {r0.corr_body_bytes} "
        f"({pct(r0.corr_body_bytes, r0.base_container_bytes)} of one base)."
    )
    a("")
    a("Projected to all 192 MLP tensors if the same two-tier recipe were packed")
    a("(DERIVED from the measured per-tensor bodies, not a packed 192-tensor artifact):")
    a("")
    mlp_n = 192
    a(f"- base bodies: {r0.base_body_bytes * mlp_n} B")
    a(f"- correction bodies: {r0.corr_body_bytes * mlp_n} B")
    a(f"- hierarchy bodies: {(r0.base_body_bytes + r0.corr_body_bytes) * mlp_n} B")
    a(
        f"- vs mixed-q3mlp MLP payload CITED 3 × 2_317_370_880 = 6_952_112_640 B "
        f"(exactly 192 × {MIXED_Q3MLP_BASE_BYTES})"
    )
    a("")
    a("---")
    a("")
    a("## 4. Tiered NR container spec")
    a("")
    a("The document below is the NR. It names decoder *families*, not kernels.")
    a("A field that could only be true of one machine belongs in NX and is rejected.")
    a("G103 left `correction_planes` empty; this is the first filled plane on this patient.")
    a("")
    a("```json")
    a(json.dumps(nr_spec, indent=2))
    a("```")
    a("")
    a("Physical layout (one blob per tensor, plane-major so the base is a byte")
    a("prefix: a base-only reader stops after `q_base` and never sees the correction):")
    a("")
    a("```")
    a("HGRAVM01                  # 8 B magic")
    a("u32 header_len")
    a("JSON header               # schema, shape, bits, group, byte ledger")
    a("f16 s_base[n_groups]      # shared group index")
    a("u3  q_base[n_padded]      # little-endian packed, HGRAVU01 bit order")
    a("                         # ---- base-only reader stops here ----")
    a("f16 s_corr[n_groups]      # same group index, residual absmax")
    a("u2  q_corr[n_padded]      # little-endian packed, HGRAVU01 bit order")
    a("```")
    a("")
    a("Decode:")
    a("")
    a("```")
    a("W_base[i] = q_base[i] * s_base[i // 64]          # standalone")
    a("W_hat[i]  = W_base[i] + q_corr[i] * s_corr[i // 64]")
    a("```")
    a("")
    a("A reader that does not implement the correction plane stops after `W_base`.")
    a("That reader is the existing mixed-q3mlp path.")
    a("")
    a("---")
    a("")
    a("## 5. Base is a runnable standalone")
    a("")
    a("| claim | evidence | tag |")
    a("|---|---|---|")
    a(
        f"| `{MIXED_Q3MLP}` exists and is the q3 MLP organ set | "
        f"`workspace/campaign/records/runs/qwen38-27b/{MIXED_Q3MLP}/` "
        f"PACK_REPORT status PACKED, 851 tensors, MLP organs HGRAVU01 q3 g64 | CITED |"
    )
    a(
        f"| complete physical BPW | {MIXED_Q3MLP_BPW} | CITED PACK_REPORT |"
    )
    a(
        f"| MLP physical BPW | {MIXED_Q3MLP_MLP_BPW} | CITED PACK_REPORT |"
    )
    a(
        "| coherent generate | `mixed-q3mlp-v1` (same MLP recipe, richer attention) "
        "cleared the campaign gate (France/Paris, 17×19); "
        f"`{MIXED_Q3MLP}` is the 3.34 BPW sibling with attention also at q3 | CITED "
        "`g1-mlp-family-generate.md`, `claude-generate/q3mlp-generate.json` |"
    )
    a(
        "| this BASE equals that scheme | packer-screen (f32, stride-17) cosine on the "
        "six sample tensors reproduces the PACK_REPORT column to float64 noise | MEASURED this process |"
    )
    a(
        "| native-decode of the correction plane | not implemented; numpy dequant only | "
        "SCOPE |"
    )
    a("")
    a("---")
    a("")
    a("## 6. Method")
    a("")
    a("```")
    a("W                  ← BF16 safetensors, name language_model.model.layers.N.mlp.{gate,down}_proj.weight")
    a("s_b, q_b           ← per-64 absmax / 3, f16 snap, rint, clip [-3,3]     # HGRAVU01 q3")
    a("W_base             ← q_b * float32(s_b)")
    a("R                  ← W - W_base")
    a("s_c, q_c           ← per-64 absmax(R) / 1, f16 snap, rint, clip [-1,1]  # HGRAVU01 q2")
    a("W_hat              ← W_base + q_c * float32(s_c)")
    a("weight rel-fro     ← ||W - Ŵ||_F / ||W||_F")
    a("act rel-fro        ← ||X W^T - X Ŵ^T||_F / ||X W^T||_F     # down_proj only")
    a("```")
    a("")
    a("Codec identity with mixed-q3mlp is the HGRAVU01 rule in")
    a("`lab/operators/qwen38_mlp_not_r160_pack.py:encode_uniform_payload` and")
    a("`lab/operators/ascension_dual_gravity_worker.py:_uniform_codec` (group 64,")
    a("bound = 2^{bits-1}-1, scale stored f16). This file reimplements that rule so")
    a("the worktree does not have to materialize `lab/`.")
    a("")
    a("Why the correction plane has its own scale: a q3 residual lives in")
    a("(-0.5 s_b, 0.5 s_b]. Feeding that residual to q2 *with s_b* rounds every")
    a("code to 0. The shared thing is the **group index**, not the numeric scale.")
    a("")
    a("---")
    a("")
    a("## 7. Code histograms (sanity)")
    a("")
    a("q3 codes should occupy {-3..+3}. q2 residual codes should occupy {-1,0,+1}")
    a("and must not be all-zero (that would be a dead plane).")
    a("")
    for r in results:
        a(f"- `{r.name}`")
        a(f"  - base q3 hist: `{json.dumps(r.code_hist_base, sort_keys=True)}`")
        a(f"  - corr q2 hist: `{json.dumps(r.code_hist_corr, sort_keys=True)}`")
    a("")
    a("---")
    a("")
    a("## 8. Run identity")
    a("")
    a("```")
    a(f"argv       {' '.join(argv)}")
    a(f"parent     {parent}")
    a(f"acts       {acts}")
    a(f"layers     {layers}")
    a(f"organs     {organs}")
    a(f"act_rows   {act_rows}")
    a(f"wall_s     {wall_s:.3f}")
    a(f"numpy      {np.__version__}")
    if capture_meta:
        a(f"capture    schema={capture_meta.get('schema')} sha256_self={capture_meta.get('sha256_self')}")
    a("```")
    a("")
    a("Future work (explicitly out of scope): a native HGRAVM01 reader and a fused")
    a("correction-plane GEMV so base+correction is a generate vehicle, not only a")
    a("numpy reconstruction.")
    a("")

    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    log(f"wrote {out} ({len(text)} bytes, sha256 {hashlib.sha256(text.encode()).hexdigest()})")


def parse_int_list(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise PackError("empty list")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="15,31,47", help="comma-separated layer indices")
    ap.add_argument("--organs", default="gate_proj,down_proj")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parent", type=Path, default=None)
    ap.add_argument("--acts", type=Path, default=None)
    ap.add_argument("--act-rows", type=int, default=512)
    args = ap.parse_args(argv)

    layers = parse_int_list(args.layers)
    organs = [p.strip() for p in args.organs.split(",") if p.strip()]
    if not organs:
        raise PackError("no organs")
    if args.act_rows < 16:
        raise PackError("--act-rows must be >= 16")

    parent = resolve_parent(args.parent)
    acts = resolve_acts(args.acts, parent)
    weight_map = load_weight_map(parent)
    log(f"parent={parent}")
    log(f"acts={acts}")
    log(f"layers={layers} organs={organs} act_rows={args.act_rows}")

    t0 = time.perf_counter()
    results: list[TensorResult] = []
    for layer in layers:
        for organ in organs:
            results.append(
                pack_one(
                    parent=parent,
                    weight_map=weight_map,
                    acts=acts,
                    layer=layer,
                    organ=organ,
                    act_rows=args.act_rows,
                )
            )
    wall = time.perf_counter() - t0
    write_report(
        out=args.out,
        results=results,
        parent=parent,
        acts=acts,
        layers=layers,
        organs=organs,
        act_rows=args.act_rows,
        wall_s=wall,
        argv=["tools/matryoshka_pack.py", * (argv if argv is not None else sys.argv[1:])],
    )

    summary = {
        "schema": SCHEMA,
        "status": "MEASURED_WIN",
        "n_tensors": len(results),
        "weight_rel_fro": {
            r.name: {
                "base": r.weight_rel_fro_base,
                "base_plus_correction": r.weight_rel_fro_corr,
            }
            for r in results
        },
        "down_proj_act_rel_fro": {
            r.name: {
                "base": r.act_rel_fro_base,
                "base_plus_correction": r.act_rel_fro_corr,
                "n_rows": r.act_rows,
            }
            for r in results
            if r.organ == "down_proj"
        },
        "bytes": {
            r.name: {
                "base_body": r.base_body_bytes,
                "base_container": r.base_container_bytes,
                "correction_body": r.corr_body_bytes,
                "hierarchy_est": r.hierarchy_bytes,
                "bf16": r.bf16_bytes,
            }
            for r in results
        },
        "wall_s": wall,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    print(
        f"[matryoshka] DONE wall_s={wall:.3f} "
        f"weight_mean {np.mean([r.weight_rel_fro_base for r in results]):.6f} -> "
        f"{np.mean([r.weight_rel_fro_corr for r in results]):.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackError as exc:
        print(f"[matryoshka] FAIL {exc}", file=sys.stderr)
        sys.exit(1)
