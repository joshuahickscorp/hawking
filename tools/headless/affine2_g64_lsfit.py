#!/usr/bin/env python3
"""AFFINE2_G64_LSFIT: group-64 affine-2 with least-squares scale/bias.

How scale/bias were CURRENTLY chosen (the g32 native mix): min/max range.

    lo = group.min(); hi = group.max()
    scale = (hi - lo) / 3; bias = lo
    q = clip(round((w - bias) / scale), 0, 3)
    w_hat = q * scale + bias

That is an absmax/range rule, not least squares. The hypothesis is therefore
NOT refuted by reading the packer.

This harness:

1. Fits (scale, bias) by least squares per group: minmax-init the 4 equally
   spaced levels, assign q in {0,1,2,3}, solve the 2x2 normal equations for
   (scale, bias), snap to f16, iterate assignment+refit until codes stabilize.
2. Packs HGRAVF01 at group 64 (2 bits + f16 scale + f16 bias = 2.5 bpw body)
   using the existing affine2_group32_matvec kernel family, which now accepts
   group_size 32 or 64. No new codec family.
3. Builds the artifact on all 192 MLP tensors, attention left at q4, and
   decodes 16 tokens on the native runtime.

Does not load a second 27B. Does not write under ~/models. Does not touch
receipts/ascent-2026-08-16 or workspace/campaign.

    python3 tools/headless/affine2_g64_lsfit.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from first_noetic_executable import (  # noqa: E402
    PARENT_BF16,
    PARENT_PARAMS,
    PROMPT,
    Q4_INCUMBENT,
    Q4_INCUMBENT_EBPW,
    Q4_ROOT,
    TOKENIZER,
    SourceBF16,
    find_decode_binary,
    git_head,
    hardlink_or_copy,
    judge_coherence,
    load_q4_manifest,
    now_iso,
    organ_of,
    sha256_hex,
    write_atomic,
    write_catalog,
)

RECEIPT = REPO / "receipts" / "headless" / "AFFINE2_G64_LSFIT.json"
SCHEMA = "hawking.headless.affine2_g64_lsfit.v1"
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "QWEN38_AFFINE2_G64_ARTIFACT_ROOT",
        str(REPO / "artifacts" / "qwen38-affine2-g64-lsfit"),
    )
)

LAYERS = 64
GROUP_AFFINE = 64
AFFINE_BITS = 2
SCALE_BITS = 16
BIAS_BITS = 16
CODEC_AFFINE = 5
CODEC_Q4 = 3
CODEC_F32 = 4
MAGIC_AFFINE = b"HGRAVF01"
SCHEMA_AFFINE = "hawking.gravity.affine_scale_bias.v1"
AFFINE_REPR = "affine_q2_group64_fp16_scale_bias"
MAX_NEW = 16
MAX_SEQ = 128
MIX_ID = "mix_all_mlp_affine_g64_ls"
MAX_LS_ITERS = 16
LS_DET_EPS = 1e-6

NATIVE_KERNEL_GEO = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"
NATIVE_KERNEL_SERIAL = "qwen_affine_q2_group32_matvec"

Q3_EBPW = 3.616489579525877
AFFINE2_G32_EBPW = 3.457428992214862
Q3_TEXT = (
    "<think>\nThe user wants a detailed, prose explanation of how a compiler "
    "transforms a"
)
AFFINE2_G32_TEXT = (
    "<think>\n\n</thinking><think>\n\n</think>\n\nThis is a request for a detailed"
)
Q4_TEXT = Q3_TEXT  # q3 g64 reproduced the incumbent exactly

CENSUS_RE = re.compile(
    r"qwen38-decode mixed census: "
    r"tensors=(?P<tensors>\d+) binary=(?P<binary>\d+) residual=(?P<residual>\d+) "
    r"hgravs=(?P<hgravs>\d+) uniform=(?P<uniform>\d+)"
    r"(?: affine=(?P<affine>\d+))?"
    r" q4=(?P<q4>\d+) "
    r"f32=(?P<f32>\d+) refused=(?P<refused>\d+) "
    r"expanded_to_q4=(?P<expanded_to_q4>\d+) "
    r"expanded_to_float_gemv=(?P<expanded_to_float_gemv>\d+)"
)


class PackError(RuntimeError):
    pass


def affine_storage_bpw(group: int, *, bias: bool = True) -> float:
    extra = SCALE_BITS / float(group)
    if bias:
        extra += BIAS_BITS / float(group)
    return float(AFFINE_BITS) + extra


def current_fit_is_least_squares() -> bool:
    """The g32 packer is minmax/range. Hypothesis stands."""
    return False


def _assign_q(grouped: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(scale) > 0.0, scale, np.float32(1.0))
    q = np.clip(
        np.rint((grouped - bias[..., None]) / denom[..., None]),
        0,
        3,
    ).astype(np.uint8)
    return np.where(np.abs(scale)[..., None] > 0.0, q, np.uint8(0))


def _snap_f16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).astype(np.float32)


def fit_affine_minmax(
    grouped: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = grouped.min(axis=-1)
    hi = grouped.max(axis=-1)
    scale = _snap_f16(np.maximum((hi - lo) / np.float32(3.0), np.float32(1e-7)))
    bias = _snap_f16(lo.astype(np.float32))
    q = _assign_q(grouped, scale, bias)
    return scale, bias, q


def fit_affine_ls(
    grouped: np.ndarray,
    *,
    max_iters: int = MAX_LS_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Least-squares (scale, bias) for w ≈ q*scale + bias, q in {0,1,2,3}.

    Init from the 4-level minmax grid, then iterate assignment and the 2x2
    normal equations until codes stabilize or max_iters.
    """
    scale, bias, q = fit_affine_minmax(grouped)
    n = np.float32(grouped.shape[-1])
    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        qf = q.astype(np.float32)
        sq = qf.sum(axis=-1)
        sqq = (qf * qf).sum(axis=-1)
        sw = grouped.sum(axis=-1)
        sqw = (qf * grouped).sum(axis=-1)
        det = n * sqq - sq * sq
        mean = grouped.mean(axis=-1)
        ok = det > np.float32(LS_DET_EPS)
        det_safe = np.maximum(det, np.float32(1e-12))
        scale_ls = np.where(ok, (n * sqw - sq * sw) / det_safe, np.float32(0.0))
        bias_ls = np.where(ok, (sqq * sw - sq * sqw) / det_safe, mean)
        scale = _snap_f16(scale_ls)
        bias = _snap_f16(bias_ls)
        q_new = _assign_q(grouped, scale, bias)
        if np.array_equal(q_new, q):
            q = q_new
            break
        q = q_new
    return scale, bias, q, n_iters


def reconstruct_from_qsb(
    q: np.ndarray, scale: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    return q.astype(np.float32) * scale[..., None] + bias[..., None]


def pack_codes_lsb2(codes: np.ndarray) -> bytes:
    codes = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    n = int(codes.size)
    if n % 4 != 0:
        raise PackError(f"2-bit packer wants a multiple of 4 codes, got {n}")
    packed = np.zeros(n // 4, dtype=np.uint8)
    for shift in range(4):
        packed |= (codes[shift::4] & np.uint8(3)) << np.uint8(2 * shift)
    return packed.tobytes()


def pack_hgrafv01(
    weights: np.ndarray,
    group_size: int = GROUP_AFFINE,
    *,
    fit: str = "ls",
) -> bytes:
    if weights.ndim != 2:
        raise PackError(f"affine packer wants rank-2, got {weights.shape}")
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    if group_size not in (32, 64):
        raise PackError(f"affine kernel family accepts group 32 or 64, got {group_size}")
    if cols % group_size != 0:
        raise PackError(f"cols={cols} is not a multiple of group_size={group_size}")
    flat = np.ascontiguousarray(weights, dtype=np.float32)
    if not np.isfinite(flat).all():
        raise PackError("affine source is non-finite")
    groups_per_row = cols // group_size
    grouped = flat.reshape(rows, groups_per_row, group_size)
    if fit == "ls":
        scale, bias, q, _iters = fit_affine_ls(grouped)
        source = "ls_fit_parent_bf16"
    elif fit == "minmax":
        scale, bias, q = fit_affine_minmax(grouped)
        source = "fitted_minmax_parent_bf16"
    else:
        raise PackError(f"unknown affine fit {fit!r}")
    groups = rows * groups_per_row
    scale_f16 = scale.astype(np.float16)
    bias_f16 = bias.astype(np.float16)
    packed = pack_codes_lsb2(q)
    header = {
        "schema": SCHEMA_AFFINE,
        "representation": (
            "affine_q2_group64_fp16_scale_bias"
            if group_size == 64
            else "affine_q2_group32_fp16_scale_bias"
        ),
        "shape": [rows, cols],
        "elements": rows * cols,
        "bits": AFFINE_BITS,
        "group_size": group_size,
        "groups": int(groups),
        "scale_bytes": int(groups * 2),
        "bias_bytes": int(groups * 2),
        "code_bytes": int(len(packed)),
        "source": source,
        "fit": fit,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = scale_f16.tobytes() + bias_f16.tobytes() + packed
    if len(body) != header["scale_bytes"] + header["bias_bytes"] + header["code_bytes"]:
        raise PackError("HGRAVF01 body ledger drifted")
    return MAGIC_AFFINE + struct.pack("<I", len(header_bytes)) + header_bytes + body


def parse_hgrafv01(payload: bytes) -> dict[str, Any]:
    if payload[:8] != MAGIC_AFFINE:
        raise PackError(f"magic {payload[:8]!r} is not HGRAVF01")
    header_len = struct.unpack_from("<I", payload, 8)[0]
    header = json.loads(payload[12 : 12 + header_len])
    body = payload[12 + header_len :]
    expected = int(header["scale_bytes"]) + int(header["bias_bytes"]) + int(header["code_bytes"])
    if len(body) != expected:
        raise PackError("HGRAVF01 body length disagrees with ledger")
    return header


def reconstruct_hgrafv01(payload: bytes) -> np.ndarray:
    header = parse_hgrafv01(payload)
    header_len = struct.unpack_from("<I", payload, 8)[0]
    body = payload[12 + header_len :]
    rows, cols = int(header["shape"][0]), int(header["shape"][1])
    group = int(header["group_size"])
    groups = int(header["groups"])
    scale_bytes = int(header["scale_bytes"])
    bias_bytes = int(header["bias_bytes"])
    scales = np.frombuffer(body[:scale_bytes], dtype=np.float16).astype(np.float32)
    biases = np.frombuffer(body[scale_bytes : scale_bytes + bias_bytes], dtype=np.float16).astype(
        np.float32
    )
    packed = np.frombuffer(body[scale_bytes + bias_bytes :], dtype=np.uint8)
    n = rows * cols
    codes = np.empty(n, dtype=np.uint8)
    for shift in range(4):
        codes[shift::4] = (packed >> np.uint8(2 * shift)) & np.uint8(3)
    gpr = cols // group
    out = np.empty((rows, cols), dtype=np.float32)
    grouped = out.reshape(rows, gpr, group)
    q = codes.reshape(rows, gpr, group).astype(np.float32)
    grouped[...] = q * scales.reshape(rows, gpr)[..., None] + biases.reshape(rows, gpr)[..., None]
    if groups != rows * gpr:
        raise PackError("group count drifted in reconstruct")
    return out


def parent_key(catalog_name: str) -> str:
    key = catalog_name.replace("language_model.model.", "model.language_model.")
    if key == catalog_name and catalog_name.startswith("language_model."):
        key = "model." + catalog_name
    return key


def is_mlp_proj(name: str) -> bool:
    return (
        name.endswith("mlp.gate_proj.weight")
        or name.endswith("mlp.up_proj.weight")
        or name.endswith("mlp.down_proj.weight")
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float((d * d).mean())


def compile_mix(
    *,
    q4_root: Path = Q4_ROOT,
    parent: Path = PARENT_BF16,
    out_root: Path | None = None,
) -> dict[str, Any]:
    dest = Path(out_root or (ARTIFACTS_ROOT / MIX_ID))
    dest.mkdir(parents=True, exist_ok=True)
    segments_dir = dest / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_q4_manifest(q4_root)
    rows = list(manifest["tensors"])
    src = SourceBF16(parent)
    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    affine_names: list[str] = []
    t0 = time.perf_counter()
    payload_bytes = 0
    affine_bytes = 0
    q4_bytes = 0
    f32_bytes = 0
    n_hardlink = 0
    n_affine = 0
    n_attn_q4 = 0
    probe: dict[str, Any] | None = None
    for i, row in enumerate(rows):
        name = row["name"]
        shape = [int(x) for x in row["shape"]]
        elements = int(row["elements"])
        src_artifact = q4_root / "tensors" / row["artifact"]
        if not src_artifact.is_file():
            raise PackError(f"incumbent missing {src_artifact}")
        if not is_mlp_proj(name):
            filename = row["artifact"]
            dest_path = segments_dir / filename
            hardlink_or_copy(src_artifact, dest_path)
            n_hardlink += 1
            nbytes = int(dest_path.stat().st_size)
            codec = CODEC_Q4 if row["kind"] == "q4" else CODEC_F32
            codec_bpw = 8.0 * nbytes / max(elements, 1)
            digest = sha256_hex(filename.encode())
            if codec == CODEC_Q4:
                q4_bytes += nbytes
                if "self_attn." in name or "linear_attn." in name:
                    n_attn_q4 += 1
            else:
                f32_bytes += nbytes
        else:
            print(f"  [{MIX_ID}] affine-ls {name} group={GROUP_AFFINE}", flush=True)
            w = src.load(parent_key(name))
            if list(w.shape) != shape:
                raise PackError(f"{name} parent shape {list(w.shape)} != catalog {shape}")
            if probe is None:
                grouped = np.ascontiguousarray(w, dtype=np.float32).reshape(
                    int(w.shape[0]), int(w.shape[1]) // GROUP_AFFINE, GROUP_AFFINE
                )
                s_mm, b_mm, q_mm = fit_affine_minmax(grouped)
                s_ls, b_ls, q_ls, n_iters = fit_affine_ls(grouped)
                recon_mm = reconstruct_from_qsb(q_mm, s_mm, b_mm)
                recon_ls = reconstruct_from_qsb(q_ls, s_ls, b_ls)
                probe = {
                    "tensor": name,
                    "shape": shape,
                    "ls_iters": n_iters,
                    "minmax_mse": mse(recon_mm, grouped),
                    "ls_mse": mse(recon_ls, grouped),
                    "ls_beats_minmax": mse(recon_ls, grouped) <= mse(recon_mm, grouped) + 1e-12,
                }
                del grouped, s_mm, b_mm, q_mm, s_ls, b_ls, q_ls, recon_mm, recon_ls
            payload = pack_hgrafv01(w, GROUP_AFFINE, fit="ls")
            del w
            filename = hashlib.sha256(name.encode("utf-8")).hexdigest() + ".hgrafv01"
            dest_path = segments_dir / filename
            write_atomic(dest_path, payload)
            nbytes = len(payload)
            digest = sha256_hex(payload)
            codec = CODEC_AFFINE
            codec_bpw = affine_storage_bpw(GROUP_AFFINE)
            affine_names.append(name)
            affine_bytes += nbytes
            n_affine += 1
        payload_bytes += nbytes
        segments.append(
            {
                "id": i,
                "filename": filename,
                "bytes": nbytes,
                "sha256": digest,
            }
        )
        records.append(
            {
                "name": name,
                "codec": codec,
                "organ": organ_of(name),
                "shape": shape,
                "elements": elements,
                "segment_id": i,
                "offset": 0,
                "nbytes": nbytes,
                "sha256": digest,
                "codec_bpw": codec_bpw,
            }
        )
    catalog_path = dest / "catalog.hq38m20"
    write_catalog(catalog_path, records, segments)
    complete_ebpw = 8.0 * payload_bytes / PARENT_PARAMS
    storage_bpw = complete_ebpw
    active_bpw = complete_ebpw
    codecs = Counter(int(r["codec"]) for r in records)
    mlp_elements = sum(int(r["elements"]) for r in records if is_mlp_proj(r["name"]))
    report = {
        "mix_id": MIX_ID,
        "recipe": {
            "id": MIX_ID,
            "tensors": "mlp.gate_proj, mlp.up_proj, mlp.down_proj on layers 0..63",
            "codec": "HGRAVF01 affine_q2_group64 LS (w = q * scale + bias, unsigned q in {0,1,2,3})",
            "group": GROUP_AFFINE,
            "fit": "least_squares_scale_bias",
            "layers": list(range(LAYERS)),
            "organs": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            "attention": "HQ30UQ4 g64 (hardlinked incumbent)",
            "embed_head": "HQ30UQ4 / f32v2 (hardlinked incumbent)",
            "kernel": NATIVE_KERNEL_GEO,
            "kernel_family": "affine2_group32_matvec (group_size bind 32 or 64)",
            "why": (
                "affine2 g32 minmax at 3.00 bpw produced worse text than q3 g64 "
                "at 3.25 bpw. That implicated the FIT. This mix refits scale and "
                "bias by least squares at group 64 (2.5 bpw body)."
            ),
        },
        "artifact_root": str(dest),
        "catalog": str(catalog_path),
        "n_tensors": len(records),
        "n_affine": n_affine,
        "n_hardlink": n_hardlink,
        "n_attn_q4_hardlinked": n_attn_q4,
        "codecs": {str(k): int(v) for k, v in sorted(codecs.items())},
        "affine_tensors": affine_names,
        "affine_group": GROUP_AFFINE,
        "affine_tensor_storage_bpw": affine_storage_bpw(GROUP_AFFINE),
        "affine_bpw_billing": {
            "codes_bpw": 2.0,
            "scale_bpw": SCALE_BITS / float(GROUP_AFFINE),
            "bias_bpw": BIAS_BITS / float(GROUP_AFFINE),
            "total_bpw": affine_storage_bpw(GROUP_AFFINE),
            "group": GROUP_AFFINE,
            "scale_dtype": "fp16",
            "bias_dtype": "fp16",
        },
        "how_scale_bias_were_currently_chosen": {
            "g32_packer": "minmax / range: scale=(max-min)/3, bias=min, then round",
            "already_least_squares": False,
            "hypothesis_refuted_by_reading_the_packer": False,
            "note": (
                "tools/affine2_transcode.py copies MLX QuantizedLinear bytes; "
                "tools/headless/affine2_native_mlp.py pack_hgrafv01 is "
                "lo, hi = min/max; scale=(hi-lo)/3; bias=lo. Not LS."
            ),
        },
        "ls_fit": {
            "method": (
                "minmax-init 4 equally spaced levels, assign q in {0,1,2,3}, "
                "solve 2x2 normal equations for (scale, bias), snap f16, "
                f"iterate up to {MAX_LS_ITERS} until codes stabilize"
            ),
            "max_iters": MAX_LS_ITERS,
            "probe": probe,
        },
        "mlp_elements": mlp_elements,
        "payload_bytes": payload_bytes,
        "affine_bytes": affine_bytes,
        "q4_bytes": q4_bytes,
        "f32_bytes": f32_bytes,
        "parent_params": PARENT_PARAMS,
        "storage_bpw": storage_bpw,
        "active_bpw": active_bpw,
        "active_fused_bpw": active_bpw,
        "complete_ebpw": complete_ebpw,
        "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
        "wall_s": time.perf_counter() - t0,
        "did_not_load_second_27b": True,
        "parent_streamed_one_tensor_at_a_time": True,
        "wrote_under_models": False,
    }
    write_atomic(dest / "MIX_REPORT.json", json.dumps(report, indent=2).encode())
    print(
        f"[{MIX_ID}] tensors={len(records)} affine={n_affine} "
        f"ebpw={complete_ebpw:.6f} (q4 {Q4_INCUMBENT_EBPW:.6f}) "
        f"mlp_bpw={affine_storage_bpw(GROUP_AFFINE):.4f} "
        f"in {report['wall_s']:.1f}s",
        flush=True,
    )
    return report


def parse_census(stderr: str) -> dict[str, int] | None:
    m = CENSUS_RE.search(stderr)
    if not m:
        return None
    out: dict[str, int] = {}
    for k, v in m.groupdict().items():
        if v is None:
            continue
        out[k] = int(v)
    return out


def parse_bind(stderr: str) -> str | None:
    affine = None
    generic = None
    for line in stderr.splitlines():
        if "HGRAVF01 affine2" in line:
            affine = line.strip()
        if generic is None and "qwen38-decode mixed bind:" in line:
            generic = line.strip()
    return affine or generic


def decode_mix(
    artifact_root: Path,
    *,
    binary: Path | None = None,
    prompt: str = PROMPT,
    max_new: int = MAX_NEW,
    max_seq: int = MAX_SEQ,
    tokenizer: Path = TOKENIZER,
) -> dict[str, Any]:
    exe = binary or find_decode_binary()
    out_json = artifact_root / "decode.json"
    cmd = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--tokenizer",
        str(tokenizer),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new),
        "--max-seq-len",
        str(max_seq),
        "--out",
        str(out_json),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    wall_s = time.perf_counter() - t0
    stdout = proc.stdout
    stderr = proc.stderr
    result: dict[str, Any] = {
        "command": cmd,
        "binary": str(exe),
        "exit_code": proc.returncode,
        "wall_s": wall_s,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }
    if proc.returncode != 0:
        result["ok"] = False
        result["generated_text"] = None
        result["generated_text_verbatim"] = None
        result["coherence"] = {
            "coherent": False,
            "reason": f"decode exit {proc.returncode}",
        }
        result["census"] = parse_census(stderr)
        result["bind"] = parse_bind(stderr)
        return result
    body: dict[str, Any] = {}
    if out_json.is_file():
        body = json.loads(out_json.read_text())
    text = body.get("generated_text")
    if text is None:
        for line in stdout.splitlines():
            if line.startswith("GENERATED_TEXT_VERBATIM: "):
                text = line[len("GENERATED_TEXT_VERBATIM: ") :]
                break
    ids = [int(x) for x in body.get("new_token_ids") or []]
    decode_steps = int(body.get("decode_steps") or max(len(ids), 1))
    decode_wall_ns = int(body.get("decode_wall_ns") or 0)
    tok_s = None
    if decode_wall_ns > 0 and decode_steps > 0:
        tok_s = decode_steps / (decode_wall_ns / 1e9)
    elif wall_s > 0 and ids:
        tok_s = len(ids) / wall_s
    census = parse_census(stderr)
    bind = parse_bind(stderr)
    saw_affine = (
        NATIVE_KERNEL_GEO in stderr
        or NATIVE_KERNEL_SERIAL in stderr
        or "HGRAVF01 affine2" in stderr
        or (census is not None and census.get("affine", 0) > 0)
    )
    native = "qwen38-decode mixed HQ38M20" in stderr or "mixed bind" in stderr
    expanded = int((census or {}).get("expanded_to_q4") or 0)
    expanded_float = int((census or {}).get("expanded_to_float_gemv") or 0)
    dequant = expanded > 0 or expanded_float > 0
    result.update(
        {
            "ok": True,
            "prompt": body.get("prompt") or prompt,
            "generated_text": text if text is not None else "",
            "generated_text_verbatim": text if text is not None else "",
            "new_token_ids": ids,
            "n_new_tokens": len(ids),
            "fallbacks": int(body.get("fallbacks") or 0),
            "dense_w_materialized": int(body.get("dense_w_materialized") or 0),
            "expanded_to_q4": expanded,
            "expanded_to_float_gemv": expanded_float,
            "prompt_ids": body.get("prompt_ids"),
            "decode_wall_ns": decode_wall_ns,
            "decode_steps": decode_steps,
            "tok_s": tok_s,
            "median_gpu_ns_per_token": body.get("median_gpu_ns_per_token"),
            "native_kernel_ran": bool((native or saw_affine) and not dequant),
            "dequant_path": bool(dequant),
            "stderr_saw_mixed_catalog": "HQ38M20" in stderr,
            "stderr_saw_affine_kernel": saw_affine,
            "coherence": judge_coherence(text or "", ids),
            "census": census,
            "bind": bind,
        }
    )
    return result


def run_parity(binary: Path | None = None) -> dict[str, Any]:
    candidates = []
    if binary is not None:
        candidates.append(binary)
    env = os.environ.get("AFFINE2_PARITY_BIN")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            REPO / "workspace/ops/build/rust/release-fast/examples/affine2_parity",
            Path.home()
            / "Downloads/hawking-copy/workspace/ops/build/rust/release-fast/examples/affine2_parity",
        ]
    )
    exe = next((p for p in candidates if p.is_file()), None)
    if exe is None:
        return {"ok": False, "reason": "affine2_parity binary not built"}
    proc = subprocess.run(
        [str(exe), "--synthetic", "--group", "64"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    parsed: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "binary": str(exe),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
    }
    for line in proc.stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            parsed[k.strip()] = v.strip()
    return parsed


def comparison_table(
    compiled: dict[str, Any], decoded: dict[str, Any] | None
) -> list[dict[str, Any]]:
    ours_text = None
    if decoded:
        ours_text = decoded.get("generated_text_verbatim")
    return [
        {
            "codec": "q4 incumbent",
            "body_bpw": 4.25,
            "complete_ebpw": Q4_INCUMBENT_EBPW,
            "text_verbatim": Q4_TEXT,
            "note": "reference",
        },
        {
            "codec": "q3 g64",
            "body_bpw": 3.25,
            "complete_ebpw": Q3_EBPW,
            "text_verbatim": Q3_TEXT,
            "note": "identical to incumbent",
        },
        {
            "codec": "affine2 g32",
            "body_bpw": 3.00,
            "complete_ebpw": AFFINE2_G32_EBPW,
            "text_verbatim": AFFINE2_G32_TEXT,
            "note": "degraded, think-tag stutter; minmax/range fit",
        },
        {
            "codec": "affine2 g64 LS",
            "body_bpw": 2.50,
            "complete_ebpw": compiled.get("complete_ebpw"),
            "text_verbatim": ours_text,
            "note": "least-squares scale+bias, group 64",
        },
    ]


def run_all(*, decode: bool = True, out_receipt: Path = RECEIPT) -> dict[str, Any]:
    t0 = time.perf_counter()
    print(f"== compile {MIX_ID} ==", flush=True)
    compiled = compile_mix()
    decoded = None
    if decode:
        print(f"== decode {MIX_ID} ==", flush=True)
        decoded = decode_mix(Path(compiled["artifact_root"]))
        gen = (decoded or {}).get("generated_text_verbatim")
        coh = (decoded or {}).get("coherence") or {}
        print(
            f"[{MIX_ID}] exit={decoded.get('exit_code')} "
            f"coherent={coh.get('coherent')} text={gen!r}",
            flush=True,
        )
    print("== affine2_parity --synthetic --group 64 ==", flush=True)
    parity = run_parity()
    print(
        f"parity ok={parity.get('ok')} status={parity.get('status')} "
        f"max_abs_diff={parity.get('max_abs_diff')}",
        flush=True,
    )

    chosen = None
    if decoded and decoded.get("ok"):
        chosen = {
            "mix_id": compiled["mix_id"],
            "recipe": compiled["recipe"],
            "artifact_root": compiled["artifact_root"],
            "exact_mix": {
                "tensors": compiled["recipe"]["tensors"],
                "codec": compiled["recipe"]["codec"],
                "group": compiled["recipe"]["group"],
                "fit": compiled["recipe"]["fit"],
                "layers": compiled["recipe"]["layers"],
                "attention": compiled["recipe"]["attention"],
                "affine_tensors": compiled["affine_tensors"],
            },
            "storage_bpw": compiled["storage_bpw"],
            "active_bpw": compiled["active_bpw"],
            "complete_ebpw": compiled["complete_ebpw"],
            "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
            "affine_tensor_storage_bpw": compiled["affine_tensor_storage_bpw"],
            "affine_bpw_billing": compiled["affine_bpw_billing"],
            "prompt": decoded.get("prompt"),
            "generated_text_verbatim": decoded.get("generated_text_verbatim"),
            "new_token_ids": decoded.get("new_token_ids"),
            "n_new_tokens": decoded.get("n_new_tokens"),
            "tok_s": decoded.get("tok_s"),
            "native_kernel_ran": decoded.get("native_kernel_ran"),
            "dequant_path": decoded.get("dequant_path"),
            "fallbacks": decoded.get("fallbacks"),
            "dense_w_materialized": decoded.get("dense_w_materialized"),
            "expanded_to_q4": decoded.get("expanded_to_q4", 0),
            "expanded_to_float_gemv": decoded.get("expanded_to_float_gemv", 0),
            "prompt_ids": decoded.get("prompt_ids"),
            "coherence": decoded.get("coherence"),
            "census": decoded.get("census"),
            "bind": decoded.get("bind"),
        }

    table = comparison_table(compiled, decoded)
    gen_text = (decoded or {}).get("generated_text_verbatim")
    coh = ((decoded or {}).get("coherence") or {}) if decoded else {}
    finding = (
        "LS fitting at group 64 is the measurement. The 16-token text is "
        "reported verbatim; a think-tag stutter is a real defect and is not "
        "summarized as coherent."
    )
    if decoded and decoded.get("ok"):
        if gen_text == Q4_TEXT:
            finding = (
                "LS-fitted affine2 g64 reproduced the incumbent 16-token text. "
                "The g32 degradation was the minmax/range fit, not the 4-level "
                "quantizer itself."
            )
        elif gen_text == AFFINE2_G32_TEXT:
            finding = (
                "LS fitting did NOT fix the degradation: the 16-token text "
                "matches affine2 g32's think-tag stutter. The loss is in the "
                "4-level quantizer itself rather than in how its parameters "
                "are chosen."
            )
        elif coh.get("repeated_single_token"):
            finding = (
                "LS-fitted affine2 g64 collapsed to a repeated token. "
                "Sixteen copies of one token is not coherence."
            )
        elif "thinking" in (gen_text or "") and "</thinking>" in (gen_text or ""):
            finding = (
                "LS fitting did NOT fix the degradation: the 16-token text "
                f"still stutters the think tag. Verbatim: {gen_text!r}. "
                "That points at the 4-level quantizer itself, not only at "
                "how scale/bias were chosen."
            )
        else:
            finding = (
                "LS-fitted affine2 g64 generated 16 tokens that match neither "
                "the incumbent nor the g32 stutter. Verbatim: "
                f"{gen_text!r}. Coherence judge: {coh}."
            )

    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Fit affine2 scale/bias by least squares, move the existing "
            "affine2_group32_matvec family from group 32 to group 64, build "
            "the 192-MLP artifact with attention at q4, and decode 16 tokens."
        ),
        "hypothesis": {
            "claim": (
                "affine2 g32 spends more parameters per weight than q3 g64 "
                "and still produces worse text, so the FIT is implicated, "
                "not the bit budget."
            ),
            "current_fit": "minmax/range (scale=(max-min)/3, bias=min)",
            "already_least_squares": False,
            "refuted_by_reading_the_packer": False,
        },
        "kernel_family": {
            "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
            "production_kernels": [NATIVE_KERNEL_SERIAL, NATIVE_KERNEL_GEO],
            "container": "HGRAVF01 hawking.gravity.affine_scale_bias.v1",
            "reconstruction": "w = float(q) * scale + bias, q in {0,1,2,3}, group 32 or 64",
            "did_not_write_a_new_codec_family": True,
            "group_size_now": 64,
        },
        "q4_incumbent": Q4_INCUMBENT,
        "parent_bf16": str(PARENT_BF16),
        "parent_params": PARENT_PARAMS,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "compile": compiled,
        "decode": decoded,
        "parity": parity,
        "chosen": chosen,
        "comparison": table,
        "generation_finding": finding,
        "elapsed_s": time.perf_counter() - t0,
    }
    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_receipt.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp, out_receipt)
    print(f"wrote {out_receipt}", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    decode = "--pack-only" not in args
    run_all(decode=decode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
