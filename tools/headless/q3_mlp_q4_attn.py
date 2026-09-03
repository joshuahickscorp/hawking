#!/usr/bin/env python3
"""Q3-MLP g64 + Q4-attention mix: compile, native-decode, write the receipt.

Doctor v2 prescribes in-register grouped-absmax q3 at group 64 for the MLP
and q4 for GQA attention (do not apply the MLP policy to attention).
NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64 showed q3 g64 on every MLP tensor
survives the hidden-state loop (argmax 9714 = teacher). This harness
builds that mix as an HQ38M20 catalog the native runtime can bind, then
decodes at least 16 tokens.

MLP tensors are packed as HGRAVU01 bits=3 group=64 (same family as the
shipping q4 kernel; fused launch is qwen_uniform_q3_group64_matvec_geo_tpr64_tg128).
Attention (and every non-MLP GEMV) stays HQ30UQ4 from the incumbent,
hardlinked. Does not load a second 27B. Streams one parent tensor at a
time. Does not write under ~/models.

    python3 tools/headless/q3_mlp_q4_attn.py
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

from first_noetic_executable import (
    CODEC_F32,
    CODEC_Q4,
    PARENT_BF16,
    PARENT_PARAMS,
    PROMPT,
    Q4_INCUMBENT,
    Q4_INCUMBENT_EBPW,
    Q4_ROOT,
    REPO,
    TOKENIZER,
    SourceBF16,
    find_decode_binary,
    hardlink_or_copy,
    judge_coherence,
    load_q4_manifest,
    now_iso,
    organ_of,
    sha256_hex,
    write_atomic,
    write_catalog,
)

RECEIPT = REPO / "receipts" / "headless" / "NOETIC_Q3_MLP_Q4_ATTN.json"
SCHEMA = "hawking.headless.noetic_q3_mlp_q4_attn.v1"
MIX_ID = "mix_all_mlp_hgravu01_q3_g64"
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "QWEN38_Q3MLP_ARTIFACT_ROOT",
        str(REPO / "artifacts" / "qwen38-q3mlp-q4attn"),
    )
)

LAYERS = 64
GROUP_Q3 = 64
BITS_Q3 = 3
SCALE_BITS = 16
Q3_BOUND = (1 << (BITS_Q3 - 1)) - 1  # 3
MAX_NEW = 16
MAX_SEQ = 128

MAGIC_UNIFORM = b"HGRAVU01"
SCHEMA_UNIFORM = "hawking.gravity.uniform_group.v1"
REPRESENTATION_Q3 = "uniform_q3_group_scale"
NATIVE_Q3_KERNEL = "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128"
# First 16 greedy ids of the sealed uniform-q4 incumbent on this prompt
# (FIRST_NOETIC_EXECUTABLE mix_a). Used only as a comparison, not a gate.
Q4_INCUMBENT_FIRST_16_IDS = [
    248068,
    198,
    760,
    1156,
    6587,
    264,
    11346,
    11,
    58655,
    15673,
    314,
    1204,
    264,
    18826,
    27545,
    264,
]

CODEC_UNIFORM = 3  # HQ38M20 codec 3 is HGRAVU01 or HQ30UQ4, distinguished by magic


class PackError(RuntimeError):
    pass


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def artifact_filename(name: str, ext: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest() + f".{ext}"


def q3_storage_bpw(group: int = GROUP_Q3, bits: int = BITS_Q3) -> float:
    return float(bits) + (SCALE_BITS / float(group))


def packed_byte_count(count: int, bits: int) -> int:
    if bits <= 0 or bits > 8:
        raise PackError(f"packed bit width {bits} is not 1..=8")
    return (count * bits + 7) // 8


def pack_unsigned_lsb(codes: np.ndarray, bits: int) -> bytes:
    """LSB-first unsigned pack. Matches hawking-core pack_unsigned_lsb."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    n = int(codes.size)
    if n == 0:
        return b""
    if bits == 3 and n % 8 == 0:
        c = codes.reshape(-1, 8).astype(np.uint16)
        b0 = (c[:, 0] & 7) | ((c[:, 1] & 7) << 3) | ((c[:, 2] & 3) << 6)
        b1 = ((c[:, 2] >> 2) & 1) | ((c[:, 3] & 7) << 1) | ((c[:, 4] & 7) << 4) | (
            (c[:, 5] & 1) << 7
        )
        b2 = ((c[:, 5] >> 1) & 3) | ((c[:, 6] & 7) << 2) | ((c[:, 7] & 7) << 5)
        return np.stack([b0, b1, b2], axis=1).astype(np.uint8).ravel().tobytes()
    out = bytearray(packed_byte_count(n, bits))
    for i, code in enumerate(codes.tolist()):
        for b in range(bits):
            if (int(code) >> b) & 1:
                bit_index = i * bits + b
                out[bit_index >> 3] |= 1 << (bit_index & 7)
    return bytes(out)


def extract_unsigned(packed: bytes, element: int, bits: int) -> int:
    bit0 = element * bits
    value = 0
    for b in range(bits):
        bit_index = bit0 + b
        value |= ((packed[bit_index >> 3] >> (bit_index & 7)) & 1) << b
    return value


def is_mlp_proj(name: str) -> bool:
    return (
        name.endswith("mlp.gate_proj.weight")
        or name.endswith("mlp.up_proj.weight")
        or name.endswith("mlp.down_proj.weight")
    )


def pack_hgravu01(
    weights: np.ndarray,
    bits: int = BITS_Q3,
    group_size: int = GROUP_Q3,
) -> bytes:
    """HGRAVU01 grouped-absmax. Matches hawking-core pack_uniform_factor.

    Quantizes with the f32 group scale, stores that scale as f16. Codes are
    offset-binary (signed + bound) packed LSB-first. Requires cols % group.
    """
    if weights.ndim != 2:
        raise PackError(f"HGRAVU01 packer wants rank-2, got {weights.shape}")
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    if bits < 2 or bits > 8 or group_size <= 0:
        raise PackError(f"uniform geometry bits={bits} group={group_size} is invalid")
    if cols % group_size != 0:
        raise PackError(f"cols={cols} is not a multiple of group_size={group_size}")
    flat = np.ascontiguousarray(weights, dtype=np.float32)
    if not np.isfinite(flat).all():
        raise PackError("HGRAVU01 source is non-finite")
    elements = rows * cols
    groups_per_row = cols // group_size
    groups = rows * groups_per_row
    bound = (1 << (bits - 1)) - 1
    grouped = flat.reshape(rows, groups_per_row, group_size)
    max_abs = np.abs(grouped).max(axis=-1).astype(np.float32)
    scale_f32 = max_abs / np.float32(bound)
    scales_f16 = scale_f32.astype(np.float16)
    denom = np.where(scale_f32 > 0.0, scale_f32, np.float32(1.0))
    signed = np.rint(grouped / denom[..., None]).clip(-bound, bound).astype(np.int16)
    codes = (signed + np.int16(bound)).astype(np.uint8).reshape(-1)
    packed_codes = pack_unsigned_lsb(codes, bits)
    expected = packed_byte_count(groups * group_size, bits)
    if len(packed_codes) != expected:
        raise PackError(
            f"HGRAVU01 packed codes {len(packed_codes)} != expected {expected}"
        )
    scale_bytes = groups * 2
    header = {
        "schema": SCHEMA_UNIFORM,
        "representation": REPRESENTATION_Q3,
        "shape": [rows, cols],
        "elements": elements,
        "bits": int(bits),
        "group_size": int(group_size),
        "groups": int(groups),
        "bound": int(bound),
        "scale_bytes": int(scale_bytes),
        "code_bytes": int(len(packed_codes)),
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    body = scales_f16.tobytes(order="C") + packed_codes
    if len(body) != scale_bytes + len(packed_codes):
        raise PackError("HGRAVU01 body ledger drifted")
    return MAGIC_UNIFORM + struct.pack("<I", len(header_bytes)) + header_bytes + body


def parse_hgravu01(payload: bytes) -> dict[str, Any]:
    if payload[:8] != MAGIC_UNIFORM:
        raise PackError(f"magic {payload[:8]!r} is not HGRAVU01")
    header_len = struct.unpack_from("<I", payload, 8)[0]
    header = json.loads(payload[12 : 12 + header_len])
    body = payload[12 + header_len :]
    scale_bytes = int(header["scale_bytes"])
    code_bytes = int(header["code_bytes"])
    if len(body) != scale_bytes + code_bytes:
        raise PackError("HGRAVU01 body length disagrees with ledger")
    header["_body_off"] = 12 + header_len
    return header


def dequant_hgravu01(payload: bytes) -> np.ndarray:
    """CPU oracle of the packed codes. Tests only; not a runtime path."""
    header = parse_hgravu01(payload)
    rows, cols = int(header["shape"][0]), int(header["shape"][1])
    bits = int(header["bits"])
    group = int(header["group_size"])
    bound = int(header.get("bound") or ((1 << (bits - 1)) - 1))
    groups = int(header["groups"])
    body_off = int(header["_body_off"])
    scales = np.frombuffer(
        payload[body_off : body_off + groups * 2], dtype=np.float16
    ).astype(np.float32)
    packed = payload[body_off + groups * 2 :]
    out = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            element = row * cols + col
            g = element // group
            code = extract_unsigned(packed, element, bits)
            out[row, col] = (int(code) - bound) * float(scales[g])
    return out


def mix_recipe() -> dict[str, Any]:
    return {
        "id": MIX_ID,
        "tensors": "mlp.gate_proj, mlp.up_proj, mlp.down_proj on layers 0..63",
        "codec": "HGRAVU01 grouped-absmax q3",
        "bits": BITS_Q3,
        "group": GROUP_Q3,
        "layers": list(range(LAYERS)),
        "organs": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "attention": (
            "HQ30UQ4 g64 (hardlinked incumbent). Doctor v2 names q4 g128 as a "
            "2.9% scale-overhead save, not a new bit regime; this mix does not "
            "apply the MLP q3 policy to attention."
        ),
        "other_mlp": "none — every MLP GEMV is HGRAVU01 q3 g64",
        "why": (
            "DOCTOR_V2_PRESCRIPTION: in-register grouped-absmax q3 matvec, "
            "group 64, for the MLP; q4 for GQA attention. "
            "NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64: q3 g64 on every MLP tensor "
            "survives the 64-layer hidden-state loop (scale-aware 0.9263 vs "
            "null 0.4474, student argmax 9714 = teacher). Ternary at 1.85 bpw "
            "fails (argmax 10895)."
        ),
        "do_not": (
            "do not apply the MLP q3 policy to attention "
            "(ATTENTION_FLOOR_REFIT: GQA stays gated at Q4)"
        ),
        "native_kernel": NATIVE_Q3_KERNEL,
    }


def compile_mix(
    *,
    q4_root: Path = Q4_ROOT,
    parent: Path = PARENT_BF16,
    out_root: Path | None = None,
) -> dict[str, Any]:
    recipe = mix_recipe()
    dest = Path(out_root or (ARTIFACTS_ROOT / MIX_ID))
    dest.mkdir(parents=True, exist_ok=True)
    segments_dir = dest / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_q4_manifest(q4_root)
    rows = list(manifest["tensors"])
    src = SourceBF16(parent)
    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    q3_names: list[str] = []
    t0 = time.perf_counter()
    payload_bytes = 0
    q3_bytes = 0
    q4_bytes = 0
    f32_bytes = 0
    n_hardlink = 0
    n_q3 = 0
    n_attn_q4 = 0
    for i, row in enumerate(rows):
        name = row["name"]
        shape = [int(x) for x in row["shape"]]
        elements = int(row["elements"])
        src_artifact = q4_root / "tensors" / row["artifact"]
        if not src_artifact.is_file():
            raise PackError(f"incumbent missing {src_artifact}")
        if is_mlp_proj(name):
            parent_key = name.replace("language_model.model.", "model.language_model.")
            if parent_key == name and name.startswith("language_model."):
                parent_key = "model." + name
            print(f"  [{MIX_ID}] q3 {name} group={GROUP_Q3}", flush=True)
            w = src.load(parent_key)
            if list(w.shape) != shape:
                raise PackError(
                    f"{name} parent shape {list(w.shape)} != catalog {shape}"
                )
            payload = pack_hgravu01(w, BITS_Q3, GROUP_Q3)
            del w
            filename = artifact_filename(name, "hgravu01")
            dest_path = segments_dir / filename
            write_atomic(dest_path, payload)
            nbytes = len(payload)
            digest = sha256_hex(payload)
            codec = CODEC_UNIFORM
            codec_bpw = q3_storage_bpw()
            q3_names.append(name)
            q3_bytes += nbytes
            n_q3 += 1
        else:
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
    report = {
        "mix_id": MIX_ID,
        "recipe": recipe,
        "artifact_root": str(dest),
        "catalog": str(catalog_path),
        "n_tensors": len(records),
        "n_q3_mlp": n_q3,
        "n_hardlink": n_hardlink,
        "n_attn_q4_hardlinked": n_attn_q4,
        "codecs": {str(k): int(v) for k, v in sorted(codecs.items())},
        "q3_tensors": q3_names,
        "q3_group": GROUP_Q3,
        "q3_bits": BITS_Q3,
        "q3_storage_bpw": q3_storage_bpw(),
        "payload_bytes": payload_bytes,
        "q3_bytes": q3_bytes,
        "q4_bytes": q4_bytes,
        "f32_bytes": f32_bytes,
        "parent_params": PARENT_PARAMS,
        "storage_bpw": storage_bpw,
        "active_bpw": active_bpw,
        "active_fused_bpw": active_bpw,
        "complete_ebpw": complete_ebpw,
        "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
        "beside_q4": {
            "storage_bpw": {
                "mix": storage_bpw,
                "q4_incumbent": Q4_INCUMBENT_EBPW,
            },
            "active_bpw": {
                "mix": active_bpw,
                "q4_incumbent": Q4_INCUMBENT_EBPW,
            },
            "complete_ebpw": {
                "mix": complete_ebpw,
                "q4_incumbent": Q4_INCUMBENT_EBPW,
            },
        },
        "wall_s": time.perf_counter() - t0,
        "did_not_load_second_27b": True,
        "parent_streamed_one_tensor_at_a_time": True,
        "wrote_under_models": False,
        "native_q3_kernel": NATIVE_Q3_KERNEL,
        "accounting": {
            "complete_ebpw": "8 * executable_payload_bytes / parent_params",
            "storage_bpw": (
                "equals complete_ebpw: fused in-register kernels, scales counted, "
                "no dense-W cache"
            ),
            "active_bpw": (
                "equals storage_bpw on the fused path "
                f"({NATIVE_Q3_KERNEL} + HQ30UQ4)"
            ),
            "q3_tensor_storage_bpw": "3 + 16/group (offset-binary + f16 scale)",
            "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
        },
    }
    write_atomic(dest / "MIX_REPORT.json", json.dumps(report, indent=2).encode())
    print(
        f"[{MIX_ID}] tensors={len(records)} q3_mlp={n_q3} "
        f"ebpw={complete_ebpw:.6f} (q4 {Q4_INCUMBENT_EBPW:.6f}) "
        f"in {report['wall_s']:.1f}s",
        flush=True,
    )
    return report


# hawking-copy binary prints an extra affine= field the worktree source
# does not yet. Both layouts must parse.
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
    for line in stderr.splitlines():
        if "qwen38-decode mixed bind:" in line:
            return line.strip()
    return None


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
    inner = [
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
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    if lock.is_file():
        cmd = ["bash", str(lock), "g012-q3mlp-q4attn", *inner]
    else:
        cmd = inner
    env = os.environ.copy()
    # Default-on recon fuse binds the geo_tpr64 q3 kernel. Do not opt out.
    env.pop("HAWKING_QWEN38_RECON_FUSE", None)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
        env=env,
    )
    wall_s = time.perf_counter() - t0
    stdout = proc.stdout
    stderr = proc.stderr
    result: dict[str, Any] = {
        "command": cmd,
        "binary": str(exe),
        "exit_code": proc.returncode,
        "wall_s": wall_s,
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
    }
    census = parse_census(stderr)
    bind = parse_bind(stderr)
    saw_q3_kernel = NATIVE_Q3_KERNEL in stderr
    saw_mixed = "qwen38-decode mixed HQ38M20" in stderr or "mixed HQ38M20" in stderr
    recon_fuse_on = "recon_fuse=ON" in (bind or "") or (
        "recon_fuse=ON" in stderr
    )
    expanded_to_q4 = int((census or {}).get("expanded_to_q4") or 0)
    expanded_to_float = int((census or {}).get("expanded_to_float_gemv") or 0)
    dequant = (
        expanded_to_q4 > 0
        or expanded_to_float > 0
        or ("reconstruct-to-Q4" in stderr and "no reconstruct-to-Q4" not in stderr)
    )
    if proc.returncode != 0:
        result.update(
            {
                "ok": False,
                "generated_text": None,
                "generated_text_verbatim": None,
                "census": census,
                "bind": bind,
                "native_kernel_ran": False,
                "dequant_path": bool(dequant),
                "stderr_saw_q3_kernel": saw_q3_kernel,
                "coherence": {
                    "coherent": False,
                    "reason": f"decode exit {proc.returncode}",
                },
            }
        )
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
    uniform_n = int((census or {}).get("uniform") or 0)
    native = (
        saw_mixed
        and recon_fuse_on
        and uniform_n == 192
        and expanded_to_q4 == 0
        and expanded_to_float == 0
        and not dequant
    )
    result.update(
        {
            "ok": True,
            "prompt": body.get("prompt") or prompt,
            "prompt_ids": body.get("prompt_ids"),
            "generated_text": text if text is not None else "",
            "generated_text_verbatim": text if text is not None else "",
            "new_token_ids": ids,
            "n_new_tokens": len(ids),
            "fallbacks": int(body.get("fallbacks") or 0),
            "dense_w_materialized": int(body.get("dense_w_materialized") or 0),
            "expanded_to_q4": expanded_to_q4,
            "expanded_to_float_gemv": expanded_to_float,
            "decode_wall_ns": decode_wall_ns,
            "decode_steps": decode_steps,
            "tok_s": tok_s,
            "median_gpu_ns_per_token": body.get("median_gpu_ns_per_token"),
            "native_kernel_ran": bool(native),
            "dequant_path": bool(dequant),
            "decode_kernel": NATIVE_Q3_KERNEL if native else None,
            "stderr_saw_mixed_catalog": saw_mixed,
            "stderr_saw_q3_kernel": saw_q3_kernel,
            "recon_fuse_on": recon_fuse_on,
            "census": census,
            "census_line": None
            if census is None
            else (
                f"tensors={census['tensors']} binary={census['binary']} "
                f"residual={census['residual']} hgravs={census['hgravs']} "
                f"uniform={census['uniform']} q4={census['q4']} "
                f"f32={census['f32']} refused={census['refused']} "
                f"expanded_to_q4={census['expanded_to_q4']} "
                f"expanded_to_float_gemv={census['expanded_to_float_gemv']}"
            ),
            "bind": bind,
            "coherence": judge_coherence(text or "", ids),
        }
    )
    return result


def write_receipt(
    compiled: dict[str, Any],
    decoded: dict[str, Any] | None,
    *,
    elapsed_s: float,
    out_receipt: Path = RECEIPT,
) -> dict[str, Any]:
    sixteen_copies_is_not_coherence = (
        "Sixteen copies of one token is not coherence. "
        "Hidden-state survival (NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64) does "
        "not automatically predict generation; a collapse here is a real "
        "finding, not a packing bug to paper over."
    )
    coherence = (decoded or {}).get("coherence") if decoded else None
    generation_finding = None
    if decoded and decoded.get("ok"):
        if coherence and coherence.get("repeated_single_token"):
            generation_finding = (
                "NOT COHERENT: sixteen copies of one token. "
                + sixteen_copies_is_not_coherence
            )
        elif coherence and not coherence.get("coherent"):
            generation_finding = (
                f"NOT COHERENT: {coherence.get('reason')}. "
                "q3-MLP surviving the hidden-state loop did not predict "
                "generation on this mix."
            )
        else:
            generation_finding = (
                "COHERENT: varied tokens, not a single-token copy and not "
                "whitespace-only. The q3-MLP / q4-attn mix is both material "
                "and coherent on this 16-token sample."
            )
    elif decoded:
        generation_finding = (
            f"DECODE FAILED exit={decoded.get('exit_code')}. "
            "Report the failure; do not search for a prettier mix."
        )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Build the artifact whose MLP tensors use q3 group 64 and whose "
            "attention stays q4, then decode it on the native runtime for at "
            "least 16 tokens."
        ),
        "q4_incumbent": Q4_INCUMBENT,
        "parent_bf16": str(PARENT_BF16),
        "parent_params": PARENT_PARAMS,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "native_q3_kernel_exists": True,
        "native_q3_kernel": NATIVE_Q3_KERNEL,
        "compile": compiled,
        "decode": decoded,
        "generation_finding": generation_finding,
        "sixteen_copies_is_not_coherence": sixteen_copies_is_not_coherence,
        "elapsed_s": elapsed_s,
        "written_to": str(out_receipt),
    }
    if decoded and decoded.get("ok"):
        receipt["chosen"] = {
            "mix_id": compiled["mix_id"],
            "recipe": compiled["recipe"],
            "artifact_root": compiled["artifact_root"],
            "exact_mix": {
                "tensors": compiled["recipe"]["tensors"],
                "codec": compiled["recipe"]["codec"],
                "bits": BITS_Q3,
                "group": GROUP_Q3,
                "layers": compiled["recipe"]["layers"],
                "attention": compiled["recipe"]["attention"],
                "q3_mlp_tensors": compiled["q3_tensors"],
                "n_q3_mlp_tensors": compiled["n_q3_mlp"],
            },
            "storage_bpw": compiled["storage_bpw"],
            "active_bpw": compiled["active_bpw"],
            "complete_ebpw": compiled["complete_ebpw"],
            "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
            "q3_tensor_storage_bpw": compiled["q3_storage_bpw"],
            "beside_q4": compiled["beside_q4"],
            "prompt": decoded.get("prompt"),
            "prompt_ids": decoded.get("prompt_ids"),
            "generated_text_verbatim": decoded.get("generated_text_verbatim"),
            "new_token_ids": decoded.get("new_token_ids"),
            "n_new_tokens": decoded.get("n_new_tokens"),
            "tok_s": decoded.get("tok_s"),
            "median_gpu_ns_per_token": decoded.get("median_gpu_ns_per_token"),
            "native_kernel_ran": decoded.get("native_kernel_ran"),
            "dequant_path": decoded.get("dequant_path"),
            "decode_kernel": decoded.get("decode_kernel"),
            "fallbacks": decoded.get("fallbacks"),
            "dense_w_materialized": decoded.get("dense_w_materialized"),
            "expanded_to_q4": decoded.get("expanded_to_q4"),
            "expanded_to_float_gemv": decoded.get("expanded_to_float_gemv"),
            "census": decoded.get("census"),
            "census_line": decoded.get("census_line"),
            "bind": decoded.get("bind"),
            "coherence": decoded.get("coherence"),
            "matches_q4_incumbent_first_16_ids": list(
                decoded.get("new_token_ids") or []
            )[:16]
            == Q4_INCUMBENT_FIRST_16_IDS,
            "native_kernel_evidence": {
                "kernel": NATIVE_Q3_KERNEL,
                "how_bound": (
                    "dispatch_uniform -> qwen38_hgravu01_geo_tpr64_launch("
                    "bits=3, group_size=64) when HAWKING_QWEN38_RECON_FUSE "
                    "is default-on. MLP cols are 5120 or 17408, both "
                    "multiples of 64."
                ),
                "census_uniform": (decoded.get("census") or {}).get("uniform"),
                "expanded_to_q4": decoded.get("expanded_to_q4"),
                "expanded_to_float_gemv": decoded.get("expanded_to_float_gemv"),
                "recon_fuse_on": decoded.get("recon_fuse_on"),
                "kernel_name_printed_on_stderr": decoded.get(
                    "stderr_saw_q3_kernel"
                ),
                "note": (
                    "The production greedy binary does not print kernel "
                    "names unless HAWKING_TRACE_DISPATCH is set. The bind "
                    "is source-deterministic for Uniform bits=3 group=64."
                ),
            },
        }
    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_receipt.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp, out_receipt)
    print(f"wrote {out_receipt}", flush=True)
    return receipt


def run(
    *,
    decode: bool = True,
    out_receipt: Path = RECEIPT,
    out_root: Path | None = None,
    compiled: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if compiled is None:
        print(f"== compile {MIX_ID} ==", flush=True)
        compiled = compile_mix(out_root=out_root)
    decoded: dict[str, Any] | None = None
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
    return write_receipt(
        compiled,
        decoded,
        elapsed_s=time.perf_counter() - t0,
        out_receipt=out_receipt,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    decode = "--pack-only" not in args
    compiled = None
    if "--decode-only" in args:
        report_path = ARTIFACTS_ROOT / MIX_ID / "MIX_REPORT.json"
        if not report_path.is_file():
            raise PackError(f"missing {report_path}; run without --decode-only first")
        compiled = json.loads(report_path.read_text())
        decode = True
    run(decode=decode, compiled=compiled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
