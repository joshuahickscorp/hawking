#!/usr/bin/env python3
"""Q2F_G64_GENERATION: ask the 2.25-bpw 4-level LS-fitted arm to generate.

COMPOSITION_LADDER classified this arm UNTESTED_ABOVE coherent_generation:
it survived the whole-model token loop (argmax 9714=9714) and nobody asked
it to decode. This harness packs every MLP GEMV from the BF16 parent with
the 4-level LS-fitted 2-bit codec at group 64, attention left at q4, and
runs 16 greedy tokens on the native runtime.

`_fourlevel_fitted` is the composition reference. Its rint-to-half
assignment can land on {0, ±1} as well as {±0.5, ±1.5}; a 2-bit kernel
only has four codes. The packer takes that function's LS-fitted delta,
assigns onto the odd grid {-1.5,-0.5,+0.5,+1.5}, and iterates assign+LS
until the codes stabilize. Reconstruction is w = (q-1.5)*delta.

Two kernels are measured on the SAME artifact (no second 27B):
  bias-free  qwen_q2f_group64_matvec_geo_tpr64_tg128     (production)
  reuse      affine2 geo with derived bias = -1.5*delta  (HAWKING_Q2F_REUSE_AFFINE2=1)

Does not load a second 27B. Does not write under ~/models. Does not touch
receipts/ascent-2026-08-16 or workspace/campaign.

    python3 tools/headless/q2f_g64_generation.py
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
from fractional_bit_canon import _fourlevel_fitted, as_groups, snap_f16  # noqa: E402

RECEIPT = REPO / "receipts" / "headless" / "Q2F_G64_GENERATION.json"
SCHEMA = "hawking.headless.q2f_g64_generation.v1"
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "QWEN38_Q2F_ARTIFACT_ROOT",
        str(REPO / "artifacts" / "qwen38-q2f-g64"),
    )
)

LAYERS = 64
GROUP_Q2F = 64
Q2F_BITS = 2
SCALE_BITS = 16
CODEC_AFFINE = 5
CODEC_Q4 = 3
CODEC_F32 = 4
MAGIC_AFFINE = b"HGRAVF01"
SCHEMA_AFFINE = "hawking.gravity.affine_scale_bias.v1"
Q2F_REPR = "fourlevel_q2_group64_fp16_delta"
AFFINE_REPR = "affine_q2_group64_fp16_scale_bias"
MAX_NEW = 16
MAX_SEQ = 128
MIX_ID = "mix_all_mlp_q2f_g64"
MAX_LS_ITERS = 16
LEADER_EBPW = 3.139300850311054
LEADER_ID = "NOETIC_PARENT_A (affine2_g64_LS + fused graph)"
NATIVE_KERNEL_Q2F_GEO = "qwen_q2f_group64_matvec_geo_tpr64_tg128"
NATIVE_KERNEL_Q2F_SERIAL = "qwen_q2f_group64_matvec"
NATIVE_KERNEL_AFFINE_GEO = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"

CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)

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


def q2f_storage_bpw(group: int = GROUP_Q2F) -> float:
    return float(Q2F_BITS) + SCALE_BITS / float(group)


def affine_storage_bpw(group: int = GROUP_Q2F) -> float:
    return float(Q2F_BITS) + SCALE_BITS / float(group) + SCALE_BITS / float(group)


def pack_codes_lsb2(codes: np.ndarray) -> bytes:
    codes = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    n = int(codes.size)
    if n % 4 != 0:
        raise PackError(f"2-bit packer wants a multiple of 4 codes, got {n}")
    packed = np.zeros(n // 4, dtype=np.uint8)
    for shift in range(4):
        packed |= (codes[shift::4] & np.uint8(3)) << np.uint8(2 * shift)
    return packed.tobytes()


def fit_q2f(
    weights: np.ndarray,
    group_size: int = GROUP_Q2F,
    *,
    max_iters: int = MAX_LS_ITERS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """4-level codes + f16 delta, initialized from `_fourlevel_fitted`.

    Returns (q uint8 [rows,gpr,g], delta f32 [rows,gpr], probe).
    """
    if weights.ndim != 2:
        raise PackError(f"q2f packer wants rank-2, got {weights.shape}")
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    if cols % group_size != 0:
        raise PackError(f"cols={cols} is not a multiple of group_size={group_size}")
    flat = np.ascontiguousarray(weights, dtype=np.float32)
    if not np.isfinite(flat).all():
        raise PackError("q2f source is non-finite")
    gpr = cols // group_size
    G = flat.reshape(rows, gpr, group_size)
    # Reference reconstruction the composition arm actually used.
    what7 = _fourlevel_fitted(flat.reshape(rows, cols), group_size)
    # Same init as _fourlevel_fitted: amax/1.5, rint-to-half, LS delta.
    amax = np.max(np.abs(G), axis=-1, keepdims=True)
    delta = np.where(amax > 0, amax / np.float32(1.5), np.float32(1.0))
    unit7 = np.clip(np.rint(G / delta * np.float32(2.0)) / np.float32(2.0), -1.5, 1.5)
    num = (G * unit7).sum(axis=-1, keepdims=True)
    den = (unit7 * unit7).sum(axis=-1, keepdims=True)
    delta = np.where(den > 0, num / np.maximum(den, np.float32(1e-30)), delta)
    delta = snap_f16(delta)
    q = np.zeros(G.shape, dtype=np.uint8)
    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        denom = np.where(np.abs(delta) > 0, delta, np.float32(1.0))
        q_new = np.clip(np.rint(G / denom + np.float32(1.5)), 0, 3).astype(np.uint8)
        q_new = np.where(np.abs(delta) > 0, q_new, np.uint8(0))
        unit = q_new.astype(np.float32) - np.float32(1.5)
        num = (G * unit).sum(axis=-1, keepdims=True)
        den = (unit * unit).sum(axis=-1, keepdims=True)
        delta_ls = np.where(den > 0, num / np.maximum(den, np.float32(1e-30)), delta)
        delta_new = snap_f16(delta_ls)
        if np.array_equal(q_new, q) and np.allclose(delta_new, delta):
            q = q_new
            delta = delta_new
            break
        q = q_new
        delta = delta_new
    recon = (q.astype(np.float32) - np.float32(1.5)) * delta
    recon_flat = recon.reshape(rows, cols)
    w_norm = float(np.linalg.norm(flat.reshape(rows, cols)))
    probe = {
        "fourlevel_fitted_rel_l2": float(
            np.linalg.norm(what7 - flat.reshape(rows, cols)) / max(w_norm, 1e-12)
        ),
        "q2f_4level_rel_l2": float(
            np.linalg.norm(recon_flat - flat.reshape(rows, cols)) / max(w_norm, 1e-12)
        ),
        "q2f_vs_fourlevel_fitted_rel_l2": float(
            np.linalg.norm(recon_flat - what7) / max(w_norm, 1e-12)
        ),
        "ls_iters": n_iters,
        "n_codes_not_on_odd_grid_in_fourlevel_fitted": int(
            np.sum(
                ~np.isin(
                    np.round(unit7, 5),
                    np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32),
                )
            )
        ),
        "note": (
            "_fourlevel_fitted is the composition reference; its rint-to-half "
            "assignment is not 4-level. Native 2-bit is the iterated odd grid."
        ),
    }
    return q, delta.reshape(rows, gpr), probe


def pack_hgrafv01_q2f(weights: np.ndarray, group_size: int = GROUP_Q2F) -> tuple[bytes, dict[str, Any]]:
    q, delta, probe = fit_q2f(weights, group_size)
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    gpr = cols // group_size
    groups = rows * gpr
    delta_f16 = delta.astype(np.float16)
    packed = pack_codes_lsb2(q)
    header = {
        "schema": SCHEMA_AFFINE,
        "representation": Q2F_REPR,
        "shape": [rows, cols],
        "elements": rows * cols,
        "bits": Q2F_BITS,
        "group_size": group_size,
        "groups": int(groups),
        "scale_bytes": int(groups * 2),
        "bias_bytes": 0,
        "code_bytes": int(len(packed)),
        "source": "fourlevel_ls_fitted_parent_bf16",
        "fit": "assign_odd_grid_ls_delta_iterate",
        "reconstruction": "w = (q - 1.5) * delta, q in {0,1,2,3}",
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = delta_f16.tobytes() + packed
    if len(body) != header["scale_bytes"] + header["code_bytes"]:
        raise PackError("HGRAVF01 q2f body ledger drifted")
    payload = MAGIC_AFFINE + struct.pack("<I", len(header_bytes)) + header_bytes + body
    return payload, probe


def parse_hgrafv01_q2f(payload: bytes) -> dict[str, Any]:
    if payload[:8] != MAGIC_AFFINE:
        raise PackError(f"magic {payload[:8]!r} is not HGRAVF01")
    header_len = struct.unpack_from("<I", payload, 8)[0]
    header = json.loads(payload[12 : 12 + header_len])
    body = payload[12 + header_len :]
    expected = int(header["scale_bytes"]) + int(header.get("bias_bytes") or 0) + int(
        header["code_bytes"]
    )
    if len(body) != expected:
        raise PackError("HGRAVF01 body length disagrees with ledger")
    return header


def reconstruct_hgrafv01_q2f(payload: bytes) -> np.ndarray:
    header = parse_hgrafv01_q2f(payload)
    header_len = struct.unpack_from("<I", payload, 8)[0]
    body = payload[12 + header_len :]
    rows, cols = int(header["shape"][0]), int(header["shape"][1])
    group = int(header["group_size"])
    scale_bytes = int(header["scale_bytes"])
    bias_bytes = int(header.get("bias_bytes") or 0)
    deltas = np.frombuffer(body[:scale_bytes], dtype=np.float16).astype(np.float32)
    packed = np.frombuffer(body[scale_bytes + bias_bytes :], dtype=np.uint8)
    n = rows * cols
    codes = np.empty(n, dtype=np.uint8)
    for shift in range(4):
        codes[shift::4] = (packed >> np.uint8(2 * shift)) & np.uint8(3)
    gpr = cols // group
    q = codes.reshape(rows, gpr, group).astype(np.float32)
    out = np.empty((rows, cols), dtype=np.float32)
    grouped = out.reshape(rows, gpr, group)
    if bias_bytes == 0:
        grouped[...] = (q - np.float32(1.5)) * deltas.reshape(rows, gpr)[..., None]
    else:
        biases = np.frombuffer(body[scale_bytes : scale_bytes + bias_bytes], dtype=np.float16).astype(
            np.float32
        )
        grouped[...] = q * deltas.reshape(rows, gpr)[..., None] + biases.reshape(rows, gpr)[
            ..., None
        ]
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
    q2f_names: list[str] = []
    t0 = time.perf_counter()
    payload_bytes = 0
    q2f_bytes = 0
    q4_bytes = 0
    f32_bytes = 0
    n_hardlink = 0
    n_q2f = 0
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
            print(f"  [{MIX_ID}] q2f {name} group={GROUP_Q2F}", flush=True)
            w = src.load(parent_key(name))
            if list(w.shape) != shape:
                raise PackError(f"{name} parent shape {list(w.shape)} != catalog {shape}")
            payload, this_probe = pack_hgrafv01_q2f(w, GROUP_Q2F)
            if probe is None:
                probe = {"tensor": name, "shape": shape, **this_probe}
            del w
            filename = hashlib.sha256(name.encode("utf-8")).hexdigest() + ".hgrafv01"
            dest_path = segments_dir / filename
            write_atomic(dest_path, payload)
            nbytes = len(payload)
            digest = sha256_hex(payload)
            codec = CODEC_AFFINE
            codec_bpw = q2f_storage_bpw(GROUP_Q2F)
            q2f_names.append(name)
            q2f_bytes += nbytes
            n_q2f += 1
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
    codecs = Counter(int(r["codec"]) for r in records)
    mlp_elements = sum(int(r["elements"]) for r in records if is_mlp_proj(r["name"]))
    report = {
        "mix_id": MIX_ID,
        "recipe": {
            "id": MIX_ID,
            "tensors": "mlp.gate_proj, mlp.up_proj, mlp.down_proj on layers 0..63",
            "codec": (
                "HGRAVF01 fourlevel_q2_group64_fp16_delta "
                "(w = (q-1.5)*delta, unsigned q in {0,1,2,3}, no bias)"
            ),
            "group": GROUP_Q2F,
            "fit": "assign_odd_grid_ls_delta_iterate (init from _fourlevel_fitted)",
            "layers": list(range(LAYERS)),
            "organs": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            "attention": "HQ30UQ4 g64 (hardlinked incumbent)",
            "embed_head": "HQ30UQ4 / f32v2 (hardlinked incumbent)",
            "kernel": NATIVE_KERNEL_Q2F_GEO,
            "kernel_reuse_affine2": NATIVE_KERNEL_AFFINE_GEO,
            "why": (
                "cheapest arm known to survive the whole-model token loop; "
                "body 2.25 bpw vs leader affine2 g64 at 2.50 (the 0.25 is the "
                "bias term this codec does not carry)"
            ),
        },
        "artifact_root": str(dest),
        "catalog": str(catalog_path),
        "n_tensors": len(records),
        "n_q2f": n_q2f,
        "n_affine": n_q2f,
        "n_hardlink": n_hardlink,
        "n_attn_q4_hardlinked": n_attn_q4,
        "codecs": {str(k): int(v) for k, v in sorted(codecs.items())},
        "q2f_tensors": q2f_names,
        "q2f_group": GROUP_Q2F,
        "q2f_tensor_storage_bpw": q2f_storage_bpw(GROUP_Q2F),
        "affine_tensor_storage_bpw": q2f_storage_bpw(GROUP_Q2F),
        "q2f_bpw_billing": {
            "codes_bpw": 2.0,
            "scale_bpw": SCALE_BITS / float(GROUP_Q2F),
            "bias_bpw": 0.0,
            "total_bpw": q2f_storage_bpw(GROUP_Q2F),
            "group": GROUP_Q2F,
            "scale_dtype": "fp16",
            "bias_dtype": None,
        },
        "reference_quantizer": "tools/headless/fractional_bit_canon.py::_fourlevel_fitted",
        "fit_probe": probe,
        "mlp_elements": mlp_elements,
        "payload_bytes": payload_bytes,
        "q2f_bytes": q2f_bytes,
        "q4_bytes": q4_bytes,
        "f32_bytes": f32_bytes,
        "parent_params": PARENT_PARAMS,
        "storage_bpw": complete_ebpw,
        "active_bpw": complete_ebpw,
        "active_fused_bpw": complete_ebpw,
        "complete_ebpw": complete_ebpw,
        "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
        "leader_complete_ebpw": LEADER_EBPW,
        "cheaper_than_leader": complete_ebpw < LEADER_EBPW,
        "wall_s": time.perf_counter() - t0,
        "did_not_load_second_27b": True,
        "parent_streamed_one_tensor_at_a_time": True,
        "wrote_under_models": False,
    }
    write_atomic(dest / "MIX_REPORT.json", json.dumps(report, indent=2).encode())
    print(
        f"[{MIX_ID}] tensors={len(records)} q2f={n_q2f} "
        f"ebpw={complete_ebpw:.6f} (q4 {Q4_INCUMBENT_EBPW:.6f}, "
        f"leader {LEADER_EBPW:.6f}) mlp_bpw={q2f_storage_bpw():.4f} "
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
    q2f = None
    affine = None
    generic = None
    for line in stderr.splitlines():
        if "HGRAVF01 q2f" in line:
            q2f = line.strip()
        if "HGRAVF01 affine2" in line:
            affine = line.strip()
        if generic is None and "qwen38-decode mixed bind:" in line:
            generic = line.strip()
    return q2f or affine or generic


def cargo_build(example: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    cmd = [
        "cargo",
        "build",
        "--profile",
        "release-fast",
        "-p",
        "hawking-core",
        "--example",
        example,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-8000:],
        "ok": proc.returncode == 0,
    }


def decode_mix(
    artifact_root: Path,
    *,
    binary: Path | None = None,
    prompt: str = PROMPT,
    max_new: int = MAX_NEW,
    max_seq: int = MAX_SEQ,
    tokenizer: Path = TOKENIZER,
    reuse_affine2: bool = False,
    fuse_mlp: str | None = None,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    exe = binary or find_decode_binary()
    tag = "q2f"
    if reuse_affine2:
        tag = "reuse_affine2"
    if fuse_mlp:
        tag += f"_fuse_{fuse_mlp}"
    if ignore_eos:
        tag += "_ignore_eos"
    out_json = artifact_root / f"decode_{tag}.json"
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd: list[str] = []
    if lock.is_file():
        cmd.extend(["bash", str(lock), f"qwen38-q2f-{tag}"])
    cmd.extend(
        [
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
    )
    env = os.environ.copy()
    if reuse_affine2:
        env["HAWKING_Q2F_REUSE_AFFINE2"] = "1"
    else:
        env.pop("HAWKING_Q2F_REUSE_AFFINE2", None)
    if fuse_mlp:
        env["HAWKING_QWEN38_FUSE_MLP"] = fuse_mlp
    else:
        env.pop("HAWKING_QWEN38_FUSE_MLP", None)
    if ignore_eos:
        env["HAWKING_QWEN38_IGNORE_EOS"] = "1"
    else:
        env.pop("HAWKING_QWEN38_IGNORE_EOS", None)
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
        "command": [str(x) for x in cmd],
        "binary": str(exe),
        "reuse_affine2": reuse_affine2,
        "fuse_mlp": fuse_mlp,
        "ignore_eos": ignore_eos,
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
    dispatches = body.get("dispatches_per_step") or []
    last_disp = int(dispatches[-1]) if dispatches else None
    saw_q2f = NATIVE_KERNEL_Q2F_GEO in stderr or NATIVE_KERNEL_Q2F_SERIAL in stderr or (
        bind is not None and "q2f" in bind
    )
    saw_affine = NATIVE_KERNEL_AFFINE_GEO in stderr or (
        bind is not None and "affine2" in bind
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
            "dispatches_per_step": dispatches,
            "dispatches_per_token": last_disp,
            "native_kernel_ran": bool((native or saw_q2f or saw_affine) and not dequant),
            "dequant_path": bool(dequant),
            "stderr_saw_mixed_catalog": "HQ38M20" in stderr,
            "stderr_saw_q2f_kernel": saw_q2f,
            "stderr_saw_affine_kernel": saw_affine,
            "coherence": judge_coherence(text or "", ids),
            "census": census,
            "bind": bind,
        }
    )
    return result


def run_parity() -> dict[str, Any]:
    candidates = [
        CARGO_TARGET / "release-fast" / "examples" / "q2f_parity",
        CARGO_TARGET / "release" / "examples" / "q2f_parity",
        Path.home()
        / "Downloads/hawking-copy/workspace/ops/build/rust/release-fast/examples/q2f_parity",
    ]
    exe = next((p for p in candidates if p.is_file()), None)
    if exe is None:
        built = cargo_build("q2f_parity")
        exe = next((p for p in candidates if p.is_file()), None)
        if exe is None:
            return {"ok": False, "reason": "q2f_parity binary not built", "build": built}
    proc = subprocess.run(
        [str(exe), "--synthetic"],
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


def run_kernel_competence() -> dict[str, Any]:
    script = HERE / "kernel_competence.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    receipt = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"
    body = json.loads(receipt.read_text()) if receipt.is_file() else {}
    wanted = (
        "qwen_q2f_group64_matvec_geo_tpr64_tg128",
        "qwen_q2f_group64_matvec",
        "q2f_group64_matvec_geo_tpr64_tg128",
        "q2f_group64_matvec",
    )
    verdicts = {}
    for f in body.get("per_file") or []:
        for k in f.get("kernels") or []:
            if k.get("kernel") in wanted:
                verdicts[k["kernel"]] = {
                    "file": f.get("file"),
                    "verdict": k.get("verdict"),
                    "n_findings": k.get("n_findings"),
                }
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "q2f_verdicts": verdicts,
        "any_q2f_defective": any(
            "dequant" not in name and v.get("verdict") == "DEFECTIVE"
            for name, v in verdicts.items()
        ),
    }


def summarize_decode(decoded: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decoded or not decoded.get("ok"):
        return decoded
    return {
        "ok": True,
        "reuse_affine2": decoded.get("reuse_affine2"),
        "fuse_mlp": decoded.get("fuse_mlp"),
        "generated_text_verbatim": decoded.get("generated_text_verbatim"),
        "new_token_ids": decoded.get("new_token_ids"),
        "n_new_tokens": decoded.get("n_new_tokens"),
        "tok_s": decoded.get("tok_s"),
        "dispatches_per_token": decoded.get("dispatches_per_token"),
        "dispatches_per_step": decoded.get("dispatches_per_step"),
        "native_kernel_ran": decoded.get("native_kernel_ran"),
        "dequant_path": decoded.get("dequant_path"),
        "fallbacks": decoded.get("fallbacks"),
        "dense_w_materialized": decoded.get("dense_w_materialized"),
        "expanded_to_q4": decoded.get("expanded_to_q4", 0),
        "expanded_to_float_gemv": decoded.get("expanded_to_float_gemv", 0),
        "coherence": decoded.get("coherence"),
        "census": decoded.get("census"),
        "bind": decoded.get("bind"),
        "prompt": decoded.get("prompt"),
        "median_gpu_ns_per_token": decoded.get("median_gpu_ns_per_token"),
        "decode_wall_ns": decoded.get("decode_wall_ns"),
        "decode_steps": decoded.get("decode_steps"),
    }


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
            "note": "reference",
        },
        {
            "codec": "NOETIC_PARENT_A affine2 g64 LS (leader)",
            "body_bpw": 2.50,
            "complete_ebpw": LEADER_EBPW,
            "note": "has bias term; fused graph 34.873 tok/s",
        },
        {
            "codec": "q2_4level_fitted_g64 (this arm)",
            "body_bpw": 2.25,
            "complete_ebpw": compiled.get("complete_ebpw"),
            "text_verbatim": ours_text,
            "note": "no bias; 2 bits + f16 delta / 64",
        },
    ]


def run_all(*, decode: bool = True, out_receipt: Path = RECEIPT) -> dict[str, Any]:
    t0 = time.perf_counter()
    print("== cargo build hybrid_greedy + q2f_parity ==", flush=True)
    build_greedy = cargo_build("ascension_qwen38_hybrid_greedy")
    build_parity = cargo_build("q2f_parity")
    print(
        f"build greedy ok={build_greedy.get('ok')} {build_greedy.get('wall_s'):.1f}s  "
        f"parity ok={build_parity.get('ok')} {build_parity.get('wall_s'):.1f}s",
        flush=True,
    )
    print(f"== compile {MIX_ID} ==", flush=True)
    compiled = compile_mix()
    decoded_q2f = None
    decoded_reuse = None
    decoded_fused = None
    configs_tried: list[dict[str, Any]] = []
    if decode:
        binary = find_decode_binary()
        print(f"== decode bias-free q2f kernel ({MIX_ID}) ==", flush=True)
        decoded_q2f = decode_mix(Path(compiled["artifact_root"]), binary=binary)
        configs_tried.append(
            {
                "id": "q2f_bias_free_unfused",
                "kernel": NATIVE_KERNEL_Q2F_GEO,
                "lost": not bool(decoded_q2f.get("ok")),
                "summary": summarize_decode(decoded_q2f),
            }
        )
        gen = (decoded_q2f or {}).get("generated_text_verbatim")
        coh = (decoded_q2f or {}).get("coherence") or {}
        print(
            f"[q2f] exit={decoded_q2f.get('exit_code')} "
            f"coherent={coh.get('coherent')} text={gen!r}",
            flush=True,
        )
        print("== decode reuse-affine2 (derived bias, same artifact) ==", flush=True)
        decoded_reuse = decode_mix(
            Path(compiled["artifact_root"]), binary=binary, reuse_affine2=True
        )
        configs_tried.append(
            {
                "id": "q2f_reuse_affine2_unfused",
                "kernel": NATIVE_KERNEL_AFFINE_GEO,
                "lost": not bool(decoded_reuse.get("ok")),
                "summary": summarize_decode(decoded_reuse),
            }
        )
        print(
            f"[reuse-affine2] exit={decoded_reuse.get('exit_code')} "
            f"coherent={(decoded_reuse.get('coherence') or {}).get('coherent')} "
            f"text={decoded_reuse.get('generated_text_verbatim')!r}",
            flush=True,
        )
        print("== decode q2f fused gate+up+swiglu ==", flush=True)
        decoded_fused = decode_mix(
            Path(compiled["artifact_root"]), binary=binary, fuse_mlp="swiglu"
        )
        configs_tried.append(
            {
                "id": "q2f_bias_free_fused_swiglu",
                "kernel": "qwen_q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
                "lost": not bool(decoded_fused.get("ok")),
                "summary": summarize_decode(decoded_fused),
            }
        )
        print(
            f"[fused] exit={decoded_fused.get('exit_code')} "
            f"text={decoded_fused.get('generated_text_verbatim')!r}",
            flush=True,
        )
        print("== decode q2f ignore-EOS 16-token sample ==", flush=True)
        decoded_16 = decode_mix(
            Path(compiled["artifact_root"]), binary=binary, ignore_eos=True
        )
        configs_tried.append(
            {
                "id": "q2f_bias_free_unfused_ignore_eos",
                "kernel": NATIVE_KERNEL_Q2F_GEO,
                "lost": not bool(decoded_16.get("ok")),
                "note": "HAWKING_QWEN38_IGNORE_EOS=1 so greedy is not truncated at im_end",
                "summary": summarize_decode(decoded_16),
            }
        )
        print(
            f"[ignore-eos] exit={decoded_16.get('exit_code')} "
            f"n={decoded_16.get('n_new_tokens')} "
            f"text={decoded_16.get('generated_text_verbatim')!r}",
            flush=True,
        )
        if decoded_16.get("ok") and (decoded_16.get("n_new_tokens") or 0) >= 16:
            decoded_q2f = decoded_16
    print("== q2f_parity --synthetic ==", flush=True)
    parity = run_parity()
    print(
        f"parity ok={parity.get('ok')} status={parity.get('status')} "
        f"max_abs_diff={parity.get('max_abs_diff')}",
        flush=True,
    )
    print("== kernel_competence ==", flush=True)
    competence = run_kernel_competence()
    print(f"competence q2f verdicts={competence.get('q2f_verdicts')}", flush=True)

    chosen = None
    primary = decoded_q2f
    if primary and primary.get("ok"):
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
                "q2f_tensors": compiled["q2f_tensors"],
            },
            "storage_bpw": compiled["storage_bpw"],
            "active_bpw": compiled["active_bpw"],
            "complete_ebpw": compiled["complete_ebpw"],
            "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
            "leader_complete_ebpw": LEADER_EBPW,
            "q2f_tensor_storage_bpw": compiled["q2f_tensor_storage_bpw"],
            "q2f_bpw_billing": compiled["q2f_bpw_billing"],
            "prompt": primary.get("prompt"),
            "generated_text_verbatim": primary.get("generated_text_verbatim"),
            "new_token_ids": primary.get("new_token_ids"),
            "n_new_tokens": primary.get("n_new_tokens"),
            "tok_s": primary.get("tok_s"),
            "dispatches_per_token": primary.get("dispatches_per_token"),
            "native_kernel_ran": primary.get("native_kernel_ran"),
            "dequant_path": primary.get("dequant_path"),
            "fallbacks": primary.get("fallbacks"),
            "dense_w_materialized": primary.get("dense_w_materialized"),
            "expanded_to_q4": primary.get("expanded_to_q4", 0),
            "expanded_to_float_gemv": primary.get("expanded_to_float_gemv", 0),
            "prompt_ids": primary.get("prompt_ids"),
            "coherence": primary.get("coherence"),
            "census": primary.get("census"),
            "bind": primary.get("bind"),
        }

    table = comparison_table(compiled, primary)
    gen_text = (primary or {}).get("generated_text_verbatim")
    coh = ((primary or {}).get("coherence") or {}) if primary else {}
    if primary and primary.get("ok"):
        if coh.get("repeated_single_token"):
            finding = (
                "Q2F g64 collapsed to a repeated token. "
                "Sixteen copies of one token is not coherence. "
                f"Verbatim: {gen_text!r}."
            )
        elif not coh.get("coherent"):
            finding = (
                "Q2F g64 generated 16 tokens but failed the coherence judge: "
                f"{coh.get('reason')}. Verbatim: {gen_text!r}."
            )
        else:
            finding = (
                "Q2F g64 generated 16 tokens that the coherence judge calls "
                f"varied. Verbatim: {gen_text!r}."
            )
    else:
        finding = "Q2F g64 native decode did not run to completion."

    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Does the cheapest arm that survived the whole-model token loop "
            "(q2 4-level fitted g64, 2.25 bpw body) generate 16 tokens natively?"
        ),
        "codec": {
            "name": "q2_4level_fitted_g64",
            "grid": [-1.5, -0.5, 0.5, 1.5],
            "group": GROUP_Q2F,
            "reconstruction": "w = (q - 1.5) * delta, q in {0,1,2,3}",
            "billing_bpw": q2f_storage_bpw(),
            "reference": "tools/headless/fractional_bit_canon.py::_fourlevel_fitted",
            "native_fit": (
                "init delta from _fourlevel_fitted, assign onto the odd 4-level "
                "grid, LS-refit delta, iterate until codes stabilize"
            ),
        },
        "kernel_family": {
            "shader_mixed": "crates/hawking-core/shaders/q80_mixed_decode.metal",
            "shader_standalone": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
            "production_kernels": [NATIVE_KERNEL_Q2F_SERIAL, NATIVE_KERNEL_Q2F_GEO],
            "reuse_affine2_kernel": NATIVE_KERNEL_AFFINE_GEO,
            "container": "HGRAVF01 hawking.gravity.affine_scale_bias.v1 bias_bytes=0",
            "reconstruction": "w = (float(q) - 1.5) * delta, group 64 compile-time",
            "specialized_on_group": True,
            "group_size_bind_time_divide": False,
        },
        "q4_incumbent": Q4_INCUMBENT,
        "leader": {
            "id": LEADER_ID,
            "complete_ebpw": LEADER_EBPW,
            "body_bpw": 2.50,
        },
        "parent_bf16": str(PARENT_BF16),
        "parent_params": PARENT_PARAMS,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "build": {"hybrid_greedy": build_greedy, "q2f_parity": build_parity},
        "compile": compiled,
        "decode": summarize_decode(primary),
        "configs_tried": configs_tried,
        "parity": parity,
        "kernel_competence": competence,
        "chosen": chosen,
        "comparison": table,
        "generation_finding": finding,
        "elapsed_s": time.perf_counter() - t0,
    }
    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_receipt.with_suffix(f".json.{os.getpid()}.tmp")
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
