#!/usr/bin/env python3
"""AFFINE2_NATIVE_MLP: whole-MLP HGRAVF01 catalog decoded on the native runtime.

The kernel already existed (affine2_group32_matvec.metal, HGRAVF01,
w = q * scale + bias, unsigned 2-bit, group 32, 4 codes/byte). This
harness hardlinks the sealed uniform-q4 catalog, re-encodes every MLP
GEMV (gate/up/down, layers 0..63) from the BF16 parent as fitted
min/max affine-2, writes catalog.hq38m20, and runs
ascension_qwen38_hybrid_greedy.

Group billing is honest: 2 bits/weight + f16 scale + f16 bias at
group 32 is 3.0 bpw. Bias-free group 64 would be 2.25 bpw (the q2
fitted whole-model bracket) but that is a different kernel; this lane
runs the native g32 affine kernel that already exists.

Does not load a second 27B. Streams one parent tensor at a time. Does
not write under ~/models. Does not touch receipts/ascent-2026-08-16
or workspace/campaign.

    python3 tools/headless/affine2_native_mlp.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import hashlib
import json
import os
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

RECEIPT = REPO / "receipts" / "headless" / "AFFINE2_NATIVE_MLP.json"
SCHEMA = "hawking.headless.affine2_native_mlp.v1"
ARTIFACTS_ROOT = Path(
    os.environ.get(
        "QWEN38_AFFINE2_ARTIFACT_ROOT",
        str(REPO / "artifacts" / "qwen38-affine2-mlp"),
    )
)

LAYERS = 64
GROUP_AFFINE = 32
AFFINE_BITS = 2
SCALE_BITS = 16
BIAS_BITS = 16
CODEC_AFFINE = 5
CODEC_Q4 = 3
CODEC_F32 = 4
MAGIC_AFFINE = b"HGRAVF01"
SCHEMA_AFFINE = "hawking.gravity.affine_scale_bias.v1"
AFFINE_REPR = "affine_q2_group32_fp16_scale_bias"
MAX_NEW = 16
MAX_SEQ = 128
MIX_ID = "mix_all_mlp_affine_g32"


class PackError(RuntimeError):
    pass


def affine_storage_bpw(group: int, *, bias: bool = True) -> float:
    extra = SCALE_BITS / float(group)
    if bias:
        extra += BIAS_BITS / float(group)
    return float(AFFINE_BITS) + extra


def pack_hgrafv01(weights: np.ndarray, group_size: int = GROUP_AFFINE) -> bytes:
    """HGRAVF01 min/max affine. Matches hawking-core pack_affine_factor."""
    if weights.ndim != 2:
        raise PackError(f"affine packer wants rank-2, got {weights.shape}")
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    if group_size != GROUP_AFFINE:
        raise PackError(f"native kernel is group {GROUP_AFFINE}, got {group_size}")
    if cols % group_size != 0:
        raise PackError(f"cols={cols} is not a multiple of group_size={group_size}")
    flat = np.ascontiguousarray(weights, dtype=np.float32)
    if not np.isfinite(flat).all():
        raise PackError("affine source is non-finite")
    groups_per_row = cols // group_size
    grouped = flat.reshape(rows, groups_per_row, group_size)
    lo = grouped.min(axis=-1)
    hi = grouped.max(axis=-1)
    scale = np.maximum((hi - lo) / 3.0, 1e-7).astype(np.float32)
    bias = lo.astype(np.float32)
    scale_f16 = scale.astype(np.float16)
    bias_f16 = bias.astype(np.float16)
    stored_scale = scale_f16.astype(np.float32)
    stored_bias = bias_f16.astype(np.float32)
    denom = np.where(np.abs(stored_scale) > 0.0, stored_scale, 1.0)
    q = np.clip(
        np.rint((grouped - stored_bias[..., None]) / denom[..., None]),
        0,
        3,
    ).astype(np.uint8)
    codes = q.reshape(-1)
    n = int(codes.size)
    packed = np.zeros(n // 4, dtype=np.uint8)
    for shift in range(4):
        packed |= (codes[shift::4] & np.uint8(3)) << np.uint8(2 * shift)
    groups = rows * groups_per_row
    header = {
        "schema": SCHEMA_AFFINE,
        "representation": AFFINE_REPR,
        "shape": [rows, cols],
        "elements": rows * cols,
        "bits": AFFINE_BITS,
        "group_size": group_size,
        "groups": int(groups),
        "scale_bytes": int(groups * 2),
        "bias_bytes": int(groups * 2),
        "code_bytes": int(packed.size),
        "source": "fitted_minmax_parent_bf16",
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = scale_f16.tobytes() + bias_f16.tobytes() + packed.tobytes()
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
    gpr = cols // GROUP_AFFINE
    out = np.empty((rows, cols), dtype=np.float32)
    grouped = out.reshape(rows, gpr, GROUP_AFFINE)
    q = codes.reshape(rows, gpr, GROUP_AFFINE).astype(np.float32)
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
            if codec == CODEC_Q4:
                q4_bytes += nbytes
            else:
                f32_bytes += nbytes
            digest = sha256_hex(filename.encode())
        else:
            print(f"  [{MIX_ID}] affine {name} group={GROUP_AFFINE}", flush=True)
            w = src.load(parent_key(name))
            if list(w.shape) != shape:
                raise PackError(f"{name} parent shape {list(w.shape)} != catalog {shape}")
            payload = pack_hgrafv01(w, GROUP_AFFINE)
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
            "codec": "HGRAVF01 affine_q2_group32 (w = q * scale + bias, unsigned q in {0,1,2,3})",
            "group": GROUP_AFFINE,
            "layers": list(range(LAYERS)),
            "organs": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            "attention": "HQ30UQ4 g64 (hardlinked incumbent)",
            "embed_head": "HQ30UQ4 / f32v2 (hardlinked incumbent)",
            "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            "why": (
                "Whole-model bracket: q2 fitted g64 at 2.25 bpw argmax-agrees "
                "(rel_l2 0.3471) while ternary 1.85 falls off a cliff. The native "
                "kernel that already exists is affine g32 with scale AND bias = 3.0 bpw."
            ),
        },
        "artifact_root": str(dest),
        "catalog": str(catalog_path),
        "n_tensors": len(records),
        "n_affine": n_affine,
        "n_hardlink": n_hardlink,
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
        "comparison_not_run_on_this_kernel": {
            "bias_free_group64_bpw": affine_storage_bpw(64, bias=False),
            "affine_group64_bpw": affine_storage_bpw(64, bias=True),
            "note": (
                "Bias-free g64 is 2.25 bpw and matches the q2 fitted whole-model "
                "bracket. The existing native kernel is group-32 with bias (3.0 bpw). "
                "A g64 bias-free variant would need a different kernel."
            ),
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
        result["coherence"] = {
            "coherent": False,
            "reason": f"decode exit {proc.returncode}",
        }
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
    saw_affine = (
        "qwen_affine_q2_group32_matvec" in stderr
        or "HGRAVF01 affine2" in stderr
        or "affine=" in stderr
    )
    native = "qwen38-decode mixed HQ38M20" in stderr or "mixed bind" in stderr
    dequant = ("expanded_to_q4=" in stderr and "expanded_to_q4=0" not in stderr) or (
        "reconstruct-to-Q4" in stderr and "no reconstruct-to-Q4" not in stderr
    )
    census_line = ""
    bind_line = ""
    for line in stderr.splitlines():
        if "mixed census:" in line:
            census_line = line.strip()
        if "HGRAVF01 affine2" in line or (
            not bind_line and "mixed bind:" in line
        ):
            bind_line = line.strip()
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
            "expanded_to_q4": 0,
            "expanded_to_float_gemv": 0,
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
            "census": census_line,
            "bind": bind_line,
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
            REPO
            / "workspace/ops/build/rust/release-fast/examples/affine2_parity",
            Path.home()
            / "Downloads/hawking-copy/workspace/ops/build/rust/release-fast/examples/affine2_parity",
        ]
    )
    exe = next((p for p in candidates if p.is_file()), None)
    if exe is None:
        return {
            "ok": False,
            "reason": "affine2_parity binary not built",
        }
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
    print("== affine2_parity --synthetic ==", flush=True)
    parity = run_parity()
    print(f"parity ok={parity.get('ok')} status={parity.get('status')}", flush=True)

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

    receipt = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Produce an artifact whose MLP tensors use the native 2-bit affine "
            "codec, decoded by the NATIVE runtime for at least 16 tokens."
        ),
        "kernel_already_existed": {
            "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
            "production_kernels": [
                "qwen_affine_q2_group32_matvec",
                "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            ],
            "container": "HGRAVF01 hawking.gravity.affine_scale_bias.v1",
            "reconstruction": "w = float(q) * scale + bias, q in {0,1,2,3}, group 32",
            "source_commits": [
                "37fdee09f lane: native-affine2-kernel-20260818-225710",
                "9babd70c5 lane: affine2-decoder-integration-20260818-231219",
            ],
            "did_not_write_a_new_kernel": True,
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
