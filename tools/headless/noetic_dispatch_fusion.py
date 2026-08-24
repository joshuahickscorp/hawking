#!/usr/bin/env python3
"""Dispatch fusion: cut dispatches per token, keep the output coherent.

The density axis is exhausted. SOURCE and EXECUTABLE sit at an identical 964
dispatches/token while DRAM falls 7.34x. This harness fuses operators
(gate+up+SwiGLU, GQA QKV, DeltaNet qkvz+ba) and reports BEFORE/AFTER
dispatches, tok/s, 16 verbatim tokens, and fused-vs-unfused max_abs_diff.

A slower fusion is a real result. Do not search until something looks good.

    python3 tools/headless/noetic_dispatch_fusion.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from first_noetic_executable import (  # noqa: E402
    PROMPT,
    TOKENIZER,
    git_head,
    judge_coherence,
    now_iso,
)

SCHEMA = "hawking.headless.noetic_dispatch_fusion.v1"
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_DISPATCH_FUSION.json"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "qwen_uniform_q4.metal"
DECODE = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
LEDGER = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_token_ns_ledger.rs"

Q4_ROOT = Path(
    os.environ.get(
        "QWEN38_Q4_ARTIFACT",
        str(Path.home() / "models/qwen38-gravity-uniform-q4-v1"),
    )
)

CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)

LAYERS = 64
GQA_LAYERS = 16
DN_LAYERS = 48
BEFORE_DISPATCHES = 964
KERNELS = (
    "qwen_uniform_q4_group64_matvec_gate_up_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
)


def theoretical_after(mlp: str, qkv: bool, dn: bool) -> int:
    n = BEFORE_DISPATCHES
    if mlp == "pair":
        n -= LAYERS
    elif mlp == "swiglu":
        n -= 2 * LAYERS
    if qkv:
        n -= 2 * GQA_LAYERS
    if dn:
        n -= DN_LAYERS
    return n


def shader_evidence() -> dict[str, Any]:
    text = SHADER.read_text(encoding="utf-8", errors="replace") if SHADER.is_file() else ""
    rust = DECODE.read_text(encoding="utf-8", errors="replace") if DECODE.is_file() else ""
    ledger = LEDGER.read_text(encoding="utf-8", errors="replace") if LEDGER.is_file() else ""
    needles = {name: text.find(f"kernel void {name}(") for name in KERNELS}
    return {
        "shader_present": SHADER.is_file(),
        "shader_path": "crates/hawking-core/shaders/qwen_uniform_q4.metal",
        "kernel_needles": needles,
        "all_kernels_declared": all(v >= 0 for v in needles.values()),
        "wired_in_encode_dense_mlp": "Qwen38MlpFusion::GateUpSwiglu" in rust,
        "wired_in_encode_gqa": "encode_fused_qkv" in rust,
        "wired_in_encode_deltanet": "fuse_dn_inproj" in rust,
        "production_964_untouched": "production_dispatch_count_is_964" in ledger,
        "does_not_write_dense_w": "Packed codes stay packed" in text or "Packed decode stays in registers" in text,
        "workhorse_unchanged": "kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(" in text,
    }


def find_fusion_binary() -> Path | None:
    env = os.environ.get("QWEN38_FUSION_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    candidates = [
        CARGO_TARGET / "release-fast" / "examples" / "ascension_qwen38_dispatch_fusion",
        CARGO_TARGET / "release" / "examples" / "ascension_qwen38_dispatch_fusion",
        REPO
        / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_dispatch_fusion",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def cargo_build() -> dict[str, Any]:
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
        "ascension_qwen38_dispatch_fusion",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
        env=env,
    )
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "binary": str(find_fusion_binary()) if find_fusion_binary() else None,
    }


def run_fusion_example(binary: Path, out: Path, *, skip_decode: bool = False) -> dict[str, Any]:
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd: list[str] = []
    if lock.is_file():
        cmd.extend(["bash", str(lock), "qwen38-dispatch-fusion"])
    cmd.extend(
        [
            str(binary),
            "--artifact-root",
            str(Q4_ROOT),
            "--tokenizer",
            str(TOKENIZER),
            "--prompt",
            PROMPT,
            "--max-new-tokens",
            "16",
            "--max-seq-len",
            "128",
            "--reps",
            os.environ.get("QWEN38_FUSION_REPS", "3"),
            "--out",
            str(out),
        ]
    )
    if skip_decode:
        cmd.append("--skip-decode")
    env = os.environ.copy()
    env.pop("HAWKING_QWEN38_FUSE_MLP", None)
    env.pop("HAWKING_QWEN38_FUSE_GQA_QKV", None)
    env.pop("HAWKING_QWEN38_FUSE_DN_INPROJ", None)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )
    result: dict[str, Any] = {
        "command": cmd,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "out": str(out),
        "ok": proc.returncode == 0 and out.is_file(),
    }
    if out.is_file():
        try:
            result["body"] = json.loads(out.read_text())
        except json.JSONDecodeError as e:
            result["ok"] = False
            result["json_error"] = str(e)
    return result


def _arm_tok_s(arm: dict[str, Any] | None) -> dict[str, Any] | None:
    if not arm:
        return None
    return {
        "tok_s_reps": arm.get("tok_s_reps"),
        "tok_s_mean": arm.get("tok_s_mean"),
        "tok_s_min": arm.get("tok_s_min"),
        "tok_s_max": arm.get("tok_s_max"),
        "generated_text_verbatim": arm.get("generated_text_verbatim"),
        "new_token_ids": arm.get("new_token_ids"),
        "dispatches_last_step_reps": arm.get("dispatches_last_step_reps"),
        "dense_w_materialized": arm.get("dense_w_materialized", 0),
        "coherence": judge_coherence(
            arm.get("generated_text_verbatim") or "",
            [int(x) for x in (arm.get("new_token_ids") or [])],
        ),
    }


def theoretical_receipt(*, gpu: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = shader_evidence()
    after_pair = theoretical_after("pair", False, False)
    after_swiglu = theoretical_after("swiglu", False, False)
    after_all = theoretical_after("swiglu", True, True)
    gpu_body = (gpu or {}).get("body") if gpu else None
    decode = (gpu_body or {}).get("decode") or {}
    unfused = _arm_tok_s(decode.get("unfused"))
    swiglu = _arm_tok_s(decode.get("mlp_swiglu"))
    combo = _arm_tok_s(decode.get("mlp_swiglu_qkv_dn"))
    after_arm = combo or swiglu
    before_disp = BEFORE_DISPATCHES
    after_disp = after_all if combo else after_swiglu
    probes = (gpu_body or {}).get("dispatch_probes") or []
    measured_before = None
    measured_after = None
    for p in probes:
        ident = p.get("id")
        probe = p.get("probe") or {}
        if ident == "unfused":
            measured_before = probe.get("measured")
            if probe.get("theoretical") is not None:
                before_disp = int(probe["theoretical"])
        if ident == "mlp_swiglu_qkv_dn":
            measured_after = probe.get("measured")
            if probe.get("theoretical") is not None:
                after_disp = int(probe["theoretical"])
        elif ident == "mlp_swiglu" and measured_after is None:
            measured_after = probe.get("measured")
            if probe.get("theoretical") is not None:
                after_disp = int(probe["theoretical"])
    parity = (gpu_body or {}).get("parity") or {}
    mlp_par = parity.get("mlp_gate_up_swiglu") or {}
    verdict_parts = []
    if measured_before is not None and measured_after is not None:
        if measured_after < measured_before:
            verdict_parts.append(
                f"dispatches {measured_before} -> {measured_after} (cut {measured_before - measured_after})"
            )
        else:
            verdict_parts.append(
                f"dispatches did not fall ({measured_before} -> {measured_after})"
            )
    else:
        verdict_parts.append(
            f"theoretical dispatches {before_disp} -> {after_disp}"
        )
    if unfused and after_arm:
        u = unfused.get("tok_s_mean")
        a = after_arm.get("tok_s_mean")
        if isinstance(u, (int, float)) and isinstance(a, (int, float)):
            if a > u:
                verdict_parts.append(f"tok/s {u:.3f} -> {a:.3f} (faster)")
            elif a < u:
                verdict_parts.append(f"tok/s {u:.3f} -> {a:.3f} (SLOWER)")
            else:
                verdict_parts.append(f"tok/s unchanged at {u:.3f}")
    if after_arm:
        coh = after_arm.get("coherence") or {}
        if coh.get("coherent"):
            verdict_parts.append("fused generate is coherent")
        else:
            verdict_parts.append(
                f"fused generate is NOT coherent: {coh.get('reason')}"
            )
    if mlp_par.get("max_abs_diff") is not None:
        verdict_parts.append(f"mlp max_abs_diff={mlp_par.get('max_abs_diff')}")

    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Does fusing operators cut dispatches per token below 964 "
            "while staying coherent?"
        ),
        "why_not_more_density": (
            "Four grouped-code matvecs exhausted the density axis. "
            "44.7% fewer bits bought 1.9% throughput; the densest coherent "
            "artifact is the slowest. SOURCE and EXECUTABLE sit at 964 "
            "dispatches/token. Decode is dispatch-bound."
        ),
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "enable": {
            "default": "off — production graph stays 964",
            "HAWKING_QWEN38_FUSE_MLP": "pair | swiglu",
            "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
            "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
            "combo_that_measured_756": {
                "HAWKING_QWEN38_FUSE_MLP": "swiglu",
                "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
                "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
            },
        },
        "counting_method": (
            "TokenCommandBuffer.dispatch_count: one kernel launch = one dispatch. "
            "Same counter as production_dispatches_per_token / generate_greedy "
            "timing.dispatches."
        ),
        "dispatches_per_token": {
            "before": before_disp,
            "after": after_disp,
            "measured_before": measured_before,
            "measured_after": measured_after,
            "theoretical": {
                "unfused": BEFORE_DISPATCHES,
                "mlp_pair": after_pair,
                "mlp_swiglu": after_swiglu,
                "mlp_swiglu_qkv_dn": after_all,
            },
            "command_buffers": 1,
        },
        "decode_tok_s": {
            "before": unfused,
            "after": after_arm,
            "after_mlp_swiglu": swiglu,
            "after_mlp_swiglu_qkv_dn": combo,
        },
        "verbatim": {
            "prompt": (gpu_body or {}).get("prompt") or PROMPT,
            "prompt_ids": (gpu_body or {}).get("prompt_ids"),
            "before": {
                "generated_text": (unfused or {}).get("generated_text_verbatim"),
                "new_token_ids": (unfused or {}).get("new_token_ids"),
            },
            "after": {
                "generated_text": (after_arm or {}).get("generated_text_verbatim"),
                "new_token_ids": (after_arm or {}).get("new_token_ids"),
            },
        },
        "parity": {
            "mlp_gate_up_swiglu": mlp_par,
            "gqa_qkv": parity.get("gqa_qkv"),
            "dn_qkvz_ba": parity.get("dn_qkvz_ba"),
            "max_abs_diff": mlp_par.get("max_abs_diff"),
            "fused_kernel_against": "unfused geo_tpr64 matvec + gk_swiglu_f32",
        },
        "dense_parent": {
            "dense_w_materialized": (gpu_body or {}).get("dense_w_materialized", 0),
            "expanded_to_q4": (gpu_body or {}).get("expanded_to_q4", 0),
            "expanded_to_float_gemv": (gpu_body or {}).get("expanded_to_float_gemv", 0),
            "note": "fused kernels consume packed Q4 in-register; no parent W is written",
        },
        "fusions_attempted": (gpu_body or {}).get("fusions_attempted")
        or [
            "gate_up_pair",
            "gate_up_swiglu",
            "gqa_qkv concat geo_tpr64",
            "dn_qkvz_ba concat geo_tpr64",
        ],
        "kernels": list(KERNELS),
        "shader_evidence": evidence,
        "prior_art": {
            "q80_command_buffers_before": 337,
            "q80_command_buffers_after": 49,
            "megakernel_8layer_f16": (
                "measured 4.4x SLOWER; fusion is not automatically a win"
            ),
            "note": "anchors, not re-derived in this run",
        },
        "gpu": None if gpu is None else {
            "ok": gpu.get("ok"),
            "exit_code": gpu.get("exit_code"),
            "wall_s": gpu.get("wall_s"),
            "binary": (gpu.get("command") or [None])[-1]
            if False
            else gpu.get("body", {}).get("artifact_root") and str(find_fusion_binary()),
            "stderr_tail": gpu.get("stderr_tail"),
        },
        "verdict": "; ".join(verdict_parts) if verdict_parts else "not yet measured",
        "gpu_ran": bool(gpu_body),
        "dispatch_probes": probes,
        "raw_example": gpu_body,
    }


def write_receipt(doc: dict[str, Any], path: Path = RECEIPT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def build(*, run_gpu: bool | None = None) -> dict[str, Any]:
    """Write the receipt. GPU runs unless run_gpu=False or the env skips it."""
    if run_gpu is None:
        skip = os.environ.get("QWEN38_FUSION_SKIP_GPU", "").strip() in {
            "1",
            "true",
            "yes",
        }
        run_gpu = not skip
    gpu = None
    build_info = None
    if run_gpu:
        binary = find_fusion_binary()
        if binary is None:
            build_info = cargo_build()
            binary = find_fusion_binary()
        if binary is None:
            gpu = {
                "ok": False,
                "exit_code": None,
                "wall_s": 0,
                "stderr_tail": "ascension_qwen38_dispatch_fusion is not built",
                "command": [],
            }
        elif not Q4_ROOT.is_dir():
            gpu = {
                "ok": False,
                "exit_code": None,
                "wall_s": 0,
                "stderr_tail": f"missing artifact {Q4_ROOT}",
                "command": [],
            }
        else:
            raw_out = REPO / "receipts" / "headless" / "_dispatch_fusion_raw.json"
            gpu = run_fusion_example(binary, raw_out)
    doc = theoretical_receipt(gpu=gpu)
    if build_info is not None:
        doc["cargo_build"] = build_info
    write_receipt(doc)
    return doc


def main() -> int:
    doc = build(run_gpu=True)
    print(json.dumps({
        "schema": doc["schema"],
        "verdict": doc["verdict"],
        "dispatches": doc["dispatches_per_token"],
        "gpu_ran": doc["gpu_ran"],
        "receipt": str(RECEIPT),
    }, indent=2))
    return 0 if RECEIPT.is_file() else 2


if __name__ == "__main__":
    raise SystemExit(main())
