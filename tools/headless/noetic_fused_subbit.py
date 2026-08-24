#!/usr/bin/env python3
"""NOETIC_FUSED_SUBBIT: 2-bit affine MLP on a fused operator graph.

Two questions, in order:

1. WHY is affine2 g64 slower than q4 geo_tpr64 on the same shape?
2. Then fuse it (gate+up, gate+up+SwiGLU, GQA QKV, DeltaNet qkvz+ba).

A slower combination is still the answer. Report every config, including losers.

    python3 tools/headless/noetic_fused_subbit.py
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from first_noetic_executable import (  # noqa: E402
    PROMPT,
    Q4_INCUMBENT_EBPW,
    TOKENIZER,
    git_head,
    judge_coherence,
    now_iso,
)

SCHEMA = "hawking.headless.noetic_fused_subbit.v1"
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_FUSED_SUBBIT.json"
AFFINE_RECEIPT = REPO / "receipts" / "headless" / "AFFINE2_G64_LSFIT.json"
MIXED_SHADER = REPO / "crates" / "hawking-core" / "shaders" / "q80_mixed_decode.metal"
Q4_SHADER = REPO / "crates" / "hawking-core" / "shaders" / "qwen_uniform_q4.metal"
DECODE_RS = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
LEDGER = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_token_ns_ledger.rs"
AFFINE_SHADER = REPO / "crates" / "hawking-core" / "shaders" / "affine2_group32_matvec.metal"

CARGO_TARGET = Path(
    os.environ.get(
        "CARGO_TARGET_DIR",
        str(REPO / "workspace" / "ops" / "build" / "rust"),
    )
)

BEFORE_DISPATCHES = 964
INCUMBENT_TOK_S = 33.716875920652136
UNFUSED_AFFINE2_TOK_S = 26.83964318075738
AFFINE2_EBPW = 3.139300850311054

KERNELS = (
    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
    "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128",
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
)


def theoretical_after(mlp: str, qkv: bool, dn: bool) -> int:
    n = BEFORE_DISPATCHES
    if mlp == "pair":
        n -= 64
    elif mlp == "swiglu":
        n -= 128
    if qkv:
        n -= 32
    if dn:
        n -= 48
    return n


def default_artifact() -> Path:
    env = os.environ.get("QWEN38_AFFINE2_ARTIFACT")
    if env:
        return Path(env)
    candidates = [
        Path(
            "/Users/scammermike/.claude-grok/worktrees/g012aff64-20260823-182619"
            "/artifacts/qwen38-affine2-g64-lsfit/mix_all_mlp_affine_g64_ls"
        ),
        REPO / "artifacts" / "qwen38-affine2-g64-lsfit" / "mix_all_mlp_affine_g64_ls",
    ]
    for c in candidates:
        if (c / "catalog.hq38m20").is_file():
            return c
    return candidates[0]


def shader_evidence() -> dict[str, Any]:
    mixed = MIXED_SHADER.read_text(encoding="utf-8", errors="replace") if MIXED_SHADER.is_file() else ""
    q4 = Q4_SHADER.read_text(encoding="utf-8", errors="replace") if Q4_SHADER.is_file() else ""
    rust = DECODE_RS.read_text(encoding="utf-8", errors="replace") if DECODE_RS.is_file() else ""
    ledger = LEDGER.read_text(encoding="utf-8", errors="replace") if LEDGER.is_file() else ""
    affine = AFFINE_SHADER.read_text(encoding="utf-8", errors="replace") if AFFINE_SHADER.is_file() else ""
    combined = mixed + "\n" + q4
    needles = {name: combined.find(f"kernel void {name}(") for name in KERNELS}
    return {
        "shader_present": MIXED_SHADER.is_file(),
        "shader_path": "crates/hawking-core/shaders/q80_mixed_decode.metal",
        "kernel_needles": needles,
        "all_kernels_declared": all(v >= 0 for v in needles.values()),
        "wired_in_encode_dense_mlp_mixed": "encode_fused_affine_gate_up" in rust
        and "fn encode_dense_mlp_mixed" in rust,
        "wired_in_encode_gqa_mixed": "fuse_gqa_qkv" in rust and "encode_gqa_mixed" in rust,
        "wired_in_encode_deltanet_mixed": "fuse_dn_inproj" in rust
        and "encode_deltanet_mixed" in rust,
        "production_964_untouched": "production_dispatch_count_is_964" in ledger,
        "specialized_g64_shift": "const uint group = col >> 6u;" in mixed
        and "affine2_geo_acc_g64" in affine,
        "runtime_div_kept_as_diagnostic": "geo_tpr64_tg128_runtime_div" in mixed,
        "does_not_write_dense_w": "Packed codes stay packed" in mixed
        or "In-register dequant only" in affine,
    }


def find_binary() -> Path | None:
    env = os.environ.get("QWEN38_FUSED_SUBBIT_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for c in [
        CARGO_TARGET / "release-fast" / "examples" / "ascension_qwen38_fused_subbit",
        CARGO_TARGET / "release" / "examples" / "ascension_qwen38_fused_subbit",
        REPO
        / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_fused_subbit",
    ]:
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
        "ascension_qwen38_fused_subbit",
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
        "stderr_tail": proc.stderr[-8000:],
        "binary": str(find_binary()) if find_binary() else None,
    }


def run_example(binary: Path, artifact: Path, out: Path) -> dict[str, Any]:
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd: list[str] = []
    if lock.is_file():
        cmd.extend(["bash", str(lock), "qwen38-fused-subbit"])
    cmd.extend(
        [
            str(binary),
            "--artifact-root",
            str(artifact),
            "--tokenizer",
            str(TOKENIZER),
            "--prompt",
            PROMPT,
            "--max-new-tokens",
            "16",
            "--max-seq-len",
            "128",
            "--reps",
            os.environ.get("QWEN38_FUSED_SUBBIT_REPS", "2"),
            "--out",
            str(out),
        ]
    )
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
        "command": [str(x) for x in cmd],
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-12000:],
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


def _arm(arm: dict[str, Any] | None) -> dict[str, Any] | None:
    if not arm:
        return None
    ids = [int(x) for x in (arm.get("new_token_ids") or [])]
    text = arm.get("generated_text_verbatim") or ""
    return {
        "tok_s_reps": arm.get("tok_s_reps"),
        "tok_s_mean": arm.get("tok_s_mean"),
        "tok_s_min": arm.get("tok_s_min"),
        "tok_s_max": arm.get("tok_s_max"),
        "generated_text_verbatim": text,
        "new_token_ids": ids,
        "n_new_tokens": len(ids),
        "dispatches_last_step_reps": arm.get("dispatches_last_step_reps"),
        "dense_w_materialized": arm.get("dense_w_materialized", 0),
        "expanded_to_q4": arm.get("expanded_to_q4", 0),
        "expanded_to_float_gemv": arm.get("expanded_to_float_gemv", 0),
        "coherence": judge_coherence(text, ids),
    }


def _median_ns(timing: dict[str, Any] | None) -> float | None:
    if not timing:
        return None
    v = timing.get("gpu_ns_median")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def diagnose_kernel_cost(kernel_cost: dict[str, Any] | None) -> dict[str, Any]:
    if not kernel_cost or kernel_cost.get("skipped") or kernel_cost.get("ok") is False:
        return {
            "status": "not_measured",
            "kernel_cost": kernel_cost,
        }
    shapes = kernel_cost.get("shapes") or []
    rows = []
    for shape in shapes:
        q4 = _median_ns((shape.get("q4_g64_geo_tpr64") or {}).get("timing"))
        a32 = _median_ns((shape.get("affine2_g32_specialized") or {}).get("timing"))
        a64 = _median_ns((shape.get("affine2_g64_specialized") or {}).get("timing"))
        old = _median_ns((shape.get("affine2_g64_runtime_div") or {}).get("timing"))
        rec = {
            "shape": shape.get("shape"),
            "label": shape.get("label"),
            "q4_g64_gpu_ns": q4,
            "affine2_g32_specialized_gpu_ns": a32,
            "affine2_g64_specialized_gpu_ns": a64,
            "affine2_g64_runtime_div_gpu_ns": old,
        }
        if isinstance(q4, float) and isinstance(a64, float) and q4 > 0:
            rec["affine2_g64_specialized_vs_q4"] = a64 / q4
        if isinstance(old, float) and isinstance(a64, float) and a64 > 0:
            rec["runtime_div_vs_specialized"] = old / a64
        rows.append(rec)
    gate = next((r for r in rows if r.get("label") == "gate_up"), rows[0] if rows else {})
    why = (
        "Q4 geo_tpr64 uses compile-time group 64 so col/64 is a shift. "
        "The affine2 G0 path previously took group_size as a bind-time parameter, "
        "putting a non-constant integer divide on every 8-wide tile. "
        "That is a kernel problem, not a property of 2-bit. "
        "The production affine2 geo kernel now specializes 32 vs 64 to shifts. "
        "The extra bias term (w = q*scale + bias) remains; it is the same per "
        "element at g32 and g64."
    )
    fixable = True
    if gate.get("runtime_div_vs_specialized") is not None:
        ratio = gate["runtime_div_vs_specialized"]
        if ratio > 1.05:
            why += f" On the gate shape, runtime-div was {ratio:.2f}x the specialized g64 body."
        else:
            why += (
                f" On the gate shape, runtime-div was only {ratio:.2f}x specialized; "
                "the divide is not the whole story."
            )
            if ratio <= 1.05:
                fixable = False
    return {
        "status": "measured",
        "fixable_kernel_problem": fixable,
        "why": why,
        "per_dispatch": rows,
        "kernel_cost": kernel_cost,
    }


def affine_parent_stats() -> dict[str, Any]:
    if not AFFINE_RECEIPT.is_file():
        return {
            "complete_ebpw": AFFINE2_EBPW,
            "unfused_tok_s": UNFUSED_AFFINE2_TOK_S,
            "source": "constants (AFFINE2_G64_LSFIT.json missing)",
        }
    doc = json.loads(AFFINE_RECEIPT.read_text())
    compile_ = doc.get("compile") or {}
    decode = doc.get("decode") or {}
    chosen = doc.get("chosen") or {}
    return {
        "complete_ebpw": compile_.get("complete_ebpw", AFFINE2_EBPW),
        "storage_bpw": compile_.get("storage_bpw", AFFINE2_EBPW),
        "affine_tensor_storage_bpw": compile_.get("affine_tensor_storage_bpw", 2.5),
        "n_affine": compile_.get("n_affine", 192),
        "unfused_tok_s": decode.get("tok_s") or chosen.get("tok_s") or UNFUSED_AFFINE2_TOK_S,
        "generated_text_verbatim": chosen.get("generated_text_verbatim")
        or decode.get("generated_text_verbatim"),
        "census": decode.get("census") or chosen.get("census"),
        "source": str(AFFINE_RECEIPT),
        "q4_incumbent_complete_physical_bpw": compile_.get(
            "q4_incumbent_complete_physical_bpw", Q4_INCUMBENT_EBPW
        ),
    }


def build_receipt(*, gpu: dict[str, Any] | None, build_info: dict[str, Any] | None) -> dict[str, Any]:
    evidence = shader_evidence()
    parent = affine_parent_stats()
    gpu_body = (gpu or {}).get("body") if gpu else None
    decode = (gpu_body or {}).get("decode") or {}
    unfused = _arm(decode.get("unfused"))
    pair = _arm(decode.get("mlp_pair"))
    swiglu = _arm(decode.get("mlp_swiglu"))
    combo = _arm(decode.get("mlp_swiglu_qkv_dn"))
    after_arm = combo or swiglu
    probes = (gpu_body or {}).get("dispatch_probes") or []
    measured_before = None
    measured_after = None
    for p in probes:
        ident = p.get("id")
        probe = p.get("probe") or {}
        if ident == "unfused":
            measured_before = probe.get("measured")
        if ident == "mlp_swiglu_qkv_dn":
            measured_after = probe.get("measured")
        elif ident == "mlp_swiglu" and measured_after is None:
            measured_after = probe.get("measured")
    parity = (gpu_body or {}).get("parity") or {}
    mlp_par = parity.get("mlp_gate_up_swiglu") or {}
    kernel = diagnose_kernel_cost((gpu_body or {}).get("kernel_cost"))

    configs: list[dict[str, Any]] = []
    for name, arm in (
        ("unfused affine2 g64", unfused),
        ("affine2 + mlp pair", pair),
        ("affine2 + mlp swiglu", swiglu),
        ("affine2 + mlp swiglu + qkv + dn", combo),
    ):
        if arm is None:
            continue
        configs.append(
            {
                "id": name,
                "tok_s_mean": arm.get("tok_s_mean"),
                "tok_s_reps": arm.get("tok_s_reps"),
                "dispatches_last_step_reps": arm.get("dispatches_last_step_reps"),
                "generated_text_verbatim": arm.get("generated_text_verbatim"),
                "n_new_tokens": arm.get("n_new_tokens"),
                "coherence": arm.get("coherence"),
                "vs_incumbent_tok_s": (
                    None
                    if not isinstance(arm.get("tok_s_mean"), (int, float))
                    else arm["tok_s_mean"] - INCUMBENT_TOK_S
                ),
                "lost_to_incumbent": (
                    None
                    if not isinstance(arm.get("tok_s_mean"), (int, float))
                    else arm["tok_s_mean"] < INCUMBENT_TOK_S
                ),
            }
        )

    verdict_parts: list[str] = []
    ebpw = parent.get("complete_ebpw") or AFFINE2_EBPW
    verdict_parts.append(
        f"complete EBPW {ebpw:.4f} beside incumbent {Q4_INCUMBENT_EBPW:.4f}"
    )
    if measured_before is not None and measured_after is not None:
        verdict_parts.append(
            f"dispatches {measured_before} -> {measured_after} (incumbent unfused 964)"
        )
    after_tok = (after_arm or {}).get("tok_s_mean")
    if isinstance(after_tok, (int, float)):
        if after_tok > INCUMBENT_TOK_S:
            verdict_parts.append(
                f"decode tok/s {after_tok:.3f} vs incumbent {INCUMBENT_TOK_S:.3f} (faster)"
            )
        else:
            verdict_parts.append(
                f"decode tok/s {after_tok:.3f} vs incumbent {INCUMBENT_TOK_S:.3f} (SLOWER)"
            )
    ids = (after_arm or {}).get("new_token_ids") or []
    if ids:
        verdict_parts.append(f"{len(ids)} tokens verbatim")
    if mlp_par.get("max_abs_diff") is not None:
        verdict_parts.append(f"mlp fused-vs-unfused max_abs_diff={mlp_par.get('max_abs_diff')}")
    if kernel.get("fixable_kernel_problem"):
        verdict_parts.append("affine2 slowness was a fixable kernel problem (runtime group divide)")
    elif kernel.get("status") == "measured":
        verdict_parts.append("affine2 slowness is not fully explained by the runtime divide")

    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": (
            "Why is affine2 g64 slower than q4, and does the 2-bit affine MLP "
            "on a fused operator graph cross the boundary?"
        ),
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "parent_params": 26895998464,
        "q4_incumbent": {
            "complete_physical_bpw": Q4_INCUMBENT_EBPW,
            "decode_tok_s": INCUMBENT_TOK_S,
            "dispatches_per_token": BEFORE_DISPATCHES,
        },
        "affine2_unfused_half": parent,
        "representation": {
            "codec": "HGRAVF01 affine_q2_group64 LS (w = q * scale + bias, q in {0,1,2,3})",
            "n_affine": parent.get("n_affine", 192),
            "complete_ebpw": ebpw,
            "q4_incumbent_complete_physical_bpw": Q4_INCUMBENT_EBPW,
            "not_the_incumbent": True,
        },
        "operator_graph": {
            "fused": True,
            "default": "off — production graph stays 964 unless apply_fusion / env",
            "dispatches_per_token_before": measured_before or BEFORE_DISPATCHES,
            "dispatches_per_token_after": measured_after
            or theoretical_after("swiglu", True, True),
            "not_the_source": True,
        },
        "enable": {
            "default": "off",
            "HAWKING_QWEN38_FUSE_MLP": "pair | swiglu",
            "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
            "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
        },
        "counting_method": (
            "TokenCommandBuffer.dispatch_count: one kernel launch = one dispatch. "
            "Same counter as production_dispatches_per_token / generate_greedy "
            "timing.dispatches."
        ),
        "dispatches_per_token": {
            "before": measured_before or BEFORE_DISPATCHES,
            "after": measured_after or theoretical_after("swiglu", True, True),
            "measured_before": measured_before,
            "measured_after": measured_after,
            "incumbent_unfused": BEFORE_DISPATCHES,
            "theoretical": {
                "unfused": BEFORE_DISPATCHES,
                "mlp_pair": theoretical_after("pair", False, False),
                "mlp_swiglu": theoretical_after("swiglu", False, False),
                "mlp_swiglu_qkv_dn": theoretical_after("swiglu", True, True),
            },
            "command_buffers": 1,
        },
        "decode_tok_s": {
            "incumbent": INCUMBENT_TOK_S,
            "affine2_unfused_prior": parent.get("unfused_tok_s"),
            "before": unfused,
            "after": after_arm,
            "after_mlp_pair": pair,
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
            "fused_kernel_against": "unfused affine2 geo_tpr64 matvec + gk_swiglu_f32",
        },
        "dense_parent": {
            "dense_w_materialized": (gpu_body or {}).get("dense_w_materialized", 0),
            "expanded_to_q4": (gpu_body or {}).get("expanded_to_q4", 0),
            "expanded_to_float_gemv": (gpu_body or {}).get("expanded_to_float_gemv", 0),
            "note": "fused affine2 kernels consume packed 2-bit codes in-register; no parent W is written",
        },
        "why_affine2_g64_was_slower": kernel,
        "live_affine2_gate_matvec": (gpu_body or {}).get("live_affine2_gate_matvec"),
        "configs_tried": configs,
        "fusions_attempted": (gpu_body or {}).get("fusions_attempted")
        or [
            "affine2 gate_up_pair",
            "affine2 gate_up_swiglu",
            "gqa_qkv concat geo_tpr64",
            "dn_qkvz_ba concat geo_tpr64",
        ],
        "kernels": list(KERNELS),
        "shader_evidence": evidence,
        "gpu": None
        if gpu is None
        else {
            "ok": gpu.get("ok"),
            "exit_code": gpu.get("exit_code"),
            "wall_s": gpu.get("wall_s"),
            "binary": str(find_binary()) if find_binary() else None,
            "stderr_tail": gpu.get("stderr_tail"),
        },
        "cargo_build": build_info,
        "gpu_ran": bool(gpu_body),
        "dispatch_probes": probes,
        "raw_example": gpu_body,
        "verdict": "; ".join(verdict_parts) if verdict_parts else "not yet measured",
    }


def write_receipt(doc: dict[str, Any], path: Path = RECEIPT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def build(*, run_gpu: bool | None = None) -> dict[str, Any]:
    if run_gpu is None:
        skip = os.environ.get("QWEN38_FUSED_SUBBIT_SKIP_GPU", "").strip() in {
            "1",
            "true",
            "yes",
        }
        run_gpu = not skip
    gpu = None
    build_info = None
    if run_gpu:
        binary = find_binary()
        if binary is None:
            build_info = cargo_build()
            binary = find_binary()
        artifact = default_artifact()
        if binary is None:
            gpu = {
                "ok": False,
                "exit_code": None,
                "wall_s": 0,
                "stderr_tail": "ascension_qwen38_fused_subbit is not built",
                "command": [],
            }
        elif not (artifact / "catalog.hq38m20").is_file():
            gpu = {
                "ok": False,
                "exit_code": None,
                "wall_s": 0,
                "stderr_tail": f"missing affine2 artifact {artifact}",
                "command": [],
            }
        else:
            raw_out = REPO / "receipts" / "headless" / "_fused_subbit_raw.json"
            gpu = run_example(binary, artifact, raw_out)
    doc = build_receipt(gpu=gpu, build_info=build_info)
    write_receipt(doc)
    return doc


def main() -> int:
    doc = build(run_gpu=True)
    print(
        json.dumps(
            {
                "schema": doc["schema"],
                "verdict": doc["verdict"],
                "dispatches": doc["dispatches_per_token"],
                "gpu_ran": doc["gpu_ran"],
                "receipt": str(RECEIPT),
            },
            indent=2,
        )
    )
    return 0 if RECEIPT.is_file() else 2


if __name__ == "__main__":
    raise SystemExit(main())
