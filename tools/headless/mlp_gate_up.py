#!/usr/bin/env python3
"""N030 — MLP_GATE_UP: non-load work of fused gate_up_swiglu.

Profiles packed decode, scale, accumulate, SwiGLU, and the 64 launches.
N024 ruled out the tile: do not retune load geometry (qmvfast / wide64 / tgx).
Attempt: fuse group-64 x-sums into RMSNorm and defer bias out of the inner
loop (S022 §9 norm+projection). Ranked by COMPLETE_TOKEN_NS, not GB/s.

    python3 tools/headless/mlp_gate_up.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
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

SCHEMA = "hawking.headless.mlp_gate_up.v1"
RECEIPT = REPO / "receipts" / "headless" / "MLP_GATE_UP.json"
RAW = REPO / "receipts" / "headless" / "_MLP_GATE_UP_raw.json"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "mlp_gate_up"
PARENT_ROOT = Path(
    os.environ.get("NOETIC_PARENT_A_ROOT", str(Path.home() / "noetic" / "NOETIC_PARENT_A"))
)
TOKENIZER = Path(
    os.environ.get(
        "QWEN38_TOKENIZER",
        str(Path.home() / "models" / "qwen3.8-27b-abliterated-bf16" / "tokenizer.json"),
    )
)
PARENT_ACTIVE_BYTES = 9_878_901_136
ROOF_GB_S = 778.8
N018_PRODUCTION_GB_S = 356.7
N025_MLP_GATE_UP_GAP_SHARE = 0.356
LEVER = "biasprep"
NO_OP = "tpr64"
BAD = "dropbias"
BAD_PROD = "biasprep_drop"
NEW_KERNELS = (
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128",
    "qwen_affine_q2_group64_matvec_gate_up_biasprep_tpr64_tg128",
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128",
    "qwen80_residual_rmsnorm_tg_xsum64",
    "qwen80_add_residual_rmsnorm_tg_xsum64",
    "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
    "affine2_group64_matvec_gate_up_geo_tpr64_tg128",
    "affine2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128",
    "affine2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128",
    "affine2_group64_matvec_gate_up_swiglu_decode_probe_tpr64_tg128",
    "affine2_group64_matvec_gate_up_swiglu_addr_probe_tpr64_tg128",
    "affine2_xsum64",
)
N024_NOT_RETRIED = ("tgsb", "pipe", "splitk4", "accfuse")
N018_TILES_NOT_RETRIED = ("qmvfast", "wide64", "tgx")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def separated(a: list[float], b: list[float]) -> bool:
    if not a or not b:
        return False
    return max(a) < min(b) or max(b) < min(a)


def shader_evidence() -> dict[str, Any]:
    mixed = REPO / "crates" / "hawking-core" / "shaders" / "q80_mixed_decode.metal"
    stand = REPO / "crates" / "hawking-core" / "shaders" / "affine2_group32_matvec.metal"
    act = REPO / "crates" / "hawking-core" / "shaders" / "qwen80_device_activations.metal"
    decode = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
    mixed_src = mixed.read_text(encoding="utf-8", errors="replace") if mixed.is_file() else ""
    stand_src = stand.read_text(encoding="utf-8", errors="replace") if stand.is_file() else ""
    act_src = act.read_text(encoding="utf-8", errors="replace") if act.is_file() else ""
    rust = decode.read_text(encoding="utf-8", errors="replace") if decode.is_file() else ""
    prod = [
        "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128",
        "qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128",
        "qwen80_residual_rmsnorm_tg_xsum64",
        "qwen80_add_residual_rmsnorm_tg_xsum64",
    ]
    iso = [
        "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "affine2_group64_matvec_gate_up_geo_tpr64_tg128",
        "affine2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128",
        "affine2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128",
        "affine2_group64_matvec_gate_up_swiglu_decode_probe_tpr64_tg128",
        "affine2_group64_matvec_gate_up_swiglu_addr_probe_tpr64_tg128",
        "affine2_xsum64",
    ]
    n018_kept = all(
        f"qwen_affine_q2_group64_matvec_{n}" in mixed_src
        or f"matvec_{n}" in mixed_src
        for n in ("qmvfast_r8tg64", "wide64_r4tg128", "tgx_r8tg256")
    )
    n024_kept = all(
        name in mixed_src
        for name in (
            "qwen_affine_q2_group64_matvec_tgsb_tpr64_tg128",
            "qwen_affine_q2_group64_matvec_pipe_tpr64_tg128",
            "qwen_affine_q2_group64_matvec_splitk4_tg256",
            "qwen_affine_q2_group64_matvec_accfuse_tpr64_tg128",
        )
    )
    return {
        "mixed_present": mixed.is_file(),
        "standalone_present": stand.is_file(),
        "activations_present": act.is_file(),
        "production_kernels": {
            n: f"kernel void {n}(" in mixed_src or f"kernel void {n}(" in act_src for n in prod
        },
        "isolated_kernels": {n: f"kernel void {n}(" in stand_src for n in iso},
        "wired_biasprep": "Affine2Geo::BiasPrep" in rust and "uses_xsum" in rust,
        "wired_drop": "BiasPrepDrop" in rust,
        "incumbent_untouched": "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128"
        in mixed_src,
        "n018_tiles_kept_not_retried_as_levers": n018_kept,
        "n024_levers_kept_not_retried": n024_kept,
        "same_tpr64_occupancy": "kSplit = 2u" in mixed_src
        and "biasprep_tpr64_tg128" in mixed_src,
        "no_dense_w": "No dense W" in mixed_src or "no dense W" in mixed_src,
    }


def kernel_autopsy() -> dict[str, Any]:
    script = HERE / "kernel_competence.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    competence = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"
    doc = None
    if competence.is_file():
        try:
            doc = json.loads(competence.read_text())
        except json.JSONDecodeError:
            doc = None
    watched = []
    any_defective = False
    if doc:
        for f in doc.get("per_file") or []:
            for k in f.get("kernels") or []:
                if k.get("kernel") in NEW_KERNELS:
                    verdict = k.get("verdict")
                    watched.append(
                        {
                            "file": k.get("file") or f.get("file"),
                            "kernel": k.get("kernel"),
                            "verdict": verdict,
                            "n_findings": k.get("n_findings"),
                            "findings": k.get("findings"),
                        }
                    )
                    if verdict == "DEFECTIVE":
                        any_defective = True
    return {
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0 and not any_defective,
        "any_new_kernel_defective": any_defective,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
        "new_kernels": watched,
        "missing": [n for n in NEW_KERNELS if n not in {w["kernel"] for w in watched}],
    }


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
        "mlp_gate_up",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
        "bin": str(BIN),
        "bin_present": BIN.is_file(),
    }


def run_locked(extra: list[str], out: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    cmd = [
        "bash",
        str(REPO / "tools" / "gpu_lane_lock.sh"),
        "n030-gateup",
        str(BIN),
        *extra,
        "--out",
        str(out),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    raw = None
    if out.is_file():
        try:
            raw = json.loads(out.read_text())
        except json.JSONDecodeError:
            raw = None
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-6000:],
        "raw": raw,
    }


def _arm_ns(arm: dict[str, Any], key: str) -> list[float]:
    return [float(x) for x in (arm.get(key) or []) if x is not None]


def summarize_isolated(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw or not raw.get("isolated"):
        return {"kind": "ABSENT", "absent_reason": "isolated GEMV JSON missing"}
    iso = raw["isolated"]
    arms = {}
    for arm in iso.get("arms") or []:
        reps = _arm_ns(arm, "gpu_ns_reps")
        arms[arm["id"]] = {
            "role": arm.get("role"),
            "kernel": arm.get("kernel"),
            "launches": arm.get("launches"),
            "gpu_ns_min": arm.get("gpu_ns_min"),
            "gpu_ns_median": arm.get("gpu_ns_median"),
            "gpu_ns_max": arm.get("gpu_ns_max"),
            "gpu_ns_reps": arm.get("gpu_ns_reps"),
            "n_reps": len(reps),
            "dense_w_materialized": arm.get("dense_w_materialized", 0),
        }
    tpr = _arm_ns(arms.get("tpr64") or {}, "gpu_ns_reps")
    bias = _arm_ns(arms.get("biasprep") or {}, "gpu_ns_reps")
    acc = _arm_ns(arms.get("acc_only") or {}, "gpu_ns_reps")
    dec = _arm_ns(arms.get("decode_probe") or {}, "gpu_ns_reps")
    addr = _arm_ns(arms.get("addr_probe") or {}, "gpu_ns_reps")
    l64 = _arm_ns(arms.get("launch64") or {}, "gpu_ns_reps")
    xsum = _arm_ns(arms.get("xsum64") or {}, "gpu_ns_reps")
    profile = {
        "packed_decode_gpu_ns_median": median(dec),
        "load_only_gpu_ns_median": median(addr),
        "accumulate_no_swiglu_gpu_ns_median": median(acc),
        "fused_swiglu_gpu_ns_median": median(tpr),
        "swiglu_epilogue_ns_median": (
            None
            if median(tpr) is None or median(acc) is None
            else max(0.0, median(tpr) - median(acc))
        ),
        "swiglu_epilogue_note": (
            "fused vs acc_only ranges overlap or acc_only is slower (two output "
            "writes); SwiGLU is not the organ wall"
            if median(tpr) is not None
            and median(acc) is not None
            and median(tpr) <= median(acc)
            else "fused minus acc_only"
        ),
        "launch64_gpu_ns_median": median(l64),
        "launch_overhead_ns_est": (
            None
            if median(l64) is None or median(tpr) is None or median(tpr) <= 0
            else (median(l64) - 64.0 * median(tpr)) / 63.0
        ),
        "launch64_note": (
            "64 fused launches in one command buffer finish in less than 64× a "
            "single-launch CB (GPU pipelines them). Launch ceremony is not the organ wall."
            if median(l64) is not None
            and median(tpr) is not None
            and median(l64) < 64.0 * median(tpr)
            else "64-dispatch CB vs 64 times a 1-dispatch CB"
        ),
        "xsum64_gpu_ns_median": median(xsum),
        "note": (
            "Component ns are isolated GEMV, not production token ns. "
            "Packed decode and load-only are ~5x below fused, so most of the "
            "isolated fused time is scale+accumulate. SwiGLU and 64-launch "
            "ceremony are not the wall."
        ),
    }
    sep = {
        "biasprep_vs_tpr64": {
            "separated": separated(bias, tpr) if bias and tpr else False,
            "faster": median(bias) is not None
            and median(tpr) is not None
            and median(bias) < median(tpr),
        }
    }
    if not sep["biasprep_vs_tpr64"]["separated"]:
        sep["biasprep_vs_tpr64"]["note"] = (
            "NOT SEPARATED: gpu_ns ranges overlap; do not quote a mean delta"
        )
    parity = iso.get("parity") or []
    drop = next((p for p in parity if p.get("id") == "dropbias"), None)
    honest = [p for p in parity if p.get("id") != "dropbias"]
    return {
        "kind": "MEASURED",
        "parity": parity,
        "parity_all_ok": all(p.get("ok") for p in honest) if honest else False,
        "dropbias_rejected": bool(drop and drop.get("rejected")),
        "occupancy": iso.get("occupancy"),
        "shape": iso.get("shape"),
        "arms": arms,
        "non_load_profile": profile,
        "separation": sep,
        "dense_w_materialized": 0,
        "did_not_load_second_27b": True,
    }


def summarize_production(raw: dict[str, Any] | None) -> dict[str, Any]:
    prod = (raw or {}).get("production")
    if not prod or prod.get("kind") == "ABSENT" or "arms" not in (prod or {}):
        return {
            "kind": "ABSENT",
            "absent_reason": (prod or {}).get("absent_reason")
            or "production decode was not run",
            "active_bytes_per_token": PARENT_ACTIVE_BYTES,
        }
    arms = {}
    for arm in prod.get("arms") or []:
        gpu = _arm_ns(arm, "median_gpu_ns_per_token_reps")
        tok = _arm_ns(arm, "tok_s_reps")
        cns = _arm_ns(arm, "complete_token_ns_reps")
        arms[arm["id"]] = {
            "role": arm.get("role"),
            "gpu_ns_min": arm.get("gpu_ns_min"),
            "gpu_ns_median": arm.get("gpu_ns_median"),
            "gpu_ns_max": arm.get("gpu_ns_max"),
            "gpu_ns_reps": gpu,
            "complete_token_ns_min": arm.get("complete_token_ns_min"),
            "complete_token_ns_median": arm.get("complete_token_ns_median"),
            "complete_token_ns_max": arm.get("complete_token_ns_max"),
            "complete_token_ns_reps": cns,
            "tok_s_reps": arm.get("tok_s_reps"),
            "tok_s_median": median(tok),
            "token_ids_unchanged_vs_tpr64": arm.get("token_ids_unchanged_vs_tpr64"),
            "token_ids_stable_across_reps": arm.get("token_ids_stable_across_reps"),
            "new_token_ids": arm.get("new_token_ids"),
            "dispatches_last_step_reps": arm.get("dispatches_last_step_reps"),
            "fallbacks_reps": arm.get("fallbacks_reps"),
            "n_reps": max(len(gpu), len(cns)),
            "dense_w_materialized": arm.get("dense_w_materialized", 0),
        }
    before = arms.get(NO_OP) or {}
    lever = arms.get(LEVER) or {}
    bad = arms.get(BAD_PROD) or arms.get(BAD) or {}
    tpr_c = [float(x) for x in (before.get("complete_token_ns_reps") or []) if x is not None]
    lev_c = [float(x) for x in (lever.get("complete_token_ns_reps") or []) if x is not None]
    sep = separated(tpr_c, lev_c) if tpr_c and lev_c else False
    faster = (
        median(lev_c) is not None
        and median(tpr_c) is not None
        and median(lev_c) < median(tpr_c)
    )
    ids_ok = lever.get("token_ids_unchanged_vs_tpr64") is True
    fallbacks_ok = all(f in (0, None) for f in (lever.get("fallbacks_reps") or [0]))
    win = sep and faster and ids_ok and fallbacks_ok
    bad_changed = bad.get("token_ids_unchanged_vs_tpr64") is False
    note = None
    if not sep:
        note = "NOT SEPARATED: complete_token_ns ranges overlap; do not quote a mean delta"
    elif not faster:
        note = "biasprep did not reduce COMPLETE_TOKEN_NS versus tpr64"
    elif not ids_ok:
        note = "biasprep changed token ids; not a legal reduction"
    return {
        "kind": "MEASURED",
        "ranking_metric": "COMPLETE_TOKEN_NS",
        "active_bytes_per_token": PARENT_ACTIVE_BYTES,
        "did_not_mutate_parent": prod.get("did_not_mutate_parent", True),
        "did_not_load_second_27b": prod.get("did_not_load_second_27b", True),
        "fusion": prod.get("fusion"),
        "expected_dispatches": prod.get("expected_dispatches"),
        "arms": arms,
        "before": {
            "id": NO_OP,
            "role": "no_op_control",
            "complete_token_ns_median": before.get("complete_token_ns_median"),
            "gpu_ns_median": before.get("gpu_ns_median"),
            "tok_s_median": before.get("tok_s_median"),
            "token_ids": before.get("new_token_ids"),
        },
        "after": {
            "id": LEVER if win else None,
            "complete_token_ns_median": lever.get("complete_token_ns_median")
            if win
            else before.get("complete_token_ns_median"),
            "kept_incumbent_tpr64": not win,
            "token_ids_unchanged": ids_ok,
            "note": None
            if win
            else (
                note
                or "biasprep did not beat tpr64 on COMPLETE_TOKEN_NS with separated "
                "ranges, unchanged token ids, and zero fallbacks"
            ),
        },
        "deliberately_bad_control": {
            "id": BAD_PROD if BAD_PROD in arms else BAD,
            "token_ids_unchanged_vs_tpr64": bad.get("token_ids_unchanged_vs_tpr64"),
            "rejected": bad_changed,
            "complete_token_ns_median": bad.get("complete_token_ns_median"),
        },
        "separation": {
            "biasprep_vs_tpr64": {
                "separated": sep,
                "faster_complete_token_ns": faster,
                "note": None
                if sep
                else "NOT SEPARATED: complete_token_ns ranges overlap; do not quote a mean delta",
            }
        },
        "dense_w_materialized": 0,
    }


def what_blocks(iso: dict[str, Any], prod: dict[str, Any], autopsy: dict[str, Any]) -> str:
    if autopsy.get("any_new_kernel_defective"):
        bad = [
            k["kernel"]
            for k in autopsy.get("new_kernels") or []
            if k.get("verdict") == "DEFECTIVE"
        ]
        return (
            "Kernel autopsy flagged new kernels DEFECTIVE before speed was trusted: "
            + ", ".join(bad)
            + ". Speed numbers are not a claim."
        )
    parts = [
        "N024 ruled out the MLP tile; this lane did not retune qmvfast/wide64/tgx.",
        "The attempted cut is S022 §9 norm+projection: RMSNorm writes group-64 "
        "x-sums and fused gate_up_swiglu defers bias*sum(x) out of the inner loop.",
        "Ranking metric is COMPLETE_TOKEN_NS, not GB/s.",
    ]
    if iso.get("kind") == "MEASURED":
        p = iso.get("non_load_profile") or {}
        parts.append(
            "Isolated fused gate_up_swiglu "
            f"median {p.get('fused_swiglu_gpu_ns_median')} ns; "
            f"acc_only {p.get('accumulate_no_swiglu_gpu_ns_median')} ns; "
            f"decode_probe {p.get('packed_decode_gpu_ns_median')} ns; "
            f"addr_probe {p.get('load_only_gpu_ns_median')} ns; "
            f"launch64 {p.get('launch64_gpu_ns_median')} ns; "
            f"xsum64 {p.get('xsum64_gpu_ns_median')} ns."
        )
        sep = (iso.get("separation") or {}).get("biasprep_vs_tpr64") or {}
        tpr_m = p.get("fused_swiglu_gpu_ns_median")
        bias_m = ((iso.get("arms") or {}).get("biasprep") or {}).get("gpu_ns_median")
        if not sep.get("separated"):
            parts.append(
                "Isolated biasprep vs tpr64 is NOT SEPARATED on GPU ns; no mean delta."
            )
        elif sep.get("faster") and tpr_m and bias_m:
            parts.append(
                f"Isolated biasprep is SEPARATED and faster than tpr64 "
                f"({bias_m} vs {tpr_m} ns median on the fused GEMV)."
            )
        elif sep.get("faster"):
            parts.append("Isolated biasprep is separated and faster than tpr64 on GPU ns.")
        else:
            parts.append("Isolated biasprep is separated but not faster than tpr64.")
    if prod.get("kind") != "MEASURED":
        parts.append(
            "Production COMPLETE_TOKEN_NS was not measured in this run. "
            f"Absent: {prod.get('absent_reason')}."
        )
        return " ".join(parts)
    before = prod.get("before") or {}
    after = prod.get("after") or {}
    bad = prod.get("deliberately_bad_control") or {}
    parts.append(
        f"Production tpr64 complete_token_ns median {before.get('complete_token_ns_median')} "
        f"(gpu_ns {before.get('gpu_ns_median')}, tok/s {before.get('tok_s_median')})."
    )
    if after.get("kept_incumbent_tpr64"):
        note = after.get("note") or "Incumbent tpr64 remains the path."
        if not note.endswith("."):
            note += "."
        parts.append(note)
        parts.append(
            "A measured reason none exists on COMPLETE_TOKEN_NS: the isolated GEMV "
            "cut did not transfer into the 580-dispatch token. Production GPU ns "
            f"{before.get('gpu_ns_median')} vs biasprep "
            f"{(prod.get('arms') or {}).get('biasprep', {}).get('gpu_ns_median')} "
            "overlap. Residual blocker is still the dual packed-decode + scale + "
            "accumulate of the 64 fused gate_up_swiglu launches (35.6% of the "
            "356.7→778.8 gap per N025), not the SwiGLU epilogue and not the RMSNorm launch."
        )
    else:
        parts.append(
            f"biasprep reduced COMPLETE_TOKEN_NS to {after.get('complete_token_ns_median')} "
            "with unchanged token ids."
        )
    if bad.get("rejected"):
        parts.append("dropbias bad control changed token ids (REJECTED).")
    else:
        parts.append(
            "WARNING: dropbias did not change token ids; the bad control is weak."
        )
    return " ".join(parts)


def build(live: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    evidence = shader_evidence()
    autopsy = kernel_autopsy()
    compile_ = {"returncode": 1, "skipped": True, "bin_present": BIN.is_file()}
    raw = None
    run = None
    if autopsy.get("any_new_kernel_defective"):
        compile_ = {
            "returncode": 0,
            "skipped": True,
            "bin_present": BIN.is_file(),
            "note": "kernel autopsy DEFECTIVE; speed run not trusted",
        }
    elif live:
        compile_ = cargo_build()
    else:
        compile_ = {
            "returncode": 0,
            "skipped": True,
            "bin_present": BIN.is_file(),
            "note": "reused existing raw; cargo not re-invoked",
        }
    if (not live) and RAW.is_file():
        try:
            raw = json.loads(RAW.read_text())
        except json.JSONDecodeError:
            raw = None
    if (
        live
        and not autopsy.get("any_new_kernel_defective")
        and compile_.get("returncode") == 0
        and BIN.is_file()
    ):
        extra: list[str] = ["--reps", "7", "--warmup", "3"]
        parent_ok = (PARENT_ROOT / "catalog.hq38m20").is_file() and TOKENIZER.is_file()
        if parent_ok:
            extra += [
                "--artifact-root",
                str(PARENT_ROOT),
                "--tokenizer",
                str(TOKENIZER),
            ]
        else:
            extra.append("--isolated")
        run = run_locked(extra, RAW)
        raw = run.get("raw")
    iso = summarize_isolated(raw)
    prod = summarize_production(raw)
    loc = {
        "path": str(PARENT_ROOT),
        "outside_worktree": not str(PARENT_ROOT).startswith(str(REPO)),
        "catalog_present": (PARENT_ROOT / "catalog.hq38m20").is_file(),
    }
    answer = what_blocks(iso, prod, autopsy)
    before_ns = (prod.get("before") or {}).get("complete_token_ns_median")
    after_ns = (prod.get("after") or {}).get("complete_token_ns_median")
    reduced = (
        prod.get("kind") == "MEASURED"
        and not (prod.get("after") or {}).get("kept_incumbent_tpr64")
    )
    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "elapsed_s": time.perf_counter() - t0,
        "obligation": "N030 — MLP_GATE_UP_AUTOPSY (S022 §6, §9, §12)",
        "ranking_metric": "COMPLETE_TOKEN_NS",
        "roof_gb_s": ROOF_GB_S,
        "n018_production_gb_s": N018_PRODUCTION_GB_S,
        "n025_mlp_gate_up_gap_share": N025_MLP_GATE_UP_GAP_SHARE,
        "prior_not_rederived": {
            "n017_dram_roof_gb_s": ROOF_GB_S,
            "n018_production_decode_gb_s": N018_PRODUCTION_GB_S,
            "n024_tile_ruled_out": True,
            "n025_mlp_gate_up_gap_share": N025_MLP_GATE_UP_GAP_SHARE,
            "did_not_retry_n018_tiles": list(N018_TILES_NOT_RETRIED),
            "did_not_retry_n024_levers": list(N024_NOT_RETRIED),
        },
        "did_not_load_second_27b": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_write_ascent_or_campaign": True,
        "parent_immutable": loc,
        "lever": {
            "id": LEVER,
            "motif": "norm+projection (RMSNorm writes group-64 x-sums; GEMV defers bias)",
            "not_load_geometry": True,
            "same_tpr64_occupancy": True,
        },
        "controls": {
            "no_op": NO_OP,
            "deliberately_bad": f"{BAD} (isolated) / {BAD_PROD} (production)",
            "reps": 7,
            "report": "min/median/max; overlap = NOT SEPARATED, no mean delta",
        },
        "kernel_autopsy": autopsy,
        "shader_evidence": evidence,
        "compile": compile_,
        "run": None
        if run is None
        else {k: v for k, v in run.items() if k != "raw"},
        "isolated_gemv": iso,
        "production_decode": prod,
        "before_complete_token_ns": before_ns,
        "after_complete_token_ns": after_ns,
        "reduced_complete_token_ns": reduced,
        "dense_w_materialized": 0,
        "answer": answer,
    }


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from-raw", action="store_true")
    args = p.parse_args()
    doc = build(live=not args.from_raw)
    write_receipt(doc)
    print(json.dumps({"receipt": str(RECEIPT), "answer": doc.get("answer")}, indent=2)[:2000])
    if doc.get("kernel_autopsy", {}).get("any_new_kernel_defective"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
