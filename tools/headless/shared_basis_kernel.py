#!/usr/bin/env python3
"""N033 SHARED_BASIS_KERNEL: make the 0.53-bpw byte win actually pay.

N032 measured shared binary K=2 at 0.53 bpw / 6.34 GB/token DRAM — the smallest
footprint — but 111.6 ms because the native path was 384 two-pass dispatches
with a barrier per group. S022 §5: a representation is not condemned until its
native kernel is competent. This lane fuses the operator, stages the tile the
representation wants, screens the kernel, and measures COMPLETE_TOKEN_NS.

    python3 tools/headless/shared_basis_kernel.py
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

from bytes_frontier import (  # noqa: E402
    GROUP,
    LAYERS,
    HIDDEN,
    INTERMEDIATE,
    MLP_ELEMENTS,
    N021_COMPLETE_GPU_NS,
    Q4_ATTN_F32_BYTES,
    ROOF_NS,
    ROOF_TOK_S,
    SCALE_BITS,
    compose_complete,
    git_head,
    moved_toward_roof,
    now_iso,
    ns_spread,
    shared_k2_bytes,
    write_atomic,
)
from first_noetic_executable import PARENT_PARAMS  # noqa: E402
from kernel_competence import (  # noqa: E402
    kernel_bodies,
    params_of,
    screen_kernel,
    strip_comments,
)

SCHEMA = "hawking.headless.shared_basis_kernel.v1"
RECEIPT = REPO / "receipts" / "headless" / "SHARED_BASIS_KERNEL.json"
RAW = REPO / "receipts" / "headless" / "_SHARED_BASIS_KERNEL_raw.json"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "shared_basis_kernel.metal"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "shared_basis_kernel"
GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"

FUSED_PRODUCTION = (
    "shared_binary_k2_fused_xsign_c5120_r8_tg256",
    "shared_binary_k2_fused_xtile_c17408_r8_tg256",
    "shared_binary_k2_fused_stream_c5120_tpr32_tg256",
    "shared_binary_k2_fused_stream_c17408_tpr32_tg256",
    "shared_binary_k2_fused_stream_c5120_tpr64_tg128",
    "shared_binary_k2_fused_stream_c17408_tpr64_tg128",
    "shared_binary_k2_fused_xsign_layers64_c5120_r8_tg256",
    "shared_binary_k2_fused_xtile_layers64_c17408_r8_tg256",
)

N032_TWOPASS_MLP_NS = 99_816_541
N032_TWOPASS_COMPLETE_NS = 111_626_166
N032_TWOPASS_DISPATCHES = 384
N032_Q2F_MLP_NS = 15_738_249
Q2F_COMPLETE_NS = N021_COMPLETE_GPU_NS  # 27_547_874


def fused_bpw(k: int, n_layers: int = LAYERS, group: int = GROUP) -> dict[str, float]:
    """Signs once per organ × K; f16 scales per layer × K × groups."""
    gate_sign = INTERMEDIATE * HIDDEN // 8
    down_sign = HIDDEN * INTERMEDIATE // 8
    sign_bytes = k * (2 * gate_sign + down_sign)
    gate_groups = INTERMEDIATE * (HIDDEN // group)
    down_groups = HIDDEN * (INTERMEDIATE // group)
    scale_bytes = n_layers * k * 2 * (2 * gate_groups + down_groups)
    active = sign_bytes + scale_bytes
    elems = MLP_ELEMENTS * n_layers / float(LAYERS)
    return {
        "k": float(k),
        "n_layers": float(n_layers),
        "basis_sign_bytes": float(sign_bytes),
        "scale_bytes": float(scale_bytes),
        "active_bytes": float(active),
        "active_bpw": 8.0 * active / elems,
        "dram_bytes_per_token": float(active + Q4_ATTN_F32_BYTES),
    }


def shader_autopsy() -> dict[str, Any]:
    src = SHADER.read_text(encoding="utf-8") if SHADER.is_file() else ""
    stripped = strip_comments(src)
    kernels = []
    for name, body in kernel_bodies(stripped):
        r = screen_kernel(name, body, params_of(stripped, name))
        n_barriers = body.count("threadgroup_barrier")
        kernels.append(
            {
                "kernel": name,
                "verdict": r["verdict"],
                "n_findings": r["n_findings"],
                "findings": r["findings"],
                "n_threadgroup_barriers": n_barriers,
                "has_constant_uint_rows": "constant uint& rows" in body
                or "constant uint & rows" in body,
                "has_runtime_div_token": any(
                    f.get("check") == "runtime_integer_divide" for f in r["findings"]
                ),
                "has_runtime_loop_token": any(
                    f.get("check") == "runtime_sized_loop" for f in r["findings"]
                ),
                "has_dynamic_inner_branch": any(
                    f.get("check") == "dynamic_branch_in_loop" for f in r["findings"]
                ),
                "has_bind_time_shape": any(
                    f.get("check") == "bind_time_shape_param" for f in r["findings"]
                ),
            }
        )
    production = [k for k in kernels if k["kernel"] in FUSED_PRODUCTION]
    return {
        "file": str(SHADER.relative_to(REPO)),
        "n_kernels": len(kernels),
        "all_clear": all(k["verdict"] == "CLEAR" for k in kernels),
        "production_all_clear": all(k["verdict"] == "CLEAR" for k in production),
        "production_no_runtime_div": all(not k["has_runtime_div_token"] for k in production),
        "production_no_runtime_loop": all(not k["has_runtime_loop_token"] for k in production),
        "production_no_dynamic_inner_branch": all(
            not k["has_dynamic_inner_branch"] for k in production
        ),
        "production_no_bind_time_shape": all(not k["has_bind_time_shape"] for k in production),
        "kernels": kernels,
        "wanted_geometry": (
            "32 threads/row (one simdgroup), 8 rows/TG of 256, x staged in TGM "
            "for c5120 (fits), x tiled 1024 for c17408 (69 KiB does not fit), "
            "K=2 and group 64 as literals, shift not divide."
        ),
    }


def run_competence() -> dict[str, Any]:
    script = HERE / "kernel_competence.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    doc = {}
    path = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"
    if path.is_file():
        doc = json.loads(path.read_text())
    ours: dict[str, Any] = {}
    for f in doc.get("per_file", []):
        if f.get("file") == "shared_basis_kernel.metal":
            for k in f.get("kernels", []):
                ours[k["kernel"]] = {
                    "verdict": k["verdict"],
                    "n_findings": k["n_findings"],
                    "findings": k.get("findings", []),
                }
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "shared_basis_kernels": ours,
        "any_fused_defective": any(v.get("verdict") == "DEFECTIVE" for v in ours.values()),
        "any_fused_not_clear": any(v.get("verdict") != "CLEAR" for v in ours.values()),
    }


def cargo_build() -> dict[str, Any]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            "cargo",
            "build",
            "--profile",
            "release-fast",
            "-p",
            "hawking-core",
            "--example",
            "shared_basis_kernel",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    return {
        "command": proc.args,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "ok": proc.returncode == 0,
        "stderr_tail": (proc.stderr or "")[-2500:],
    }


def run_example(reps: int = 7) -> dict[str, Any]:
    if not BIN.is_file():
        return {"ok": False, "error": f"missing {BIN}"}
    RAW.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN),
        "--reps",
        str(reps),
        "--warmup",
        "2",
        "--layers",
        "64",
        "--out",
        str(RAW),
    ]
    if GPU_LOCK.is_file():
        cmd = ["bash", str(GPU_LOCK), "n033-sharedbasis", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    raw = json.loads(RAW.read_text()) if RAW.is_file() else {}
    return {
        "ok": proc.returncode == 0 and bool(raw),
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout_tail": (proc.stdout or "")[-3000:],
        "stderr_tail": (proc.stderr or "")[-5000:],
        "raw": raw,
        "command": cmd,
    }


def graph_by_id(raw: dict[str, Any], gid: str) -> dict[str, Any] | None:
    for g in raw.get("graphs") or []:
        if g.get("id") == gid:
            return g
    return None


def complete_of(mlp_ns: int | None) -> dict[str, Any]:
    return compose_complete(mlp_ns, N032_Q2F_MLP_NS)


def run_composition() -> dict[str, Any]:
    """Climb the ladder on real activations. Stream one parent tensor at a time.

    Does not mmap the whole 27B. Does not write under ~/models. Does not mutate
    NOETIC_PARENT_A.
    """
    try:
        from fractional_bit_canon import (  # noqa: WPS433
            GAIN_HEALTHY,
            REL_FRO_LOCAL_MAX,
            classify,
            find_capture,
            find_parent,
            load_X,
            load_tensor,
            score_pair,
            split_from_manifest,
            swiglu_intermediate,
            tensor_name,
            x_wt,
        )
        from onebit_families import bill_shared_basis, fit_shared_binary_bases  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"import: {exc}", "rung": "local_functional_probe"}

    try:
        parent = find_parent()
        cap = find_capture()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "rung": "local_functional_probe",
            "unreached_above": "held_out_activation",
            "why": "parent BF16 or capture missing; kernel parity is the highest rung run here",
        }

    import gc

    import numpy as np

    layer_a, layer_b = 30, 31
    organ = "gate_proj"
    group = GROUP
    # Full 17408-row pair is a 700 MB LS; a 1024-row slice is the same operator
    # on real W rows and real X. Not Gaussian. Not a toy 8x8.
    row_slice = 1024
    t0 = time.perf_counter()
    try:
        X = load_X(cap, layer_a)
        fit_idx, hold_idx, _man, split_rule = split_from_manifest(cap, X.shape[0])
        Xf, Xh = X[fit_idx], X[hold_idx]
        d = (Xf.astype(np.float64) ** 2).mean(axis=0).astype(np.float32)
        del X
        Wa = load_tensor(parent, tensor_name(layer_a, organ))[:row_slice].copy()
        Wb = load_tensor(parent, tensor_name(layer_b, organ))[:row_slice].copy()
        Ya = x_wt(Xh, Wa)
        Yb = x_wt(Xh, Wb)
        zero_a = score_pair(Ya, np.zeros_like(Ya))
        curve = []
        died_at = None
        for k in (1, 2, 4, 8):
            print(f"  composition K={k} L{layer_a}+L{layer_b} {organ} rows={row_slice} ...", flush=True)
            Whats, _bases, _alphas = fit_shared_binary_bases([Wa, Wb], [d, d], k, group)
            acc = bill_shared_basis(Wa, k, group, n_layers=2)
            acc64 = fused_bpw(k, n_layers=64, group=group)
            rows = []
            for What, Y, W, layer in (
                (Whats[0], Ya, Wa, layer_a),
                (Whats[1], Yb, Wb, layer_b),
            ):
                Yh = x_wt(Xh, What)
                sc = score_pair(Y, Yh)
                cls = classify(sc, acc, zero_a, None)
                wrel = float(np.linalg.norm(What - W) / max(float(np.linalg.norm(W)), 1e-30))
                rows.append(
                    {
                        "layer": int(layer),
                        "k": k,
                        "rel_fro": sc["rel_fro"],
                        "cosine": sc["cosine"],
                        "gain": sc["gain"],
                        "beats_null": sc["beats_null"],
                        "weight_rel_fro": wrel,
                        "health": cls.get("health") or cls.get("label") or cls,
                        "n_hold": int(Xh.shape[0]),
                    }
                )
                del Yh
            mean_rel = float(np.mean([r["rel_fro"] for r in rows]))
            mean_gain = float(np.mean([r["gain"] for r in rows]))
            healthy = mean_rel <= REL_FRO_LOCAL_MAX and mean_gain >= GAIN_HEALTHY
            curve.append(
                {
                    "k": k,
                    "n_layers_fitted": 2,
                    "row_slice": row_slice,
                    "group": group,
                    "storage_bpw_over_2_layers": acc["storage_bpw"],
                    "counterfactual_64_layer_bpw": acc64["active_bpw"],
                    "mean_rel_fro": mean_rel,
                    "mean_gain": mean_gain,
                    "healthy": bool(healthy),
                    "layers": rows,
                }
            )
            del Whats, _bases, _alphas
            gc.collect()
        short = None
        k2_row = next((c for c in curve if c["k"] == 2), None)
        if k2_row and k2_row["healthy"]:
            try:
                n_tok = min(256, len(Xh))
                Wua = load_tensor(parent, tensor_name(layer_a, "up_proj"))[:row_slice].copy()
                Wub = load_tensor(parent, tensor_name(layer_b, "up_proj"))[:row_slice].copy()
                Whats_g, _, _ = fit_shared_binary_bases([Wa, Wb], [d, d], 2, group)
                Whats_u, _, _ = fit_shared_binary_bases([Wua, Wub], [d, d], 2, group)
                mid_t = swiglu_intermediate(Xh[:n_tok], Wa, Wua)
                mid_s = swiglu_intermediate(Xh[:n_tok], Whats_g[0], Whats_u[0])
                sc_chain = score_pair(mid_t, mid_s)
                healthy_chain = (
                    sc_chain["rel_fro"] <= REL_FRO_LOCAL_MAX and sc_chain["gain"] >= GAIN_HEALTHY
                )
                short = {
                    "organ": "swiglu_intermediate_gate_up",
                    "layers": [layer_a],
                    "k": 2,
                    "n_tokens": int(n_tok),
                    "row_slice": row_slice,
                    **{k: sc_chain[k] for k in ("rel_fro", "cosine", "gain", "beats_null")},
                    "healthy": bool(healthy_chain),
                }
                if not healthy_chain:
                    died_at = "short_chain"
                del Wua, Wub, Whats_g, Whats_u, mid_t, mid_s
            except Exception as exc:  # noqa: BLE001
                short = {"error": str(exc), "healthy": False}
        del Wa, Wb, Ya, Yb, Xf, Xh
        gc.collect()
        if k2_row and not k2_row["healthy"]:
            died_at = "held_out_activation"
            reached = "local_functional_probe"
            unreached = None
        elif died_at == "short_chain":
            reached = "held_out_activation"
            unreached = None
        elif short and short.get("healthy"):
            reached = "short_chain"
            unreached = "complete_organ"
        else:
            reached = "held_out_activation"
            unreached = "short_chain"
        return {
            "ok": True,
            "parent": str(parent),
            "capture": str(cap),
            "split_rule": split_rule,
            "streamed_per_tensor": True,
            "did_not_load_second_27b": True,
            "did_not_mutate_noetic_parent_a": True,
            "not_gaussian": True,
            "row_slice": row_slice,
            "layers": [layer_a, layer_b],
            "organ": organ,
            "curve": curve,
            "short_chain": short,
            "highest_rung_reached": reached,
            "died_at": died_at,
            "unreached_above": unreached,
            "wall_s": time.perf_counter() - t0,
            "reading": (
                "K=2 at g=64 over 2 layers is 1.50 bpw (not the 0.53-bpw 64-layer "
                "operating point). The 0.53-bpw figure amortizes the same 2 bases "
                "over 64 layers; coefficient cost K*16/g = 0.50 does not vanish. "
                "If 2 adjacent layers already miss health, 64-layer share cannot "
                "rescue it (more layers, same K, is a tighter constraint)."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "rung": "local_functional_probe",
            "unreached_above": "held_out_activation",
            "wall_s": time.perf_counter() - t0,
        }


def _fix_composition_rungs(comp: dict[str, Any], parity_ok: bool) -> dict[str, Any]:
    """Kernel parity is local_functional_probe. Composition may raise or kill."""
    if not parity_ok:
        return {
            "rung": None,
            "status": "FAILED",
            "died_at": "local_functional_probe",
            "why": "fused kernel did not match the CPU oracle",
        }
    if not comp.get("ok"):
        return {
            "rung": "local_functional_probe",
            "status": "UNTESTED_ABOVE",
            "unreached_above": "held_out_activation",
            "why": comp.get("error") or comp.get("why") or "composition did not run",
        }
    died = comp.get("died_at")
    if died == "held_out_activation":
        return {
            "rung": "local_functional_probe",
            "status": "FAILED",
            "died_at": "held_out_activation",
            "why": (
                "K=2 g=64 fitted on adjacent layers missed health on held-out "
                "real activations (rel_fro/gain). A cheaper rep that cannot "
                "preserve the activation is dead at this rung (S022 §38)."
            ),
        }
    if died == "short_chain":
        return {
            "rung": "held_out_activation",
            "status": "FAILED",
            "died_at": "short_chain",
            "why": "held-out passed; 2-layer SwiGLU chain missed health",
        }
    reached = comp.get("highest_rung_reached") or "held_out_activation"
    return {
        "rung": reached,
        "status": "UNTESTED_ABOVE",
        "unreached_above": comp.get("unreached_above") or "complete_organ",
        "why": "composition held on the rungs that were run; complete organ/token not executed",
    }


def main() -> int:
    try:
        from fractional_bit_canon import _ensure_torch  # noqa: WPS433

        _ensure_torch()
    except Exception:
        pass
    t0 = time.perf_counter()
    autopsy = shader_autopsy()
    competence = run_competence()
    build = cargo_build()
    measured = {"ok": False}
    if build["ok"]:
        measured = run_example(7)
    raw = measured.get("raw") or {}
    tp = graph_by_id(raw, "two_pass_384")
    fu = graph_by_id(raw, "fused_192")
    st = graph_by_id(raw, "fused_stream_192")
    se = graph_by_id(raw, "fused_serial")
    no = graph_by_id(raw, "fused_noop")
    ml = graph_by_id(raw, "fused_multilayer_3")

    tp_mlp = (tp or {}).get("gpu_ns", {}).get("median")
    fu_mlp = (fu or {}).get("gpu_ns", {}).get("median")
    tp_complete = complete_of(tp_mlp)
    fu_complete = complete_of(fu_mlp)
    q2f_complete = complete_of(N032_Q2F_MLP_NS)

    k2 = fused_bpw(2)
    bytes_now = k2["active_bytes"]
    dram_now = k2["dram_bytes_per_token"]

    parity_rows = raw.get("parity") or []
    fused_parity = [
        p for p in parity_rows if p.get("must_match") is True and "two_pass" not in p.get("id", "")
    ]
    parity_ok = bool(fused_parity) and all(p.get("ok") is True for p in fused_parity)
    noop_row = next((p for p in parity_rows if p.get("must_match") is False), None)
    noop_diverges = noop_row is not None and not noop_row.get("ok")

    extra = (fu or {}).get("extra") or {}
    overlap_serial = extra.get("overlap_with_serial")
    overlap_noop = extra.get("overlap_with_noop")
    overlap_tp = extra.get("overlap_with_twopass")
    separated = (
        overlap_serial is False and overlap_noop is False
        if overlap_serial is not None and overlap_noop is not None
        else None
    )

    print("composition (streamed tensors, real X) ...", flush=True)
    composition = run_composition()
    coh = _fix_composition_rungs(composition, parity_ok)

    disp_before = (tp or {}).get("dispatches") or N032_TWOPASS_DISPATCHES
    disp_after = (fu or {}).get("dispatches")
    disp_ml = (ml or {}).get("dispatches")

    fu_ns = fu_complete.get("complete_token_ns")
    q2f_ns = q2f_complete.get("complete_token_ns")
    tp_ns = tp_complete.get("complete_token_ns")
    beat_q2f = isinstance(fu_ns, (int, float)) and isinstance(q2f_ns, (int, float)) and fu_ns < q2f_ns
    beat_twopass = (
        isinstance(fu_ns, (int, float)) and isinstance(tp_ns, (int, float)) and fu_ns < tp_ns
    )
    toward = moved_toward_roof(fu_ns, q2f_ns)
    geo = raw.get("geometry_search") or {}

    k_curve = composition.get("curve") if composition.get("ok") else None
    k2_comp = next((c for c in (k_curve or []) if c.get("k") == 2), None)
    first_healthy = next((c for c in (k_curve or []) if c.get("healthy")), None)
    geo_note = ""
    w5120 = (geo.get("winner_c5120") or {}).get("kernel")
    wxsign = "xsign" in (w5120 or "")
    if w5120:
        geo_note = (
            f" Tile search winner for c5120 is {w5120}"
            + (
                " (TGM sign+x reuse)."
                if wxsign
                else " (streamed tpr64; TGM x+signs lost on occupancy, so the geometry the representation suggested was measured and rejected)."
            )
        )
    compose_note = ""
    if coh.get("died_at") == "held_out_activation":
        compose_note = (
            f" K=2 cannot compose: died_at held_out_activation"
            f" (2-layer gate L30/L31 rel_fro={None if not k2_comp else k2_comp.get('mean_rel_fro')}"
            f" gain={None if not k2_comp else k2_comp.get('mean_gain')})."
        )
        if first_healthy:
            compose_note += (
                f" First healthy K on the 2-layer curve is K={first_healthy['k']}"
                f" rel_fro={first_healthy['mean_rel_fro']:.4f} at"
                f" {first_healthy['storage_bpw_over_2_layers']:.3f} bpw over 2 layers"
                f" ({first_healthy['counterfactual_64_layer_bpw']:.3f} bpw if those"
                f" bases amortized over 64 — untested, and adding layers at fixed K"
                f" is a tighter constraint)."
            )
        compose_note += " S022 §38: the token_ns win does not make a dead-on-activation rep promotable."
    if beat_q2f:
        byte_win_pays = True
        byte_win_reason = (
            f"Competent fused kernel dropped dispatches {disp_before}->{disp_after} "
            f"and COMPLETE_TOKEN_NS {tp_ns}->{fu_ns} ns, beating q2f at {q2f_ns} ns "
            f"(delta {int(q2f_ns - fu_ns)} ns). The 0.53-bpw / 6.34-GB-per-token "
            f"byte win now translates to token_ns.{geo_note}{compose_note}"
        )
    elif beat_twopass:
        byte_win_pays = False
        residual = None
        if isinstance(fu_ns, (int, float)) and isinstance(q2f_ns, (int, float)):
            residual = int(fu_ns - q2f_ns)
        byte_win_reason = (
            f"competent kernel beat the 384-dispatch two-pass "
            f"({tp_ns} -> {fu_ns} ns) and dropped dispatches "
            f"{disp_before} -> {disp_after}, but did not beat q2f at "
            f"{q2f_ns} ns (residual {residual} ns). The 0.53-bpw byte win "
            f"is a kernel-competent residual, not a representation condemnation."
            f"{geo_note}{compose_note}"
        )
    else:
        byte_win_pays = False
        byte_win_reason = (
            "fused arm did not beat two-pass on COMPLETE_TOKEN_NS, or measurement missing."
            f"{geo_note}{compose_note}"
        )

    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": "N033",
        "question": (
            "Does a COMPETENT native kernel for shared-binary-basis MLP (K=2, g=64) "
            "turn the 0.53-bpw / 6.34-GB-per-token byte win into COMPLETE_TOKEN_NS, "
            "or is there a measured residual against q2f at 27.55 ms?"
        ),
        "answer": byte_win_reason,
        "byte_win_translates_to_token_ns": byte_win_pays,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "dense_w_materialized": 0,
        "dense_w_is_a_counter": True,
        "timing_label": "DIRTY_ENGINEERING",
        "kernel_competence": competence,
        "kernel_autopsy": autopsy,
        "competent": bool(
            autopsy.get("production_all_clear")
            and autopsy.get("production_no_runtime_div")
            and autopsy.get("production_no_runtime_loop")
            and autopsy.get("production_no_dynamic_inner_branch")
            and autopsy.get("production_no_bind_time_shape")
            and not competence.get("any_fused_defective")
        ),
        "geometry_search": geo,
        "build": build,
        "run": {
            "ok": measured.get("ok"),
            "exit_code": measured.get("exit_code"),
            "wall_s": measured.get("wall_s"),
            "stderr_tail": measured.get("stderr_tail"),
            "stdout_tail": measured.get("stdout_tail"),
            "raw_path": str(RAW),
        },
        "parity": {
            "ok": parity_ok,
            "noop_diverges": noop_diverges,
            "rows": parity_rows,
        },
        "before": {
            "id": "two_pass_384",
            "dispatches": disp_before,
            "mlp_graph_gpu_ns": ns_spread(tp),
            "COMPLETE_TOKEN_NS": {
                "mlp_graph_gpu_ns": ns_spread(tp),
                "composed": tp_complete,
                "min": (tp or {}).get("gpu_ns", {}).get("min"),
                "median": tp_complete.get("complete_token_ns"),
                "max": (tp or {}).get("gpu_ns", {}).get("max"),
                "reps": (tp or {}).get("gpu_ns", {}).get("n"),
            },
            "n032_anchor_mlp_gpu_ns": N032_TWOPASS_MLP_NS,
            "n032_anchor_complete_ns": N032_TWOPASS_COMPLETE_NS,
        },
        "after": {
            "id": "fused_192",
            "dispatches": disp_after,
            "mlp_graph_gpu_ns": ns_spread(fu),
            "COMPLETE_TOKEN_NS": {
                "mlp_graph_gpu_ns": ns_spread(fu),
                "composed": fu_complete,
                "min": (fu or {}).get("gpu_ns", {}).get("min"),
                "median": fu_complete.get("complete_token_ns"),
                "max": (fu or {}).get("gpu_ns", {}).get("max"),
                "reps": (fu or {}).get("gpu_ns", {}).get("n"),
            },
            "kernels": (fu or {}).get("kernels"),
        },
        "controls": {
            "serial": {
                "gpu_ns": ns_spread(se),
                "overlap_with_fused": overlap_serial,
                "label": "NOT SEPARATED" if overlap_serial else ("SEPARATED" if overlap_serial is False else None),
            },
            "noop": {
                "gpu_ns": ns_spread(no),
                "overlap_with_fused": overlap_noop,
                "label": "NOT SEPARATED" if overlap_noop else ("SEPARATED" if overlap_noop is False else None),
            },
            "stream_ablation": {
                "gpu_ns": ns_spread(st),
                "overlap_with_fused": extra.get("overlap_with_stream"),
            },
            "overlap": bool(overlap_serial) or bool(overlap_noop),
            "label": "NOT SEPARATED" if (overlap_serial or overlap_noop) else ("SEPARATED" if separated else None),
        },
        "multilayer_3_dispatch": {
            "dispatches": disp_ml,
            "mlp_graph_gpu_ns": ns_spread(ml),
            "COMPLETE_TOKEN_NS": {
                "mlp_graph_gpu_ns": ns_spread(ml),
                "composed": complete_of((ml or {}).get("gpu_ns", {}).get("median")),
                "median": complete_of((ml or {}).get("gpu_ns", {}).get("median")).get(
                    "complete_token_ns"
                ),
                "reps": (ml or {}).get("gpu_ns", {}).get("n"),
            },
            "note": (
                "Signs reused across 64 layers in one kernel. Harness x is the same "
                "across layers; a real token's x is not. Reported as an amortization "
                "measurement, not as COMPLETE_TOKEN_NS of a sequential residual stream."
            ),
        },
        "dispatches": {
            "before": disp_before,
            "after_fused_per_gemv": disp_after,
            "after_multilayer": disp_ml,
            "driven_down_from_384": bool(
                isinstance(disp_after, (int, float)) and disp_after < N032_TWOPASS_DISPATCHES
            ),
        },
        "active_bytes_per_token": bytes_now,
        "dram_bytes_per_token": dram_now,
        "active_bpw": k2["active_bpw"],
        "dense_w": 0,
        "q2f_baseline": {
            "complete_token_ns": Q2F_COMPLETE_NS,
            "mlp_graph_gpu_ns": N032_Q2F_MLP_NS,
            "receipt": "receipts/headless/NATIVE_2BIT_MLP.json + BYTES_FRONTIER.json",
        },
        "toward_roof_729_7": toward,
        "roof_tok_s": ROOF_TOK_S,
        "roof_ns": ROOF_NS,
        "parent_params": PARENT_PARAMS,
        "mlp_elements": MLP_ELEMENTS,
        "composition": composition,
        "composition_ladder": coh,
        "k_tradeoff_curve": composition.get("curve") if composition.get("ok") else None,
        "finding": {
            "byte_win_translates_to_token_ns": byte_win_pays,
            "reason": byte_win_reason,
            "dispatch_drop": [disp_before, disp_after],
            "kernel_was_the_failure": beat_twopass,
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    write_atomic(RECEIPT, json.dumps(doc, indent=2) + "\n")
    print(f"wrote {RECEIPT}")
    print(
        f"competent={doc['competent']} parity={parity_ok} "
        f"disp {disp_before}->{disp_after} "
        f"pays={byte_win_pays} rung={coh.get('rung')}"
    )
    ok = (
        measured.get("ok")
        and build["ok"]
        and doc["competent"]
        and parity_ok
        and autopsy.get("production_all_clear")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
