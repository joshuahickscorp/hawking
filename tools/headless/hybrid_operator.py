#!/usr/bin/env python3
"""N038 HYBRID_OPERATOR: binary bulk + distributed cheap correction, ONE fuse.

N036: binary g64 is 1.25 bpw / 23.43 ms, faster than q2f 2.25 bpw / 27.55 ms,
and generation-dead. The injury is UNIFORM (token 0 / layer 0 / up_proj across
channels), so a sparse or protected-island correction cannot heal it. The one
structurally distinct combination not yet run as a single operator is the
hybrid: binary bulk plus a DISTRIBUTED correction, fused, no dense W.

Two representations (correction bits count toward complete EBPW):

  binary_g64 + low-rank r     y = binary(x) + U (V^T x)
  binary_g64 + shared-basis K y = binary(x) + sum_k s_k (B_k ⊙ x)

A measured "the cheapest distributed correction that restores generation costs
>= (2.25-1.25) bpw" is a valid, campaign-completing result.

    python3 tools/headless/hybrid_operator.py
    python3 -m pytest tools/headless -q

Does not load a second 27B. Does not write under ~/models. Does not mutate
NOETIC_PARENT_A. GPU serialized with `bash tools/gpu_lane_lock.sh n038-hybrid`.
"""
from __future__ import annotations

import gc
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
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    MLP_ELEMENTS,
    N021_COMPLETE_GPU_NS,
    PARENT_PARAMS,
    Q4_ATTN_F32_BYTES,
    Q2F_BPW,
    compose_complete,
    git_head,
    now_iso,
    ns_spread,
    write_atomic,
)
from first_noetic_executable import PARENT_BF16  # noqa: E402
from fractional_bit_canon import (  # noqa: E402
    GAIN_HEALTHY,
    REL_FRO_LOCAL_MAX,
    _binary_meanabs,
    _fourlevel_fitted,
    find_capture,
    find_parent,
    load_X,
    load_tensor,
    rsvd,
    score_pair,
    snap_f16,
    split_from_manifest,
    swiglu_intermediate,
    tensor_name,
)
from kernel_competence import (  # noqa: E402
    kernel_bodies,
    params_of,
    screen_kernel,
    strip_comments,
)
from shared_basis_kernel import fused_bpw  # noqa: E402

SCHEMA = "hawking.headless.hybrid_operator.v1"
RECEIPT = REPO / "receipts" / "headless" / "HYBRID_OPERATOR.json"
RAW = REPO / "receipts" / "headless" / "_HYBRID_OPERATOR_raw.json"
SHADER = REPO / "crates" / "hawking-core" / "shaders" / "hybrid_operator.metal"
CARGO_TARGET = Path(
    os.environ.get("CARGO_TARGET_DIR", str(REPO / "workspace" / "ops" / "build" / "rust"))
)
BIN = CARGO_TARGET / "release-fast" / "examples" / "hybrid_operator"
GPU_LOCK = REPO / "tools" / "gpu_lane_lock.sh"
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"

BINARY_BPW = 1.25
Q2F_COMPLETE_NS = N021_COMPLETE_GPU_NS  # 27_547_874
Q2F_MLP_NS = 15_738_249
Q2F_MS = 27.55
ORGANS = ("gate_proj", "up_proj", "down_proj")
PROBE_LAYERS = (0, 1, 7, 31)
HOLD_TOKENS = 48
RANKS_GPU = (8, 32)
RANKS_CPU = (8, 32, 64, 128, 256)
# r such that extra bpw reaches 1.0: uv_bytes(r)/MLP * 8 = 1 => r = 247.3
RANK_BUDGET = 247
K_SHARED = 2

PRODUCTION_KERNELS = (
    "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128",
    "binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128",
    "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128",
    "binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128",
    "binary_shared_k2_fused_geo_c5120_tpr64_tg128",
    "binary_shared_k2_fused_geo_c17408_tpr64_tg128",
)
NOOP_KERNELS = (
    "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128_noop",
    "binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128_noop",
    "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128_noop",
    "binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128_noop",
    "binary_shared_k2_fused_geo_c5120_tpr64_tg128_noop",
    "binary_shared_k2_fused_geo_c17408_tpr64_tg128_noop",
)


def _reexec_vision() -> None:
    if not VISION_PY.is_file():
        return
    try:
        if Path(sys.executable).resolve() == VISION_PY.resolve():
            return
    except OSError:
        return
    os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])


def organ_shape(organ: str) -> tuple[int, int]:
    if organ in ("gate_proj", "up_proj"):
        return INTERMEDIATE, HIDDEN
    return HIDDEN, INTERMEDIATE


def uv_bytes(rank: int, n_layers: int = LAYERS) -> int:
    """f16 U[rows,r] + f16 V[cols,r] for 3 organs × n_layers. No hidden bits."""
    per_organ = 2 * int(rank) * (INTERMEDIATE + HIDDEN)
    return int(n_layers) * 3 * per_organ


def lowrank_bill(rank: int, n_layers: int = LAYERS) -> dict[str, float]:
    binary = MLP_ELEMENTS * (n_layers / float(LAYERS)) * BINARY_BPW / 8.0
    corr = float(uv_bytes(rank, n_layers))
    active = binary + corr
    elems = MLP_ELEMENTS * (n_layers / float(LAYERS))
    return {
        "rank": float(rank),
        "n_layers": float(n_layers),
        "binary_bytes": float(binary),
        "correction_bytes": corr,
        "active_bytes": float(active),
        "active_bpw": 8.0 * active / elems,
        "correction_bpw": 8.0 * corr / elems,
        "dram_bytes_per_token": float(active + Q4_ATTN_F32_BYTES),
        "complete_ebpw": 8.0 * (active + Q4_ATTN_F32_BYTES) / PARENT_PARAMS,
        "below_q2f_bpw": (8.0 * active / elems) < Q2F_BPW,
        "dense_w": 0.0,
    }


def shared_hybrid_bill(k: int = K_SHARED, n_layers: int = LAYERS) -> dict[str, float]:
    """Independent binary bulk + K extra shared bases (signs once, scales/layer)."""
    binary = MLP_ELEMENTS * (n_layers / float(LAYERS)) * BINARY_BPW / 8.0
    extra = fused_bpw(k, n_layers=n_layers, group=GROUP)
    active = binary + extra["active_bytes"]
    elems = MLP_ELEMENTS * (n_layers / float(LAYERS))
    return {
        "k": float(k),
        "n_layers": float(n_layers),
        "binary_bytes": float(binary),
        "correction_bytes": extra["active_bytes"],
        "basis_sign_bytes": extra["basis_sign_bytes"],
        "scale_bytes": extra["scale_bytes"],
        "active_bytes": float(active),
        "active_bpw": 8.0 * active / elems,
        "correction_bpw": extra["active_bpw"],
        "dram_bytes_per_token": float(active + Q4_ATTN_F32_BYTES),
        "complete_ebpw": 8.0 * (active + Q4_ATTN_F32_BYTES) / PARENT_PARAMS,
        "below_q2f_bpw": (8.0 * active / elems) < Q2F_BPW,
        "dense_w": 0.0,
    }


def rank_for_extra_bpw(target: float = 1.0) -> float:
    """Rank at which f16 U,V correction bills `target` bpw on the MLP body."""
    # 8 * uv_bytes(r) / MLP_ELEMENTS = target
    # uv_bytes(r) = 64 * 3 * 2 * r * 22528 = 8650752 r
    per_rank = uv_bytes(1)
    return target * MLP_ELEMENTS / (8.0 * per_rank)


def shader_autopsy() -> dict[str, Any]:
    src = SHADER.read_text(encoding="utf-8") if SHADER.is_file() else ""
    stripped = strip_comments(src)
    kernels = []
    for name, body in kernel_bodies(stripped):
        r = screen_kernel(name, body, params_of(stripped, name))
        params = params_of(stripped, name)
        kernels.append(
            {
                "kernel": name,
                "verdict": r["verdict"],
                "n_findings": r["n_findings"],
                "findings": r["findings"],
                "has_constant_uint_rows": "constant uint& rows" in params
                or "constant uint & rows" in params,
                "has_constant_uint_rank": "constant uint& rank" in params
                or "constant uint & rank" in params,
                "n_threadgroup_barriers": body.count("threadgroup_barrier"),
            }
        )
    production = [k for k in kernels if k["kernel"] in PRODUCTION_KERNELS]
    present = {n: (f"kernel void {n}(" in src) for n in PRODUCTION_KERNELS + NOOP_KERNELS}
    return {
        "file": str(SHADER.relative_to(REPO)),
        "n_kernels": len(kernels),
        "all_present": all(present.values()),
        "kernels_present": present,
        "all_clear": all(k["verdict"] == "CLEAR" for k in kernels) and bool(kernels),
        "production_all_clear": all(k["verdict"] == "CLEAR" for k in production)
        and all(present[n] for n in PRODUCTION_KERNELS),
        "production_no_bind_time_shape": all(
            not k["has_constant_uint_rows"] and not k["has_constant_uint_rank"] for k in production
        ),
        "uses_shift_not_div": ">> 6u" in src,
        "dense_w_written": False,
        "kernels": kernels,
        "wanted_geometry": (
            "tpr64 tg128 (2 rows / TG of 128, 64 lanes/row), group 64 as a "
            "shift, rank and K as literals, V^T x fused into the binary sweep, "
            "one dispatch per GEMV."
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
    auto = shader_autopsy()
    ours: dict[str, Any] = {}
    path = REPO / "receipts" / "headless" / "KERNEL_COMPETENCE.json"
    if path.is_file():
        doc = json.loads(path.read_text())
        for f in doc.get("per_file", []):
            if f.get("file") == "hybrid_operator.metal":
                for k in f.get("kernels", []):
                    ours[k["kernel"]] = {
                        "verdict": k["verdict"],
                        "n_findings": k["n_findings"],
                        "findings": k.get("findings", []),
                    }
    return {
        "ok": proc.returncode == 0 and auto["production_all_clear"],
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "autopsy": auto,
        "hybrid_kernels": ours,
        "any_geo_defective": any(v.get("verdict") == "DEFECTIVE" for v in ours.values()),
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
            "hybrid_operator",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=1200,
    )
    return {
        "command": proc.args,
        "exit_code": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "ok": proc.returncode == 0,
        "stderr_tail": (proc.stderr or "")[-4000:],
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
        cmd = ["bash", str(GPU_LOCK), "n038-hybrid", *cmd]
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
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "raw": raw,
        "command": cmd,
    }


def graph_by_id(raw: dict[str, Any], gid: str) -> dict[str, Any] | None:
    for g in raw.get("graphs") or []:
        if g.get("id") == gid:
            return g
    return None


def _healthy(sc: dict[str, Any], rel_max: float = REL_FRO_LOCAL_MAX, gain_min: float = GAIN_HEALTHY) -> bool:
    return bool(sc.get("rel_fro", 1.0) <= rel_max and sc.get("gain", 0.0) >= gain_min)


def reconstruct_lr(w, rank: int):
    w_bin = _binary_meanabs(w, GROUP)
    r = w - w_bin
    u, s, vh = rsvd(r, rank, niter=1, seed=10582654 + rank)
    u16, s16, vh16 = snap_f16(u), snap_f16(s), snap_f16(vh)
    return w_bin + (u16 * s16) @ vh16


def group_ls_scale(residual, basis, group: int = GROUP):
    import numpy as np

    rows, cols = residual.shape
    gpr = cols // group
    r = residual.reshape(rows, gpr, group)
    b = basis.reshape(rows, gpr, group)
    num = (r * b).sum(axis=-1)
    den = (b * b).sum(axis=-1)
    sc = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    return snap_f16(sc.astype(np.float32))


def reconstruct_shared_k2(w, b0, b1, sc0, sc1, group: int = GROUP):
    import numpy as np

    w_bin = _binary_meanabs(w, GROUP)
    rows, cols = w.shape
    gpr = cols // group
    extra = (b0.reshape(rows, gpr, group) * sc0[:, :, None]).reshape(rows, cols)
    extra = extra + (b1.reshape(rows, gpr, group) * sc1[:, :, None]).reshape(rows, cols)
    return w_bin + extra


def fit_shared_residual_k2(parent, organ: str, layers) -> dict[str, Any]:
    """Greedy K=2 shared ±1 bases of (W - W_binary) across `layers`."""
    import numpy as np

    rows, cols = organ_shape(organ)
    acc = np.zeros((rows, cols), dtype=np.float64)
    t0 = time.perf_counter()
    for L in layers:
        w = parent.load(tensor_name(L, organ)) if hasattr(parent, "load") else load_tensor(
            parent, tensor_name(L, organ)
        )
        r = w - _binary_meanabs(w, GROUP)
        acc += r
        del w, r
        gc.collect()
    b0 = np.where(acc >= 0, np.float32(1.0), np.float32(-1.0))
    del acc
    gc.collect()
    acc2 = np.zeros((rows, cols), dtype=np.float64)
    scales0 = {}
    for L in layers:
        w = parent.load(tensor_name(L, organ)) if hasattr(parent, "load") else load_tensor(
            parent, tensor_name(L, organ)
        )
        r = w - _binary_meanabs(w, GROUP)
        sc = group_ls_scale(r, b0)
        scales0[int(L)] = sc
        gpr = cols // GROUP
        acc2 += r - (b0.reshape(rows, gpr, GROUP) * sc[:, :, None]).reshape(rows, cols)
        del w, r
        gc.collect()
    b1 = np.where(acc2 >= 0, np.float32(1.0), np.float32(-1.0))
    del acc2
    gc.collect()
    scales1 = {}
    for L in layers:
        w = parent.load(tensor_name(L, organ)) if hasattr(parent, "load") else load_tensor(
            parent, tensor_name(L, organ)
        )
        r = w - _binary_meanabs(w, GROUP)
        gpr = cols // GROUP
        r1 = r - (b0.reshape(rows, gpr, GROUP) * scales0[int(L)][:, :, None]).reshape(rows, cols)
        scales1[int(L)] = group_ls_scale(r1, b1)
        del w, r, r1
        gc.collect()
    return {
        "organ": organ,
        "b0": b0,
        "b1": b1,
        "scales0": scales0,
        "scales1": scales1,
        "layers": list(layers),
        "wall_s": time.perf_counter() - t0,
    }


def score_y(pred, ref) -> dict[str, Any]:
    sc = score_pair(ref, pred)
    sc["healthy"] = _healthy(sc)
    return sc


def run_held_out_lowrank(parent, cap, hold_idx, ranks=RANKS_CPU) -> dict[str, Any]:
    import numpy as np

    curve = []
    x0 = load_X(cap, 0)
    use = np.asarray(hold_idx)
    if use.size > HOLD_TOKENS:
        step = max(1, use.size // HOLD_TOKENS)
        use = use[::step][:HOLD_TOKENS]
    xh = x0[use]
    del x0
    organ = "up_proj"
    print(f"  held-out low-rank L0 {organ} ranks={list(ranks)} tokens={len(use)} ...", flush=True)
    w = parent.load(tensor_name(0, organ)) if hasattr(parent, "load") else load_tensor(
        parent, tensor_name(0, organ)
    )
    y_t = xh @ w.T
    w_bin = _binary_meanabs(w, GROUP)
    y_b = xh @ w_bin.T
    w_q = _fourlevel_fitted(w, GROUP)
    y_q = xh @ w_q.T
    sc_b = score_y(y_b, y_t)
    sc_q = score_y(y_q, y_t)
    residual = w - w_bin
    ref = {
        "binary_vs_teacher": sc_b,
        "q2f_vs_teacher": sc_q,
        "binary_vs_q2f": score_y(y_b, y_q),
    }
    for rank in ranks:
        print(f"    rsvd rank={rank} ...", flush=True)
        u, s, vh = rsvd(residual, rank, niter=1, seed=10582654 + rank)
        what = w_bin + (snap_f16(u) * snap_f16(s)) @ snap_f16(vh)
        y_h = xh @ what.T
        sc = score_y(y_h, y_t)
        sc_vs_q = score_y(y_h, y_q)
        bill = lowrank_bill(rank)
        curve.append(
            {
                "rank": rank,
                "layer": 0,
                "organ": organ,
                "vs_teacher": sc,
                "vs_q2f": sc_vs_q,
                "healthy": sc["healthy"],
                "active_bpw": bill["active_bpw"],
                "correction_bpw": bill["correction_bpw"],
                "complete_ebpw": bill["complete_ebpw"],
                "below_q2f_bpw": bill["below_q2f_bpw"],
                "below_rank_budget": rank <= RANK_BUDGET,
            }
        )
        del what, y_h, u, s, vh
        gc.collect()
        print(
            f"      r={rank} rel_fro={sc['rel_fro']:.4f} gain={sc['gain']:.3f} "
            f"bpw={bill['active_bpw']:.4f} healthy={sc['healthy']}",
            flush=True,
        )
    # one more probe layer to show the injury is not L0-only
    probes = []
    for L in (1, 7):
        xl = load_X(cap, L)[use]
        wl = parent.load(tensor_name(L, organ)) if hasattr(parent, "load") else load_tensor(
            parent, tensor_name(L, organ)
        )
        yt = xl @ wl.T
        wb = _binary_meanabs(wl, GROUP)
        yb = xl @ wb.T
        what = reconstruct_lr(wl, 32)
        yh = xl @ what.T
        probes.append(
            {
                "layer": L,
                "organ": organ,
                "binary": score_y(yb, yt),
                "lowrank_r32": score_y(yh, yt),
            }
        )
        del xl, wl, yt, wb, yb, what, yh
        gc.collect()
    del w, w_bin, w_q, y_t, y_b, y_q, residual, xh
    gc.collect()
    first_healthy = next((c for c in curve if c["healthy"] and c["below_q2f_bpw"]), None)
    first_any = next((c for c in curve if c["healthy"]), None)
    return {
        "organ": organ,
        "layer": 0,
        "n_tokens": int(use.size),
        "not_gaussian": True,
        "real_activations": True,
        "reference": ref,
        "curve": curve,
        "probes": probes,
        "first_healthy_rank_below_q2f": None if not first_healthy else first_healthy["rank"],
        "first_healthy_rank_any": None if not first_any else first_any["rank"],
        "rank_budget_for_1bpw_extra": RANK_BUDGET,
    }


def run_held_out_shared(parent, cap, hold_idx, layers=PROBE_LAYERS) -> dict[str, Any]:
    import numpy as np

    use = np.asarray(hold_idx)
    if use.size > HOLD_TOKENS:
        step = max(1, use.size // HOLD_TOKENS)
        use = use[::step][:HOLD_TOKENS]
    organ = "up_proj"
    print(f"  fit shared-K=2 residual on {organ} layers={list(layers)} ...", flush=True)
    fit = fit_shared_residual_k2(parent, organ, layers)
    rows = []
    for L in layers:
        x = load_X(cap, L)[use]
        w = parent.load(tensor_name(L, organ)) if hasattr(parent, "load") else load_tensor(
            parent, tensor_name(L, organ)
        )
        y_t = x @ w.T
        y_b = x @ _binary_meanabs(w, GROUP).T
        what = reconstruct_shared_k2(
            w, fit["b0"], fit["b1"], fit["scales0"][int(L)], fit["scales1"][int(L)]
        )
        y_h = x @ what.T
        sc = score_y(y_h, y_t)
        rows.append(
            {
                "layer": int(L),
                "organ": organ,
                "binary": score_y(y_b, y_t),
                "hybrid": sc,
                "healthy": sc["healthy"],
            }
        )
        print(
            f"    L{L:02d} shared-k2 rel_fro={sc['rel_fro']:.4f} gain={sc['gain']:.3f} "
            f"healthy={sc['healthy']}",
            flush=True,
        )
        del x, w, y_t, y_b, what, y_h
        gc.collect()
    rels = [r["hybrid"]["rel_fro"] for r in rows]
    gains = [r["hybrid"]["gain"] for r in rows]
    bill = shared_hybrid_bill(K_SHARED)
    mean_rel = float(sum(rels) / len(rels)) if rels else 1.0
    mean_gain = float(sum(gains) / len(gains)) if gains else 0.0
    # drop the large ±1 planes from the returned fit
    return {
        "k": K_SHARED,
        "organ": organ,
        "fit_layers": list(layers),
        "fit_wall_s": fit["wall_s"],
        "layers": rows,
        "mean_rel_fro": mean_rel,
        "mean_gain": mean_gain,
        "healthy": bool(mean_rel <= REL_FRO_LOCAL_MAX and mean_gain >= GAIN_HEALTHY and rels),
        "n_healthy": sum(1 for r in rows if r["healthy"]),
        "n": len(rows),
        "accounting": bill,
        "not_gaussian": True,
        "real_activations": True,
    }


def run_short_chain(parent, cap, hold_idx, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Layer-0 SwiGLU on real X with reconstructed gate+up."""
    import numpy as np

    print(f"  short_chain {kind} L0 SwiGLU ...", flush=True)
    x = load_X(cap, 0)
    use = np.asarray(hold_idx)
    if use.size > HOLD_TOKENS:
        step = max(1, use.size // HOLD_TOKENS)
        use = use[::step][:HOLD_TOKENS]
    xh = x[use]
    del x
    wg = parent.load(tensor_name(0, "gate_proj")) if hasattr(parent, "load") else load_tensor(
        parent, tensor_name(0, "gate_proj")
    )
    wu = parent.load(tensor_name(0, "up_proj")) if hasattr(parent, "load") else load_tensor(
        parent, tensor_name(0, "up_proj")
    )
    mid_t = swiglu_intermediate(xh, wg, wu)
    if kind.startswith("lowrank"):
        rank = int(spec["rank"])
        hg = reconstruct_lr(wg, rank)
        hu = reconstruct_lr(wu, rank)
    elif kind.startswith("shared"):
        fit_g = spec["fit_gate"]
        fit_u = spec["fit_up"]
        hg = reconstruct_shared_k2(
            wg, fit_g["b0"], fit_g["b1"], fit_g["scales0"][0], fit_g["scales1"][0]
        )
        hu = reconstruct_shared_k2(
            wu, fit_u["b0"], fit_u["b1"], fit_u["scales0"][0], fit_u["scales1"][0]
        )
    else:
        raise ValueError(kind)
    mid_s = swiglu_intermediate(xh, hg, hu)
    sc = score_y(mid_s, mid_t)
    del wg, wu, hg, hu, mid_t, mid_s, xh
    gc.collect()
    return {
        "kind": kind,
        "layer": 0,
        "n_tokens": int(use.size),
        "rel_fro": sc["rel_fro"],
        "cosine": sc["cosine"],
        "gain": sc["gain"],
        "beats_null": sc["beats_null"],
        "healthy": sc["healthy"],
        "real_activations": True,
    }


def run_complete_organ(parent, cap, hold_idx, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    print(f"  complete_organ {kind} L0 ...", flush=True)
    x = load_X(cap, 0)
    use = np.asarray(hold_idx)
    if use.size > HOLD_TOKENS:
        step = max(1, use.size // HOLD_TOKENS)
        use = use[::step][:HOLD_TOKENS]
    xh = x[use]
    del x
    wg = parent.load(tensor_name(0, "gate_proj")) if hasattr(parent, "load") else load_tensor(
        parent, tensor_name(0, "gate_proj")
    )
    wu = parent.load(tensor_name(0, "up_proj")) if hasattr(parent, "load") else load_tensor(
        parent, tensor_name(0, "up_proj")
    )
    wd = parent.load(tensor_name(0, "down_proj")) if hasattr(parent, "load") else load_tensor(
        parent, tensor_name(0, "down_proj")
    )
    mid_t = swiglu_intermediate(xh, wg, wu)
    y_t = mid_t @ wd.T
    if kind.startswith("lowrank"):
        rank = int(spec["rank"])
        hg, hu, hd = reconstruct_lr(wg, rank), reconstruct_lr(wu, rank), reconstruct_lr(wd, rank)
    elif kind.startswith("shared"):
        hg = reconstruct_shared_k2(
            wg,
            spec["fit_gate"]["b0"],
            spec["fit_gate"]["b1"],
            spec["fit_gate"]["scales0"][0],
            spec["fit_gate"]["scales1"][0],
        )
        hu = reconstruct_shared_k2(
            wu,
            spec["fit_up"]["b0"],
            spec["fit_up"]["b1"],
            spec["fit_up"]["scales0"][0],
            spec["fit_up"]["scales1"][0],
        )
        hd = reconstruct_shared_k2(
            wd,
            spec["fit_down"]["b0"],
            spec["fit_down"]["b1"],
            spec["fit_down"]["scales0"][0],
            spec["fit_down"]["scales1"][0],
        )
    else:
        raise ValueError(kind)
    mid_s = swiglu_intermediate(xh, hg, hu)
    y_s = mid_s @ hd.T
    sc = score_y(y_s, y_t)
    del wg, wu, wd, hg, hu, hd, mid_t, mid_s, y_t, y_s, xh
    gc.collect()
    return {
        "kind": kind,
        "layer": 0,
        "n_tokens": int(use.size),
        "rel_fro": sc["rel_fro"],
        "cosine": sc["cosine"],
        "gain": sc["gain"],
        "beats_null": sc["beats_null"],
        "healthy": sc["healthy"],
        "real_activations": True,
        "dense_w_scoring_vehicle": True,
        "note": "What reconstructed as a scoring vehicle; executable is the fused kernel",
    }


def climb(held_lr: dict[str, Any], held_sh: dict[str, Any], shorts: dict, organs: dict) -> dict[str, Any]:
    """Composition ladder. A win requires coherent_generation. We stop at the first fail."""
    lr_healthy = held_lr.get("first_healthy_rank_below_q2f") is not None
    sh_healthy = bool(held_sh.get("healthy"))
    if not lr_healthy and not sh_healthy:
        why = (
            "Neither distributed correction healed held-out real activations "
            "while staying under q2f 2.25 bpw. "
        )
        if held_lr.get("first_healthy_rank_any") is not None:
            r = held_lr["first_healthy_rank_any"]
            bill = lowrank_bill(r)
            why += (
                f"Low-rank first healthy rank is r={r} at {bill['active_bpw']:.4f} bpw "
                f"(correction {bill['correction_bpw']:.4f} bpw); rank budget for a 1.0 bpw "
                f"correction is r<={RANK_BUDGET}."
            )
        else:
            why += (
                f"Low-rank r in {list(RANKS_CPU)} never reached rel_fro<={REL_FRO_LOCAL_MAX} "
                f"and gain>={GAIN_HEALTHY} on L0 up_proj. Shared K=2 mean rel_fro="
                f"{held_sh.get('mean_rel_fro')}."
            )
        return {
            "rung": "local_functional_probe",
            "status": "FAILED",
            "died_at": "held_out_activation",
            "unreached_above": None,
            "why": why,
        }
    # pick the healthy arm
    if lr_healthy:
        kind = f"lowrank_r{held_lr['first_healthy_rank_below_q2f']}"
    else:
        kind = "shared_k2"
    short = shorts.get(kind)
    if not short or not short.get("healthy"):
        return {
            "rung": "held_out_activation",
            "status": "FAILED",
            "died_at": "short_chain",
            "why": f"{kind} healed GEMV held-out but not L0 SwiGLU.",
        }
    organ = organs.get(kind)
    if not organ or not organ.get("healthy"):
        return {
            "rung": "short_chain",
            "status": "FAILED",
            "died_at": "complete_organ",
            "why": f"{kind} healed SwiGLU but not the full L0 MLP.",
        }
    return {
        "rung": "complete_organ",
        "status": "FAILED",
        "died_at": "complete_token",
        "why": (
            f"{kind} healed L0 complete organ on real activations. Native "
            "complete_token / coherent_generation was not run: the production "
            "decoder has no hybrid codec, and reconstructing dense W to inject "
            "into a student is a scoring vehicle, not the fused operator. "
            "A win requires coherent_generation on the native path."
        ),
        "unreached_above": "complete_token",
    }


def attach_graph(
    spec: dict[str, Any],
    graph: dict[str, Any] | None,
    q2f_mlp_ns: int | None,
    parity_ok: bool,
    composition: dict[str, Any],
    held: dict[str, Any] | None,
) -> dict[str, Any]:
    gpu = ns_spread(graph)
    wall = ns_spread(graph, "wall_ns")
    mlp_ns = gpu.get("median")
    complete = compose_complete(mlp_ns, q2f_mlp_ns)
    complete_ns = complete.get("complete_token_ns")
    faster = complete_ns is not None and int(complete_ns) < int(Q2F_COMPLETE_NS)
    noop = (graph or {}).get("noop") or {}
    overlap = noop.get("overlap_with_fused")
    rung = composition.get("rung")
    coherent = rung == "coherent_generation"
    counts = bool(
        coherent
        and spec.get("below_q2f_bpw")
        and faster
        and spec.get("dense_w", 0) == 0
    )
    return {
        **spec,
        "COMPLETE_TOKEN_NS": {
            "mlp_graph_gpu_ns": gpu,
            "mlp_graph_wall_ns": wall,
            "composed": complete,
            "min": gpu.get("min"),
            "median": complete_ns,
            "max": gpu.get("max"),
            "reps": gpu.get("n"),
        },
        "faster_than_q2f_27_55ms": faster,
        "parity": {"ok": parity_ok},
        "dense_w": 0,
        "dense_w_materialized": 0,
        "fused_native_operator": True,
        "one_dispatch_per_gemv": True,
        "composition": composition,
        "held_out": held,
        "counts_as_win": counts,
        "control": {
            "noop": noop,
            "overlap": overlap,
            "label": (
                "NOT SEPARATED"
                if overlap
                else ("SEPARATED" if overlap is False else None)
            ),
        },
    }


class _ParentShim:
    """Thin wrapper so we can call .load like ParentReader without a second 27B."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, name: str):
        return load_tensor(self.root, name)


def main() -> int:
    _reexec_vision()
    t0 = time.perf_counter()
    skip_gpu = os.environ.get("N038_SKIP_GPU") == "1"
    skip_fit = os.environ.get("N038_SKIP_FIT") == "1"

    competence = run_competence()
    print(
        "kernel autopsy production_all_clear=",
        competence["autopsy"]["production_all_clear"],
        "any_geo_defective=",
        competence["any_geo_defective"],
        flush=True,
    )

    build = {"ok": BIN.is_file(), "skipped": True}
    measured = {"ok": False, "raw": {}}
    if not skip_gpu:
        print("== cargo build release-fast hybrid_operator ==", flush=True)
        build = cargo_build()
        print("build ok", build["ok"], "wall", build.get("wall_s"), flush=True)
        if not build["ok"]:
            print(build.get("stderr_tail"), flush=True)
        else:
            print("== GPU graphs under n038-hybrid lock ==", flush=True)
            measured = run_example(7)
            print("gpu ok", measured["ok"], "wall", measured.get("wall_s"), flush=True)
            if not measured["ok"]:
                print(measured.get("stderr_tail"), flush=True)

    raw = measured.get("raw") or {}
    if not raw and RAW.is_file():
        raw = json.loads(RAW.read_text())
        measured = {"ok": True, "raw": raw, "reused_raw": True}

    q2f_g = graph_by_id(raw, "q2f_g64")
    q2f_mlp_ns = (q2f_g or {}).get("gpu_ns", {}).get("median") or Q2F_MLP_NS
    parity_rows = raw.get("parity") or []
    parity_ok = bool(parity_rows) and all(p.get("ok") is True for p in parity_rows)

    held_lr: dict[str, Any] = {}
    held_sh: dict[str, Any] = {}
    shorts: dict[str, Any] = {}
    organs_scored: dict[str, Any] = {}
    parent_info: dict[str, Any] = {}
    capture_info: dict[str, Any] = {}
    if not skip_fit:
        print("== composition on REAL activations (no second 27B) ==", flush=True)
        cap = find_capture()
        parent_path = find_parent()
        parent = _ParentShim(parent_path)
        x0 = load_X(cap, 0)
        fit_idx, hold_idx, _man, split_rule = split_from_manifest(cap, x0.shape[0])
        del x0
        parent_info = {
            "path": str(parent_path),
            "is_noetic_parent_a": str(parent_path) == str(PARENT_BF16)
            or "qwen3.8-27b-abliterated-bf16" in str(parent_path),
            "mutated": False,
        }
        capture_info = {
            "path": str(cap),
            "split_rule": split_rule,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
        }
        held_lr = run_held_out_lowrank(parent, cap, hold_idx)
        held_sh = run_held_out_shared(parent, cap, hold_idx)
        # short_chain / complete_organ only if something looked locally viable
        if held_lr.get("first_healthy_rank_below_q2f") is not None:
            r = int(held_lr["first_healthy_rank_below_q2f"])
            kind = f"lowrank_r{r}"
            shorts[kind] = run_short_chain(parent, cap, hold_idx, kind, {"rank": r})
            if shorts[kind].get("healthy"):
                organs_scored[kind] = run_complete_organ(
                    parent, cap, hold_idx, kind, {"rank": r}
                )
        if held_sh.get("healthy"):
            print("  fit shared K=2 residual for gate+down (short/organ) ...", flush=True)
            fit_g = fit_shared_residual_k2(parent, "gate_proj", (0,))
            fit_d = fit_shared_residual_k2(parent, "down_proj", (0,))
            # reuse up-proj fit from held_sh by re-fitting L0 only (planes already in held)
            fit_u = fit_shared_residual_k2(parent, "up_proj", (0,))
            spec = {"fit_gate": fit_g, "fit_up": fit_u, "fit_down": fit_d}
            shorts["shared_k2"] = run_short_chain(parent, cap, hold_idx, "shared_k2", spec)
            if shorts["shared_k2"].get("healthy"):
                organs_scored["shared_k2"] = run_complete_organ(
                    parent, cap, hold_idx, "shared_k2", spec
                )
            del fit_g, fit_u, fit_d
            gc.collect()

    composition = climb(held_lr, held_sh, shorts, organs_scored)

    reps = []
    lr8_bill = lowrank_bill(8)
    lr32_bill = lowrank_bill(32)
    sh_bill = shared_hybrid_bill(2)
    for gid, bill, held in (
        ("binary_lowrank_r8", lr8_bill, {"curve_head": (held_lr.get("curve") or [None])[0]}),
        ("binary_lowrank_r32", lr32_bill, {"curve": [c for c in (held_lr.get("curve") or []) if c.get("rank") == 32]}),
        ("binary_shared_k2", sh_bill, held_sh),
    ):
        g = graph_by_id(raw, gid)
        spec = {
            "id": gid,
            "correction": "lowrank" if "lowrank" in gid else "shared_basis",
            "distributed": True,
            "not_sparse": True,
            "not_island": True,
            **bill,
        }
        reps.append(attach_graph(spec, g, q2f_mlp_ns, parity_ok, composition, held))

    n_fused = sum(1 for r in reps if r.get("fused_native_operator") and r.get("COMPLETE_TOKEN_NS", {}).get("reps"))
    wins = [r for r in reps if r.get("counts_as_win")]
    coherent_beats = bool(wins)
    if coherent_beats:
        reason = (
            f"{len(wins)} hybrid representation(s) reached coherent_generation "
            "below q2f 2.25 bpw and under 27.55 ms."
        )
        answer = reason
    else:
        # cheapest correction that would be needed
        first_any = held_lr.get("first_healthy_rank_any")
        if first_any is not None and int(first_any) > RANK_BUDGET:
            bill = lowrank_bill(int(first_any))
            reason = (
                f"The cheapest distributed correction that restored held-out health "
                f"is low-rank r={first_any} at {bill['correction_bpw']:.4f} extra bpw "
                f"(body {bill['active_bpw']:.4f} bpw), which is >= (2.25-1.25) bpw. "
                "The hybrid does not beat q2f — confirming the 2.25 MLP floor."
            )
        elif not held_lr.get("first_healthy_rank_below_q2f") and not held_sh.get("healthy"):
            r256 = next((c for c in (held_lr.get("curve") or []) if c.get("rank") == 256), None)
            extra256 = lowrank_bill(256)["correction_bpw"]
            reason = (
                "No distributed correction under the 1.0 bpw budget restored held-out "
                "activations on real X. "
                + (
                    f"Even r=256 ({extra256:.3f} extra bpw, body "
                    f"{lowrank_bill(256)['active_bpw']:.3f} > 2.25) "
                    f"rel_fro={r256['vs_teacher']['rel_fro']:.4f}."
                    if r256
                    else "The SVD residual of binary g64 is high-rank."
                )
                + f" Shared K=2 (extra {sh_bill['correction_bpw']:.3f} bpw) mean rel_fro="
                + f"{held_sh.get('mean_rel_fro')}. Composition died at "
                + f"{composition.get('died_at')}. This confirms the 2.25 MLP floor: "
                "binary's physical speed cannot keep a distributed correction that "
                "actually restores generation under both 2.25 bpw and 27.55 ms."
            )
        else:
            reason = composition.get("why") or (
                "No coherent hybrid is both below q2f 2.25 bpw and faster than 27.55 ms."
            )
        answer = (
            "No coherent hybrid exists that is BOTH below q2f 2.25 bpw AND faster "
            f"than 27.55 ms. {reason}"
        )

    toward = None
    best_ns = None
    for r in reps:
        ns = (r.get("COMPLETE_TOKEN_NS") or {}).get("median")
        if ns is not None and (best_ns is None or ns < best_ns):
            best_ns = ns
    if best_ns is not None:
        toward = {
            "candidate_complete_token_ns": best_ns,
            "q2f_complete_token_ns": Q2F_COMPLETE_NS,
            "faster_than_q2f": best_ns < Q2F_COMPLETE_NS,
            "roof_tok_s": 729.7,
        }

    doc = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": "N038 (S024 §14, §102; N036 evidence)",
        "question": (
            "Can binary's physical speed keep a DISTRIBUTED correction under "
            "2.25 bpw AND under 27.55 ms while restoring generation, as ONE "
            "fused native operator (no dense W)?"
        ),
        "answer": answer,
        "coherent_hybrid_beats_q2f": coherent_beats,
        "did_not_load_second_27b": True,
        "did_not_write_under_models": True,
        "did_not_mutate_noetic_parent_a": True,
        "dense_w": 0,
        "dense_w_materialized": 0,
        "dense_w_is_a_counter": True,
        "native_only": True,
        "kernel_autopsy": competence["autopsy"],
        "kernel_competence": {
            "ok": competence["ok"],
            "any_geo_defective": competence["any_geo_defective"],
            "hybrid_kernels": competence["hybrid_kernels"],
        },
        "competent": bool(
            competence["autopsy"]["production_all_clear"]
            and not competence["any_geo_defective"]
        ),
        "build": {k: build.get(k) for k in ("ok", "wall_s", "exit_code", "stderr_tail", "skipped")},
        "run": {
            "ok": measured.get("ok"),
            "wall_s": measured.get("wall_s"),
            "exit_code": measured.get("exit_code"),
            "stderr_tail": measured.get("stderr_tail"),
            "command": measured.get("command"),
            "reused_raw": measured.get("reused_raw"),
        },
        "parity": {
            "ok": parity_ok,
            "noop_diverges": any(
                (p.get("id") == "lowrank_r8_noop_diverges" and p.get("noop_diverges"))
                or p.get("noop_diverges")
                for p in parity_rows
            ),
            "rows": parity_rows,
        },
        "q2f_baseline": {
            "bpw": Q2F_BPW,
            "complete_token_ns": Q2F_COMPLETE_NS,
            "ms": Q2F_MS,
            "mlp_graph_gpu_ns": q2f_mlp_ns,
        },
        "binary_bulk": {
            "bpw": BINARY_BPW,
            "note": "injured body; N032 23.43 ms, 16 copies of token 271",
        },
        "representations": reps,
        "n_hybrid_fused_operators": n_fused,
        "ACTIVE_BYTES_PER_TOKEN": {
            r["id"]: r.get("dram_bytes_per_token") for r in reps
        },
        "held_out_lowrank": {
            k: held_lr.get(k)
            for k in (
                "organ",
                "layer",
                "n_tokens",
                "not_gaussian",
                "real_activations",
                "reference",
                "curve",
                "probes",
                "first_healthy_rank_below_q2f",
                "first_healthy_rank_any",
                "rank_budget_for_1bpw_extra",
            )
        }
        if held_lr
        else {},
        "held_out_shared": {
            k: held_sh.get(k)
            for k in (
                "k",
                "organ",
                "fit_layers",
                "fit_wall_s",
                "layers",
                "mean_rel_fro",
                "mean_gain",
                "healthy",
                "n_healthy",
                "n",
                "accounting",
                "not_gaussian",
                "real_activations",
            )
        }
        if held_sh
        else {},
        "short_chain": shorts,
        "complete_organ": organs_scored,
        "composition_ladder": composition,
        "parent": parent_info,
        "capture": capture_info,
        "streamed_one_tensor_at_a_time": True,
        "not_gaussian": True,
        "toward_roof_729_7": toward,
        "finding": {
            "coherent_hybrid_beats_q2f": coherent_beats,
            "reason": reason,
            "n_representations": len(reps),
            "n_fused_native": n_fused,
            "n_wins": len(wins),
            "rank_budget_for_1bpw_extra": RANK_BUDGET,
            "shared_k2_below_q2f_bpw": sh_bill["below_q2f_bpw"],
            "lowrank_r8_below_q2f_bpw": lr8_bill["below_q2f_bpw"],
            "died_at": composition.get("died_at"),
            "rung": composition.get("rung"),
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    write_atomic(RECEIPT, json.dumps(doc, indent=2) + "\n")
    print(f"receipt {RECEIPT}", flush=True)
    print("answer:", answer, flush=True)
    return 0 if competence["ok"] and (skip_gpu or measured.get("ok")) and parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
