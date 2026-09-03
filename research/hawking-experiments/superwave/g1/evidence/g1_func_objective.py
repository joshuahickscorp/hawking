#!/usr/bin/env python3
"""G-FUNC: weight-error vs function-error at identical bit width.

Scores come from tools/gravity_doctor_gate.py (observed / probed / worst_unit).
Fit is prompt-held-out. In-sample numbers are labeled and never used as the score.

CPU / numpy only. No GPU, no generate, no pack, no resident touch.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

REPO = Path("/Users/scammermike/.claude-grok/worktrees/205-func-objective-20260817-181028")
sys.path.insert(0, str(REPO / "tools"))

from gravity_doctor_gate import (  # noqa: E402
    AXIS_MARGIN,
    axes,
    c_faithful_q4,
    c_uniform,
    c_visible_subspace,
    gate,
    load_tensor,
    _probe,
    _rowcos,
    _worst_unit,
)

ART = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b")
BF16 = ART / "bf16"
CAP_V1 = ART / "activation-capture-v1"
CAP_V2 = ART / "activation-capture-v2" / "parent_bf16"
OUT = Path("/tmp/g1_func_objective")
OUT.mkdir(parents=True, exist_ok=True)

N = 26_895_998_464
GROUP = 128
MULT = (0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00)
HOLD_FRAC = 0.25
V1_PROMPT_LENS = (57, 60, 68, 61, 10)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def tname(layer: int, role: str) -> str:
    if role == "gate":
        return f"language_model.model.layers.{layer}.mlp.gate_proj.weight"
    if role == "up":
        return f"language_model.model.layers.{layer}.mlp.up_proj.weight"
    if role == "down":
        return f"language_model.model.layers.{layer}.mlp.down_proj.weight"
    if role == "q":
        return f"language_model.model.layers.{layer}.self_attn.q_proj.weight"
    if role == "k":
        return f"language_model.model.layers.{layer}.self_attn.k_proj.weight"
    if role == "v":
        return f"language_model.model.layers.{layer}.self_attn.v_proj.weight"
    if role == "o":
        return f"language_model.model.layers.{layer}.self_attn.o_proj.weight"
    if role == "qkv":
        return f"language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"
    if role == "z":
        return f"language_model.model.layers.{layer}.linear_attn.in_proj_z.weight"
    if role == "out":
        return f"language_model.model.layers.{layer}.linear_attn.out_proj.weight"
    raise KeyError(role)


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def load_W(name: str) -> np.ndarray:
    return load_tensor(name, root=str(BF16))


def load_v1_X(layer: int) -> np.ndarray:
    rec = json.loads((CAP_V1 / "capture-result.json").read_text())
    per = rec["per_layer"][str(layer)]
    X = np.fromfile(per["path"], dtype=np.float32).reshape(per["n_rows"], rec["hidden"])
    return np.ascontiguousarray(X, dtype=np.float32)


def load_v2_site(site: str, layer: int) -> np.ndarray:
    rec = json.loads((CAP_V2 / "capture-result.json").read_text())
    per = rec["sites"][site]["per_layer"][str(layer)]
    width = int(per["width"])
    n = int(per["n_rows"])
    raw = np.fromfile(per["path"], dtype=np.float16)
    if raw.size != n * width:
        raise RuntimeError(f"{site} L{layer} size {raw.size} != {n}*{width}")
    return np.ascontiguousarray(raw.reshape(n, width).astype(np.float32))


def v2_receipt() -> dict:
    return json.loads((CAP_V2 / "capture-result.json").read_text())


def site_row_splits(prompts: list, store_n: int) -> np.ndarray:
    """Replay capture-v2 site-split: first take_n tokens of each prompt, prompt order."""
    fit_left = int(math.floor((1.0 - HOLD_FRAC) * store_n))
    hold_left = store_n - fit_left
    splits = []
    for spec in prompts:
        n_here = int(spec["n_tokens"])
        if spec["split"] == "fit":
            take = min(n_here, fit_left)
            fit_left -= take
        else:
            take = min(n_here, hold_left)
            hold_left -= take
        splits.extend([spec["split"]] * take)
    if len(splits) != store_n:
        raise RuntimeError(f"split reconstruction {len(splits)} != store_n {store_n}")
    return np.array(splits)


def v2_masks(site: str) -> tuple[np.ndarray, np.ndarray, dict]:
    rec = v2_receipt()
    store_n = int(rec["sites"][site]["store_n"])
    splits = site_row_splits(rec["prompts"], store_n)
    fit = splits == "fit"
    hold = splits == "hold"
    meta = {
        "site": site,
        "store_n": store_n,
        "n_fit": int(fit.sum()),
        "n_hold": int(hold.sum()),
        "receipt_n_fit": rec["sites"][site]["n_fit"],
        "receipt_n_hold": rec["sites"][site]["n_hold"],
        "match_receipt": int(fit.sum()) == rec["sites"][site]["n_fit"]
        and int(hold.sum()) == rec["sites"][site]["n_hold"],
    }
    return fit, hold, meta


def v1_masks() -> dict:
    all_i = np.arange(256)
    sl, s = [], 0
    for n in V1_PROMPT_LENS:
        sl.append((s, s + n))
        s += n
    prompt_fit = np.concatenate([np.arange(a, b) for a, b in sl[:3]])
    prompt_hold = np.concatenate([np.arange(a, b) for a, b in sl[3:]])
    return {
        "evenodd": {"fit": all_i[0::2], "hold": all_i[1::2], "leak": "every prompt"},
        "prompt": {"fit": prompt_fit, "hold": prompt_hold, "leak": "none"},
        "s0": {"fit": all_i[:192], "hold": all_i[192:], "leak": "prompt4 split"},
    }


def payload_bytes(n_elem: int, bits: int, g: int = GROUP) -> int:
    """HQ30-family header 40 B + f16 scale + packed codes. MEASURED formula from distillation lane."""
    ng = (n_elem + g - 1) // g
    code_b = (bits * g + 7) // 8
    return 40 + ng * (2 + code_b)


def complete_bpw_tensor(n_elem: int, bits: int, g: int = GROUP) -> float:
    return 8.0 * payload_bytes(n_elem, bits, g) / n_elem


def compute_grams(X_fit: np.ndarray, g: int = GROUP) -> np.ndarray:
    n_in = X_fit.shape[1]
    n_blocks = n_in // g
    grams = np.empty((n_blocks, g, g), dtype=np.float64)
    Xf = np.ascontiguousarray(X_fit[:, : n_blocks * g], dtype=np.float64)
    for b in range(n_blocks):
        xb = Xf[:, b * g : (b + 1) * g]
        grams[b] = xb.T @ xb
    return grams


def search_scales(
    W: np.ndarray,
    bits: int,
    grams: np.ndarray | None,
    multipliers=MULT,
    snap_f16: bool = True,
    g: int = GROUP,
    row_chunk: int = 1024,
) -> tuple[np.ndarray, dict]:
    """Per-group 8-point scale search. grams=None => weight MSE (e^T e)."""
    t0 = time.time()
    bound = (1 << (bits - 1)) - 1
    n_out, n_in = W.shape
    if n_in % g != 0:
        raise RuntimeError(f"in_dim {n_in} not divisible by {g}")
    n_blocks = n_in // g
    n_groups = n_out * n_blocks
    chosen = np.empty((n_out, n_blocks), dtype=np.float32)
    picked = np.zeros(len(multipliers), dtype=np.int64)
    Wf = np.ascontiguousarray(W, dtype=np.float32)
    use_gram = grams is not None
    if use_gram and grams.shape[0] != n_blocks:
        raise RuntimeError(f"grams {grams.shape} vs n_blocks {n_blocks}")
    for r0 in range(0, n_out, row_chunk):
        r1 = min(n_out, r0 + row_chunk)
        Wc3 = Wf[r0:r1].reshape(r1 - r0, n_blocks, g)
        amax = np.max(np.abs(Wc3), axis=2)
        s0 = amax / max(bound, 1)
        best_c = np.full((r1 - r0, n_blocks), np.inf, dtype=np.float64)
        best_s = np.zeros((r1 - r0, n_blocks), dtype=np.float32)
        best_i = np.zeros((r1 - r0, n_blocks), dtype=np.int16)
        for i, m in enumerate(multipliers):
            s = (s0 * float(m)).astype(np.float32)
            if snap_f16:
                s = s.astype(np.float16).astype(np.float32)
            den = np.where(s > 0.0, s, 1.0)
            codes = np.clip(np.rint(Wc3 / den[..., None]), -bound, bound)
            e = Wc3.astype(np.float64) - codes.astype(np.float64) * s.astype(np.float64)[..., None]
            if use_gram:
                ge = np.einsum("bkl,cbl->cbk", grams, e, optimize=True)
                cost = np.einsum("cbk,cbk->cb", ge, e, optimize=True)
            else:
                cost = np.einsum("cbg,cbg->cb", e, e, optimize=True)
            cost = np.where(s0 > 0.0, cost, 0.0)
            better = cost < best_c
            best_c = np.where(better, cost, best_c)
            best_s = np.where(better, s, best_s)
            best_i = np.where(better, i, best_i)
        chosen[r0:r1] = best_s
        for i in range(len(multipliers)):
            picked[i] += int(np.sum(best_i == i))
        del Wc3, amax, s0, best_c, best_s, best_i
    i1 = list(multipliers).index(1.0) if 1.0 in multipliers else None
    n_not = int(n_groups - picked[i1]) if i1 is not None else int(n_groups)
    n_down = int(sum(int(picked[i]) for i, m in enumerate(multipliers) if m < 1.0))
    meta = {
        "n_groups": int(n_groups),
        "n_blocks": int(n_blocks),
        "n_out": int(n_out),
        "n_in": int(n_in),
        "multipliers": [float(m) for m in multipliers],
        "snap_f16": bool(snap_f16),
        "n_groups_not_absmax": n_not,
        "frac_groups_not_absmax": float(n_not) / float(n_groups),
        "n_groups_smaller_than_absmax": n_down,
        "frac_groups_smaller_than_absmax": float(n_down) / float(n_groups),
        "n_picked_per_multiplier": [int(x) for x in picked],
        "wall_s": time.time() - t0,
        "objective": "func_eGe" if use_gram else "weight_mse",
        "complete_bpw_tensor": complete_bpw_tensor(n_out * n_in, bits, g),
        "payload_bytes": payload_bytes(n_out * n_in, bits, g),
        "nominal_bpw": float(bits) + 16.0 / float(g),
    }
    return chosen, meta


def quant_with_scales(W: np.ndarray, scales: np.ndarray, bits: int, g: int = GROUP) -> np.ndarray:
    bound = (1 << (bits - 1)) - 1
    n_out, n_in = W.shape
    n_blocks = n_in // g
    W3 = W.reshape(n_out, n_blocks, g)
    s = scales.reshape(n_out, n_blocks)
    den = np.where(s > 0.0, s, 1.0)
    codes = np.clip(np.rint(W3 / den[..., None]), -bound, bound)
    rec = codes * s[..., None]
    return np.ascontiguousarray(rec.reshape(n_out, n_in), dtype=np.float32)


def absmax_scales(W: np.ndarray, bits: int, g: int = GROUP, snap_f16: bool = True) -> np.ndarray:
    bound = (1 << (bits - 1)) - 1
    n_out, n_in = W.shape
    n_blocks = n_in // g
    W3 = W.reshape(n_out, n_blocks, g)
    s = np.max(np.abs(W3), axis=2) / max(bound, 1)
    if snap_f16:
        s = s.astype(np.float16).astype(np.float32)
    return s.astype(np.float32)


def pack_axes(a: dict) -> dict:
    return {k: float(a[k]) for k in ("observed", "probed", "worst_unit")}


def pack_gate(g: dict) -> dict:
    out = {
        "observed": float(g["observed"]),
        "probed": float(g["probed"]),
        "worst_unit": float(g["worst_unit"]),
        "gate": float(g["gate"]),
        "worst_axis": g.get("worst_axis"),
        "healthy": bool(g["healthy"]),
        "mode": g["mode"],
    }
    if "deficit" in g:
        out["deficit"] = {k: float(v) for k, v in g["deficit"].items()}
    return out


def score_pair(W: np.ndarray, Wh: np.ndarray, X_hold: np.ndarray, X_fit: np.ndarray | None, seed: int = 0) -> dict:
    ref_W = c_faithful_q4(W, group=GROUP)
    ref = axes(W, ref_W, X_hold, seed=seed)
    g = gate(W, Wh, X_hold, ref=ref, seed=seed)
    rec = {
        "hold": pack_gate(g),
        "ref_q4_g128": pack_axes(ref),
        "margins": dict(AXIS_MARGIN),
        "weight_cosine": _rowcos(W.reshape(1, -1), Wh.reshape(1, -1)),
        "weight_rel_l2": float(
            np.linalg.norm(W - Wh) / (np.linalg.norm(W) + 1e-30)
        ),
    }
    if X_fit is not None:
        rec["fit_observed_IN_SAMPLE"] = float(axes(W, Wh, X_fit, seed=seed)["observed"])
        rec["hold_minus_fit_observed"] = rec["hold"]["observed"] - rec["fit_observed_IN_SAMPLE"]
    del ref_W
    return rec


def numerical_rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X, tol=1e-3 * np.linalg.norm(X, 2)))


def visible_energy(W: np.ndarray, X: np.ndarray) -> dict:
    """Fraction of ||W||_F^2 in the capture-visible input subspace."""
    # SVD of X: X = U S Vt, visible = first r right singular vectors
    # Use economy SVD on a centered-or-not X; doctor uses raw X.
    # For wide X (n < d) compute via X.T @ X eigens if cheaper.
    n, d = X.shape
    t0 = time.time()
    if n >= d:
        # tall or square
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        r = int(np.sum(s > 1e-3 * s[0])) if s.size else 0
        # doctor uses matrix_rank with tol=1e-3 * ||X||_2 == 1e-3 * s[0]
        B = vt[:r]
        P = B.T @ B
    else:
        # fat: rank <= n. SVD of X (n x d) is fine for n=256, d=5120.
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        r = int(np.sum(s > 1e-3 * (s[0] if s.size else 1.0)))
        B = vt[:r]
        P = B.T @ B
    Wp = W @ P
    Wn = W - Wp
    w2 = float(np.square(W, dtype=np.float64).sum())
    vis = float(np.square(Wp, dtype=np.float64).sum()) / max(w2, 1e-30)
    nul = float(np.square(Wn, dtype=np.float64).sum()) / max(w2, 1e-30)
    return {
        "rank": r,
        "dim": int(d),
        "n_rows": int(n),
        "visible_energy": vis,
        "null_energy": nul,
        "s0": float(s[0]) if s.size else 0.0,
        "wall_s": time.time() - t0,
    }


def role_site(role: str) -> str:
    if role in ("gate", "up"):
        return "post_attn_norm"
    if role == "down":
        return "post_swiglu"
    if role in ("q", "k", "v", "qkv", "z"):
        return "post_input_norm"
    if role in ("o", "out"):
        return "mixer_x"
    raise KeyError(role)


def role_class(role: str) -> str:
    if role in ("gate", "up", "down"):
        return "mlp"
    return "attention"


CELLS = [
    # three+ depths, MLP + attention
    (0, "gate"),
    (0, "down"),
    (0, "qkv"),
    (0, "out"),
    (15, "gate"),
    (15, "down"),
    (15, "v"),
    (15, "o"),
    (31, "gate"),
    (31, "down"),
    (31, "v"),
    (31, "q"),
    (31, "o"),
    (63, "gate"),
    (63, "down"),
    (63, "v"),
    (63, "o"),
]


def run_cell(layer: int, role: str, bits: int, capture: str = "v2") -> dict:
    name = tname(layer, role)
    log(f"CELL {capture} L{layer} {role} q{bits} load {name}")
    W = load_W(name)
    if capture == "v2":
        site = role_site(role)
        X = load_v2_site(site, layer)
        fit_m, hold_m, split_meta = v2_masks(site)
        X_fit = np.ascontiguousarray(X[fit_m])
        X_hold = np.ascontiguousarray(X[hold_m])
        del X
        split_kind = "v2_prompt"
    else:
        X = load_v1_X(layer)
        if W.shape[1] != X.shape[1]:
            raise RuntimeError(f"v1 width {X.shape[1]} != W in {W.shape[1]} for {name}")
        m = v1_masks()["prompt"]
        X_fit = np.ascontiguousarray(X[m["fit"]])
        X_hold = np.ascontiguousarray(X[m["hold"]])
        del X
        split_meta = {"n_fit": int(X_fit.shape[0]), "n_hold": int(X_hold.shape[0]), "leak": "none"}
        split_kind = "v1_prompt"
        site = "v1_post_norm_hidden"
    if X_fit.shape[1] != W.shape[1]:
        raise RuntimeError(f"X in {X_fit.shape} vs W {W.shape} site={site}")

    out = {
        "layer": layer,
        "role": role,
        "class": role_class(role),
        "name": name,
        "shape": [int(x) for x in W.shape],
        "bits": bits,
        "group": GROUP,
        "capture": capture,
        "site": site,
        "split": split_kind,
        "split_meta": split_meta,
        "n_fit": int(X_fit.shape[0]),
        "n_hold": int(X_hold.shape[0]),
        "complete_bpw_tensor": complete_bpw_tensor(W.size, bits),
        "payload_bytes": payload_bytes(W.size, bits),
        "nominal_bpw": bits + 16.0 / GROUP,
        "fits": {},
    }

    # three fits at identical width
    plans = [
        ("weight_absmax", "absmax"),
        ("weight_mse", "wmse"),
        ("func_mse", "fmse"),
    ]
    grams = None
    for fit_name, kind in plans:
        t1 = time.time()
        if kind == "absmax":
            scales = absmax_scales(W, bits, snap_f16=True)
            meta = {
                "objective": "weight_absmax",
                "snap_f16": True,
                "frac_groups_not_absmax": 0.0,
                "n_groups": int(W.shape[0] * (W.shape[1] // GROUP)),
                "wall_s": 0.0,
            }
        elif kind == "wmse":
            scales, meta = search_scales(W, bits, grams=None, snap_f16=True)
        else:
            grams = compute_grams(X_fit, GROUP)
            scales, meta = search_scales(W, bits, grams=grams, snap_f16=True)
            del grams
            grams = None
        Wh = quant_with_scales(W, scales, bits)
        scored = score_pair(W, Wh, X_hold, X_fit)
        scored["search"] = meta
        scored["wall_s"] = time.time() - t1
        out["fits"][fit_name] = scored
        ghold = scored["hold"]
        log(
            f"  {fit_name:14s} gate={ghold['gate']:+.6f} obs={ghold['observed']:.6f} "
            f"prb={ghold['probed']:.6f} wu={ghold['worst_unit']:.6f} "
            f"axis={ghold['worst_axis']} healthy={ghold['healthy']} "
            f"wcos={scored['weight_cosine']:.6f} frac_moved={meta.get('frac_groups_not_absmax', 0):.4f}"
        )
        del Wh, scales
        gc.collect()

    # deltas: func - weight_mse  (the priced objective change)
    fa, fb = out["fits"]["func_mse"]["hold"], out["fits"]["weight_mse"]["hold"]
    fc = out["fits"]["weight_absmax"]["hold"]
    out["delta_func_minus_weight_mse"] = {
        "gate": fa["gate"] - fb["gate"],
        "observed": fa["observed"] - fb["observed"],
        "probed": fa["probed"] - fb["probed"],
        "worst_unit": fa["worst_unit"] - fb["worst_unit"],
    }
    out["delta_func_minus_absmax"] = {
        "gate": fa["gate"] - fc["gate"],
        "observed": fa["observed"] - fc["observed"],
        "probed": fa["probed"] - fc["probed"],
        "worst_unit": fa["worst_unit"] - fc["worst_unit"],
    }
    del W, X_fit, X_hold
    gc.collect()
    return out


def swiglu(x: np.ndarray, G: np.ndarray, U: np.ndarray, D: np.ndarray) -> np.ndarray:
    a = x @ G.T
    return (silu(a) * (x @ U.T)) @ D.T


def block_axes(x: np.ndarray, G, U, D, Gh, Uh, Dh, seed: int = 0) -> dict:
    y = swiglu(x, G, U, D)
    yh = swiglu(x, Gh, Uh, Dh)
    P = _probe(G.shape[1], n=256, seed=seed)
    yp = swiglu(P, G, U, D)
    yph = swiglu(P, Gh, Uh, Dh)
    return {
        "observed": _rowcos(y, yh),
        "probed": _rowcos(yp, yph),
        "worst_unit": min(_worst_unit(y, yh), _worst_unit(yp, yph)),
    }


def block_gate(x, G, U, D, Gh, Uh, Dh, ref, seed=0):
    a = block_axes(x, G, U, D, Gh, Uh, Dh, seed=seed)
    deficits = {k: a[k] - (ref[k] - AXIS_MARGIN[k]) for k in a}
    worst = min(deficits, key=deficits.get)
    return {
        **{k: float(a[k]) for k in a},
        "deficit": {k: float(v) for k, v in deficits.items()},
        "gate": float(deficits[worst]),
        "worst_axis": worst,
        "healthy": deficits[worst] >= 0.0,
        "mode": "relative_block",
    }


def run_block(layer: int, bits: int) -> dict:
    log(f"BLOCK L{layer} q{bits}")
    G = load_W(tname(layer, "gate"))
    U = load_W(tname(layer, "up"))
    D = load_W(tname(layer, "down"))
    X = load_v2_site("post_attn_norm", layer)
    fit_m, hold_m, split_meta = v2_masks("post_attn_norm")
    X_fit = np.ascontiguousarray(X[fit_m])
    X_hold = np.ascontiguousarray(X[hold_m])
    del X

    def fit_tensor(W, Xf, kind):
        if kind == "absmax":
            s = absmax_scales(W, bits, snap_f16=True)
            meta = {"objective": "weight_absmax", "frac_groups_not_absmax": 0.0}
        elif kind == "wmse":
            s, meta = search_scales(W, bits, grams=None, snap_f16=True)
        else:
            grams = compute_grams(Xf, GROUP)
            s, meta = search_scales(W, bits, grams=grams, snap_f16=True)
            del grams
        Wh = quant_with_scales(W, s, bits)
        return Wh, meta

    # per-tensor fits
    G_abs, mGa = fit_tensor(G, X_fit, "absmax")
    U_abs, mUa = fit_tensor(U, X_fit, "absmax")
    # down X is post_swiglu (teacher intermediate)
    H_teacher = load_v2_site("post_swiglu", layer)
    hf, hh, hmeta = v2_masks("post_swiglu")
    H_fit = np.ascontiguousarray(H_teacher[hf])
    del H_teacher
    D_abs, mDa = fit_tensor(D, H_fit, "absmax")

    G_w, mGw = fit_tensor(G, X_fit, "wmse")
    U_w, mUw = fit_tensor(U, X_fit, "wmse")
    D_w, mDw = fit_tensor(D, H_fit, "wmse")

    G_f, mGf = fit_tensor(G, X_fit, "func")
    U_f, mUf = fit_tensor(U, X_fit, "func")
    D_f, mDf = fit_tensor(D, H_fit, "func")  # per-tensor func on teacher H

    # coupled: fit D against h_hat produced by quantized G,U on the FIT split
    h_hat_fit = silu(X_fit @ G_f.T) * (X_fit @ U_f.T)
    grams_h = compute_grams(h_hat_fit, GROUP)
    s_c, mDc = search_scales(D, bits, grams=grams_h, snap_f16=True)
    D_c = quant_with_scales(D, s_c, bits)
    del grams_h, h_hat_fit, s_c

    # Q4 reference block
    G4 = c_faithful_q4(G, group=GROUP)
    U4 = c_faithful_q4(U, group=GROUP)
    D4 = c_faithful_q4(D, group=GROUP)
    ref = block_axes(X_hold, G, U, D, G4, U4, D4)
    del G4, U4, D4

    variants = {
        "weight_absmax_indep": (G_abs, U_abs, D_abs, {"G": mGa, "U": mUa, "D": mDa}),
        "weight_mse_indep": (G_w, U_w, D_w, {"G": mGw, "U": mUw, "D": mDw}),
        "func_mse_indep": (G_f, U_f, D_f, {"G": mGf, "U": mUf, "D": mDf}),
        "func_mse_coupled_down_on_hhat": (G_f, U_f, D_c, {"G": mGf, "U": mUf, "D": mDc}),
    }
    out = {
        "layer": layer,
        "bits": bits,
        "group": GROUP,
        "split_meta": split_meta,
        "h_split_meta": hmeta,
        "n_fit": int(X_fit.shape[0]),
        "n_hold": int(X_hold.shape[0]),
        "ref_q4_g128": {k: float(ref[k]) for k in ref},
        "variants": {},
    }
    for name, (Gh, Uh, Dh, meta) in variants.items():
        gsc = block_gate(X_hold, G, U, D, Gh, Uh, Dh, ref)
        # in-sample diagnostic
        gin = block_axes(X_fit, G, U, D, Gh, Uh, Dh)
        rec = {
            "hold": gsc,
            "fit_observed_IN_SAMPLE": float(gin["observed"]),
            "hold_minus_fit_observed": gsc["observed"] - float(gin["observed"]),
            "search": {
                k: {
                    "objective": v.get("objective"),
                    "frac_groups_not_absmax": v.get("frac_groups_not_absmax"),
                    "n_picked_per_multiplier": v.get("n_picked_per_multiplier"),
                    "wall_s": v.get("wall_s"),
                }
                for k, v in meta.items()
            },
        }
        out["variants"][name] = rec
        log(
            f"  BLOCK {name:32s} gate={gsc['gate']:+.6f} obs={gsc['observed']:.6f} "
            f"prb={gsc['probed']:.6f} wu={gsc['worst_unit']:.6f} axis={gsc['worst_axis']} "
            f"healthy={gsc['healthy']}"
        )
        del Gh, Uh, Dh
        gc.collect()

    fa = out["variants"]["func_mse_indep"]["hold"]
    fb = out["variants"]["weight_mse_indep"]["hold"]
    fc = out["variants"]["func_mse_coupled_down_on_hhat"]["hold"]
    out["delta_func_minus_weight_mse"] = {
        "gate": fa["gate"] - fb["gate"],
        "observed": fa["observed"] - fb["observed"],
        "probed": fa["probed"] - fb["probed"],
        "worst_unit": fa["worst_unit"] - fb["worst_unit"],
    }
    out["delta_coupled_minus_indep_func"] = {
        "gate": fc["gate"] - fa["gate"],
        "observed": fc["observed"] - fa["observed"],
        "probed": fc["probed"] - fa["probed"],
        "worst_unit": fc["worst_unit"] - fa["worst_unit"],
    }
    del G, U, D, G_abs, U_abs, D_abs, G_w, U_w, D_w, G_f, U_f, D_f, D_c
    del X_fit, X_hold, H_fit
    gc.collect()
    return out


def run_rank_probe() -> dict:
    log("RANK / visible energy")
    out = {}
    # v1 L0
    Xv1 = load_v1_X(0)
    Wg = load_W(tname(0, "gate"))
    r_v1 = numerical_rank(Xv1)
    ve = visible_energy(Wg, Xv1)
    cheat = c_visible_subspace(Wg, Xv1)
    # score cheat on v1 even/odd hold to show Goodhart
    m = v1_masks()["prompt"]
    Xh = Xv1[m["hold"]]
    ref = axes(Wg, c_faithful_q4(Wg, group=GROUP), Xh)
    gcheat = gate(Wg, cheat, Xh, ref=ref)
    # also full-X incumbent (the in-sample trap)
    gcheat_full = axes(Wg, cheat, Xv1)
    out["v1_L0"] = {
        "X_shape": [int(x) for x in Xv1.shape],
        "numerical_rank": r_v1,
        "visible_energy_gate": ve,
        "visible_subspace_cheat": {
            "full_X_observed_IN_SAMPLE": float(gcheat_full["observed"]),
            "full_X_probed": float(gcheat_full["probed"]),
            "full_X_worst_unit": float(gcheat_full["worst_unit"]),
            "prompt_hold_gate": pack_gate(gcheat),
            "weight_cosine": _rowcos(Wg.reshape(1, -1), cheat.reshape(1, -1)),
        },
    }
    log(
        f"  v1 L0 rank {r_v1}/{Xv1.shape[1]} visE={ve['visible_energy']:.5f} "
        f"nullE={ve['null_energy']:.5f} cheat_full_obs={gcheat_full['observed']:.6f} "
        f"cheat_hold_gate={gcheat['gate']:+.6f} probed={gcheat['probed']:.6f}"
    )
    del Xv1, cheat, Xh

    # v2 L0 post_attn_norm (MLP in) and post_input_norm
    for site in ("post_attn_norm", "post_input_norm"):
        X = load_v2_site(site, 0)
        fit_m, hold_m, sm = v2_masks(site)
        r_all = numerical_rank(X)
        r_fit = numerical_rank(X[fit_m])
        ve2 = visible_energy(Wg, X[fit_m])
        cheat2 = c_visible_subspace(Wg, X[fit_m])
        ref2 = axes(Wg, c_faithful_q4(Wg, group=GROUP), X[hold_m])
        gc2 = gate(Wg, cheat2, X[hold_m], ref=ref2)
        gc2_fit = axes(Wg, cheat2, X[fit_m])
        out[f"v2_L0_{site}"] = {
            "X_shape": [int(x) for x in X.shape],
            "split": sm,
            "rank_all": r_all,
            "rank_fit": r_fit,
            "visible_energy_gate_on_fit": ve2,
            "visible_subspace_cheat": {
                "fit_observed_IN_SAMPLE": float(gc2_fit["observed"]),
                "hold_gate": pack_gate(gc2),
                "weight_cosine": _rowcos(Wg.reshape(1, -1), cheat2.reshape(1, -1)),
            },
        }
        log(
            f"  v2 L0 {site} rank_all={r_all} rank_fit={r_fit}/{X.shape[1]} "
            f"visE={ve2['visible_energy']:.5f} cheat_fit_obs={gc2_fit['observed']:.6f} "
            f"cheat_hold_gate={gc2['gate']:+.6f} probed={gc2['probed']:.6f}"
        )
        del X, cheat2
        gc.collect()
    del Wg
    gc.collect()
    return out


def run_v1_diag() -> list:
    """Thin-capture Goodhart diagnostic: same codec, v1 prompt hold, L0/L31/L63."""
    cells = [(0, "gate", 2), (0, "down", 2), (31, "q", 2), (63, "gate", 2)]
    # down on v1: reconstruct SwiGLU from hidden + BF16 gate/up
    out = []
    for layer, role, bits in cells:
        log(f"V1DIAG L{layer} {role} q{bits}")
        name = tname(layer, role)
        W = load_W(name)
        H = load_v1_X(layer)
        m = v1_masks()["prompt"]
        if role == "down":
            G = load_W(tname(layer, "gate"))
            U = load_W(tname(layer, "up"))
            X = silu(H @ G.T) * (H @ U.T)
            del G, U
            site = "v1_reconstructed_swiglu"
        else:
            if W.shape[1] != H.shape[1]:
                log(f"  skip {name}: in {W.shape[1]} != hidden {H.shape[1]}")
                del W, H
                continue
            X = H
            site = "v1_post_norm_hidden"
        X_fit = np.ascontiguousarray(X[m["fit"]])
        X_hold = np.ascontiguousarray(X[m["hold"]])
        # also even/odd as the leaky split
        me = v1_masks()["evenodd"]
        Xe_fit = np.ascontiguousarray(X[me["fit"]])
        Xe_hold = np.ascontiguousarray(X[me["hold"]])
        del X, H

        rec = {
            "layer": layer,
            "role": role,
            "bits": bits,
            "site": site,
            "n_fit_prompt": int(X_fit.shape[0]),
            "n_hold_prompt": int(X_hold.shape[0]),
            "fits": {},
        }
        grams_p = compute_grams(X_fit, GROUP)
        grams_e = compute_grams(Xe_fit, GROUP)
        for fit_name, grams, Xf, Xh, split in (
            ("weight_mse_prompt", None, X_fit, X_hold, "prompt"),
            ("func_mse_prompt", grams_p, X_fit, X_hold, "prompt"),
            ("func_mse_evenodd_LEAKY", grams_e, Xe_fit, Xe_hold, "evenodd"),
            ("func_mse_evenodd_scored_on_prompt_hold", grams_e, Xe_fit, X_hold, "even_fit_prompt_hold"),
        ):
            if grams is None and fit_name.startswith("weight"):
                scales, meta = search_scales(W, bits, grams=None, snap_f16=True)
            else:
                scales, meta = search_scales(W, bits, grams=grams, snap_f16=True)
            Wh = quant_with_scales(W, scales, bits)
            scored = score_pair(W, Wh, Xh, Xf)
            scored["search"] = meta
            scored["split"] = split
            rec["fits"][fit_name] = scored
            ghold = scored["hold"]
            log(
                f"  {fit_name:40s} gate={ghold['gate']:+.6f} obs={ghold['observed']:.6f} "
                f"prb={ghold['probed']:.6f} in_sample={scored.get('fit_observed_IN_SAMPLE', float('nan')):.6f}"
            )
            del Wh, scales
        del grams_p, grams_e, W, X_fit, X_hold, Xe_fit, Xe_hold
        gc.collect()
        out.append(rec)
    return out


def main():
    t0 = time.time()
    (OUT / "run.log").write_text("")
    log("START g1_func_objective")

    # verify split reconstruction
    rec = v2_receipt()
    split_check = {}
    for site, s in rec["sites"].items():
        _, _, meta = v2_masks(site)
        split_check[site] = meta
        log(f"split {site}: {meta}")
        if not meta["match_receipt"]:
            raise RuntimeError(f"split mismatch {site}: {meta}")

    # capture hashes
    provenance = {
        "bf16": str(BF16),
        "v1_sha256_self": json.loads((CAP_V1 / "capture-result.json").read_text())["sha256_self"],
        "v1_receipt_sha256": sha256_file(CAP_V1 / "capture-result.json"),
        "v2_sha256_self": rec["sha256_self"],
        "v2_receipt_sha256": sha256_file(CAP_V2 / "capture-result.json"),
        "v2_n_tokens": rec["n_tokens"],
        "v2_status": rec["status"],
        "doctor_gate": str(REPO / "tools" / "gravity_doctor_gate.py"),
        "doctor_gate_sha256": sha256_file(REPO / "tools" / "gravity_doctor_gate.py"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "N": N,
        "group": GROUP,
        "multipliers": list(MULT),
        "axis_margin": dict(AXIS_MARGIN),
        "split_check": split_check,
        "complete_bpw_q2": 2.0 + 16.0 / GROUP,
        "complete_bpw_q3": 3.0 + 16.0 / GROUP,
        "note_complete_bpw": "per-weight code: bits + 16/group = bits+0.125 at g=128; identical for both objectives",
        "kernel": "existing uniform-Qn group-128 geo_tpr64; f16 scale plane; no shader change; no expand-to-Q4",
    }

    rank = run_rank_probe()
    (OUT / "rank.json").write_text(json.dumps(rank, indent=2))

    v1diag = run_v1_diag()
    (OUT / "v1diag.json").write_text(json.dumps(v1diag, indent=2))

    cells = []
    for layer, role in CELLS:
        for bits in (2, 3):
            try:
                cells.append(run_cell(layer, role, bits, capture="v2"))
                (OUT / "cells.json").write_text(json.dumps(cells, indent=2))
            except Exception as e:
                log(f"FAIL L{layer} {role} q{bits}: {type(e).__name__}: {e}")
                cells.append(
                    {
                        "layer": layer,
                        "role": role,
                        "bits": bits,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                (OUT / "cells.json").write_text(json.dumps(cells, indent=2))

    blocks = []
    for layer in (0, 31, 63):
        for bits in (2, 3):
            try:
                blocks.append(run_block(layer, bits))
                (OUT / "blocks.json").write_text(json.dumps(blocks, indent=2))
            except Exception as e:
                log(f"FAIL BLOCK L{layer} q{bits}: {type(e).__name__}: {e}")
                blocks.append({"layer": layer, "bits": bits, "error": f"{type(e).__name__}: {e}"})
                (OUT / "blocks.json").write_text(json.dumps(blocks, indent=2))

    report = {
        "schema": "hawking.g1.func_objective.v1",
        "wall_s": time.time() - t0,
        "rss_max_gb": rss_gb(),
        "provenance": provenance,
        "rank": rank,
        "v1diag": v1diag,
        "cells": cells,
        "blocks": blocks,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    log(f"DONE wall={report['wall_s']:.1f}s rss_max={report['rss_max_gb']:.3f}G")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        # one cell + rank, no full grid
        t0 = time.time()
        rec = v2_receipt()
        for site, s in rec["sites"].items():
            _, _, meta = v2_masks(site)
            log(f"split {site}: {meta}")
        rank = run_rank_probe()
        cell = run_cell(0, "gate", 2, capture="v2")
        print(json.dumps({"rank_keys": list(rank), "cell_delta": cell["delta_func_minus_weight_mse"],
                          "absmax_gate": cell["fits"]["weight_absmax"]["hold"],
                          "wmse_gate": cell["fits"]["weight_mse"]["hold"],
                          "func_gate": cell["fits"]["func_mse"]["hold"],
                          "wall": time.time() - t0, "rss": rss_gb()}, indent=2))
        sys.exit(0)
    raise SystemExit(main())
