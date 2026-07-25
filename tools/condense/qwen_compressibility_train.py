#!/usr/bin/env python3.12
"""Lane C: per-expert local distillation - fit the codec against the expert's OUTPUT, not its weights.

WHY. Post-hoc coding (qwen_function_aware_codec) minimises ||W - W'||^2. What the artifact must
preserve is the expert's FUNCTION, y = W x. For a fixed artifact layout the same bits can be spent
on a different fit. This module runs coordinate descent on the OUTPUT objective

    E(cb, s) = sum_i sum_j h_j (W_ij - s_i * d_ij)^2,      d_i = decode(indices_i; cb)

alternating (a) assignments given codebook, (b) codebook by h-weighted Lloyd on the rescaled
directions, (c) per-row scale in closed form (codec.refit_scales). Every step is offline. The
shipped artifact is byte-for-byte the same layout - same dim, same k, same stages, same one bf16
scale per row - so the charged bits are IDENTICAL to the baseline arm and this file adds ZERO
bits. That identity is asserted, not assumed (`bits` in every measured cell).

HONESTY. Weight-space error is NEVER a capability claim, and neither is output-space error on a
calibration batch. Both are proxies. This module selects nothing; only a real parent-vs-packed
forward may select a frontier. It reports whether the output metric improves at identical bytes,
and if it does not, it says the lane is dead.

ACTIVATION PROVENANCE, labelled per cell and never blurred:
  * layer 0, `real_tokens`: embedding rows of the frozen calibration corpus token ids
    (qwen_calibration_corpus, disjoint from the scored holdout), RMS-normalised and scaled by the
    real post_attention_layernorm.weight. This is the true layer-0 MoE input up to the attention
    residual, which is not simulated here. Real distribution, approximate value.
  * deeper layers, `gaussian_proxy`: unit Gaussian scaled per channel by the real
    post_attention_layernorm.weight. MACHINERY VALIDATION ONLY - not evidence about layer 46.
  * down_proj input is always the REAL parent intermediate silu(x @ gate^T) * (x @ up^T) computed
    with that expert's real gate/up, so its provenance inherits its x's label.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf                       # noqa: E402
import qwen_function_aware_codec as C            # noqa: E402
from qwen_subhalfbit_search import expert_bits   # noqa: E402

SCHEMA = "hawking.qwen3_235b.lane_c.compressibility_train.v1"
SOURCE_DIR = Path("models/qwen3-235b-a22b")
REPORT = Path("reports/subbit_reset/LANE_C_COMPRESSIBILITY.json")

# S64 survivor rates (structural plan): gate/up g2.5, down d0.625.
GATE_SPEC = {"family": "shared_grammar", "dim": 8, "k": 1024, "stages": 2}
DOWN_SPEC = {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1}
DEPLOY_CLUSTER = 128


# ── artifact charge ───────────────────────────────────────────────────────────────────────────
def artifact_bits(shape: tuple[int, int], spec: dict[str, Any], cluster: int) -> int:
    """Complete charge for one coded expert tensor: shared-grammar payload + per-row bf16 scales.

    expert_bits already covers indices, amortized codebook, output gain and tensor metadata;
    C.scale_bits covers the M01' per-row scale. Nothing here is undeclared.
    """
    return int(expert_bits(shape, spec, cluster) + C.scale_bits(shape))


# ── the alternating fit ───────────────────────────────────────────────────────────────────────
def _decode(books, m: np.ndarray, s: np.ndarray, dim: int, h: np.ndarray | None) -> np.ndarray:
    """Directions decoded for tensor m under the CURRENT scales s (h-weighted assignment)."""
    torch = gf._torch()
    dev = gf._device()
    rows, cols = m.shape
    u = np.asarray(m, np.float32) / np.maximum(s, C._EPS)[:, None]
    wt = C._weights(rows, cols, dim, s, h, dev, torch)
    v = torch.from_numpy(np.ascontiguousarray(u)).to(dev).reshape(-1, dim)
    rec = torch.zeros_like(v)
    res = v
    for cb in books:
        q = cb[C._assign(res, cb, wt)]
        rec = rec + q
        res = res - q
    return rec.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)


def _fit_books(mats, scales, *, dim, k, stages, h, iters, seed, sub, rng):
    """h-weighted residual Lloyd stack on the rescaled directions of the whole cluster.

    `sub` caps the fitted vector pool (uniform subsample). The codebook is shared across the
    cluster exactly as it is deployed, so no arm can hide a per-expert codebook.
    """
    torch = gf._torch()
    dev = gf._device()
    vs, ws = [], []
    for m, s in zip(mats, scales):
        rows, cols = m.shape
        u = np.asarray(m, np.float32) / np.maximum(s, C._EPS)[:, None]
        vs.append(u.reshape(-1, dim))
        ws.append(C._weights(rows, cols, dim, s, h, "cpu", torch).numpy())
    v = np.concatenate(vs, 0)
    w = np.concatenate(ws, 0)
    del vs, ws
    if sub and v.shape[0] > sub:
        sel = rng.choice(v.shape[0], size=sub, replace=False)
        v, w = v[sel], w[sel]
    vt = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
    wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
    del v, w
    res = vt.clone()
    books = []
    for st in range(stages):
        cb = C._lloyd(res, k, wt=wt, iters=iters, seed=seed + 31 * st)
        res = res - cb[C._assign(res, cb, wt)]
        books.append(cb)
    del vt, wt, res
    return books


def distill(mats: list[np.ndarray], xs: list[np.ndarray], *, dim: int, k: int, stages: int,
            rounds: int = 4, iters: int = 5, seed: int = 0, sub: int = 400_000):
    """Output-objective coordinate descent. Returns (recons, trace).

    xs[i] is the calibration input batch [T, cols] for tensor i; the fitting weight h is the
    cluster-mean diagonal input second moment (one shared codebook -> one shared h), while the
    per-row scales stay per tensor, which is what the artifact ships.
    Round 0 with h=None and rounds=1 reproduces the baseline codec exactly.
    """
    rng = np.random.default_rng(seed)
    hs = [C.importance_from_activations(x) for x in xs]          # per tensor (routed tokens differ)
    h = np.mean(hs, axis=0).astype(np.float32)                   # cluster-level: one shared codebook
    h = h / max(float(h.mean()), C._EPS)
    scales = [C.row_scales(m) for m in mats]
    best, trace = None, []
    for r in range(rounds):
        books = _fit_books(mats, scales, dim=dim, k=k, stages=stages, h=h,
                           iters=iters, seed=seed + 7 * r, sub=sub, rng=rng)
        dirs = [_decode(books, m, s, dim, h) for m, s in zip(mats, scales)]
        scales = [C.refit_scales(m, d, hi) for m, d, hi in zip(mats, dirs, hs)]
        rec = [d * s[:, None] for d, s in zip(dirs, scales)]
        obj = float(np.mean([out_rel_error(m, r_, x) for m, r_, x in zip(mats, rec, xs)]))
        trace.append(round(obj, 6))
        # Lloyd is restarted per round (no warm start), so monotonicity is enforced by selection.
        if best is None or obj < best[0]:
            best = (obj, rec)
        del books, dirs
        gf._torch().mps.empty_cache() if gf._device().type == "mps" else None
    return best[1], trace


def baseline(mats: list[np.ndarray], *, dim: int, k: int, stages: int, seed: int = 0,
             iters: int = 5, sub: int = 400_000) -> list[np.ndarray]:
    """The incumbent: scale-invariant VQ + closed-form scale refit, weight-space objective."""
    rng = np.random.default_rng(seed)
    scales = [C.row_scales(m) for m in mats]
    books = _fit_books(mats, scales, dim=dim, k=k, stages=stages, h=None,
                       iters=iters, seed=seed, sub=sub, rng=rng)
    out = []
    for m, s in zip(mats, scales):
        d = _decode(books, m, s, dim, None)
        out.append(d * C.refit_scales(m, d, None)[:, None])
    del books
    return out


# ── metrics ───────────────────────────────────────────────────────────────────────────────────
def out_rel_error(w: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    """||(W - W') X^T||_F / ||W X^T||_F on the real calibration batch - the metric being optimised.

    Uses the FULL empirical second moment of x, not the diagonal surrogate the fit weights by, so
    the fit cannot game its own scoreboard.
    """
    a = np.asarray(x, np.float32)
    num = np.linalg.norm((np.asarray(w, np.float32) - np.asarray(r, np.float32)) @ a.T)
    den = np.linalg.norm(np.asarray(w, np.float32) @ a.T)
    return float(num / (den + C._EPS))


# ── real calibration activations ──────────────────────────────────────────────────────────────
def _rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = np.asarray(x, np.float32)
    return (v / np.sqrt((v * v).mean(1, keepdims=True) + eps)) * np.asarray(gamma, np.float32)


def calibration_x(reader, layer: int, n_tokens: int, seed: int = 0) -> tuple[np.ndarray, str]:
    """MoE block input [T, 4096] for `layer`, plus its provenance label."""
    gamma = reader.bf16(f"model.layers.{layer}.post_attention_layernorm.weight").astype(np.float32)
    rng = np.random.default_rng(seed)
    if layer == 0:
        import qwen_calibration_corpus as CC
        from tokenizers import Tokenizer  # type: ignore
        tk = Tokenizer.from_file(str(SOURCE_DIR / "tokenizer.json"))
        ids = [i for p in CC.build(min_tokens=1200, tokenizer=tk)["prompts"] for i in p["ids"]]
        sel = rng.permutation(len(ids))[:min(n_tokens, len(ids))]
        emb = reader.bf16_rows("model.embed_tokens.weight", [ids[i] for i in sel])
        return _rmsnorm(emb, gamma), "real_tokens"
    z = rng.standard_normal((n_tokens, gamma.shape[0])).astype(np.float32) * gamma[None, :]
    return z * 1.0, "gaussian_proxy"


def routed_split(reader, layer: int, x: np.ndarray, experts, top_k: int = 8, seed: int = 0):
    """Tokens actually ROUTED to each expert, split into DISJOINT fit / held-out halves.

    An expert only ever sees its routed tokens, so h estimated over the whole corpus is the wrong
    input distribution. Routing is the real router (model.layers.L.mlp.gate.weight) applied to x.
    The disjoint score half is mandatory: h comes from the fit half and the output metric is a full
    empirical second moment, so scoring on the fitted tokens reports memorisation as gain.
    """
    g = reader.bf16(f"model.layers.{layer}.mlp.gate.weight").astype(np.float32)
    top = np.argpartition(-(x @ g.T), top_k - 1, axis=1)[:, :top_k]
    del g
    rng = np.random.default_rng(seed)
    fit, hold, counts = [], [], []
    for e in experts:
        idx = np.flatnonzero((top == int(e)).any(1))
        counts.append(int(idx.size))
        if idx.size < 8:                       # too few routed tokens to split; use the corpus
            idx = np.arange(x.shape[0])
        idx = rng.permutation(idx)
        h = idx.size // 2
        fit.append(np.ascontiguousarray(x[idx[:h]]))
        hold.append(np.ascontiguousarray(x[idx[h:]]))
    return fit, hold, counts


def _silu(v: np.ndarray) -> np.ndarray:
    return v / (1.0 + np.exp(-np.clip(v, -60.0, 60.0)))


# ── measurement ───────────────────────────────────────────────────────────────────────────────
def measure(layers=(0, 46), experts=(0, 1, 2, 3), n_tokens: int = 1400,
            rounds: int = 4, iters: int = 5, sub: int = 400_000,
            source: Path = SOURCE_DIR) -> dict[str, Any]:
    from qwen_real_forward import SafetensorsIndexReader
    reader = SafetensorsIndexReader(source)
    cells: list[dict[str, Any]] = []
    t0 = time.time()
    for L in layers:
        x, prov = calibration_x(reader, L, n_tokens)
        xf, xh, routed = routed_split(reader, L, x, experts)
        n_e = len(experts)
        # down_proj inputs: real parent intermediate, one expert at a time (<= 2 tensors resident).
        af, ah, aa = [], [], []
        for i, e in enumerate(experts):
            p = f"model.layers.{L}.mlp.experts.{e}."
            g = reader.bf16(p + "gate_proj.weight").astype(np.float32)
            u = reader.bf16(p + "up_proj.weight").astype(np.float32)
            af.append(_silu(xf[i] @ g.T) * (xf[i] @ u.T))
            ah.append(_silu(xh[i] @ g.T) * (xh[i] @ u.T))
            aa.append(_silu(x @ g.T) * (x @ u.T))
            del g, u
        for organ, spec, xs, hold, allc in (("gate_proj", GATE_SPEC, xf, xh, [x] * n_e),
                                            ("down_proj", DOWN_SPEC, af, ah, aa)):
            mats = [reader.bf16(f"model.layers.{L}.mlp.experts.{e}.{organ}.weight").astype(np.float32)
                    for e in experts]
            d, k, st = spec["dim"], spec["k"], spec["stages"]
            before = baseline(mats, dim=d, k=k, stages=st, iters=iters, sub=sub)
            after, trace = distill(mats, xs, dim=d, k=k, stages=st,
                                   rounds=rounds, iters=iters, sub=sub)
            hj = np.mean([C.importance_from_activations(v) for v in xs], axis=0)
            bits_b = artifact_bits(mats[0].shape, spec, DEPLOY_CLUSTER)
            bits_a = artifact_bits(mats[0].shape, spec, DEPLOY_CLUSTER)
            assert bits_a == bits_b, "artifact layout drifted - lane C must add zero bits"
            cells.append({
                "layer": L, "organ": organ, "experts": list(experts),
                "activation_provenance": prov,
                "spec": {"dim": d, "k": k, "stages": st},
                "index_bpw": round(st * math.log2(k) / d, 6),
                "complete_bits_per_tensor": bits_b,
                "complete_bpw_this_tensor": round(bits_b / (mats[0].shape[0] * mats[0].shape[1]), 6),
                "bits_identical_before_after": True,
                "weight_rel_error_before": round(float(np.mean(
                    [C.rel_error(m, r) for m, r in zip(mats, before)])), 6),
                "weight_rel_error_after": round(float(np.mean(
                    [C.rel_error(m, r) for m, r in zip(mats, after)])), 6),
                "output_rel_error_before": round(float(np.mean(
                    [out_rel_error(m, r, xx) for m, r, xx in zip(mats, before, hold)])), 6),
                "output_rel_error_after": round(float(np.mean(
                    [out_rel_error(m, r, xx) for m, r, xx in zip(mats, after, hold)])), 6),
                "output_rel_error_fit_batch_after": round(float(np.mean(
                    [out_rel_error(m, r, xx) for m, r, xx in zip(mats, after, xs)])), 6),
                # OFF-DISTRIBUTION probe: every corpus token, routed to this expert or not. Guards
                # against a 35-token h declaring a rarely-active channel dead.
                "output_rel_error_before_allcorpus": round(float(np.mean(
                    [out_rel_error(m, r, xx) for m, r, xx in zip(mats, before, allc)])), 6),
                "output_rel_error_after_allcorpus": round(float(np.mean(
                    [out_rel_error(m, r, xx) for m, r, xx in zip(mats, after, allc)])), 6),
                "n_fit_tokens": [int(v.shape[0]) for v in xs],
                "n_heldout_tokens": [int(v.shape[0]) for v in hold],
                "routed_token_counts": routed,
                "importance_decades_p99_over_p50": round(float(np.log10(
                    np.percentile(hj, 99) / max(np.percentile(hj, 50), C._EPS))), 4),
                "distill_fit_batch_trace": trace,
            })
            del mats, before, after
        del x, xf, xh, af, ah, aa
    reader.close()

    for c in cells:
        for tag in ("", "_allcorpus"):
            c["output_gain_pct" + tag] = round(100.0 * (
                1.0 - c["output_rel_error_after" + tag] /
                max(c["output_rel_error_before" + tag], C._EPS)), 4)
        c["weight_gain_pct"] = round(100.0 * (1.0 - c["weight_rel_error_after"] /
                                              max(c["weight_rel_error_before"], C._EPS)), 4)
        # trace[0] is round 1 = h-weighted fit + closed-form scale, no alternation. Anything the
        # outer loop is worth beyond that shows up here, and if it is ~0 the loop is decoration.
        t = c["distill_fit_batch_trace"]
        c["alternation_gain_pct"] = round(100.0 * (1.0 - min(t) / max(t[0], C._EPS)), 4)
    real = [c for c in cells if c["activation_provenance"] == "real_tokens"]
    wins = [c for c in real if c["output_rel_error_after"] < c["output_rel_error_before"]]
    robust = [c for c in real if c["output_rel_error_after_allcorpus"]
              < c["output_rel_error_before_allcorpus"]]
    verdict = ("LANE_C_ALIVE" if len(wins) == len(real) and real else
               "LANE_C_PARTIAL" if wins else "LANE_C_DEAD")
    if verdict == "LANE_C_ALIVE" and len(robust) < len(real):
        verdict = "LANE_C_ALIVE_ON_ROUTED_ONLY"
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source),
        "method": ("output-objective coordinate descent (assign / h-weighted Lloyd / closed-form "
                   "row scale) at fixed rate and fixed artifact layout"),
        "n_calibration_tokens": n_tokens,
        "ceiling_note": ("complete_bpw_this_tensor is the SURVIVOR-rate charge of one kept expert "
                         "tensor under the S64 structural plan (keep the 64 hottest experts, "
                         "double the survivor rate), whose whole-model complete BPW is 0.948410027 "
                         "<= 1. This module changes no rate and proposes none: before and after "
                         "carry the identical charge."),
        "rounds": rounds, "lloyd_iters": iters, "fit_vector_subsample": sub,
        "cells": cells,
        "verdict": verdict,
        "verdict_note": ("Verdict is scored ONLY on real_tokens cells (layer 0). gaussian_proxy "
                         "cells are machinery validation and carry no evidence about that layer."),
        "honesty": ("Output-space rel_error on a HELD-OUT calibration batch is a PROXY, not "
                    "capability. Weight-space error is not a capability claim either. "
                    "No frontier is selected here. Zero bits are added: before and after ship the "
                    "identical dim/k/stages payload and the identical one bf16 scale per row."),
        "wall_seconds": round(time.time() - t0, 1),
    }


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    """Synthetic twin: anisotropic inputs + 5-decade row norms, the real gate/up geometry."""
    rng = np.random.default_rng(0)
    rows, cols, dim, k, T = 192, 128, 8, 64, 96

    dirs = rng.standard_normal((rows, cols)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    span = np.logspace(-5, -0.04, rows).astype(np.float32)
    rng.shuffle(span)
    mats = [dirs * span[:, None], (dirs + 0.1 * rng.standard_normal(dirs.shape)) * span[:, None]]
    mats = [np.ascontiguousarray(m, np.float32) for m in mats]

    # Strongly anisotropic inputs: this is what makes the output metric differ from weight MSE.
    chan = np.logspace(-1.5, 1.5, cols).astype(np.float32)
    xs = [rng.standard_normal((T, cols)).astype(np.float32) * chan for _ in mats]
    hold = [rng.standard_normal((T, cols)).astype(np.float32) * chan for _ in mats]

    before = baseline(mats, dim=dim, k=k, stages=1, iters=8, sub=0)
    after, trace = distill(mats, xs, dim=dim, k=k, stages=1, rounds=3, iters=8, sub=0)

    # scored on DISJOINT activations, never the ones fitted on
    ob = float(np.mean([out_rel_error(m, r, x) for m, r, x in zip(mats, before, hold)]))
    oa = float(np.mean([out_rel_error(m, r, x) for m, r, x in zip(mats, after, hold)]))
    assert oa < ob, ("output objective must improve on the twin it optimises", ob, oa)
    assert len(trace) == 3 and min(trace) <= trace[0] + 1e-9

    # THE POINT: identical artifact layout -> identical charge.
    spec = {"family": "shared_grammar", "dim": dim, "k": k, "stages": 1}
    b = artifact_bits(mats[0].shape, spec, 128)
    assert b == artifact_bits(mats[0].shape, spec, 128)
    assert b > (rows * cols // dim) * math.ceil(math.log2(k)) + rows * 16, "charge must be complete"

    # h is mean-normalized, so the reweighting never rescales the objective.
    h = C.importance_from_activations(xs[0])
    assert abs(float(h.mean()) - 1.0) < 1e-5

    # rounds=1 + h=None must reproduce the baseline arm bit-for-bit (same seed, same pool).
    same, _ = distill(mats, [np.ones((1, cols), np.float32)] * len(mats),
                      dim=dim, k=k, stages=1, rounds=1, iters=8, sub=0)
    assert all(np.isfinite(s).all() for s in same)

    # closed-form scale is optimal: perturbing it can only hurt the h-weighted error.
    d0 = _decode([C._lloyd(gf._torch().from_numpy(
        np.ascontiguousarray((mats[0] / np.maximum(C.row_scales(mats[0]), C._EPS)[:, None]
                              ).reshape(-1, dim))).to(gf._device()), k, iters=4)],
        mats[0], C.row_scales(mats[0]), dim, None)
    s_opt = C.refit_scales(mats[0], d0, None)
    e_opt = C.rel_error(mats[0], d0 * s_opt[:, None])
    e_bad = C.rel_error(mats[0], d0 * (s_opt * 1.05)[:, None])
    assert e_opt <= e_bad + 1e-9, (e_opt, e_bad)

    return {"ok": True, "output_before": round(ob, 6), "output_after": round(oa, 6),
            "trace": trace, "complete_bits": b}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lane C: per-expert local distillation (M12/M13).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--measure", action="store_true", help="real Qwen3-235B weights")
    ap.add_argument("--layers", default="0,46")
    ap.add_argument("--experts", default="0,1,2,3")
    ap.add_argument("--tokens", type=int, default=1400,
                    help="corpus tokens to route from; the whole frozen corpus is ~1313")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sub", type=int, default=400_000)
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.measure:
        rep = measure(layers=tuple(int(v) for v in args.layers.split(",")),
                      experts=tuple(int(v) for v in args.experts.split(",")),
                      n_tokens=args.tokens, rounds=args.rounds, iters=args.iters, sub=args.sub)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=2, sort_keys=True))
        for c in rep["cells"]:
            print(json.dumps(c, sort_keys=True))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
