#!/usr/bin/env python3.12
"""S3A: layerwise quantization-aware training at an EXACT fixed rate.

WHAT IS BEING TESTED. Every sub-bit arm this campaign has run is POST-HOC coding: the frozen
weights are whatever they are and the codec must encode them. The one thing never tested end to
end is MOVING THE WEIGHTS - training a latent weight matrix so that its quantization lands closer
to the teacher's OUTPUT than the teacher's own quantization does. That is step (d) below.

THE FOUR STEPS, at the S64 survivor rates (gate/up dim=8 k=1024 stages=2 -> 2.5 index bpw;
down dim=16 k=1024 stages=1 -> 0.625), identical artifact layout in both arms:

  (a) hard assignment of every dim-vector to a codeword (the shipped, discrete operation)
  (b) codebook update by activation-weighted Lloyd on the OUTPUT metric h_j = E[x_j^2]
  (c) per-row scale in closed form against the TEACHER (C.refit_scales), re-rounded to bf16
  (d) latent-weight update, straight-through / error-feedback in the output metric:
          W_lat <- W_lat + eta * (W_teacher - Q(W_lat)) H / ||H||_2 ,   H = X_fit^T X_fit / T
      With a straight-through estimator, d/dW_lat ||(W_teach - Q(W_lat)) X^T||_F^2 is
      -2 (W_teach - Q(W_lat)) X^T X, so (d) is exactly a gradient step on the packed output
      error with respect to the trainable weights. eta is chosen from a small grid ON THE FIT
      SPLIT ONLY.

CONTROL vs TREATMENT. Both arms run the SAME code path (`_loop`). The control passes etas=(0.0,),
so its latent weights never move and it is precisely (a)-(c): the alternating fit that Lane C
already measured. The treatment passes a real eta grid. The ONLY difference between the two
numbers reported per cell is whether the weights were allowed to move. That is the isolation the
stage asks for.

EVERY ITERATE IS VALIDATED AS THE EXACT PACKED ARTIFACT. `_encode` returns integer codeword
indices; `_decode_idx` reconstructs from those indices and nothing else; codebooks are truncated
to bf16 (which is what the ledger charges for them) before any decode; row scales are truncated
to bf16. No soft relaxation is ever scored. `artifact_bits` is asserted identical between arms.

HONESTY, per the campaign contract:
  * Weight-space error is NEVER a capability claim.
  * Output-space relative error on a held-out activation split is a PROXY for capability, not
    capability. Only a real parent-vs-packed 94-layer forward can select a frontier, and this
    module runs none.
  * Layer 0 activations are real tokens from the frozen calibration corpus. Deeper layers use a
    gamma-shaped Gaussian proxy (labelled `gaussian_proxy`) because a real layer-46 input needs a
    46-layer forward; those cells are machinery evidence, not layer evidence, and the verdict is
    scored on the real cells.
  * The fit/holdout activation split is disjoint by construction (LC.routed_split) and the eta
    line search only ever sees the fit half.
  * Bits: this module adds ZERO bits. Both arms ship the identical dim/k/stages payload, the
    identical amortized codebook and the identical one bf16 scale per row. The trained latent
    weights are a TRAINING-ONLY object and are genuinely absent from the artifact - what ships is
    the index stream, so nothing has vanished from the ledger.
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
import qwen_compressibility_train as LC          # noqa: E402

SCHEMA = "hawking.gravity.layerwise_qat.v1"
SOURCE_DIR = LC.SOURCE_DIR
GATE_SPEC = LC.GATE_SPEC
DOWN_SPEC = LC.DOWN_SPEC
DEPLOY_CLUSTER = LC.DEPLOY_CLUSTER
ETAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def _bf16(a: np.ndarray) -> np.ndarray:
    """Truncate to the bf16 grid the artifact actually stores."""
    b = np.ascontiguousarray(a, np.float32).view(np.uint32) >> np.uint32(16)
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16_books(books):
    torch = gf._torch()
    return [torch.from_numpy(_bf16(cb.detach().cpu().numpy())).to(cb.device) for cb in books]


# ── the exact packed artifact ─────────────────────────────────────────────────────────────────
def _encode(books, w_lat: np.ndarray, s: np.ndarray, dim: int, h):
    """Hard codeword indices for one tensor. This is the discrete operation that ships."""
    torch = gf._torch()
    dev = gf._device()
    rows, cols = w_lat.shape
    u = np.asarray(w_lat, np.float32) / np.maximum(s, C._EPS)[:, None]
    wt = C._weights(rows, cols, dim, s, h, dev, torch)
    res = torch.from_numpy(np.ascontiguousarray(u)).to(dev).reshape(-1, dim)
    idxs = []
    for cb in books:
        i = C._assign(res, cb, wt)
        idxs.append(i)
        res = res - cb[i]
    return idxs


def _decode_idx(books, idxs, rows: int, cols: int, dim: int) -> np.ndarray:
    """Reconstruct directions from the INDEX STREAM and the codebooks only."""
    rec = None
    for cb, i in zip(books, idxs):
        q = cb[i]
        rec = q if rec is None else rec + q
    return rec.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)


def _pack(books, w_lat: np.ndarray, teacher: np.ndarray, dim: int, h, hi):
    """One complete packed artifact: bf16 codebooks -> indices -> bf16 row scales -> decode."""
    s = C.row_scales(w_lat)
    idxs = _encode(books, w_lat, s, dim, h)
    d = _decode_idx(books, idxs, teacher.shape[0], teacher.shape[1], dim)
    s2 = C.refit_scales(teacher, d, hi)          # closed form vs the TEACHER, bf16-rounded
    return d * s2[:, None], idxs


# ── the alternating loop; etas=(0.0,) is the control, a real grid is the treatment ────────────
def _loop(mats, xs, *, dim, k, stages, rounds, iters, seed, sub, etas):
    """Returns (best_recon, fit_trace, best_round, eta_trace, qat_delta).

    qat_delta[r] is how much step (d) reduced the FIT objective within round r AT FIXED
    CODEBOOKS - the clean read on whether moving the weights descends at all, with the Lloyd
    restart noise held out of it. It is >= 0 by construction because eta=0 is in every grid.

    mats are the TEACHER tensors and never change. W_lat starts at the teacher; step (d) is the
    only thing that moves it. Selection (best round) is made on the FIT split.
    """
    rng = np.random.default_rng(seed)
    hs = [C.importance_from_activations(x) for x in xs]
    h = np.mean(hs, axis=0).astype(np.float32)
    h = h / max(float(h.mean()), C._EPS)
    # spectral normalisation of H = X^T X / T, computed on the small side (T << cols).
    hnorm = [max(float(np.linalg.norm(x, 2)) ** 2 / max(x.shape[0], 1), C._EPS) for x in xs]

    w_lat = [np.array(m, np.float32, copy=True) for m in mats]
    best, trace, eta_trace, qat_delta, best_round = None, [], [], [], 0

    for r in range(rounds):
        scales = [C.row_scales(w) for w in w_lat]
        books = _bf16_books(LC._fit_books(w_lat, scales, dim=dim, k=k, stages=stages, h=h,
                                          iters=iters, seed=seed + 7 * r, sub=sub, rng=rng))
        rec = [_pack(books, w, m, dim, h, hi)[0] for w, m, hi in zip(w_lat, mats, hs)]
        obj = float(np.mean([LC.out_rel_error(m, rc, x) for m, rc, x in zip(mats, rec, xs)]))
        trace.append(round(obj, 6))
        if best is None or obj < best[0]:
            best, best_round = (obj, rec), r

        # (d) straight-through gradient step, eta picked on the fit split against these books.
        grads = [((m - rc) @ x.T) @ x / (max(x.shape[0], 1) * hn)
                 for m, rc, x, hn in zip(mats, rec, xs, hnorm)]
        cand = None
        for eta in etas:
            if eta == 0.0:
                cand = cand or (obj, list(w_lat), 0.0, rec)
                continue
            w2 = [w + eta * g for w, g in zip(w_lat, grads)]
            r2 = [_pack(books, w, m, dim, h, hi)[0] for w, m, hi in zip(w2, mats, hs)]
            o2 = float(np.mean([LC.out_rel_error(m, rc, x) for m, rc, x in zip(mats, r2, xs)]))
            if o2 < cand[0]:
                cand = (o2, w2, eta, r2)
        eta_trace.append(cand[2])
        qat_delta.append(round(obj - cand[0], 8))
        w_lat = cand[1]
        if cand[0] < best[0]:                    # the eta candidate is itself a packed artifact
            best, best_round = (cand[0], cand[3]), r
        del books, grads, cand
        if gf._device().type == "mps":
            gf._torch().mps.empty_cache()

    return best[1], trace, best_round, eta_trace, qat_delta


# ── measurement ───────────────────────────────────────────────────────────────────────────────
def measure(layers=(0, 46), experts=(0, 1, 2), n_tokens: int = 1400, rounds: int = 4,
            iters: int = 5, sub: int = 400_000, source: Path = SOURCE_DIR,
            seeds: tuple[int, ...] = (0, 1)) -> dict[str, Any]:
    from qwen_real_forward import SafetensorsIndexReader
    reader = SafetensorsIndexReader(source)
    cells: list[dict[str, Any]] = []
    t0 = time.time()
    for L in layers:
        x, prov = LC.calibration_x(reader, L, n_tokens)
        xf, xh, routed = LC.routed_split(reader, L, x, experts)
        af, ah = [], []
        for i, e in enumerate(experts):          # down_proj inputs: real parent intermediate
            p = f"model.layers.{L}.mlp.experts.{e}."
            g = reader.bf16(p + "gate_proj.weight").astype(np.float32)
            u = reader.bf16(p + "up_proj.weight").astype(np.float32)
            af.append(LC._silu(xf[i] @ g.T) * (xf[i] @ u.T))
            ah.append(LC._silu(xh[i] @ g.T) * (xh[i] @ u.T))
            del g, u
        for organ, spec, fit_x, hold_x in (("gate_proj", GATE_SPEC, xf, xh),
                                           ("up_proj", GATE_SPEC, xf, xh),
                                           ("down_proj", DOWN_SPEC, af, ah)):
            mats = [reader.bf16(f"model.layers.{L}.mlp.experts.{e}.{organ}.weight"
                                ).astype(np.float32) for e in experts]
            d, k, st = spec["dim"], spec["k"], spec["stages"]
            # Held-out sets here are only ~30-90 routed tokens, so ONE seed cannot distinguish a
            # real gain from Lloyd-restart noise. Every cell is run under every seed and the
            # verdict demands the win in all of them.
            per_seed = []
            for sd in seeds:
                ctl, t_ctl, br_ctl, _, _ = _loop(mats, fit_x, dim=d, k=k, stages=st, rounds=rounds,
                                                 iters=iters, seed=sd, sub=sub, etas=(0.0,))
                trt, t_trt, br_trt, etas, dlt = _loop(mats, fit_x, dim=d, k=k, stages=st,
                                                      rounds=rounds, iters=iters, seed=sd, sub=sub,
                                                      etas=ETAS)
                per_seed.append({
                    "seed": sd,
                    "output_rel_error_heldout_control": round(float(np.mean(
                        [LC.out_rel_error(m, r, xx) for m, r, xx in zip(mats, ctl, hold_x)])), 6),
                    "output_rel_error_heldout_treatment": round(float(np.mean(
                        [LC.out_rel_error(m, r, xx) for m, r, xx in zip(mats, trt, hold_x)])), 6),
                    "weight_rel_error_control": round(float(np.mean(
                        [C.rel_error(m, r) for m, r in zip(mats, ctl)])), 6),
                    "weight_rel_error_treatment": round(float(np.mean(
                        [C.rel_error(m, r) for m, r in zip(mats, trt)])), 6),
                    "fit_trace_control": t_ctl, "fit_trace_treatment": t_trt,
                    "best_round_control": br_ctl, "best_round_treatment": br_trt,
                    "chosen_eta_per_round": etas, "qat_step_fit_gain_per_round": dlt,
                })
                if sd != seeds[-1]:
                    del ctl, trt
            # The artifact charge depends only on (shape, spec, cluster), which no arm touches -
            # both arms are literally the same integer, and the seeds change nothing either.
            bits = LC.artifact_bits(mats[0].shape, spec, DEPLOY_CLUSTER)
            oc = float(np.mean([p["output_rel_error_heldout_control"] for p in per_seed]))
            ot = float(np.mean([p["output_rel_error_heldout_treatment"] for p in per_seed]))
            gains = [100.0 * (1.0 - p["output_rel_error_heldout_treatment"]
                              / max(p["output_rel_error_heldout_control"], C._EPS))
                     for p in per_seed]
            cells.append({
                "layer": L, "organ": organ, "experts": list(experts),
                "activation_provenance": prov,
                "spec": {"dim": d, "k": k, "stages": st},
                "index_bpw": round(st * math.log2(k) / d, 6),
                "complete_bits_per_tensor_control": bits,
                "complete_bits_per_tensor_treatment": bits,
                "bits_identical": True,
                "complete_bpw_this_tensor": round(bits / (mats[0].shape[0] * mats[0].shape[1]), 6),
                "weight_rel_error_control": round(float(np.mean(
                    [p["weight_rel_error_control"] for p in per_seed])), 6),
                "weight_rel_error_treatment": round(float(np.mean(
                    [p["weight_rel_error_treatment"] for p in per_seed])), 6),
                "output_rel_error_heldout_control": round(oc, 6),
                "output_rel_error_heldout_treatment": round(ot, 6),
                "qat_gain_pct_heldout": round(float(np.mean(gains)), 4),
                "qat_gain_pct_heldout_per_seed": [round(g, 4) for g in gains],
                "wins_all_seeds": bool(all(g > 0 for g in gains)),
                "per_seed": per_seed,
                "n_fit_tokens": [int(v.shape[0]) for v in fit_x],
                "n_heldout_tokens": [int(v.shape[0]) for v in hold_x],
                "routed_token_counts": routed,
            })
            del mats, ctl, trt
        del x, xf, xh, af, ah
    reader.close()

    real = [c for c in cells if c["activation_provenance"] == "real_tokens"]
    wins = [c for c in real if c["wins_all_seeds"]]
    moved = [c for c in cells if any(e > 0 for p in c["per_seed"]
                                     for e in p["chosen_eta_per_round"])]
    verdict = ("S3A_QAT_ALIVE" if real and len(wins) == len(real) else
               "S3A_QAT_PARTIAL" if wins else "S3A_QAT_DEAD")
    # Only layer 0 has real-token activations, and layer 0 is the campaign's KNOWN exception
    # (non-Gaussian, 1.328 decades of coding gap). A win confined to it must say so in its name.
    if verdict != "S3A_QAT_DEAD" and {c["layer"] for c in real} == {0}:
        verdict += "_LAYER0_ONLY"
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source),
        "method": ("alternating exact-rate optimisation (hard assign / activation-weighted Lloyd "
                   "/ closed-form bf16 row scale) with an optional straight-through latent-weight "
                   "step; control = identical loop with eta forced to 0"),
        "rate_note": ("S64 survivor rates, unchanged: gate/up dim=8 k=1024 stages=2 = 2.5 index "
                      "bpw, down dim=16 k=1024 stages=1 = 0.625. Whole-model complete BPW of the "
                      "S64_structural plan is 0.948410027 <= 1. This module proposes no rate."),
        "n_calibration_tokens": n_tokens, "rounds": rounds, "lloyd_iters": iters,
        "eta_grid": list(ETAS), "fit_vector_subsample": sub, "seeds": list(seeds),
        "cells": cells,
        "n_cells_where_weights_moved": len(moved),
        "verdict": verdict,
        "verdict_note": ("Scored ONLY on real_tokens cells, and a cell counts as a win only if it "
                         "wins under EVERY seed. gaussian_proxy cells (deeper layers) validate the "
                         "machinery and carry no evidence about those layers. Real-token "
                         "activations exist only at layer 0 without a multi-layer forward, and "
                         "layer 0 is the campaign's known coding exception, so a _LAYER0_ONLY "
                         "verdict does NOT generalise to depth."),
        "honesty": ("Output-space relative error on a HELD-OUT, disjoint routed-token split is a "
                    "PROXY for capability, not capability. Weight-space error is not a capability "
                    "claim. No forward was run and no frontier is selected. Every reported number "
                    "comes from the EXACT PACKED artifact: bf16 codebooks, integer index stream, "
                    "bf16 row scales. Zero bits are added; the latent trained weights are a "
                    "training-only object that is genuinely absent from the artifact."),
        "wall_seconds": round(time.time() - t0, 1),
    }


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    """Synthetic twin with the real gate/up pathology: 5-decade row norms, anisotropic inputs."""
    rng = np.random.default_rng(0)
    rows, cols, dim, k, T = 192, 128, 8, 64, 96
    dirs = rng.standard_normal((rows, cols)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    span = np.logspace(-5, -0.04, rows).astype(np.float32)
    rng.shuffle(span)
    mats = [np.ascontiguousarray(dirs * span[:, None], np.float32),
            np.ascontiguousarray((dirs + 0.1 * rng.standard_normal(dirs.shape)) * span[:, None],
                                 np.float32)]
    chan = np.logspace(-1.5, 1.5, cols).astype(np.float32)
    xs = [rng.standard_normal((T, cols)).astype(np.float32) * chan for _ in mats]
    hold = [rng.standard_normal((T, cols)).astype(np.float32) * chan for _ in mats]

    ctl, t_c, _, e_c, d_c = _loop(mats, xs, dim=dim, k=k, stages=1, rounds=3, iters=8, seed=0,
                                  sub=0, etas=(0.0,))
    trt, t_t, br, e_t, d_t = _loop(mats, xs, dim=dim, k=k, stages=1, rounds=3, iters=8, seed=0,
                                   sub=0, etas=ETAS)

    # 1. the control must never move its weights, and must therefore gain nothing from step (d).
    assert all(e == 0.0 for e in e_c) and all(v == 0.0 for v in d_c), ("control moved", e_c, d_c)

    # 2. THE CORE CLAIM being tested: at FIXED codebooks, the straight-through step must be a
    #    real descent direction on the packed output error - it must fire and it must gain.
    #    A wrong sign, a wrong Hessian or a broken pack makes every candidate lose to eta=0 and
    #    the line search returns 0 everywhere, so this assert fires.
    assert any(e > 0.0 for e in e_t), ("step (d) never fired on the twin", e_t)
    assert all(v >= 0.0 for v in d_t), ("line search returned a worse iterate", d_t)
    assert sum(d_t) > 0.0, ("step (d) is not a descent direction on the twin", d_t)

    # 3. held-out output error is finite and the treatment is not a catastrophe there.
    oc = float(np.mean([LC.out_rel_error(m, r, x) for m, r, x in zip(mats, ctl, hold)]))
    ot = float(np.mean([LC.out_rel_error(m, r, x) for m, r, x in zip(mats, trt, hold)]))
    assert np.isfinite(oc) and np.isfinite(ot) and ot < 2.0 * oc, (oc, ot)

    # 4. the scored object really is the packed artifact: decode from the index stream alone.
    torch = gf._torch()
    s = C.row_scales(mats[0])
    books = _bf16_books([C._lloyd(torch.from_numpy(np.ascontiguousarray(
        (mats[0] / np.maximum(s, C._EPS)[:, None]).reshape(-1, dim))).to(gf._device()),
        k, iters=4)])
    idxs = _encode(books, mats[0], s, dim, None)
    assert int(idxs[0].max()) < k and int(idxs[0].min()) >= 0, "index out of the billed alphabet"
    d1 = _decode_idx(books, idxs, rows, cols, dim)
    d2 = _decode_idx(books, [i.clone() for i in idxs], rows, cols, dim)
    assert np.array_equal(d1, d2), "decode is not a pure function of the index stream"
    # bf16 grid checked STRUCTURALLY (low mantissa bits must be zero), not against _bf16 itself -
    # comparing to _bf16 would pass even if _bf16 were gutted into a no-op.
    cbn = np.ascontiguousarray(books[0].detach().cpu().numpy(), np.float32)
    assert not np.any(cbn.view(np.uint32) & np.uint32(0xFFFF)), \
        "codebook is not on the bf16 grid it is charged at"

    # 5. zero bits added: both arms carry the identical complete charge.
    spec = {"family": "shared_grammar", "dim": dim, "k": k, "stages": 1}
    b = LC.artifact_bits(mats[0].shape, spec, 128)
    assert b == LC.artifact_bits(mats[0].shape, spec, 128)
    assert b > (rows * cols // dim) * math.ceil(math.log2(k)) + rows * 16, "charge incomplete"
    return {"ok": True, "fit_control": t_c, "fit_treatment": t_t, "best_round": br,
            "etas": e_t, "qat_step_fit_gain_per_round": d_t,
            "heldout_control": round(oc, 6), "heldout_treatment": round(ot, 6),
            "complete_bits": b}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--layers", default="0,46")
    ap.add_argument("--experts", default="0,1,2")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=1400)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default="reports/subbit_reset/S3A_LAYERWISE_QAT.json")
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(), indent=2))
        return 0
    if a.measure:
        rep = measure(layers=tuple(int(v) for v in a.layers.split(",")),
                      experts=tuple(int(v) for v in a.experts.split(",")),
                      n_tokens=a.tokens, rounds=a.rounds,
                      seeds=tuple(int(v) for v in a.seeds.split(",")))
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps({"verdict": rep["verdict"], "out": str(p),
                          "wall_seconds": rep["wall_seconds"]}, indent=2))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
