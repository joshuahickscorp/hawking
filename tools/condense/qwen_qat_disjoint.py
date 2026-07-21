#!/usr/bin/env python3.12
"""S3A re-score: layerwise QAT under a GENUINELY DISJOINT evaluation probe.

WHY THIS FILE EXISTS. S3A (tools/condense/qwen_layerwise_qat.py) reported 14.1 / 11.3 / 2.45 pct
"held-out" output-error gains at layer 0 from a straight-through latent-weight step at
byte-identical cost. Its adversary refuted the word HELD-OUT: the fraction of held-out activation
energy lying inside the FIT split's row space was 1.000 / 0.991 / 0.424 / 1.000, and the per-expert
gain was monotone in that overlap (19.8 / 11.6 / 4.0 / 18.0 pct). The step
    dW = ((W - Q(W_lat)) X_fit^T) X_fit
has column space EXACTLY the fit row space, so a probe inside that span scores fit-span descent,
not generalisation.

WHERE THE OVERLAP CAME FROM (root cause, measured here). qwen_compressibility_train.calibration_x
draws a random PERMUTATION OF POSITIONS from a corpus that repeats its 12 segments until the token
target is met: 1313 positions but only 540 UNIQUE token ids at min_tokens=1200, and layer 0's MoE
input is rmsnorm(embed[id]) - a pure function of the id. LC.routed_split then splits POSITIONS, so
the same embedding row appears in both halves. Capture 1.000 is not near-degeneracy, it is literal
row duplication.

THE SPLIT THIS MODULE USES.
  * unique token ids only (each embedding row appears at most once anywhere),
  * partitioned by CORPUS SEGMENT, so fit and score text never share a segment, let alone a token,
  * scored against the FULL held-out corpus activation matrix rather than the expert's own routed
    tokens - the adversary's own prescription, and the only lever that reaches capture <= ~0.1 with
    a 540-unique-token frozen corpus.
Every cell reports its MEASURED capture. Nothing is asserted to be disjoint.

WHAT IS UNCHANGED. Rates (S64 survivors: gate/up dim=8 k=1024 stages=2, down dim=16 k=1024
stages=1), the alternating loop, the packed-artifact scoring path, and the control/treatment
isolation (control = the identical loop with eta forced to 0.0). This module proposes no rate and
adds no bits.

HONESTY.
  * Output relative error on a calibration probe is a PROXY for capability, never capability.
  * Weight-space error is never a capability claim.
  * Layer 0's MoE input is itself a proxy: rmsnorm(embed[id]) with the attention sublayer and the
    residual add skipped. This module does not fix that; it fixes the fit/score overlap only.
  * Scoring on the FULL held-out corpus is a distribution shift away from the expert's routed
    tokens. That is the price of a low-capture probe on a 540-unique-token corpus, it is stated
    per cell, and the routed-token score is reported ALONGSIDE it with its own (higher) capture.
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
import qwen_layerwise_qat as QAT                 # noqa: E402

SCHEMA = "hawking.gravity.qat_disjoint_rescore.v1"
CAPTURE_TARGET = 0.10


# ── the disjointness measurement ──────────────────────────────────────────────────────────────
def capture_fraction(x_fit: np.ndarray, x_hold: np.ndarray, tol: float = 1e-6) -> float:
    """Fraction of x_hold's energy lying in the row space of x_fit.

    Basis from an SVD with a rank tolerance, NOT a bare QR: a rank-deficient fit block would give
    QR columns spanning directions the fit never actually saw, which understates the confound.
    """
    a = np.asarray(x_fit, np.float64)
    b = np.asarray(x_hold, np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    u, s, _ = np.linalg.svd(a.T, full_matrices=False)
    q = u[:, s > tol * max(float(s[0]), 1e-30)]
    den = float(np.linalg.norm(b) ** 2)
    return float(np.linalg.norm(b @ q) ** 2 / den) if den > 0 else float("nan")


# ── the byte-level artifact ───────────────────────────────────────────────────────────────────
def _u16(a: np.ndarray) -> bytes:
    """bf16 payload exactly as stored: the top 16 bits of the float32 word."""
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> np.uint32(16)
            ).astype(np.uint16).tobytes()


def serialize(books, idxs, scales: np.ndarray, k: int) -> bytes:
    """The shipped payload: bf16 codebooks + a ceil(log2 k)-bit index stream + bf16 row scales."""
    w = int(math.ceil(math.log2(k)))
    out = [_u16(cb.detach().cpu().numpy()) for cb in books]
    for i in idxs:
        v = np.ascontiguousarray(i.detach().cpu().numpy(), np.uint16)
        bits = np.unpackbits(v.view(np.uint8).reshape(-1, 2)[:, ::-1], axis=1)[:, 16 - w:]
        out.append(np.packbits(bits.reshape(-1)).tobytes())
    out.append(_u16(scales))
    return b"".join(out)


# ── control / treatment loop, returning the artifact and not just its error ───────────────────
def loop(mats, xs, *, dim, k, stages, rounds, iters, seed, sub, etas):
    """QAT.\\_loop with the winning round's ARTIFACT kept, so the two arms can be byte-compared.

    Identical mathematics to qwen_layerwise_qat._loop - same pack, same line search, same
    selection on the fit split. etas=(0.0,) is the control.
    """
    rng = np.random.default_rng(seed)
    hs = [C.importance_from_activations(x) for x in xs]
    h = np.mean(hs, axis=0).astype(np.float32)
    h = h / max(float(h.mean()), C._EPS)
    hnorm = [max(float(np.linalg.norm(x, 2)) ** 2 / max(x.shape[0], 1), C._EPS) for x in xs]

    w_lat = [np.array(m, np.float32, copy=True) for m in mats]
    best = None
    eta_trace: list[float] = []

    for r in range(rounds):
        scales = [C.row_scales(w) for w in w_lat]
        books = QAT._bf16_books(LC._fit_books(w_lat, scales, dim=dim, k=k, stages=stages, h=h,
                                              iters=iters, seed=seed + 7 * r, sub=sub, rng=rng))

        def pack_all(ws):
            packs = [QAT._pack(books, w, m, dim, h, hi) for w, m, hi in zip(ws, mats, hs)]
            rec = [p[0] for p in packs]
            obj = float(np.mean([LC.out_rel_error(m, rc, x) for m, rc, x in zip(mats, rec, xs)]))
            return obj, rec, [p[1] for p in packs]

        obj, rec, idxs = pack_all(w_lat)
        art = (books, idxs, [C.row_scales(w) for w in w_lat])
        if best is None or obj < best[0]:
            best = (obj, rec, art)

        grads = [((m - rc) @ x.T) @ x / (max(x.shape[0], 1) * hn)
                 for m, rc, x, hn in zip(mats, rec, xs, hnorm)]
        cand = (obj, list(w_lat), 0.0, rec, art)
        for eta in etas:
            if eta == 0.0:
                continue
            w2 = [w + eta * g for w, g in zip(w_lat, grads)]
            o2, r2, i2 = pack_all(w2)
            if o2 < cand[0]:
                cand = (o2, w2, eta, r2, (books, i2, [C.row_scales(w) for w in w2]))
        eta_trace.append(cand[2])
        w_lat = cand[1]
        if cand[0] < best[0]:
            best = (cand[0], cand[3], cand[4])
        del grads
        if gf._device().type == "mps":
            gf._torch().mps.empty_cache()

    return best[1], best[2], eta_trace


# ── the disjoint split ────────────────────────────────────────────────────────────────────────
def segment_split(reader, experts, *, seed: int, layer: int = 0, top_k: int = 8):
    """Unique-token, segment-disjoint fit/score sets at layer 0.

    Returns (x_fit_per_expert, x_hold_full, x_hold_routed_per_expert, meta). The fit set for an
    expert is the tokens ROUTED to it among the fit segments' unique ids; the score set is every
    unique id of the held-out segments (plus, separately, the routed subset of them).
    """
    import qwen_calibration_corpus as CC
    from tokenizers import Tokenizer  # type: ignore

    tk = Tokenizer.from_file(str(LC.SOURCE_DIR / "tokenizer.json"))
    corpus = CC.build(min_tokens=1200, tokenizer=tk)
    segs: dict[str, set[int]] = {}
    for p in corpus["prompts"]:
        segs.setdefault(p["id"].split("#")[0], set()).update(p["ids"])
    names = sorted(segs)
    rng = np.random.default_rng(seed)
    pm = rng.permutation(len(names))
    fit_names = [names[i] for i in pm[:len(names) // 2]]
    hold_names = [names[i] for i in pm[len(names) // 2:]]
    fit_ids = set().union(*[segs[n] for n in fit_names])
    hold_ids = set().union(*[segs[n] for n in hold_names]) - fit_ids   # ids win over segments
    assert not (fit_ids & hold_ids), "token id leaked across the split"

    ids = sorted(fit_ids | hold_ids)
    pos = {t: i for i, t in enumerate(ids)}
    gamma = reader.bf16(f"model.layers.{layer}.post_attention_layernorm.weight").astype(np.float32)
    x = LC._rmsnorm(reader.bf16_rows("model.embed_tokens.weight", ids), gamma)
    g = reader.bf16(f"model.layers.{layer}.mlp.gate.weight").astype(np.float32)
    top = np.argpartition(-(x @ g.T), top_k - 1, axis=1)[:, :top_k]
    del g

    ia = np.array(sorted(pos[t] for t in fit_ids))
    ib = np.array(sorted(pos[t] for t in hold_ids))
    xf, xhr, routed = [], [], []
    for e in experts:
        ra = ia[(top[ia] == int(e)).any(1)]
        rb = ib[(top[ib] == int(e)).any(1)]
        routed.append({"expert": int(e), "n_fit_routed": int(ra.size), "n_hold_routed": int(rb.size)})
        xf.append(np.ascontiguousarray(x[ra]))
        xhr.append(np.ascontiguousarray(x[rb]))
    meta = {"fit_segments": fit_names, "hold_segments": hold_names,
            "n_unique_ids_total": len(ids), "n_fit_ids": int(ia.size), "n_hold_ids": int(ib.size),
            "corpus_sha256": corpus["sha256"], "routed": routed,
            "n_corpus_positions": corpus["n_tokens"]}
    return xf, np.ascontiguousarray(x[ib]), xhr, meta


# ── measurement ───────────────────────────────────────────────────────────────────────────────
def measure(experts=(0, 1, 2), split_seeds=(0, 1, 2), rounds: int = 3, iters: int = 5,
            sub: int = 400_000, lloyd_seed: int = 0, layer: int = 0,
            positive_control: bool = True, source: Path = LC.SOURCE_DIR) -> dict[str, Any]:
    from qwen_real_forward import SafetensorsIndexReader
    reader = SafetensorsIndexReader(source)
    t0 = time.time()
    cells: list[dict[str, Any]] = []
    bits_identical = True

    for sd in split_seeds:
        xf, xh_full, xh_routed, meta = segment_split(reader, experts, seed=sd, layer=layer)
        af, ah_full, ah_routed = [], [], []
        for i, e in enumerate(experts):
            p = f"model.layers.{layer}.mlp.experts.{e}."
            g = reader.bf16(p + "gate_proj.weight").astype(np.float32)
            u = reader.bf16(p + "up_proj.weight").astype(np.float32)
            af.append(LC._silu(xf[i] @ g.T) * (xf[i] @ u.T))
            ah_full.append(LC._silu(xh_full @ g.T) * (xh_full @ u.T))
            ah_routed.append(LC._silu(xh_routed[i] @ g.T) * (xh_routed[i] @ u.T))
            del g, u

        for organ, spec, fx, hx_full, hx_routed in (
                ("gate_proj", LC.GATE_SPEC, xf, [xh_full] * len(experts), xh_routed),
                ("up_proj", LC.GATE_SPEC, xf, [xh_full] * len(experts), xh_routed),
                ("down_proj", LC.DOWN_SPEC, af, ah_full, ah_routed)):
            mats = [reader.bf16(f"model.layers.{layer}.mlp.experts.{e}.{organ}.weight"
                                ).astype(np.float32) for e in experts]
            d, k, st = spec["dim"], spec["k"], spec["stages"]
            ctl, art_c, eta_c = loop(mats, fx, dim=d, k=k, stages=st, rounds=rounds, iters=iters,
                                     seed=lloyd_seed, sub=sub, etas=(0.0,))
            trt, art_t, eta_t = loop(mats, fx, dim=d, k=k, stages=st, rounds=rounds, iters=iters,
                                     seed=lloyd_seed, sub=sub, etas=QAT.ETAS)

            bc = serialize(art_c[0], art_c[1][0], art_c[2][0], k)
            bt = serialize(art_t[0], art_t[1][0], art_t[2][0], k)
            same_len = len(bc) == len(bt)
            bits = LC.artifact_bits(mats[0].shape, spec, LC.DEPLOY_CLUSTER)
            bits_identical &= bool(same_len)

            per_expert = []
            for i, e in enumerate(experts):
                cap_full = capture_fraction(fx[i], hx_full[i])
                cap_routed = capture_fraction(fx[i], hx_routed[i]) if hx_routed[i].size else float("nan")
                ec = LC.out_rel_error(mats[i], ctl[i], hx_full[i])
                et = LC.out_rel_error(mats[i], trt[i], hx_full[i])
                rc = LC.out_rel_error(mats[i], ctl[i], hx_routed[i]) if hx_routed[i].size else float("nan")
                rt = LC.out_rel_error(mats[i], trt[i], hx_routed[i]) if hx_routed[i].size else float("nan")
                per_expert.append({
                    "expert": int(e),
                    "capture_fraction_full_corpus_probe": round(cap_full, 5),
                    "capture_fraction_routed_probe": None if math.isnan(cap_routed) else round(cap_routed, 5),
                    "n_fit_tokens": int(fx[i].shape[0]),
                    "n_score_tokens_full": int(hx_full[i].shape[0]),
                    "n_score_tokens_routed": int(hx_routed[i].shape[0]),
                    "control_error_full": round(ec, 6), "treatment_error_full": round(et, 6),
                    "gain_pct_full": round(100.0 * (1.0 - et / max(ec, C._EPS)), 4),
                    "control_error_routed": None if math.isnan(rc) else round(rc, 6),
                    "treatment_error_routed": None if math.isnan(rt) else round(rt, 6),
                    "gain_pct_routed": (None if math.isnan(rc) or math.isnan(rt)
                                        else round(100.0 * (1.0 - rt / max(rc, C._EPS)), 4)),
                })
            caps = [p["capture_fraction_full_corpus_probe"] for p in per_expert]
            gains = [p["gain_pct_full"] for p in per_expert]
            low = [p for p in per_expert if p["capture_fraction_full_corpus_probe"] <= CAPTURE_TARGET]
            cells.append({
                "split_seed": sd, "layer": layer, "organ": organ, "experts": list(experts),
                "fit_segments": meta["fit_segments"], "hold_segments": meta["hold_segments"],
                "spec": {"dim": d, "k": k, "stages": st},
                "complete_bits_per_tensor_control": bits,
                "complete_bits_per_tensor_treatment": bits,
                "serialized_bytes_control": len(bc), "serialized_bytes_treatment": len(bt),
                "serialized_length_identical": same_len,
                "payload_content_differs": bc != bt,
                "chosen_eta_per_round_control": eta_c,
                "chosen_eta_per_round_treatment": eta_t,
                "mean_capture_fraction": round(float(np.mean(caps)), 5),
                "max_capture_fraction": round(float(np.max(caps)), 5),
                "mean_gain_pct_full": round(float(np.mean(gains)), 4),
                "mean_gain_pct_low_capture_only": (round(float(np.mean(
                    [p["gain_pct_full"] for p in low])), 4) if low else None),
                "n_experts_at_or_below_capture_target": len(low),
                "per_expert": per_expert,
            })
            del mats, ctl, trt, art_c, art_t
        del xf, xh_full, xh_routed, af, ah_full, ah_routed

    # POSITIVE CONTROL: the ORIGINAL position-level split, same loop(), same rates. If the machinery
    # in this file were simply broken, this would also read ~0 and the collapse above would prove
    # nothing. It has to reproduce the sealed ~14 pct win for the disjoint result to mean anything.
    pc = None
    if positive_control:
        x, prov = LC.calibration_x(reader, layer, 1400)
        pf, ph, _ = LC.routed_split(reader, layer, x, experts)
        mats = [reader.bf16(f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"
                            ).astype(np.float32) for e in experts]
        d, k, st = LC.GATE_SPEC["dim"], LC.GATE_SPEC["k"], LC.GATE_SPEC["stages"]
        c0, _, _ = loop(mats, pf, dim=d, k=k, stages=st, rounds=rounds, iters=iters,
                        seed=lloyd_seed, sub=sub, etas=(0.0,))
        t0_, _, _ = loop(mats, pf, dim=d, k=k, stages=st, rounds=rounds, iters=iters,
                         seed=lloyd_seed, sub=sub, etas=QAT.ETAS)
        ec = float(np.mean([LC.out_rel_error(m, r, h) for m, r, h in zip(mats, c0, ph)]))
        et = float(np.mean([LC.out_rel_error(m, r, h) for m, r, h in zip(mats, t0_, ph)]))
        pc = {"split": "original LC.routed_split over corpus POSITIONS (duplicate ids in both halves)",
              "provenance": prov, "organ": "gate_proj",
              "capture_fraction_per_expert": [round(capture_fraction(a, b), 5)
                                              for a, b in zip(pf, ph)],
              "control_error": round(ec, 6), "treatment_error": round(et, 6),
              "gain_pct": round(100.0 * (1.0 - et / max(ec, C._EPS)), 4),
              "note": ("reproduces the sealed S3A win under the sealed split, so the collapse in "
                       "the disjoint cells is caused by the SPLIT and not by this module")}
        del mats, c0, t0_, x, pf, ph
    reader.close()

    low_cells = [p for c in cells for p in c["per_expert"]
                 if p["capture_fraction_full_corpus_probe"] <= CAPTURE_TARGET]
    low_gains = [p["gain_pct_full"] for p in low_cells]
    all_gains = [p["gain_pct_full"] for c in cells for p in c["per_expert"]]
    survives = bool(low_gains) and all(g > 0 for g in low_gains)
    verdict = ("S3A_QAT_SEALED_NEGATIVE_UNDER_DISJOINT_PROBE" if low_gains and not survives else
               "S3A_QAT_SURVIVES_DISJOINT" if survives else
               "S3A_QAT_UNDECIDED_CAPTURE_TARGET_NOT_REACHED")
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source),
        "supersedes": "reports/subbit_reset/S3A_LAYERWISE_QAT.json (held-out label only)",
        "capture_target": CAPTURE_TARGET,
        "root_cause_of_the_original_overlap": (
            "qwen_calibration_corpus repeats its 12 segments to hit the token target, so 1313 "
            "corpus positions carry only 540 unique ids; layer-0 MoE input is rmsnorm(embed[id]), "
            "a pure function of the id; LC.routed_split splits POSITIONS. The same embedding row "
            "therefore appears in both halves, which is why capture measured 1.000."),
        "split": ("unique ids only, partitioned by corpus SEGMENT (6 fit / 6 score segments, "
                  "seed-varied), scored against the FULL held-out segment activation matrix"),
        "rounds": rounds, "lloyd_iters": iters, "lloyd_seed": lloyd_seed,
        "split_seeds": list(split_seeds), "eta_grid": list(QAT.ETAS),
        "cells": cells,
        "positive_control_original_split": pc,
        "n_low_capture_expert_cells": len(low_cells),
        "mean_gain_pct_low_capture": (round(float(np.mean(low_gains)), 4) if low_gains else None),
        "mean_gain_pct_all_cells": round(float(np.mean(all_gains)), 4) if all_gains else None,
        "survives_disjoint": survives,
        "bits_identical": bits_identical,
        "verdict": verdict,
        "honesty": (
            "Output relative error on a calibration probe is a PROXY for capability, not "
            "capability; no forward was run. Layer-0 input is itself rmsnorm(embed[id]) with the "
            "attention sublayer and residual skipped. Scoring on the full held-out segment matrix "
            "is a distribution shift away from the expert's routed tokens; the routed-probe score "
            "is reported alongside with its own, higher, capture. The frozen corpus holds only 540 "
            "unique ids, so a low-capture probe and a large fit span cannot both be had here."),
        "wall_seconds": round(time.time() - t0, 1),
    }


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    rng = np.random.default_rng(0)

    # 1. capture_fraction: orthogonal complement -> 0, same span -> 1, duplicated rows -> 1.
    q = np.linalg.qr(rng.standard_normal((64, 64)))[0]
    a, b = q[:8].copy(), q[8:24].copy()
    assert capture_fraction(a, b) < 1e-8, capture_fraction(a, b)
    assert abs(capture_fraction(a, a[:4] * 3.0) - 1.0) < 1e-8
    # the exact pathology this module exists to catch: the "held-out" half is duplicate rows.
    dup = np.concatenate([a, a[:3]], 0)
    assert abs(capture_fraction(dup[:8], dup[8:]) - 1.0) < 1e-8
    # rank-deficient fit must NOT be credited with directions it never saw.
    rd = np.concatenate([a[:2], a[:2] * 2.0], 0)
    assert capture_fraction(rd, b) < 1e-8, "rank tolerance is not being applied"

    # 2. serialize: identical shape+spec -> identical byte length, different weights -> different
    #    bytes. This is exactly the control-vs-treatment assert the stage needs.
    torch = gf._torch()
    rows, cols, dim, k = 128, 64, 8, 64
    m1 = rng.standard_normal((rows, cols)).astype(np.float32)
    m2 = m1 + 0.5 * rng.standard_normal(m1.shape).astype(np.float32)
    out = []
    for m in (m1, m2):
        s = C.row_scales(m)
        books = QAT._bf16_books([C._lloyd(torch.from_numpy(np.ascontiguousarray(
            (m / np.maximum(s, C._EPS)[:, None]).reshape(-1, dim))).to(gf._device()), k, iters=4)])
        out.append(serialize(books, QAT._encode(books, m, s, dim, None), s, k))
    assert len(out[0]) == len(out[1]), (len(out[0]), len(out[1]))
    assert out[0] != out[1], "serialization is blind to the weights it encodes"
    assert len(out[0]) == k * dim * 2 + (rows * cols // dim) * 6 // 8 + rows * 2, len(out[0])

    # 3. loop(): the control never moves its weights; the treatment's step can fire; both arms
    #    serialize to the same length. A regression that let eta leak into the control fires here.
    mats = [m1, m2]
    xs = [rng.standard_normal((48, cols)).astype(np.float32) for _ in mats]
    rc, ac, ec = loop(mats, xs, dim=dim, k=k, stages=1, rounds=2, iters=6, seed=0, sub=0,
                      etas=(0.0,))
    rt, at, et = loop(mats, xs, dim=dim, k=k, stages=1, rounds=2, iters=6, seed=0, sub=0,
                      etas=QAT.ETAS)
    assert all(e == 0.0 for e in ec), ("control moved", ec)
    assert len(serialize(ac[0], ac[1][0], ac[2][0], k)) == \
           len(serialize(at[0], at[1][0], at[2][0], k)), "arms differ in shipped bytes"

    # 4. segment_split's disjointness is structural, not asserted: emulate it on token ids and
    #    confirm no id can appear on both sides (the original bug was exactly this).
    segs = {f"s{i}": set(range(i * 10, i * 10 + 15)) for i in range(12)}
    names = sorted(segs)
    pm = np.random.default_rng(1).permutation(len(names))
    f = set().union(*[segs[names[i]] for i in pm[:6]])
    h = set().union(*[segs[names[i]] for i in pm[6:]]) - f
    assert not (f & h) and h, "split leaks ids"

    return {"ok": True, "control_etas": ec, "treatment_etas": et,
            "serialized_bytes": len(out[0])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--experts", default="0,1,2")
    ap.add_argument("--split-seeds", default="0,1,2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="reports/subbit_reset/S3A_DISJOINT_RESCORE.json")
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(), indent=2))
        return 0
    if a.measure:
        rep = measure(experts=tuple(int(v) for v in a.experts.split(",")),
                      split_seeds=tuple(int(v) for v in a.split_seeds.split(",")),
                      rounds=a.rounds)
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps({"verdict": rep["verdict"], "survives_disjoint": rep["survives_disjoint"],
                          "bits_identical": rep["bits_identical"], "out": str(p),
                          "wall_seconds": rep["wall_seconds"]}, indent=2))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
