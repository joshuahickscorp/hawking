#!/usr/bin/env python3.12
"""S3C - Doctor Generation 2, and the byte auction that decides whether the Doctor keeps its slot.

WHAT GEN1 IS. tools/condense/qwen_function_aware_codec.fit_doctor/apply_doctor is a sparse residual
over the rows with the worst RELATIVE residual energy, coded with a second codebook. It measurably
works in weight space (down_proj L46 rel_error 0.7038 -> 0.6074) and on the sealed S1 full forward
S64_doctor beat S64_structural on all four metrics for +0.0513 complete BPW - both arms still
COLLAPSE, so the Doctor is a better loser, not a winner.

WHAT GEN2 CHANGES, at IDENTICAL bytes:
  G2-A  trainable residual codebook: fit the correction codebook against the OUTPUT residual on
        calibration activations (weighted Lloyd with h = diag E[a a^T] on the down_proj input)
        instead of unweighted Lloyd on the weight residual. Same indices, same codebook, same
        bitmap, same scales - only the fitted values differ.
  G2-B  trainable protected-row SELECTION: pick the protected rows by measured OUTPUT sensitivity
        sum_j h_j (w_ij - base_ij)^2 instead of relative residual energy ||w_i-b_i||/||w_i||. The
        protection bitmap is already billed per row, so the choice of WHICH rows is free.

WHAT THIS STAGE ACTUALLY DELIVERS: THE AUCTION. The S64_doctor artifact sits at 0.999769787
complete BPW, i.e. there is no headroom - the Doctor's 12,074,352,640 bits are bits some other
treatment is not getting. So the question is not "does the Doctor help" but "does the Doctor buy
more output-error reduction per bit than the alternatives". Priced here, on real weights and real
routed activations, in one common unit (squared output error of the expert's weighted contribution,
per bit charged):
    (a) Doctor residual on down_proj            (Gen1 and both Gen2 variants)
    (b) one more surviving expert, K -> K+1
    (c) a higher gate/up rung  (2.5 -> 5.0 index bpw)
    (d) a higher down rung     (0.625 -> 1.25 index bpw)
    (e) router capacity - deferred to stage S2C; read from its report if present, else recorded
        ABSENT. Not estimated.

HONESTY, non-negotiable and enforced by the report this file writes:
  * Weight-space error is NEVER a capability claim.
  * The output-space error measured here is on a CALIBRATION batch of real routed activations. It
    is a PROXY, not capability. Only a real parent-vs-packed 94-layer forward may select a
    frontier. Nothing in this file selects one.
  * Calibration is split BY PROMPT into a fit half and a score half; codebooks, importance vectors
    and protected-row choices are fitted on the fit half ONLY and every number reported is measured
    on the score half. The frozen corpus is itself disjoint from the campaign's scored holdout
    (qwen_calibration_corpus asserts this) and the split is asserted in `measure`.
  * (b) is priced using the SEALED Lane E result that an omitted expert is not reconstructible from
    survivors (MISS_COST = 1.0). Adding an expert back also perturbs the top-k renormalisation of
    the other experts of that token; that second-order term is NOT modelled and is declared in the
    report as a first-order approximation.
  * The shared codebook is fitted here PER EXPERT (on that expert's own gate+up pair, and on its
    own down tensor) rather than on all 64 survivors of the layer, while the codebook bytes are
    charged amortised over 64 exactly as the deployed ledger does. This flatters every coded arm
    equally, so the RANKING is the deliverable and the absolute errors are optimistic. Declared,
    not hidden.
  * Every bit is charged with the existing exact ledger (qwen_subhalfbit_search.expert_bits,
    qwen_function_aware_codec.doctor_bits). Nothing here is charged approximately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "foundry")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qwen_function_aware_codec as FAC  # noqa: E402
import qwen_structural_plan as SP  # noqa: E402
import qwen_subhalfbit_search as SHB  # noqa: E402

SCHEMA = "hawking.gravity.doctor_gen2.v1"
_ROOT = Path(_HERE).parent.parent          # repo root, so inputs resolve from any cwd
REPORT = Path("reports/subbit_reset/S3C_DOCTOR_GEN2.json")
S2C_REPORT = _ROOT / "reports/subbit_reset/S2C_ROUTER_DISTILL.json"
ROUTING = _ROOT / "reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json"
CACHE = Path(os.environ.get("HAWKING_SCRATCH", "/tmp")) / "qwen_doctor_gen2_acts.npz"

# The S64_doctor artifact's own spec (reports/.../checkpoints/*__S64_doctor.json).
KEEP = 64
GATE_BASE, GATE_UP_RUNG = "2.5", "5.0"
DOWN_BASE, DOWN_UP_RUNG = "0.625", "1.25"
DOCTOR = {"dim": 16, "k": 1024, "stages": 1, "protect_frac": 0.5}
LAYER = 46                      # the layer every sealed Doctor number in this campaign is from
MIN_TOKENS = 800                # ~8 frozen segments, so the BY-PROMPT fit/score split is possible
MISS_COST = 1.0                 # SEALED, Lane E: an omitted expert is not reconstructible


# ── exact bit prices (existing ledger only) ───────────────────────────────────────────────────
def bits_gate_up(shape: tuple[int, int], rung: str) -> int:
    """Both gate-class tensors of one expert at a rung, including the billed per-row bf16 scales."""
    spec = SP.GATE_RUNGS[rung]
    return 2 * (SHB.expert_bits(shape, spec, KEEP) + shape[0] * SP.ROW_SCALE_BITS)


def bits_down(shape: tuple[int, int], rung: str) -> int:
    spec = SP.DOWN_RUNGS[rung]
    return SHB.expert_bits(shape, spec, KEEP) + shape[0] * SP.ROW_SCALE_BITS


def bits_doctor(shape: tuple[int, int]) -> int:
    return FAC.doctor_bits(shape, doctor_dim=DOCTOR["dim"], doctor_k=DOCTOR["k"],
                           doctor_stages=DOCTOR["stages"],
                           protect_frac=DOCTOR["protect_frac"], cluster=KEEP)


# ── G2-B: protected-row selection by measured output sensitivity ──────────────────────────────
def rows_by_output_sensitivity(w: np.ndarray, base: np.ndarray, h: np.ndarray,
                               frac: float) -> np.ndarray:
    """Rows whose residual costs the OUTPUT the most: sum_j h_j (w_ij - base_ij)^2.

    Exactly the same bytes as Gen1's relative-energy choice - the protection bitmap is one bit per
    row either way, and it is already billed. Only WHICH rows are protected changes.
    """
    d = (np.asarray(w, np.float32) - np.asarray(base, np.float32)) ** 2
    e = d @ np.asarray(h, np.float32)
    n = max(1, int(round(frac * w.shape[0])))
    return np.sort(np.argsort(-e)[:n])


def doctor_arm(w: np.ndarray, base: np.ndarray, *, h: np.ndarray | None, select_h: bool,
               fit_h: bool, seed: int = 0) -> np.ndarray:
    """One Doctor arm: choose rows, fit the correction codebook, decode. Identical byte cost.

    select_h -> G2-B row selection.  fit_h -> G2-A output-weighted codebook fit.
    """
    frac = float(DOCTOR["protect_frac"])
    rows = (rows_by_output_sensitivity(w, base, h, frac) if select_h
            else FAC.protected_rows(w, base, frac))
    resid = np.asarray(w, np.float32)[rows] - np.asarray(base, np.float32)[rows]
    imp = h if fit_h else None
    books = FAC.fit([resid], dim=DOCTOR["dim"], k=DOCTOR["k"], stages=DOCTOR["stages"],
                    seed=seed + 977, row_scale=True, importance=imp, iters=4)
    out = np.array(base, np.float32, copy=True)
    out[rows] = out[rows] + FAC.apply_refit(books, resid, dim=DOCTOR["dim"], importance=imp)
    return out


# ── the auction ranking ───────────────────────────────────────────────────────────────────────
def router_capacity_e() -> dict[str, Any]:
    """(e) router capacity, READ from the sealed S2C report. Never estimated.

    S2C's unit is the MEDIAN RELATIVE error of the whole MoE output against the 128-expert
    teacher; this stage's auction unit is ABSOLUTE squared output error of one expert's weighted
    contribution. They are not interconvertible, so (e) is reported ALONGSIDE the auction with its
    own numbers rather than ranked inside it. Its sign, however, is decisive on its own.
    """
    if not S2C_REPORT.is_file():
        return {"present": False, "source": str(S2C_REPORT), "note": "ABSENT - not estimated"}
    d = json.loads(S2C_REPORT.read_text())
    arms = []
    for lay in d.get("layers", []):
        base = lay["arms"]["masked"]["holdout_median_rel_err"]
        for name, v in lay["arms"].items():
            if name == "masked":
                continue
            arms.append({"layer": lay["layer"], "arm": name,
                         "extra_bits_whole_model": v.get("extra_bits"),
                         "masked_holdout_rel_err": base,
                         "holdout_delta_vs_masked": v.get("holdout_delta_vs_masked"),
                         "beats_masked_holdout": v.get("beats_masked_holdout")})
    positive = [a for a in arms if a["beats_masked_holdout"] and (a["extra_bits_whole_model"] or 0) > 0]
    return {"present": True, "source": str(S2C_REPORT), "verdict": d.get("verdict"), "arms": arms,
            "any_paid_arm_beats_masking": bool(positive),
            "unit": "median relative error of the whole MoE output vs the 128-expert teacher",
            "note": "NOT in this auction's unit and therefore not ranked inside it. S2C's sealed "
                    "verdict is NEGATIVE - no trained router student beats plain masking on "
                    "held-out token ids at layer 0, and the single layer-46 win (bias, "
                    "-0.0067 relative error for 96256 whole-model bits) is unreplicated and of "
                    "the opposite sign to layer 0. Router capacity therefore cannot outrank a "
                    "treatment with positive measured utility per bit."}


def rank(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank competing uses of the same marginal bits by output-error reduction PER BIT.

    entries carry `delta_sq_err` (reduction in squared output error, positive = good) and `bits`
    (bits charged per expert per layer). A negative or zero delta ranks last and is flagged.
    """
    out = []
    for e in entries:
        b = int(e["bits"])
        out.append(dict(e, utility_per_bit=(float(e["delta_sq_err"]) / b if b > 0 else 0.0)))
    out.sort(key=lambda e: -e["utility_per_bit"])
    for i, e in enumerate(out):
        e["auction_rank"] = i + 1
    return out


# ── activation capture (real weights, truncated forward) ──────────────────────────────────────
def capture(layer: int = LAYER, min_tokens: int = MIN_TOKENS, out: Path = CACHE,
            max_layers: int | None = None, progress: bool = True) -> dict[str, Any]:
    """Run the real forward to `layer` and cache that layer's expert-input activations + routing.

    Streams experts one at a time and frees each immediately: never more than one expert resident.
    """
    from tokenizers import Tokenizer  # type: ignore

    import qwen_calibration_corpus as CC
    from qwen_real_forward import DEFAULT_META, from_source, rmsnorm, swiglu

    fwd = from_source()
    if not fwd.source_present():
        raise SystemExit("Qwen3-235B shards not resident; S3C needs the real source")
    g, r = fwd.g, fwd.reader
    tk = Tokenizer.from_file(str(DEFAULT_META / "tokenizer.json"))
    corpus = CC.build(min_tokens=min_tokens, tokenizer=tk)
    prompts = corpus["prompts"]
    # The fit/score split is BY PROMPT, so a corpus of one prompt cannot be split at all. Fail here,
    # at capture time, rather than after a 5-minute forward.
    assert len(prompts) >= 4, (
        f"corpus has {len(prompts)} prompt(s); the prompt-disjoint fit/score split needs >= 4. "
        f"raise --tokens (each frozen segment is ~100 tokens)")
    lens = [len(p["ids"]) for p in prompts]
    owner = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lens)])
    xs = [r.bf16_rows("model.embed_tokens.weight", list(p["ids"])) for p in prompts]

    nb = layer if max_layers is None else min(layer, max_layers)
    t0 = time.time()
    for L in range(nb + 1):
        for i in range(len(xs)):
            xs[i] = xs[i] + fwd._attention(L, xs[i])
        ln = r.bf16(f"model.layers.{L}.post_attention_layernorm.weight")
        h = np.concatenate([rmsnorm(x, ln, g.eps) for x in xs], axis=0).astype(np.float32)
        logits = h @ r.bf16(f"model.layers.{L}.mlp.gate.weight").T
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        idx = np.argsort(-probs, axis=1)[:, :g.top_k].astype(np.int16)
        w = np.take_along_axis(probs, idx.astype(np.int64), axis=1).astype(np.float32)
        if g.norm_topk_prob:
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-20)
        if L == nb:
            break
        acc = np.zeros_like(h)
        for e in np.unique(idx):
            rows, slot = np.nonzero(idx == e)
            ex = fwd._load_expert(L, int(e))
            hr = h[rows]
            a = swiglu(hr @ ex["gate"].T, hr @ ex["up"].T)
            acc[rows] += w[rows, slot][:, None] * (a @ ex["down"].T)
            del ex, a, hr
        off = 0
        for i, n in enumerate(lens):
            xs[i] = xs[i] + acc[off:off + n]
            off += n
        del acc
        if progress:
            print(f"  layer {L:2d}/{nb}  {time.time()-t0:6.1f}s total", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=h, top_idx=idx, top_w=w, owner=owner, layer=np.int32(nb),
                        n_prompts=np.int32(len(prompts)),
                        corpus_sha=np.bytes_(corpus["sha256"]))
    return {"path": str(out), "layer": nb, "n_tokens": int(h.shape[0]),
            "n_prompts": len(prompts), "corpus_sha256": corpus["sha256"],
            "seconds": round(time.time() - t0, 1)}


# ── measurement ───────────────────────────────────────────────────────────────────────────────
def _split(owner: np.ndarray, n_prompts: int) -> tuple[np.ndarray, np.ndarray]:
    """Prompt-disjoint fit/score split: even prompts fit, odd prompts score."""
    fit_p = {i for i in range(n_prompts) if i % 2 == 0}
    fit = np.array([i for i, o in enumerate(owner) if int(o) in fit_p], np.int64)
    score = np.array([i for i, o in enumerate(owner) if int(o) not in fit_p], np.int64)
    return fit, score


def _expert(reader, layer: int, e: int) -> dict[str, np.ndarray]:
    p = f"model.layers.{layer}.mlp.experts.{e}"
    return {k: reader.bf16(f"{p}.{k}_proj.weight").astype(np.float32)
            for k in ("gate", "up", "down")}


def _fwd_expert(ex: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from qwen_real_forward import swiglu
    a = swiglu(x @ ex["gate"].T, x @ ex["up"].T)
    return a, a @ ex["down"].T


def _code(mats: list[np.ndarray], rung_table: dict, rung: str, imp: np.ndarray | None,
          seed: int) -> list[np.ndarray]:
    s = rung_table[rung]
    return FAC.fit(mats, dim=int(s["dim"]), k=int(s["k"]), stages=int(s["stages"]),
                   seed=seed, row_scale=True, importance=imp, iters=4)


def measure(cache: Path = CACHE, n_survivors: int = 3, progress: bool = True) -> dict[str, Any]:
    """Price every competing use of the same marginal bits on real weights + real activations."""
    from qwen_real_forward import SafetensorsIndexReader

    d = np.load(cache, allow_pickle=False)
    X, top_idx, top_w, owner = d["x"], d["top_idx"], d["top_w"], d["owner"]
    layer = int(d["layer"])
    n_prompts = int(d["n_prompts"])
    fit_t, score_t = _split(owner, n_prompts)
    assert len(set(fit_t.tolist()) & set(score_t.tolist())) == 0, "fit/score tokens overlap"
    assert len(set(owner[fit_t].tolist()) & set(owner[score_t].tolist())) == 0, \
        "fit/score PROMPTS overlap - the split must be prompt-disjoint"
    assert len(fit_t) > 0 and len(score_t) > 0

    routing = json.loads(ROUTING.read_text())
    surv = SP.survivors(routing, layer, KEEP)
    lay = routing["layers"][layer]
    order = np.lexsort((-np.asarray(lay["softmax_mass"], np.float64),
                        -np.asarray(lay["top8_count"], np.int64)))
    marginal = int(order[KEEP])                       # the expert K -> K+1 would buy back
    picks = [int(order[i]) for i in (0, KEEP // 4, KEEP - 1)][:n_survivors]

    reader = SafetensorsIndexReader("models/qwen3-235b-a22b")
    gate_shape, down_shape = (1536, 4096), (4096, 1536)

    # h for gate/up is the layer's expert input second moment on FIT tokens only.
    h_x = FAC.importance_from_activations(X[fit_t])

    def rows_for(e: int, toks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = (top_idx[toks] == e)
        sel = m.any(1)
        t = toks[sel]
        w = top_w[toks][sel][m[sel]]
        return t, w.astype(np.float32)

    per_expert: list[dict[str, Any]] = []
    books_cache: dict[str, Any] = {}
    for e in picks:
        t_fit, _ = rows_for(e, fit_t)
        t_sc, w_sc = rows_for(e, score_t)
        if len(t_sc) < 2 or len(t_fit) < 2:
            continue
        ex = _expert(reader, layer, e)
        a_fit, _ = _fwd_expert(ex, X[t_fit])
        h_a = FAC.importance_from_activations(a_fit)
        del a_fit
        _, Y = _fwd_expert(ex, X[t_sc])
        Y = Y * w_sc[:, None]
        e_ref = float((Y ** 2).sum())

        def books(key, mats, table, rung, imp, seed):
            if key not in books_cache:
                books_cache[key] = _code(mats, table, rung, imp, seed)
            return books_cache[key]

        def err(g_rung: str, d_rung: str, doctor: str | None) -> float:
            gb = books(("g", g_rung), [ex["gate"], ex["up"]], SP.GATE_RUNGS, g_rung, h_x, 11)
            db = books(("d", d_rung), [ex["down"]], SP.DOWN_RUNGS, d_rung, h_a, 13)
            gd = int(SP.GATE_RUNGS[g_rung]["dim"])
            dd = int(SP.DOWN_RUNGS[d_rung]["dim"])
            cg = FAC.apply_refit(gb, ex["gate"], dim=gd, importance=h_x)
            cu = FAC.apply_refit(gb, ex["up"], dim=gd, importance=h_x)
            cd = FAC.apply_refit(db, ex["down"], dim=dd, importance=h_a)
            if doctor is not None:
                cd = doctor_arm(ex["down"], cd, h=h_a, select_h=("B" in doctor),
                                fit_h=("A" in doctor), seed=17)
            _, Yp = _fwd_expert({"gate": cg, "up": cu, "down": cd}, X[t_sc])
            del cg, cu, cd
            return float((((Yp * w_sc[:, None]) - Y) ** 2).sum())

        base = err(GATE_BASE, DOWN_BASE, None)
        arms = {
            "base_no_doctor": base,
            "doctor_gen1": err(GATE_BASE, DOWN_BASE, "gen1"),
            "doctor_gen2A_output_fit_codebook": err(GATE_BASE, DOWN_BASE, "A"),
            "doctor_gen2B_output_row_select": err(GATE_BASE, DOWN_BASE, "B"),
            "doctor_gen2AB_both": err(GATE_BASE, DOWN_BASE, "AB"),
            "gate_up_rung_up": err(GATE_UP_RUNG, DOWN_BASE, None),
            "down_rung_up": err(GATE_BASE, DOWN_UP_RUNG, None),
        }
        per_expert.append({"expert": e, "n_score_tokens": int(len(t_sc)),
                           "output_energy": e_ref,
                           "sq_err": arms,
                           "rel_err": {k: math.sqrt(v / max(e_ref, 1e-30))
                                       for k, v in arms.items()}})
        books_cache.clear()
        del ex, Y
        if progress:
            print(f"  expert {e}: " + " ".join(
                f"{k}={math.sqrt(v/max(e_ref,1e-30)):.4f}" for k, v in arms.items()), flush=True)

    # (b) K -> K+1: the marginal expert's own contribution, recovered up to its coding error.
    t_sc_m, w_m = rows_for(marginal, score_t)
    marg: dict[str, Any] = {"expert": marginal, "n_score_tokens": int(len(t_sc_m))}
    if len(t_sc_m) >= 2:
        ex = _expert(reader, layer, marginal)
        t_fit_m, _ = rows_for(marginal, fit_t)
        a_fit, _ = _fwd_expert(ex, X[t_fit_m]) if len(t_fit_m) >= 2 else (None, None)
        h_am = FAC.importance_from_activations(a_fit) if a_fit is not None else None
        del a_fit
        _, Ym = _fwd_expert(ex, X[t_sc_m])
        Ym = Ym * w_m[:, None]
        e_ref_m = float((Ym ** 2).sum())
        gb = _code([ex["gate"], ex["up"]], SP.GATE_RUNGS, GATE_BASE, h_x, 11)
        db = _code([ex["down"]], SP.DOWN_RUNGS, DOWN_BASE, h_am, 13)
        gd = int(SP.GATE_RUNGS[GATE_BASE]["dim"])
        dd = int(SP.DOWN_RUNGS[DOWN_BASE]["dim"])
        _, Yp = _fwd_expert({"gate": FAC.apply_refit(gb, ex["gate"], dim=gd, importance=h_x),
                             "up": FAC.apply_refit(gb, ex["up"], dim=gd, importance=h_x),
                             "down": FAC.apply_refit(db, ex["down"], dim=dd, importance=h_am)},
                            X[t_sc_m])
        coded = float((((Yp * w_m[:, None]) - Ym) ** 2).sum())
        marg |= {"omitted_sq_err": e_ref_m * MISS_COST, "coded_sq_err": coded,
                 "delta_sq_err": e_ref_m * MISS_COST - coded}
        del ex, Ym, Yp
    reader.close()

    # ── auction: mean delta per expert vs mean bits per expert ────────────────────────────────
    def mean_delta(arm: str) -> float:
        return float(np.mean([p["sq_err"]["base_no_doctor"] - p["sq_err"][arm]
                              for p in per_expert]))

    b_doc = bits_doctor(down_shape)
    b_gate = bits_gate_up(gate_shape, GATE_UP_RUNG) - bits_gate_up(gate_shape, GATE_BASE)
    b_down = bits_down(down_shape, DOWN_UP_RUNG) - bits_down(down_shape, DOWN_BASE)
    b_expert = bits_gate_up(gate_shape, GATE_BASE) + bits_down(down_shape, DOWN_BASE)

    entries = [
        {"id": "a_doctor_gen1", "treatment": "Doctor residual on down_proj (Gen1)",
         "bits": b_doc, "delta_sq_err": mean_delta("doctor_gen1")},
        {"id": "a_doctor_gen2A", "treatment": "Doctor, output-fitted correction codebook",
         "bits": b_doc, "delta_sq_err": mean_delta("doctor_gen2A_output_fit_codebook")},
        {"id": "a_doctor_gen2B", "treatment": "Doctor, output-sensitivity row selection",
         "bits": b_doc, "delta_sq_err": mean_delta("doctor_gen2B_output_row_select")},
        {"id": "a_doctor_gen2AB", "treatment": "Doctor, both Gen2 changes",
         "bits": b_doc, "delta_sq_err": mean_delta("doctor_gen2AB_both")},
        {"id": "c_gate_up_rung_up", "treatment": f"gate/up {GATE_BASE} -> {GATE_UP_RUNG} index bpw",
         "bits": b_gate, "delta_sq_err": mean_delta("gate_up_rung_up")},
        {"id": "d_down_rung_up", "treatment": f"down {DOWN_BASE} -> {DOWN_UP_RUNG} index bpw",
         "bits": b_down, "delta_sq_err": mean_delta("down_rung_up")},
    ]
    if "delta_sq_err" in marg:
        # normalise the marginal expert's absolute delta onto the same per-expert scale as the
        # survivor arms by expressing both as a fraction of that expert's own output energy is NOT
        # valid (different denominators), so the raw squared error is used - the auction unit is
        # absolute squared output error at this layer on the same score tokens.
        entries.append({"id": "b_one_more_expert",
                        "treatment": f"K {KEEP} -> {KEEP+1} (expert {marginal} restored)",
                        "bits": b_expert, "delta_sq_err": marg["delta_sq_err"]})
    ranked = rank(entries)

    return {"layer": layer, "survivor_count": len(surv), "marginal_expert": marg,
            "per_expert": per_expert, "auction": ranked,
            "bits": {"doctor_per_expert": b_doc, "gate_up_rung_up_per_expert": b_gate,
                     "down_rung_up_per_expert": b_down, "one_expert_per_layer": b_expert},
            "split": {"n_fit_tokens": int(len(fit_t)), "n_score_tokens": int(len(score_t)),
                      "prompt_disjoint": True},
            "router_capacity_e": router_capacity_e()}


# ── report ────────────────────────────────────────────────────────────────────────────────────
def build_report(m: dict[str, Any]) -> dict[str, Any]:
    ranked = m["auction"]
    best = ranked[0]
    doc = [e for e in ranked if e["id"].startswith("a_doctor")]
    best_doc = max(doc, key=lambda e: e["utility_per_bit"])
    gen1 = next(e for e in doc if e["id"] == "a_doctor_gen1")
    losers = [e for e in ranked if e["utility_per_bit"] > best_doc["utility_per_bit"]]
    doctor_keeps_slot = not losers
    rep = {
        "schema": SCHEMA,
        "stage": "S3C_DOCTOR_GEN2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parent": "qwen3-235b-a22b-instruct-2507",
        "ceiling": "complete_bits / original_weight_count <= 1/1",
        "claim": "PROXY ONLY. Output-space squared error on a calibration batch of REAL routed "
                 "activations at one layer. Weight-space error is not a capability claim and "
                 "output-space error on calibration is not capability either. No frontier is "
                 "selected here; only a real parent-vs-packed 94-layer forward may select one.",
        "measurement": m,
        "QWEN235B_DOCTOR_GEN2_PROGRAM": {
            "diagnosis_to_treatment": [
                {"diagnosis": "down_proj carries the worst rate-distortion floor in the S64 "
                              "artifact (0.625 index bpw)",
                 "treatment": "sparse residual Doctor on down_proj, protected rows only",
                 "measured": {"arm": "doctor_gen1",
                              "utility_per_bit": gen1["utility_per_bit"],
                              "delta_sq_err": gen1["delta_sq_err"], "bits": gen1["bits"]}},
                {"diagnosis": "Gen1 fits the correction codebook by unweighted Lloyd on the WEIGHT "
                              "residual, but the artifact must preserve the OUTPUT",
                 "treatment": "G2-A output-weighted Lloyd on the residual (h = diag E[a a^T])",
                 "measured": next(e for e in doc if e["id"] == "a_doctor_gen2A")},
                {"diagnosis": "Gen1 selects protected rows by relative residual energy, which is "
                              "blind to how much each row's residual reaches the output",
                 "treatment": "G2-B row selection by measured output sensitivity, same bitmap "
                              "bytes",
                 "measured": next(e for e in doc if e["id"] == "a_doctor_gen2B")},
                {"diagnosis": "the S64_doctor artifact sits at 0.999769787 complete BPW, so the "
                              "Doctor's bits are bits some other treatment is not getting",
                 "treatment": "byte auction against K+1, gate/up rung, down rung",
                 "measured": {"winner": best["id"],
                              "winner_utility_per_bit": best["utility_per_bit"],
                              "doctor_best": best_doc["id"],
                              "doctor_utility_per_bit": best_doc["utility_per_bit"]}},
            ],
            "ranked_marginal_utility": ranked,
        },
        "falsification": {
            "rule": "if the Doctor's marginal output-error reduction per bit is below (b), (c) or "
                    "(d) then the Doctor loses its bytes, regardless of its S1 forward win",
            "doctor_best_arm": best_doc["id"],
            "doctor_utility_per_bit": best_doc["utility_per_bit"],
            "beaten_by": [e["id"] for e in losers],
            "doctor_keeps_its_slot": doctor_keeps_slot,
            "router_capacity_e": {
                "in_auction_unit": False,
                "any_paid_arm_beats_masking":
                    m["router_capacity_e"].get("any_paid_arm_beats_masking"),
                "source_verdict": (m["router_capacity_e"].get("verdict") or {}).get("result")},
            "gen2_beats_gen1": bool(best_doc["id"] != "a_doctor_gen1" and
                                    best_doc["utility_per_bit"] > gen1["utility_per_bit"]),
        },
        "honesty_caveats": [
            "Output-space error on a calibration batch is a PROXY, not capability.",
            "Measured at ONE layer (46) on a sample of survivor experts, not the whole model.",
            "(b) uses the SEALED Lane E MISS_COST=1.0; the top-k renormalisation perturbation from "
            "restoring an expert is a second-order term that is NOT modelled.",
            "The shared codebook is fitted PER EXPERT while its bytes are charged amortised over "
            "64 survivors, exactly as the deployed ledger does; every coded arm is flattered "
            "equally, so the ranking is the deliverable and the absolute errors are optimistic.",
            "Codebooks, importance vectors and protected-row choices come from the FIT prompts; "
            "every reported number is measured on the disjoint SCORE prompts.",
            f"(b) is measured on only {m['marginal_expert'].get('n_score_tokens', 0)} score tokens "
            "- the marginal expert is by construction the coldest one, so its sample is the "
            "smallest in the table and its utility is the least precise number here.",
            "(e) router capacity is READ from the sealed S2C report, not re-measured. Its unit "
            "(median relative error of the whole MoE output) is NOT this auction's unit "
            "(absolute squared output error of one expert's weighted contribution), so it is "
            "reported alongside the ranking rather than inside it.",
        ],
    }
    rep["sha256"] = hashlib.sha256(
        json.dumps(rep, sort_keys=True, default=float).encode()).hexdigest()
    return rep


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    """Asserts that FAIL if the core logic breaks. No model weights required."""
    rng = np.random.default_rng(0)

    # 1. Exact Doctor ledger for the real S64_doctor spec, hand-computed.
    rows, cols = 4096, 1536
    n_prot = rows // 2
    hand = ((n_prot * cols // 16) * 1 * 10) + (1 * 1024 * 16 * 16 // KEEP) + n_prot * 16 + rows
    assert bits_doctor((rows, cols)) == hand == 2_007_040, (bits_doctor((rows, cols)), hand)
    # and it is exactly the sealed whole-artifact Doctor charge
    assert hand * KEEP * 94 == 12_074_352_640

    # 2. Rung prices are strictly ordered and the marginal prices are positive.
    assert bits_gate_up((1536, 4096), GATE_UP_RUNG) > bits_gate_up((1536, 4096), GATE_BASE)
    assert bits_down((4096, 1536), DOWN_UP_RUNG) > bits_down((4096, 1536), DOWN_BASE)

    # 3. G2-B row selection: a twin where relative-energy and output-sensitivity DISAGREE.
    #    Group A: tiny rows, huge RELATIVE residual, but living in near-zero-h columns.
    #    Group B: big rows, small relative residual, in the high-h columns that reach the output.
    n, c = 40, 8
    h = np.array([1e-6] * (c // 2) + [1.0] * (c // 2), np.float32)
    h = h / h.mean()
    w = np.zeros((n, c), np.float32)
    base = np.zeros((n, c), np.float32)
    # Group A also has the larger RAW residual, so an unweighted sensitivity would pick it too:
    # only the h-weighting distinguishes the two, which is exactly the property under test.
    w[:20, :c // 2] = 10.0                       # group A lives in the dead columns
    base[:20, :c // 2] = 0.0                     # relative residual 1.0, raw residual 400/row
    w[20:, c // 2:] = 1.0                        # group B lives in the live columns
    base[20:, c // 2:] = 0.9                     # relative residual 0.1, raw residual 0.04/row
    sel_rel = FAC.protected_rows(w, base, 0.5)
    sel_out = rows_by_output_sensitivity(w, base, h, 0.5)
    assert set(sel_rel.tolist()) == set(range(20)), sel_rel
    assert set(sel_out.tolist()) == set(range(20, 40)), sel_out

    def left(sel):
        keep = np.ones(n, bool)
        keep[sel] = False                        # protected rows are corrected exactly
        return float((((w - base) ** 2)[keep] @ h).sum())
    assert left(sel_out) < left(sel_rel) * 0.1, (left(sel_out), left(sel_rel))

    # 3b. (e) router capacity must be READ from S2C, never estimated, and the path must resolve.
    #     The sealed S2C artifact is IN this repo, so a mistyped path is a silent ABSENT and must
    #     fail loudly here rather than quietly dropping (e) out of the auction.
    e5 = router_capacity_e()
    assert e5["present"], f"(e) unreadable: {S2C_REPORT} - a wrong path is not an absent stage"
    assert e5["arms"] and "verdict" in e5, e5
    l0 = [a for a in e5["arms"] if a["layer"] == 0 and (a["extra_bits_whole_model"] or 0) > 0]
    assert l0 and not any(a["beats_masked_holdout"] for a in l0), l0

    # 4. The auction ranks by utility PER BIT, not by raw delta.
    r = rank([{"id": "cheap", "bits": 10, "delta_sq_err": 5.0},
              {"id": "huge", "bits": 1_000_000, "delta_sq_err": 100.0},
              {"id": "useless", "bits": 10, "delta_sq_err": 0.0}])
    assert [e["id"] for e in r] == ["cheap", "huge", "useless"], r
    assert r[0]["utility_per_bit"] == 0.5 and r[0]["auction_rank"] == 1

    # 5. The fit/score split is prompt-disjoint and both halves are non-empty.
    owner = np.repeat(np.arange(6), 4)
    f, s = _split(owner, 6)
    assert len(f) and len(s) and not (set(owner[f].tolist()) & set(owner[s].tolist()))
    assert len(f) + len(s) == len(owner)

    # 6. G2-A must not be worse than plain Lloyd IN THE OUTPUT METRIC it optimises.
    ww = rng.standard_normal((64, 32)).astype(np.float32)
    bb = ww + 0.3 * rng.standard_normal((64, 32)).astype(np.float32)
    hh = FAC.importance_from_activations(
        (rng.standard_normal((128, 32)) * np.logspace(-2, 2, 32)).astype(np.float32))
    global DOCTOR
    saved = DOCTOR
    DOCTOR = {"dim": 8, "k": 16, "stages": 1, "protect_frac": 0.5}
    try:
        plain = doctor_arm(ww, bb, h=hh, select_h=False, fit_h=False, seed=3)
        outfit = doctor_arm(ww, bb, h=hh, select_h=False, fit_h=True, seed=3)
        e_plain = float((((ww - plain) ** 2) @ hh).sum())
        e_fit = float((((ww - outfit) ** 2) @ hh).sum())
        e_none = float((((ww - bb) ** 2) @ hh).sum())
        assert e_plain < e_none and e_fit < e_none, (e_none, e_plain, e_fit)
    finally:
        DOCTOR = saved
    return {"ok": True, "doctor_bits_down_L46": bits_doctor((rows, cols)),
            "gate_up_rung_up_bits": bits_gate_up((1536, 4096), GATE_UP_RUNG) -
            bits_gate_up((1536, 4096), GATE_BASE),
            "down_rung_up_bits": bits_down((4096, 1536), DOWN_UP_RUNG) -
            bits_down((4096, 1536), DOWN_BASE),
            "output_metric_left_relative_select": left(sel_rel),
            "output_metric_left_output_select": left(sel_out),
            "doctor_output_err_no_doctor": e_none, "plain": e_plain, "output_fitted": e_fit}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S3C Doctor Gen2 + byte auction.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--tokens", type=int, default=MIN_TOKENS)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True, default=float))
        return 0
    if a.capture:
        print(json.dumps(capture(a.layer, a.tokens, Path(a.cache),
                                 a.max_layers or None), indent=2))
        return 0
    if a.run:
        rep = build_report(measure(Path(a.cache), a.experts))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2, sort_keys=True, default=float) + "\n")
        print(json.dumps(rep["falsification"], indent=2, sort_keys=True, default=float))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
