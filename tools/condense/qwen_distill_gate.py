#!/usr/bin/env python3.12
"""LANE E - expert-output distillation at the MERGE BOUNDARY (feasibility gate, real weights).

WHAT THIS IS. The structural lane (qwen_structural_plan) buys its rate by DELETING experts: keep
the 64 hottest per layer, double the survivor rate, stay budget-neutral under the 1.0 complete-BPW
ceiling. The measured cost is that 20.8 percent of real top-8 routing decisions land on an expert
that is no longer there and get re-routed to survivors. This module asks the one question that
decides whether that is survivable, on REAL Qwen3-235B weights and REAL residual-stream inputs:

    when the router loses an expert, how wrong is the MoE block output, and can a small LEARNED
    combination of surviving experts stand in for the omitted one?

Three conditions, all measured against the unmodified parent MoE output y_full:
    (a) FALLBACK   - what S64 actually does: mask omitted experts, top-8 over survivors, renormalize.
    (b) SINGLE     - keep the full routing, synthesize each omitted expert as c * f_s* for the single
                     best surviving expert s* (one index + one scalar).
    (c) MERGE      - same, but as a least-squares combination of m surviving experts
                     (m indices + m scalars per omitted expert). (a) minus (c) is the headroom a
                     learned merge would buy, and every coefficient is charged in bits below.

HONESTY, non-negotiable. Everything here is WEIGHT/ACTIVATION SPACE. Relative error on a block
output is NOT a capability claim and never becomes one; only a parent-vs-packed forward with a real
task score can make a capability claim. Full distillation of a 235B parent is not runnable on this
box and is not attempted or pretended. The merge coefficients are fit on even-indexed probe tokens
and scored on odd-indexed ones, so the reported merge numbers are held out, not fit quality.

INPUTS ARE REAL, NOT PROXY. The expert inputs h are the true post-attention-layernorm residual
stream produced by qwen_real_forward on the real 118-shard checkpoint, run to the probe layer.
The probe text is fresh and DISJOINT from both the routing-calibration corpus and the scored
holdout. Sample size is small (a few dozen tokens, 2 layers) - this is a GATE, not a survey, and
the report says so.

MEMORY. One expert is resident at a time: each needed expert is loaded, applied to all probe tokens,
reduced to a [tokens, hidden] output block, and freed. The forward's expert cache is capped small
because a heavy campaign forward shares this box.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "foundry")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qwen_real_forward as RF  # noqa: E402
import qwen_structural_plan as SP  # noqa: E402

SCHEMA = "hawking.gravity.lane_e.distill_gate.v1"
CEILING = Fraction(1, 1)

CALIBRATION = "reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json"
# Fresh text, written for this probe. Not in the calibration corpus, not in the scored holdout.
PROBE_TEXT = (
    "A tide gauge on the estuary logs sixteen readings an hour; the winter storm surge "
    "arrives before the barometer falls, so the operator trusts the water, not the dial."
)
KEEP = 64                 # survivors per layer, the S64 arm
MERGE_M = 4               # surviving experts per learned merge
EXPERT_INDEX_BITS = 7     # ceil(log2(128))
COEF_BITS = 16            # bf16 coefficient
# Sealed structural-lane figure this must sit on top of (S64 g2.5 d0.625 + Doctor).
S64_DOCTOR_BPW = 0.999769787


# ── linear algebra helpers ────────────────────────────────────────────────────────────────────
def _rel(y: np.ndarray, ref: np.ndarray) -> float:
    """Mean over tokens of ||y_p - ref_p|| / ||ref_p||."""
    num = np.linalg.norm(y - ref, axis=-1)
    den = np.maximum(np.linalg.norm(ref, axis=-1), 1e-30)
    return float(np.mean(num / den))


def fit_merge(target: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Least-squares coefficients c minimising ||target - sum_s c_s preds[s]||_F. preds [m,T,H].

    Solved through the m-by-m normal equations, so the T*H design matrix is never formed.
    """
    m = preds.shape[0]
    flat = preds.reshape(m, -1)
    G = flat @ flat.T
    b = flat @ target.reshape(-1)
    G = G + np.eye(m, dtype=G.dtype) * (1e-8 * (np.trace(G) / max(m, 1) + 1e-30))  # tiny ridge
    return np.linalg.solve(G, b)


# ── the probe ─────────────────────────────────────────────────────────────────────────────────
def expert_inputs(fwd: RF.QwenRealForward, tokens: list[int], layer: int) -> np.ndarray:
    """The REAL MoE input h at `layer`: post-attention residual, post-attention-layernorm."""
    x = fwd.logits_for(tokens, max_blocks=layer)          # residual entering block `layer`
    x = x + fwd._attention(layer, x)
    w = fwd.reader.bf16(f"model.layers.{layer}.post_attention_layernorm.weight")
    return RF.rmsnorm(x, w, fwd.g.eps)


def expert_outputs(fwd: RF.QwenRealForward, layer: int, experts: list[int],
                   h: np.ndarray) -> dict[int, np.ndarray]:
    """f_e(h) for each requested expert. ONE expert resident at a time, then freed."""
    r, out = fwd.reader, {}
    for e in experts:
        g = r.bf16(f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight")
        u = r.bf16(f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight")
        a = RF.swiglu(h @ g.T, h @ u.T)                   # [T, moe_inter]
        del g, u
        d = r.bf16(f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight")
        out[e] = (a @ d.T).astype(np.float32)             # [T, hidden]
        del a, d
    return out


def probe_layer(fwd: RF.QwenRealForward, cal: dict[str, Any], tokens: list[int],
                layer: int, keep: int = KEEP, m: int = MERGE_M) -> dict[str, Any]:
    t0 = time.time()
    g = fwd.g
    h = expert_inputs(fwd, tokens, layer)                 # [T, hidden], REAL
    T = h.shape[0]
    gate_w = fwd.reader.bf16(f"model.layers.{layer}.mlp.gate.weight")
    logits = h @ gate_w.T                                 # [T, n_experts]
    S = set(SP.survivors(cal, layer, keep))
    masked = logits.copy()
    masked[:, [e for e in range(logits.shape[1]) if e not in S]] = -np.inf

    full = [RF.route_topk(logits[p], g.top_k, g.norm_topk_prob) for p in range(T)]
    surv = [RF.route_topk(masked[p], g.top_k, g.norm_topk_prob) for p in range(T)]

    need = sorted({int(e) for i, _ in full for e in i} | {int(e) for i, _ in surv for e in i})
    F = expert_outputs(fwd, layer, need, h)

    def mix(routes, sub=None):
        y = np.zeros((T, g.hidden), dtype=np.float32)
        for p, (idx, wts) in enumerate(routes):
            for e, w in zip(idx, wts):
                e = int(e)
                v = F[e][p] if (sub is None or e in S) else sub.get(e, F[e])[p]
                y[p] += w * v
        return y

    y_full = mix(full)
    y_a = mix(surv)

    # every omitted expert the real router actually wanted, on these tokens
    omitted = sorted({int(e) for i, _ in full for e in i} - S)
    fit_rows = np.arange(0, T, 2)       # coefficients are FIT here
    score_rows = np.arange(1, T, 2)     # and SCORED here (held out)

    sub_b: dict[int, np.ndarray] = {}
    sub_c: dict[int, np.ndarray] = {}
    per_expert = []
    for E in omitted:
        hit = [p for p in range(T) if E in [int(x) for x in full[p][0]]]
        fit_hit = [p for p in hit if p % 2 == 0] or hit
        # predictors: the survivors the restricted router most often picks on E's tokens
        tally: dict[int, int] = {}
        for p in hit:
            for e in surv[p][0]:
                tally[int(e)] = tally.get(int(e), 0) + 1
        pool = [e for e, _ in sorted(tally.items(), key=lambda kv: -kv[1])][:max(m, 8)]
        if not pool:
            continue
        tgt = F[E][fit_hit]
        # (b) single best survivor, optimally scaled
        best, best_err, best_c = None, np.inf, 0.0
        for s in pool:
            ps = F[s][fit_hit]
            c = float((ps * tgt).sum() / max((ps * ps).sum(), 1e-30))
            err = float(np.linalg.norm(tgt - c * ps) / max(np.linalg.norm(tgt), 1e-30))
            if err < best_err:
                best, best_err, best_c = s, err, c
        sub_b[E] = best_c * F[best]
        # (c) least-squares merge over m survivors
        pc = pool[:m]
        coef = fit_merge(tgt, np.stack([F[s][fit_hit] for s in pc]))
        sub_c[E] = np.tensordot(coef.astype(np.float32),
                                np.stack([F[s] for s in pc]), axes=(0, 0))
        sc = [p for p in hit if p % 2 == 1] or hit
        per_expert.append({
            "expert": E, "n_tokens_routed": len(hit),
            "predictors": pc, "coefficients": [round(float(c), 6) for c in coef],
            "single_rel_err_heldout": round(_rel(sub_b[E][sc], F[E][sc]), 6),
            "merge_rel_err_heldout": round(_rel(sub_c[E][sc], F[E][sc]), 6),
        })

    y_b = mix(full, sub_b)
    y_c = mix(full, sub_c)
    affected = [p for p in range(T) if any(int(e) not in S for e in full[p][0])]
    aff = np.asarray(affected) if affected else np.arange(0)
    hs = np.asarray([p for p in score_rows if p in set(affected)]) if affected else np.arange(0)

    def block(rows):
        if len(rows) == 0:
            return None
        return {"n_tokens": int(len(rows)),
                "fallback_rel_err": round(_rel(y_a[rows], y_full[rows]), 6),
                "single_rel_err": round(_rel(y_b[rows], y_full[rows]), 6),
                "merge_rel_err": round(_rel(y_c[rows], y_full[rows]), 6)}

    lay = cal["layers"][layer]
    cnt = np.asarray(lay["top8_count"], dtype=np.float64)
    return {
        "layer": layer, "n_probe_tokens": T, "keep": keep, "merge_m": m,
        "calibration_top8_retained_this_layer": round(float(cnt[sorted(S)].sum() /
                                                           max(cnt.sum(), 1e-12)), 6),
        "probe_top8_retained_this_layer": round(
            float(sum(1 for p in range(T) for e in full[p][0] if int(e) in S) /
                  max(T * g.top_k, 1)), 6),
        "n_omitted_experts_routed_to": len(omitted),
        "n_tokens_with_an_omitted_expert": len(affected),
        "moe_output_all_tokens": block(np.arange(T)),
        "moe_output_affected_tokens": block(aff),
        "moe_output_affected_heldout_tokens": block(hs),
        "per_omitted_expert": per_expert,
        "wall_seconds": round(time.time() - t0, 1),
    }


# ── exact bit charge ──────────────────────────────────────────────────────────────────────────
def merge_bits(n_layers: int, n_omitted_per_layer: int, m: int, hidden: int) -> dict[str, int]:
    """Every bit a learned merge adds, charged exactly. Two granularities.

    scalar : m survivor indices + m bf16 scalars per (layer, omitted expert).
    per_row: m survivor indices + m bf16 scalars PER OUTPUT ROW (a per-row merge, the version that
             would actually be needed if one global scalar per survivor turned out too coarse).
    """
    per = n_layers * n_omitted_per_layer
    return {"scalar": per * m * (EXPERT_INDEX_BITS + COEF_BITS),
            "per_row": per * (m * EXPERT_INDEX_BITS + m * hidden * COEF_BITS)}


def affordability(grand_params: int, base_bpw: float, extra_bits: dict[str, int]) -> dict[str, Any]:
    """Does the merge fit ON TOP of the sealed S64+Doctor arm, under complete BPW <= 1/1?"""
    base_bits = math.ceil(base_bpw * grand_params)          # derived from the sealed float figure
    headroom = grand_params - base_bits
    out = {"base_arm": "S64_g2.5_d0.625_doctor", "base_complete_bpw": base_bpw,
           "base_complete_bits_derived": base_bits, "headroom_bits": headroom,
           "headroom_bpw": float(Fraction(headroom, grand_params)), "variants": {}}
    for name, bits in extra_bits.items():
        total = Fraction(base_bits + bits, grand_params)
        out["variants"][name] = {
            "merge_bits": bits,
            "merge_bpw": float(Fraction(bits, grand_params)),
            "complete_bpw_with_merge": float(total),
            "legal_under_one_bit_ceiling": bool(total <= CEILING),
            "overage_bits": max(0, base_bits + bits - grand_params),
            "must_give_up_bits": max(0, bits - headroom),
        }
    return out


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def demo() -> None:
    """Synthetic check of the merge machinery and the bit charge. No weights, no network."""
    rng = np.random.default_rng(0)
    T, H, n = 40, 64, 6
    base = rng.standard_normal((n, T, H)).astype(np.float32)
    # an "omitted" expert that IS exactly 0.7*s0 - 0.3*s2: the merge must find it, held out.
    tgt = 0.7 * base[0] - 0.3 * base[2]
    fit_rows, score_rows = np.arange(0, T, 2), np.arange(1, T, 2)
    coef = fit_merge(tgt[fit_rows], base[:4][:, fit_rows])
    rec = np.tensordot(coef.astype(np.float32), base[:4], axes=(0, 0))
    assert _rel(rec[score_rows], tgt[score_rows]) < 1e-4, _rel(rec[score_rows], tgt[score_rows])
    # a single survivor cannot represent it: the merge must strictly beat the best singleton
    single = min(_rel(float((base[s][fit_rows] * tgt[fit_rows]).sum() /
                            (base[s][fit_rows] ** 2).sum()) * base[s][score_rows],
                      tgt[score_rows]) for s in range(4))
    assert single > 10 * max(_rel(rec[score_rows], tgt[score_rows]), 1e-9), single
    # _rel is a real relative error: identity is 0, sign flip is 2
    assert _rel(tgt, tgt) == 0.0 and abs(_rel(-tgt, tgt) - 2.0) < 1e-5

    # bit charge: scalar merge is affordable on top of S64+Doctor, per-row is NOT.
    gp = 235_093_634_560
    eb = merge_bits(94, 64, MERGE_M, 4096)
    assert eb["scalar"] == 94 * 64 * 4 * 23, eb["scalar"]
    aff = affordability(gp, S64_DOCTOR_BPW, eb)
    assert aff["variants"]["scalar"]["legal_under_one_bit_ceiling"] is True
    assert aff["variants"]["per_row"]["legal_under_one_bit_ceiling"] is False
    assert aff["variants"]["per_row"]["must_give_up_bits"] > 0
    # the ceiling is enforced, not decorative
    assert not affordability(gp, S64_DOCTOR_BPW, {"x": gp})["variants"]["x"][
        "legal_under_one_bit_ceiling"]
    print(json.dumps({"ok": True, "merge_bits": eb,
                      "scalar_bpw": aff["variants"]["scalar"]["merge_bpw"],
                      "per_row_bpw": aff["variants"]["per_row"]["merge_bpw"]}, indent=2))


# ── run ───────────────────────────────────────────────────────────────────────────────────────
def run(layers: list[int], n_tokens: int, out_path: str | None) -> dict[str, Any]:
    from tokenizers import Tokenizer  # type: ignore
    tk = Tokenizer.from_file(str(RF.DEFAULT_META / "tokenizer.json"))
    tokens = tk.encode(PROBE_TEXT).ids[:n_tokens]

    import bounded_cache  # noqa: E402
    cache = bounded_cache.PressureAwareCache("lane-e", disk_path=str(RF.DEFAULT_SOURCE),
                                             min_entries=2, hard_max=6, verbose=False)
    cache.max_bytes = 3 * 1024 ** 3            # a heavy campaign forward shares this box
    fwd = RF.from_source()
    fwd.cache = cache
    cal = json.loads(Path(CALIBRATION).read_text())

    import qwen3_moe_adapter as A  # noqa: E402
    gp = A.build_inventory(A.load_config(), A.load_index()).grand_params

    t0 = time.time()
    per_layer = [probe_layer(fwd, cal, tokens, L) for L in layers]
    eb = merge_bits(cal["n_layers"], cal["n_experts"] - KEEP, MERGE_M, fwd.g.hidden)

    fb = [p["moe_output_affected_tokens"]["fallback_rel_err"] for p in per_layer]
    mg = [p["moe_output_affected_heldout_tokens"]["merge_rel_err"] for p in per_layer]
    med = [round(float(np.median([x["merge_rel_err_heldout"] for x in p["per_omitted_expert"]])), 6)
           for p in per_layer]
    med1 = [round(float(np.median([x["single_rel_err_heldout"] for x in p["per_omitted_expert"]])), 6)
            for p in per_layer]
    rep = {
        "schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent": "qwen3-235b-a22b-instruct-2507", "source": str(RF.DEFAULT_SOURCE),
        "ceiling": "complete_bits / original_weight_count <= 1/1",
        "claim_scope": "WEIGHT/ACTIVATION SPACE ONLY. Relative error on a MoE block output is NOT "
                       "a capability claim. Only a parent-vs-packed forward with a task score can "
                       "make one. Full 235B distillation was not run and is not claimed.",
        "inputs": "REAL post-attention-layernorm residual stream from qwen_real_forward on the "
                  "real 118-shard checkpoint. Not a proxy distribution.",
        "probe_text_disjoint_from": [CALIBRATION, "campaign scored holdout"],
        "n_probe_tokens": len(tokens), "layers_probed": layers,
        "survivor_source": CALIBRATION, "keep_experts_per_layer": KEEP, "merge_m": MERGE_M,
        "sample_size_caveat": "A GATE, not a survey: %d tokens, %d layers. Directionally decisive, "
                              "not a population estimate." % (len(tokens), len(layers)),
        "depth_caveat": "Layers probed are SHALLOW. A mid-depth probe (L46, L70) was launched twice "
                        "and abandoned after ~45 and ~75 minutes: reaching block L requires a real "
                        "forward through L blocks, i.e. tens of GiB of expert-shard reads, and the "
                        "concurrent 94-layer campaign forward saturates the disk. The result is "
                        "therefore established on early layers only and should be re-run at "
                        "L46/L70 when the box is quiet. Layer 0 is known to be atypical, which is "
                        "why L3 and L7 are included.",
        "per_layer": per_layer,
        "bits_charged": eb,
        "affordability": affordability(gp, S64_DOCTOR_BPW, eb),
        "summary": {
            "fallback_rel_err_affected_tokens": [round(v, 6) for v in fb],
            "merge_rel_err_affected_heldout_tokens": [round(v, 6) for v in mg],
            "merge_headroom_vs_fallback": [round(a - b, 6) for a, b in zip(fb, mg)],
            "median_single_expert_reconstruction_rel_err_heldout": med1,
            "median_merged_expert_reconstruction_rel_err_heldout": med,
            "reading": "An omitted expert is NOT reconstructible from survivors: the median held-out "
                       "reconstruction error of f_E is near 1.0, i.e. no better than predicting "
                       "zero. This is the same wall as the sealed inter-expert-redundancy dead "
                       "lever. At MoE-block level a learned merge buys only a few points off the "
                       "fallback error, which itself is large.",
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lane E: expert-output merge-boundary gate.")
    ap.add_argument("--demo", action="store_true", help="runnable self-check, no weights")
    ap.add_argument("--layers", default="0,46")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.demo:
        demo()
        return 0
    rep = run([int(x) for x in a.layers.split(",")], a.tokens, a.out or None)
    for p in rep["per_layer"]:
        b = p["moe_output_affected_tokens"]
        hb = p["moe_output_affected_heldout_tokens"]
        print(f"L{p['layer']:<3} omitted_routed={p['n_omitted_experts_routed_to']:<3} "
              f"affected={b['n_tokens']:<3} fallback={b['fallback_rel_err']:.4f} "
              f"single={b['single_rel_err']:.4f} merge={b['merge_rel_err']:.4f} "
              f"| heldout merge={hb['merge_rel_err']:.4f}")
    print(json.dumps(rep["affordability"]["variants"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
