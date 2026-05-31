#!/usr/bin/env python3
"""Bible §8.1 L1.3 oracle — cross-layer weight delta-encoding (kill-or-keep).

Hypothesis (L1.3): adjacent transformer layers' weight matrices are correlated,
so layer L+1 can be stored as layer L + a delta D = W[L+1] - W[L] that is more
compressible (lower entropy / low-rank) than the full W[L+1]. If true, fewer
*unique* bytes/token cross the memory bus (decode is bandwidth-bound) -> a real
lever. If Qwen's layers are too independent (cosine ~ 0, D no more compressible
than the original), the lever dies cheaply with ZERO kernel written. Same
discipline that killed block-256 FFN sparsity.

What it measures, per consecutive layer pair (L, L+1) and per weight type
(q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj):

  1. cosine similarity of flattened W[L] and W[L+1] (are they aligned at all?).
  2. delta D = W[L+1] - W[L]:
       - low-rank energy: SVD of D; fraction of Frobenius energy in the top-r
         singular values for r in {16, 32, 64}, plus the rank needed for 99%.
       - quantizability: std(D) vs std(W[L+1]); approx bits/weight needed to
         quantize D at the *same* absolute reconstruction error Q4_K achieves on
         the original (uniform-quantizer model: at matched RMSE, bits scale as
         log2(std), so delta_bits = base_bits + log2(std(D)/std(W[L+1]))).
  3. byte comparison at equal reconstruction error:
       - baseline: store W[L+1] at its native GGUF bits/weight (Q4_K=4.5,
         Q6_K=6.5625).
       - delta-lowrank: store D as rank-r factors at fp16 -> 16*r*(m+n)/(m*n)
         bits/weight (pick the smallest r whose D-energy >= 99%).
       - delta-quant: store D at the matched-error bit estimate from (2).
     The delta path "wins" only if (its bits/weight) < (baseline bits/weight)
     at >= the baseline's reconstruction fidelity.

DECISION:
  GO    if deltas are SUBSTANTIALLY more compressible than the originals across
        MOST tensor types (cosine clearly > 0 AND a delta path beats baseline
        bytes at equal error for the majority of types).
  NO-GO if layers are too independent: cosine ~ 0 and no delta path is cheaper
        than just storing W[L+1] directly.

RAM discipline: ONE tensor pair resident at a time; del + gc.collect() between
pairs; never hold the full f32 model. SVD is run on the smaller Gram matrix
(min(m,n) x min(m,n)) so even the 11008x2048 FFN tensors stay cheap.

CPU-only. No Metal, no model run, no cargo. Reads the GGUF the engine serves.

Run:  /tmp/ggufenv/bin/python tools/bench/oracle_interlayer_delta.py [gguf] [--full]
By default samples a representative set of layer pairs (early/mid/late) to stay
well under the RAM budget and finish fast; --full does all consecutive pairs.
"""
import argparse
import gc
import json
import os
import resource
import sys

import numpy as np
from gguf import GGUFReader, GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize


# GGUF tensor short-name -> bible/HF projection name.
TYPE_MAP = {
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_output": "o_proj",
    "ffn_gate": "gate_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
}
RANKS = (16, 32, 64)
ENERGY_TARGET = 0.99  # fraction of delta Frobenius energy a low-rank store must keep
# Uniform-quantizer range model: assume weights span ~k*std for clipless coverage.
# k cancels in the std-ratio comparison, so its exact value never affects the
# verdict; we only need it to print an absolute bits estimate. 6 std ~ full range.
QUANT_RANGE_K = 6.0


def rss_gb():
    # macOS ru_maxrss is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def bits_per_weight(qtype):
    blk, tbytes = GGML_QUANT_SIZES[qtype]
    return tbytes * 8.0 / blk


def to_2d(t):
    """Dequantize a GGUF tensor to a 2D f32 matrix in logical (rows, cols)."""
    W = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
    if W.ndim != 2:
        W = W.reshape(tuple(int(x) for x in t.shape)[::-1])
    return W


def topr_energy_fractions(D, ranks):
    """Fraction of Frobenius energy (sum sigma^2) captured by top-r singular
    values, plus the rank needed for ENERGY_TARGET. SVD via eig of the smaller
    Gram matrix: singular_values^2 == eigenvalues of the (min-dim) Gram."""
    m, n = D.shape
    if m <= n:
        gram = D @ D.T          # m x m, m = min
    else:
        gram = D.T @ D          # n x n, n = min
    eig = np.linalg.eigvalsh(gram)          # ascending, >= 0 up to fp noise
    eig = np.clip(eig[::-1], 0.0, None)      # descending
    total = float(eig.sum()) + 1e-30
    cum = np.cumsum(eig) / total
    fracs = {}
    for r in ranks:
        rr = min(r, len(eig))
        fracs[r] = float(cum[rr - 1])
    rank99 = int(np.searchsorted(cum, ENERGY_TARGET) + 1)
    return fracs, rank99, len(eig)


def lowrank_bits_per_weight(m, n, r, fp_bytes=2):
    """Bits/weight to store an m x n matrix as rank-r factors at fp_bytes each."""
    return fp_bytes * 8.0 * r * (m + n) / (m * n)


def analyze_pair(t0, t1):
    """All metrics for one (W[L], W[L+1]) pair. Holds <=3 dense matrices."""
    W0 = to_2d(t0)
    W1 = to_2d(t1)
    m, n = W1.shape
    f0 = W0.ravel()
    f1 = W1.ravel()
    n0 = float(np.linalg.norm(f0))
    n1 = float(np.linalg.norm(f1))
    cosine = float(np.dot(f0, f1) / (n0 * n1 + 1e-30))
    del f0
    D = W1 - W0
    del W0
    gc.collect()
    fd = D.ravel()
    std_w1 = float(f1.std())
    std_d = float(fd.std())
    # Relative Frobenius magnitude of the delta vs the original.
    rel_delta_norm = float(np.linalg.norm(fd) / (n1 + 1e-30))
    del f1, fd
    gc.collect()

    base_qtype = t1.tensor_type
    base_bpw = bits_per_weight(base_qtype)

    # (2) matched-error quant bits for the delta: at equal RMSE, bits scale as
    # log2(std). base_bpw already encodes W1 at its quant error; the delta needs
    # base_bpw + log2(std_d/std_w1) bits to hit the SAME absolute RMSE. If the
    # delta has higher std it needs MORE bits -> anti-compressible.
    std_ratio = std_d / (std_w1 + 1e-30)
    delta_quant_bpw = base_bpw + float(np.log2(std_ratio + 1e-30))

    # (1)+(3) low-rank energy of the delta and its byte cost.
    fracs, rank99, mindim = topr_energy_fractions(D, RANKS)
    del D
    gc.collect()

    # Smallest preset rank whose energy >= target; else rank99 (capped at mindim).
    chosen_r = next((r for r in RANKS if fracs[r] >= ENERGY_TARGET), None)
    if chosen_r is None:
        chosen_r = min(rank99, mindim)
        chosen_r_is_preset = False
    else:
        chosen_r_is_preset = True
    delta_lowrank_bpw = lowrank_bits_per_weight(m, n, chosen_r)

    # Best delta path = cheaper of the two delta encodings.
    delta_best_bpw = min(delta_quant_bpw, delta_lowrank_bpw)
    wins = delta_best_bpw < base_bpw

    return {
        "shape": [int(m), int(n)],
        "base_qtype": base_qtype.name,
        "base_bpw": round(base_bpw, 4),
        "cosine": round(cosine, 5),
        "std_w1": round(std_w1, 6),
        "std_delta": round(std_d, 6),
        "std_ratio_delta_over_w1": round(std_ratio, 4),
        "rel_delta_frob_norm": round(rel_delta_norm, 4),
        "delta_topr_energy": {str(r): round(fracs[r], 4) for r in RANKS},
        "delta_rank_for_99pct_energy": int(rank99),
        "min_dim": int(mindim),
        "chosen_rank": int(chosen_r),
        "chosen_rank_hits_99": bool(chosen_r_is_preset),
        "delta_quant_bpw_matched_err": round(delta_quant_bpw, 4),
        "delta_lowrank_bpw_fp16": round(delta_lowrank_bpw, 4),
        "delta_best_bpw": round(delta_best_bpw, 4),
        "delta_beats_baseline": bool(wins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="?",
                    default="models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--full", action="store_true",
                    help="all consecutive pairs (default: sampled early/mid/late)")
    ap.add_argument("--out", default="reports/oracle_interlayer_delta.md")
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    by_name = {t.name: t for t in r.tensors}

    # Discover layer count from blk.N.attn_q.weight presence.
    layer_ids = sorted(
        int(name.split(".")[1])
        for name in by_name
        if name.startswith("blk.") and name.endswith(".attn_q.weight")
    )
    n_layers = (max(layer_ids) + 1) if layer_ids else 0
    if n_layers < 2:
        print("ERROR: fewer than 2 layers found; cannot form a pair.")
        sys.exit(1)

    if args.full:
        pairs = [(L, L + 1) for L in range(n_layers - 1)]
    else:
        # Representative early / mid / late consecutive pairs.
        mid = n_layers // 2
        cand = [(0, 1), (mid - 1, mid), (n_layers - 2, n_layers - 1)]
        pairs = sorted({p for p in cand if p[0] >= 0 and p[1] < n_layers})

    # results[short_type] = list of per-pair dicts
    results = {k: [] for k in TYPE_MAP}
    max_rss = rss_gb()
    print(f"gguf={args.gguf}  layers={n_layers}  pairs={pairs}  "
          f"mode={'FULL' if args.full else 'SAMPLED'}")
    for (L, Lp1) in pairs:
        for short in TYPE_MAP:
            n0 = f"blk.{L}.{short}.weight"
            n1 = f"blk.{Lp1}.{short}.weight"
            if n0 not in by_name or n1 not in by_name:
                continue
            res = analyze_pair(by_name[n0], by_name[n1])
            res["pair"] = [L, Lp1]
            results[short].append(res)
            max_rss = max(max_rss, rss_gb())
            print(f"  [{L}->{Lp1}] {TYPE_MAP[short]:<10} "
                  f"cos={res['cosine']:+.4f} "
                  f"std_ratio={res['std_ratio_delta_over_w1']:.2f} "
                  f"E64={res['delta_topr_energy']['64']:.3f} "
                  f"base={res['base_bpw']:.2f}b "
                  f"deltaBest={res['delta_best_bpw']:.2f}b "
                  f"{'WIN' if res['delta_beats_baseline'] else 'lose'} "
                  f"| rss={rss_gb():.2f}G")
            gc.collect()

    # ---- Aggregate per tensor type ----
    agg = {}
    for short, lst in results.items():
        if not lst:
            continue
        cos = np.array([x["cosine"] for x in lst])
        sr = np.array([x["std_ratio_delta_over_w1"] for x in lst])
        e64 = np.array([float(x["delta_topr_energy"]["64"]) for x in lst])
        base_bpw = np.array([x["base_bpw"] for x in lst])
        best_bpw = np.array([x["delta_best_bpw"] for x in lst])
        wins = np.array([x["delta_beats_baseline"] for x in lst])
        agg[short] = {
            "proj": TYPE_MAP[short],
            "n_pairs": len(lst),
            "cosine_mean": float(cos.mean()),
            "cosine_min": float(cos.min()),
            "cosine_max": float(cos.max()),
            "std_ratio_mean": float(sr.mean()),
            "delta_E64_mean": float(e64.mean()),
            "base_bpw": float(base_bpw.mean()),
            "delta_best_bpw_mean": float(best_bpw.mean()),
            "win_frac": float(wins.mean()),
        }

    # ---- Verdict ----
    # GO needs MOST tensor types to show real correlation AND a cheaper delta.
    # "Correlation" gate: mean cosine clearly above zero (>= 0.30 is generous;
    # genuine cross-layer sharing in residual nets shows cos in the 0.5-0.9 band).
    # "Byte" gate: delta beats baseline at equal error in the majority of pairs.
    COS_GATE = 0.30
    types_with_corr = sum(1 for a in agg.values() if a["cosine_mean"] >= COS_GATE)
    types_with_bytes = sum(1 for a in agg.values() if a["win_frac"] >= 0.5)
    n_types = len(agg)
    go_corr = types_with_corr > n_types / 2
    go_bytes = types_with_bytes > n_types / 2
    go = go_corr and go_bytes
    verdict = "GO" if go else "NO-GO"

    overall_cos = float(np.mean([a["cosine_mean"] for a in agg.values()]))
    overall_stdratio = float(np.mean([a["std_ratio_mean"] for a in agg.values()]))
    overall_e64 = float(np.mean([a["delta_E64_mean"] for a in agg.values()]))

    # ---- Markdown report ----
    lines = []
    lines.append("# Oracle — §8.1 L1.3 cross-layer weight delta-encoding")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"- Model: `{args.gguf}` ({n_layers} layers)")
    lines.append(f"- Mode: {'FULL (all consecutive pairs)' if args.full else 'SAMPLED'}; "
                 f"pairs analyzed: {pairs}")
    lines.append(f"- Peak RSS during run: {max_rss:.2f} GB")
    lines.append(f"- Q4_K baseline = 4.5 bits/weight; Q6_K = 6.5625 bits/weight "
                 f"(verified from gguf GGML_QUANT_SIZES).")
    lines.append("")
    lines.append("## Deciding numbers (means across analyzed pairs)")
    lines.append("")
    lines.append(f"- Mean cosine(W[L], W[L+1]) across all types: **{overall_cos:+.4f}** "
                 f"(GO gate: >= {COS_GATE}).")
    lines.append(f"- Mean std(delta)/std(W[L+1]): **{overall_stdratio:.3f}** "
                 f"(> 1.0 means the delta needs MORE bits than the original at equal error).")
    lines.append(f"- Mean top-64 SVD energy of the delta: **{overall_e64:.3f}** "
                 f"(fraction of delta energy in its 64 largest singular values; "
                 f"min(dim) is 256-2048, so 64 is a generous rank budget).")
    lines.append(f"- Tensor types with mean cosine >= {COS_GATE}: "
                 f"{types_with_corr}/{n_types}.")
    lines.append(f"- Tensor types where a delta encoding beats Q4_K/Q6_K bytes at "
                 f"equal error (majority of pairs): {types_with_bytes}/{n_types}.")
    lines.append("")
    lines.append("## Per-tensor-type summary")
    lines.append("")
    lines.append("| proj | pairs | cos mean | cos range | std(D)/std(W) | "
                 "delta top-64 E | base bpw | delta best bpw | delta wins |")
    lines.append("|------|-------|---------|-----------|---------------|"
                 "----------------|----------|----------------|------------|")
    order = ["attn_q", "attn_k", "attn_v", "attn_output",
             "ffn_gate", "ffn_up", "ffn_down"]
    for short in order:
        a = agg.get(short)
        if not a:
            continue
        lines.append(
            f"| {a['proj']} | {a['n_pairs']} | {a['cosine_mean']:+.4f} | "
            f"[{a['cosine_min']:+.3f}, {a['cosine_max']:+.3f}] | "
            f"{a['std_ratio_mean']:.3f} | {a['delta_E64_mean']:.3f} | "
            f"{a['base_bpw']:.3f} | {a['delta_best_bpw_mean']:.3f} | "
            f"{int(round(a['win_frac']*100))}% |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if go:
        lines.append(
            "Layers show material correlation and the delta compresses below the "
            "Q4_K/Q6_K byte budget at equal reconstruction error across most "
            "tensor types. Proceed to a repack + decode-side prototype (read base "
            "+ delta contiguously)."
        )
    else:
        reasons = []
        if not go_corr:
            reasons.append(
                f"consecutive layers are essentially uncorrelated (mean cosine "
                f"{overall_cos:+.4f}, far below the {COS_GATE} gate) — there is no "
                f"shared structure for a delta to exploit"
            )
        if overall_stdratio >= 1.0:
            reasons.append(
                f"the delta has HIGHER variance than the original "
                f"(std ratio {overall_stdratio:.2f} >= 1.0), so quantizing D at "
                f"equal error costs MORE bits than storing W[L+1] directly — the "
                f"delta is anti-compressible, not compressible"
            )
        if overall_e64 < 0.5:
            reasons.append(
                f"the delta is full-rank (only {overall_e64:.1%} of its energy in "
                f"the top-64 singular values), so a low-rank store of D cannot "
                f"approach the Q4_K byte budget either"
            )
        if not go_bytes:
            reasons.append(
                f"no delta encoding beats the native quant bytes at equal error in "
                f"the majority of tensor types ({types_with_bytes}/{n_types})"
            )
        lines.append(
            "L1.3 dies cheaply on Qwen2.5-3B with zero kernel written, because " +
            "; ".join(reasons) + "."
        )
        lines.append("")
        lines.append(
            "This is the expected outcome for a well-trained transformer: each "
            "layer learns a distinct transform, so W[L+1] - W[L] is roughly the "
            "difference of two near-independent random-looking matrices, whose "
            "variance adds (std grows by ~sqrt(2)) and whose spectrum stays flat. "
            "Same discipline that killed block-256 FFN sparsity: measured, not "
            "assumed."
        )
        lines.append("")
        lines.append(
            "Note: the std-ratio comparison is range-model-independent — the "
            "assumed quant range k*std cancels, so the verdict rests only on the "
            "measured std(D) vs std(W) and the measured SVD spectrum, not on any "
            "tunable constant."
        )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- Source: the Q4_K_M GGUF the engine actually serves; each tensor "
        "dequantized to f32 via gguf's own type-trait dequantizer (Q4_K/Q6_K). "
        "Measuring structure of the served weights is correct here."
    )
    lines.append(
        "- cosine: dot(flat W[L], flat W[L+1]) / (||.|| ||.||)."
    )
    lines.append(
        "- delta low-rank: SVD via eig of the smaller Gram (min-dim^2); top-r "
        "energy = sum of r largest sigma^2 / total. Low-rank store cost = "
        "16*r*(m+n)/(m*n) bits/weight at fp16 factors."
    )
    lines.append(
        "- delta quant cost at equal error: a uniform quantizer hits a target "
        "RMSE with bits = log2(range / (RMSE*sqrt(12))); holding the target = "
        "Q4_K's error on W[L+1], the delta needs base_bpw + log2(std(D)/std(W)) "
        "bits. std(D) > std(W) => more bits, i.e. anti-compressible."
    )
    lines.append(
        "- RAM: one pair resident at a time, del + gc.collect() between pairs; "
        f"peak RSS {max_rss:.2f} GB."
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("")
    print(f"VERDICT: {verdict}")
    print(f"  mean cosine={overall_cos:+.4f} (gate>={COS_GATE})  "
          f"mean std(D)/std(W)={overall_stdratio:.3f}  "
          f"mean delta-top64-energy={overall_e64:.3f}")
    print(f"  types with correlation: {types_with_corr}/{n_types}  "
          f"types where delta beats bytes: {types_with_bytes}/{n_types}")
    print(f"  peak RSS={max_rss:.2f} GB")
    print(f"-> {args.out}")

    # Also drop a JSON sidecar next to the bench oracle JSONs for machine reads.
    json_out = "reports/oracle/interlayer_delta.json"
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    json.dump({
        "oracle": "interlayer_delta_l1_3",
        "model": args.gguf,
        "n_layers": n_layers,
        "pairs": [list(p) for p in pairs],
        "mode": "full" if args.full else "sampled",
        "verdict": verdict,
        "gates": {"cosine_gate": COS_GATE,
                  "types_with_correlation": types_with_corr,
                  "types_with_byte_win": types_with_bytes,
                  "n_types": n_types},
        "overall": {"cosine_mean": overall_cos,
                    "std_ratio_mean": overall_stdratio,
                    "delta_top64_energy_mean": overall_e64},
        "per_type": agg,
        "per_pair": results,
        "peak_rss_gb": max_rss,
    }, open(json_out, "w"), indent=2)
    print(f"-> {json_out}")


if __name__ == "__main__":
    main()
