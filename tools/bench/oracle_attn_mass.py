#!/usr/bin/env python3
"""L1.1 attention-mass concentration oracle — reader / verdict.

Consumes the JSON written by the in-engine capture instrument
(`crate::stateful::attn_capture`, gated behind DISMANTLE_QWEN_ATTN_CAPTURE=1)
and prints the §2.5 verdict for the KV-working-set lever:

  Does a small bounded set of cached positions hold >=99% of the attention
  mass per layer on Qwen2.5-3B, and is that set the StreamingLLM/H2O
  structure (a few attention sinks + a recent window) rather than scattered?

GO  iff, aggregated over captured long-context query positions, a bounded
    working set (sinks + a recent window of size W) captures >=99% mass on
    (nearly) every layer at a budget << context length.
NO-GO if mass is spread broadly (the min #positions for 99% is a large
    fraction of context, or sinks+recent miss a lot of mass) -> any bounded
    budget would drop load-bearing context. Same discipline that killed
    block-256 FFN sparsity.

Usage:
  tools/bench/oracle_attn_mass.py [reports/bench/attn_capture.json]
"""
import json
import sys


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/bench/attn_capture.json"
    with open(path) as f:
        d = json.load(f)

    layers = d.get("layers", [])
    if not layers:
        print(f"no captured layers in {path} — did you run with "
              f"DISMANTLE_QWEN_ATTN_CAPTURE=1 on a long prompt?")
        return 2

    thr = d["mass_thresholds"]            # e.g. [0.90, 0.99, 0.999]
    recent_spans = d["recent_spans"]      # e.g. [32, 64, 128]
    sink_span = d["sink_span"]
    min_ctx = d.get("min_ctx_len", 0)
    # index of the 0.99 threshold
    i99 = min(range(len(thr)), key=lambda i: abs(thr[i] - 0.99))

    print(f"== L1.1 attention-mass concentration oracle ==")
    print(f"source: {path}")
    print(f"min_ctx_len filter: {min_ctx}  (only positions with >= this "
          f"many cached tokens are scored)")
    print(f"sink_span={sink_span}  recent_spans={recent_spans}  "
          f"mass_thresholds={thr}")
    print()

    # Per-layer table.
    hdr = (f"{'layer':>5} {'samp':>6} {'ctx':>7} {'top1':>6} "
           f"{'sink':>6} " +
           " ".join(f"s+r{w:>4}" for w in recent_spans) + "  " +
           f"{'pos99':>7} {'frac99':>7}")
    print(hdr)
    print("-" * len(hdr))

    n_layers = len(layers)
    # Aggregates for the verdict.
    fracs99 = []
    best_sr_cover = []     # best sinks+recent coverage across spans, per layer
    mean_ctx_all = []
    for L in layers:
        sr = L["mean_sink_plus_recent"]
        frac99 = L["mean_min_frac"][i99]
        fracs99.append(frac99)
        best_sr = max(sr)
        best_sr_cover.append(best_sr)
        mean_ctx_all.append(L["mean_ctx_len"])
        print(f"{L['layer']:>5} {L['samples']:>6} {L['mean_ctx_len']:>7.0f} "
              f"{L['mean_top1']:>6.3f} {L['mean_sink_mass']:>6.3f} " +
              " ".join(f"{x:>7.3f}" for x in sr) + "  " +
              f"{L['mean_min_pos'][i99]:>7.1f} {frac99:>7.4f}")

    print()
    mean_ctx = sum(mean_ctx_all) / len(mean_ctx_all)
    # Verdict aggregates: use the worst (max) frac99 across layers — a bounded
    # budget must work on EVERY layer, so the hardest layer governs.
    worst_frac99 = max(fracs99)
    median_frac99 = sorted(fracs99)[len(fracs99) // 2]
    # sinks+recent coverage: worst (min across layers) of the best-span cover.
    worst_sr = min(best_sr_cover)
    median_sr = sorted(best_sr_cover)[len(best_sr_cover) // 2]
    # implied bounded budget to hold 99% on the hardest layer:
    implied_budget = worst_frac99 * mean_ctx

    print(f"mean context length scored: {mean_ctx:.0f} tokens")
    print(f"99%-mass position fraction  — median {median_frac99:.4f}, "
          f"WORST-layer {worst_frac99:.4f}")
    print(f"  -> implied bounded budget for 99% on the hardest layer: "
          f"~{implied_budget:.0f} positions ({worst_frac99*100:.1f}% of ctx)")
    print(f"sinks(+recent) 99% coverage  — median {median_sr:.4f}, "
          f"WORST-layer {worst_sr:.4f}  "
          f"(best recent span per layer)")
    print()

    # ---- Verdict thresholds (documented, conservative) ----
    # GO if the hardest layer needs < 25% of context for 99% mass AND a fixed
    # sinks+recent window covers >= 0.97 mass on the hardest layer (the
    # StreamingLLM structure actually holds). Tunable budget can trade the
    # rest. NO-GO if mass is broadly spread.
    FRAC_GO = 0.25
    SR_GO = 0.97
    go_concentration = worst_frac99 < FRAC_GO
    go_structure = worst_sr >= SR_GO

    print(f"thresholds: GO needs worst-layer frac99 < {FRAC_GO} AND "
          f"worst-layer sinks+recent cover >= {SR_GO}")
    if go_concentration and go_structure:
        verdict = "GO"
        why = ("a bounded sinks+recent working set captures ~99% mass on "
               "every layer at a budget well below context length")
    elif go_concentration and not go_structure:
        verdict = "GO (H2O-style, not pure StreamingLLM)"
        why = ("mass IS concentrated (few positions hold 99%) but the heavy "
               "positions are NOT just sinks+recent — a cumulative-mass "
               "(H2O) policy is needed, not pure positional StreamingLLM")
    else:
        verdict = "NO-GO"
        why = ("attention mass is spread broadly — the hardest layer needs "
               f"{worst_frac99*100:.0f}% of context for 99% mass; any bounded "
               "budget drops load-bearing context (Type-1 on this model/ctx, "
               "same as block-256 FFN sparsity)")
    print()
    print(f"VERDICT: {verdict}")
    print(f"  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
