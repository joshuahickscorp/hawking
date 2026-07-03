#!/usr/bin/env python3.12
"""The Hawking condense parameter-sweep MANIFEST + size/tier math.

Philosophy (the contract): match each model's params to the LOWEST BIT POSSIBLE at
near-1:1 quality via the doctor. The smallest viable artifact = the highest tps. So the
sweep is a BIT-FLOOR SEARCH (climb 1→4 bit, stop at the lowest bit the doctor holds
near-1:1), not a fixed grid. The central hypothesis: the bit-floor DESCENDS as params
rise (bigger = more redundant = compresses harder). See docs/plans/parameter_sweep_pipeline.md.

Two ceilings on a 96 GB box:
  CONDENSE (needs f16 resident ≈ 2×params): ~32-40B naive; ~235B+ via phase-2 block-wise.
  SERVE   (needs only the .tq ≈ bpw/8×params): ~200B @3-bit, ~285B @2-bit, ~500B @1.34-bit.

CLI:
  python ladder.py            # summary by family/tier
  python ladder.py --plan     # the floor-search cells the driver will run
  python ladder.py --tsv      # flat table (family, model, params, tier, serve-fit per bpw)
  python ladder.py --fit 405  # show serve footprint for an arbitrary param count
"""
import sys

# ── bit → bpw payload (matches condense.sh / TrellisConfig::for_bpw_quality) ──────────
BPW = {1: 1.34, 2: 2.34, 3: 3.34, 4: 4.50}
BITS_CLIMB = [1, 2, 3, 4]                 # floor-search climbs in this order, stops at floor

# ── quality thresholds (healed Δ ppl over f16), both reported ─────────────────────────
NEAR_1to1 = 0.02                          # ≤ +2 %  = true near-lossless ("have your cake")
WIN       = 0.08                          # ≤ +8 %  = beats llama Q4_K's degradation @ lower bpw

# ── "extract the most" intensify grid (applied only at the floor, where it matters) ────
ALPHAS_COARSE = [0.5]
ALPHAS_FINE   = [0.25, 0.5, 0.75]         # AWQ alpha-sweep at the margin

# ── RECIPES — the floor-search climbs EFFECTIVE bpw across BOTH methods, stops at 1:1 ──
# single   = AWQ 1-bake (the base; serves via the existing GPU bitslice .tq path).
# residual = STRAND_b1 + STRAND_b2(W − STRAND_b1) — full-rank, codec-native, train-free
#            (commit 3bc128a; 0.5B 3+2 = +1.6%≈1:1, 2+2 = +8.9% beats Q4_K). Costs +b2 bpw, so
#            it buys quality where there's headroom; the extreme-fit frontier (405B ≤1.34 bpw)
#            can't afford it → single-bit viability stays THE frontier question.
# NOTE residual SERVE = two-part .tq (decode base+residual, sum in GEMV) — not yet built; today
#      residual is the QUALITY (Stream-A) ceiling-breaker, single-bake is the SERVE (Stream-B) path.
# (future) "awq+residual" = residual on the AWQ base — the 7B chat's active next step.
def eff_bpw(recipe):
    kind, a, b = recipe
    return BPW[a] if kind == "single" else round(BPW[a] + BPW[b], 2)

RECIPES = sorted([
    ("single",   1, None),    # ~1.34 — extreme-fit frontier (405B/671B); serves
    ("residual", 1, 1),       # ~2.68
    ("single",   2, None),    # ~2.34 — serves
    ("residual", 2, 1),       # ~3.68
    ("single",   3, None),    # ~3.34 — serves
    ("residual", 2, 2),       # ~4.68 — measured 0.5B +8.9% (beats Q4_K @4.5)
    ("single",   4, None),    # ~4.50 — llama Q4_K reference density; serves
    ("residual", 3, 2),       # ~5.68 — measured 0.5B +1.6% (≈1:1)
], key=eff_bpw)

def recipe_label(recipe):
    kind, a, b = recipe
    return f"awq{a}b" if kind == "single" else f"res{a}+{b}b"

def serves(recipe):
    return recipe[0] == "single"          # single-bake has a built .tq serve path today

# ── hardware envelope (M1 Ultra, the DELIVERED box: 128 GB unified, ~800 GB/s, 8 TB) ────
# Re-derived from the M2-Max-96GB constants the plan was written against (see
# M1ULTRA_POTENTIAL_AUDIT.md §6 re-derivation 1). The bigger envelope moves 405B@1.34 and
# 671B@1.0 out of TIGHT/EDGE into RESIDENT with no expert pager.
RAM_GB        = 128.0
WEIGHT_BUDGET = 112.0                      # leave ~16 GB for KV + activations + OS
CONDENSE_RESIDENT_MAX_B = 48              # naive condense cap (f16 ≈ 2×params must fit ~112 GB) -> ~48B


def tq_gb(params_b, bpw):
    return params_b * bpw / 8.0


def f16_gb(params_b):
    return params_b * 2.0


def serve_fits(params_b, bpw, budget=WEIGHT_BUDGET):
    return tq_gb(params_b, bpw) <= budget


def serve_headroom_bpw(params_b):
    """The HIGHEST bit-rung bpw at which this model still serves on 96 GB = its quality
    headroom / constraint. 32B → 4.50 (unconstrained); 405B → 1.34 (1-bit only); a model
    that needs the sub-rung 1.0 edge → 1.0; None if even 1.0 bpw overflows."""
    for bits in (4, 3, 2, 1):
        if serve_fits(params_b, BPW[bits]):
            return BPW[bits]
    return 1.0 if tq_gb(params_b, 1.0) <= WEIGHT_BUDGET else None


def condense_tier(params_b, has_f16_source=True):
    if not has_f16_source:
        return "serve_only"
    if params_b <= CONDENSE_RESIDENT_MAX_B:
        return "resident"          # phase 1: f16 fits, naive AWQ + doctor
    return "streamed"              # phase 2: block-wise, one block resident at a time


# ── THE LADDER ────────────────────────────────────────────────────────────────────────
# priority: P0 spine · P1 cross-family · P2 100B+/MoE · P3 1-bit frontier.
# active_b = active params for MoE (drives tps; total params drive footprint).
# keep_f16: None → auto (keep ≤7B, purge >7B to respect 1 TB).
def M(family, name, hf_id, params_b, priority, active_b=None, keep_f16=None, note=""):
    return dict(family=family, name=name, hf_id=hf_id, params_b=params_b, priority=priority,
                active_b=active_b, keep_f16=keep_f16, note=note)

MODELS = [
    # ── P0 — Qwen2.5 spine (clean 144× scaling curve) ──────────────────────────────────
    M("qwen2.5", "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, 0, note="hidden 896 → serve-invariant FAILS; condense dev-probe only"),
    M("qwen2.5", "qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, 0),
    M("qwen2.5", "qwen2.5-3b",   "Qwen/Qwen2.5-3B-Instruct",   3.0, 0),
    M("qwen2.5", "qwen2.5-7b",   "Qwen/Qwen2.5-7B-Instruct",   7.6, 0),
    M("qwen2.5", "qwen2.5-14b",  "Qwen/Qwen2.5-14B-Instruct", 14.8, 0),
    M("qwen2.5", "qwen2.5-32b",  "Qwen/Qwen2.5-32B-Instruct", 32.5, 0),
    M("qwen2.5", "qwen2.5-72b",  "Qwen/Qwen2.5-72B-Instruct", 72.7, 0, note="72B FFN in_features 29568 % 256 ≠ 0 → verify serve kernel (pad/%128)"),
    # ── P1 — Llama-3.x ladder (cross-family generality) ────────────────────────────────
    M("llama3", "llama3.2-1b",   "meta-llama/Llama-3.2-1B-Instruct",   1.2, 1),
    M("llama3", "llama3.2-3b",   "meta-llama/Llama-3.2-3B-Instruct",   3.2, 1),
    M("llama3", "llama3.1-8b",   "meta-llama/Llama-3.1-8B-Instruct",   8.0, 1),
    M("llama3", "llama3.3-70b",  "meta-llama/Llama-3.3-70B-Instruct", 70.6, 1),
    # ── P1 — Gemma-2 ───────────────────────────────────────────────────────────────────
    M("gemma2", "gemma2-2b",     "google/gemma-2-2b-it",   2.6, 1),
    M("gemma2", "gemma2-9b",     "google/gemma-2-9b-it",   9.2, 1),
    M("gemma2", "gemma2-27b",    "google/gemma-2-27b-it", 27.2, 1),
    # ── P1 — Mistral ───────────────────────────────────────────────────────────────────
    M("mistral", "mistral-7b",   "mistralai/Mistral-7B-Instruct-v0.3", 7.2, 1),
    M("mistral", "mistral-nemo", "mistralai/Mistral-Nemo-Instruct-2407", 12.2, 1),
    M("mistral", "mistral-small","mistralai/Mistral-Small-24B-Instruct-2501", 23.6, 1),
    # ── P1 — Phi ───────────────────────────────────────────────────────────────────────
    M("phi", "phi3.5-mini",      "microsoft/Phi-3.5-mini-instruct", 3.8, 1),
    M("phi", "phi3-medium",      "microsoft/Phi-3-medium-4k-instruct", 14.0, 1),
    # ── P2 — 100B+ / MoE (serve-stream + phase-2 block-wise condense) ──────────────────
    M("qwen3", "qwen3-30b-a3b",  "Qwen/Qwen3-30B-A3B", 30.5, 2, active_b=3.3, note="MoE: total 30B fits, decodes like ~3B"),
    M("qwen3", "qwen3-235b",     "Qwen/Qwen3-235B-A22B", 235.0, 2, active_b=22.0, keep_f16=False,
      note="MoE dream case: 235B total @2-bit ≈69GB fits, decodes like ~22B; condense=phase-2 block-wise"),
    M("gptoss", "gpt-oss-20b",   "openai/gpt-oss-20b", 20.9, 2, active_b=3.6),
    M("gptoss", "gpt-oss-120b",  "openai/gpt-oss-120b", 116.8, 2, active_b=5.1, keep_f16=False,
      note="120B @≤3-bit fits; phase-2 block-wise condense"),
    M("deepseek", "deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite-Chat", 15.7, 2, active_b=2.4),
    # ── P3 — the 1-bit frontier (gated on 1-bit viable at scale) ───────────────────────
    M("llama3", "llama3.1-405b",  "meta-llama/Llama-3.1-405B-Instruct", 405.0, 3, keep_f16=False,
      note="FRONTIER: serves ONLY at ≤1.34 bpw (68GB). Needs 1-bit viable + block-wise condense"),
    M("deepseek", "deepseek-v3",  "deepseek-ai/DeepSeek-V3", 671.0, 3, active_b=37.0, keep_f16=False,
      note="EDGE: 1.0 bpw ≈ 84GB = absolute ceiling of the box; MoE active 37B"),
]


def keep_f16(m):
    if m["keep_f16"] is not None:
        return m["keep_f16"]
    return m["params_b"] <= 7.5          # auto: keep small (cheap re-run), purge big (disk)


def cells(model):
    """The floor-search recipes for one model, ascending effective bpw (climb → stop at 1:1)."""
    p = model["params_b"]
    out = []
    for r in RECIPES:
        eb = eff_bpw(r)
        out.append(dict(model=model["name"], recipe=recipe_label(r), kind=r[0],
                        eff_bpw=eb, tq_gb=round(tq_gb(p, eb), 1),
                        serve_fits=serve_fits(p, eb), serves=serves(r)))
    return out


def _fmt(x):
    return f"{x:5.1f}" if x is not None else "  -  "


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--fit":
        p = float(sys.argv[2])
        print(f"# serve footprint for {p}B (weights = params×bpw/8, budget {WEIGHT_BUDGET} GB)")
        for bits in BITS_CLIMB:
            g = tq_gb(p, BPW[bits])
            print(f"  {bits}-bit ({BPW[bits]:.2f} bpw): {g:6.1f} GB  {'FITS' if g<=WEIGHT_BUDGET else 'no'}")
        print(f"  f16 (condense resident): {f16_gb(p):6.1f} GB  tier={condense_tier(p)}")
        return
    if arg == "--tsv":
        print("family\tmodel\tparams_b\tactive_b\tprio\tf16_gb\ttier\ttq@1\ttq@2\ttq@3\tserve_max_bpw")
        for m in MODELS:
            p = m["params_b"]
            print(f"{m['family']}\t{m['name']}\t{p}\t{m['active_b'] or ''}\t{m['priority']}\t"
                  f"{f16_gb(p):.0f}\t{condense_tier(p)}\t{tq_gb(p,1.34):.0f}\t{tq_gb(p,2.34):.0f}\t"
                  f"{tq_gb(p,3.34):.0f}\t{serve_headroom_bpw(p)}")
        return
    if arg == "--plan":
        for m in MODELS:
            tier = condense_tier(m["params_b"])
            print(f"\n# {m['name']}  ({m['params_b']}B, P{m['priority']}, condense={tier}"
                  f"{', '+m['note'] if m['note'] else ''})")
            for c in cells(m):
                fit = "fits✓" if c["serve_fits"] else "fits✗(96GB)"
                sv = "" if c["serves"] else " [residual serve pending]"
                print(f"    {c['recipe']:9s} eff~{c['eff_bpw']:.2f}bpw  .tq≈{c['tq_gb']}GB  {fit}{sv}")
        return
    # default: summary
    print(f"# Hawking condense ladder — {len(MODELS)} models, RAM {RAM_GB:.0f} GB, "
          f"weight budget {WEIGHT_BUDGET:.0f} GB")
    print(f"# naive-condense ≤{CONDENSE_RESIDENT_MAX_B}B · serve ≤~200B@3bit ≤~285B@2bit ≤~500B@1.34bit")
    for pr in (0, 1, 2, 3):
        ms = [m for m in MODELS if m["priority"] == pr]
        tag = {0: "P0 Qwen2.5 spine", 1: "P1 cross-family", 2: "P2 100B+/MoE", 3: "P3 1-bit frontier"}[pr]
        print(f"\n{tag}:")
        for m in ms:
            hr = serve_headroom_bpw(m["params_b"])
            hrs = "unconstr." if hr and hr >= 4.5 else (f"≤{hr}bpw" if hr else "1.0-edge")
            print(f"  {m['name']:18s} {m['params_b']:6.1f}B  condense={condense_tier(m['params_b']):8s}"
                  f"  serve-max={hrs:9s}"
                  + (f"  (MoE active {m['active_b']}B)" if m["active_b"] else ""))


if __name__ == "__main__":
    main()
