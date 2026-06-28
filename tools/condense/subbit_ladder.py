#!/usr/bin/env python3.12
"""subbit_ladder.py — the SUB-1-BIT rung math + feasibility table (a PROBE, not a serve win).

The sibling of tools/condense/ladder.py. Where ladder.py stops at the 1.34-bpw "1-bit" rung
(the lowest rung with a BUILT GPU bitslice .tq serve path), THIS file extends the rung set DOWN
into the sub-1-bit territory that the studio_maximization "crazy ladder" sketches — 1.00, 0.75,
0.50, 0.33 effective bpw — and encodes, as runnable code, the feasibility classes that decide
which (model, eff-bpw) cells are PRODUCT, RESEARCH, FANTASY, or physically BELOW-FLOOR.

WHAT THIS TOOL IS: a size/fit/feasibility CALCULATOR. Given any (params, eff-bpw) it computes the
artifact size, whether it serves on an 84 GB weight budget, and its lane. It does NOT bake, doctor,
measure ppl, or claim a serving win — the serve paths below 1.34 bpw are UNBUILT and the recovery
at these rungs is UNPROVEN. The "PRODUCT" lane means "the math + a built codec rung exist", NOT
"this artifact ships at near-1:1 quality today". Treat every sub-1-bit number as a PROBE.

EFFECTIVE BPW DISCIPLINE (matches ladder.py / audit_ladder.py): rungs here are EFFECTIVE bpw,
i.e. they already include side-info (trellis scales + outlier-channel positions + residual-pass
overhead — the baker's AGGREGATE number). We never report a nominal payload bpw. The hard physical
fact this tool enforces: a DENSE codec cannot encode a weight matrix below the side-info floor
(~0.28 eff-bpw measured ~0.25-0.31), because the scales/positions ALONE cost that many bits even
if every weight were free. A dense rung below that floor is BELOW-FLOOR = impossible. A MoE model's
AMORTIZED-over-total bpw MAY sit below the floor only because the active experts carry real bits and
the dormant experts dilute the average — the active slice itself is still floor-bound.

KILL LINE (the criterion that refutes the sub-1-bit lever):
  If, on a 7B+ model, the lowest EFFECTIVE-bpw config that holds <=+2% ppl (multiwindow, after the
  full doctor stack) lands at or ABOVE ~1.34 bpw — i.e. no measured config ever crosses the 1.34
  line at near-1:1 — then every rung below 1.34 here is FANTASY/RESEARCH theater and the sub-1-bit
  ladder is DEAD as a product axis. This file computes sizes; scaling_law.py --fit decides the kill.

CLI:
  python subbit_ladder.py                 # summary: rungs x models, lane per cell
  python subbit_ladder.py --tsv           # flat table (model, params, active, per-rung gb + class + lane)
  python subbit_ladder.py --fit <p_b> [--active A]   # footprint per rung + the bpw that just fits 84/70/30GB
  python subbit_ladder.py --dream         # the headline callouts (671B@1.0, 744B@0.33, 235B-A22B@1.34)
  python subbit_ladder.py --selftest      # run --tsv + --dream and assert the anchor numbers
"""
import sys

# ── hardware envelope (matches ladder.py) ─────────────────────────────────────────────
WEIGHT_BUDGET = 84.0          # serve weight budget on the 96 GB box (leave headroom for KV/acts/OS)
SERVE_COMFY   = 70.0          # <= this = comfortable (room for long-ctx KV)
SERVE_DREAM   = 30.0          # the "fits on a 32 GB laptop class" callout threshold

# ── the SUB-1-BIT ladder (EFFECTIVE bpw, side-info included) ───────────────────────────
# 4.50/3.34/2.34/1.34 are ladder.py's built rungs (BPW dict); below 1.34 is the new frontier.
# 1.00 = the absolute edge that makes 671B fit 84 GB. 0.33 ~= ternary's MoE-amortized dream.
RUNGS = [4.50, 3.34, 2.34, 1.34, 1.00, 0.75, 0.50, 0.33]

# rungs at/above this have a BUILT GPU bitslice .tq serve path (ladder.py serves()=single-bake)
BUILT_SERVE_FLOOR = 1.34

# ── the side-info floor — the hard physical wall for DENSE codecs ──────────────────────
# Measured aggregate side-info (trellis scales + outlier positions + residual overhead) is
# ~0.25-0.31 eff-bpw across the 7B bakes; we take ~0.28 as the floor. A DENSE rung below this is
# physically impossible: the scales/positions cost this many bits even at zero weight payload.
SIDE_INFO_FLOOR = 0.28
SIDE_INFO_RANGE = (0.25, 0.31)

# ── the quality gate (echoed from scaling_law.GATE; this file doesn't measure, only cites) ─
GATE_PCT = 2.0                # <= +2% ppl vs f16 parent = the ~1:1 "floor held" bar

# ── the models (footprint = TOTAL params; tps ~ ACTIVE params for MoE) ─────────────────
# (family, name, total_b, active_b or None for dense, note)
def M(family, name, total_b, active_b=None, note=""):
    return dict(family=family, name=name, total_b=total_b, active_b=active_b, note=note)

MODELS = [
    M("qwen2.5", "Qwen2.5-32B",      32.5,  None, "dense; unconstrained on the box"),
    M("qwen2.5", "Qwen2.5-72B",      72.7,  None, "dense; serve-tight even at 4.5 bpw"),
    M("llama3",  "Llama3.3-70B",     70.6,  None, "dense; the cross-family 70B point"),
    M("qwen3",   "Qwen3-235B-A22B",  235.0, 22.0, "MoE: 235B footprint, decodes like ~22B"),
    M("llama3",  "Llama3.1-405B",    405.0, None, "dense FRONTIER: needs <=1.0 bpw to even fit"),
    M("deepseek","DeepSeek-V3",      671.0, 37.0, "MoE: 671B footprint @1.0 ~= 84GB = box edge; active 37B"),
    M("glm",     "GLM-744B",         744.0, 32.0, "MoE: largest tail; active ~32B; the 0.33 dream"),
]


# ── size math (artifact_gb matches ladder.py tq_gb: params * bpw / 8) ──────────────────
def tq_gb(params_b, bpw):
    return params_b * bpw / 8.0


def serve_class(gb):
    """SERVE-COMFY (<=70GB) / SERVE-TIGHT (70-84) / SERVE-OVERFLOW (>84)."""
    if gb <= SERVE_COMFY:
        return "SERVE-COMFY"
    if gb <= WEIGHT_BUDGET:
        return "SERVE-TIGHT"
    return "SERVE-OVERFLOW"


def bpw_that_fits(params_b, budget):
    """The HIGHEST EFFECTIVE bpw at which `params_b` still fits `budget` GB (params*bpw/8<=budget).
    Inverse of tq_gb: bpw <= 8*budget/params. None if even the smallest rung overflows."""
    cap = 8.0 * budget / params_b
    for r in RUNGS:                       # rungs are descending; first <= cap is the highest that fits
        if r <= cap:
            return r
    return None                           # even 0.33 overflows -> nothing fits this budget


def below_floor(bpw, is_moe):
    """A rung is BELOW-FLOOR (physically impossible) when it's under the dense side-info floor AND
    the model is dense. A MoE may amortize below the floor (active experts carry the bits, dormant
    experts dilute the per-total average) — so MoE is NOT auto-killed by the dense floor."""
    return (bpw < SIDE_INFO_FLOOR) and not is_moe


# ── the lane verdict (the studio_maximization "crazy ladder" classification) ───────────
# Index each sub-1-bit rung as SUBBIT-N by descending bpw (the studio enumeration):
#   SUBBIT-0 = 1.34 (built serve), 1 = 1.00, 2 = 0.75, 3 = 0.50, 4 = 0.33  ... plus the studio's
#   extended-research rungs. The studio verdict: SUBBIT-0/1/5 are PRODUCT-lane (a built/near-built
#   serve rung + a plausible recovery story), SUBBIT-2/3/4 are RESEARCH-lane (the MDL prune+quant /
#   codec-native frontier — unbuilt, unproven), SUBBIT-6/7 (and ANY dense rung under the floor) are
#   FANTASY-lane (below where any measured config has reached, or physically impossible).
# Concretely on THIS rung set:
SUBBIT_INDEX = {1.34: 0, 1.00: 1, 0.75: 2, 0.50: 3, 0.33: 4}
PRODUCT_SUBBIT  = {0, 1, 5}
RESEARCH_SUBBIT = {2, 3, 4}
FANTASY_SUBBIT  = {6, 7}


def lane(bpw, is_moe):
    """PRODUCT / RESEARCH / FANTASY / BELOW-FLOOR for a (rung, model-kind) cell.

    BELOW-FLOOR overrides everything for a DENSE rung under the side-info floor (impossible).
    Rungs at/above 1.34 are PRODUCT (the built/serve tier). Sub-1-bit rungs map by SUBBIT index
    to the studio lanes. A MoE whose per-total rung dips under the floor stays RESEARCH/FANTASY
    by its index (it is not impossible, just unproven) rather than BELOW-FLOOR."""
    if below_floor(bpw, is_moe):
        return "BELOW-FLOOR"
    if bpw >= BUILT_SERVE_FLOOR:          # 1.34, 2.34, 3.34, 4.50 — the built serve rungs
        return "PRODUCT"
    idx = SUBBIT_INDEX.get(bpw)
    if idx is None:
        return "RESEARCH"                 # an off-grid sub-1-bit rung defaults to research
    if idx in PRODUCT_SUBBIT:
        return "PRODUCT"
    if idx in RESEARCH_SUBBIT:
        return "RESEARCH"
    return "FANTASY"


def serves_today(bpw):
    """Honest: is there a BUILT .tq serve path at this rung? (>=1.34 only.) Everything below is
    a probe — the sub-1-bit serve kernels are UNBUILT, this tool measures size, not a serve win."""
    return bpw >= BUILT_SERVE_FLOOR


def is_moe(model):
    return model["active_b"] is not None


# ── per-model cells ────────────────────────────────────────────────────────────────────
def cells(model):
    p = model["total_b"]
    moe = is_moe(model)
    out = []
    for r in RUNGS:
        gb = tq_gb(p, r)
        out.append(dict(bpw=r, gb=round(gb, 1), serve=serve_class(gb),
                        lane=lane(r, moe), serves_today=serves_today(r)))
    return out


def _moe_tag(model):
    return f"MoE act {model['active_b']}B" if is_moe(model) else "dense"


# ── CLI handlers ───────────────────────────────────────────────────────────────────────
def cmd_tsv():
    cols = "\t".join(f"gb@{r}" for r in RUNGS)
    lanes = "\t".join(f"lane@{r}" for r in RUNGS)
    print(f"family\tmodel\ttotal_b\tactive_b\tkind\t{cols}\tfit84\tfit70\tfit30\t{lanes}")
    for m in MODELS:
        p = m["total_b"]
        gbs = "\t".join(f"{tq_gb(p, r):.1f}" for r in RUNGS)
        lns = "\t".join(lane(r, is_moe(m)) for r in RUNGS)
        f84 = bpw_that_fits(p, WEIGHT_BUDGET)
        f70 = bpw_that_fits(p, SERVE_COMFY)
        f30 = bpw_that_fits(p, SERVE_DREAM)
        print(f"{m['family']}\t{m['name']}\t{p}\t{m['active_b'] or ''}\t{_moe_tag(m)}\t{gbs}\t"
              f"{f84 if f84 else 'none'}\t{f70 if f70 else 'none'}\t{f30 if f30 else 'none'}\t{lns}")


def cmd_fit(params_b, active_b=None):
    moe = active_b is not None
    print(f"# subbit footprint for {params_b}B "
          f"({'MoE active '+str(active_b)+'B' if moe else 'dense'}), "
          f"weight budget {WEIGHT_BUDGET:.0f} GB  [PROBE — sub-1.34-bpw serve is UNBUILT]")
    print(f"# artifact_gb = total_params * eff_bpw / 8 ; footprint=total, tps~active")
    for r in RUNGS:
        gb = tq_gb(params_b, r)
        bf = "  BELOW-FLOOR(dense impossible)" if below_floor(r, moe) else ""
        print(f"  {r:4.2f} bpw: {gb:7.1f} GB  {serve_class(gb):13s} {lane(r, moe):11s}"
              f"{'' if serves_today(r) else ' [no built serve path]'}{bf}")
    for budget, tag in ((WEIGHT_BUDGET, "84GB box"), (SERVE_COMFY, "70GB comfy"), (SERVE_DREAM, "30GB laptop")):
        b = bpw_that_fits(params_b, budget)
        if b is None:
            print(f"  just-fits {tag:12s}: none on this rung set (even {RUNGS[-1]} bpw overflows)")
        else:
            print(f"  just-fits {tag:12s}: <= {b:.2f} eff-bpw  ({tq_gb(params_b, b):.1f} GB){'' if serves_today(b) else '  [probe rung]'}")


def cmd_dream():
    print("# Headline sub-1-bit callouts (PROBE math — sizes only, no serve-win claim):")
    callouts = [
        ("DeepSeek-V3 671B",      671.0, 1.00, 37.0),
        ("GLM-744B",              744.0, 0.33, 32.0),
        ("Qwen3-235B-A22B",       235.0, 1.34, 22.0),
    ]
    for name, p, bpw, act in callouts:
        gb = tq_gb(p, bpw)
        moe = act is not None
        print(f"  {name:22s} @ {bpw:.2f} eff-bpw = {gb:5.1f} GB  "
              f"{serve_class(gb):13s} {lane(bpw, moe):11s}  (MoE active {act}B)"
              f"{'' if serves_today(bpw) else '  [serve path UNBUILT]'}")
    print(f"# Side-info floor: dense codec cannot go below ~{SIDE_INFO_FLOOR} eff-bpw "
          f"(measured {SIDE_INFO_RANGE[0]}-{SIDE_INFO_RANGE[1]}); MoE may amortize lower (active experts carry the bits).")
    print(f"# KILL: if no 7B+ config holds <=+{GATE_PCT}% ppl below {BUILT_SERVE_FLOOR} bpw, every sub-1-bit rung is fantasy.")


def cmd_summary():
    print(f"# subbit_ladder — sub-1-bit rung math (PROBE; sizes/fit/lane, NOT a serve-win claim)")
    print(f"# budget {WEIGHT_BUDGET:.0f}GB · built serve floor {BUILT_SERVE_FLOOR} bpw · "
          f"dense side-info floor ~{SIDE_INFO_FLOOR} bpw (measured {SIDE_INFO_RANGE[0]}-{SIDE_INFO_RANGE[1]})")
    print(f"# lanes: PRODUCT(built/near) · RESEARCH(unbuilt frontier) · FANTASY(beyond reach) · "
          f"BELOW-FLOOR(dense impossible)")
    hdr = "model".ljust(20) + "  total  " + "  ".join(f"{r:>5.2f}" for r in RUNGS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in MODELS:
        p = m["total_b"]
        row = m["name"].ljust(20) + f"  {p:5.0f}B "
        marks = []
        for r in RUNGS:
            gb = tq_gb(p, r)
            ln = lane(r, is_moe(m))
            sym = {"PRODUCT": " P ", "RESEARCH": " R ", "FANTASY": " F ", "BELOW-FLOOR": " x "}[ln]
            # mark serve-overflow rungs with a trailing '!'
            sym = sym if gb <= WEIGHT_BUDGET else sym.rstrip() + "!"
            marks.append(f"{sym:>5s}")
        print(row + "  ".join(marks))
    print("\n# legend: P=product R=research F=fantasy x=below-floor  '!'=overflows 84GB")
    print("# fit-on-84GB (highest eff-bpw that serves):")
    for m in MODELS:
        b = bpw_that_fits(m["total_b"], WEIGHT_BUDGET)
        print(f"    {m['name']:20s} <= {b if b else 'none':>4} bpw   ({_moe_tag(m)})")
    print(f"\n# KILL: if no 7B+ config holds <=+{GATE_PCT}% ppl below {BUILT_SERVE_FLOOR} bpw "
          f"(scaling_law.py --fit), the whole sub-1-bit ladder is dead.")


def cmd_selftest():
    """Synthetic, runs entirely here (touches no model). Asserts the anchor numbers from the
    prompt: 671@1.0 ~ 83.9GB, 744@0.33 ~ 30.7GB, 235@1.34 ~ 39GB, plus lane/floor invariants."""
    ok = True
    def check(name, got, want, tol=0.15):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.2f}, want ~{want:.2f}")

    check("671B @ 1.00 bpw GB", tq_gb(671.0, 1.00), 83.9)
    check("744B @ 0.33 bpw GB", tq_gb(744.0, 0.33), 30.7)
    check("235B @ 1.34 bpw GB", tq_gb(235.0, 1.34), 39.4, tol=0.5)

    # lane invariants
    inv = [
        ("671B@1.00 serves the box (TIGHT)", serve_class(tq_gb(671.0, 1.00)) == "SERVE-TIGHT"),
        ("671B@1.00 is PRODUCT (SUBBIT-1)", lane(1.00, True) == "PRODUCT"),
        ("0.75 dense rung is RESEARCH", lane(0.75, False) == "RESEARCH"),
        ("0.50 dense rung is RESEARCH", lane(0.50, False) == "RESEARCH"),
        ("0.33 rung NOT below floor (>0.28)", not below_floor(0.33, False)),
        ("0.20 dense rung is BELOW-FLOOR", lane(0.20, False) == "BELOW-FLOOR"),
        ("0.20 MoE rung is NOT below-floor", lane(0.20, True) != "BELOW-FLOOR"),
        ("1.34 is the built serve floor", serves_today(1.34) and not serves_today(1.00)),
        ("405B needs <=1.34 to fit 84GB", (bpw_that_fits(405.0, WEIGHT_BUDGET) or 9) <= 1.34),
        ("32B fits at top rung 4.50", bpw_that_fits(32.5, WEIGHT_BUDGET) == 4.50),
    ]
    for name, cond in inv:
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print(f"\n# --tsv smoke + --dream smoke:")
    cmd_tsv()
    print()
    cmd_dream()
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return
    if arg == "--tsv":
        cmd_tsv()
        return
    if arg == "--fit":
        p = float(sys.argv[2])
        act = None
        if "--active" in sys.argv:
            act = float(sys.argv[sys.argv.index("--active") + 1])
        cmd_fit(p, act)
        return
    if arg == "--dream":
        cmd_dream()
        return
    if arg == "--selftest":
        sys.exit(0 if cmd_selftest() else 1)
    cmd_summary()


if __name__ == "__main__":
    main()
