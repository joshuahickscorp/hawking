#!/usr/bin/env python3.12
"""auto_bits.py — AUTO MODE: from an uncompressed model, recommend the bit format AND the serve regime,
so a user doesn't have to guess. Answers "what's the best bpw for THIS model on MY device?" and
"how will it even run?" in one shot. The recommendation is a STARTING point the studio run then
confirms/falls-back with the real NIAH/ppl gate (exactly: try 2-bit; if it can't hold quality, step
to 3-bit) - never a silent claim.

How it decides the bpw (cheap, no full bake):
  - an official, deployable, quality-passing floor receipt may select a measured product rung;
  - SUBBIT-0 theory reports never select a rung or become a measured hard floor;
  - otherwise use the redundancy-hypothesis heuristic floor(N) ~ clamp(4.0 - 0.8*log10(params_B*1e9/1e9)...)
    -> bigger models are recommended lower bpw (they carry more redundancy), snapped to the ladder
    {1:1.34, 2:2.34, 3:3.34, 4:4.5};
  - then adjust the bpw for the device: raise if storage would overflow; for frontier MoE models,
    prefer a lower sub-bit rung when that is what turns MOE-PAGED into fully RESIDENT.

How it decides the regime: delegates to size_frontier (RESIDENT / MOE-PAGED / DENSE-OOC) for the
device budget. If dense + out-of-core would be sub-0.1 tok/s, it says so and suggests a smaller/ MoE
alternative. Advisor only; the bake + gate is the studio run. KILL: if even 4-bit + out-of-core
overflows the device storage, the model does not fit this device at any quality (need bigger storage).
"""
import sys, os, json, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import size_frontier as SF
except Exception:
    SF = None
LADDER = [(1, 1.34), (2, 2.34), (3, 3.34), (4, 4.5)]
SUBBIT_RUNG = [(1, 1.34), (1, 1.00), (1, 0.75), (1, 0.50), (1, 0.33)]
OUT = "reports/condense"


def heuristic_floor_bpw(params_b):
    """Redundancy-hypothesis STARTING floor: bigger -> lower. Snap to the ladder. Confirmed by the run."""
    x = math.log10(max(0.5, params_b))                 # 0.5B->-0.3, 7B->0.85, 70B->1.85, 700B->2.85
    raw = 3.6 - 0.9 * x                                 # ~3.9 @0.5B, ~2.8 @7B, ~1.9 @70B, ~1.1 @700B
    for bits, bpw in LADDER:                            # snap UP to the nearest ladder rung
        if bpw >= raw:
            return bits, bpw
    return 4, 4.5


def measured_floor(label):
    """Return a measured floor only from an explicit deployable official receipt.

    The similarly named ``*_subbit0.json`` file is deliberately never consulted: SUBBIT-0 is an
    entropy/theory record, not a codec artifact or quality result.  Requiring the official receipt
    location and fail-closed product fields also prevents a hand-edited theory report from becoming a
    hard capacity recommendation merely by setting one boolean.
    """
    p = f"receipts/official/{label}-floor.json"
    if os.path.exists(p):
        try:
            with open(p) as handle:
                d = json.load(handle)
            if not (
                d.get("project") == "hawking"
                and d.get("receipt_version") == "0.2"
                and d.get("deployable") is True
                and d.get("quality_gate") == "pass"
                and d.get("claim_type") in {"density", "scale-point"}
            ):
                return None
            floor = d.get("effective_bpw")
            if isinstance(floor, bool) or not isinstance(floor, (int, float)) \
                    or not math.isfinite(float(floor)) or float(floor) <= 0:
                return None
            return float(floor)
        except (OSError, ValueError, TypeError):
            return None
    return None


def recommend(total_b, active_b, arch, label, device="m1ultra"):
    bits, bpw = heuristic_floor_bpw(total_b)
    mfloor = measured_floor(label)
    src = "heuristic(redundancy-law)"
    if mfloor:
        for b, bp in LADDER:
            if bp >= mfloor:
                bits, bpw, src = b, bp, "measured(official deployable floor receipt)"; break
    # Raise bpw until it at least fits storage on the device; then, for huge MoE models, prefer
    # a lower sub-bit rung if that is what keeps the entire artifact resident on the Studio.
    regime, chosen = None, None
    for b, bp in [(bits, bpw)] + [(b, bp) for b, bp in LADDER if bp > bpw]:
        r = SF.analyze(total_b, active_b, bp, device) if SF else {"best_regime": "?", "fits_ssd": True}
        if r.get("best_regime") != "TOO-BIG":
            chosen, regime = (b, bp), r; break
    if not chosen:
        chosen, regime = (4, 4.5), (SF.analyze(total_b, active_b, 4.5, device) if SF else {"best_regime": "TOO-BIG"})
    prefer_resident = os.environ.get("AUTO_BITS_PREFER_RESIDENT", "1") != "0"
    if prefer_resident and active_b and regime.get("best_regime") != "RESIDENT":
        for b, bp in SUBBIT_RUNG:
            if bp > chosen[1]:
                continue
            r = SF.analyze(total_b, active_b, bp, device) if SF else {"best_regime": "?", "fits_ssd": True}
            if r.get("best_regime") == "RESIDENT":
                chosen, regime = (b, bp), r
                src += "+resident-fit"
                break
    b, bp = chosen
    rec = {
        "model": label, "total_b": total_b, "active_b": active_b, "arch": arch,
        "recommended_bits": b, "recommended_bpw": bp, "bpw_source": src,
        "fallback_ladder": "if the run's NIAH/ppl gate fails at this bpw, step UP one rung (2->3->4)",
        "serve_regime": regime.get("best_regime"), "resident_gb": regime.get("resident_gb"),
        "tok_s": regime.get("est_tok_s", regime.get("est_tok_s_cold")),
        "tq_on_disk_gb": regime.get("tq_on_disk_gb"), "device": device,
        "advisor_only": True, "confirmed_by": "studio_run.py go (real bake + NIAH/ppl gate)",
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{label}_autobits.json", "w"), indent=2)
    print(f"[auto] {label}: try {b}-bit ({bp} eff-bpw, {src}) | regime {rec['serve_regime']} "
          f"resident~{rec['resident_gb']}GB tok/s~{rec['tok_s']} | fallback: step up if gate fails",
          file=sys.stderr)
    if rec["serve_regime"] == "TOO-BIG":
        print(f"# KILL: even 4-bit out-of-core overflows {device} storage — needs bigger/faster storage",
              file=sys.stderr)
    return rec


def _from_dir(model_dir, device):
    c = json.load(open(os.path.join(model_dir, "config.json")))
    arch = (c.get("architectures") or ["?"])[0]
    n_experts = c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts")
    # rough total/active from config when present; else caller passes --params
    label = os.path.basename(model_dir.rstrip("/"))
    return arch, n_experts, label


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    dev = sys.argv[sys.argv.index("--device")+1] if "--device" in sys.argv else "m1ultra"
    if a == "--params":
        total = float(sys.argv[2])
        active = float(sys.argv[sys.argv.index("--active")+1]) if "--active" in sys.argv else None
        lbl = sys.argv[sys.argv.index("--label")+1] if "--label" in sys.argv else f"{int(total)}b"
        recommend(total, active, "MoE" if active else "dense", lbl, dev)
    elif a == "--help":
        print(__doc__)
    else:  # model dir: read config; params must be passed for total/active (config rarely has a clean total)
        arch, n_exp, lbl = _from_dir(a, dev)
        total = float(sys.argv[sys.argv.index("--params")+1]) if "--params" in sys.argv else 7.0
        active = float(sys.argv[sys.argv.index("--active")+1]) if "--active" in sys.argv else None
        recommend(total, active, arch, lbl, dev)
