#!/usr/bin/env python3
"""Global bit allocator: descend by marginal functional damage per bit removed.

Not "pick a bit width and apply it everywhere". Two facts make uniform allocation
provably wrong on this model:

  q_inject spans 16.1x with depth (1.597e-04 at L0 to 2.577e-03 at L63). The same
  relative weight error injected late costs the residual stream ~16x more.

  damage per bit is not linear and differs per tensor class, so the cheapest next
  bit is rarely in the tensor you last took one from.

So allocation is a descent: score every (tensor, bit width) cell, weight its damage
by where it sits in the network, then repeatedly remove the bit with the lowest
damage-per-byte until the budget is met.

Honesty constraints baked in:
  - damage is measured with the adequacy gate, which is probe-inclusive, so a cell
    cannot look good by hiding in the capture's nullspace.
  - tensors whose input activations were never captured (down_proj takes the 17408
    intermediate, which the capture does not contain) are scored PROBE-ONLY and
    flagged, never silently mixed with activation-conditioned scores.
  - the reported BPW is computed through the IR cost model, so metadata counts.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform  # noqa: E402
from gravity_ir import quant_tensor, SOURCE_PARAM_COUNT              # noqa: E402

# q_inject measured at real captured operating points (tools/gravity_error_chain.py
# re-measurement); linear interpolation between sampled depths.
Q_INJECT = {0: 1.597e-04, 7: 1.910e-04, 15: 2.715e-04, 23: 4.281e-04, 31: 4.634e-04,
            39: 5.279e-04, 47: 6.534e-04, 55: 9.065e-04, 63: 2.577e-03}
_QL = sorted(Q_INJECT)


def q_inject(layer):
    if layer <= _QL[0]:
        return Q_INJECT[_QL[0]]
    if layer >= _QL[-1]:
        return Q_INJECT[_QL[-1]]
    lo = max(l for l in _QL if l <= layer)
    hi = min(l for l in _QL if l >= layer)
    if lo == hi:
        return Q_INJECT[lo]
    t = (layer - lo) / (hi - lo)
    return Q_INJECT[lo] * (1 - t) + Q_INJECT[hi] * t


def layer_classes(layer):
    """Discover this layer's 2-D GEMV tensors from the index.

    The architecture is HYBRID at full_attention_interval 4: some layers carry
    linear_attn (DeltaNet: in_proj_qkv/a/b/z, out_proj), others self_attn (GQA:
    q/k/v/o_proj). Assuming one fixed class list crashes on the other kind, and
    silently skipping would drop whole tensor families out of the allocation.
    """
    import json as _j, re as _re
    idx = _j.load(open(os.path.join(BF16_ROOT, "model.safetensors.index.json")))
    pre = f"language_model.model.layers.{layer}."
    out = []
    for k in idx["weight_map"]:
        if not k.startswith(pre) or not k.endswith(".weight"):
            continue
        cls = k[len(pre):-len(".weight")]
        if cls.endswith("norm") or "layernorm" in cls or cls.endswith("conv1d"):
            continue
        out.append(cls)
    return sorted(out)


BF16_ROOT = "workspace/campaign/records/runs/qwen38-27b/bf16"


def damage_curve(layer, cls, bits_list, group=128, seed=0):
    """Damage vs bits for one tensor. damage = 1 - gate score, floored at 0."""
    name = f"language_model.model.layers.{layer}.{cls}.weight"
    W = load_tensor(name)
    d_in = W.shape[1]
    probe_only = d_in != 5120
    X = None if probe_only else load_X(layer)
    if X is None:                       # probe-only: isotropic directions stand in
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((64, d_in)).astype(np.float32)
    rows = {}
    for b in bits_list:
        a = axes(W, c_uniform(W, b, group), X, seed=seed)
        score = min(a["observed"], a["probed"], a["worst_unit"])
        rows[b] = {"score": float(score), "damage": float(max(0.0, 1.0 - score)), **a}
    return {"tensor": name, "elements": int(W.size), "d_in": int(d_in),
            "probe_only": probe_only, "layer": layer, "cls": cls, "curve": rows}


def bytes_at(elements, bits, group=128):
    return quant_tensor(elements, bits, group, "x").stored_bytes


def inventory(root=None):
    """Every GEMV tensor in the model with its element count, from headers only."""
    import glob, struct
    root = root or BF16_ROOT
    inv = []
    for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__" or not name.startswith("language_model."):
                continue
            if len(meta["shape"]) != 2:
                continue
            e = meta["shape"][0] * meta["shape"][1]
            m = name.split(".")
            layer = int(m[3]) if len(m) > 4 and m[2] == "layers" else None
            cls = ".".join(m[4:-1]) if layer is not None else ".".join(m[2:-1])
            inv.append({"name": name, "elements": e, "layer": layer, "cls": cls})
    return inv


def descend(curves, targets, bits_list, inv=None):
    """Remove the cheapest bit model-wide until each target BPW is met.

    Damage for an unsampled (layer, class) comes from the nearest sampled layer of
    the SAME class. Endpoint tables (embed, lm_head) have no sampled curve and no
    captured input; they are held at the richest width and reported separately
    rather than being given a made-up damage number.
    """
    inv = inv or inventory()
    by_cls = {}
    for c in curves:
        by_cls.setdefault(c["cls"], {})[c["layer"]] = c

    items, held = [], []
    for t in inv:
        cand = by_cls.get(t["cls"])
        if cand is None or t["layer"] is None:
            held.append(t)
            continue
        near = min(cand, key=lambda s: abs(s - t["layer"]))
        items.append({**t, "curve": cand[near]["curve"], "src_layer": near})

    top = max(bits_list)
    state = {i["name"]: top for i in items}
    held_bytes = sum(bytes_at(t["elements"], top) for t in held)

    def total():
        return held_bytes + sum(bytes_at(i["elements"], state[i["name"]]) for i in items)

    out = {}
    for tgt in sorted(targets, reverse=True):
        while 8 * total() / SOURCE_PARAM_COUNT > tgt:
            best = None
            for i in items:
                b = state[i["name"]]
                k = bits_list.index(b)
                if k == 0:
                    continue
                nb = bits_list[k - 1]
                dd = (i["curve"][nb]["damage"] - i["curve"][b]["damage"]) * q_inject(i["layer"])
                db = bytes_at(i["elements"], b) - bytes_at(i["elements"], nb)
                c = dd / max(db, 1)
                if best is None or c < best[0]:
                    best = (c, i["name"], nb)
            if best is None:
                break
            state[best[1]] = best[2]
        dist = {}
        for i in items:
            dist.setdefault(state[i["name"]], 0)
            dist[state[i["name"]]] += i["elements"]
        wd = sum(i["curve"][state[i["name"]]]["damage"] * q_inject(i["layer"]) for i in items)
        out[tgt] = {"bpw": 8 * total() / SOURCE_PARAM_COUNT,
                    "elems_by_bits": dict(sorted(dist.items())),
                    "weighted_damage": wd,
                    "held_elems": sum(t["elements"] for t in held)}
    return out


# ---------------------------------------------------------------- gauntlet
#
# odyssey_patient_runner's --gravity <spec> builds ONE mix (its own docstring:
# "not a sweep"). This is the search that replaces "one sample": propose an
# MLP-density mix, measure complete EBPW the same way tools/future/
# complete_ebpw.py bills the sealed incumbent (mix_report -> cost), localize
# a miss with the SAME organ-sensitivity ranking odyssey_ctl already uses for
# capability failures, and stop on exactly one of three distinct outcomes.
#
# TARGET_HIT requires adequacy to be POSITIVELY verified (healthy is True),
# never just "not proven bad" -- an UNKNOWN adequacy (no real tensors/capture
# on disk in this lane) must never be silently read as a pass.

TARGET_HIT = "TARGET_HIT"
PROVEN_UNABLE = "PROVEN_UNABLE"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
GAUNTLET_OUTCOMES = (TARGET_HIT, PROVEN_UNABLE, BUDGET_EXHAUSTED)

# Default MLP-bpw ladder: identical values to complete_ebpw.bar_reachability's
# own mlp_density_sensitivity sweep, so the two are directly cross-checkable.
# Stops at 0.5 -- this codec's scale+bias overhead is a fixed 0.5 bpw at its
# current group size (affine_group=64 in MIX_REPORT), so 0.5 is the lowest
# total this rebilling can express without also searching group size.
MLP_BPW_LADDER = (2.5, 2.25, 2.0, 1.5, 1.0, 0.5)


def non_searchable_floor_bpw(mix=None):
    """complete EBPW if the searched MLP term were free (zero bytes).

    Same bucket tools/future/complete_ebpw.py:bar_reachability uses (q4
    attention+DeltaNet; f32 is <0.003 bpw and stays folded into the searched
    side). This is arithmetic over MEASURED bytes in the real MIX_REPORT, not
    an estimate: no value the searched term can take will beat it.
    """
    from tools.future.complete_ebpw import mix_report
    mix = mix or mix_report()
    return 8.0 * int(mix["q4_bytes"]) / int(mix["parent_params"])


def mlp_mix_candidate(total_mlp_bpw, mix=None):
    """A sealed-incumbent candidate with ONLY the MLP body rebilled at
    `total_mlp_bpw` (codes+scale+bias combined). Every other byte -- q4
    attention/DeltaNet, f32, headers -- is the real incumbent value from
    MIX_REPORT; only the searched term moves.
    """
    from tools.future import complete_ebpw as ce
    mix = mix or ce.mix_report()
    billing = mix["affine_bpw_billing"]
    aux_bpw = float(billing["scale_bpw"]) + float(billing["bias_bpw"])
    codes_bpw = float(total_mlp_bpw) - aux_bpw
    if codes_bpw < 0:
        raise ce.CompleteEbpwRefused(
            f"total_mlp_bpw {total_mlp_bpw} does not leave room for the "
            f"{aux_bpw} bpw of scale+bias overhead this codec already carries "
            "(fixed by its group size; shrinking below this needs a bigger "
            "group, which this rebilling does not search)"
        )
    cand = ce.incumbent_candidate()
    for region in cand["regions"]:
        if region["name"] == "mlp_codes":
            region["bitwidth"] = codes_bpw
    cand["stated_total_bytes"] = sum(
        int(p["bytes"]) for p in ce._parts_of(cand)
    )
    cand["id"] = f"gravity_mlp_{total_mlp_bpw}bpw"
    return cand


def mlp_ladder_propose(attempt, history, localization, ladder=MLP_BPW_LADDER):
    """Default proposer: descend the MLP-bpw ladder in order.

    Deterministic, not blind sampling -- each rung is strictly denser than
    the last. `localization` (the prior localize_gravity_failure() result) is
    accepted so a multi-lever proposer can steer on it; this one has a single
    lever (MLP density) and always takes the next rung down.
    """
    if attempt > len(ladder):
        return None
    return ladder[attempt - 1]


def _adequacy_unknown(candidate):
    return {
        "healthy": None,
        "magnitude_aware": None,
        "evidence": (
            "UNKNOWN (no adequacy_fn supplied; a missing check is not a pass -- "
            "see tools/gravity_doctor_gate.py:axes for the magnitude-aware check "
            "this should be wired to once real tensors/activation-capture exist "
            "for this candidate)"
        ),
    }


def gauntlet(
    *,
    target=1.00,
    max_attempts=8,
    propose_fn=mlp_ladder_propose,
    measure_fn=None,
    adequacy_fn=None,
    localize_fn=None,
    per_organ_sensitivity=None,
    threshold=None,
):
    """Target-seeking search over complete EBPW. Does not stop at one sample.

    Exactly one of TARGET_HIT / PROVEN_UNABLE / BUDGET_EXHAUSTED terminates
    the loop, recorded distinctly in the return value's "outcome". Only
    TARGET_HIT is a pass -- BUDGET_EXHAUSTED is returned with is_pass=False
    and must never be read as success by a caller.
    """
    from tools.future import complete_ebpw as ce
    from tools.odyssey_ctl import (
        localize_gravity_failure as _default_localize,
        gravity_pass_threshold,
    )

    localize_fn = localize_fn or _default_localize
    adequacy_fn = adequacy_fn or _adequacy_unknown
    thresh = gravity_pass_threshold() if threshold is None else threshold
    mix = ce.mix_report()

    floor = non_searchable_floor_bpw(mix)
    if floor > target:
        return {
            "outcome": PROVEN_UNABLE,
            "is_pass": False,
            "attempts": 0,
            "history": [],
            "bound": {
                "floor_ebpw": floor,
                "target": target,
                "reason": (
                    "non-searched bytes (q4 attention+DeltaNet) alone cost "
                    f"{floor:.6f} ebpw, already above the {target} target "
                    "with the entire searched MLP term at zero bytes; no "
                    "value of the searched term can close this gap"
                ),
                "evidence": "MEASURED (mix_report q4_bytes / parent_params)",
            },
        }

    def _default_measure(candidate):
        return float(ce.cost(candidate)["complete_ebpw"])

    measure_fn = measure_fn or _default_measure

    history = []
    localization = None
    for attempt in range(1, max_attempts + 1):
        proposal = propose_fn(attempt, history, localization)
        if proposal is None:
            return {
                "outcome": BUDGET_EXHAUSTED,
                "is_pass": False,
                "attempts": attempt - 1,
                "history": history,
                "exhausted": "proposer_ladder",
            }
        candidate = mlp_mix_candidate(proposal, mix=mix)
        ebpw = measure_fn(candidate)
        adequacy = adequacy_fn(candidate)
        under_target = ebpw <= target
        hit = under_target and adequacy.get("healthy") is True
        record = {
            "attempt": attempt,
            "proposal": proposal,
            "ebpw": ebpw,
            "under_target": under_target,
            "adequacy": adequacy,
        }
        if hit:
            history.append(record)
            return {
                "outcome": TARGET_HIT,
                "is_pass": True,
                "attempts": attempt,
                "history": history,
                "candidate": candidate,
                "ebpw": ebpw,
                "adequacy": adequacy,
            }
        localization = localize_fn(
            record.get("delta_hits"), per_organ_sensitivity, thresh
        )
        record["localization"] = localization
        history.append(record)

    return {
        "outcome": BUDGET_EXHAUSTED,
        "is_pass": False,
        "attempts": max_attempts,
        "history": history,
        "exhausted": "max_attempts",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--bits", default="2,3,4,6")
    ap.add_argument("--targets", default="4.0,3.0,2.0,1.5,1.0")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    bits = sorted(int(x) for x in a.bits.split(","))
    targets = [float(x) for x in a.targets.split(",")]

    curves = []
    print(f"{'tensor':<44} {'probe_only':>10} " + " ".join(f"{b}b_dmg".rjust(9) for b in bits))
    for l in layers:
        for cls in layer_classes(l):
            try:
                c = damage_curve(l, cls, bits)
            except Exception as ex:
                print(f"{cls+'@L'+str(l):<44} SKIP {ex}")
                continue
            if c["elements"] < 1_000_000:      # 1-D / tiny tensors are not GEMVs
                continue
            curves.append(c)
            dm = " ".join(f"{c['curve'][b]['damage']:>9.4f}" for b in bits)
            print(f"{cls+'@L'+str(l):<44} {str(c['probe_only']):>10} {dm}")

    print(f"\nq_inject weighting: L0 {q_inject(0):.3e}  L31 {q_inject(31):.3e}  L63 {q_inject(63):.3e}"
          f"  ({q_inject(63)/q_inject(0):.1f}x)")

    inv = inventory()
    gemv = sum(t["elements"] for t in inv)
    print(f"inventory: {len(inv)} GEMV tensors, {gemv} elements "
          f"({100*gemv/SOURCE_PARAM_COUNT:.1f}% of N)")
    res = descend(curves, targets, bits)
    print(f"\n{'target':>7} {'achieved':>9} {'wdamage':>10}  elements by bit width")
    for t in sorted(res, reverse=True):
        r = res[t]
        dist = "  ".join(f"{b}b:{100*e/gemv:.1f}%" for b, e in r["elems_by_bits"].items())
        print(f"{t:>7.2f} {r['bpw']:>9.4f} {r['weighted_damage']:>10.5f}  {dist}")
    print(f"held at {max(bits)}b (no curve, no captured input): "
          f"{res[max(res)]['held_elems']} elems")
    if a.json:
        json.dump({"curves": curves, "alloc": {str(k): v for k, v in (res or {}).items()}}, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
