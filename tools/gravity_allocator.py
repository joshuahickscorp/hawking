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
