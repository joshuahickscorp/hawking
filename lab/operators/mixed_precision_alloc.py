"""Output-sensitivity mixed-precision allocation (absorbed from mixed_precision.py).

Core methodology kept:
  - sens(t,b) ≈ relRMS(t,b) · actnorm(t)  (output-space damage proxy)
  - greedy marginal water-fill under a target average effective bpw
  - emit per-tensor and coarse per-role configs

Dropped: strand baker subprocess, HF model load for capture_actnorm/ppl,
full-model bake prove path, retired campaign CLI. Live path uses pure torch
grid-quant relRMS (conservative upper bound on trellis recon) and optional
caller-supplied actnorm; when actnorm is absent, per-row weight energy is used
as a stand-in so the water-fill is still exercisable on real organs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

# Effective bpw per nominal bit (RHT+outlier folded) — same table the doctor used.
BPW: dict[int, float] = {
    1: 1.34,
    2: 2.34,
    3: 3.34,
    4: 4.50,
    5: 5.50,
    6: 6.50,
    8: 8.50,
}


def grid_quant_rel_l2(W, bits: int) -> float:
    """Per-output-row uniform round-to-grid rel_L2 = ||W-Q(W)||/||W||.

    Conservative upper bound on baker trellis+RHT error. Torch-only.
    """
    import torch

    if not isinstance(W, torch.Tensor):
        W = torch.as_tensor(W)
    W = W.float()
    if W.ndim != 2 or min(W.shape) < 1:
        raise ValueError(f"expected 2-D weight, got shape {tuple(W.shape)}")
    levels = max(2, 2 ** int(bits))
    half = (levels - 1) / 2.0
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / half
    Q = (W / scale).round().clamp_(-half, half) * scale
    num = (W - Q).pow(2).sum().sqrt()
    den = W.pow(2).sum().sqrt().clamp_min(1e-12)
    return float((num / den).item())


def weight_actnorm_proxy(W) -> float:
    """When calib activations are unavailable: RMS of per-input-channel mean|W|."""
    import torch

    if not isinstance(W, torch.Tensor):
        W = torch.as_tensor(W)
    W = W.float()
    # mean over out_features of |W|, then RMS over in_features
    col = W.abs().mean(dim=0)
    return float((col.pow(2).mean().sqrt()).item())


def sensitivity_rows(
    tensors: Mapping[str, Any],
    bits_set: Sequence[int],
    actnorm: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build damage-ranked rows from named weight tensors.

    tensors: {name: array/tensor [out, in]}
    actnorm: optional {name: float}; missing names fall back to weight_actnorm_proxy.
    """
    import torch

    bits_set = sorted(int(b) for b in bits_set)
    rows: list[dict[str, Any]] = []
    for name, W in tensors.items():
        t = W if isinstance(W, torch.Tensor) else torch.as_tensor(W)
        t = t.float()
        elems = int(t.numel())
        a = float(actnorm[name]) if actnorm and name in actnorm else weight_actnorm_proxy(t)
        relrms = {b: grid_quant_rel_l2(t, b) for b in bits_set}
        sens = {b: relrms[b] * a for b in bits_set}
        rows.append(
            {
                "name": name,
                "elems": elems,
                "shape": list(t.shape),
                "actnorm": a,
                "relrms": relrms,
                "sens": sens,
            }
        )
    return rows


def holdout_margin_rows(
    organs: Sequence[Mapping[str, Any]],
    bits_set: Sequence[int],
    *,
    layer_target: float,
) -> list[dict[str, Any]]:
    """Build allocate() rows from activation-holdout output cosine.

    NOT the weight-space grid_quant relRMS proxy — input distribution decides.
    Sensitivity at bit-width b is remaining deficit to ``layer_target`` (c^48),
    scaled so extra bits buy more on worse organs. Water-fill then lifts the
    worst organs first.
    """
    bits_set = sorted(int(b) for b in bits_set)
    if not bits_set:
        raise ValueError("bits_set must be non-empty")
    hi = bits_set[-1]
    rows: list[dict[str, Any]] = []
    for o in organs:
        name = str(o["name"])
        elems = int(o["elems"])
        if elems <= 0:
            raise ValueError(f"{name}: elems must be positive")
        holdout = float(o["holdout_cosine"])
        deficit = max(float(layer_target) - holdout, 0.0) + 1e-6
        sens = {b: float(deficit) * float(hi + 1 - b) for b in bits_set}
        rows.append(
            {
                "name": name,
                "elems": elems,
                "holdout_cosine": holdout,
                "margin": holdout - float(layer_target),
                "layer_target": float(layer_target),
                "sens": sens,
                "component": o.get("component"),
                "layer": o.get("layer"),
                "band": o.get("band"),
            }
        )
    return rows


def _avg_eff_bpw(
    rows: Sequence[Mapping[str, Any]], alloc: Mapping[str, int], total_elems: int
) -> float:
    return sum(BPW[alloc[str(r["name"])]] * int(r["elems"]) for r in rows) / total_elems


def _waterfill_maxmin(
    rows: Sequence[Mapping[str, Any]],
    bits_set: Sequence[int],
    target_bpw: float,
) -> tuple[dict[str, int], float]:
    """Lift the worst holdout-margin organ first under avg-eff-bpw ≤ target."""
    floor, ceil = bits_set[0], bits_set[-1]
    alloc = {str(r["name"]): floor for r in rows}
    total_elems = sum(int(r["elems"]) for r in rows)
    by = {str(r["name"]): r for r in rows}

    def margin(name: str) -> float:
        r = by[name]
        b = alloc[name]
        if "holdout_cosine" in r:
            holdout = float(r["holdout_cosine"])
            target = float(r.get("layer_target", 0.9857))
            gap = max(target - holdout, 0.0)
            # Predicted close of the gap as bits rise, so the current
            # worst does not hoard the whole budget at a static cosine.
            span = max(ceil - floor, 1)
            return holdout + gap * (b - floor) / span
        return -float(r["sens"][b])

    while True:
        pick: tuple[tuple[float, int, str], str, int] | None = None
        for r in rows:
            name = str(r["name"])
            b = alloc[name]
            if b >= ceil:
                continue
            nb = bits_set[bits_set.index(b) + 1]
            trial = dict(alloc)
            trial[name] = nb
            if _avg_eff_bpw(rows, trial, total_elems) > float(target_bpw) + 1e-9:
                continue
            # Worst holdout first; ties go to the larger organ.
            key = (margin(name), -int(r["elems"]), name)
            if pick is None or key < pick[0]:
                pick = (key, name, nb)
        if pick is None:
            break
        alloc[pick[1]] = pick[2]
    return alloc, _avg_eff_bpw(rows, alloc, total_elems)


def allocate(
    rows: Sequence[Mapping[str, Any]],
    bits_set: Sequence[int],
    target_bpw: float,
    *,
    objective: str = "greedy_marginal",
) -> tuple[dict[str, int], float]:
    """Greedy / max-min water-fill under param-weighted average effective-bpw ≤ target.

    objective:
      greedy_marginal — original ds/dc gain (needs per-bit ``sens``)
      maxmin_margin   — always lift the current worst holdout-cosine organ
    """
    bits_set = sorted(int(b) for b in bits_set)
    if not bits_set:
        raise ValueError("bits_set must be non-empty")
    if not rows:
        return {}, 0.0
    total_elems = sum(int(r["elems"]) for r in rows)
    if total_elems <= 0:
        raise ValueError("rows must have positive element counts")
    if objective == "maxmin_margin":
        return _waterfill_maxmin(rows, bits_set, target_bpw)
    if objective != "greedy_marginal":
        raise ValueError(f"unknown allocate objective {objective!r}")

    floor, ceil = bits_set[0], bits_set[-1]
    alloc = {str(r["name"]): floor for r in rows}
    by = {str(r["name"]): r for r in rows}

    def gain(name: str, b: int) -> tuple[float, int]:
        r = by[name]
        nb = bits_set[bits_set.index(b) + 1]
        ds = float(r["sens"][b]) - float(r["sens"][nb])
        dc = (BPW[nb] - BPW[b]) * int(r["elems"]) / total_elems
        return (ds / dc if dc > 0 else 0.0), nb

    while True:
        best: tuple[float, str, int] | None = None
        for r in rows:
            name = str(r["name"])
            b = alloc[name]
            if b >= ceil:
                continue
            g, nb = gain(name, b)
            trial = dict(alloc)
            trial[name] = nb
            if (
                _avg_eff_bpw(rows, trial, total_elems) <= target_bpw + 1e-9
                and (best is None or g > best[0])
            ):
                best = (g, name, nb)
        if best is None:
            break
        alloc[best[1]] = best[2]
    return alloc, _avg_eff_bpw(rows, alloc, total_elems)


def allocate_from_holdout(
    organs: Sequence[Mapping[str, Any]],
    *,
    bits_set: Sequence[int] = (1, 2, 3, 4),
    target_bpw: float = 1.5,
    layer_target: float,
) -> dict[str, Any]:
    """End-to-end per-organ water-fill from activation-holdout cosine."""
    bits_set = tuple(sorted(int(b) for b in bits_set))
    rows = holdout_margin_rows(organs, bits_set, layer_target=layer_target)
    alloc, avg = allocate(rows, bits_set, target_bpw, objective="maxmin_margin")
    hist: dict[str, int] = {}
    for b in alloc.values():
        hist[str(int(b))] = hist.get(str(int(b)), 0) + 1
    return {
        "schema": "hawking.doctor.mixed_precision_alloc.v1",
        "allocator_invoked": True,
        "allocator": "mixed_precision_alloc.allocate",
        "objective": "maxmin_margin_to_layer_target",
        "sensitivity": "activation_holdout_output_cosine",
        "not_weight_space_relrms": True,
        "rung": "mixed_prec",
        "bits_set": list(bits_set),
        "target_bpw": float(target_bpw),
        "layer_target": float(layer_target),
        "achieved_avg_eff_bpw": float(avg),
        "within_budget": bool(avg <= float(target_bpw) + 1e-9),
        "allocation": alloc,
        "histogram": hist,
        "rung_config": emit_config(alloc, "rung"),
        "bit_width_map": [
            {
                "name": r["name"],
                "elems": r["elems"],
                "holdout_cosine": r["holdout_cosine"],
                "margin_to_layer_target": r["margin"],
                "bits": alloc[r["name"]],
                "eff_bpw": BPW[alloc[r["name"]]],
                "component": r.get("component"),
                "layer": r.get("layer"),
                "band": r.get("band"),
            }
            for r in rows
        ],
        "metric": "holdout_output_cosine_margin_to_c48",
    }


def emit_config(alloc: Mapping[str, int], kind: str = "tensor") -> dict[str, Any]:
    if kind == "rung":
        roles: dict[str, list[int]] = {}
        for name, b in alloc.items():
            role = name.split(".")[-2] if "." in name else name
            # strip trailing .weight if present
            if role == "weight" and "." in name:
                role = name.split(".")[-2]
            # for names like ...gate_proj.weight
            parts = name.split(".")
            for p in reversed(parts):
                if p.endswith("_proj") or p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    role = p
                    break
            roles.setdefault(role, []).append(int(b))
        return {r: max(set(bs), key=bs.count) for r, bs in roles.items()}
    return {str(k): int(v) for k, v in alloc.items()}


def run_on_organs(
    tensors: Mapping[str, Any],
    *,
    bits_set: Sequence[int] = (1, 2, 3, 4),
    target_bpw: float = 1.0,
    actnorm: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """End-to-end L2 mixed_prec rung on a named organ set."""
    rows = sensitivity_rows(tensors, bits_set, actnorm=actnorm)
    alloc, avg = allocate(rows, bits_set, target_bpw)
    return {
        "schema": "hawking.doctor.mixed_precision_alloc.v1",
        "rung": "mixed_prec",
        "layer": 2,
        "bits_set": list(bits_set),
        "target_bpw": float(target_bpw),
        "achieved_avg_eff_bpw": float(avg),
        "within_budget": bool(avg <= target_bpw + 1e-9),
        "allocation": alloc,
        "rung_config": emit_config(alloc, "rung"),
        "rows": [
            {
                "name": r["name"],
                "elems": r["elems"],
                "shape": r["shape"],
                "actnorm": r["actnorm"],
                "relrms": r["relrms"],
                "sens": r["sens"],
                "bits": alloc[r["name"]],
            }
            for r in rows
        ],
        "metric": "relrms_grid * actnorm_or_weight_proxy",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L2 mixed-precision water-fill (live)")
    ap.add_argument("--target-bpw", type=float, default=1.0)
    ap.add_argument("--bits", default="1,2,3,4")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    bits = [int(x) for x in args.bits.split(",") if x.strip()]
    if args.selftest:
        import torch

        torch.manual_seed(0)
        tensors = {
            "layer0.gate_proj.weight": torch.randn(64, 32) * 0.02,
            "layer0.up_proj.weight": torch.randn(64, 32) * 0.05,
            "layer0.down_proj.weight": torch.randn(32, 64) * 0.1,
        }
        # make down more sensitive via larger actnorm
        act = {
            "layer0.gate_proj.weight": 0.1,
            "layer0.up_proj.weight": 0.2,
            "layer0.down_proj.weight": 1.5,
        }
        out = run_on_organs(tensors, bits_set=bits, target_bpw=args.target_bpw, actnorm=act)
        print(json.dumps(out, indent=2))
        assert out["within_budget"], out
        return 0
    ap.error("pass --selftest or use lab.operators.doctor_ladder")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
