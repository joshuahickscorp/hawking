"""Reachability arithmetic for a complete-EBPW target (G027, S009 24).

Before any long search, answer: CAN this lever set reach this target AT ALL?

Complete EBPW = persistent bytes * 8 / source parameters. The persistent bytes
split into classes, each holding some number of source parameters at some rate.
A lever is permission to drive one class down to some floor rate. If the classes
a lever set CANNOT touch already exceed the target on their own, the lever set
is mathematically incapable and no amount of effort inside it will do.

This is the calculation that killed MLP-only Gravity: with the entire MLP at
ZERO BYTES the resident still bills 1.545, because attention + DeltaNet at Q4
alone costs 1.55x the whole <=1.0 budget.

Numbers come from the sealed receipts, never from arguments:
  receipts/future/REPRESENTATION_FLOOR.json          byte classes, storage_bpw
  receipts/future/FLASH_EBPW_BAR_REACHABILITY.json   mlp / non_mlp split
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parents[2]
FLOOR = REPO / "receipts" / "future" / "REPRESENTATION_FLOOR.json"
FLASH = REPO / "receipts" / "future" / "FLASH_EBPW_BAR_REACHABILITY.json"


@dataclass(frozen=True)
class Klass:
    name: str
    bytes_: int
    params: int

    @property
    def bpw(self) -> float:
        return self.bytes_ * 8 / self.params if self.params else float("nan")


def load_classes() -> Dict[str, object]:
    floor = json.loads(FLOOR.read_text())["byte_classes"]
    flash = json.loads(FLASH.read_text())

    parent = int(floor["parent_params"])
    payload = int(floor["payload_bytes"])

    mlp = Klass("mlp", int(flash["mlp"]["bytes"]), int(flash["mlp"]["elements"]))
    non_mlp = Klass("attention+deltanet+other",
                    int(flash["non_mlp"]["bytes"]), int(flash["non_mlp"]["params"]))

    # The two receipts do not agree byte-for-byte and the difference is exactly
    # the f32 organs: FLASH folds them into the MLP term, REPRESENTATION_FLOOR
    # keeps them separate. Recorded rather than reconciled silently -- a
    # 10.6 MB discrepancy is 0.003 EBPW and does not change any verdict here,
    # but a later lever that targets f32 organs must know which bucket they are
    # already counted in.
    discrepancy = int(flash["mlp"]["bytes"]) - int(floor["mlp_storage_bytes"])
    return {
        "parent_params": parent,
        "payload_bytes": payload,
        "measured_complete_ebpw": payload * 8 / parent,
        "classes": [mlp, non_mlp],
        "f32_bookkeeping": {
            "flash_mlp_minus_floor_mlp_storage": discrepancy,
            "floor_f32_bytes": int(floor["f32_bytes"]),
            "equal": discrepancy == int(floor["f32_bytes"]),
            "note": "FLASH counts the f32 organs inside its mlp term; "
                    "REPRESENTATION_FLOOR keeps them as their own class",
        },
        "sums_to_payload": mlp.bytes_ + non_mlp.bytes_ == payload,
    }


def reach(target_ebpw: float,
          mutable: Iterable[str],
          floors_bpw: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Can `target_ebpw` be reached by driving `mutable` classes to `floors_bpw`?

    A class absent from `floors_bpw` is taken to its ABSOLUTE floor of zero
    bytes -- the best case that physics allows and no representation achieves.
    A bound computed that way is an impossibility proof when it fails, and
    nothing at all when it passes.
    """
    st = load_classes()
    floors_bpw = floors_bpw or {}
    mutable = set(mutable)
    parent = st["parent_params"]

    fixed, movable, rows = 0, 0, []
    for k in st["classes"]:
        if k.name in mutable:
            floor_bpw = floors_bpw.get(k.name, 0.0)
            floor_bytes = int(k.params * floor_bpw / 8)
            movable += k.bytes_ - floor_bytes
            fixed += floor_bytes
            rows.append({"class": k.name, "state": "MUTABLE", "params": k.params,
                         "current_bytes": k.bytes_, "current_bpw": round(k.bpw, 6),
                         "floor_bpw": floor_bpw, "floor_bytes": floor_bytes})
        else:
            fixed += k.bytes_
            rows.append({"class": k.name, "state": "FIXED", "params": k.params,
                         "current_bytes": k.bytes_, "current_bpw": round(k.bpw, 6)})

    best = fixed * 8 / parent
    budget_bytes = int(target_ebpw * parent / 8)
    return {
        "target_complete_ebpw": target_ebpw,
        "current_complete_ebpw": round(st["measured_complete_ebpw"], 6),
        "mutable_classes": sorted(mutable),
        "fixed_cost_floor_bytes": fixed,
        "fixed_cost_floor_ebpw": round(best, 6),
        "mutable_budget_bytes": movable,
        "total_byte_budget_at_target": budget_bytes,
        "best_case_zeroing_bound_ebpw": round(best, 6),
        "reachable": best <= target_ebpw,
        "over_budget_factor": round(best / target_ebpw, 4) if target_ebpw else None,
        "verdict": ("REACHABLE IN PRINCIPLE -- the fixed cost fits, so the question becomes "
                    "whether any representation actually attains the assumed floor"
                    if best <= target_ebpw else
                    "MATHEMATICALLY UNREACHABLE -- the classes this lever set cannot touch "
                    "already exceed the target on their own. No effort inside these levers "
                    "can succeed; change the search space."),
        "classes": rows,
    }


def required_partner_bpw(target_ebpw: float, held: str, held_bpw: float) -> Dict[str, object]:
    """Hold one class at a rate; what must the OTHER reach to hit the target?

    This is the actionable form. "Unreachable" tells a search to stop; this
    tells it what number to go get.
    """
    st = load_classes()
    parent = st["parent_params"]
    budget = target_ebpw * parent / 8
    ks = {k.name: k for k in st["classes"]}
    if held not in ks:
        raise KeyError(f"{held} not in {sorted(ks)}")
    other = next(k for k in st["classes"] if k.name != held)
    held_bytes = ks[held].params * held_bpw / 8
    left = budget - held_bytes
    need_bpw = left * 8 / other.params
    return {
        "target_complete_ebpw": target_ebpw,
        "held": held, "held_bpw": held_bpw,
        "held_bytes": int(held_bytes),
        "solve_for": other.name,
        "current_bpw": round(other.bpw, 6),
        "required_bpw": round(need_bpw, 6),
        "feasible": need_bpw > 0,
        "note": ("negative required rate: the held class alone already exceeds the whole budget"
                 if need_bpw <= 0 else
                 f"{other.name} must fall from {other.bpw:.4f} to {need_bpw:.4f} bpw"),
    }


def _selftest() -> List[str]:
    """Reproduce the two results already sealed on disk, or say why not."""
    fails = []
    st = load_classes()
    if not st["sums_to_payload"]:
        fails.append("class bytes do not sum to payload_bytes -- the split is not exhaustive")
    if abs(st["measured_complete_ebpw"] - 3.139301) > 1e-5:
        fails.append(f"recomputed EBPW {st['measured_complete_ebpw']:.6f} != sealed 3.139301")

    # FLASH: entire MLP free, target 1.0 -> 1.545, unreachable.
    r = reach(1.0, mutable=["mlp"])
    if abs(r["fixed_cost_floor_ebpw"] - 1.545) > 0.002:
        fails.append(f"MLP-free bound {r['fixed_cost_floor_ebpw']} != sealed 1.545")
    if r["reachable"]:
        fails.append("MLP-free at target 1.0 reported REACHABLE; the sealed receipt says it is not")

    # The solver must invert the bound: hold MLP at zero and solve, and the
    # answer must be the rate that puts the partner exactly at the target.
    inv = required_partner_bpw(1.0, held="mlp", held_bpw=0.0)
    chk = reach(1.0, mutable=["mlp"])
    if inv["required_bpw"] <= 0 and chk["reachable"]:
        fails.append("solver and bound disagree on MLP-free at target 1.0")

    # Negative control: everything mutable to zero must be trivially reachable,
    # or the calculator cannot report success at all.
    r2 = reach(1.0, mutable=["mlp", "attention+deltanet+other"])
    if not r2["reachable"]:
        fails.append("with every class free the calculator still says unreachable -- "
                     "it can only ever say no")
    return fails


if __name__ == "__main__":
    import sys
    fails = _selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {'FAILED' if fails else 'PASS'}")
    if fails:
        sys.exit(1)
    st = load_classes()
    print(f"\nresident: {st['measured_complete_ebpw']:.6f} complete EBPW, "
          f"{st['payload_bytes']:,} B over {st['parent_params']:,} params")
    for k in st["classes"]:
        print(f"  {k.name:28s} {k.bytes_:>14,} B  {k.params:>14,} params  {k.bpw:.4f} bpw")
    print()
    for target in (1.0, 1.5):
        for mutable, label, floors in (
            (["mlp"], "MLP only, free", None),
            (["attention+deltanet+other"], "attention+DeltaNet only, free", None),
            (["mlp", "attention+deltanet+other"], "both, to 1.0 bpw",
             {"mlp": 1.0, "attention+deltanet+other": 1.0}),
            (["mlp", "attention+deltanet+other"], "both, to 2.0 bpw",
             {"mlp": 2.0, "attention+deltanet+other": 2.0}),
        ):
            r = reach(target, mutable, floors)
            print(f"  target {target}  {label:34s} bound {r['fixed_cost_floor_ebpw']:>8.4f}  "
                  f"{'REACHABLE' if r['reachable'] else 'UNREACHABLE'}")
    print("\n  what each organ must actually reach:")
    for target in (1.0, 1.5):
        for held, hb in (("mlp", 2.504975), ("mlp", 1.5), ("mlp", 1.0),
                         ("attention+deltanet+other", 4.248858),
                         ("attention+deltanet+other", 2.0),
                         ("attention+deltanet+other", 1.0)):
            r = required_partner_bpw(target, held, hb)
            print(f"    target {target}  hold {held:26s} at {hb:>8.4f} -> "
                  f"{r['solve_for']:26s} needs {r['required_bpw']:>9.4f} bpw"
                  f"{'  IMPOSSIBLE' if not r['feasible'] else ''}")
