#!/usr/bin/env python3
"""G125: the two harnesses never disagreed about the GPU. They disagreed about the host.

PROTECTED_BITCAST_ABSOLUTE named its own blocker and refused to promote:

    "two harnesses report controls 0.545 ms apart, and picking one silently
     would be choosing the flattering number."

That refusal was right and the number was real. What nobody did was DECOMPOSE it.
Both harnesses report a wall time and a GPU time for the same widen_f4 arm of the
same resident, so the difference splits exactly two ways and the split is
arithmetic on receipts already sealed - no new lease, no new run.

    canonical (resident_reprofile.py)   gpu 26.5943   wall 27.2896   host 0.6953
    lease     (organ profiler)          gpu 26.5410   wall 26.7447   host 0.2037
    ----------------------------------------------------------------------------
    GPU  differ by 0.0533 ms      0.20%
    HOST differ by 0.4916 ms      90.2% OF THE ENTIRE WALL DISAGREEMENT

THE TWO INSTRUMENTS AGREE ABOUT THE RESIDENT TO 0.20%. Everything that looked
like an instrument conflict is host gap - the CPU-side cost of getting work to a
GPU that both harnesses time identically - and host gap is a property of what
else the machine was doing, not of the resident.

That is why the canonical receipt carries `timing_label: DIRTY_ENGINEERING` with
the reason "other system lanes were live", while the lease SIGSTOPped the
ModelLake supervisor and both workers. The lease has a quieter host. It is
supposed to.

WHAT THIS UNBLOCKS, AND WHAT IT DOES NOT.

UNBLOCKED: the GPU absolute. Two independent harnesses, different code paths,
different sessions, agree to 0.0533 ms. `decode_gpu_ms_per_token` is promotable
as a resident property.

NOT UNBLOCKED: a single wall TPS. Wall = GPU + host, and host moved 3.4x between
these two windows. A wall number is a joint claim about the resident AND the
machine's other tenants, so it stays a RANGE with the host gap named beside it:

    36.644 TPS   canonical wall, host 0.6953   noisy machine
    37.391 TPS   lease wall,     host 0.2037   protected window

Both are honest. Neither is the resident alone.

THE RULE THIS INSTALLS. A wall-time disagreement between harnesses is not an
instrument conflict until the GPU components have been differenced. Difference
them first. If the GPU halves agree, the argument is about host contention and
belongs in a range, not in a reconciliation.

    python3 tools/future/harness_reconciliation.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/harness_reconciliation.py"
RECEIPT_NAME = "HARNESS_RECONCILIATION.json"

CANON_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
LEASE_REL = "receipts/future/PROTECTED_BITCAST_ABSOLUTE.json"

# If the GPU halves agree within this, the wall argument is about the host.
GPU_AGREEMENT_REL = 0.01
# The share of the wall gap that host must explain before we call it host-borne.
HOST_SHARE_FLOOR = 0.75


class ReconciliationRefused(RuntimeError):
    """An input is missing, or the two harnesses did not measure the same arm."""


def _load(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise ReconciliationRefused(f"{rel} is not on disk; nothing to reconcile")
    return json.loads(p.read_text())


def harnesses() -> dict[str, dict[str, Any]]:
    canon = _load(CANON_REL)
    lease = _load(LEASE_REL)
    m = lease["measured"]
    arm = m.get("arm")
    if arm != "widen_f4":
        raise ReconciliationRefused(
            f"the lease control arm is {arm!r}, not widen_f4. Differencing two "
            "harnesses that timed DIFFERENT arms would attribute a real lever to "
            "instrument noise."
        )
    out = {
        "canonical": {
            "recorded_by": canon["recorded_by"],
            "gpu_ms": float(canon["decode_gpu_ms_per_token"]),
            "wall_ms": float(canon["decode_wall_ms_per_token"]),
            "timing_label": canon.get("timing_label"),
            "timing_label_reason": canon.get("timing_label_reason"),
        },
        "lease": {
            "recorded_by": lease["recorded_by"],
            "gpu_ms": float(m["control_gpu_ms"]),
            "wall_ms": float(m["control_wall_ms"]),
            "timing_label": "PROTECTED_WINDOW",
            "timing_label_reason": (
                "ModelLake supervisor SIGSTOPped first, then both workers, then "
                "measured, then resumed in reverse order"
            ),
        },
    }
    for v in out.values():
        v["host_ms"] = round(v["wall_ms"] - v["gpu_ms"], 4)
    return out


def decomposition() -> dict[str, Any]:
    h = harnesses()
    c, l = h["canonical"], h["lease"]
    d_gpu = c["gpu_ms"] - l["gpu_ms"]
    d_wall = c["wall_ms"] - l["wall_ms"]
    d_host = c["host_ms"] - l["host_ms"]
    gpu_rel = abs(d_gpu) / l["gpu_ms"]
    host_share = abs(d_host) / abs(d_wall)
    return {
        "wall_disagreement_ms": round(d_wall, 4),
        "gpu_component_ms": round(d_gpu, 4),
        "host_component_ms": round(d_host, 4),
        "gpu_relative_difference": round(gpu_rel, 5),
        "host_share_of_wall_gap": round(host_share, 4),
        "gpu_halves_agree": gpu_rel <= GPU_AGREEMENT_REL,
        "gap_is_host_borne": host_share >= HOST_SHARE_FLOOR,
        "host_gap_ratio": round(c["host_ms"] / l["host_ms"], 3),
        "reading": (
            f"the wall numbers differ by {abs(d_wall):.4f} ms and "
            f"{host_share:.1%} of that is host gap. The GPU halves differ by "
            f"{abs(d_gpu):.4f} ms, which is {gpu_rel:.2%}. These instruments "
            "agree about the resident and disagree about the machine."
        ),
    }


def what_is_promotable() -> dict[str, Any]:
    h = harnesses()
    d = decomposition()
    c, l = h["canonical"], h["lease"]
    if not (d["gpu_halves_agree"] and d["gap_is_host_borne"]):
        return {
            "gpu_absolute": "NOT_PROMOTABLE",
            "why": (
                "the GPU halves do not agree, so the disagreement is a real "
                "instrument conflict and the original refusal stands unchanged"
            ),
        }
    lo, hi = sorted([1000.0 / c["wall_ms"], 1000.0 / l["wall_ms"]])
    return {
        "gpu_absolute": "PROMOTABLE",
        "decode_gpu_ms_per_token": round((c["gpu_ms"] + l["gpu_ms"]) / 2.0, 4),
        "gpu_agreement_ms": round(abs(d["gpu_component_ms"]), 4),
        "corroborated_by": [c["recorded_by"], l["recorded_by"]],
        "wall_tps": "RANGE_ONLY",
        "wall_tps_range": [round(lo, 3), round(hi, 3)],
        "wall_tps_range_reason": (
            "wall = gpu + host, and host moved "
            f"{d['host_gap_ratio']}x between these windows "
            f"({c['host_ms']} ms noisy vs {l['host_ms']} ms protected). A single "
            "wall TPS is a joint claim about the resident AND the machine's "
            "other tenants. The range is the honest form."
        ),
        "the_original_refusal_was_right": (
            "PROTECTED_BITCAST_ABSOLUTE refused to pick one of two wall numbers "
            "and called that choosing the flattering number. That still holds - "
            "this receipt does not pick one. It shows the choice was never "
            "between two measurements of the resident."
        ),
    }


def rule() -> dict[str, Any]:
    return {
        "id": "DIFFERENCE_THE_GPU_HALVES_BEFORE_CALLING_IT_AN_INSTRUMENT_CONFLICT",
        "statement": (
            "a wall-time disagreement between two harnesses is not an instrument "
            "conflict until their GPU components have been differenced. Wall = "
            "GPU + host. If the GPU halves agree, the argument is about host "
            "contention and belongs in a range with the host gap named, not in a "
            "reconciliation that tries to make one harness wrong."
        ),
        "who_it_would_have_helped": (
            "the 0.545 ms blocker sat open as 'the two harnesses should be "
            "reconciled' when the reconciliation was one subtraction over two "
            "sealed receipts. Nothing had to be re-measured."
        ),
        "cost_of_not_applying_it": (
            "an unpromoted GPU absolute that two independent harnesses had "
            "already corroborated to 0.20%"
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G125",
        "question": (
            "PROTECTED_BITCAST_ABSOLUTE refused to promote an absolute because "
            "two harnesses reported controls 0.545 ms apart. Is that an "
            "instrument conflict?"
        ),
        "verdict": "NO_CONFLICT_THE_GPU_HALVES_AGREE_TO_0_20_PERCENT",
        "harnesses": harnesses(),
        "decomposition": decomposition(),
        "what_is_promotable": what_is_promotable(),
        "rule": rule(),
        "evidence_class": "DERIVED_FROM_SEALED_RECEIPTS",
        "no_new_measurement": (
            "this is arithmetic over two receipts already on disk. No GPU lease "
            "was taken and no hardware number is claimed here that was not "
            "measured by one of the two cited harnesses."
        ),
        "inputs": [CANON_REL, LEASE_REL],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "decomposition", "what_is_promotable")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
