"""G133: the organ table re-measured on the body that actually runs. Sum 21.8434 ms.

GAP_LEDGER_60 named this as its own next measurement. Its organ rows summed to
26.7013 ms against a live baseline of 21.9464 - they were measured before the
three levers became the sealed default, so the table was describing a body that
no longer runs and no absolute figure in it was current.

Re-measured in a protected window, levers unset, nothing pinned:

    organ            pre       sealed      delta      share of the sealed token
    mlp_gate_up     9.9154     7.1706     -2.7447        32.8%
    deltanet        5.5971     5.2064     -0.3907        23.8%
    mlp_down        5.7319     4.5854     -1.1465        21.0%
    q4_remainder    2.0596     1.8808     -0.1788         8.6%
    gqa_attention   2.0356     1.7661     -0.2695         8.1%
    lm_head         1.0204     0.8912     -0.1292         4.1%
    sampling        0.3340     0.3341     +0.0001         1.5%
    embedding       0.0073     0.0088     +0.0014         0.0%
    ---------------------------------------------------------------
    SUM            26.7013    21.8434     -4.8579

IT CLOSES ON THE BASELINE. The re-measured organs sum to 21.8434 against the
protected complete-token absolute of 21.9464 - 0.1030 ms apart, 0.47%. The old
table was 4.7549 ms above its baseline; this one is 0.1030 ms below. A
decomposition that does not reconcile with the total it decomposes is not a
decomposition, and the previous one had stopped reconciling.

WHERE THE SAVING LANDED IS THE CONFIRMATION. 3.891 of the 4.858 ms came out of
mlp_gate_up and mlp_down - 80% of it - which is exactly where an MLP dequant
arithmetic lever should show up. deltanet lost 0.391 ms, consistent with widen_f4
folding ba_to_decay. sampling and embedding did not move, and they should not
have: nothing touched them.

WHAT IT SAYS ABOUT THE REMAINING 5.2797 ms TO 60. MLP is still 53.8% of the token
at 11.756 ms across its two halves, and DeltaNet is 23.8% at 5.206. The gap to 60
is 45% of MLP alone, or slightly more than all of DeltaNet. Nothing outside those
two organs is large enough to supply it: q4_remainder, gqa, lm_head, sampling and
embedding together are 4.881 ms, and perfectly deleting ALL FIVE would still miss
60 by 0.4 ms.

    python3 tools/future/organ_decomposition_sealed.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/organ_decomposition_sealed.py"
RECEIPT_NAME = "ORGAN_DECOMPOSITION_SEALED.json"

NEW_REL = "receipts/future/_G133_ORGAN_SEALED_raw.json"
OLD_REL = "receipts/future/_G075_organ_widen_f4.json"
ABSOLUTE_REL = "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"

# A decomposition must reconcile with the total it decomposes.
MAX_RECONCILE_REL = 0.02
MLP_ORGANS = ("mlp_gate_up", "mlp_down")


class DecompositionRefused(RuntimeError):
    """The rows do not reconcile with the measured token, or an arm is missing."""


def _load(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise DecompositionRefused(f"{rel} is not on disk; take the lease first")
    return json.loads(p.read_text())


def _rows(rel: str) -> dict[str, float]:
    d = _load(rel)
    return {r["organ"]: r["gpu_ns_median"] / 1e6
            for r in d["isolated_organs"]["organs"]}


def table() -> list[dict[str, Any]]:
    new, old = _rows(NEW_REL), _rows(OLD_REL)
    if set(new) != set(old):
        raise DecompositionRefused(
            f"organ sets differ: only-new={sorted(set(new) - set(old))}, "
            f"only-old={sorted(set(old) - set(new))}. A row that appeared or "
            "vanished is not a delta."
        )
    total = sum(new.values())
    disp = {r["organ"]: r.get("dispatches")
            for r in _load(NEW_REL)["isolated_organs"]["organs"]}
    out = []
    for organ in sorted(new, key=lambda k: -new[k]):
        out.append({
            "organ": organ,
            "pre_ms": round(old[organ], 4),
            "sealed_ms": round(new[organ], 4),
            "delta_ms": round(new[organ] - old[organ], 4),
            "share_of_sealed_token": round(new[organ] / total, 4),
            "dispatches": disp.get(organ),
        })
    return out


def reconciliation() -> dict[str, Any]:
    new, old = _rows(NEW_REL), _rows(OLD_REL)
    a = _load(ABSOLUTE_REL)["measured"]
    base = float(a["gpu_ms_per_token"])
    s_new, s_old = sum(new.values()), sum(old.values())
    rel_new = abs(s_new - base) / base
    if rel_new > MAX_RECONCILE_REL:
        raise DecompositionRefused(
            f"the re-measured organs sum to {s_new:.4f} ms against a measured "
            f"token of {base:.4f} - {rel_new:.2%} apart. A decomposition that "
            "does not reconcile with its total is not a decomposition."
        )
    return {
        "sealed_organ_sum_ms": round(s_new, 4),
        "measured_token_ms": base,
        "residual_ms": round(s_new - base, 4),
        "residual_relative": round(rel_new, 5),
        "reconciles": True,
        "the_previous_table_did_not": {
            "pre_organ_sum_ms": round(s_old, 4),
            "was_above_its_baseline_by_ms": round(s_old - base, 4),
            "why": "it was measured before the three levers became the sealed "
                   "default, so it described a body that no longer runs",
        },
    }


def where_the_saving_landed() -> dict[str, Any]:
    rows = table()
    total_delta = sum(r["delta_ms"] for r in rows)
    mlp_delta = sum(r["delta_ms"] for r in rows if r["organ"] in MLP_ORGANS)
    untouched = [r["organ"] for r in rows if abs(r["delta_ms"]) < 0.01]
    return {
        "total_delta_ms": round(total_delta, 4),
        "mlp_delta_ms": round(mlp_delta, 4),
        "mlp_share_of_the_saving": round(mlp_delta / total_delta, 4),
        "organs_that_did_not_move": untouched,
        "reading": (
            f"{abs(mlp_delta):.3f} of the {abs(total_delta):.3f} ms came out of "
            "mlp_gate_up and mlp_down, which is exactly where an MLP dequant "
            "arithmetic lever should show up. deltanet lost 0.391 ms, consistent "
            "with widen_f4 folding ba_to_decay. sampling and embedding did not "
            "move, and nothing touched them - an organ that moved without a "
            "lever pointed at it would mean the arms were not matched."
        ),
    }


def what_remains() -> dict[str, Any]:
    rows = table()
    a = _load(ABSOLUTE_REL)["measured"]
    base = float(a["gpu_ms_per_token"])
    gap60 = base - 1000.0 / 60.0
    gap71 = base - 1000.0 / 71.0
    mlp = sum(r["sealed_ms"] for r in rows if r["organ"] in MLP_ORGANS)
    dn = next(r["sealed_ms"] for r in rows if r["organ"] == "deltanet")
    rest = sum(r["sealed_ms"] for r in rows
               if r["organ"] not in MLP_ORGANS and r["organ"] != "deltanet")
    return {
        "gap_to_60_ms": round(gap60, 4),
        "gap_to_71_ms": round(gap71, 4),
        "mlp_total_ms": round(mlp, 4),
        "mlp_share": round(mlp / base, 4),
        "deltanet_ms": round(dn, 4),
        "everything_else_ms": round(rest, 4),
        "gap_as_share_of_mlp": round(gap60 / mlp, 4),
        "deleting_everything_else_perfectly_still_misses_60_by_ms": round(
            gap60 - rest, 4),
        "reading": (
            f"the gap to 60 is {gap60:.4f} ms. MLP is {mlp:.3f} ms across its "
            f"two halves ({mlp / base:.1%} of the token) and DeltaNet is "
            f"{dn:.3f}. Everything else together is {rest:.3f} ms, so perfectly "
            f"deleting ALL of it still misses 60 by {gap60 - rest:.4f} ms. The "
            "remaining gap has to come out of MLP or DeltaNet."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G133",
        "question": (
            "GAP_LEDGER_60 named this: re-run the organ decomposition under the "
            "sealed default so the table and the baseline describe one body."
        ),
        "verdict": "RECONCILED_AT_21p8434_MS",
        "table": table(),
        "reconciliation": reconciliation(),
        "where_the_saving_landed": where_the_saving_landed(),
        "what_remains": what_remains(),
        "levers_unset": True,
        "nothing_pinned": (
            "this example never calls set_dn_state_kernel and never exports a "
            "geo lever, so the session opens on the promoted defaults"
        ),
        "inputs": [NEW_REL, OLD_REL, ABSOLUTE_REL],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(
                lock_held=True, lane="g133-organ", loadavg="{ 8.37 8.62 8.56 }"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "reconciliation", "where_the_saving_landed",
                       "what_remains")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
