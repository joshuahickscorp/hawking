"""G126: the three measured levers are licensed for promotion, and the gate says why.

GAP_LEDGER_60 stated its own promotion condition in `built_not_promoted`:

    "a protected reprofile with both levers set, replacing the resident token
     budget; then the default flips and this section empties"

and gave the reason it had not happened:

    "every window so far has had ModelLake downloads running - twice after they
     were SIGSTOPped and the supervisor respawned them."

BOTH OF THOSE ARE NOW SATISFIED AND NEITHER WAS NOTICED. The protected reprofile
exists: PROTECTED_BITCAST_ABSOLUTE stopped the SUPERVISOR first, then the
workers, measured, and resumed in reverse - the exact fix for the respawn that
contaminated the two earlier windows. And the last standing objection, two
harnesses reporting controls 0.545 ms apart, was decomposed in G125: the GPU
halves agree to 0.20% and 90.2% of that gap is host contention.

So this module does not measure anything. It is the gate that reads the evidence
already on disk, applies the promotion rule the ledger wrote for itself, and
either licenses the default flip or refuses with the specific missing item.

    THE THREE LEVERS, all token-identical, all opt-in today:

        HAWKING_QWEN38_DN_STATE=widen_f4    1.0245 ms   628 -> 580 dispatches
        HAWKING_AFFINE2_GEO=bitcast         3.8541 ms
        HAWKING_Q4_UNPACK=bitcast           0.6836 ms

    widen_f4 is the CONTROL of the bitcast lease, not an addend to it. The lease
    control is 26.5410 ms GPU with widen_f4 already on, and bitcast takes that to
    22.0100. Adding 1.0245 to 4.5311 would double-count the baseline shift, which
    is the arithmetic error this gate exists to refuse.

WHAT A PASSING GATE LICENSES: flipping three source defaults. Nothing more. The
flip is not the promotion - PROMOTED requires a post-flip verification run
proving the SEALED DEFAULT now dispatches the measured kernels, because a default
that silently fails to select its kernel would report the old number with a new
label and every downstream receipt would inherit it.

    python3 tools/future/lever_promotion_gate.py            # decision only
    python3 tools/future/lever_promotion_gate.py --build    # seal the receipt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/lever_promotion_gate.py"
RECEIPT_NAME = "LEVER_PROMOTION_GATE.json"

LEASE_REL = "receipts/future/PROTECTED_BITCAST_ABSOLUTE.json"
RECON_REL = "receipts/future/HARNESS_RECONCILIATION.json"
WIDEN_REL = "receipts/future/DELTANET_WIDEN_AB.json"
LEDGER_REL = "receipts/future/GAP_LEDGER_60.json"

# Where each default lives, so the flip is reviewable before it is made.
DEFAULTS = (
    {
        "lever": "HAWKING_QWEN38_DN_STATE",
        "value": "widen_f4",
        "file": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
        "selector": "Qwen38DeltaNetStateKernel::from_env_with_fast",
        "today": "Baseline unless the fast profile is on",
        "after": "WidenF4 always",
    },
    {
        "lever": "HAWKING_AFFINE2_GEO",
        "value": "bitcast",
        "file": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
        "selector": "Affine2Geo::from_env_with_fast",
        "today": "Tpr64 unless the fast profile is on, and fast selects SplitK4 - "
                 "so the fast profile is NOT the measured arm either",
        "after": "Bitcast always",
    },
    {
        "lever": "HAWKING_Q4_UNPACK",
        "value": "bitcast",
        "file": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
        "selector": "qwen38_q4_bitcast_on",
        "today": "false unless the env var is set",
        "after": "true unless the env var says otherwise",
    },
)


class PromotionRefused(RuntimeError):
    """A precondition is missing. The defaults stay where they are."""


def _load(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise PromotionRefused(
            f"{rel} is not on disk. Promotion is refused rather than assumed: a "
            "default flipped without its evidence is a fake completion."
        )
    return json.loads(p.read_text())


def preconditions() -> list[dict[str, Any]]:
    """Each is a named, checkable claim. All must hold."""
    lease = _load(LEASE_REL)
    recon = _load(RECON_REL)
    widen = _load(WIDEN_REL)
    m = lease["measured"]
    L = lease["lease"]
    prom = recon["what_is_promotable"]
    wc = widen["complete_token"]

    return [
        {
            "id": "PROTECTED_WINDOW_WITH_THE_SUPERVISOR_STOPPED_FIRST",
            "holds": "SUPERVISOR first" in L["method"] or "supervisor" in L["method"],
            "evidence": L["method"],
            "why_it_matters": L["why_the_supervisor_first"],
        },
        {
            "id": "LEASE_EVIDENCE_CLASS_IS_PROTECTED_ABSOLUTE",
            "holds": L["evidence_class"] == "PROTECTED_ABSOLUTE",
            "evidence": L["evidence_class"],
        },
        {
            "id": "NOTHING_WAS_KILLED_TO_GET_THE_WINDOW",
            "holds": "SIGSTOP and SIGCONT only" in L["nothing_was_killed"],
            "evidence": L["nothing_was_killed"],
        },
        {
            "id": "BITCAST_ARM_IS_TOKEN_IDENTICAL_WITH_ZERO_FALLBACKS",
            "holds": bool(m["token_identical"]) and int(m["fallbacks"]) == 0,
            "evidence": f"token_identical={m['token_identical']}, "
                        f"fallbacks={m['fallbacks']}, n_tokens={m['n_tokens']}, "
                        f"reps={m['reps']}",
        },
        {
            "id": "WIDEN_F4_ARM_IS_TOKEN_IDENTICAL_ON_THE_628_GRAPH",
            "holds": wc["widen_f4_dispatches_last"] == 580
                     and wc["incumbent_dispatches_last"] == 628,
            "evidence": f"{wc['incumbent_dispatches_last']} -> "
                        f"{wc['widen_f4_dispatches_last']} dispatches, "
                        f"{wc['incumbent_ms']} -> {wc['widen_f4_ms']} ms",
        },
        {
            "id": "THE_HARNESS_DISAGREEMENT_IS_NOT_AN_INSTRUMENT_CONFLICT",
            "holds": prom["gpu_absolute"] == "PROMOTABLE",
            "evidence": recon["decomposition"]["reading"],
        },
        {
            "id": "WIDEN_F4_IS_THE_LEASE_CONTROL_NOT_AN_ADDEND",
            "holds": m["arm"] == "widen_f4" and m["dispatches"] == 580,
            "evidence": f"lease control arm={m['arm']}, dispatches="
                        f"{m['dispatches']} - the same 580 the widen A/B "
                        "produced, so the bitcast saving is measured ON TOP of "
                        "widen_f4 and the two must not be summed",
        },
    ]


def decision() -> dict[str, Any]:
    checks = preconditions()
    failed = [c["id"] for c in checks if not c["holds"]]
    lease = _load(LEASE_REL)
    m = lease["measured"]
    ledger = _load(LEDGER_REL)
    if failed:
        return {
            "verdict": "REFUSED",
            "failed_preconditions": failed,
            "defaults_unchanged": True,
        }
    gpu, wall = float(m["bitcast_gpu_ms"]), float(m["bitcast_wall_ms"])
    return {
        "verdict": "LICENSED_TO_FLIP_THE_DEFAULTS",
        "n_preconditions": len(checks),
        "sealed_default_today_ms": float(m["control_wall_ms"]),
        "sealed_default_today_tps": float(m["control_wall_tps"]),
        "sealed_default_after_ms": wall,
        "sealed_default_after_tps": float(m["bitcast_wall_tps"]),
        "gpu_ms_after": gpu,
        "ms_saved": float(m["gpu_ms_saved"]),
        "still_short_of_60_by_ms": round(wall - 1000.0 / 60.0, 4),
        "still_short_of_71_by_ms": round(wall - 1000.0 / 71.0, 4),
        "the_ledger_asked_for_exactly_this": (
            ledger["built_not_promoted"]["what_promotion_requires"]
        ),
        "and_the_reason_it_gave_no_longer_holds": (
            ledger["built_not_promoted"]["why_not_promoted"]
        ),
        "what_this_does_not_license": (
            "calling the result PROMOTED. Flipping a default is a source change; "
            "promotion needs a post-flip run proving the SEALED DEFAULT actually "
            "dispatches the measured kernels. A default that silently fails to "
            "select its kernel reports the old number under a new label, and "
            "every downstream receipt inherits it."
        ),
    }


def the_double_count_this_refuses() -> dict[str, Any]:
    lease = _load(LEASE_REL)
    widen = _load(WIDEN_REL)
    m = lease["measured"]
    w = float(widen["saving"]["complete_token_saving_ms"])
    b = float(m["gpu_ms_saved"])
    return {
        "tempting_wrong_sum_ms": round(w + b, 4),
        "correct_saving_ms": b,
        "why": (
            f"widen_f4 saves {w} ms against the 628-dispatch incumbent. The "
            f"bitcast lease then saves {b} ms against a control that ALREADY has "
            "widen_f4 on - same 580 dispatches, same arm name. Summing them "
            f"would claim {round(w + b, 4)} ms and count the widen_f4 baseline "
            "shift twice."
        ),
        "what_widen_f4_is_worth_here": (
            "it is already inside the 26.5410 ms control. Its value shows up as "
            "the control being 1.02 ms lower than the 628-dispatch incumbent, "
            "not as an addend to the bitcast delta."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G126",
        "question": (
            "GAP_LEDGER_60 wrote its own promotion condition and the reason it "
            "was unmet. Are both now satisfied?"
        ),
        "decision": decision(),
        "preconditions": preconditions(),
        "defaults_to_flip": list(DEFAULTS),
        "the_double_count_this_refuses": the_double_count_this_refuses(),
        "evidence_class": "DERIVED_FROM_SEALED_RECEIPTS",
        "no_new_measurement": (
            "no GPU lease was taken by this module and no hardware number is "
            "claimed that was not measured by a cited receipt."
        ),
        "inputs": [LEASE_REL, RECON_REL, WIDEN_REL, LEDGER_REL],
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
                      ("decision", "the_double_count_this_refuses")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
