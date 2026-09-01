"""G131: the promoted default under a protected lease. 21.9464 ms GPU, 45.566 TPS.

G126 flipped three defaults and proved they reach the dispatcher, but its
verification ran alongside the live sovereign loop and read 35.8-37.2 ms. Only
identity was claimed. This is the absolute, taken the same way
PROTECTED_BITCAST_ABSOLUTE took its own: ModelLake SUPERVISOR stopped FIRST, then
the workers, measure, resume in reverse.

    9 reps, levers unset, state kernel never pinned

    GPU  ms/token   21.8741 .. 21.9727      median of medians  21.9464
    GPU  TPS                                                   45.566
    WALL ms/token   22.8538 .. 22.9565      median             22.9024
    WALL TPS                                                   43.663

    580 dispatches   0 fallbacks   dense_w_materialized 0
    dn_state at open widen_f4, read from env and never pinned

IT CORROBORATES THE ARM THAT LICENSED THE PROMOTION. PROTECTED_BITCAST_ABSOLUTE
measured its bitcast arm at 22.0100 ms GPU. This run, on a different day, through
a different instrument, with the levers UNSET rather than exported, reads 21.9464
- a difference of 0.0636 ms, or 0.29%. Two protected windows and two harnesses
agree about the resident to well under half a percent.

THE HOST GAP IS THE PART THAT MOVES, exactly as G125 said it would. Wall minus
GPU is 0.956 ms here against 0.204 ms in the earlier lease and 0.695 ms in the
canonical budget. GPU is a resident property; wall is a joint claim about the
resident and whatever else the machine is doing, so the wall figure is reported
and NOT promoted as the resident's number.

    the promoted GPU absolute      21.9464 ms   45.566 TPS
    still short of 60 by            5.2797 ms
    still short of 71 by            7.8614 ms

ONE HONEST BLEMISH. A download worker exited after the lease resumed - not during
it, which is what stopping the supervisor first is for. The supervisor was alive
and rescheduled to another specimen; the interrupted specimen's partial is intact
at 5.9 GB with 16 .incomplete files, so it resumes rather than restarts. Nothing
was killed and nothing was lost, and the measurement window closed before any of
it happened.

    python3 tools/future/sealed_default_absolute.py --build
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/sealed_default_absolute.py"
RECEIPT_NAME = "SEALED_DEFAULT_ABSOLUTE.json"

RAW_REL = "receipts/future/_G131_LEASE_SEALED_raw.json"
PRIOR_LEASE_REL = "receipts/future/PROTECTED_BITCAST_ABSOLUTE.json"
VERIFIED_REL = "receipts/future/SEALED_DEFAULT_VERIFIED.json"

MIN_REPS = 9
MAX_SPREAD = 1.02          # 9 reps of the same arm must not span more than 2%
CORROBORATION_REL = 0.01   # two protected windows must agree to 1%


class AbsoluteRefused(RuntimeError):
    """The window was not protected, or the run does not support an absolute."""


def _load(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise AbsoluteRefused(f"{rel} is not on disk; take the lease first")
    return json.loads(p.read_text())


def raw() -> dict[str, Any]:
    d = _load(RAW_REL)
    if not d.get("the_state_kernel_was_never_pinned") or not d.get("levers_unset"):
        raise AbsoluteRefused(
            "this raw pinned an arm or inherited a lever, so it measures "
            "something other than the sealed default"
        )
    if len(d["complete_token_gpu_ns_medians"]) < MIN_REPS:
        raise AbsoluteRefused(
            f"{len(d['complete_token_gpu_ns_medians'])} reps < {MIN_REPS}"
        )
    return d


def measured() -> dict[str, Any]:
    d = raw()
    gpu = sorted(x / 1e6 for x in d["complete_token_gpu_ns_medians"])
    spread = gpu[-1] / gpu[0]
    if spread > MAX_SPREAD:
        raise AbsoluteRefused(
            f"rep spread {spread:.4f} > {MAX_SPREAD}; not a steady window"
        )
    wall = sorted(r["decode_wall_ns"] / r["decode_steps"] / 1e6 for r in d["runs"])
    gpu_med = d["complete_token_gpu_ns_median_of_medians"] / 1e6
    wall_med = statistics.median(wall)
    last = d["last"]
    return {
        "n_reps": len(gpu),
        "gpu_ms_per_token": round(gpu_med, 4),
        "gpu_ms_min": round(gpu[0], 4),
        "gpu_ms_max": round(gpu[-1], 4),
        "gpu_rep_spread": round(spread, 4),
        "gpu_tps": round(1000.0 / gpu_med, 3),
        "wall_ms_per_token": round(wall_med, 4),
        "wall_tps": round(1000.0 / wall_med, 3),
        "host_gap_ms": round(wall_med - gpu_med, 4),
        "dispatches": last["complete_token_dispatches_last"],
        "fallbacks": last["fallbacks"],
        "dense_w_materialized": last["dense_w_materialized"],
        "dn_state_kernel_at_open": d["dn_state_kernel_at_open"],
        "levers_unset": True,
    }


def corroboration() -> dict[str, Any]:
    m = measured()
    prior = _load(PRIOR_LEASE_REL)["measured"]
    prior_gpu = float(prior["bitcast_gpu_ms"])
    delta = abs(m["gpu_ms_per_token"] - prior_gpu)
    rel = delta / prior_gpu
    return {
        "prior_protected_gpu_ms": prior_gpu,
        "this_protected_gpu_ms": m["gpu_ms_per_token"],
        "difference_ms": round(delta, 4),
        "relative_difference": round(rel, 5),
        "agrees": rel <= CORROBORATION_REL,
        "why_this_matters": (
            "the prior lease measured the levers as EXPORTED ENV VARS on the "
            "bitcast arm; this one measures them UNSET as the sealed default, "
            "through a different instrument on a different day. Agreement to "
            f"{rel:.2%} is evidence that the promotion changed the default and "
            "not the number."
        ),
    }


def host_gap_is_not_the_resident() -> dict[str, Any]:
    m = measured()
    canon = _load("receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json")
    prior = _load(PRIOR_LEASE_REL)["measured"]
    gaps = {
        "canonical_noisy": round(
            float(canon["decode_wall_ms_per_token"])
            - float(canon["decode_gpu_ms_per_token"]), 4),
        "prior_lease": round(
            float(prior["bitcast_wall_ms"]) - float(prior["bitcast_gpu_ms"]), 4),
        "this_lease": m["host_gap_ms"],
    }
    return {
        "host_gap_ms_by_window": gaps,
        "span": round(max(gaps.values()) / min(gaps.values()), 3),
        "reading": (
            "G125 established that a wall-time disagreement between harnesses "
            "is host contention, not an instrument conflict. Three protected or "
            "semi-protected windows put the host gap across a "
            f"{max(gaps.values()) / min(gaps.values()):.2f}x span while GPU time "
            "moved under 0.3%. The GPU figure is promoted; the wall figure is "
            "reported beside its host gap and is not the resident's number."
        ),
    }


def lease() -> dict[str, Any]:
    return {
        "evidence_class": "PROTECTED_ABSOLUTE",
        "method": (
            "SIGSTOP the ModelLake SUPERVISOR first (tools/odyssey/"
            "modellake_watch.py pid 82743), then both hf download workers, "
            "settle 25s, measure under the GPU lane lock, then SIGCONT workers "
            "then supervisor"
        ),
        "states_at_open": "supervisor T, both workers T",
        "states_after_resume": "supervisor S, one worker running, one exited",
        "loadavg_before": "10.28 9.12 8.75",
        "loadavg_at_open": "9.10 8.95 8.71",
        "loadavg_at_close": "8.70 8.87 8.68",
        "nothing_was_killed": (
            "SIGSTOP and SIGCONT only. No signal other than STOP/CONT was sent "
            "to any download process."
        ),
        "one_worker_exited_after_resume": {
            "when": "AFTER the window closed, not during it - which is what "
                    "stopping the supervisor first is for",
            "specimen": "thinkingmachines/Inkling-Small",
            "supervisor_alive": True,
            "supervisor_rescheduled_to": "zai-org/GLM-5.3-Flash",
            "partial_intact": "5.9 GB with 16 .incomplete files, so the "
                              "specimen resumes rather than restarts",
            "work_lost": None,
        },
    }


def build() -> dict[str, Any]:
    m = measured()
    return {
        "obligation": "G131",
        "question": (
            "with the three levers promoted to sealed defaults, what does the "
            "resident actually cost in a protected window?"
        ),
        "verdict": "PROTECTED_ABSOLUTE_FOR_THE_SEALED_DEFAULT",
        "headline_gpu_tps": m["gpu_tps"],
        "measured": m,
        "lease": lease(),
        "corroboration": corroboration(),
        "host_gap_is_not_the_resident": host_gap_is_not_the_resident(),
        "identity_verified_separately": {
            "receipt": VERIFIED_REL,
            "note": "G126 proved this arm dispatches the measured kernels. This "
                    "receipt measures it; it does not re-prove identity.",
        },
        "still_short_of_60_by_ms": round(m["gpu_ms_per_token"] - 1000.0 / 60.0, 4),
        "still_short_of_71_by_ms": round(m["gpu_ms_per_token"] - 1000.0 / 71.0, 4),
        "inputs": [RAW_REL, PRIOR_LEASE_REL, VERIFIED_REL],
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
                lock_held=True, lane="g126-lease", loadavg="{ 9.10 8.95 8.71 }"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "headline_gpu_tps", "measured",
                       "corroboration")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
