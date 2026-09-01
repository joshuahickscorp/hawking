"""G126 part two: the sealed default really does dispatch the measured kernels.

LEVER_PROMOTION_GATE licensed the flip and refused the word PROMOTED, because a
default that silently fails to select its kernel reports the old number under a
new label and every downstream receipt inherits it. This module is the check it
demanded, and it needed a new instrument: EVERY existing A/B pins its arms with
`set_dn_state_kernel`, so not one of them could answer "what does the default
do?" - they all overwrite the value under test before generating a token.

`ascension_qwen38_sealed_default` is that instrument. It unsets all nine levers,
opens a session, applies the sealed fusion config MINUS the state-kernel pin, and
reports what the dispatcher actually launched.

    dn_state_kernel at open              widen_f4     (read from env, never pinned)
    launched gated_delta kernel          qwen38_gated_delta_decode_vi_simd_ba_f4
    complete-token dispatches            580
    fallbacks / dense_w_materialized     0 / 0

    bitcast matvec kernels launched
        qwen_affine_q2_group32_matvec_geo_tpr64_tg128_bitcast                4160
        qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_bitcast 4160
        qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_bitcast               4225
        qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128_bitcast   3120
        qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128_bitcast           1040

    non-bitcast matvec still launched    NONE

The only non-bitcast q4 kernel in the histogram is `qwen_uniform_q4_embedding_
lookup`, which is a gather and has no bitcast sibling - the standing law that a
name without a sibling is returned unchanged rather than having "_bitcast"
appended to a kernel that does not exist.

TOKEN IDENTITY: the sealed default's 32 token ids are identical to all three arms
of BOTH lease runs - control and bitcast, widen_f4 / baseline / coalesce_tg32 -
and all seven reps are identical to each other.

WHAT IS NOT CLAIMED HERE. This run's timings. It executed alongside the live HCLI
sovereign loop and read ~36 ms/token, far above the lease's 22.01. Kernel identity
and dispatch count are load-insensitive; milliseconds are not. The protected
absolute remains the lease's 22.3347 ms / 44.773 TPS, and what this receipt adds
is that the arm that measurement timed is now the arm you get with nothing set.

    python3 tools/future/sealed_default_verified.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/sealed_default_verified.py"
RECEIPT_NAME = "SEALED_DEFAULT_VERIFIED.json"

RAW_REL = "receipts/future/_G126_SEALED_DEFAULT_raw.json"
LEASE_BITCAST_REL = "receipts/future/_G095_LEASE_BITCAST_raw.json"
LEASE_CTRL_REL = "receipts/future/_G095_LEASE_CTRL_raw.json"
GATE_REL = "receipts/future/LEVER_PROMOTION_GATE.json"

EXPECTED_STATE_KERNEL = "widen_f4"
EXPECTED_GATED_DELTA = "qwen38_gated_delta_decode_vi_simd_ba_f4"
EXPECTED_DISPATCHES = 580
# A gather, not a matvec. It has no bitcast sibling and must not be counted as
# a missed conversion.
NON_MATVEC_Q4 = ("qwen_uniform_q4_embedding_lookup",)


class VerificationRefused(RuntimeError):
    """The raw is missing, or it was produced by an instrument that pins arms."""


def _load(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise VerificationRefused(f"{rel} is not on disk; run the probe first")
    return json.loads(p.read_text())


def raw() -> dict[str, Any]:
    d = _load(RAW_REL)
    if not d.get("the_state_kernel_was_never_pinned"):
        raise VerificationRefused(
            "this raw came from an instrument that pinned the state kernel. "
            "Such a run cannot verify a DEFAULT - it overwrites the value under "
            "test before generating a token."
        )
    if not d.get("levers_unset"):
        raise VerificationRefused(
            "the raw does not attest that the levers were unset. An inherited "
            "env var makes a promoted default indistinguishable from something "
            "someone left exported."
        )
    return d


def dispatch_identity() -> dict[str, Any]:
    d = raw()
    last = d["last"]
    hist = {r["kernel"]: int(r["count"]) for r in last["kernel_histogram"]}
    bitcast = {k: v for k, v in hist.items() if k.endswith("_bitcast")}
    matvec_plain = {
        k: v for k, v in hist.items()
        if ("affine_q2" in k or "uniform_q4" in k or "q2f" in k)
        and not k.endswith("_bitcast")
        and k not in NON_MATVEC_Q4
    }
    return {
        "dn_state_kernel_at_open": d["dn_state_kernel_at_open"],
        "dn_state_kernel_after_sealed_config": d["dn_state_kernel_after_sealed_config"],
        "launched_gated_delta_kernel": last["launched_gated_delta_kernel"],
        "complete_token_dispatches_last": last["complete_token_dispatches_last"],
        "fallbacks": last["fallbacks"],
        "dense_w_materialized": last["dense_w_materialized"],
        "bitcast_kernels_launched": dict(sorted(bitcast.items())),
        "non_bitcast_matvec_kernels_launched": dict(sorted(matvec_plain.items())),
        "excluded_as_not_a_matvec": list(NON_MATVEC_Q4),
    }


def token_identity() -> dict[str, Any]:
    d = raw()
    mine = d["last"]["new_token_ids"]
    reps = [r["new_token_ids"] for r in d["runs"]]
    arms: dict[str, bool] = {}
    for label, rel in (("lease_bitcast", LEASE_BITCAST_REL),
                       ("lease_control", LEASE_CTRL_REL)):
        dec = _load(rel)["decode"]
        for arm, v in dec.items():
            ids = v.get("new_token_ids")
            if ids and len(ids) == len(mine):
                arms[f"{label}.{arm}"] = ids == mine
    return {
        "n_tokens": len(mine),
        "n_reps": len(reps),
        "all_reps_identical_to_each_other": all(r == reps[0] for r in reps),
        "identical_to": dict(sorted(arms.items())),
        "identical_to_every_compared_arm": bool(arms) and all(arms.values()),
    }


def checks() -> list[dict[str, Any]]:
    di, ti = dispatch_identity(), token_identity()
    return [
        {"id": "OPEN_READ_THE_PROMOTED_STATE_KERNEL_FROM_ENV",
         "holds": di["dn_state_kernel_at_open"] == EXPECTED_STATE_KERNEL,
         "evidence": di["dn_state_kernel_at_open"]},
        {"id": "THE_SEALED_CONFIG_DID_NOT_OVERWRITE_IT",
         "holds": di["dn_state_kernel_after_sealed_config"] == EXPECTED_STATE_KERNEL,
         "evidence": di["dn_state_kernel_after_sealed_config"]},
        {"id": "THE_MEASURED_GATED_DELTA_KERNEL_WAS_ACTUALLY_LAUNCHED",
         "holds": di["launched_gated_delta_kernel"] == EXPECTED_GATED_DELTA,
         "evidence": di["launched_gated_delta_kernel"]},
        {"id": "DISPATCH_COUNT_MATCHES_THE_MEASURED_GRAPH",
         "holds": di["complete_token_dispatches_last"] == EXPECTED_DISPATCHES,
         "evidence": di["complete_token_dispatches_last"]},
        {"id": "NO_MATVEC_STAYED_ON_THE_NON_BITCAST_PATH",
         "holds": not di["non_bitcast_matvec_kernels_launched"],
         "evidence": di["non_bitcast_matvec_kernels_launched"]},
        {"id": "AT_LEAST_ONE_BITCAST_KERNEL_WAS_LAUNCHED",
         "holds": len(di["bitcast_kernels_launched"]) > 0,
         "evidence": sorted(di["bitcast_kernels_launched"])},
        {"id": "NO_FALLBACKS_AND_NO_DENSE_MATERIALIZATION",
         "holds": int(di["fallbacks"]) == 0 and int(di["dense_w_materialized"]) == 0,
         "evidence": f"fallbacks={di['fallbacks']}, "
                     f"dense_w={di['dense_w_materialized']}"},
        {"id": "TOKEN_IDENTICAL_TO_EVERY_LEASE_ARM",
         "holds": ti["identical_to_every_compared_arm"],
         "evidence": ti["identical_to"]},
        {"id": "EVERY_REP_PRODUCED_THE_SAME_TOKENS",
         "holds": ti["all_reps_identical_to_each_other"],
         "evidence": f"{ti['n_reps']} reps, {ti['n_tokens']} tokens"},
    ]


def verdict() -> dict[str, Any]:
    cs = checks()
    failed = [c["id"] for c in cs if not c["holds"]]
    if failed:
        return {"verdict": "NOT_VERIFIED", "failed": failed,
                "consequence": "the flip must be reverted; a default that does "
                               "not reach the dispatcher reports the old number "
                               "under a new label"}
    return {
        "verdict": "PROMOTED",
        "n_checks": len(cs),
        "what_is_now_the_default": [
            "HAWKING_QWEN38_DN_STATE=widen_f4",
            "HAWKING_AFFINE2_GEO=bitcast",
            "HAWKING_Q4_UNPACK=bitcast",
        ],
        "the_protected_absolute_this_arm_carries": {
            "source": "receipts/future/PROTECTED_BITCAST_ABSOLUTE.json",
            "wall_ms": 22.3347,
            "wall_tps": 44.773,
            "gpu_ms": 22.01,
            "note": "measured under a lease with the ModelLake supervisor "
                    "stopped first. This receipt does not re-measure it; it "
                    "shows that arm is now what you get with nothing set.",
        },
    }


def what_this_run_does_not_claim() -> dict[str, Any]:
    d = raw()
    med = [round(x / 1e6, 4) for x in d["complete_token_gpu_ns_medians"]]
    return {
        "this_run_ms_per_token": med,
        "why_they_are_not_promoted": (
            "this verification executed alongside the live HCLI sovereign loop, "
            f"reading {min(med)}-{max(med)} ms against the lease's 22.01. Kernel "
            "identity and dispatch count are load-insensitive; milliseconds are "
            "not. Only the identity claims are made here."
        ),
        "evidence_class": "IDENTITY_ONLY_TIMINGS_ARE_CONTAMINATED",
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G126",
        "part": "two of two - the gate licensed the flip, this verifies it landed",
        "question": (
            "with every lever unset, does the dispatcher actually launch the "
            "kernels the protected lease timed?"
        ),
        "verdict": verdict(),
        "checks": checks(),
        "dispatch_identity": dispatch_identity(),
        "token_identity": token_identity(),
        "what_this_run_does_not_claim": what_this_run_does_not_claim(),
        "why_a_new_instrument_was_needed": (
            "every existing A/B pins its arms with set_dn_state_kernel, so none "
            "of them can verify a DEFAULT - they overwrite the value under test "
            "before generating a token. ascension_qwen38_sealed_default unsets "
            "the levers, opens a session, applies the sealed fusion config MINUS "
            "the state-kernel pin, and reports what was launched."
        ),
        "inputs": [RAW_REL, LEASE_BITCAST_REL, LEASE_CTRL_REL, GATE_REL],
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
                      ("verdict", "dispatch_identity", "token_identity",
                       "what_this_run_does_not_claim")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
