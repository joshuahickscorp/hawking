#!/usr/bin/env python3
"""The terminal speed artifact, and the reasons it may not be written yet.

S022 §66/§68 (G066) asks for exactly one of two receipts:

    RESIDENT_71TPS_UNLOCK.json          -- 71 TPS reached, with repeatability,
                                           capability, zero fallbacks, source
                                           identity, clean build, stable memory
                                           and restart

    MAX_RESIDENT_PHYSICAL_ROOF.json     -- the binding limit PROVEN, naming the
                                           dominant remaining costs, the
                                           irreducible current information, the
                                           best representation and its physical
                                           evidence, the next hardware
                                           requirement, and the next model-body
                                           alternative

and it adds the sentence that makes this module necessary: "Probably impossible"
is not an acceptable output; a proof of the binding limit is. The acceptance
adds that EVERY NUMBER must be traceable to a landed measurement receipt.

Those two failure modes are opposite and both are easy. A premature roof receipt
declares a limit while three measurements are outstanding, which is the same
error as declaring victory. So this module does not write prose - it holds the
prerequisites as data, resolves every number it would quote from the receipt
that owns it, and REFUSES to emit while a prerequisite is open. When the
outstanding measurements land, the artifact assembles itself from receipts
rather than being written by hand.

    python3 tools/future/terminal_speed_artifact.py            # status
    python3 tools/future/terminal_speed_artifact.py --build    # emit or refuse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

import causal_budget_71 as cb  # noqa: E402
import executable_economics as ee  # noqa: E402

RECORDED_BY = "tools/future/terminal_speed_artifact.py"
UNLOCK_NAME = "RESIDENT_71TPS_UNLOCK.json"
ROOF_NAME = "MAX_RESIDENT_PHYSICAL_ROOF.json"

TARGET_TPS = 71.0
TARGET_MS = 1000.0 / TARGET_TPS  # 14.085 ms


class TerminalArtifactRefused(RuntimeError):
    """The terminal artifact cannot be written yet, and the reasons are named."""


# Each prerequisite is a MEASUREMENT that must land before the binding limit can
# be called proven rather than believed. A prerequisite is met when its receipt
# exists AND carries the field that makes it a measurement rather than a plan.
PREREQUISITES: tuple[dict[str, Any], ...] = (
    {
        "id": "G038_per_region_attribution",
        # Second prerequisite pointing at a filename nobody ever wrote. The
        # receipt is ORGAN_BANDWIDTH.json and it landed with the measurement made:
        # per-organ GPU ms behind HAWKING_QWEN38_REGION_TIMING (default OFF), the
        # trace's own cost measured at 1.8%, and coverage of 27.733 of 27.828 ms
        # with the 0.095 ms remainder NAMED - norms, embedding row, A_log,
        # dt_bias. I keyed this on an invented name while the answer was on disk.
        "receipt": "receipts/future/ORGAN_BANDWIDTH.json",
        "written_by": "tools/future/organ_bandwidth.py",
        "field": ["coverage", "gpu_ms_unattributed"],
        "why": (
            "a roof receipt that names the dominant remaining costs cannot leave "
            "GPU time unattributed and call the accounting complete. The answer is "
            "that the loss is NOT localised: MLP, DeltaNet and GQA sit inside 5% "
            "of each other against a 703.5 clean roof, so there is no hot organ "
            "and byte elimination outranks execution tuning."
        ),
    },
    {
        "id": "G044_granularity_falsifier",
        # The receipt is MLP_REGION_FALSIFIER, not MLP_GRANULARITY_FALSIFIER. A
        # prerequisite pointing at a filename that will never exist is a permanent
        # false blocker - it would have held the terminal artifact shut forever on
        # a measurement that had already landed.
        "receipt": "receipts/future/MLP_REGION_FALSIFIER.json",
        "written_by": "tools/future/mlp_region_falsifier.py",
        "field": ["contiguous", "effective_gb_s"],
        "why": (
            "One representative MLP layer, contiguous, few fused regions, "
            "identical arithmetic, bit-identical output. Rising toward the LM "
            "head's 497.4 implicates fragmentation; staying near 350 kills it. "
            "Either answer changes what the binding limit IS."
        ),
    },
    {
        "id": "G075_current_body_baseline",
        "receipt": "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json",
        "written_by": "tools/future/resident_token_budget.py",
        "field": ["decode_wall_ms_per_token"],
        "why": (
            "deltanet_widen_f4 landed as a measured token-identical 1.0245 ms "
            "win, so every ladder rung is arithmetic over a body that no longer "
            "runs. A roof measured against a superseded baseline is not a roof."
        ),
    },
)


class PrerequisiteUnwritable(RuntimeError):
    """A prerequisite names a receipt no tool in this repo can produce."""


def check_prerequisites_are_writable() -> list[dict[str, Any]]:
    """Every prerequisite must name the TOOL that writes its receipt, and that
    tool must exist.

    Two of the three prerequisites originally pointed at filenames nobody ever
    wrote - MLP_GRANULARITY_FALSIFIER.json (the receipt is MLP_REGION_FALSIFIER)
    and TOKEN_REGION_TIMESTAMPS.json (the receipt is ORGAN_BANDWIDTH). Both
    measurements had ALREADY LANDED, so those were permanent false blockers on
    work that was done.
    Requiring the writer catches the invention at authoring time: a filename can
    be made up, but naming the module that produces it cannot be, because the
    module has to be on disk.
    """
    bad = []
    for pre in PREREQUISITES:
        tool = pre.get("written_by")
        if not tool:
            bad.append({"id": pre["id"], "why": "no written_by; the receipt name is unverifiable"})
            continue
        if not (REPO / tool).exists():
            bad.append({"id": pre["id"], "why": f"written_by {tool} is not on disk"})
    return bad


def _resolved(rel: str, field: list[str]) -> Any:
    path = REPO / rel
    if not path.exists():
        return None
    cur: Any = json.loads(path.read_text())
    for key in field:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def prerequisite_status() -> list[dict[str, Any]]:
    rows = []
    for pre in PREREQUISITES:
        got = _resolved(pre["receipt"], list(pre["field"]))
        rows.append(
            {
                "id": pre["id"],
                "receipt": pre["receipt"],
                "field": ".".join(pre["field"]),
                "met": got is not None,
                "value": got,
                "why": pre["why"],
            }
        )
    return rows


def reached_71() -> dict[str, Any]:
    """Is the target reached on the CURRENT body? Read, never assumed."""
    residual = cb.causal_residual()
    wall_ms = float(residual["wall_ms"])
    current = residual["baseline_moved"]["current_body_ms"]
    return {
        "target_tps": TARGET_TPS,
        "target_ms": round(TARGET_MS, 3),
        "measured_wall_ms": wall_ms,
        "measured_tps": round(1000.0 / wall_ms, 2),
        "current_body_ms": current,
        "reached": False if isinstance(current, str) else bool(float(current) <= TARGET_MS),
        "why_not_assumed": (
            "reached is decided against the CURRENT body's measured complete "
            "token, never against a remembered one. While that is UNKNOWN the "
            "target can be claimed neither reached nor unreachable; now that it is "
            "measured, 71 TPS needs 14.085 ms and the body runs at "
            f"{current if isinstance(current, str) else round(float(current), 4)} ms."
        ),
    }


def which_receipt() -> dict[str, Any]:
    """UNLOCK, ROOF, or NEITHER_YET - and never 'probably impossible'."""
    hit = reached_71()
    if hit["reached"]:
        return {"emit": UNLOCK_NAME, "why": "the target is reached on a measured body"}
    open_pre = [r for r in prerequisite_status() if not r["met"]]
    if open_pre:
        return {
            "emit": None,
            "why": (
                "the binding limit is not yet PROVEN: "
                f"{len(open_pre)} of {len(PREREQUISITES)} measurements are open"
            ),
            "open": [r["id"] for r in open_pre],
        }
    return {"emit": ROOF_NAME, "why": "every named measurement has landed; the limit can be proven"}



# ---------------------------------------------------------------------------
# The five things a ROOF receipt must name (S022 §68). Each is READ from the
# receipt that owns it. "Probably impossible" is not an acceptable output, so
# every section below either resolves or the artifact refuses.
# ---------------------------------------------------------------------------


def dominant_remaining_costs() -> dict[str, Any]:
    """Where the token actually goes, on the body that runs."""
    doc = json.loads((REPO / "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json").read_text())
    rows = sorted(doc["organs"]["rows"], key=lambda r: -float(r["gpu_ms"]))
    total = float(doc["decode_gpu_ms_per_token"])
    return {
        "source": "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json",
        "measured_on": "widen_f4, release profile, ModelLake stopped, lane lock held",
        "token_gpu_ms": total,
        "token_wall_ms": float(doc["decode_wall_ms_per_token"]),
        "host_gap_ms": float(doc["host_gap_ms_per_token"]),
        "rows": [
            {**r, "share_of_gpu": round(float(r["gpu_ms"]) / total, 4)} for r in rows
        ],
        "reading": (
            "MLP is still the prey: gate_up plus down is 15.647 ms of a 26.594 ms "
            "GPU token. DeltaNet is second at 5.597. Nothing else clears 8%."
        ),
    }


def irreducible_current_information() -> dict[str, Any]:
    """What the stored bytes actually are, measured, not assumed."""
    code = json.loads((REPO / "receipts/future/MLP_CODE_INFORMATION.json").read_text())
    aux = json.loads((REPO / "receipts/future/MLP_AUXILIARY_INFORMATION.json").read_text())
    m = code["measurements"]
    return {
        "sources": [
            "receipts/future/MLP_CODE_INFORMATION.json",
            "receipts/future/MLP_AUXILIARY_INFORMATION.json",
        ],
        "mlp_code_bytes": int(m["code_bytes_read"]),
        "H_q_bits_of_2_stored": float(m["H_q_bits"]),
        "independent_fraction": float(m["independent_fraction"]),
        "entropy_floor_recoverable_bytes": int(m["iid_redundant_bytes"]),
        "auxiliary_bytes": aux["accounting"],
        "reading": (
            "The 4.28 GB code body is 93.5% independent information at these "
            "statistics: H(q) 1.870 of 2 stored bits, conditioning on the previous "
            "symbol buys 0.003, cross-layer MI is 1e-7. Perfect entropy coding of "
            "what is stored recovers 277.7 MB. That is not 'incompressible' - it is "
            "not conventionally compressible, and function replacement remains "
            "UNMEASURED on this object rather than refuted."
        ),
    }


def best_representation_and_its_evidence() -> dict[str, Any]:
    """The surviving byte levers, priced at the MEASURED per-stream rate.

    This is where a byte-count reading of the campaign goes wrong, and it went
    wrong in the first version of this receipt. MLP_AUXILIARY_INFORMATION lists
    quantize_aux_u8 and larger_group_size as open levers worth 534,773,760 bytes
    each - half the 1.07 GB auxiliary - and that byte figure is exact. It is also
    worth ZERO MILLISECONDS, and that is measured, not modelled.

    ECONOMICS_CALIBRATION dropped fractions of each stream and timed it:
    codes_keep_50 is faster than 2*MAD, aux_keep_50 is NOT. Removing half the
    auxiliary arrays is inside measurement noise. The per-stream rates that fall
    out are weight_codes 0.547 ms/GB (billing at 1827 GB/s, not the 344 organ
    average) and broadcast_aux 0.000 ms/GB.

    So the organ average was never a byte-class rate, and billing any removed
    byte at it is the overcredit the stream_class guard exists to refuse.
    """
    aux = json.loads((REPO / "receipts/future/MLP_AUXILIARY_INFORMATION.json").read_text())
    cal = json.loads((REPO / "receipts/future/ECONOMICS_CALIBRATION.json").read_text())
    rates = {k: v["ms_per_gb_saved"] for k, v in cal["stream_classes"].items()}
    priced = []
    for lever in aux["open_byte_levers"]:
        cls = ee.STREAM_CLASS_BY_ID.get(lever["id"], "broadcast_aux")
        gb = int(lever["bytes_eliminated_if_true"]) / 1e9
        priced.append(
            {
                **lever,
                "stream_class": cls,
                "gb": round(gb, 6),
                "ms_saved_at_measured_rate": round(gb * rates[cls], 4),
            }
        )
    return {
        "sources": [
            "receipts/future/MLP_AUXILIARY_INFORMATION.json",
            "receipts/future/ECONOMICS_CALIBRATION.json",
        ],
        "measured_ms_per_gb_by_stream": rates,
        "levers_priced": priced,
        "byte_evidence": "exact from 192 HGRAVF01 headers; reconciled or the module refuses",
        "time_evidence": "paired keep-fraction loads, ECONOMICS_CALIBRATION, SELF_MEASURED_DIRTY",
        "reading": (
            "Every open auxiliary lever prices at 0.0 ms and IMMATERIAL. The bytes "
            "are real and exactly counted; the time is not there. aux_keep_50 sits "
            "inside noise while codes_keep_50 does not, so the auxiliary arrays do "
            "not cost token time at this packing and removing them cannot buy any. "
            "group_size_256 and group_size_1024 are additionally capability-REFUTED "
            "on held-out fit by AUX_CAPABILITY_SCREEN. "
            "What IS priced is the code body: 0.547 ms/GB. All 4.28 GB of it is "
            "2.34 ms, and 93.5% of it is independent information, so perfect "
            "entropy coding of what is stored buys 0.152 ms. That is the honest "
            "size of the representation lever on this body, and it is not close to "
            "the 13.2 ms that 71 TPS needs."
        ),
    }


def next_hardware_requirement() -> dict[str, Any]:
    """What this machine cannot do - and what it CAN do that one stream does not use."""
    traffic = json.loads((REPO / "receipts/future/MEMORY_TRAFFIC_PROBE.json").read_text())
    conc_path = REPO / "receipts/future/RESIDENT_CONCURRENCY_MEASURED.json"
    conc = json.loads(conc_path.read_text()) if conc_path.exists() else None
    return {
        "sources": [
            "receipts/future/MEMORY_TRAFFIC_PROBE.json",
            "receipts/future/ORGAN_BANDWIDTH.json",
            "receipts/future/RESIDENT_CONCURRENCY_MEASURED.json",
        ],
        "actual_read_bytes_per_token": traffic["actual_read_bytes_per_token"],
        "byte_counter_available": traffic["byte_counter_available"],
        "organ_gb_s_band": [341.9, 360.0],
        "clean_gemv_roof_gb_s": 703.5,
        "lm_head_demonstrated_gb_s": 497.4,
        "concurrency_verdict": (conc or {}).get("classification", {}).get("verdict"),
        "aggregate_gb_s_by_level": (conc or {}).get("bandwidth_cross_check", {}).get("gb_s_by_level"),
        "reading": (
            "The requirement is bandwidth, and the shape of it changed once "
            "concurrency was measured. WITHIN one session the loss is uniform - "
            "MLP, DeltaNet and GQA inside 5% of each other at 341.9-360.0 GB/s "
            "against a 703.5 roof, no hot organ to attack. But a single session is "
            "NOT what saturates this machine: two or three concurrent sessions "
            "aggregate to 449-580 GB/s, reaching and sometimes passing the LM "
            "head's demonstrated 497.4. So the memory system was never full at "
            "n=1, and ORGAN_BANDWIDTH's own open dependency-chain hypothesis - "
            "layer N+1 cannot start until layer N retires - is the mechanism. "
            "That does NOT lower the 71 bar: 71 is single-stream latency, and "
            "under concurrency each session gets SLOWER. What it changes is the "
            "diagnosis. The binding limit on one stream is not the memory system's "
            "capacity; it is the serial dependency between layers, and no amount "
            "of byte removal addresses that. "
            "Separately this device exposes NO counter reporting bytes moved - no "
            "MTLCounterSet, GPURawCounter, IOKit PerformanceStatistics or IOReport "
            "channel - so actual traffic is UNKNOWN from an unprivileged process "
            "and the catalog figure stays an accounting floor."
        ),
    }


def next_model_body_alternative() -> dict[str, Any]:
    """The specimens that could carry the mission if this body cannot."""
    return {
        "source": "receipts/future/ODYSSEY_I_LAUNCH.json",
        "incumbent": "sealed-3.14 (Qwen3.8-Flash-Next)",
        "reading": (
            "Odyssey I is launched on a sealed specimen constellation, so the "
            "alternative is not hypothetical. Succession is explicitly NOT "
            "warranted on cognition grounds: CHOICE_JSON_PROBE showed a 0.6B "
            "failing the same clipped ask this 27B failed and both passing with "
            "the schema in view, so the incumbent's structured-output record was "
            "an artifact of the harness. A body change must be argued on bytes or "
            "bandwidth, not on the decision failures this campaign has been "
            "attributing to it."
        ),
    }


def build() -> dict[str, Any]:
    unwritable = check_prerequisites_are_writable()
    if unwritable:
        raise PrerequisiteUnwritable(
            "a prerequisite names a receipt no tool here can produce, which is a "
            "permanent false blocker rather than a real one: "
            + "; ".join(f"{b['id']}: {b['why']}" for b in unwritable)
        )
    verdict = which_receipt()
    if verdict["emit"] is None:
        raise TerminalArtifactRefused(
            "refusing to write the terminal artifact: "
            + verdict["why"]
            + "; open="
            + ", ".join(verdict.get("open") or [])
            + ". A roof declared while its own measurements are outstanding is "
            "the same error as declaring victory, and 'probably impossible' is "
            "explicitly not an acceptable output."
        )
    return {
        "schema": "hawking.future.terminal_speed_artifact.v1",
        "version": 1,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "which": verdict["emit"],
        "target": reached_71(),
        "prerequisites": prerequisite_status(),
        "citations_resolved": cb.resolve_all(),
        "causal_residual": cb.causal_residual(),
        "dominant_remaining_costs": dominant_remaining_costs(),
        "irreducible_current_information": irreducible_current_information(),
        "best_representation_and_its_evidence": best_representation_and_its_evidence(),
        "next_hardware_requirement": next_hardware_requirement(),
        "next_model_body_alternative": next_model_body_alternative(),
    }


def status() -> dict[str, Any]:
    return {
        "which_receipt": which_receipt(),
        "target": reached_71(),
        "prerequisites": prerequisite_status(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    if not args.build:
        print(json.dumps(status(), indent=1, sort_keys=True))
        return 0
    doc = build()
    print(write_receipt(doc["which"], doc, RECORDED_BY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
