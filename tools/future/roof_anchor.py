"""THE ROOF ITSELF MUST BE RE-ANCHORED.

Every ceiling must name the roof it rests on. A ceiling with an unstated roof
is the defect that produced 595.9: a kernel family's scoring reference was
promoted to a machine property across three hops because nothing forced the
ceiling to say which roof it used.

This module is the roofs themselves, plus the rule made executable.
compute_ceiling() returns a number only when given a roof_id from the
registry. Passing no roof, an empty roof, a positional GB/s figure, or an
id that is not in the registry RAISES. The refusal is the deliverable.

The ceiling audit carries a source reference (artifact path, field path,
tolerance) and resolves the quoted value at evaluation time. It does not
copy a mutable number into this file. A claimed quote that disagrees with
the named source still raises QuoteDrift.

No GPU lease. Numbers are copied from named historical receipts. Those
receipts keep the wrong numbers; they are the evidence.

    python3 tools/future/roof_anchor.py --build
    python3 -m pytest tools/future/test_roof_anchor.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
from typing import Any, Mapping

from tools.future._common import git, load_json, write_receipt, REPO

RECEIPT = "ROOF_ANCHOR.json"
SCHEMA = "hawking.future.roof_anchor.v1"
VERSION = 1
RECORDED_BY = "tools/future/roof_anchor.py"

# Sealed-3.14 active weight bytes per token, from the atlas headline.
# Used only when a caller asks for a raw-TPS ceiling; not a roof.
ACTIVE_BYTES_ATLAS = 9_878_901_136

HONEST_REL = "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json"
ATLAS_REL = "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json"
CENSUS_REL = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
GENOME_REL = "receipts/headless/MACHINE_GENOME.json"
G072_REL = "receipts/ascent-2026-08-16/G072_MULTI_PLANE_GEMV.json"
G044_REL = "receipts/ascent-2026-08-16/G044_ROOFLINE_KNEE.json"
GENESIS_REL = "receipts/ascent-2026-08-18/Genesis.m3ultra.nx"
CANON_REL = "docs/ultragoals/NOETIC_CANON.md"
CATALOG_REL = "receipts/future/CATALOG_ADDRESSING.json"
ALU_REL = "receipts/future/MLP_ALU_ROOFLINE.json"
ORGAN_BW_REL = "receipts/future/ORGAN_BANDWIDTH.json"
BUDGET_REL = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"
PATH_REL = "receipts/future/PATH_TO_71.json"
ADDRESSING_GAP_REL = "receipts/future/ADDRESSING_GAP.json"
CAP_MAP_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"
METABOLISM_REL = "receipts/future/IMPROVEMENT_METABOLISM.json"

GEMV_PAYLOAD_BYTES = 13_611_663_360
MLP_ARM_A_BYTES = 83_558_400  # 83.56 MB; gate+up+down of one layer
DELTANET_ARM_A_BYTES = 44_564_480
LM_HEAD_BYTES = 675_430_440
TRIAD_BYTES_PER_REP = 805_306_368  # 64M f32 * 3 (c = a + b)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Every GB/s figure is "
    "copied from a named prior receipt or refused. Arithmetic over those "
    "copies is DERIVED_FROM_CITED_RECEIPTS, not a new experiment. A ceiling "
    "computed here names its roof_id; a ceiling without a roof_id cannot be "
    "expressed. Historical receipts that carry the wrong numbers are the "
    "evidence and are not rewritten."
)


class RoofAnchorError(ValueError):
    """Contract violation around roofs and ceilings."""


class UnstatedRoof(RoofAnchorError):
    """A ceiling with an unstated roof is the defect that produced 595.9."""


class UnknownRoof(RoofAnchorError):
    """roof_id is not in the registry. Inventing a roof is how 595.9 happened."""


# ---------------------------------------------------------------------------
# Registry. Each roof is a cited number, not a measurement this lane took.
# hops_from_origin = copy/promotion steps from the originating measurement
# (the origin itself is 0). hops_to_nearest_ceiling = steps from that origin
# to the first on-record ceiling that treats the number as a roof.
# ---------------------------------------------------------------------------


def _measured(
    *,
    shape: str,
    bytes: int | None,
    dispatches: int | None,
    activation_loaded: bool | None,
    arithmetic_ran: bool | None,
    kernel: str | None = None,
    topology: str | None = None,
    reps: int | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "shape": shape,
        "bytes": bytes,
        "dispatches": dispatches,
        "activation_loaded": activation_loaded,
        "arithmetic_ran": arithmetic_ran,
        "kernel": kernel,
        "topology": topology,
        "reps": reps,
        "note": note,
    }


def _roof(
    *,
    id: str,
    value_gb_s: float,
    source_receipt: str,
    source_field: str,
    kind: str,
    hops_from_origin: int,
    hops_to_nearest_ceiling: int,
    what_was_measured: dict[str, Any],
    origin_receipt: str,
    origin_field: str,
    caveats: tuple[str, ...],
    usable_as_production_streaming_roof: bool,
    campaign_label: float | None = None,
    also_cited_from: tuple[dict[str, str], ...] = (),
    statistic: str | None = None,
) -> dict[str, Any]:
    if kind not in {"measured", "published", "promoted", "derived"}:
        raise RoofAnchorError(f"bad kind {kind!r} for roof {id}")
    return {
        "id": id,
        "value_gb_s": value_gb_s,
        "campaign_label": campaign_label if campaign_label is not None else value_gb_s,
        "source_receipt": source_receipt,
        "source_field": source_field,
        "kind": kind,
        "measured_or_published": "published" if kind == "published" else "measured",
        "hops_from_origin": hops_from_origin,
        "hops_to_nearest_ceiling": hops_to_nearest_ceiling,
        "what_was_measured": what_was_measured,
        "origin_receipt": origin_receipt,
        "origin_field": origin_field,
        "statistic": statistic,
        "also_cited_from": list(also_cited_from),
        "caveats": list(caveats),
        "usable_as_production_streaming_roof": usable_as_production_streaming_roof,
    }


# Kind "promoted" still reports measured_or_published = "measured" because the
# *origin* was a measurement; the promotion is in hops_from_origin / caveats.
# Census 595.9 is the one case whose *this record* did not measure.

ROOFS: dict[str, dict[str, Any]] = {}


def _register(row: dict[str, Any]) -> None:
    ROOFS[row["id"]] = row


_register(
    _roof(
        id="published_peak_819",
        value_gb_s=819.0,
        source_receipt=HONEST_REL,
        source_field="hardware.published_peak_gb_s",
        kind="published",
        hops_from_origin=0,
        hops_to_nearest_ceiling=1,
        what_was_measured=_measured(
            shape="none — vendor datasheet peak for the SoC",
            bytes=None,
            dispatches=None,
            activation_loaded=None,
            arithmetic_ran=None,
            note="Not measured here. Apple M3 Ultra published peak. CATALOG_ADDRESSING copies it as cited.published_peak_gb_s_datasheet.",
        ),
        origin_receipt=HONEST_REL,
        origin_field="hardware.published_peak_gb_s",
        statistic="datasheet",
        also_cited_from=(
            {"receipt": CATALOG_REL, "field": "cited.published_peak_gb_s_datasheet"},
            {
                "receipt": "receipts/headless/ORGAN_ROOF_LEDGER.json",
                "field": "three_roofs.DEVICE_THEORETICAL.value",
            },
        ),
        caveats=(
            "vendor number, not a measurement taken in this campaign",
            "not a decode roof and not a GEMV roof",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="q4_single_gemv_addr_13p6gb_max",
        value_gb_s=703.6072736347875,
        campaign_label=703.5,
        source_receipt=HONEST_REL,
        source_field="q4_single_gemv_addr_probe[label=gemv_payload_13p612gb].spread.max",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=1,
        what_was_measured=_measured(
            shape="one concatenated Q4 GEMV at the geometry payload, addr_probe kernel",
            bytes=GEMV_PAYLOAD_BYTES,
            dispatches=1,
            activation_loaded=False,
            arithmetic_ran=False,
            kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
            topology="single_gemv",
            reps=5,
            note=(
                "Loads scales + packed codes and sinks them. NO nibble unpack, "
                "NO input-vector load, NO FMA. Shader: crates/hawking-core/shaders/"
                "qwen_uniform_q4.metal. Median of the same probe is 699.5736545106142; "
                "min 693.1508595217028. Campaign 703.5 is this MAX rounded, not the median. "
                "ADDRESSING_GAP refused 703.5-as-median. Timing label "
                "GPU_PROTECTED_CPU_CONTENDED; clean_box false."
            ),
        ),
        origin_receipt=HONEST_REL,
        origin_field="q4_single_gemv_addr_probe[label=gemv_payload_13p612gb].spread.max",
        statistic="max",
        also_cited_from=(
            {"receipt": CATALOG_REL, "field": "cited.single_gemv_addr.max_gb_s"},
            {"receipt": ALU_REL, "field": "clean_gemv_gb_s"},
            {"receipt": BUDGET_REL, "field": "ladder[rung=every organ at the clean GEMV roof 703.5 GB/s]"},
        ),
        caveats=(
            "no input-vector load — kernel shape does not exist in production",
            "no nibble unpack, no FMA",
            "campaign 703.5 is not the median (median 699.57, max 703.61)",
            "contended host; absolute value wants a clean paired rerun",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="q4_single_gemv_addr_13p6gb_median",
        value_gb_s=699.5736545106142,
        campaign_label=699.6,
        source_receipt=HONEST_REL,
        source_field="q4_single_gemv_addr_probe[label=gemv_payload_13p612gb].spread.median",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=1,
        what_was_measured=_measured(
            shape="same probe as q4_single_gemv_addr_13p6gb_max; the median of five reps",
            bytes=GEMV_PAYLOAD_BYTES,
            dispatches=1,
            activation_loaded=False,
            arithmetic_ran=False,
            kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
            topology="single_gemv",
            reps=5,
            note="Same kernel shape as the 703.5 max. Still no activation load.",
        ),
        origin_receipt=HONEST_REL,
        origin_field="q4_single_gemv_addr_probe[label=gemv_payload_13p612gb].spread.median",
        statistic="median",
        also_cited_from=(
            {"receipt": CATALOG_REL, "field": "cited.single_gemv_addr.median_gb_s"},
        ),
        caveats=(
            "no input-vector load — kernel shape does not exist in production",
            "this is the honest statistic of the probe the campaign labelled 703.5",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="g072_family_scoring_595p9",
        value_gb_s=595.9,
        source_receipt=G072_REL,
        source_field="measured_roof_gb_s",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=3,
        what_was_measured=_measured(
            shape=(
                "same-run roofline sweep at low arithmetic intensity, used as the "
                "scoring reference for qwen_binary_planes_k{1,2,3}_matvec_geo_tpr64_tg128"
            ),
            bytes=None,
            dispatches=None,
            activation_loaded=False,
            arithmetic_ran=None,
            kernel="roofline sweep in the G072 run (not the plane GEMVs themselves)",
            note=(
                "G072 roof_basis: the roofline sweep in the SAME run peaks at 595.9 GB/s "
                "at low arithmetic intensity, so that is the ceiling these kernels are "
                "scored against rather than a figure from another day. Genesis.m3ultra.nx "
                "roof_provenance still names this. It is a family's scoring reference, "
                "not a machine property."
            ),
        ),
        origin_receipt=G072_REL,
        origin_field="measured_roof_gb_s",
        statistic="same_run_roofline_peak",
        caveats=(
            "kernel-family scoring reference, not a machine DRAM roof",
            "promoted to a machine property across three hops (see TRACE_595P9)",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="census_promoted_595p9",
        value_gb_s=595.9,
        source_receipt=CENSUS_REL,
        source_field="artifact.anchors_not_rederived.measured_roof_GB_s",
        kind="promoted",
        hops_from_origin=3,
        hops_to_nearest_ceiling=0,
        what_was_measured=_measured(
            shape="nothing in this receipt — copied, not measured",
            bytes=None,
            dispatches=None,
            activation_loaded=None,
            arithmetic_ran=None,
            note=(
                "NOETIC_ORGAN_CENSUS lists 595.9 under anchors_not_rederived. The census "
                "did not measure bandwidth. This is hop 3 of the promotion that turned "
                "G072's family scoring reference into 'the measured ceiling of the hardware'."
            ),
        ),
        origin_receipt=G072_REL,
        origin_field="measured_roof_gb_s",
        statistic="anchor_not_rederived",
        caveats=(
            "this record did not measure 595.9",
            "unstated roof: the census does not name G072",
            "the defect that produced 595.9 as a machine property",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="machine_genome_f32_triad_589p73",
        value_gb_s=589.73,
        source_receipt=GENOME_REL,
        source_field="measured_bandwidth.median_gb_s",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=2,
        what_was_measured=_measured(
            shape="triad c = a + b, f32; 67,108,864 elements; 2 reads + 1 write",
            bytes=TRIAD_BYTES_PER_REP,
            dispatches=None,
            activation_loaded=False,
            arithmetic_ran=True,
            kernel="STREAM triad (not a GEMV, not Q4, not production decode)",
            topology="one_access_pattern",
            reps=30,
            note=(
                "warmup 8. q1 585.34, q3 596.41, IQR 1.89%. reliable=true. "
                "is_theoretical_roof=false. Genome note: 'one access pattern on one "
                "dtype; not the SoC roof and not a workload-reachable roof'. "
                "measurement_conditions.contended=false (lake fill SIGSTOPped)."
            ),
        ),
        origin_receipt=GENOME_REL,
        origin_field="measured_bandwidth.median_gb_s",
        statistic="f32_triad_median",
        also_cited_from=(
            {"receipt": ATLAS_REL, "field": "identities.machine.measured_dram_gbps"},
            {"receipt": ATLAS_REL, "field": "THE_CEILING.measured_roof_gb_s"},
        ),
        caveats=(
            "honestly measured of an f32 STREAM triad, not of Q4 weight streaming",
            "the genome itself says this is not the SoC roof and not workload-reachable",
            "ATLAS THE_CEILING used 589.73 as 'a perfect 589.73 GB/s machine' without naming the triad",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="q4_catalog_addr_401",
        value_gb_s=530.6544688491846,
        campaign_label=530.7,
        source_receipt=HONEST_REL,
        source_field="q4_production_catalog_addr_probe.spread.median",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=1,
        what_was_measured=_measured(
            shape="401 production-shaped GEMVs (192 MLP + 144 DN + 64 GQA + 1 lm_head), addr_probe",
            bytes=GEMV_PAYLOAD_BYTES,
            dispatches=401,
            activation_loaded=False,
            arithmetic_ran=False,
            kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
            topology="production_shape_catalog",
            note=(
                "Same kernel as the 13.6 GB single GEMV, mixed organs, unique synthetic Q4. "
                "Still NO nibble unpack, NO input-vector load, NO FMA. max 539.031, min 509.271. "
                "Not sealed-3.14 production decode. Same receipt's catalog_full (with dequant + "
                "input + FMA) median is 505.81."
            ),
        ),
        origin_receipt=HONEST_REL,
        origin_field="q4_production_catalog_addr_probe.spread.median",
        statistic="median",
        also_cited_from=(
            {"receipt": CATALOG_REL, "field": "cited.production_catalog_addr.median_gb_s"},
        ),
        caveats=(
            "catalog topology, but still addr_probe — no activation load",
            "a catalog-path addressing floor, not a production decode roof",
            "not sealed-3.14; uniform-Q4 synthetic payload",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="mlp_arm_a_stripped_497p4",
        value_gb_s=497.4,
        source_receipt=ALU_REL,
        source_field="mlp.arm_a_stripped.effective_gb_s",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=0,
        what_was_measured=_measured(
            shape=(
                "one representative sealed-3.14 MLP layer (gate+up+down), ARM A: "
                "production access pattern and byte count, decode+dequant+FMA replaced "
                "with a XOR/add sink"
            ),
            bytes=MLP_ARM_A_BYTES,
            dispatches=3,
            activation_loaded=True,
            arithmetic_ran=False,
            kernel="alu_roofline_affine_q2_geo_tpr64_tg128_stripped",
            topology="production_mlp_layer",
            reps=11,
            note=(
                "83.56 MB. 1 command buffer, 1 encoder. Loads proven live: stripped "
                "time 168000 ns exceeds zero-load floor 32416 ns and drops when bytes "
                "are halved (arm_a_halfk 91875 ns). evidence_class SELF_MEASURED_DIRTY; "
                "absolute GB/s is measured-under-load. Independently equal to the LM "
                "head's demonstrated production rate."
            ),
        ),
        origin_receipt=ALU_REL,
        origin_field="mlp.arm_a_stripped.effective_gb_s",
        statistic="median_effective_gb_s",
        caveats=(
            "arithmetic stripped — this is a streaming rate with the activation load, not production FMA",
            "one representative layer, measured-under-load (SELF_MEASURED_DIRTY)",
            "not a datasheet peak and not a clean-box roof",
        ),
        usable_as_production_streaming_roof=True,
    )
)

_register(
    _roof(
        id="lm_head_production_497p4",
        value_gb_s=497.4,
        source_receipt=ORGAN_BW_REL,
        source_field="organs[organ=lm_head].effective_gb_s",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=0,
        what_was_measured=_measured(
            shape="production lm_head on a traced sealed-3.14 decode token; activation + arithmetic both live",
            bytes=LM_HEAD_BYTES,
            dispatches=2,
            activation_loaded=True,
            arithmetic_ran=True,
            kernel="production lm_head (same catalog, same low-bit representation, same build)",
            topology="production_decode_organ",
            note=(
                "675,430,440 bytes in 2 dispatches, 1.358 ms GPU, 337.7 MB/dispatch. "
                "ORGAN_BANDWIDTH: 'the LM head proves the roof is reachable'. "
                "MLP_ALU_ROOFLINE copies the same 497.4 as lm_head_gb_s. Independent "
                "of ARM A landing on the same number with arithmetic stripped."
            ),
        ),
        origin_receipt=ORGAN_BW_REL,
        origin_field="organs[organ=lm_head].effective_gb_s",
        statistic="effective_gb_s",
        also_cited_from=(
            {"receipt": ALU_REL, "field": "lm_head_gb_s"},
            {"receipt": BUDGET_REL, "field": "measured_now.organs[organ=lm_head].gb_s"},
        ),
        caveats=(
            "one organ, not every organ — causal-budget 47.97 assumes the others can match it",
            "production arithmetic is live, so this is a demonstrated organ rate, not a stripped streaming ceiling",
        ),
        usable_as_production_streaming_roof=True,
    )
)

_register(
    _roof(
        id="deltanet_arm_a_stripped_943p2",
        value_gb_s=943.2,
        source_receipt=ALU_REL,
        source_field="deltanet.arm_a_stripped.effective_gb_s",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=0,
        what_was_measured=_measured(
            shape="one representative DeltaNet qkvz, ARM A stripped (XOR/add sink, production access pattern)",
            bytes=DELTANET_ARM_A_BYTES,
            dispatches=1,
            activation_loaded=True,
            arithmetic_ran=False,
            kernel="alu_roofline_q4_geo_tpr64_tg128_stripped",
            topology="production_deltanet_qkvz",
            reps=11,
            note=(
                "44.56 MB, 1 CB, 1 encoder, median 47250 ns. 943.2 / 819.0 = 1.152. "
                "Exceeds the published peak, so it cannot be a DRAM streaming rate. "
                "Residency or accounting (bytes-believed over GPU ns with cache hits), not a roof."
            ),
        ),
        origin_receipt=ALU_REL,
        origin_field="deltanet.arm_a_stripped.effective_gb_s",
        statistic="median_effective_gb_s",
        caveats=(
            "exceeds published peak 819 by 1.15x — refused as a DRAM roof",
            "residency or byte-accounting, not a streaming ceiling",
        ),
        usable_as_production_streaming_roof=False,
    )
)

_register(
    _roof(
        id="g044_f4_sweep_594p35",
        value_gb_s=594.3492381201206,
        campaign_label=594.35,
        source_receipt=G044_REL,
        source_field="bandwidth_ceiling_gb_s",
        kind="measured",
        hops_from_origin=0,
        hops_to_nearest_ceiling=1,
        what_was_measured=_measured(
            shape="hawking_roofline_sweep_f4: 1 GiB float4, K FMAs per vector, bandwidth plateau",
            bytes=1_073_741_824,
            dispatches=None,
            activation_loaded=False,
            arithmetic_ran=True,
            kernel="hawking_roofline_sweep_f4",
            note=(
                "First sweep point 594.35 GB/s at 0.5 flop/B, 1 806 584 ns. Not a GEMV "
                "genome and not a decode roof. Neighbour of G072's 595.9 scoring peak; "
                "not the same run and not interchangeable with it."
            ),
        ),
        origin_receipt=G044_REL,
        origin_field="bandwidth_ceiling_gb_s",
        statistic="roofline_sweep_f4_plateau",
        caveats=(
            "float4 streaming sweep, not a GEMV, not production decode",
            "do not collapse onto 595.9 — different receipt, different run",
        ),
        usable_as_production_streaming_roof=False,
    )
)

REQUIRED_ROOF_FIELDS = (
    "id",
    "value_gb_s",
    "source_receipt",
    "source_field",
    "what_was_measured",
    "measured_or_published",
    "kind",
    "hops_from_origin",
    "hops_to_nearest_ceiling",
    "origin_receipt",
    "origin_field",
    "caveats",
    "usable_as_production_streaming_roof",
)

REQUIRED_MEASURED_FIELDS = (
    "shape",
    "bytes",
    "dispatches",
    "activation_loaded",
    "arithmetic_ran",
)


# ---------------------------------------------------------------------------
# 589.73 and 595.9 hop traces. Same method: name every copy.
# ---------------------------------------------------------------------------

TRACE_589P73: dict[str, Any] = {
    "value_gb_s": 589.73,
    "is_scoring_reference_promoted": False,
    "is_honestly_measured": True,
    "measured_of": (
        "f32 STREAM triad c = a + b, 67,108,864 elements, 805,306,368 bytes "
        "moved per rep (2 reads + 1 write), 30 reps, warmup 8"
    ),
    "same_defect_class_as_595p9": (
        "Not a second scoring-reference origin. It is a second instance of the "
        "UNSTATED-ROOF promotion: a one-shape measurement treated as the machine "
        "roof for a different workload. The genome warned (is_theoretical_roof="
        "false; 'not the SoC roof and not a workload-reachable roof') and ATLAS "
        "THE_CEILING used it as 'a perfect 589.73 GB/s machine' anyway."
    ),
    "hops_to_atlas_ceiling": 2,
    "hops": [
        {
            "hop": 0,
            "role": "origin_measurement",
            "receipt": GENOME_REL,
            "field": "measured_bandwidth.median_gb_s",
            "value_gb_s": 589.73,
            "what": (
                "pattern 'triad c = a + b, f32'. is_theoretical_roof=false. "
                "contended=false for the window."
            ),
        },
        {
            "hop": 1,
            "role": "renamed_to_measured_dram",
            "receipt": ATLAS_REL,
            "field": "identities.machine.measured_dram_gbps",
            "value_gb_s": 589.73,
            "what": (
                "Copied and renamed. identities.machine.receipt still names "
                "MACHINE_GENOME.json, but the field is now 'measured DRAM'."
            ),
        },
        {
            "hop": 2,
            "role": "unstated_machine_roof_for_a_q4_decode_ceiling",
            "receipt": ATLAS_REL,
            "field": "THE_CEILING.measured_roof_gb_s",
            "value_gb_s": 589.73,
            "ceiling_field": "THE_CEILING.raw_tps_ceiling_at_100pct_of_roof",
            "ceiling_value": 59.69591069708626,
            "what": (
                "THE_CEILING.measured_roof_gb_s is a number. It does not name the "
                "triad. Consequence text: 'a perfect 589.73 GB/s machine tops out "
                "at 59.70 raw TPS' against 9.879 GB of Q4 production-decode weight "
                "traffic. Census 595.9 and honest 699.57 were not used. This is "
                "the unstated-roof defect on a honestly-measured one-pattern number."
            ),
        },
    ],
}

TRACE_595P9: dict[str, Any] = {
    "value_gb_s": 595.9,
    "is_scoring_reference_promoted": True,
    "is_honestly_measured": (
        "as a same-run roofline-sweep peak used to score one kernel family; "
        "not as a machine DRAM roof"
    ),
    "hops_to_machine_property": 3,
    "traced_by": CANON_REL,
    "hops": [
        {
            "hop": 0,
            "role": "family_scoring_reference",
            "receipt": G072_REL,
            "field": "measured_roof_gb_s",
            "value_gb_s": 595.9,
            "what": (
                "G072 roof_basis: the roofline sweep in the SAME run peaks at "
                "595.9 GB/s at low arithmetic intensity, so that is the ceiling "
                "qwen_binary_planes_k{1,2,3}_matvec_geo_tpr64_tg128 are scored against."
            ),
        },
        {
            "hop": 1,
            "role": "promoted_into_the_nx_machine_genome",
            "receipt": GENESIS_REL,
            "field": "compiled_for_machine_genome.measured_roof_gb_s",
            "value_gb_s": 595.9,
            "what": (
                "Genesis.m3ultra.nx writes 595.9 under compiled_for_machine_genome. "
                "roof_provenance still names 'roofline sweep in the G072 run, low "
                "arithmetic intensity peak'. The number has moved into a machine genome; "
                "the provenance has not yet been dropped."
            ),
        },
        {
            "hop": 2,
            "role": "hardcoded_as_anchor_roof",
            "receipt": CANON_REL,
            "field": "S018 §10 ANCHOR_ROOF_GB_S = 595.9",
            "value_gb_s": 595.9,
            "what": (
                "NOETIC_CANON: three design files hardcoded ANCHOR_ROOF_GB_S = 595.9 "
                "(72.8% of spec), a constant with no provenance in this campaign."
            ),
        },
        {
            "hop": 3,
            "role": "machine_property_in_later_receipts",
            "receipt": CENSUS_REL,
            "field": "artifact.anchors_not_rederived.measured_roof_GB_s",
            "value_gb_s": 595.9,
            "what": (
                "NOETIC_ORGAN_CENSUS copies 595.9 as an unre-derived machine anchor. "
                "Later receipts called it 'the measured ceiling of the hardware'. "
                "A kernel family's scoring reference became the machine's roof across "
                "three honest hops. This is the defect an unstated roof produces."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Recommended anchor. Checked against the alternatives, not inherited.
# ---------------------------------------------------------------------------

RECOMMENDED_ANCHOR_ID = "mlp_arm_a_stripped_497p4"

RECOMMENDED_ANCHOR: dict[str, Any] = {
    "roof_id": RECOMMENDED_ANCHOR_ID,
    "value_gb_s": 497.4,
    "corroborated_by": "lm_head_production_497p4",
    "why": (
        "497.4 is the honest production-shaped streaming roof: it is measured "
        "WITH the activation load on a real organ shape (one sealed-3.14 MLP "
        "layer, 83.56 MB, 3 dispatches, loads proven live against the zero-load "
        "floor), and it is independently what the LM head already achieves in "
        "production (675.4 MB, 2 dispatches, arithmetic live). Two measurements, "
        "same number, both load x. That is the shape production has."
    ),
    "agrees_with_the_brief": True,
    "disagreement": None,
    "against": {
        "published_peak_819": {
            "value_gb_s": 819.0,
            "rejected_because": (
                "vendor datasheet, not measured here. A ceiling resting on 819 "
                "is a datasheet ceiling, not a demonstrated one."
            ),
        },
        "q4_single_gemv_addr_13p6gb_max": {
            "value_gb_s": 703.5,
            "rejected_because": (
                "addr_probe: no nibble unpack, no input-vector load, no FMA. "
                "ADDRESSING_GAP states that kernel shape does not exist in "
                "production. Also not the median of its own probe (699.57). "
                "A ceiling resting on 703.5 is a ceiling for a kernel that "
                "never loads the activation."
            ),
        },
        "g072_family_scoring_595p9": {
            "value_gb_s": 595.9,
            "rejected_because": (
                "already scarred: a kernel family's scoring reference promoted "
                "to a machine property across three hops. Using it again repeats "
                "the defect."
            ),
        },
        "machine_genome_f32_triad_589p73": {
            "value_gb_s": 589.73,
            "rejected_because": (
                "honestly measured of an f32 STREAM triad (2 reads + 1 write), "
                "not of Q4 weight streaming with an activation. The genome "
                "itself says it is not the SoC roof and not workload-reachable. "
                "ATLAS used it as the machine roof without naming the triad."
            ),
        },
        "q4_catalog_addr_401": {
            "value_gb_s": 530.7,
            "rejected_because": (
                "production-shaped CATALOG (401 mixed organs) but still the "
                "addr_probe kernel: no activation load. It is the catalog-path "
                "addressing floor, not a decode roof. The same receipt's "
                "catalog_full (dequant + input + FMA) median is 505.81, which "
                "sits next to 497.4 and does not overturn it."
            ),
        },
        "deltanet_arm_a_stripped_943p2": {
            "value_gb_s": 943.2,
            "rejected_because": (
                "exceeds the 819 published peak by 1.15x, so it cannot be a "
                "DRAM streaming rate. Residency or accounting, not a roof."
            ),
        },
    },
    "catalog_full_sibling_does_not_overturn": {
        "receipt": HONEST_REL,
        "field": "q4_production_catalog_full.spread.median",
        "value_gb_s": 505.8100047843556,
        "reading": (
            "401-GEMV catalog with the FULL kernel (address + dequant + input "
            "load + FMA) lands at 505.81, inside 2% of 497.4. Uniform-Q4 "
            "synthetic, not sealed-3.14. Confirms the production-shaped-with-"
            "activation band; does not replace the sealed-3.14 pair."
        ),
    },
}


# ---------------------------------------------------------------------------
# Ceiling audit. Which roof each on-record ceiling rests on, including silence.
#
# Row ids DO NOT encode their value. They used to - causal_budget_demonstrated_47p97,
# _66p54, path_to_71_best_composed_42p36 - and tests keyed on those strings, so
# correcting the number broke tests that had nothing to do with the correction.
# An id that carries a measurement is a calendar entry: it is wrong the moment
# the measurement improves.
#
# quoted_value is not a number copied into this module. Each row carries a
# SOURCE REFERENCE (artifact path, field path, tolerance) and the value is
# resolved from that source at evaluation time. PATH_TO_71.best_composed_tps
# moved 42.36 -> 49.8 and a static 42.36 in this file made collection raise
# QuoteDrift against a producer that had been rebased. The consumer follows
# the producer. Passing a claimed quoted_value that disagrees with the source
# still raises QuoteDrift — that is the guard, and it is not weakened.
# ---------------------------------------------------------------------------


class QuoteDrift(RuntimeError):
    """A claimed quote disagrees with the source the audit names."""


_UNRESOLVABLE = object()
_RESOLVE = object()


def source_ref(*, artifact: Any, field: Any, tolerance: Any) -> dict[str, Any]:
    """A citation resolved at evaluation time, not a number copied here.

    Missing any of the three parts RAISES. There is no default artifact, no
    default field, and no default tolerance: a partial reference is how a
    hop drops its source.
    """
    if artifact is None or (isinstance(artifact, str) and not artifact.strip()):
        raise RoofAnchorError("source reference missing artifact path")
    if field is None or (isinstance(field, str) and not str(field).strip()):
        raise RoofAnchorError("source reference missing field path")
    if tolerance is None or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise RoofAnchorError("source reference missing numeric tolerance")
    if float(tolerance) < 0:
        raise RoofAnchorError("source reference tolerance must be >= 0")
    return {
        "artifact": str(artifact),
        "field": str(field),
        "tolerance": float(tolerance),
    }


def _bind_source(
    *,
    source: Mapping[str, Any] | None,
    receipt: str | None,
    field: str | None,
) -> dict[str, Any]:
    if source is not None:
        return source_ref(
            artifact=source.get("artifact"),
            field=source.get("field"),
            tolerance=source.get("tolerance"),
        )
    if receipt is None or field is None:
        raise RoofAnchorError(
            "audit row needs a source reference (artifact path, field path, tolerance)"
        )
    return source_ref(artifact=receipt, field=field, tolerance=1e-6)


def _load_artifact(artifact: str) -> Any:
    """Read a JSON artifact from disk, or from HEAD if this checkout is sparse.

    An absolute path is the caller's file (tests mutate a temp producer). A
    repo-relative path prefers the working tree and falls back to git show;
    sparse-absent is not campaign-absent.
    """
    if _os.path.isabs(artifact):
        if not _os.path.isfile(artifact):
            return _UNRESOLVABLE
        try:
            with open(artifact, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return _UNRESOLVABLE
    rp = REPO / artifact
    if rp.is_file():
        try:
            return json.loads(rp.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            return _UNRESOLVABLE
    blob = git("show", f"HEAD:{artifact}")
    if not blob:
        return _UNRESOLVABLE
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return _UNRESOLVABLE


def _resolve_field(receipt_rel: str, field: str) -> Any:
    """Read the quoted field out of the artifact it names.

    Grammar in use: dotted keys, name[key=value] to select one row of a list,
    name[] for the whole list. Anything else resolves to _UNRESOLVABLE and is
    RECORDED as unresolvable rather than silently passing.
    """
    cur: Any = _load_artifact(receipt_rel)
    if cur is _UNRESOLVABLE:
        return _UNRESOLVABLE
    for seg in re.findall(r"[^.\[\]]+(?:\[[^\]]*\])?", field):
        m = re.fullmatch(r"([^\[]+)(?:\[([^\]]*)\])?", seg)
        if m is None:
            return _UNRESOLVABLE
        name, sel = m.group(1).strip(), m.group(2)
        if not isinstance(cur, Mapping) or name not in cur:
            return _UNRESOLVABLE
        cur = cur[name]
        if sel is None:
            continue
        if sel == "":
            continue
        if "=" not in sel or not isinstance(cur, list):
            return _UNRESOLVABLE
        key, want = sel.split("=", 1)
        hits = [r for r in cur if isinstance(r, Mapping) and str(r.get(key.strip())) == want.strip()]
        if len(hits) != 1:
            return _UNRESOLVABLE
        cur = hits[0]
    return cur


def _leaf_for_row(got: Any) -> Any:
    if isinstance(got, (int, float, str, bool)) or got is None:
        return got
    return "<structure>"


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _drifted(got: Any, claimed: Any, tolerance: float) -> bool:
    if not (_numeric(got) and _numeric(claimed)):
        return False
    return abs(float(got) - float(claimed)) > float(tolerance) * max(abs(float(claimed)), 1.0)


def _audit_row(
    *,
    id: str,
    rests_on_roof_id: str | None,
    roof_named_in_record: bool,
    defect: str | None,
    reading: str,
    source: Mapping[str, Any] | None = None,
    receipt: str | None = None,
    field: str | None = None,
    quoted_value: Any = _RESOLVE,
    steers_priorities: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    src = _bind_source(source=source, receipt=receipt, field=field)
    artifact, field_path, tolerance = src["artifact"], src["field"], src["tolerance"]
    row = {
        "id": id,
        "receipt": artifact,
        "field": field_path,
        "source": {
            "artifact": artifact,
            "field": field_path,
            "tolerance": tolerance,
        },
        "rests_on_roof_id": rests_on_roof_id,
        "roof_named_in_record": roof_named_in_record,
        "defect": defect,
        "reading": reading,
        "steers_priorities": steers_priorities,
    }
    got = _resolve_field(artifact, field_path)
    if got is _UNRESOLVABLE:
        row["resolved"] = None
        row["quote_checked"] = False
        row["why_not_checked"] = "field is not a resolvable path in this receipt"
        row["quoted_value"] = None if quoted_value is _RESOLVE else quoted_value
    else:
        row["resolved"] = _leaf_for_row(got)
        row["quote_checked"] = True
        if quoted_value is _RESOLVE:
            row["quoted_value"] = _leaf_for_row(got)
        else:
            row["quoted_value"] = quoted_value
            if _drifted(got, quoted_value, tolerance):
                raise QuoteDrift(
                    f"{id}: {artifact}:{field_path} is {got!r} but the audit quotes "
                    f"{quoted_value!r}. An audit that records a number instead of "
                    "checking it is the defect this module exists to find."
                )
    if extra:
        row.update(extra)
    origin = row.get("inherited_from")
    displayed = row.get("quoted_value")
    if origin and row.get("quote_checked") and _numeric(displayed):
        # A quote can match its OWN receipt and still be stale, because the receipt
        # it was copied from has moved. That is the hops problem this module is
        # named for, and it fired the moment it was wired: the budget's
        # roof-on-today's-bytes rung moved 66.54 -> 65.15 when the unattributed
        # 0.321 ms entered the reconstruction, and CAPABILITY_INFORMATION_MAP and
        # IMPROVEMENT_METABOLISM were both still carrying 66.54. Both read the
        # figure rather than hard-coding it, so rebuilding them propagated 65.15
        # and this check now reads clean. Recorded, not raised - a stale
        # inheritance is a finding about the corpus, and raising would leave the
        # detector unable to report the thing it detected.
        at_origin = _resolve_field(
            str(origin),
            "ladder[rung=every organ at the clean GEMV roof 703.5 GB/s].tps",
        )
        if _numeric(at_origin):
            row["origin_value"] = at_origin
            row["inherited_quote_is_stale"] = _drifted(at_origin, displayed, tolerance)
    return row


CEILING_AUDIT: tuple[dict[str, Any], ...] = (
    _audit_row(
        id="atlas_the_ceiling",
        source=source_ref(
            artifact=ATLAS_REL,
            field="THE_CEILING.raw_tps_ceiling_at_100pct_of_roof",
            tolerance=1e-6,
        ),
        rests_on_roof_id="machine_genome_f32_triad_589p73",
        roof_named_in_record=False,
        defect="unstated_roof",
        reading=(
            "THE_CEILING.measured_roof_gb_s = 589.73 is a number. identities."
            "machine.receipt names MACHINE_GENOME, but THE_CEILING does not "
            "name the f32 triad or say is_theoretical_roof=false. Formula: "
            "589.73e9 / 9_878_901_136 = 59.70 raw TPS. Census 595.9 and "
            "honest 699.57 were not used."
        ),
        extra={
            "roof_value_gb_s": 589.73,
            "active_bytes": ACTIVE_BYTES_ATLAS,
            "formula": "roof_gb_s * 1e9 / active_weight_bytes_per_token",
        },
    ),
    _audit_row(
        id="census_anchor_595p9",
        source=source_ref(
            artifact=CENSUS_REL,
            field="artifact.anchors_not_rederived.measured_roof_GB_s",
            tolerance=1e-6,
        ),
        rests_on_roof_id="g072_family_scoring_595p9",
        roof_named_in_record=False,
        defect="unstated_roof",
        reading=(
            "The census did not measure 595.9. It anchors a kernel family's "
            "scoring reference as a machine property after three hops and does "
            "not name G072. This is the original unstated-roof defect."
        ),
        extra={"hops_from_origin": 3, "kind_of_ceiling": "machine_anchor_not_a_tps_figure"},
    ),
    _audit_row(
        id="causal_budget_demonstrated_regime",
        source=source_ref(
            artifact=BUDGET_REL,
            field="ladder[rung=every organ at the LM head's demonstrated 497.4 GB/s].tps",
            tolerance=1e-6,
        ),
        rests_on_roof_id="lm_head_production_497p4",
        roof_named_in_record=True,
        defect=None,
        reading=(
            "Named. Demonstrated regime: every organ at the LM head's 497.4 GB/s "
            "plus the 0.989 ms host gap. ARM A stripped MLP independently lands "
            "on the same 497.4. This rung names its roof. It has moved twice and "
            "both moves were accounting, not physics: 47.97 when the "
            "reconstruction summed only the organs it could name, 47.25 when an "
            "unattributed term of 0.321 ms was added, and 47.76 once that term was "
            "corrected to the 0.095 ms remainder ORGAN_BANDWIDTH measures in the "
            "run that timed both the parts and the total. The 0.321 was that "
            "remainder plus 0.226 ms of region-trace overhead and run-to-run "
            "variation, taken across two receipts."
        ),
        extra={"formula": "organ bytes / 497.4 GB/s + 0.989 ms host gap"},
    ),
    _audit_row(
        id="causal_budget_roof_on_todays_bytes",
        source=source_ref(
            artifact=BUDGET_REL,
            field="ladder[rung=every organ at the clean GEMV roof 703.5 GB/s].tps",
            tolerance=1e-6,
        ),
        rests_on_roof_id="q4_single_gemv_addr_13p6gb_max",
        roof_named_in_record=True,
        defect="wrong_roof_shape",
        steers_priorities=True,
        reading=(
            "The rung NAMES 703.5 ('clean GEMV roof') so this is not an unstated "
            "roof. It is the wrong shape: 703.5 is the addr_probe that never "
            "loads the activation. the_two_numbers_that_matter.roof_on_todays_bytes_tps "
            "is the live figure on that rung (66.54 before an unattributed term "
            "entered the reconstruction, 65.15 while that term was wrongly 0.321) "
            "and has been steering priorities. Flag: NO INPUT-VECTOR LOAD. "
            "A 66.54 TPS ceiling is a ceiling for a kernel that does not exist "
            "in production. Recomputed against the recommended 497.4 with the "
            "same host-gap formula this receipt uses, the demonstrated rung is "
            "already in this ladder, one storey down."
        ),
        extra={
            "caveat": "no_input_vector_load",
            "campaign_label_gb_s": 703.5,
            "sourced_max_gb_s": 703.6072736347875,
            "sourced_median_gb_s": 699.5736545106142,
            "host_gap_ms": 0.989,
        },
    ),
    _audit_row(
        id="causal_budget_71_target",
        source=source_ref(
            artifact=BUDGET_REL,
            field="ladder[rung=71 TPS].tps",
            tolerance=1e-6,
        ),
        rests_on_roof_id="q4_single_gemv_addr_13p6gb_max",
        roof_named_in_record=True,
        defect="wrong_roof_shape",
        reading=(
            "71 is recorded as NOT_REACHABLE_AT_THE_ROOF_ON_TODAYS_BYTES. It "
            "still rests on the 703.5 addr_probe plus either 6.7% fewer bytes "
            "or the host gap gone. Same no-input-vector-load caveat as the "
            "roof-on-today's-bytes rung."
        ),
        extra={"caveat": "no_input_vector_load"},
    ),
    _audit_row(
        id="path_to_71_campaign_target",
        source=source_ref(
            artifact=PATH_REL,
            field="gap_to_71 (target 71 TPS / 14.085 ms)",
            tolerance=1e-6,
        ),
        quoted_value=71.0,
        rests_on_roof_id=None,
        roof_named_in_record=False,
        defect="unstated_roof",
        reading=(
            "PATH_TO_71 never names a GB/s roof. 71 is a campaign target. The "
            "composed PATH_04 figure is component arithmetic over measured "
            "token_ms and listed levers, not roof_gb_s * bytes. A ceiling with "
            "no roof_id is the defect; this record cannot recover one because "
            "none was used."
        ),
        extra={"kind_of_ceiling": "campaign_target_not_roof_derived"},
    ),
    _audit_row(
        id="path_to_71_best_composed",
        source=source_ref(
            artifact=PATH_REL,
            field="gap_to_71.best_composed_tps",
            tolerance=1e-6,
        ),
        rests_on_roof_id=None,
        roof_named_in_record=False,
        defect=None,
        reading=(
            "Not a roof-derived ceiling. Component composition over measured "
            "token_ms. No roof to name; no roof was used. Listed so it is not "
            "mistaken for a 497.4 or 703.5 ceiling. The number lives on "
            "PATH_TO_71; this row names that field rather than copying it."
        ),
        extra={"kind_of_ceiling": "component_composition"},
    ),
    _audit_row(
        id="capability_map_inherits_roof_on_todays_bytes",
        source=source_ref(
            artifact=CAP_MAP_REL,
            field="answers.roof_movement_on_the_71tps_ladder.quoted_roof_on_todays_bytes",
            tolerance=1e-6,
        ),
        rests_on_roof_id="q4_single_gemv_addr_13p6gb_max",
        roof_named_in_record=True,
        defect="wrong_roof_shape",
        steers_priorities=True,
        reading=(
            "Inherits causal-budget roof-on-today's-bytes, which inherits 703.5. "
            "The note names clean GEMV 703.5 GB/s. Same no-input-vector-load "
            "caveat; this is how that rung propagated."
        ),
        extra={"caveat": "no_input_vector_load", "inherited_from": BUDGET_REL},
    ),
    _audit_row(
        id="improvement_metabolism_inherits_roof_on_todays_bytes",
        source=source_ref(
            artifact=METABOLISM_REL,
            field="cited.causal_budget.roof_on_todays_bytes_cited_tps",
            tolerance=1e-6,
        ),
        rests_on_roof_id="q4_single_gemv_addr_13p6gb_max",
        roof_named_in_record=False,
        defect="unstated_roof",
        reading=(
            "Cites roof-on-today's-bytes from the causal budget without naming "
            "703.5 in the cited field. Inherited ceiling, inherited silence "
            "about the addr_probe shape."
        ),
        extra={"caveat": "no_input_vector_load", "inherited_from": BUDGET_REL},
    ),
    _audit_row(
        id="addressing_gap_named_table",
        source=source_ref(
            artifact=ADDRESSING_GAP_REL,
            field="ceilings[]",
            tolerance=1e-6,
        ),
        quoted_value="table of raw-TPS ceilings, each with roof_name and roof_source",
        rests_on_roof_id="(each row names its own)",
        roof_named_in_record=True,
        defect=None,
        reading=(
            "Control: ADDRESSING_GAP already refuses a ceiling without a named "
            "sourced roof (UnstatedRoof). This module makes that rule the only "
            "way to compute a ceiling, not a check after the fact."
        ),
    ),
)


MINIMUM_AUDIT_IDS = (
    "atlas_the_ceiling",
    "census_anchor_595p9",
    "causal_budget_roof_on_todays_bytes",
    "path_to_71_campaign_target",
)


# ---------------------------------------------------------------------------
# Load helpers (sparse-absent is not campaign-absent).
# ---------------------------------------------------------------------------


def nested(node: Any, *path: str) -> Any:
    cur = node
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_receipt(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if path.is_file():
        try:
            return {"status": "LOADED", "rel": rel, "doc": load_json(path), "via": f"disk:{rel}"}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {"status": "REFUSED", "rel": rel, "reason": f"unreadable:{exc}", "doc": None}
    blob = git("show", f"HEAD:{rel}")
    if not blob:
        return {
            "status": "REFUSED",
            "rel": rel,
            "reason": "unseen_in_this_checkout_and_HEAD",
            "doc": None,
        }
    try:
        doc = json.loads(blob)
    except json.JSONDecodeError as exc:
        return {"status": "REFUSED", "rel": rel, "reason": f"git_unreadable:{exc}", "doc": None}
    return {"status": "LOADED", "rel": rel, "doc": doc, "via": f"git:HEAD:{rel}"}


def _is_number(value: Any) -> bool:
    return _numeric(value)


def _probe_spread(doc: Mapping[str, Any], key: str, label: str | None, field: str) -> Any:
    node = nested(doc, key)
    if isinstance(node, list):
        for row in node:
            if isinstance(row, dict) and row.get("label") == label:
                spread = row.get("spread")
                if isinstance(spread, dict):
                    return spread.get(field)
        return None
    if isinstance(node, dict):
        spread = node.get("spread")
        if isinstance(spread, dict):
            return spread.get(field)
    return None


def verify_registry_against_sources() -> dict[str, Any]:
    """Pin baked citations to the historical receipts when those receipts load.

    A miss is recorded, not silently substituted. This lane does not rewrite
    the historical receipts.
    """
    checks: list[dict[str, Any]] = []

    def check(roof_id: str, loaded: Mapping[str, Any], actual: Any) -> None:
        roof = ROOFS[roof_id]
        expected = roof["value_gb_s"]
        if loaded.get("status") != "LOADED":
            checks.append(
                {
                    "roof_id": roof_id,
                    "status": "REFUSED",
                    "reason": loaded.get("reason"),
                    "source_receipt": roof["source_receipt"],
                }
            )
            return
        if not _is_number(actual):
            checks.append(
                {
                    "roof_id": roof_id,
                    "status": "REFUSED",
                    "reason": "absent_or_non_numeric_in_source",
                    "source_receipt": roof["source_receipt"],
                    "source_field": roof["source_field"],
                }
            )
            return
        ok = abs(float(actual) - float(expected)) < 1e-9
        checks.append(
            {
                "roof_id": roof_id,
                "status": "LOADED" if ok else "MISMATCH",
                "expected": expected,
                "actual": float(actual),
                "source_receipt": roof["source_receipt"],
                "source_field": roof["source_field"],
                "via": loaded.get("via"),
            }
        )

    honest = load_receipt(HONEST_REL)
    atlas = load_receipt(ATLAS_REL)
    census = load_receipt(CENSUS_REL)
    genome = load_receipt(GENOME_REL)
    g072 = load_receipt(G072_REL)
    g044 = load_receipt(G044_REL)
    alu = load_receipt(ALU_REL)
    organ = load_receipt(ORGAN_BW_REL)
    genesis = load_receipt(GENESIS_REL)

    honest_doc = honest.get("doc") if isinstance(honest.get("doc"), dict) else {}
    check(
        "published_peak_819",
        honest,
        nested(honest_doc, "hardware", "published_peak_gb_s"),
    )
    check(
        "q4_single_gemv_addr_13p6gb_max",
        honest,
        _probe_spread(honest_doc, "q4_single_gemv_addr_probe", "gemv_payload_13p612gb", "max_gb_s"),
    )
    check(
        "q4_single_gemv_addr_13p6gb_median",
        honest,
        _probe_spread(honest_doc, "q4_single_gemv_addr_probe", "gemv_payload_13p612gb", "median_gb_s"),
    )
    check(
        "q4_catalog_addr_401",
        honest,
        _probe_spread(honest_doc, "q4_production_catalog_addr_probe", None, "median_gb_s"),
    )
    check("g072_family_scoring_595p9", g072, nested(g072.get("doc") or {}, "measured_roof_gb_s"))
    check(
        "census_promoted_595p9",
        census,
        nested(census.get("doc") or {}, "artifact", "anchors_not_rederived", "measured_roof_GB_s"),
    )
    check(
        "machine_genome_f32_triad_589p73",
        genome,
        nested(genome.get("doc") or {}, "measured_bandwidth", "median_gb_s"),
    )
    check(
        "mlp_arm_a_stripped_497p4",
        alu,
        nested(alu.get("doc") or {}, "mlp", "arm_a_stripped", "effective_gb_s"),
    )
    check(
        "deltanet_arm_a_stripped_943p2",
        alu,
        nested(alu.get("doc") or {}, "deltanet", "arm_a_stripped", "effective_gb_s"),
    )
    lm = None
    organ_doc = organ.get("doc") if isinstance(organ.get("doc"), dict) else {}
    for row in organ_doc.get("organs") or []:
        if isinstance(row, dict) and row.get("organ") == "lm_head":
            lm = row.get("effective_gb_s")
            break
    check("lm_head_production_497p4", organ, lm)
    check("g044_f4_sweep_594p35", g044, nested(g044.get("doc") or {}, "bandwidth_ceiling_gb_s"))

    atlas_roof = nested(atlas.get("doc") or {}, "THE_CEILING", "measured_roof_gb_s")
    genesis_roof = nested(
        genesis.get("doc") or {}, "compiled_for_machine_genome", "measured_roof_gb_s"
    )
    extras = {
        "atlas_the_ceiling_uses_589p73": {
            "status": "LOADED" if atlas.get("status") == "LOADED" and atlas_roof == 589.73 else atlas.get("status"),
            "actual": atlas_roof,
            "via": atlas.get("via"),
        },
        "genesis_promotes_595p9": {
            "status": "LOADED" if genesis.get("status") == "LOADED" and genesis_roof == 595.9 else genesis.get("status"),
            "actual": genesis_roof,
            "via": genesis.get("via"),
            "provenance": nested(
                genesis.get("doc") or {}, "compiled_for_machine_genome", "roof_provenance"
            ),
        },
    }
    mismatches = [c for c in checks if c["status"] == "MISMATCH"]
    refused = [c for c in checks if c["status"] == "REFUSED"]
    return {
        "checks": checks,
        "extras": extras,
        "mismatches": mismatches,
        "refused": refused,
        "ok": not mismatches,
    }


# ---------------------------------------------------------------------------
# THE RULE. A ceiling without a named roof cannot be expressed.
# ---------------------------------------------------------------------------


def get_roof(roof_id: str) -> dict[str, Any]:
    if roof_id is None or (isinstance(roof_id, str) and not roof_id.strip()):
        raise UnstatedRoof(
            "ceiling with unstated roof — this is the defect that produced 595.9"
        )
    if roof_id not in ROOFS:
        raise UnknownRoof(
            f"roof_id {roof_id!r} is not in the roof registry; "
            f"known ids: {sorted(ROOFS)}"
        )
    return ROOFS[roof_id]


def compute_ceiling(
    *args: Any,
    roof_id: str | None = None,
    active_bytes: Any = None,
) -> dict[str, Any]:
    """Raw TPS ceiling = roof_gb_s * 1e9 / active_bytes.

    roof_id is keyword-only and must be a registry id. A positional GB/s
    number, a missing id, or an empty id RAISES UnstatedRoof. An unknown id
    RAISES UnknownRoof. The result always names the roof it rests on.
    """
    if args:
        raise UnstatedRoof(
            "a ceiling must be computed with roof_id= from the registry; "
            "a raw GB/s number is not a roof (this is the defect that produced 595.9)"
        )
    roof = get_roof(roof_id)  # type: ignore[arg-type]
    if not _is_number(active_bytes) or float(active_bytes) <= 0:
        raise RoofAnchorError("active_bytes must be a positive number")
    raw = float(roof["value_gb_s"]) * 1e9 / float(active_bytes)
    return {
        "status": "LOADED",
        "roof_id": roof["id"],
        "roof_value_gb_s": roof["value_gb_s"],
        "roof_campaign_label": roof["campaign_label"],
        "roof_source_receipt": roof["source_receipt"],
        "roof_source_field": roof["source_field"],
        "roof_kind": roof["kind"],
        "roof_measured_or_published": roof["measured_or_published"],
        "hops_from_origin": roof["hops_from_origin"],
        "what_was_measured": roof["what_was_measured"],
        "caveats": list(roof["caveats"]),
        "usable_as_production_streaming_roof": roof["usable_as_production_streaming_roof"],
        "active_bytes": int(active_bytes) if float(active_bytes) == int(active_bytes) else float(active_bytes),
        "formula": "roof_gb_s * 1e9 / active_bytes",
        "raw_tps_ceiling": raw,
        "would_improve_tps": None,
        "claim": (
            "arithmetic upper bound IF this named roof were 100% utilised; "
            "not a reachable TPS"
        ),
    }


def registry_rows() -> list[dict[str, Any]]:
    return [ROOFS[k] for k in sorted(ROOFS)]


def validate_registry() -> None:
    for roof in ROOFS.values():
        missing = [f for f in REQUIRED_ROOF_FIELDS if f not in roof]
        if missing:
            raise RoofAnchorError(f"roof {roof.get('id')} missing {missing}")
        measured = roof["what_was_measured"]
        if not isinstance(measured, dict):
            raise RoofAnchorError(f"roof {roof['id']} what_was_measured is not a dict")
        miss_m = [f for f in REQUIRED_MEASURED_FIELDS if f not in measured]
        if miss_m:
            raise RoofAnchorError(f"roof {roof['id']} what_was_measured missing {miss_m}")
        if not _is_number(roof["value_gb_s"]):
            raise RoofAnchorError(f"roof {roof['id']} value_gb_s is not a number")
        if not roof["source_receipt"] or not roof["source_field"]:
            raise RoofAnchorError(f"roof {roof['id']} missing source receipt or field")
        if roof["measured_or_published"] not in {"measured", "published"}:
            raise RoofAnchorError(f"roof {roof['id']} bad measured_or_published")
        if not isinstance(roof["hops_from_origin"], int) or roof["hops_from_origin"] < 0:
            raise RoofAnchorError(f"roof {roof['id']} bad hops_from_origin")


def audit_by_id() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in CEILING_AUDIT}


def analyze() -> dict[str, Any]:
    validate_registry()
    verification = verify_registry_against_sources()
    atlas_recompute = compute_ceiling(
        roof_id="machine_genome_f32_triad_589p73",
        active_bytes=ACTIVE_BYTES_ATLAS,
    )
    recommended_recompute = compute_ceiling(
        roof_id=RECOMMENDED_ANCHOR_ID,
        active_bytes=ACTIVE_BYTES_ATLAS,
    )
    wrong_shape_recompute = compute_ceiling(
        roof_id="q4_single_gemv_addr_13p6gb_max",
        active_bytes=ACTIVE_BYTES_ATLAS,
    )
    unstated = [row for row in CEILING_AUDIT if row.get("defect") == "unstated_roof"]
    wrong_shape = [row for row in CEILING_AUDIT if row.get("defect") == "wrong_roof_shape"]
    flagged_703 = [
        row
        for row in CEILING_AUDIT
        if row.get("caveat") == "no_input_vector_load"
        or row.get("rests_on_roof_id") == "q4_single_gemv_addr_13p6gb_max"
    ]
    return {
        "rule": {
            "text": (
                "EVERY CEILING MUST STATE WHICH ROOF IT RESTS ON. A ceiling "
                "with an unstated roof is the defect that produced 595.9. "
                "compute_ceiling() raises UnstatedRoof unless roof_id is a "
                "registry key."
            ),
            "api": "compute_ceiling(*, roof_id, active_bytes)",
            "refusal": "UnstatedRoof on missing/empty/positional; UnknownRoof on an id not in the registry",
        },
        "registry": registry_rows(),
        "trace_589p73": TRACE_589P73,
        "trace_595p9": TRACE_595P9,
        "ceiling_audit": list(CEILING_AUDIT),
        "unstated_roofs_on_record": [row["id"] for row in unstated],
        "wrong_shape_on_record": [row["id"] for row in wrong_shape],
        "no_input_vector_load_flags": [row["id"] for row in flagged_703],
        "recommended_anchor": RECOMMENDED_ANCHOR,
        "recompute": {
            "atlas_against_named_589p73": atlas_recompute,
            "recommended_497p4_raw_tps": recommended_recompute,
            "703p5_raw_tps_for_comparison": wrong_shape_recompute,
            "atlas_matches_recompute": abs(
                float(atlas_recompute["raw_tps_ceiling"]) - 59.69591069708626
            )
            < 1e-6,
        },
        "source_verification": verification,
        "gpu_lock_note": (
            "/tmp/hawking-gpu-lane.lock was found WEDGED as a stale 0-byte file "
            "today and cleared. This lane took no GPU lease and fabricates no "
            "hardware number. Measurements cited here predate that clear and "
            "may be unserialised; they stay cited, not re-measured."
        ),
    }


def build() -> Any:
    analysis = analyze()
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Re-anchor the roof. Register every candidate roof with its value, "
            "source receipt and field, what was actually measured, measured-or-"
            "published, and hop count. Trace 589.73. Make an unstated roof "
            "impossible to express. Audit every ceiling on record. Recommend "
            "the production-shaped-with-activation anchor and defend it."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        **analysis,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if not args.build:
        ap.error("pass --build to write receipts/future/ROOF_ANCHOR.json")
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
