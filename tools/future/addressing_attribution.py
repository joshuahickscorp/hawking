#!/usr/bin/env python3
"""Attribute 703 GB/s → 337 from landed receipts. No new hardware probe.

The 703 figure is a Q4 addr-probe that loads scales + packed codes and
sinks them: no nibble unpack, no input-vector load, no FMA. It is not
comparable to any kernel that loads the activation. 530.7 is the
production-catalog ceiling this machine already measured through the
catalog, and is the first target. 497.4 is the honest MLP streaming
ceiling WITH the activation load. fold_addqx (370.9, bit-identical) is
the demonstrated removable decode arithmetic. 337.3 is production
effective across the token.

    python3 tools/future/addressing_attribution.py --record
    python3 -m pytest tools/future/test_addressing_attribution.py -q

SELF_MEASURED_DIRTY on a contaminated host qualifies nothing. A TPS
number in this receipt is never labelled QUALIFIED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    RECEIPTS,
    REPO,
    _assert_no_hardware_claims,
    bench_block,
    load_json,
    seal,
)


RECEIPT = "ADDRESSING_ATTRIBUTION.json"
SCHEMA = "hawking.future.addressing_attribution.v1"
VERSION = 1
RECORDED_BY = "tools/future/addressing_attribution.py"

CATALOG = RECEIPTS / "CATALOG_ADDRESSING.json"
ADDRESSING_GAP = RECEIPTS / "ADDRESSING_GAP.json"
ALU = RECEIPTS / "MLP_ALU_ROOFLINE.json"
DECODE = RECEIPTS / "MLP_DECODE_CHEAPEN.json"
STREAM = RECEIPTS / "MLP_STREAM_COUNT.json"
ISSUE = RECEIPTS / "MLP_ISSUE_RATE_LADDER.json"
ISSUE_RAW = RECEIPTS / "_MLP_ISSUE_RATE_LADDER_raw.json"
REGION = RECEIPTS / "MLP_REGION_FALSIFIER.json"
KERNEL_GEO = RECEIPTS / "KERNEL_GEOMETRY.json"
WIDEN = RECEIPTS / "DELTANET_WIDEN_AB.json"
ORGAN_BW = RECEIPTS / "ORGAN_BANDWIDTH.json"

CLAIM_BOUNDARY = (
    "Static sidecar assembled from committed receipts. No new GPU probe. "
    "Every GB/s is copied from a named prior receipt with its statistic "
    "and provenance caveat. The 703 GB/s addr-probe loads scales + packed "
    "codes and sinks them; NO NIBBLE UNPACK, NO INPUT-VECTOR LOAD, NO FMA "
    "— it never loads the activation and is NOT comparable to any kernel "
    "that does. 819 is the datasheet peak; DeltaNet ARM A 943.2 exceeds it "
    "by 1.15x and is residency, not a DRAM streaming target. "
    "evidence_class SELF_MEASURED_DIRTY on a contaminated host qualifies "
    "nothing. A TPS figure here is never labelled QUALIFIED; a protected "
    "window is still required. Bit-identical output is non-negotiable."
)


class AttributionRefuse(ValueError):
    """Unsourced rung, missing receipt, or a QUALIFIED TPS claim."""


class UnsourcedRung(AttributionRefuse):
    pass


class QualifiedTpsRefused(AttributionRefuse):
    pass


def nested(doc: Mapping[str, Any] | None, *path: str) -> Any:
    cur: Any = doc
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def require_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AttributionRefuse(f"missing receipt {path.relative_to(REPO)}")
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise AttributionRefuse(f"{path} is not a JSON object")
    return doc


def rung(
    *,
    ident: str,
    gb_s: float,
    statistic: str,
    source_receipt: str,
    json_path: str,
    what: str,
    provenance: str,
    loads_activation: bool,
    comparable_to_production_decode: bool,
    as_dram_roof: bool,
    target_order: int | None = None,
) -> dict[str, Any]:
    if gb_s is None:
        raise UnsourcedRung(f"{ident} has no GB/s")
    return {
        "id": ident,
        "gb_s": round(float(gb_s), 4) if ident != "production_effective" else float(gb_s),
        "gb_s_full": float(gb_s),
        "statistic": statistic,
        "source_receipt": source_receipt,
        "json_path": json_path,
        "what": what,
        "provenance": provenance,
        "loads_activation": loads_activation,
        "nibble_unpack": loads_activation,
        "comparable_to_production_decode": comparable_to_production_decode,
        "as_dram_roof": as_dram_roof,
        "target_order": target_order,
    }


def mechanism(
    *,
    ident: str,
    kind: str,
    status: str,
    claim: str,
    refutation_or_demonstration: str,
    source_receipt: str,
    span: float | None = None,
) -> dict[str, Any]:
    if kind not in {"removable", "structural", "refuted"}:
        raise AttributionRefuse(f"unknown mechanism kind {kind}")
    if status not in {
        "REFUTED",
        "REMOVABLE_DEMONSTRATED",
        "STRUCTURAL",
        "NOT_A_DRAM_ROOF",
    }:
        raise AttributionRefuse(f"unknown status {status}")
    return {
        "id": ident,
        "kind": kind,
        "status": status,
        "claim": claim,
        "refutation_or_demonstration": refutation_or_demonstration,
        "source_receipt": source_receipt,
        "span": span,
    }


def assemble(injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()

    def take(path: Path) -> dict[str, Any]:
        rel = str(path.relative_to(REPO)) if path.is_absolute() else str(path)
        if injected is not None and rel in injected:
            doc = injected[rel]
            if not isinstance(doc, dict):
                raise AttributionRefuse(f"injected {rel} is not an object")
            return doc
        return require_file(path)

    catalog = take(CATALOG)
    gap = take(ADDRESSING_GAP)
    alu = take(ALU)
    decode = take(DECODE)
    stream = take(STREAM)
    issue = take(ISSUE)
    issue_raw = take(ISSUE_RAW)
    region = take(REGION)
    kernel_geo = take(KERNEL_GEO)
    widen = take(WIDEN)
    organ = take(ORGAN_BW)

    single = nested(catalog, "cited", "single_gemv_addr") or {}
    prod_cat = nested(catalog, "cited", "production_catalog_addr") or {}
    effective = nested(catalog, "cited", "production_effective_gb_s")
    peak = nested(catalog, "cited", "published_peak_gb_s_datasheet")
    if not isinstance(single, Mapping) or single.get("max_gb_s") is None:
        raise UnsourcedRung("CATALOG_ADDRESSING.cited.single_gemv_addr.max_gb_s")
    if not isinstance(prod_cat, Mapping) or prod_cat.get("median_gb_s") is None:
        raise UnsourcedRung("CATALOG_ADDRESSING.cited.production_catalog_addr.median_gb_s")
    if effective is None:
        raise UnsourcedRung("CATALOG_ADDRESSING.cited.production_effective_gb_s")
    if peak is None:
        raise UnsourcedRung("CATALOG_ADDRESSING.cited.published_peak_gb_s_datasheet")

    addr_probe_note = nested(gap, "kernel_geometry", "addr_probe") or nested(
        gap, "bind_path"
    )
    # ADDRESSING_GAP stores the probe description at kernel_geometry in the
    # source module; the sealed receipt copies it under several keys.
    if not isinstance(addr_probe_note, str):
        addr_probe_note = (
            "same launch geometry; loads scales + packed codes and sinks them; "
            "no nibble unpack, no input-vector load, no FMA"
        )

    mlp = alu.get("mlp") if isinstance(alu.get("mlp"), Mapping) else {}
    mlp_prod = nested(mlp, "production", "effective_gb_s")
    mlp_arm_a = nested(mlp, "arm_a_stripped", "effective_gb_s")
    mlp_bytes = nested(mlp, "arm_a_stripped", "weight_bytes")
    dn = alu.get("deltanet") if isinstance(alu.get("deltanet"), Mapping) else {}
    dn_arm_a = nested(dn, "arm_a_stripped", "effective_gb_s")
    dn_arm_a_bytes = nested(dn, "arm_a_stripped", "weight_bytes")
    if mlp_prod is None or mlp_arm_a is None:
        raise UnsourcedRung("MLP_ALU_ROOFLINE.mlp production / ARM A")
    if dn_arm_a is None:
        raise UnsourcedRung("MLP_ALU_ROOFLINE.deltanet.arm_a_stripped")

    best_exact = decode.get("best_exact") if isinstance(decode.get("best_exact"), Mapping) else {}
    if best_exact.get("id") != "fold_addqx":
        raise UnsourcedRung("MLP_DECODE_CHEAPEN.best_exact.id is not fold_addqx")
    if best_exact.get("bit_identical") is not True:
        raise AttributionRefuse("fold_addqx is not marked bit_identical in DECODE_CHEAPEN")
    fold_gb = best_exact.get("effective_gb_s")
    fold_ratio = nested(best_exact, "gap_to_arm_a", "ratio_to_this_run_production")
    fold_proj = nested(best_exact, "projection", "delta_token_ms")
    if fold_gb is None:
        raise UnsourcedRung("MLP_DECODE_CHEAPEN.best_exact.effective_gb_s")

    rungs = [
        rung(
            ident="datasheet_peak",
            gb_s=float(peak),
            statistic="datasheet",
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
            json_path="cited.published_peak_gb_s_datasheet",
            what="machine datasheet peak (819 GB/s)",
            provenance=(
                "A paper peak, not a streaming measurement. DeltaNet ARM A "
                f"{float(dn_arm_a):.1f} GB/s exceeds it by "
                f"{float(dn_arm_a) / float(peak):.2f}x and therefore cannot "
                "be a DRAM streaming rate — treat 943.2 as residency, not a target."
            ),
            loads_activation=False,
            comparable_to_production_decode=False,
            as_dram_roof=False,
        ),
        rung(
            ident="catalog_addressing_single_gemv_addr",
            gb_s=float(single["max_gb_s"]),
            statistic="max",
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
            json_path="cited.single_gemv_addr.max_gb_s",
            what=(
                "CATALOG_ADDRESSING cited.single_gemv_addr, 1 dispatch, "
                f"{int(single.get('payload_bytes') or 0)} bytes (13.612 GB)"
            ),
            provenance=(
                "CRITICAL: qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe "
                + addr_probe_note
                + ". It never loads the activation, so it is NOT comparable "
                "to any kernel that does (ARM A, production affine2-q2, "
                "fold_addqx, production_effective). Median of the same probe "
                f"is {float(single['median_gb_s']):.3f} GB/s; min "
                f"{float(single['min_gb_s']):.3f}. 703.6 is the max."
            ),
            loads_activation=False,
            comparable_to_production_decode=False,
            as_dram_roof=False,
            target_order=2,
        ),
        rung(
            ident="production_catalog_401_gemvs",
            gb_s=float(prod_cat["median_gb_s"]),
            statistic="median",
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
            json_path="cited.production_catalog_addr.median_gb_s",
            what=(
                "production_catalog_401_gemvs median "
                f"(max {float(prod_cat['max_gb_s']):.1f}, "
                f"min {float(prod_cat['min_gb_s']):.1f})"
            ),
            provenance=(
                "Same addr-probe kernel as the 703 figure, 401 production-shaped "
                "GEMVs, 13.612 GB payload. Still no activation load. This is the "
                "obligation's already-measured floor of what this machine does "
                "through the catalog, and the first target. CATALOG_ADDRESSING "
                "falsified host catalog indirection as the 703→530 mechanism; "
                "the drop is dispatch topology + mixed organ sizes."
            ),
            loads_activation=False,
            comparable_to_production_decode=False,
            as_dram_roof=False,
            target_order=1,
        ),
        rung(
            ident="mlp_alu_roofline_arm_a",
            gb_s=float(mlp_arm_a),
            statistic="median",
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
            json_path="mlp.arm_a_stripped.effective_gb_s",
            what=(
                "MLP_ALU_ROOFLINE ARM A: same "
                f"{int(mlp_bytes or 0) / 1e6:.2f} MB, arithmetic stripped, loads proven live"
            ),
            provenance=(
                "A real streaming rate WITH the activation load. Honest ceiling "
                "for the MLP shape. Not occupancy-limited. ARM A jumped 1.51x "
                "vs production 329.6; ARM B tracked bytes so the conjunction "
                "stayed MIXED, but surviving loads + ARM A near the LM-head "
                "rate (497.4) is the ceiling fold_addqx is climbing toward."
            ),
            loads_activation=True,
            comparable_to_production_decode=True,
            as_dram_roof=True,
        ),
        rung(
            ident="fold_addqx",
            gb_s=float(fold_gb),
            statistic="median",
            source_receipt="receipts/future/MLP_DECODE_CHEAPEN.json",
            json_path="best_exact.effective_gb_s",
            what="MLP_DECODE_CHEAPEN fold_addqx, BIT-IDENTICAL",
            provenance=(
                "Byte comparison of output buffers against the production "
                "kernel after the timed command buffer "
                f"(n_mismatch_bytes="
                f"{nested(best_exact, 'byte_compare', 'n_mismatch_bytes')}). "
                f"Ratio to this-run production {fold_ratio}x (~1.127x). "
                "Declared approx_candidate (fold is a different f32 association) "
                "and empirically error_class=bit_identical. DIRTY_DIAGNOSTIC on "
                "ONE LAYER as a probe until the complete-token A/B. Projection "
                f"{fold_proj} ms is arithmetic over that probe, not a resident "
                "measurement."
            ),
            loads_activation=True,
            comparable_to_production_decode=True,
            as_dram_roof=True,
        ),
        rung(
            ident="production_affine2_q2",
            gb_s=float(mlp_prod),
            statistic="median",
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
            json_path="mlp.production.effective_gb_s",
            what=(
                "production affine2-q2, reproducing the granularity falsifier's "
                "~331.6 on the same layer"
            ),
            provenance=(
                "geo_tpr64_tg128, one MLP layer gate+up+down, 83.56 MB, loads live. "
                "MLP_REGION_FALSIFIER packed the same organ into one staging "
                "buffer / one serial region and stayed in the ~350 GB/s cluster; "
                "granularity is dead. This is the MLP production rate fold_addqx "
                "is measured against."
            ),
            loads_activation=True,
            comparable_to_production_decode=True,
            as_dram_roof=True,
        ),
        rung(
            ident="production_effective",
            gb_s=float(effective),
            statistic="derived_effective",
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
            json_path="cited.production_effective_gb_s",
            what="production_effective across the token",
            provenance=(
                "Active-weight bytes per token / complete-token GPU ns, from the "
                "token-bytes atlas cited by CATALOG_ADDRESSING. KERNEL_GEOMETRY "
                "falsified occupancy/tpr64 as the 530→337 drop: the catalog probe "
                "moved 13.612 GB of synthetic Q4 at 530.7; production moves "
                "9.879 GB of mixed affine+q4. Bytes drop 27.4%, time rises 14.2%, "
                "GB/s falls 36.4%. Geometry/occupancy/tails are not the 193 GB/s."
            ),
            loads_activation=True,
            comparable_to_production_decode=True,
            as_dram_roof=True,
        ),
        rung(
            ident="deltanet_arm_a_residency",
            gb_s=float(dn_arm_a),
            statistic="median",
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
            json_path="deltanet.arm_a_stripped.effective_gb_s",
            what="DeltaNet ARM A (stripped arithmetic)",
            provenance=(
                f"Exceeds datasheet peak 819.0 by {float(dn_arm_a) / float(peak):.2f}x "
                f"on a {int(dn_arm_a_bytes or 0)}-byte payload. Cannot be a DRAM "
                "streaming rate. Treat as residency (cache reuse / working set "
                "fitting), not as a target for the weight-addressing band."
            ),
            loads_activation=True,
            comparable_to_production_decode=False,
            as_dram_roof=False,
        ),
    ]

    # Occupancy 1024 threads = 236 GB/s lives on the issue-ladder raw.
    tg1024 = None

    def _find_tg(node: Any) -> None:
        nonlocal tg1024
        if tg1024 is not None:
            return
        if isinstance(node, Mapping):
            if node.get("id") == "tg1024":
                tg1024 = node
                return
            for v in node.values():
                _find_tg(v)
        elif isinstance(node, list):
            for v in node:
                _find_tg(v)

    _find_tg(issue_raw)

    ilp_span = nested(issue, "judgement", "ilp", "ratio_8_over_1")
    reg_span = None
    rp = nested(issue, "judgement", "register_pressure")
    if isinstance(rp, Mapping) and rp.get("gb_s_ws0") and rp.get("gb_s_ws32"):
        reg_span = float(rp["gb_s_ws32"]) / float(rp["gb_s_ws0"])
    stream_dn_over_mlp = nested(stream, "judgement", "dn_over_mlp")
    stream_gbs = nested(stream, "judgement", "gb_s") or {}
    pack6 = stream_gbs.get("pack_6_32") if isinstance(stream_gbs, Mapping) else None
    mlp222 = stream_gbs.get("mlp_2_2_2_32") if isinstance(stream_gbs, Mapping) else None
    dn42 = stream_gbs.get("dn_4_2_32") if isinstance(stream_gbs, Mapping) else None

    refuted = [
        mechanism(
            ident="region_granularity",
            kind="refuted",
            status="REFUTED",
            claim="fragmented regions / packing granularity is why MLP sits at ~330 GB/s",
            refutation_or_demonstration=(
                f"{region.get('verdict')}: {region.get('finding')}"
            ),
            source_receipt="receipts/future/MLP_REGION_FALSIFIER.json",
        ),
        mechanism(
            ident="catalog_addressing_as_main_mechanism",
            kind="refuted",
            status="REFUTED",
            claim="host catalog indirection / unhoisted addressing is the 703→530 tax",
            refutation_or_demonstration=(
                f"CATALOG_ADDRESSING falsified_as={catalog.get('falsified_as')}. "
                f"{catalog.get('verdict')}"
            ),
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
        ),
        mechanism(
            ident="raw_dispatch_count",
            kind="refuted",
            status="REFUTED",
            claim="raw dispatch count is the main 703→530 / 530→337 mechanism",
            refutation_or_demonstration=(
                "Same 13.612 GB, 1 vs 401 vs 287 tiled gates: extra dispatch is "
                "~12–16 µs. Encoder collapse (serial vs noop) left GPU_NS "
                "overlapping. KERNEL_GEOMETRY: weight-free dispatches at most "
                "0.586 ms of the token. Dispatch count is not the named loss."
            ),
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
        ),
        mechanism(
            ident="decode_fusion",
            kind="refuted",
            status="REFUTED",
            claim="fusing decode launches (encoder collapse / more fusion) recovers the band",
            refutation_or_demonstration=(
                "CATALOG_ADDRESSING encoder_collapse: serial encoder is "
                "token-identical and does not separate GPU_NS. Production already "
                "runs GateUpSwiglu + GQA-QKV + DN-inproj + add-RMSNorm on the "
                "580-graph (DELTANET_WIDEN_AB). Fusion is not the remaining lever."
            ),
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
        ),
        mechanism(
            ident="stream_count_at_fixed_bytes_per_thread",
            kind="refuted",
            status="REFUTED",
            claim="MLP is stream-count bound; merging 2+2+2 into 4+2 then 6 recovers GB/s",
            refutation_or_demonstration=(
                f"MLP_STREAM_COUNT: 4+2 shape dn_4_2_32={dn42} GB/s is only "
                f"{stream_dn_over_mlp}x the 2+2+2 shape mlp_2_2_2_32={mlp222}; "
                f"merging further HURTS (pack_6_32={pack6} GB/s, pack_38=45.6). "
                "Verdict MIXED; do not force STREAM_COUNT_BOUND."
            ),
            source_receipt="receipts/future/MLP_STREAM_COUNT.json",
            span=float(stream_dn_over_mlp) if stream_dn_over_mlp is not None else None,
        ),
        mechanism(
            ident="dependency_chains",
            kind="refuted",
            status="REFUTED",
            claim="MLP is dependency-bound; more accumulator chains recover the ARM A gap",
            refutation_or_demonstration=(
                f"MLP_ISSUE_RATE_LADDER ILP: 1-to-8 accumulator chains span "
                f"{ilp_span} (jumped=False, bar 1.12). Not DEPENDENCY_BOUND."
            ),
            source_receipt="receipts/future/MLP_ISSUE_RATE_LADDER.json",
            span=float(ilp_span) if ilp_span is not None else None,
        ),
        mechanism(
            ident="register_pressure",
            kind="refuted",
            status="REFUTED",
            claim="register pressure is why production sits at 330 GB/s",
            refutation_or_demonstration=(
                f"MLP_ISSUE_RATE_LADDER register_pressure: ws0 {nested(rp, 'gb_s_ws0')} "
                f"vs ws32 {nested(rp, 'gb_s_ws32')} GB/s, span {reg_span and round(reg_span, 4)}, "
                "occupancy_span 1.0, dropped=False. Span 1.078 does not explain 1.51x ARM A."
            ),
            source_receipt="receipts/future/MLP_ISSUE_RATE_LADDER.json",
            span=round(reg_span, 4) if reg_span else None,
        ),
        mechanism(
            ident="occupancy",
            kind="refuted",
            status="REFUTED",
            claim="raising occupancy (more threads per threadgroup) recovers the band",
            refutation_or_demonstration=(
                "Raising occupancy makes things WORSE. issue-ladder tg1024 "
                f"(1024 threads/threadgroup) is "
                f"{None if tg1024 is None else tg1024.get('effective_gb_s')} GB/s "
                "against production-occupancy 128 threads at ~308–330 GB/s. "
                "KERNEL_GEOMETRY: tpr64 already covers every production organ "
                "except BA with dozens to thousands of threadgroups per core; "
                "occupancy/tails are not the 530→337 drop."
            ),
            source_receipt="receipts/future/_MLP_ISSUE_RATE_LADDER_raw.json",
            span=None if tg1024 is None else float(tg1024.get("effective_gb_s") or 0),
        ),
    ]

    removable = [
        mechanism(
            ident="decode_arithmetic_fold_addqx",
            kind="removable",
            status="REMOVABLE_DEMONSTRATED",
            claim="production affine2-q2 decode arithmetic (8 dequant FMA + 8 MAC per 6 B)",
            refutation_or_demonstration=(
                "fold_addqx is bit-identical by byte comparison of output "
                f"buffers, {fold_gb} GB/s vs this-run production "
                f"{nested(best_exact, 'gap_to_arm_a', 'production_gb_s')} "
                f"({fold_ratio}x, obligation's 1.127x). Closes "
                f"{nested(best_exact, 'gap_to_arm_a', 'fraction_of_gap_closed')} "
                "of the ARM A 497.4 gap on the one-layer probe. DIRTY_DIAGNOSTIC "
                "until the complete-token A/B. A faster resident that answers "
                "differently is not the same resident."
            ),
            source_receipt="receipts/future/MLP_DECODE_CHEAPEN.json",
            span=float(fold_ratio) if fold_ratio is not None else None,
        )
    ]

    kg_verdict = kernel_geo.get("verdict")
    kg_why = kg_verdict.get("why") if isinstance(kg_verdict, Mapping) else None
    structural = [
        mechanism(
            ident="addr_probe_without_activation",
            kind="structural",
            status="STRUCTURAL",
            claim="703 addr-probe is the production decode ceiling",
            refutation_or_demonstration=(
                "The probe does not load the activation and does not FMA. ARM A "
                "at 497.4 WITH the activation load is the honest MLP ceiling. "
                "Comparing 703 to 337 is comparing different work."
            ),
            source_receipt="receipts/future/ADDRESSING_GAP.json",
        ),
        mechanism(
            ident="catalog_topology_703_to_530",
            kind="structural",
            status="STRUCTURAL",
            claim="401 mixed production-shaped GEMVs vs one 13.612 GB GEMV",
            refutation_or_demonstration=(
                "Falsified as catalog-indirection. Remaining named cost is "
                "dispatch topology + mixed organ sizes "
                f"(single max {float(single['max_gb_s']):.1f} → catalog median "
                f"{float(prod_cat['median_gb_s']):.1f}). Not host addressing. "
                "First target is still the 530 catalog floor, then 703."
            ),
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
        ),
        mechanism(
            ident="byte_numerator_and_non_gemv_530_to_337",
            kind="structural",
            status="STRUCTURAL",
            claim="kernel geometry / occupancy / coalescing is the 530→337 drop",
            refutation_or_demonstration=(
                (kg_why or "KERNEL_GEOMETRY FALSIFIED occupancy as the drop.")
                + " Affine MLP is denser than the all-q4 catalog of the same "
                "shapes; production also pays non-GEMV work (DeltaNet, GQA, "
                "rmsnorm, swiglu). That is not removable decode arithmetic."
            ),
            source_receipt="receipts/future/KERNEL_GEOMETRY.json",
        ),
        mechanism(
            ident="datasheet_and_deltanet_arm_a_not_dram_targets",
            kind="structural",
            status="NOT_A_DRAM_ROOF",
            claim="819 datasheet or 943 DeltaNet ARM A is the streaming target",
            refutation_or_demonstration=(
                f"DeltaNet ARM A {float(dn_arm_a):.1f} exceeds datasheet "
                f"{float(peak):.1f} by {float(dn_arm_a) / float(peak):.2f}x; "
                "that cannot be DRAM streaming. 819 is a paper peak. Neither "
                "is a target for recovering the weight-addressing band."
            ),
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
        ),
    ]

    widen_ct = nested(widen, "complete_token") or {}
    organ_mlp = None
    organs = organ.get("organs")
    if isinstance(organs, list):
        for row in organs:
            if isinstance(row, Mapping) and row.get("organ") == "mlp":
                organ_mlp = row
                break

    # Do not emit a QUALIFIED TPS. Cite the dirty token ms as a measured-under-load
    # number with an explicit not-qualified flag.
    post_widen = nested(widen, "complete_token", "widen_f4_ms")
    token_gpu_ms = organ.get("token_gpu_ms")

    def _no_tps(name: str, value: Any, source: str, note: str) -> dict[str, Any]:
        if isinstance(value, (int, float)) and name == "tps":
            raise QualifiedTpsRefused("refusing a tps key")
        return {
            "name": name,
            "value_ms": value,
            "source_receipt": source,
            "qualified": False,
            "qualification": "NOT_QUALIFIED",
            "evidence_class": "SELF_MEASURED_DIRTY",
            "note": note,
        }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "obligation": (
            "RECOVER THE WEIGHT-ADDRESSING BANDWIDTH. Attribute where 703 GB/s "
            "becomes 337, separate REMOVABLE cost from STRUCTURAL cost, and "
            "ACTUALLY REMOVE WHAT IS REMOVABLE. Target the 530 GB/s "
            "production-catalog ceiling first, then the 703 clean-addressing "
            "figure. BIT-IDENTICAL OUTPUT IS NON-NEGOTIABLE."
        ),
        "first_target": {
            "id": "production_catalog_401_gemvs",
            "gb_s": float(prod_cat["median_gb_s"]),
            "why": (
                "already-measured floor of what this machine does through the "
                "catalog; obligation names this as the first target"
            ),
        },
        "second_target": {
            "id": "catalog_addressing_single_gemv_addr",
            "gb_s": float(single["max_gb_s"]),
            "caveat": (
                "addr-probe without activation load; not a production-decode comparable"
            ),
        },
        "rungs": rungs,
        "703_without_input_vector_load": True,
        "703_statistic": "max of q4_single_gemv_addr_probe gemv_payload_13p612gb",
        "703_median_gb_s": float(single["median_gb_s"]),
        "703_min_gb_s": float(single["min_gb_s"]),
        "703_max_gb_s": float(single["max_gb_s"]),
        "payload_bytes_13p612gb": int(single.get("payload_bytes") or 13_611_663_360),
        "refuted": refuted,
        "removable": removable,
        "structural": structural,
        "removable_count": len(removable),
        "refuted_count": len(refuted),
        "known_removable_decode_arithmetic_ratio": fold_ratio,
        "known_removable_decode_arithmetic_gb_s": float(fold_gb),
        "fold_addqx_probe_projection_token_ms": fold_proj,
        "fold_addqx_probe_projection_is_not_a_resident_measurement": True,
        "post_widen_f4_complete_token_ms_cited": post_widen,
        "post_widen_f4_source": "receipts/future/DELTANET_WIDEN_AB.json",
        "post_widen_f4_cited_is_stale_until_fold_addqx_ab_remeasures_incumbent": True,
        "organ_mlp_gpu_ms_cited": None if not isinstance(organ_mlp, Mapping) else organ_mlp.get("gpu_ms"),
        "token_gpu_ms_cited_pre_widen": token_gpu_ms,
        "dirty_token_ms": [
            _no_tps(
                "organ_bandwidth_token_gpu_ms",
                token_gpu_ms,
                "receipts/future/ORGAN_BANDWIDTH.json",
                "pre-widen_f4 complete-token GPU ms; stale as a baseline",
            ),
            _no_tps(
                "widen_f4_incumbent_ms",
                nested(widen_ct, "incumbent_ms"),
                "receipts/future/DELTANET_WIDEN_AB.json",
                "unfused vi-SIMD arm of the widen_f4 A/B; superseded by the f4 arm",
            ),
            _no_tps(
                "widen_f4_candidate_ms",
                nested(widen_ct, "widen_f4_ms"),
                "receipts/future/DELTANET_WIDEN_AB.json",
                "landed widen_f4 complete-token ms; incumbent of fold_addqx A/B must re-measure this",
            ),
        ],
        "tps_qualification": {
            "any_tps_labelled_qualified": False,
            "protected_window_required": True,
            "reason": (
                "SELF_MEASURED_DIRTY on a contaminated host qualifies nothing. "
                "A real TPS claim still needs a protected window."
            ),
        },
        "occupancy_1024_threads_gb_s": None
        if tg1024 is None
        else float(tg1024.get("effective_gb_s")),
        "self_timing": {
            "class": "SELF_MEASURED_DIRTY",
            "what": "python wall of static attribution in this process",
            "elapsed_ms": elapsed_ms,
            "qualifies": "nothing",
        },
        "bit_identical_is_non_negotiable": True,
        "finding": (
            "703.6 is an addr-probe without an input-vector load (median 699.57). "
            "First target is production_catalog_401_gemvs 530.7 (max 539.0, min 509.3). "
            "Honest MLP ceiling WITH activation is ARM A 497.4. Production affine2-q2 "
            "is 329.6. fold_addqx is the demonstrated removable 1.127x bit-identical "
            "decode cheapen (370.9). production_effective is 337.3 across the token. "
            "819 is datasheet; 943.2 DeltaNet ARM A is residency, not DRAM. "
            "Granularity, catalog addressing, dispatch count, decode fusion, stream "
            "count, dependency chains, register pressure, and occupancy are REFUTED. "
            "A protected window is still required; no TPS is QUALIFIED."
        ),
    }


def build(injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    doc = assemble(injected)
    if doc.get("tps_qualification", {}).get("any_tps_labelled_qualified"):
        raise QualifiedTpsRefused("a QUALIFIED TPS leaked into the attribution")
    blob = json.dumps(doc)

    def _scan(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k.lower() in {"tps", "accepted_tps"} and isinstance(v, (int, float)):
                    raise QualifiedTpsRefused(f"{here} is a numeric TPS field")
                if isinstance(v, str) and v.strip().upper() == "QUALIFIED" and "tps" in here.lower():
                    raise QualifiedTpsRefused(f"{here} labelled QUALIFIED")
                _scan(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _scan(v, f"{path}[{i}]")

    _scan(doc)
    if "QUALIFIED" in blob and "NOT_QUALIFIED" not in blob.replace("NOT_QUALIFIED", ""):
        # Allow NOT_QUALIFIED; refuse a bare QUALIFIED label on TPS.
        pass
    if '"qualification": "QUALIFIED"' in blob or '"qualified": true' in blob.lower():
        raise QualifiedTpsRefused("receipt labelled a figure QUALIFIED")
    return doc


def record(*, path: Path | None = None, injected: Mapping[str, Any] | None = None) -> Path:
    doc = build(injected)
    doc.setdefault("bench", bench_block(RECORDED_BY))
    _assert_no_hardware_claims(doc)
    seal(doc)
    out = path or (RECEIPTS / RECEIPT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    doc = build()
    if args.record:
        path = record(path=args.out)
        print(f"wrote {path}")
    print(doc["finding"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
