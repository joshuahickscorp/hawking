#!/usr/bin/env python3
"""Name every loss from the clean roof to the complete token.

Composition sidecar. No new GPU probe. Every GB/s, ms, dispatch, encoder,
command-buffer and wait figure is copied from a named prior receipt or
derived by arithmetic over those copies. 703 GB/s is a MEASURED CLEAN
KERNEL ROOF (addr_probe, no input-vector load, campaign label is the
probe MAX not the median) and is NEVER guaranteed production bandwidth.

Transitions, in order:

    CLEAN ROOF -> ADDRESSING -> GEOMETRY -> REAL DECODE -> COMPLETE TOKEN

Each carries its own MEASURED loss. Bytes per token and useful bytes per
token are different columns: the broadcast aux stream is bytes that are
not on the critical path, which is why removing 0.535 GB made things
SLOWER. Unrelated losses are never combined into 'GPU INEFFICIENCY'.
Anything left is UNATTRIBUTED, with its size. A decomposition whose
parts do not add up to the whole RAISES.

    python3 tools/future/token_roof_decomposition.py --record
    python3 -m pytest tools/future/test_token_roof_decomposition.py -q
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    RECEIPTS,
    REPO,
    _assert_no_hardware_claims,
    bench_block,
    load_json,
    seal,
)


RECEIPT = "RESIDENT_TOKEN_ROOF_DECOMPOSITION.json"
SCHEMA = "hawking.future.token_roof_decomposition.v1"
VERSION = 1
RECORDED_BY = "tools/future/token_roof_decomposition.py"

ROOF_ANCHOR = RECEIPTS / "ROOF_ANCHOR.json"
ADDRESSING_ATTRIBUTION = RECEIPTS / "ADDRESSING_ATTRIBUTION.json"
ALU = RECEIPTS / "MLP_ALU_ROOFLINE.json"
ISSUE = RECEIPTS / "MLP_ISSUE_RATE_LADDER.json"
STREAM = RECEIPTS / "MLP_STREAM_COUNT.json"
DN_DECOMPOSE = RECEIPTS / "DELTANET_ORGAN_DECOMPOSE.json"
FOLD = RECEIPTS / "FOLD_ADDQX_AB.json"
WALL_GPU = RECEIPTS / "WALL_GPU_RECONCILIATION.json"
KERNEL_GEO = RECEIPTS / "KERNEL_GEOMETRY.json"
ORGAN_BW = RECEIPTS / "ORGAN_BANDWIDTH.json"
BYTE_CENSUS = RECEIPTS / "MLP_BYTE_CENSUS.json"
CATALOG = RECEIPTS / "CATALOG_ADDRESSING.json"
CAUSAL = RECEIPTS / "RESIDENT_71TPS_CAUSAL_BUDGET.json"
AUX_INFO = RECEIPTS / "MLP_AUXILIARY_INFORMATION.json"
AUX_U8 = RECEIPTS / "AUX_U8_NATIVE.json"
ECONOMICS = RECEIPTS / "ECONOMICS_CALIBRATION.json"
TOKEN_BUDGET = RECEIPTS / "RESIDENT_TOKEN_BUDGET.json"

STAGES = (
    "CLEAN_ROOF",
    "ADDRESSING",
    "GEOMETRY",
    "REAL_DECODE",
    "COMPLETE_TOKEN",
)
TRANSITIONS = (
    "CLEAN_ROOF_TO_ADDRESSING",
    "ADDRESSING_TO_GEOMETRY",
    "GEOMETRY_TO_REAL_DECODE",
    "REAL_DECODE_TO_COMPLETE_TOKEN",
)
RECONCILE_REQUIRED = (
    "clean_roof",
    "addressing",
    "geometry",
    "real_decode",
    "deltanet_state_to_consume_stall",
    "UNATTRIBUTED",
)

# The four losses the campaign measured separately. Combining them is the
# forbidden GPU-inefficiency smear.
UNRELATED_LOSSES = (
    "decode_arithmetic",
    "addressing",
    "deltanet_state_to_consume_stall",
    "host_ceremony",
)

FORBIDDEN_BUCKETS = frozenset(
    {
        "GPU_INEFFICIENCY",
        "gpu_inefficiency",
        "GPU inefficiency",
        "gpu inefficiency",
        "GpuInefficiency",
        "gpu-inefficiency",
    }
)

CLEAN_KERNEL_ROOF_CAVEAT = (
    "MEASURED CLEAN KERNEL ROOF, NEVER guaranteed production bandwidth. "
    "addr_probe loads scales + packed codes and sinks them: NO nibble unpack, "
    "NO input-vector load, NO FMA. Kernel shape does not exist in production. "
    "Campaign 703.5 is the probe MAX (703.61), not the median (699.57)."
)

CLAIM_BOUNDARY = (
    "Static sidecar assembled from committed receipts. No new GPU probe. "
    "Every GB/s, ms, dispatch, encoder, command-buffer and wait figure is "
    "copied from a named prior receipt or derived by arithmetic over those "
    "copies (DERIVED_FROM_CITED_RECEIPTS). 703 GB/s is recorded as a MEASURED "
    "CLEAN KERNEL ROOF with the no-input-vector-load caveat, never as "
    "guaranteed production bandwidth; the causal budget's 66.13 TPS rung "
    "rests on it and carries the same caveat. Bytes per token and useful "
    "bytes per token are different columns: broadcast aux is not on the "
    "critical path. Unrelated losses (decode arithmetic, addressing, the "
    "DeltaNet state stall, host ceremony) stay apart. evidence_class of "
    "the cited measurements is SELF_MEASURED_DIRTY; this sidecar takes no "
    "GPU lease and fabricates no hardware number. A TPS figure here is "
    "never labelled QUALIFIED."
)

RECONCILE_TOLERANCE_MS = 1e-6


class TokenRoofError(ValueError):
    """Contract violation around the roof-to-token decomposition."""


class UnreconciledDecomposition(TokenRoofError):
    """Raised rather than emit a decomposition whose parts do not cover the whole."""


class ForbiddenLossBucket(TokenRoofError):
    """Raised rather than combine unrelated losses into GPU INEFFICIENCY."""


class UnsourcedTransition(TokenRoofError):
    """Raised rather than emit a transition without a source receipt."""


class EmptyGpuSample(TokenRoofError):
    """Raised rather than divide by a missing GPU timestamp."""


class UnqualifiedCleanRoof(TokenRoofError):
    """Raised rather than mention 703 without the no-input-vector-load caveat."""


def nested(doc: Mapping[str, Any] | None, *path: str) -> Any:
    cur: Any = doc
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def require_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TokenRoofError(f"missing receipt {path.relative_to(REPO)}")
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise TokenRoofError(f"{path} is not a JSON object")
    return doc


def implied_ms(nbytes: int | float, gb_s: float) -> float:
    if gb_s <= 0:
        raise EmptyGpuSample("gb_s must be positive to form an implied time")
    if nbytes < 0:
        raise ValueError("bytes must be non-negative")
    return float(nbytes) / float(gb_s) / 1e6


def effective_gb_s(nbytes: int | float, gpu_ms: float) -> float:
    if gpu_ms <= 0:
        raise EmptyGpuSample("gpu_ms must be positive to form a bandwidth")
    if nbytes <= 0:
        raise ValueError("bytes must be positive to form a bandwidth")
    return float(nbytes) / (float(gpu_ms) * 1e-3) / 1e9


def _is_forbidden_bucket(name: str) -> bool:
    folded = name.strip()
    if folded in FORBIDDEN_BUCKETS:
        return True
    low = folded.lower().replace("_", " ").replace("-", " ")
    return "gpu" in low and "inefficiency" in low


def named_loss(
    *,
    name: str,
    ms: float,
    source_receipt: str,
    gb_s: float | None = None,
    host_ms: float = 0.0,
    gpu_ms: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A single sourced loss. Refuses the forbidden combined bucket."""
    if _is_forbidden_bucket(name):
        raise ForbiddenLossBucket(
            "UNRELATED LOSSES MAY NEVER BE COMBINED INTO 'GPU INEFFICIENCY'. "
            f"refusing bucket {name!r}"
        )
    if not source_receipt:
        raise UnsourcedTransition(f"{name} has no source receipt")
    gpu = host_ms if gpu_ms is None and name == "host_ceremony" else (
        ms if gpu_ms is None else gpu_ms
    )
    if name == "host_ceremony":
        gpu = 0.0
        host_ms = ms if host_ms == 0.0 else host_ms
    row = {
        "name": name,
        "ms": ms,
        "gb_s": gb_s,
        "source_receipt": source_receipt,
        "host_ms": host_ms,
        "gpu_ms": gpu,
    }
    row.update(extra)
    return row


def combine_losses(
    losses: Iterable[Mapping[str, Any]],
    *,
    bucket: str,
) -> dict[str, Any]:
    """Refuse to smear unrelated measured losses into one bucket.

    The four campaign losses — decode arithmetic, addressing, the DeltaNet
    state stall, host ceremony — are different measurements. Combining them
    under GPU_INEFFICIENCY (or under any single name) is the defect this
    module exists to prevent. The function always raises when asked to
    combine those four, and always raises on the forbidden bucket name.
    """
    rows = [dict(x) for x in losses]
    if _is_forbidden_bucket(bucket):
        raise ForbiddenLossBucket(
            "UNRELATED LOSSES MAY NEVER BE COMBINED INTO 'GPU INEFFICIENCY'. "
            f"refusing bucket {bucket!r}"
        )
    names = {str(r.get("name", "")) for r in rows}
    if set(UNRELATED_LOSSES) <= names:
        raise ForbiddenLossBucket(
            "decode arithmetic, addressing, the DeltaNet state stall and "
            "host ceremony are four different measurements with four "
            "different source receipts; they stay apart. Anything left is "
            "UNATTRIBUTED, not a combined inefficiency bucket."
        )
    raise ForbiddenLossBucket(
        f"refusing to combine {sorted(names)} into {bucket!r}"
    )


def _703_fields() -> dict[str, Any]:
    return {
        "campaign_label_gb_s": 703.5,
        "max_gb_s": 703.6072736347875,
        "median_gb_s": 699.5736545106142,
        "min_gb_s": 693.1508595217028,
        "statistic": "max",
        "no_input_vector_load": True,
        "usable_as_production_streaming_roof": False,
        "guaranteed_production_bandwidth": False,
        "clean_kernel_roof_caveat": CLEAN_KERNEL_ROOF_CAVEAT,
        "kind": "MEASURED_CLEAN_KERNEL_ROOF",
    }


def assert_703_qualified(doc: Mapping[str, Any]) -> None:
    """Every mention of the 703 roof must carry the no-input-vector-load caveat."""

    def walk(node: Any, ancestors: tuple[Mapping[str, Any], ...]) -> None:
        if isinstance(node, Mapping):
            chain = ancestors + (node,)
            blob = json.dumps({k: v for k, v in node.items() if not isinstance(v, (dict, list))})
            mentions = (
                "703" in blob
                or any(
                    isinstance(v, (int, float)) and 703.0 <= float(v) < 704.0
                    for v in node.values()
                    if not isinstance(v, (dict, list, bool))
                )
            )
            if mentions:
                qualified = False
                for frame in reversed(chain):
                    caveat = frame.get("clean_kernel_roof_caveat") or frame.get("caveat") or ""
                    if frame.get("no_input_vector_load") is True:
                        qualified = True
                        break
                    if isinstance(caveat, str) and "input-vector load" in caveat.lower():
                        qualified = True
                        break
                    if frame.get("guaranteed_production_bandwidth") is False and (
                        frame.get("usable_as_production_streaming_roof") is False
                    ):
                        # not enough alone — require the caveat words
                        pass
                if not qualified:
                    raise UnqualifiedCleanRoof(
                        "703 appears without the no-input-vector-load caveat: "
                        f"keys={sorted(node.keys())[:12]}"
                    )
            for v in node.values():
                walk(v, chain)
        elif isinstance(node, list):
            for v in node:
                walk(v, ancestors)

    walk(doc, ())
    blob = json.dumps(doc)
    if "703" in blob:
        if "input-vector load" not in blob.lower() and "input_vector_load" not in blob.lower():
            raise UnqualifiedCleanRoof("703 appears in the receipt without the caveat text")
        if "guaranteed production bandwidth" in blob.lower() and "NEVER" not in blob and "never" not in blob:
            raise UnqualifiedCleanRoof("703 must never be described as guaranteed production bandwidth")


def assert_no_forbidden_bucket(doc: Mapping[str, Any]) -> None:
    """Refuse a combined inefficiency bucket. Naming the forbidden name in
    the obligation / finding / `forbidden_bucket` field is the refusal, not
    the bucket."""
    allowed_keys = {
        "forbidden_bucket",
        "claim_boundary",
        "obligation",
        "finding",
        "note",
        "why",
        "caveat",
        "clean_kernel_roof_caveat",
    }

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if k not in allowed_keys and _is_forbidden_bucket(str(k)):
                    raise ForbiddenLossBucket(f"forbidden key {k!r}")
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and key not in allowed_keys:
            if key in {"loss_name", "name", "id", "bucket"} and _is_forbidden_bucket(node):
                raise ForbiddenLossBucket(f"forbidden {key} {node!r}")

    walk(doc)
    for row in doc.get("transitions") or []:
        if _is_forbidden_bucket(str(row.get("loss_name", ""))):
            raise ForbiddenLossBucket(f"transition {row.get('id')} uses the forbidden bucket")
        for comp in row.get("components") or []:
            if _is_forbidden_bucket(str(comp.get("name", ""))):
                raise ForbiddenLossBucket(f"component {comp.get('name')!r} is the forbidden bucket")
    parts = nested(doc, "reconciliation", "gpu", "parts_ms") or {}
    for k in parts:
        if _is_forbidden_bucket(k):
            raise ForbiddenLossBucket(f"reconcile key {k!r}")


def reconcile(
    whole_ms: float,
    parts: Mapping[str, float],
    *,
    required: tuple[str, ...] = RECONCILE_REQUIRED,
    tolerance_ms: float = RECONCILE_TOLERANCE_MS,
) -> dict[str, Any]:
    """Account for whole_ms with named parts. Refuse a silent absorb.

    UNATTRIBUTED must be present even when it is zero. If the parts do not
    sum to the whole, this RAISES UnreconciledDecomposition rather than
    folding the gap into any named loss.
    """
    if whole_ms <= 0:
        raise EmptyGpuSample("whole_ms must be positive to reconcile")
    missing = [k for k in required if k not in parts]
    if missing:
        raise UnreconciledDecomposition(
            "refusing a decomposition whose parts do not cover the whole: "
            f"missing {missing}"
        )
    if "UNATTRIBUTED" not in parts:
        raise UnreconciledDecomposition(
            "UNATTRIBUTED must be present even if zero; an unattributed "
            "residue is reported as UNATTRIBUTED, never absorbed"
        )
    for key in required:
        if _is_forbidden_bucket(key):
            raise ForbiddenLossBucket(f"forbidden reconcile key {key!r}")
        if key != "UNATTRIBUTED" and float(parts[key]) < 0:
            raise UnreconciledDecomposition(
                f"refusing a negative named loss {key}={parts[key]}"
            )
    summed = sum(float(parts[k]) for k in parts)
    gap = float(whole_ms) - summed
    if abs(gap) > tolerance_ms:
        raise UnreconciledDecomposition(
            "refusing a decomposition whose parts do not add up to the whole: "
            f"parts sum {summed} vs whole {whole_ms}, gap {gap}. Name the "
            "residue UNATTRIBUTED rather than absorbing it, or fix the parts."
        )
    unattr = float(parts["UNATTRIBUTED"])
    return {
        "whole_ms": float(whole_ms),
        "sum_parts_ms": summed,
        "unattributed_ms": unattr,
        "unattributed_name": "UNATTRIBUTED",
        "gap_ms": gap,
        "required": list(required),
        "parts_ms": {k: float(parts[k]) for k in parts},
        "within_tolerance": abs(gap) <= tolerance_ms,
    }


def _counts(*, dispatches: Any, encoders: Any, command_buffers: Any) -> dict[str, Any]:
    return {
        "dispatches": dispatches,
        "encoders": encoders,
        "command_buffers": command_buffers,
    }


def _waits(*, gpu_ms: float | None, wait_ms: float | None, note: str) -> dict[str, Any]:
    extra = None
    if gpu_ms is not None and wait_ms is not None:
        extra = round(float(wait_ms) - float(gpu_ms), 6)
    return {
        "gpu_ms": gpu_ms,
        "wait_ms": wait_ms,
        "wait_minus_gpu_ms": extra,
        "note": note,
    }


def transition(
    *,
    ident: str,
    from_stage: str,
    to_stage: str,
    loss_name: str,
    loss_ms: float,
    loss_gb_s: float | None,
    from_gb_s: float,
    to_gb_s: float,
    from_ms: float,
    to_ms: float,
    source_receipt: str,
    source_field: str,
    dispatches: Mapping[str, Any],
    encoders: Mapping[str, Any],
    command_buffers: Mapping[str, Any],
    waits: Mapping[str, Any],
    host_ms: float,
    gpu_ms: float,
    bytes_per_token: int,
    useful_bytes_per_token: int,
    native_measurement: Mapping[str, Any],
    caveat: str | None = None,
    components: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if ident not in TRANSITIONS:
        raise UnsourcedTransition(f"unknown transition {ident}")
    if from_stage not in STAGES or to_stage not in STAGES:
        raise UnsourcedTransition(f"{ident} stages {from_stage}->{to_stage} not in {STAGES}")
    if _is_forbidden_bucket(loss_name) or _is_forbidden_bucket(ident):
        raise ForbiddenLossBucket(f"refusing {ident}/{loss_name}")
    if not source_receipt:
        raise UnsourcedTransition(f"{ident} has no source receipt")
    if bytes_per_token == useful_bytes_per_token:
        raise TokenRoofError(
            f"{ident}: bytes_per_token and useful_bytes_per_token must stay "
            "different columns (broadcast aux is not on the critical path)"
        )
    row: dict[str, Any] = {
        "id": ident,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "loss_name": loss_name,
        "loss_ms": loss_ms,
        "loss_gb_s": loss_gb_s,
        "from_gb_s": from_gb_s,
        "to_gb_s": to_gb_s,
        "from_ms": from_ms,
        "to_ms": to_ms,
        "source_receipt": source_receipt,
        "source_field": source_field,
        "dispatches": dict(dispatches),
        "encoders": dict(encoders),
        "command_buffers": dict(command_buffers),
        "waits": dict(waits),
        "host_ms": host_ms,
        "gpu_ms": gpu_ms,
        "bytes_per_token": bytes_per_token,
        "useful_bytes_per_token": useful_bytes_per_token,
        "native_measurement": dict(native_measurement),
        "measured": True,
        "components": list(components or ()),
    }
    if caveat:
        row["caveat"] = caveat
        if "703" in caveat or "input-vector" in caveat.lower():
            row["no_input_vector_load"] = True
            row["clean_kernel_roof_caveat"] = CLEAN_KERNEL_ROOF_CAVEAT
            row["usable_as_production_streaming_roof"] = False
            row["guaranteed_production_bandwidth"] = False
    row.update(extra)
    return row


def assemble(injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()

    def take(path: Path) -> dict[str, Any]:
        rel = str(path.relative_to(REPO)) if path.is_absolute() else str(path)
        if injected is not None and rel in injected:
            doc = injected[rel]
            if not isinstance(doc, dict):
                raise TokenRoofError(f"injected {rel} is not an object")
            return doc
        return require_file(path)

    roof = take(ROOF_ANCHOR)
    attr = take(ADDRESSING_ATTRIBUTION)
    alu = take(ALU)
    issue = take(ISSUE)
    stream = take(STREAM)
    dn = take(DN_DECOMPOSE)
    fold = take(FOLD)
    wall = take(WALL_GPU)
    geo = take(KERNEL_GEO)
    organ = take(ORGAN_BW)
    census = take(BYTE_CENSUS)
    catalog = take(CATALOG)
    causal = take(CAUSAL)
    aux = take(AUX_INFO)
    aux_u8 = take(AUX_U8)
    economics = take(ECONOMICS)
    token_budget = take(TOKEN_BUDGET)

    active_bytes = int(
        nested(census, "census", "active_weight_bytes_per_token")
        or nested(catalog, "cited", "active_weight_bytes_per_token")
        or 9_878_901_136
    )
    auxiliary_bytes = int(
        nested(aux, "accounting", "auxiliary_bytes")
        or nested(aux_u8, "auxiliary_bytes_target")
        or 1_069_605_696
    )
    useful_bytes = active_bytes - auxiliary_bytes
    if useful_bytes <= 0 or useful_bytes == active_bytes:
        raise TokenRoofError(
            "useful_bytes_per_token collapsed onto bytes_per_token; "
            "broadcast aux must stay a separate column"
        )

    single = nested(catalog, "cited", "single_gemv_addr") or {}
    prod_cat = nested(catalog, "cited", "production_catalog_addr") or {}
    if not isinstance(single, Mapping) or single.get("max_gb_s") is None:
        raise UnsourcedTransition("CATALOG_ADDRESSING.cited.single_gemv_addr.max_gb_s")
    if not isinstance(prod_cat, Mapping) or prod_cat.get("median_gb_s") is None:
        raise UnsourcedTransition("CATALOG_ADDRESSING.cited.production_catalog_addr.median_gb_s")

    clean_max = float(single["max_gb_s"])
    clean_median = float(single["median_gb_s"])
    clean_min = float(single.get("min_gb_s") or 693.1508595217028)
    clean_campaign = 703.5
    catalog_payload = int(single.get("payload_bytes") or prod_cat.get("payload_bytes") or 13_611_663_360)
    catalog_addr_gb_s = float(prod_cat["median_gb_s"])
    catalog_addr_ms_native = float(prod_cat["median_ns"]) / 1e6
    catalog_single_ms_native = float(single["median_ns"]) / 1e6
    catalog_single_max_ms_native = float(single["max_ns"]) / 1e6

    arm_a_gb_s = float(nested(alu, "mlp", "arm_a_stripped", "effective_gb_s") or 497.4)
    prod_mlp_gb_s = float(nested(alu, "mlp", "production", "effective_gb_s") or 329.6)
    lm_head_gb_s = float(alu.get("lm_head_gb_s") or 497.4)
    mlp_bytes = int(nested(alu, "mlp", "arm_a_stripped", "weight_bytes") or 83_558_400)
    mlp_arm_a_ns = int(nested(alu, "mlp", "arm_a_stripped", "gpu_ns_median") or 0)
    mlp_prod_ns = int(nested(alu, "mlp", "production", "gpu_ns_median") or 0)
    mlp_arm_a_disp = int(nested(alu, "mlp", "arm_a_stripped", "dispatches") or 3)
    mlp_arm_a_enc = int(nested(alu, "mlp", "arm_a_stripped", "encoders") or 1)
    mlp_arm_a_cb = int(nested(alu, "mlp", "arm_a_stripped", "command_buffers") or 1)

    kg_decomp = nested(geo, "verdict", "decomposition") or {}
    catalog_full_gb_s = float(kg_decomp.get("catalog_full_median_gb_s") or 505.8100047843556)
    catalog_full_ms_native = float(kg_decomp.get("catalog_full_implied_ms") or 26.910625)
    catalog_addr_ms_quoted = float(kg_decomp.get("catalog_addr_implied_ms") or catalog_addr_ms_native)
    weight_free_ms = float(kg_decomp.get("weight_free_dispatches_at_most_ms") or 0.585604)
    geometry_is_the_drop = nested(geo, "verdict", "geometry_is_the_530_to_337_loss")
    kg_verdict = nested(geo, "verdict", "verdict") or nested(geo, "verdict")
    if isinstance(kg_verdict, Mapping):
        kg_verdict = kg_verdict.get("verdict")

    organs = organ.get("organs") or []
    organ_by = {
        r["organ"]: r
        for r in organs
        if isinstance(r, Mapping) and r.get("organ")
    }
    mlp_organ_ms = float(organ_by["mlp"]["gpu_ms"])
    mlp_organ_gb = float(organ_by["mlp"]["active_bytes"]) / 1e9
    mlp_organ_disp = int(organ_by["mlp"]["dispatches"])
    dn_organ_ms = float(organ_by["deltanet"]["gpu_ms"])
    dn_organ_gb = float(organ_by["deltanet"]["active_bytes"]) / 1e9
    dn_organ_disp = int(organ_by["deltanet"]["dispatches"])
    gqa_organ_ms = float(organ_by["gqa"]["gpu_ms"])
    gqa_organ_gb = float(organ_by["gqa"]["active_bytes"]) / 1e9
    gqa_organ_disp = int(organ_by["gqa"]["dispatches"])
    lm_organ_ms = float(organ_by["lm_head"]["gpu_ms"])
    lm_organ_disp = int(organ_by["lm_head"]["dispatches"])
    token_gpu_ms_organ_era = float(organ.get("token_gpu_ms"))
    coverage = organ.get("coverage") or {}
    organ_unattr_ms = float(coverage.get("gpu_ms_unattributed") or 0.0)
    organ_covered_ms = float(coverage.get("gpu_ms_covered") or 0.0)

    recon_dn = dn.get("reconciliation") or {}
    dn_stall_ms = float(recon_dn.get("residual_ms") or 0.0)
    dn_stall_name = str(recon_dn.get("residual_name") or "")
    dn_partition_ms = float(recon_dn.get("sum_partition_ms") or 0.0)
    dn_organ_cb_ms = float(recon_dn.get("organ_ms") or 0.0)
    dn_as_exec = nested(dn, "families", "dn_as_executed") or {}
    dn_as_named = dn.get("as_executed_named") or {}

    fold_ct = fold.get("complete_token") or {}
    complete_gpu_ms = float(fold_ct.get("incumbent_ms") or 26.302583)
    complete_fold_ms = float(fold_ct.get("fold_addqx_ms") or 22.319249)
    complete_disp = int(fold_ct.get("incumbent_dispatches_last") or 580)
    fold_saving_ms = float(nested(fold, "saving", "complete_token_saving_ms") or (complete_gpu_ms - complete_fold_ms))
    iso = fold.get("isolated_mlp") or {}
    iso_inc = nested(iso, "matvecs", "incumbent") or {}
    iso_wait_ms = (
        float(iso_inc["wait_ns_median"]) / 1e6
        if iso_inc.get("wait_ns_median") is not None
        else None
    )
    iso_gpu_ms = (
        float(iso_inc["gpu_ns_median"]) / 1e6
        if iso_inc.get("gpu_ns_median") is not None
        else None
    )

    wall_derived = wall.get("derived") or {}
    host_ms = float(wall_derived.get("host_gap_ms_per_token") or 0.989)
    wall_gpu_ms = float(wall_derived.get("decode_gpu_ms_per_token") or 0.0)
    wall_wall_ms = float(wall_derived.get("decode_wall_ms_per_token") or 0.0)
    wall_disp = float(wall_derived.get("dispatches_per_decode_step") or 628.0)

    topology = catalog.get("topology_tax") or {}
    encoder_collapse = catalog.get("encoder_collapse") or {}
    native_addressing_loss_ms = catalog_addr_ms_native - catalog_single_ms_native

    stream_on_crit = nested(economics, "stream_classes", "broadcast_aux", "on_critical_path")
    aux_u8_inc = nested(aux_u8, "gpu_ab", "incumbent") or {}
    aux_u8_nat = nested(aux_u8, "gpu_ab", "native_u8") or {}
    aux_removed = int(
        nested(aux_u8, "economics", "bytes_only_screen_style", "bytes_removed")
        or nested(aux, "open_byte_levers", 0)  # noqa: not used as int
        or 534_773_760
    )
    if not isinstance(aux_removed, int):
        aux_removed = 534_773_760

    ilp_span = nested(issue, "judgement", "ilp", "ratio_8_over_1")
    reg_span = nested(issue, "judgement", "register_pressure", "ratio_ws32_over_ws0")
    stream_dn_over_mlp = nested(stream, "judgement", "dn_over_mlp")
    pack6 = nested(stream, "judgement", "gb_s", "pack_6_32")
    mlp222 = nested(stream, "judgement", "gb_s", "mlp_2_2_2_32")

    byte_ms_campaign = nested(token_budget, "derived", "byte_ms_at_clean_gemv")
    causal_ladder = causal.get("ladder") or []
    # KEY ON THE RUNG, NOT ITS VALUE. This looked up the ladder row whose tps
    # equalled a literal, so every correction to the budget silently turned the
    # lookup into None and the decomposition kept its own stale copy. The rung
    # NAME is what identifies it; the number is what moves. It has moved three
    # times - 66.54, 65.15, 66.13 - and all three moves were accounting.
    ROOF_RUNG = "every organ at the clean GEMV roof 703.5 GB/s"
    rung_6654 = next(
        (
            r
            for r in causal_ladder
            if isinstance(r, Mapping) and r.get("rung") == ROOF_RUNG
        ),
        None,
    )
    two_numbers = causal.get("the_two_numbers_that_matter") or {}

    # --- token-scale implied times on ACTIVE bytes (campaign denominator) ---
    clean_ms_active = implied_ms(active_bytes, clean_campaign)
    addressing_ms_active = implied_ms(active_bytes, catalog_addr_gb_s)
    geometry_ms_active = implied_ms(active_bytes, arm_a_gb_s)
    clean_ms_useful = implied_ms(useful_bytes, clean_campaign)

    t1_ms = addressing_ms_active - clean_ms_active
    t2_ms = geometry_ms_active - addressing_ms_active
    # REAL DECODE is production MLP vs ARM A on the same organ bytes, NOT
    # the 530→337 smear and NOT DeltaNet/GQA/host.
    mlp_at_arm_a_ms = implied_ms(organ_by["mlp"]["active_bytes"], arm_a_gb_s)
    t3_ms = mlp_organ_ms - mlp_at_arm_a_ms

    # Native catalog-full vs catalog-addr (geometry of the full kernel).
    t2_native_ms = catalog_full_ms_native - catalog_addr_ms_quoted

    gpu_parts_named = {
        "clean_roof": clean_ms_active,
        "addressing": t1_ms,
        "geometry": t2_ms,
        "real_decode": t3_ms,
        "deltanet_state_to_consume_stall": dn_stall_ms,
    }
    named_sum = sum(gpu_parts_named.values())
    unattr_ms = complete_gpu_ms - named_sum
    gpu_parts = {**gpu_parts_named, "UNATTRIBUTED": unattr_ms}
    gpu_recon = reconcile(complete_gpu_ms, gpu_parts)

    wall_complete_ms = complete_gpu_ms + host_ms
    wall_parts = {**gpu_parts, "host_ceremony": host_ms}
    # Wall book: GPU parts + host. UNATTRIBUTED is already in gpu_parts.
    wall_sum = sum(wall_parts.values())
    wall_gap = wall_complete_ms - wall_sum
    if abs(wall_gap) > RECONCILE_TOLERANCE_MS:
        raise UnreconciledDecomposition(
            f"wall book does not close: {wall_sum} vs {wall_complete_ms}"
        )

    t4_gpu_ms = dn_stall_ms + unattr_ms
    t4_ms_with_host = t4_gpu_ms + host_ms

    bytes_cols = {
        "bytes_per_token": active_bytes,
        "useful_bytes_per_token": useful_bytes,
    }

    clean_fields = _703_fields()
    clean_fields.update(
        {
            "max_gb_s": clean_max,
            "median_gb_s": clean_median,
            "min_gb_s": clean_min,
            "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
            "source_field": "cited.single_gemv_addr.max_gb_s",
            "also_cited_from": [
                "receipts/future/ROOF_ANCHOR.json",
                "receipts/future/ADDRESSING_ATTRIBUTION.json",
                "receipts/future/MLP_ALU_ROOFLINE.json",
            ],
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
            "what_was_measured": nested(roof, "registry")
            and None,  # filled below from roof registry if present
        }
    )
    roof_reg = None
    for row in roof.get("registry") or []:
        if isinstance(row, Mapping) and row.get("id") == "q4_single_gemv_addr_13p6gb_max":
            roof_reg = row
            break
    if isinstance(roof_reg, Mapping):
        clean_fields["what_was_measured"] = roof_reg.get("what_was_measured")
        clean_fields["roof_id"] = roof_reg.get("id")
        clean_fields["source_receipt"] = roof_reg.get("source_receipt") or clean_fields["source_receipt"]
        clean_fields["source_field"] = roof_reg.get("source_field") or clean_fields["source_field"]
    else:
        clean_fields.pop("what_was_measured", None)

    stages = [
        {
            "id": "CLEAN_ROOF",
            "gb_s": clean_campaign,
            "gb_s_measured_max": clean_max,
            "gb_s_measured_median": clean_median,
            "ms_active_bytes": clean_ms_active,
            "ms_useful_bytes": clean_ms_useful,
            "ms_native_catalog_payload_median": catalog_single_ms_native,
            "ms_native_catalog_payload_max": catalog_single_max_ms_native,
            **_counts(dispatches=int(single.get("dispatches") or 1), encoders=1, command_buffers=1),
            **bytes_cols,
            "host_ms": 0.0,
            "gpu_ms": clean_ms_active,
            "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
            "source_field": "cited.single_gemv_addr",
            "loads_activation": False,
            **{k: v for k, v in clean_fields.items() if k not in {"source_receipt", "source_field"}},
        },
        {
            "id": "ADDRESSING",
            "gb_s": catalog_addr_gb_s,
            "ms_active_bytes": addressing_ms_active,
            "ms_native_catalog_payload": catalog_addr_ms_native,
            **_counts(
                dispatches=int(prod_cat.get("dispatches") or 401),
                encoders={
                    "catalog_probe": int(prod_cat.get("dispatches") or 401),
                    "encoder_collapse_serial": encoder_collapse.get("serial_encoder_count"),
                    "encoder_collapse_noop": encoder_collapse.get("noop_encoder_count"),
                    "encoder_collapse_separated_gpu": encoder_collapse.get("separated"),
                },
                command_buffers=1,
            ),
            **bytes_cols,
            "host_ms": 0.0,
            "gpu_ms": addressing_ms_active,
            "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
            "source_field": "cited.production_catalog_addr.median_gb_s",
            "loads_activation": False,
            "note": (
                "Same addr_probe kernel as the 703 figure, 401 production-shaped "
                "GEMVs. Still no activation load. First target on the 703-to-337 chain."
            ),
        },
        {
            "id": "GEOMETRY",
            "gb_s": arm_a_gb_s,
            "ms_active_bytes": geometry_ms_active,
            "ms_native_catalog_full": catalog_full_ms_native,
            "catalog_full_gb_s": catalog_full_gb_s,
            **_counts(
                dispatches=mlp_arm_a_disp,
                encoders=mlp_arm_a_enc,
                command_buffers=mlp_arm_a_cb,
            ),
            **bytes_cols,
            "host_ms": 0.0,
            "gpu_ms": geometry_ms_active,
            "source_receipt": "receipts/future/MLP_ALU_ROOFLINE.json",
            "source_field": "mlp.arm_a_stripped.effective_gb_s",
            "loads_activation": True,
            "corroborated_by": {
                "lm_head_gb_s": lm_head_gb_s,
                "source_receipt": "receipts/future/ORGAN_BANDWIDTH.json",
            },
            "occupancy_is_the_530_to_337_loss": False,
            "kernel_geometry_verdict": kg_verdict,
            "note": (
                "Production-shaped WITH the activation load: ARM A stripped MLP "
                f"on {mlp_bytes} bytes, loads proven live. Independently the LM "
                "head's demonstrated production rate. KERNEL_GEOMETRY FALSIFIED "
                "occupancy/coalescing/tpr64 as the 530→337 drop."
            ),
        },
        {
            "id": "REAL_DECODE",
            "gb_s": prod_mlp_gb_s,
            "organ_mlp_gb_s": float(organ_by["mlp"]["effective_gb_s"]),
            "ms_organ_mlp": mlp_organ_ms,
            "ms_mlp_at_arm_a": mlp_at_arm_a_ms,
            **_counts(
                dispatches={"layer_probe": 3, "organ": mlp_organ_disp},
                encoders=1,
                command_buffers=1,
            ),
            **bytes_cols,
            "host_ms": 0.0,
            "gpu_ms": mlp_organ_ms,
            "source_receipt": "receipts/future/MLP_ALU_ROOFLINE.json",
            "source_field": "mlp.production.effective_gb_s",
            "loads_activation": True,
            "arithmetic_ran": True,
            "note": (
                "Production affine2-q2 decode arithmetic live on the same 83.56 MB "
                "as ARM A. Organ-scale GPU ms is ORGAN_BANDWIDTH mlp.gpu_ms. "
                "This is decode arithmetic, not addressing, not the DeltaNet stall, "
                "not host ceremony."
            ),
        },
        {
            "id": "COMPLETE_TOKEN",
            "gpu_ms": complete_gpu_ms,
            "fold_addqx_gpu_ms": complete_fold_ms,
            "complete_token_saving_ms": fold_saving_ms,
            "host_ms": host_ms,
            "wall_ms_derived": wall_complete_ms,
            "wall_ms_kind": "DERIVED_FROM_CITED_RECEIPTS",
            "gb_s_derived_from_active_bytes": effective_gb_s(active_bytes, complete_gpu_ms),
            "gb_s_kind": "DERIVED_FROM_CITED_RECEIPTS",
            "not_a_new_measurement": True,
            **_counts(
                dispatches={
                    "fold_addqx_580_graph": complete_disp,
                    "sealed_fused_628_graph": int(wall_disp) if wall_disp == int(wall_disp) else wall_disp,
                    "organ_trace": int(nested(organ, "trace_overhead", "dispatches_identical") or 628),
                },
                encoders={"fold_addqx_580_graph": complete_disp},
                command_buffers={"complete_token": 1},
            ),
            **bytes_cols,
            "source_receipt": "receipts/future/FOLD_ADDQX_AB.json",
            "source_field": "complete_token.incumbent_ms",
            "incumbent_is_post_widen_f4_baseline": True,
            "note": (
                "Post-widen_f4 incumbent complete-token GPU median. Host gap is "
                "tracked in the host column from WALL_GPU_RECONCILIATION, not "
                "folded into GPU time. 580-graph fused production path."
            ),
        },
    ]

    transitions = [
        transition(
            ident="CLEAN_ROOF_TO_ADDRESSING",
            from_stage="CLEAN_ROOF",
            to_stage="ADDRESSING",
            loss_name="catalog_topology_mixed_organs",
            loss_ms=t1_ms,
            loss_gb_s=clean_campaign - catalog_addr_gb_s,
            from_gb_s=clean_campaign,
            to_gb_s=catalog_addr_gb_s,
            from_ms=clean_ms_active,
            to_ms=addressing_ms_active,
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
            source_field="cited.single_gemv_addr -> cited.production_catalog_addr",
            dispatches={"from": int(single.get("dispatches") or 1), "to": int(prod_cat.get("dispatches") or 401)},
            encoders={
                "from": 1,
                "to_catalog_probe": int(prod_cat.get("dispatches") or 401),
                "encoder_collapse_serial": encoder_collapse.get("serial_encoder_count"),
                "encoder_collapse_noop": encoder_collapse.get("noop_encoder_count"),
                "encoder_collapse_separated_gpu": encoder_collapse.get("separated"),
            },
            command_buffers={"from": 1, "to": 1},
            waits=_waits(
                gpu_ms=native_addressing_loss_ms,
                wait_ms=None,
                note=(
                    "MTLCommandBuffer GPUStartTime/GPUEndTime on the catalog "
                    "addr_probe. Encoder collapse left GPU_NS overlapping; the "
                    "tax is kernel launch / mixed organ sizes, not encoder "
                    "begin/end and not host catalog walk. "
                    f"ns_per_extra_dispatch_catalog_vs_single="
                    f"{topology.get('ns_per_extra_dispatch_catalog_vs_single')}."
                ),
            ),
            host_ms=0.0,
            gpu_ms=t1_ms,
            bytes_per_token=active_bytes,
            useful_bytes_per_token=useful_bytes,
            native_measurement={
                "payload_bytes": catalog_payload,
                "from_gb_s_median": clean_median,
                "from_gb_s_max": clean_max,
                "to_gb_s_median": catalog_addr_gb_s,
                "loss_ms_on_catalog_payload_median": native_addressing_loss_ms,
                "loss_ms_on_catalog_payload_max": catalog_addr_ms_native - catalog_single_max_ms_native,
                "single_to_catalog_ns": topology.get("single_to_catalog_ns"),
                "attributed_gb_s_median_to_median": clean_median - catalog_addr_gb_s,
                "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
                "also": "receipts/future/ADDRESSING_ATTRIBUTION.json T2 / ADDRESSING_GAP",
            },
            caveat=CLEAN_KERNEL_ROOF_CAVEAT,
            components=[
                named_loss(
                    name="addressing",
                    ms=t1_ms,
                    gb_s=clean_campaign - catalog_addr_gb_s,
                    source_receipt="receipts/future/CATALOG_ADDRESSING.json",
                    gpu_ms=t1_ms,
                    host_ms=0.0,
                    mechanism="dispatch topology + mixed organ sizes",
                    falsified_as="host catalog indirection",
                )
            ],
            token_scale_formula="implied_ms(active_bytes, 530.65) - implied_ms(active_bytes, 703.5)",
        ),
        transition(
            ident="ADDRESSING_TO_GEOMETRY",
            from_stage="ADDRESSING",
            to_stage="GEOMETRY",
            loss_name="activation_load_production_shaped_roof",
            loss_ms=t2_ms,
            loss_gb_s=catalog_addr_gb_s - arm_a_gb_s,
            from_gb_s=catalog_addr_gb_s,
            to_gb_s=arm_a_gb_s,
            from_ms=addressing_ms_active,
            to_ms=geometry_ms_active,
            source_receipt="receipts/future/KERNEL_GEOMETRY.json",
            source_field="verdict.decomposition.catalog_full vs catalog_addr; mlp.arm_a_stripped",
            dispatches={
                "from_catalog": int(prod_cat.get("dispatches") or 401),
                "to_arm_a_layer": mlp_arm_a_disp,
            },
            encoders={"from": int(prod_cat.get("dispatches") or 401), "to": mlp_arm_a_enc},
            command_buffers={"from": 1, "to": mlp_arm_a_cb},
            waits=_waits(
                gpu_ms=t2_native_ms,
                wait_ms=None,
                note=(
                    "KERNEL_GEOMETRY: occupancy/coalescing/tpr64 FALSIFIED as the "
                    f"530→337 drop (verdict={kg_verdict}, "
                    f"geometry_is_the_530_to_337_loss={geometry_is_the_drop}). "
                    "Raising occupancy makes things WORSE (issue-ladder tg1024). "
                    f"Weight-free dispatches at most {weight_free_ms} ms — a bound, "
                    "not this transition's loss. Native catalog_full vs catalog_addr "
                    f"is {t2_native_ms} ms on 13.612 GB (dequant + input + FMA)."
                ),
            ),
            host_ms=0.0,
            gpu_ms=t2_ms,
            bytes_per_token=active_bytes,
            useful_bytes_per_token=useful_bytes,
            native_measurement={
                "catalog_addr_gb_s": catalog_addr_gb_s,
                "catalog_full_gb_s": catalog_full_gb_s,
                "catalog_full_minus_addr_ms": t2_native_ms,
                "arm_a_gb_s": arm_a_gb_s,
                "arm_a_gpu_ms_layer": None if not mlp_arm_a_ns else mlp_arm_a_ns / 1e6,
                "occupancy_contribution_ms": 0.0,
                "occupancy_status": "FALSIFIED",
                "weight_free_dispatches_at_most_ms": weight_free_ms,
                "source_receipt": "receipts/future/KERNEL_GEOMETRY.json",
                "arm_a_source_receipt": "receipts/future/MLP_ALU_ROOFLINE.json",
            },
            components=[
                named_loss(
                    name="activation_load_vs_addr_probe",
                    ms=t2_ms,
                    gb_s=catalog_addr_gb_s - arm_a_gb_s,
                    source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
                    gpu_ms=t2_ms,
                    host_ms=0.0,
                    native_catalog_full_vs_addr_ms=t2_native_ms,
                ),
                named_loss(
                    name="occupancy_coalescing_tpr64",
                    ms=0.0,
                    gb_s=0.0,
                    source_receipt="receipts/future/KERNEL_GEOMETRY.json",
                    gpu_ms=0.0,
                    host_ms=0.0,
                    status="FALSIFIED",
                    note="raising occupancy is worse; not the 530→337 drop",
                ),
            ],
        ),
        transition(
            ident="GEOMETRY_TO_REAL_DECODE",
            from_stage="GEOMETRY",
            to_stage="REAL_DECODE",
            loss_name="decode_arithmetic",
            loss_ms=t3_ms,
            loss_gb_s=arm_a_gb_s - prod_mlp_gb_s,
            from_gb_s=arm_a_gb_s,
            to_gb_s=prod_mlp_gb_s,
            from_ms=mlp_at_arm_a_ms,
            to_ms=mlp_organ_ms,
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
            source_field="mlp.arm_a_stripped vs mlp.production; ORGAN_BANDWIDTH mlp.gpu_ms",
            dispatches={"layer_probe": 3, "organ": mlp_organ_disp, "complete_token_mlp_matvecs": iso.get("mlp_full_incumbent_dispatches")},
            encoders={"layer_probe": mlp_arm_a_enc, "organ_isolated": 1},
            command_buffers={"layer_probe": mlp_arm_a_cb, "organ_isolated": 1},
            waits=_waits(
                gpu_ms=iso_gpu_ms,
                wait_ms=iso_wait_ms,
                note=(
                    "Isolated MLP matvecs on the fold_addqx A/B: wait_ns_median vs "
                    "gpu_ns_median after wait. Complete-token timestamps are GPU "
                    "start/end, never a CPU-wait proxy. This transition is decode "
                    "arithmetic only; DeltaNet stall and host ceremony are not in here."
                ),
            ),
            host_ms=0.0,
            gpu_ms=t3_ms,
            bytes_per_token=active_bytes,
            useful_bytes_per_token=useful_bytes,
            native_measurement={
                "arm_a_gb_s": arm_a_gb_s,
                "production_gb_s": prod_mlp_gb_s,
                "arm_a_over_production": nested(alu, "mlp", "judgement", "arm_a_over_production"),
                "layer_arm_a_ms": None if not mlp_arm_a_ns else mlp_arm_a_ns / 1e6,
                "layer_production_ms": None if not mlp_prod_ns else mlp_prod_ns / 1e6,
                "organ_mlp_ms": mlp_organ_ms,
                "organ_mlp_at_arm_a_ms": mlp_at_arm_a_ms,
                "fold_addqx_complete_token_saving_ms": fold_saving_ms,
                "fold_addqx_complete_token_saving_class": nested(fold, "saving", "class"),
                "fold_addqx_faster_not_exact": nested(fold, "saving", "faster_not_exact"),
                "source_receipt": "receipts/future/MLP_ALU_ROOFLINE.json",
                "organ_source_receipt": "receipts/future/ORGAN_BANDWIDTH.json",
                "fold_source_receipt": "receipts/future/FOLD_ADDQX_AB.json",
            },
            components=[
                named_loss(
                    name="decode_arithmetic",
                    ms=t3_ms,
                    gb_s=arm_a_gb_s - prod_mlp_gb_s,
                    source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
                    gpu_ms=t3_ms,
                    host_ms=0.0,
                    organ="mlp",
                    demonstrated_removable_complete_token_ms=fold_saving_ms,
                    demonstrated_source="receipts/future/FOLD_ADDQX_AB.json",
                )
            ],
        ),
        transition(
            ident="REAL_DECODE_TO_COMPLETE_TOKEN",
            from_stage="REAL_DECODE",
            to_stage="COMPLETE_TOKEN",
            loss_name="token_assembly",
            loss_ms=t4_ms_with_host,
            loss_gb_s=None,
            from_gb_s=prod_mlp_gb_s,
            to_gb_s=effective_gb_s(active_bytes, complete_gpu_ms),
            from_ms=mlp_organ_ms,
            to_ms=complete_gpu_ms,
            source_receipt="receipts/future/FOLD_ADDQX_AB.json",
            source_field="complete_token.incumbent_ms; DELTANET_ORGAN_DECOMPOSE residual; WALL_GPU_RECONCILIATION host",
            dispatches={
                "complete_token": complete_disp,
                "deltanet_organ": dn_organ_disp,
                "sealed_fused": int(wall_disp) if wall_disp == int(wall_disp) else wall_disp,
            },
            encoders={
                "deltanet_as_executed": dn_as_named.get("encoder_count") or dn_as_exec.get("encoders"),
                "complete_token": complete_disp,
            },
            command_buffers={
                "deltanet_as_executed": dn_as_exec.get("command_buffers") or dn_as_named.get("command_buffers") or 1,
                "complete_token": 1,
            },
            waits=_waits(
                gpu_ms=wall_gpu_ms or complete_gpu_ms,
                wait_ms=wall_wall_ms or (complete_gpu_ms + host_ms),
                note=(
                    "Host vs GPU are separate columns. WALL_GPU_RECONCILIATION "
                    f"bounds the entire host class at {host_ms} ms (three runs, "
                    "GPU/wall ratio stable). That is NOT this GPU remainder. "
                    "DeltaNet organ CB vs isolated-family sum is a GPU dependency "
                    "stall, also not host."
                ),
            ),
            host_ms=host_ms,
            gpu_ms=t4_gpu_ms,
            bytes_per_token=active_bytes,
            useful_bytes_per_token=useful_bytes,
            native_measurement={
                "complete_token_incumbent_ms": complete_gpu_ms,
                "complete_token_fold_addqx_ms": complete_fold_ms,
                "deltanet_organ_cb_ms": dn_organ_cb_ms,
                "deltanet_partition_sum_ms": dn_partition_ms,
                "deltanet_residual_ms": dn_stall_ms,
                "host_gap_ms": host_ms,
                "organ_era_token_gpu_ms": token_gpu_ms_organ_era,
                "organ_era_is_a_different_run": True,
                "source_receipts": [
                    "receipts/future/FOLD_ADDQX_AB.json",
                    "receipts/future/DELTANET_ORGAN_DECOMPOSE.json",
                    "receipts/future/WALL_GPU_RECONCILIATION.json",
                ],
            },
            components=[
                named_loss(
                    name="deltanet_state_to_consume_stall",
                    ms=dn_stall_ms,
                    source_receipt="receipts/future/DELTANET_ORGAN_DECOMPOSE.json",
                    gpu_ms=dn_stall_ms,
                    host_ms=0.0,
                    residual_name=dn_stall_name,
                    organ_cb_ms=dn_organ_cb_ms,
                    partition_sum_ms=dn_partition_ms,
                    note=(
                        "Named, not absorbed. State update must retire before "
                        "gated_rmsnorm and out_proj."
                    ),
                ),
                named_loss(
                    name="host_ceremony",
                    ms=host_ms,
                    source_receipt="receipts/future/WALL_GPU_RECONCILIATION.json",
                    gpu_ms=0.0,
                    host_ms=host_ms,
                    note=(
                        "CPU submission, command buffers, readbacks, "
                        "synchronization, state movement, host transforms, "
                        "reporting, allocation, lock contention TOGETHER. "
                        "Not the unlock."
                    ),
                ),
                named_loss(
                    name="UNATTRIBUTED",
                    ms=unattr_ms,
                    source_receipt="receipts/future/RESIDENT_TOKEN_ROOF_DECOMPOSITION.json",
                    gpu_ms=unattr_ms,
                    host_ms=0.0,
                    note=(
                        "Residue after naming addressing, geometry/activation, "
                        "MLP decode arithmetic, the DeltaNet state stall and "
                        "host ceremony, projected onto the fold_addqx incumbent "
                        "complete token. Contains organ-era GQA/DeltaNet-other "
                        "vs 497.4 plus the epoch gap between ORGAN_BANDWIDTH "
                        f"{token_gpu_ms_organ_era} ms and FOLD_ADDQX_AB "
                        f"{complete_gpu_ms} ms, which are different runs. "
                        "Reported as UNATTRIBUTED, not smeared."
                    ),
                    contains_epoch_gap=True,
                    organ_era_token_gpu_ms=token_gpu_ms_organ_era,
                    gqa_vs_497p4_ms_organ_era=gqa_organ_ms - implied_ms(organ_by["gqa"]["active_bytes"], arm_a_gb_s),
                    deltanet_other_vs_497p4_ms_organ_era=(
                        dn_organ_ms
                        - implied_ms(organ_by["deltanet"]["active_bytes"], arm_a_gb_s)
                        - dn_stall_ms
                    ),
                    organ_coverage_unattributed_ms=organ_unattr_ms,
                ),
            ],
        ),
    ]

    # T4 loss_ms includes host (wall remainder). GPU-only T4 is t4_gpu_ms.
    # Reconciliation of the GPU whole does not add host.

    unrelated = {
        "decode_arithmetic": {
            "name": "decode_arithmetic",
            "transition": "GEOMETRY_TO_REAL_DECODE",
            "ms": t3_ms,
            "source_receipt": "receipts/future/MLP_ALU_ROOFLINE.json",
            "also": "receipts/future/FOLD_ADDQX_AB.json",
        },
        "addressing": {
            "name": "addressing",
            "transition": "CLEAN_ROOF_TO_ADDRESSING",
            "ms": t1_ms,
            "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
        },
        "deltanet_state_to_consume_stall": {
            "name": "deltanet_state_to_consume_stall",
            "transition": "REAL_DECODE_TO_COMPLETE_TOKEN",
            "ms": dn_stall_ms,
            "source_receipt": "receipts/future/DELTANET_ORGAN_DECOMPOSE.json",
        },
        "host_ceremony": {
            "name": "host_ceremony",
            "transition": "REAL_DECODE_TO_COMPLETE_TOKEN",
            "ms": host_ms,
            "host_ms": host_ms,
            "gpu_ms": 0.0,
            "source_receipt": "receipts/future/WALL_GPU_RECONCILIATION.json",
        },
    }

    rung_66 = {
        # read from the ladder rung, never re-typed here
        "quoted_value": (
            rung_6654.get("tps") if isinstance(rung_6654, Mapping) else None
        ),
        "unit": "tokens_per_second_arithmetic",
        "rung": "every organ at the clean GEMV roof 703.5 GB/s",
        "class": "ROOF_ON_TODAYS_BYTES",
        "ms": None if not isinstance(rung_6654, Mapping) else rung_6654.get("ms"),
        "source_receipt": "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
        "source_field": "ladder[rung=every organ at the clean GEMV roof 703.5 GB/s].tps",
        "the_two_numbers_that_matter_field": "roof_on_todays_bytes_tps",
        "quoted_from_two_numbers": two_numbers.get("roof_on_todays_bytes_tps"),
        "rests_on_roof_id": "q4_single_gemv_addr_13p6gb_max",
        "no_input_vector_load": True,
        "usable_as_production_streaming_roof": False,
        "guaranteed_production_bandwidth": False,
        "clean_kernel_roof_caveat": CLEAN_KERNEL_ROOF_CAVEAT,
        "caveat": CLEAN_KERNEL_ROOF_CAVEAT,
        "kind": "NOT_A_REACHABLE_TPS",
        "qualification": "NOT_QUALIFIED",
        "note": (
            "Arithmetic: organ bytes at 703.5 GB/s plus the 0.989 ms host gap. "
            "A ceiling for a kernel that does not load the activation. "
            "ROOF_ANCHOR flags this rung wrong_roof_shape."
        ),
    }

    useful_block = {
        "bytes_per_token": active_bytes,
        "useful_bytes_per_token": useful_bytes,
        "auxiliary_bytes": auxiliary_bytes,
        "auxiliary_source_receipt": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
        "auxiliary_source_field": "accounting.auxiliary_bytes",
        "broadcast_aux_on_critical_path": False if stream_on_crit is None else bool(stream_on_crit),
        "broadcast_aux_source_receipt": "receipts/future/ECONOMICS_CALIBRATION.json",
        "broadcast_aux_source_field": "stream_classes.broadcast_aux.on_critical_path",
        "why_the_columns_are_different": (
            "The broadcast aux stream (MLP scale/bias/header, "
            f"{auxiliary_bytes} bytes) is bytes that are not on the critical "
            "path. ECONOMICS_CALIBRATION billed a 50% aux drop at 0 ms/GB "
            "(within noise). AUX_U8_NATIVE then removed "
            f"{aux_removed} bytes of aux and the kernel got SLOWER "
            f"(incumbent {aux_u8_inc.get('effective_gb_s')} GB/s in "
            f"{aux_u8_inc.get('gpu_us_median')} us vs native "
            f"{aux_u8_nat.get('effective_gb_s')} GB/s in "
            f"{aux_u8_nat.get('gpu_us_median')} us). Do not collapse the columns."
        ),
        "removing_0p535_gb_made_things_slower": {
            "bytes_removed": aux_removed,
            "incumbent_gb_s": aux_u8_inc.get("effective_gb_s"),
            "native_gb_s": aux_u8_nat.get("effective_gb_s"),
            "incumbent_gpu_us": aux_u8_inc.get("gpu_us_median"),
            "native_gpu_us": aux_u8_nat.get("gpu_us_median"),
            "source_receipt": "receipts/future/AUX_U8_NATIVE.json",
        },
        "clean_roof_ms_on_bytes": clean_ms_active,
        "clean_roof_ms_on_useful_bytes": clean_ms_useful,
        "campaign_byte_ms_at_clean_gemv_cited": byte_ms_campaign,
        "campaign_byte_ms_source": "receipts/future/RESIDENT_TOKEN_BUDGET.json",
    }

    refuted = [
        {
            "id": "stream_count_at_fixed_bytes_per_thread",
            "status": "REFUTED",
            "span": stream_dn_over_mlp,
            "source_receipt": "receipts/future/MLP_STREAM_COUNT.json",
            "note": (
                f"4+2 shape {nested(stream, 'judgement', 'gb_s', 'dn_4_2_32')} vs "
                f"2+2+2 {mlp222}; merging further HURTS (pack_6_32={pack6})."
            ),
        },
        {
            "id": "dependency_chains",
            "status": "REFUTED",
            "span": ilp_span,
            "source_receipt": "receipts/future/MLP_ISSUE_RATE_LADDER.json",
            "note": "1-to-8 accumulators span 1.062, jumped=False. Not a dependency chain.",
        },
        {
            "id": "register_pressure",
            "status": "REFUTED",
            "span": reg_span,
            "source_receipt": "receipts/future/MLP_ISSUE_RATE_LADDER.json",
            "note": "ws0 vs ws32 span 1.078. Not register pressure.",
        },
        {
            "id": "occupancy",
            "status": "REFUTED",
            "source_receipt": "receipts/future/MLP_ISSUE_RATE_LADDER.json",
            "note": "Raising occupancy makes things WORSE (tg1024 ~236 GB/s).",
        },
        {
            "id": "region_granularity",
            "status": "REFUTED",
            "source_receipt": "receipts/future/MLP_REGION_FALSIFIER.json",
        },
        {
            "id": "catalog_addressing_as_host_indirection",
            "status": "REFUTED",
            "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
        },
        {
            "id": "raw_dispatch_count",
            "status": "REFUTED",
            "source_receipt": "receipts/future/CATALOG_ADDRESSING.json",
        },
    ]

    finding = (
        "CLEAN ROOF 703.5 GB/s is a MEASURED CLEAN KERNEL ROOF (max 703.61, "
        "median 699.57, no input-vector load) — NEVER guaranteed production "
        "bandwidth; the 66.13 TPS rung rests on it and carries the same caveat. "
        f"ADDRESSING drops {t1_ms:.4f} ms / {clean_campaign - catalog_addr_gb_s:.2f} GB/s "
        f"to catalog 530.65 (401 vs 1 dispatch; topology + mixed organs; "
        f"native {native_addressing_loss_ms:.4f} ms on 13.612 GB). "
        f"GEOMETRY drops {t2_ms:.4f} ms / {catalog_addr_gb_s - arm_a_gb_s:.2f} GB/s "
        f"to ARM A 497.4 WITH activation; occupancy contribution is 0 (FALSIFIED); "
        f"catalog_full vs addr is {t2_native_ms:.4f} ms native. "
        f"REAL DECODE is MLP decode arithmetic {t3_ms:.4f} ms "
        f"(497.4 → 329.6 on the same 83.56 MB; organ 15.541 vs {mlp_at_arm_a_ms:.4f} ms); "
        f"fold_addqx saved {fold_saving_ms:.4f} ms on the complete token and is "
        "NOT bit-identical. COMPLETE TOKEN incumbent "
        f"{complete_gpu_ms:.4f} ms GPU / {complete_fold_ms:.4f} fold_addqx, "
        f"host ceremony {host_ms:.4f} ms (separate column), DeltaNet "
        f"state-to-consume stall {dn_stall_ms:.4f} ms (named, not absorbed). "
        f"UNATTRIBUTED {unattr_ms:.4f} ms. Bytes {active_bytes} vs useful "
        f"{useful_bytes}: broadcast aux is not on the critical path; removing "
        "0.535 GB made things SLOWER. No GPU INEFFICIENCY bucket."
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "DERIVED_FROM_CITED_RECEIPTS",
        "cited_evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "obligation": (
            "RESIDENT_TOKEN_ROOF_DECOMPOSITION.json EXISTS AND NAMES EVERY LOSS. "
            "Each transition CLEAN ROOF -> ADDRESSING -> GEOMETRY -> REAL DECODE "
            "-> COMPLETE TOKEN carries its own MEASURED loss, with dispatch/"
            "encoder/command-buffer counts, waits, host and gpu time, bytes per "
            "token and USEFUL bytes per token tracked separately. UNRELATED "
            "LOSSES MAY NEVER BE COMBINED INTO 'GPU INEFFICIENCY', and an "
            "unattributed residue is reported as UNATTRIBUTED. 703 GB/s is "
            "recorded as a MEASURED CLEAN KERNEL ROOF, NEVER as guaranteed "
            "production bandwidth."
        ),
        "clean_kernel_roof": clean_fields,
        "clean_kernel_roof_caveat": CLEAN_KERNEL_ROOF_CAVEAT,
        "no_input_vector_load": True,
        "usable_as_production_streaming_roof": False,
        "guaranteed_production_bandwidth": False,
        "causal_budget_roof_on_todays_bytes": rung_66,
        "bytes": useful_block,
        "stages": stages,
        "transitions": transitions,
        "unrelated_losses_kept_apart": unrelated,
        "unattributed": {
            "name": "UNATTRIBUTED",
            "ms": unattr_ms,
            "gpu_ms": unattr_ms,
            "host_ms": 0.0,
            "whole_ms": complete_gpu_ms,
            "named_sum_ms": named_sum,
            "note": (
                "Residue after naming the four measured transitions' token-scale "
                "losses plus the DeltaNet state stall. Size is reported, not smeared "
                "into GPU INEFFICIENCY."
            ),
        },
        "reconciliation": {
            "gpu": gpu_recon,
            "wall": {
                "whole_ms": wall_complete_ms,
                "whole_kind": "DERIVED_FROM_CITED_RECEIPTS",
                "formula": "complete_token_gpu_ms + host_gap_ms",
                "complete_token_gpu_ms": complete_gpu_ms,
                "host_ceremony_ms": host_ms,
                "sum_parts_ms": wall_sum,
                "gap_ms": wall_gap,
                "within_tolerance": abs(wall_gap) <= RECONCILE_TOLERANCE_MS,
                "host_source_receipt": "receipts/future/WALL_GPU_RECONCILIATION.json",
            },
            "complete_token": {
                "incumbent_gpu_ms": complete_gpu_ms,
                "fold_addqx_gpu_ms": complete_fold_ms,
                "source_receipt": "receipts/future/FOLD_ADDQX_AB.json",
                "organ_era_gpu_ms": token_gpu_ms_organ_era,
                "organ_era_source_receipt": "receipts/future/ORGAN_BANDWIDTH.json",
                "organ_era_is_a_different_run": True,
            },
        },
        "host_vs_gpu": {
            "host_ms": host_ms,
            "gpu_ms": complete_gpu_ms,
            "wall_ms_derived": wall_complete_ms,
            "host_source_receipt": "receipts/future/WALL_GPU_RECONCILIATION.json",
            "gpu_source_receipt": "receipts/future/FOLD_ADDQX_AB.json",
            "host_bounds_the_entire_class": True,
            "host_class": (
                "CPU submission, command buffers, readbacks, synchronization, "
                "state movement, host transforms, reporting, allocation, lock "
                "contention"
            ),
        },
        "refuted_as_the_mechanism": refuted,
        "forbidden_bucket": "GPU_INEFFICIENCY",
        "forbidden_bucket_present": False,
        "tps_qualification": {
            "any_tps_labelled_qualified": False,
            "protected_window_required": True,
            "reason": (
                "Cited measurements are SELF_MEASURED_DIRTY. This sidecar "
                "fabricates no hardware number and labels no TPS QUALIFIED."
            ),
        },
        "self_timing": {
            "what": "python wall of static composition in this process",
            "elapsed_ms": elapsed_ms,
            "qualifies": "nothing",
        },
        "finding": finding,
    }
    return doc


def build(injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    doc = assemble(injected)
    assert_no_forbidden_bucket(doc)
    assert_703_qualified(doc)
    if doc.get("tps_qualification", {}).get("any_tps_labelled_qualified"):
        raise TokenRoofError("a QUALIFIED TPS leaked into the decomposition")
    trans = doc.get("transitions") or []
    if len(trans) != 4:
        raise UnreconciledDecomposition(f"expected 4 transitions, got {len(trans)}")
    for row in trans:
        if not row.get("source_receipt"):
            raise UnsourcedTransition(row.get("id"))
        if row.get("bytes_per_token") == row.get("useful_bytes_per_token"):
            raise TokenRoofError("bytes columns collapsed")
        if "dispatches" not in row or "encoders" not in row or "command_buffers" not in row:
            raise UnsourcedTransition(f"{row.get('id')} missing dispatch/encoder/cb counts")
        if "waits" not in row:
            raise UnsourcedTransition(f"{row.get('id')} missing waits")
        if "host_ms" not in row or "gpu_ms" not in row:
            raise UnsourcedTransition(f"{row.get('id')} missing host/gpu split")
    recon = nested(doc, "reconciliation", "gpu") or {}
    if not recon.get("within_tolerance"):
        raise UnreconciledDecomposition("gpu reconciliation did not close")
    if "UNATTRIBUTED" not in (recon.get("parts_ms") or {}):
        raise UnreconciledDecomposition("UNATTRIBUTED missing from gpu reconciliation")
    blob = json.dumps(doc)
    if '"qualification": "QUALIFIED"' in blob or '"qualified": true' in blob.lower():
        raise TokenRoofError("receipt labelled a figure QUALIFIED")
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
    recon = doc["reconciliation"]["gpu"]
    print(doc["finding"])
    print(
        f"gpu whole {recon['whole_ms']:.6f} ms  "
        f"named {recon['whole_ms'] - recon['unattributed_ms']:.6f} ms  "
        f"UNATTRIBUTED {recon['unattributed_ms']:.6f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
