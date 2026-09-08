#!/usr/bin/env python3
"""628-graph A/B: production unfused vi-SIMD vs widen_f4 on the real decode path.

DELTANET_ORGAN_DECOMPOSE measured a 0.702 ms layout lever on isolated
gated-delta family CBs (fused-ba 1.581 ms against widen_f4 0.879 ms).
Production still launched unfused vi-SIMD, so that number stayed
DIRTY_DIAGNOSTIC until this A/B ran the candidate on encode_deltanet
and reported the complete token.

    python3 tools/future/deltanet_widen_ab.py --measure --record
    python3 tools/future/deltanet_widen_ab.py --from RAW.json --record
    python3 -m pytest tools/future/test_deltanet_widen_ab.py -q

PARITY IS TOKEN-IDENTICAL OR IT IS NOT PARITY. Argmax agreement on a
sample is refused (DELTANET_MULTISTEP: a candidate held argmax at step
64 while logit relative L2 was 0.13). Fallbacks must be 0.

If the organ-level saving does not reach the complete token, that is
the result: the receipt names where it went.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "DELTANET_WIDEN_AB.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_DELTANET_WIDEN_AB_raw.json"
SCHEMA = "hawking.future.deltanet_widen_ab.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_widen_ab.py"

INCUMBENT_KERNEL = "qwen38_gated_delta_decode_vi_simd"
WIDEN_KERNEL = "qwen38_gated_delta_decode_vi_simd_ba_f4"
SEALED_DISPATCHES = 628
WIDEN_DISPATCHES = 580

# Cited from DELTANET_ORGAN_DECOMPOSE largest_demonstrated_lever.
# Fair cut is fused-ba minus f4, not unfused minus f4.
CITED_FUSED_BA_MS = 1.5806
CITED_WIDEN_F4_MS = 0.879
CITED_SAVING_MS = 0.7016
CITED_UNFUSED_MS = 1.5826
MATERIALITY_MS = 1.0

SEALED_ENV = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}

DEFAULT_ARTIFACT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")
DEFAULT_TOKENIZER = DEFAULT_ARTIFACT / "tokenizer.json"

CLAIM_BOUNDARY = (
    "One-process 628-graph A/B of incumbent unfused vi-SIMD against "
    "widen_f4 on the real encode_deltanet path, sealed-3.14 fusions "
    "(GateUpSwiglu, FUSE_GQA_QKV, FUSE_DN_INPROJ, FUSE_ADD_RMSNORM, "
    "FUSE_BA_DELTA off). GPU time is MTLCommandBuffer "
    "GPUStartTime/GPUEndTime. Complete-token ms is the median of "
    "per-generated-token GPU timestamps, not the isolated organ. "
    "Absolute ms are measured-under-load; the A/B ratio is back-to-back "
    "in the same process. Parity is token-id equality with fallbacks 0; "
    "argmax agreement is not parity. If the cited 0.702 ms organ-level "
    "saving does not appear in the complete token, the receipt names "
    "where it went and does not promote the diagnostic."
)


class WidenAbRefuse(ValueError):
    """Raised rather than emit an A/B receipt that cannot be defended."""


class ArgmaxIsNotParity(WidenAbRefuse):
    """Argmax agreement on a sample is not token-identical parity."""

    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: argmax agreement is not parity. Token-id equality "
            "across runs is required"
            f"{extra}. (DELTANET_MULTISTEP: a candidate held argmax at "
            "step 64 while logit relative L2 was 0.13)"
        )


class MissingArm(WidenAbRefuse):
    """Raised rather than compare an incomplete pair."""


class ProductionDidNotLaunch(WidenAbRefuse):
    """The candidate arm did not dispatch the f4 kernel on encode_deltanet."""


class EmptyGpuSample(WidenAbRefuse):
    """Raised rather than divide by a missing GPU timestamp."""


def ns_to_ms(ns: int | float | None) -> float | None:
    if ns is None:
        return None
    return int(ns) / 1e6


def _as_int_list(values: Sequence[Any] | None) -> list[int] | None:
    if values is None:
        return None
    out: list[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise WidenAbRefuse(f"token id is not an integer: {v!r}")
        out.append(int(v))
    return out


def first_divergence(
    incumbent: Sequence[int], candidate: Sequence[int]
) -> dict[str, Any] | None:
    n = min(len(incumbent), len(candidate))
    for i in range(n):
        if int(incumbent[i]) != int(candidate[i]):
            return {
                "index": i,
                "incumbent": int(incumbent[i]),
                "candidate": int(candidate[i]),
            }
    if len(incumbent) != len(candidate):
        return {
            "index": n,
            "reason": "length",
            "incumbent_len": len(incumbent),
            "candidate_len": len(candidate),
        }
    return None


def report_token_parity(
    *,
    incumbent_token_ids: Sequence[int] | None = None,
    candidate_token_ids: Sequence[int] | None = None,
    fallbacks: int = 0,
    argmax_agreement: float | None = None,
    runs_compared: int | None = None,
) -> dict[str, Any]:
    """Token-id equality with fallbacks 0. Argmax is never the basis.

    Passing only argmax_agreement, even 1.0, raises ArgmaxIsNotParity.
    Matching argmax with mismatched ids is not parity. Matching ids with
    nonzero fallbacks is not parity.
    """
    if incumbent_token_ids is None or candidate_token_ids is None:
        raise ArgmaxIsNotParity(
            "token-id lists are required; argmax agreement on a sample "
            "is not a substitute"
            + (
                f" (argmax_agreement={argmax_agreement})"
                if argmax_agreement is not None
                else ""
            )
        )
    inc = _as_int_list(incumbent_token_ids)
    cand = _as_int_list(candidate_token_ids)
    assert inc is not None and cand is not None
    identical = inc == cand
    div = None if identical else first_divergence(inc, cand)
    fb = int(fallbacks)
    parity = bool(identical and fb == 0)
    return {
        "token_ids_identical": identical,
        "tokens_compared": min(len(inc), len(cand)),
        "incumbent_len": len(inc),
        "candidate_len": len(cand),
        "first_divergence": div,
        "fallbacks": fb,
        "parity": parity,
        "parity_basis": "token_id_equality",
        "argmax_is_not_parity": True,
        "argmax_agreement": argmax_agreement,
        "argmax_agreement_ignored": argmax_agreement is not None,
        "runs_compared": runs_compared,
    }


def kernel_count(histogram: Sequence[Mapping[str, Any]] | None, name: str) -> int:
    if not histogram:
        return 0
    total = 0
    for row in histogram:
        if row.get("kernel") == name:
            total += int(row.get("count") or 0)
    return total


def _median_u64(values: Sequence[int]) -> int | None:
    if not values:
        return None
    s = sorted(int(v) for v in values)
    return s[len(s) // 2]


def _require_gpu_ns(arm: Mapping[str, Any], label: str) -> int:
    ns = arm.get("gpu_ns_median")
    if ns is None:
        raise EmptyGpuSample(f"{label} is missing gpu_ns_median")
    ns_i = int(ns)
    if ns_i <= 0:
        raise EmptyGpuSample(f"{label} gpu_ns_median must be positive, got {ns_i}")
    return ns_i


def locate_saving(
    *,
    organ_unfused_ms: float,
    organ_fused_ba_ms: float,
    organ_f4_ms: float,
    token_incumbent_ms: float,
    token_f4_ms: float,
    cited_saving_ms: float = CITED_SAVING_MS,
    materiality_ms: float = MATERIALITY_MS,
) -> dict[str, Any]:
    """Name whether the organ-level lever reached the complete token.

    Fair organ cut is fused-ba minus f4 (the diagnostic). Incumbent on
    the 628 graph is unfused, so the token delta is unfused-vs-f4; that
    can differ from the fair cut because fusing ba_to_decay alone saves
    no GPU ms.
    """
    organ_fair_ms = organ_fused_ba_ms - organ_f4_ms
    organ_vs_unfused_ms = organ_unfused_ms - organ_f4_ms
    token_saved_ms = token_incumbent_ms - token_f4_ms
    # Reached: complete token kept at least half of the isolated fair cut,
    # same sign. A 0.70 ms organ cut that becomes 0.05 ms on the token did
    # not survive integration.
    reached = (
        organ_fair_ms > 0
        and token_saved_ms > 0
        and token_saved_ms >= 0.5 * organ_fair_ms
    )
    displaced_ms = organ_fair_ms - token_saved_ms
    extra_ms = token_saved_ms - organ_fair_ms
    if reached:
        if extra_ms > 0.1:
            where = (
                "The organ-level layout saving reached the complete token "
                f"({round(token_saved_ms, 4)} ms vs isolated fair cut "
                f"{round(organ_fair_ms, 4)} ms). The extra "
                f"{round(extra_ms, 4)} ms is the 48-launch ba_to_decay fold "
                "on the chained 628-graph token; isolated family CBs do not "
                "pay that encoder tax. Fusing ba_to_decay alone was already "
                "measured as no GPU ms on isolated gated-delta."
            )
        else:
            where = (
                "The organ-level layout saving reached the complete token "
                f"({round(token_saved_ms, 4)} ms of isolated fair cut "
                f"{round(organ_fair_ms, 4)} ms)"
            )
    elif organ_fair_ms <= 0:
        where = (
            "The isolated fair cut did not reproduce: fused-ba "
            f"{round(organ_fused_ba_ms, 4)} ms vs widen_f4 "
            f"{round(organ_f4_ms, 4)} ms. The cited {cited_saving_ms} ms "
            "diagnostic did not replicate even as an isolated family."
        )
    elif token_saved_ms <= 0:
        where = (
            "The isolated fair cut "
            f"{round(organ_fair_ms, 4)} ms did not appear in the complete "
            f"token (incumbent {round(token_incumbent_ms, 4)} ms, widen_f4 "
            f"{round(token_f4_ms, 4)} ms, delta {round(token_saved_ms, 4)} ms). "
            "The layout change moved cost into the rest of the 628-graph "
            "token rather than removing it."
        )
    else:
        where = (
            "The complete token kept only "
            f"{round(token_saved_ms, 4)} ms of the isolated fair cut "
            f"{round(organ_fair_ms, 4)} ms "
            f"(displaced {round(displaced_ms, 4)} ms into the rest of "
            "the token graph)"
        )
    return {
        "cited_organ_saving_ms": cited_saving_ms,
        "isolated_fused_ba_ms": round(organ_fused_ba_ms, 4),
        "isolated_widen_f4_ms": round(organ_f4_ms, 4),
        "isolated_unfused_ms": round(organ_unfused_ms, 4),
        "isolated_fair_cut_ms": round(organ_fair_ms, 4),
        "isolated_unfused_minus_f4_ms": round(organ_vs_unfused_ms, 4),
        "complete_token_incumbent_ms": round(token_incumbent_ms, 4),
        "complete_token_widen_f4_ms": round(token_f4_ms, 4),
        "complete_token_saving_ms": round(token_saved_ms, 4),
        "displaced_ms": round(displaced_ms, 4),
        "reached_the_token": reached,
        "materiality_bar_ms": materiality_ms,
        "clears_materiality": bool(token_saved_ms >= materiality_ms),
        "where": where,
    }


def _run_ids(runs: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    out: list[list[int]] = []
    for run in runs:
        ids = run.get("new_token_ids")
        if not isinstance(ids, list) or not ids:
            raise MissingArm("a decode run is missing new_token_ids")
        out.append(_as_int_list(ids) or [])
    return out


def _all_identical(series: Sequence[Sequence[int]]) -> bool:
    if not series:
        return False
    first = list(series[0])
    return all(list(s) == first for s in series)


def _fallbacks_sum(runs: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for run in runs:
        total += int(run.get("fallbacks") or 0)
    return total


def _require_kernel(runs: Sequence[Mapping[str, Any]], kernel: str, arm: str) -> int:
    counts = []
    for i, run in enumerate(runs):
        n = kernel_count(run.get("kernel_histogram") or [], kernel)
        launched = run.get("launched_gated_delta_kernel")
        if n <= 0 and launched != kernel:
            raise ProductionDidNotLaunch(
                f"{arm} run {i} did not dispatch {kernel} "
                f"(launched_gated_delta_kernel={launched!r}, "
                "histogram had no matching count). Production did not "
                "launch the candidate; this is not a 628-graph A/B of it."
            )
        counts.append(n if n > 0 else 1)
    return min(counts)


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    iso = raw.get("isolated_gated_delta")
    decode = raw.get("decode")
    if not isinstance(iso, Mapping):
        raise MissingArm("raw is missing isolated_gated_delta")
    if not isinstance(decode, Mapping):
        raise MissingArm("raw is missing decode")
    unfused = iso.get("unfused")
    fused_ba = iso.get("fused_ba")
    f4 = iso.get("widen_f4")
    if not isinstance(unfused, Mapping) or not isinstance(fused_ba, Mapping) or not isinstance(f4, Mapping):
        raise MissingArm("isolated_gated_delta must carry unfused, fused_ba, widen_f4")
    unfused_ns = _require_gpu_ns(unfused, "isolated unfused")
    fused_ns = _require_gpu_ns(fused_ba, "isolated fused_ba")
    f4_ns = _require_gpu_ns(f4, "isolated widen_f4")

    inc_runs = decode.get("incumbent")
    f4_runs = decode.get("widen_f4")
    if not isinstance(inc_runs, list) or not inc_runs:
        raise MissingArm("decode.incumbent is empty")
    if not isinstance(f4_runs, list) or not f4_runs:
        raise MissingArm("decode.widen_f4 is empty")
    if len(inc_runs) != len(f4_runs):
        raise MissingArm(
            f"decode arms have different rep counts: incumbent {len(inc_runs)} "
            f"vs widen_f4 {len(f4_runs)}"
        )

    _require_kernel(inc_runs, INCUMBENT_KERNEL, "incumbent")
    _require_kernel(f4_runs, WIDEN_KERNEL, "widen_f4")

    inc_ids = _run_ids(inc_runs)
    f4_ids = _run_ids(f4_runs)
    fallbacks = _fallbacks_sum(inc_runs) + _fallbacks_sum(f4_runs)
    # Across arms AND across reps. A single mismatched rep is not parity.
    all_ids = inc_ids + f4_ids
    if not _all_identical(inc_ids) or not _all_identical(f4_ids):
        # still compare first-of-each for first_divergence, but parity is false
        parity = report_token_parity(
            incumbent_token_ids=inc_ids[0],
            candidate_token_ids=f4_ids[0],
            fallbacks=fallbacks,
            runs_compared=len(inc_runs) + len(f4_runs),
        )
        parity["token_ids_identical"] = False
        parity["parity"] = False
        parity["within_arm_identical"] = {
            "incumbent": _all_identical(inc_ids),
            "widen_f4": _all_identical(f4_ids),
        }
        if parity["first_divergence"] is None:
            parity["first_divergence"] = first_divergence(inc_ids[0], f4_ids[0]) or {
                "reason": "within-arm mismatch"
            }
    else:
        parity = report_token_parity(
            incumbent_token_ids=inc_ids[0],
            candidate_token_ids=f4_ids[0],
            fallbacks=fallbacks,
            runs_compared=len(inc_runs) + len(f4_runs),
        )
        parity["within_arm_identical"] = {"incumbent": True, "widen_f4": True}

    inc_token_ns = decode.get("incumbent_complete_token_gpu_ns_median")
    f4_token_ns = decode.get("widen_f4_complete_token_gpu_ns_median")
    if inc_token_ns is None:
        inc_token_ns = _median_u64(
            [int(r["complete_token_gpu_ns_median"]) for r in inc_runs if r.get("complete_token_gpu_ns_median")]
        )
    if f4_token_ns is None:
        f4_token_ns = _median_u64(
            [int(r["complete_token_gpu_ns_median"]) for r in f4_runs if r.get("complete_token_gpu_ns_median")]
        )
    if not inc_token_ns or not f4_token_ns:
        raise EmptyGpuSample("complete-token gpu_ns_median missing on an arm")

    inc_disp = inc_runs[0].get("complete_token_dispatches_last") or inc_runs[0].get(
        "theoretical_dispatches"
    )
    f4_disp = f4_runs[0].get("complete_token_dispatches_last") or f4_runs[0].get(
        "theoretical_dispatches"
    )

    organ = locate_saving(
        organ_unfused_ms=ns_to_ms(unfused_ns) or 0.0,
        organ_fused_ba_ms=ns_to_ms(fused_ns) or 0.0,
        organ_f4_ms=ns_to_ms(f4_ns) or 0.0,
        token_incumbent_ms=ns_to_ms(int(inc_token_ns)) or 0.0,
        token_f4_ms=ns_to_ms(int(f4_token_ns)) or 0.0,
    )

    isolated_organ = raw.get("isolated_organ") or {}
    return {
        "raw": dict(raw),
        "isolated": {
            "unfused_gpu_ns": unfused_ns,
            "fused_ba_gpu_ns": fused_ns,
            "widen_f4_gpu_ns": f4_ns,
            "unfused_ms": ns_to_ms(unfused_ns),
            "fused_ba_ms": ns_to_ms(fused_ns),
            "widen_f4_ms": ns_to_ms(f4_ns),
            "unfused_dispatches": unfused.get("dispatches"),
            "fused_ba_dispatches": fused_ba.get("dispatches"),
            "widen_f4_dispatches": f4.get("dispatches"),
            "organ": isolated_organ,
        },
        "complete_token": {
            "incumbent_gpu_ns_median": int(inc_token_ns),
            "widen_f4_gpu_ns_median": int(f4_token_ns),
            "incumbent_ms": ns_to_ms(int(inc_token_ns)),
            "widen_f4_ms": ns_to_ms(int(f4_token_ns)),
            "incumbent_reps": decode.get("incumbent_complete_token_gpu_ns_median_reps"),
            "widen_f4_reps": decode.get("widen_f4_complete_token_gpu_ns_median_reps"),
            "incumbent_dispatches_last": inc_disp,
            "widen_f4_dispatches_last": f4_disp,
        },
        "parity": parity,
        "saving": organ,
        "launched": {
            "incumbent_kernel": INCUMBENT_KERNEL,
            "widen_f4_kernel": WIDEN_KERNEL,
            "incumbent_histogram_count": kernel_count(
                inc_runs[0].get("kernel_histogram") or [], INCUMBENT_KERNEL
            ),
            "widen_f4_histogram_count": kernel_count(
                f4_runs[0].get("kernel_histogram") or [], WIDEN_KERNEL
            ),
        },
    }


def _finding(measured: Mapping[str, Any]) -> str:
    s = measured["saving"]
    p = measured["parity"]
    parity_bit = (
        "token-id identical, fallbacks 0"
        if p.get("parity")
        else "NOT token-identical"
        if not p.get("token_ids_identical")
        else f"token ids match but fallbacks={p.get('fallbacks')}"
    )
    return (
        f"628-graph A/B, production decode path, {parity_bit}. "
        f"Complete-token incumbent {s['complete_token_incumbent_ms']} ms vs "
        f"widen_f4 {s['complete_token_widen_f4_ms']} ms "
        f"(saved {s['complete_token_saving_ms']} ms). Isolated fair cut "
        f"fused-ba {s['isolated_fused_ba_ms']} ms vs f4 {s['isolated_widen_f4_ms']} ms "
        f"(saved {s['isolated_fair_cut_ms']} ms; cited {s['cited_organ_saving_ms']} ms). "
        f"{s['where']} Materiality bar {s['materiality_bar_ms']} ms: "
        f"{'clears' if s['clears_materiality'] else 'does not clear'}."
    )


def build(measured: Mapping[str, Any]) -> dict[str, Any]:
    raw = measured["raw"]
    s = measured["saving"]
    p = measured["parity"]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/ascension_qwen38_deltanet_widen_ab.rs; "
            "production encode_deltanet on sealed-3.14; incumbent unfused "
            "vi-SIMD vs HAWKING_QWEN38_DN_STATE=widen_f4 (fused-ba sibling) "
            "back-to-back in one process"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing"),
        "absolute_ms_are_measured_under_load": True,
        "concurrent_load_start": raw.get("concurrent_load_start"),
        "concurrent_load": raw.get("concurrent_load"),
        "session_open_s": raw.get("session_open_s"),
        "reps": raw.get("reps"),
        "warmup": raw.get("warmup"),
        "max_new_tokens": raw.get("max_new_tokens"),
        "dense_w_materialized": raw.get("dense_w_materialized", 0),
        "production_fusions": raw.get("production_fusions"),
        "sealed_env": SEALED_ENV,
        "lever": "HAWKING_QWEN38_DN_STATE=widen_f4",
        "lever_semantics": (
            "launches qwen38_gated_delta_decode_vi_simd_ba_f4 on the real "
            "encode_deltanet path; folds ba_to_decay because the kernel "
            "consumes projected_ba in-register. Does not require "
            "FUSE_BA_DELTA=1. Default production stays unfused vi-SIMD."
        ),
        "cited_diagnostic": {
            "source": "receipts/future/DELTANET_ORGAN_DECOMPOSE.json",
            "fused_ba_ms": CITED_FUSED_BA_MS,
            "widen_f4_ms": CITED_WIDEN_F4_MS,
            "saving_ms": CITED_SAVING_MS,
            "unfused_ms": CITED_UNFUSED_MS,
            "note": "fair cut is fused-ba minus f4; FUSE_BA_DELTA alone saves no GPU ms",
        },
        "isolated_gated_delta": measured["isolated"],
        "complete_token": measured["complete_token"],
        "parity": p,
        "saving": s,
        "launched": measured["launched"],
        "expected_dispatches": {
            "incumbent": SEALED_DISPATCHES,
            "widen_f4": WIDEN_DISPATCHES,
        },
        "finding": _finding(measured),
        "findings": [
            {
                "id": "PRODUCTION_LAUNCHED_WIDEN_F4",
                "what": (
                    f"candidate histogram counted {measured['launched']['widen_f4_histogram_count']} "
                    f"{WIDEN_KERNEL} dispatches; incumbent counted "
                    f"{measured['launched']['incumbent_histogram_count']} "
                    f"{INCUMBENT_KERNEL}"
                ),
                "why_it_matters": (
                    "the 0.702 ms diagnostic was a probe beside production; "
                    "this A/B is the production encode path"
                ),
            },
            {
                "id": "TOKEN_IDENTITY",
                "what": (
                    "token-id identical, fallbacks 0"
                    if p.get("parity")
                    else "NOT token-identical; see parity.first_divergence"
                ),
                "parity": p.get("parity"),
                "token_ids_identical": p.get("token_ids_identical"),
                "fallbacks": p.get("fallbacks"),
                "argmax_is_not_parity": True,
            },
            {
                "id": "ORGAN_SAVING_VS_COMPLETE_TOKEN",
                "what": s["where"],
                "reached_the_token": s["reached_the_token"],
                "clears_materiality": s["clears_materiality"],
                "complete_token_saving_ms": s["complete_token_saving_ms"],
                "isolated_fair_cut_ms": s["isolated_fair_cut_ms"],
                "cited_organ_saving_ms": s["cited_organ_saving_ms"],
            },
        ],
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise WidenAbRefuse("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    # A hardware number must be placeable in time. This module used to write its
    # own json.dumps with no timestamp, so when /tmp/hawking-gpu-lane.lock was
    # found wedged, placing this receipt against that window needed git landing
    # time - a proxy for when the measurement actually ran.
    doc.setdefault(
        "measurement_provenance",
        measurement_provenance(
            lock_held=bool(os.environ.get("HAWKING_GPU_LANE_LOCK_HELD")),
            lane="deltanet_widen_ab",
            # A receipt rebuilt from a stored raw capture must not stamp the
            # rebuild time as the measurement time. The raw files carry no
            # timestamp, so a retrofit records the measurement time as UNKNOWN.
            retrofit=not os.environ.get("HAWKING_MEASURED_NOW"),
        ),
    )
    write_measured_receipt(out, doc, "tools/future/deltanet_widen_ab.py")
    return out


def example_binaries() -> list[Path]:
    names = ("ascension_qwen38_deltanet_widen_ab",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "workspace" / "ops" / "build" / "rust",
            REPO / "target",
        ]
    )
    out: list[Path] = []
    for root in roots:
        for profile in ("release-fast", "release"):
            for name in names:
                p = root / profile / "examples" / name
                if p.is_file():
                    out.append(p)
    return out


def run_example(
    artifact_root: Path,
    tokenizer: Path,
    *,
    reps: int = 7,
    warmup: int = 1,
    max_new_tokens: int = 32,
    max_seq_len: int = 128,
    out: Path | None = None,
    binary: Path | None = None,
    use_lock: bool = True,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "ascension_qwen38_deltanet_widen_ab binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core "
            "--example ascension_qwen38_deltanet_widen_ab`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--tokenizer",
        str(tokenizer),
        "--reps",
        str(reps),
        "--warmup",
        str(warmup),
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-seq-len",
        str(max_seq_len),
        "--out",
        str(out),
    ]
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd = ["bash", str(lock), "y3widen", *inner] if use_lock and lock.is_file() else inner
    env = os.environ.copy()
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} exited {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
            args.tokenizer,
            reps=args.reps,
            warmup=args.warmup,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            out=RAW_DEFAULT,
            use_lock=not args.no_lock,
        )
    elif args.raw_path:
        raw = load_raw(Path(args.raw_path))
    elif RAW_DEFAULT.is_file():
        raw = load_raw(RAW_DEFAULT)

    if raw is None:
        print(
            "no measurement: pass --from RAW.json, --measure, or write "
            f"{RAW_DEFAULT}",
            file=sys.stderr,
        )
        return 2

    measured = measurement_from_raw(raw)
    if args.record:
        path = record(measured, path=args.out)
        print(f"wrote {path}")
        print(measured["saving"]["where"])
        print(build(measured)["finding"])
    else:
        s = measured["saving"]
        p = measured["parity"]
        print(
            f"complete token incumbent {s['complete_token_incumbent_ms']} ms  "
            f"widen_f4 {s['complete_token_widen_f4_ms']} ms  "
            f"saved {s['complete_token_saving_ms']} ms"
        )
        print(
            f"isolated fair cut {s['isolated_fair_cut_ms']} ms  "
            f"parity={p['parity']} fallbacks={p['fallbacks']}"
        )
        print(s["where"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
