#!/usr/bin/env python3
"""Settle the 1.708x aux-merge shape on a kernel that does real arithmetic.

z2stream's stream-count ladder (receipts/future/MLP_STREAM_COUNT.json) held
38 B/iter, stripped the FMAs, and found one unexplained shape:

    mlp_2_2_2_32   [2,2,2,32]   308.3 GB/s     production's two aux planes
    mid_2_4_32     [2,4,32]     526.6 GB/s     scale+bias in one 4-byte record
    ratio                                   1.708x

Merging further collapsed. That receipt refused to promote the merge: the
probe was an XOR/add sink, not the production matvec. This sidecar is the
matched pair it asked for. ARM A is the production geo_tpr64_tg128 body
verbatim (two half planes). ARM B is that same body with one half2 group
record, interleaved in memory from the existing planes. Same layer, same
payload, both orderings, warmup 60, in one process.

The probe predicted 1.708x. Anything is a result, including 1.00x. A
refutation here closes the last open shape from the stream ladder.

    python3 tools/future/mlp_aux_merge_ab.py --measure --build
    python3 tools/future/mlp_aux_merge_ab.py --from receipts/future/_MLP_AUX_MERGE_AB_raw.json --build
    python3 -m pytest tools/future/test_mlp_aux_merge_ab.py -q

evidence_class SELF_MEASURED_DIRTY unless a quiet window is actually held.
Does not change the production decode path. Does not promote.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    REPO,
    measurement_provenance,
    write_measured_receipt,
)


RECEIPT = REPO / "receipts" / "future" / "MLP_AUX_MERGE_AB.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_MLP_AUX_MERGE_AB_raw.json"
SCHEMA = "hawking.future.mlp_aux_merge_ab.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_aux_merge_ab.py"

BYTES_PER_ITER = 38
INCUMBENT_STREAMS = [2, 2, 2, 32]
MERGED_STREAMS = [2, 4, 32]
STEADY_MAX_SPREAD = 1.10
MIN_WARMUP = 60
MIN_REPS = 7
SAME_RATIO = 1.08
PROBE_SURVIVES_SLACK = 1.12

PROBE_INCUMBENT_GB_S = 308.3
PROBE_MERGED_GB_S = 526.6
PROBE_RATIO = 1.708
CITED_MLP_MS = 11.756
CITED_TOKEN_MS = 21.9464
CITED_GAP_TO_60_MS = 5.2797
CITED_EVERYTHING_ELSE_MS = 4.881
DEFAULT_ARTIFACT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")

INCUMBENT_KERNEL = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"
MERGED_KERNEL = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_aux_merge"

VERDICT_SURVIVES = "PROBE_SURVIVES"
VERDICT_SMALLER = "LIFT_SMALLER_THAN_PROBE"
VERDICT_NONE = "NO_LIFT"
VERDICT_HURTS = "HURTS"
VERDICT_BLOCKED = "BLOCKED"
VERDICTS = (
    VERDICT_SURVIVES,
    VERDICT_SMALLER,
    VERDICT_NONE,
    VERDICT_HURTS,
    VERDICT_BLOCKED,
)

CLAIM_BOUNDARY = (
    "One representative MLP layer (gate+up+down) on sealed-3.14, "
    "SELF_MEASURED_DIRTY. GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime "
    "for an isolated command buffer of three geo_tpr64_tg128 dispatches. Unique "
    "payload bytes (codes+scales+biases of the launched tensors) are the GB/s "
    "numerator on both arms; counted traffic per thread-iteration is 38 B on "
    "both arms (A: 2+2+2+32, B: 2+4+32). ARM A is the production unpack body; "
    "ARM B is that body with scale and bias loaded as one half2. The host "
    "interleaves the two existing planes in memory and does not rewrite any "
    "artifact on disk. Both orderings (A then B, B then A) run in one process "
    "after warmup 60. Absolute GB/s is measured-under-load; the claim is the "
    "back-to-back ratio. Does not change the production decode path. Does not "
    "promote. FMA count, total issue rate, dependency chains, register pressure "
    "and stream count are REFUTED elsewhere."
)


class AbRefused(RuntimeError):
    """The pair is not matched, or an arm is not a measurement."""


class ByteMismatch(AbRefused):
    """The two arms do not move the same unique bytes; a rate is not a claim."""


class SpreadRefused(AbRefused):
    """An arm's reps are not homogeneous; its median is a coin flip."""


class EmptyGpuSample(AbRefused):
    """Raised rather than invent a timestamp."""


class MissingArm(AbRefused):
    pass


class Blocked(AbRefused):
    """Compile failed, lease missing, or no gpu_ns. Honest, not a number."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def rep_spread(reps: list[int]) -> float:
    if not reps:
        raise EmptyGpuSample("no gpu_ns_reps to form a spread")
    lo, hi = min(reps), max(reps)
    if lo <= 0:
        raise EmptyGpuSample("gpu_ns_reps contain a non-positive sample")
    return hi / lo


def _ratio(a: float, b: float) -> float:
    if a <= 0:
        raise EmptyGpuSample("ratio denominator is not a measurement")
    return b / a


def same_rate(a: float, b: float, bar: float = SAME_RATIO) -> bool:
    lo, hi = min(a, b), max(a, b)
    if lo <= 0:
        return False
    return hi / lo <= bar


def assert_warmup(raw: Mapping[str, Any]) -> None:
    warmup = int(raw.get("warmup") or 0)
    if warmup < MIN_WARMUP:
        raise AbRefused(
            f"warmup {warmup} < {MIN_WARMUP}; the first-measured arm is not "
            "in steady state at that warmup and its median is a coin flip "
            "between two modes"
        )
    reps = int(raw.get("reps") or 0)
    if reps < MIN_REPS:
        raise AbRefused(f"reps {reps} < {MIN_REPS}")


def assert_steady(arm: Mapping[str, Any], label: str) -> float:
    spread = float(arm.get("rep_spread") or 0.0)
    if "rep_spread" not in arm:
        reps = [int(x) for x in arm.get("gpu_ns_reps") or []]
        spread = rep_spread(reps)
    if spread > STEADY_MAX_SPREAD:
        raise SpreadRefused(
            f"{label} has rep_spread {spread:.4f} > {STEADY_MAX_SPREAD}"
        )
    return spread


def assert_bytes_held(arm: Mapping[str, Any], ident: str, streams: list[int]) -> None:
    got = int(arm.get("bytes_per_thread_iteration") or 0)
    if got != BYTES_PER_ITER:
        raise ByteMismatch(
            f"{ident}: bytes_per_thread_iteration {got} != {BYTES_PER_ITER}"
        )
    listed = [int(x) for x in (arm.get("bytes_per_stream") or [])]
    if listed and listed != streams:
        raise ByteMismatch(
            f"{ident}: bytes_per_stream {listed} != {streams}"
        )
    if listed and sum(listed) != BYTES_PER_ITER:
        raise ByteMismatch(
            f"{ident}: bytes_per_stream {listed} sum to {sum(listed)} != {BYTES_PER_ITER}"
        )


def assert_bytes_matched(incumbent: Mapping[str, Any], merged: Mapping[str, Any]) -> None:
    a = int(incumbent.get("unique_payload_bytes") or incumbent.get("weight_bytes") or 0)
    b = int(merged.get("unique_payload_bytes") or merged.get("weight_bytes") or 0)
    if a <= 0 or b <= 0:
        raise ByteMismatch("unique payload bytes missing; refusing a rate claim")
    if a != b:
        raise ByteMismatch(
            f"unique payload bytes differ: incumbent {a} vs merged {b}; "
            "refusing a rate claim on an unmatched working set"
        )
    ia = int(incumbent.get("bytes_per_thread_iteration") or 0)
    ib = int(merged.get("bytes_per_thread_iteration") or 0)
    if ia != ib:
        raise ByteMismatch(
            f"bytes/iteration differ: incumbent {ia} vs merged {ib}; "
            "refusing a rate claim"
        )
    if ia != BYTES_PER_ITER:
        raise ByteMismatch(
            f"bytes/iteration {ia} != {BYTES_PER_ITER}; comparison is not matched"
        )


def _require_arm(block: Mapping[str, Any], name: str, where: str) -> dict[str, Any]:
    arm = block.get(name)
    if not isinstance(arm, Mapping):
        raise MissingArm(f"{where} is missing arm {name}")
    if int(arm.get("gpu_ns_median") or 0) <= 0:
        raise EmptyGpuSample(f"{where}.{name} has no gpu_ns")
    return dict(arm)


def _arm_view(raw: Mapping[str, Any], ident: str, streams: list[int]) -> dict[str, Any]:
    weight = int(raw.get("unique_payload_bytes") or raw.get("weight_bytes") or 0)
    gpu_ns = int(raw.get("gpu_ns_median") or 0)
    reps = [int(x) for x in raw.get("gpu_ns_reps") or []]
    gb_s = effective_gb_s(weight, gpu_ns)
    spread = float(raw.get("rep_spread") or (rep_spread(reps) if reps else 0.0))
    listed = [int(x) for x in (raw.get("bytes_per_stream") or streams)]
    out = {
        "id": ident,
        "kernel": raw.get("kernel"),
        "stream_count": int(raw.get("stream_count") or len(listed)),
        "bytes_per_stream": listed,
        "bytes_per_thread_iteration": int(
            raw.get("bytes_per_thread_iteration") or sum(listed) or 0
        ),
        "unique_payload_bytes": weight,
        "weight_bytes": weight,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_min": int(raw.get("gpu_ns_min") or (min(reps) if reps else 0)),
        "gpu_ns_max": int(raw.get("gpu_ns_max") or (max(reps) if reps else 0)),
        "gpu_ns_reps": reps,
        "gpu_us_median": round(gpu_ns / 1e3, 3),
        "rep_spread": spread,
        "steady_state": bool(raw.get("steady_state", spread <= STEADY_MAX_SPREAD)),
        "effective_gb_s": round(gb_s, 1),
        "dispatches": int(raw.get("dispatches", 3)),
        "encoders": int(raw.get("encoders", 1)),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "threads_per_threadgroup": int(raw.get("threads_per_threadgroup", 128)),
    }
    if "occupancy" in raw:
        out["occupancy"] = raw["occupancy"]
    assert_bytes_held(out, ident, streams)
    return out


def _order_view(block: Mapping[str, Any], where: str) -> dict[str, Any]:
    inc = _arm_view(_require_arm(block, "incumbent", where), "incumbent", INCUMBENT_STREAMS)
    mer = _arm_view(_require_arm(block, "merged", where), "merged", MERGED_STREAMS)
    assert_bytes_matched(inc, mer)
    assert_steady(inc, f"{where}.incumbent")
    assert_steady(mer, f"{where}.merged")
    ratio = _ratio(float(inc["effective_gb_s"]), float(mer["effective_gb_s"]))
    return {
        "sequence": list(block.get("sequence") or []),
        "incumbent": inc,
        "merged": mer,
        "ratio_merged_over_incumbent": round(ratio, 4),
    }


def _output_compare(raw: Mapping[str, Any]) -> dict[str, Any]:
    oc = raw.get("output_compare")
    if not isinstance(oc, Mapping):
        raise AbRefused("output_compare is missing; a layout change without an output check is not a result")
    n = int(oc.get("n_compared") or 0)
    if n <= 0:
        raise AbRefused("output_compare.n_compared is 0")
    n_exact = int(oc.get("n_bit_exact") or 0)
    return {
        "n_compared": n,
        "n_bit_exact": n_exact,
        "max_abs_err": float(oc.get("max_abs_err") or 0.0),
        "rel_fro": float(oc.get("rel_fro") or 0.0),
        "bit_identical": bool(oc.get("bit_identical", n_exact == n)),
        "per_projection": oc.get("per_projection") or [],
    }


def judge(measurement: Mapping[str, Any]) -> dict[str, Any]:
    pooled = measurement["pooled"]
    inc = pooled["incumbent"]
    mer = pooled["merged"]
    assert_bytes_matched(inc, mer)
    ratio = _ratio(float(inc["effective_gb_s"]), float(mer["effective_gb_s"]))
    ab = float(measurement["order_ab"]["ratio_merged_over_incumbent"])
    ba = float(measurement["order_ba"]["ratio_merged_over_incumbent"])
    orderings_agree = same_rate(ab, ba)
    vs_probe = ratio / PROBE_RATIO if PROBE_RATIO else None

    if not orderings_agree:
        verdict = VERDICT_NONE
        why = (
            f"the two orderings disagree (AB {ab:.4f}x vs BA {ba:.4f}x); "
            "first-arm residency is still in the result, refusing a rate claim"
        )
    elif same_rate(ratio, 1.0):
        verdict = VERDICT_NONE
        why = (
            f"merged/incumbent = {ratio:.4f}x, within {SAME_RATIO} of 1.00; "
            "the 1.708x stripped-probe lift does not survive on real arithmetic"
        )
    elif ratio < 1.0 / SAME_RATIO:
        verdict = VERDICT_HURTS
        why = (
            f"merged/incumbent = {ratio:.4f}x; merging the aux planes HURTS "
            "on the production body (the stripped probe's 1.708x does not survive)"
        )
    elif ratio >= PROBE_RATIO / PROBE_SURVIVES_SLACK:
        verdict = VERDICT_SURVIVES
        why = (
            f"merged/incumbent = {ratio:.4f}x, within {PROBE_SURVIVES_SLACK} of "
            f"the stripped probe's {PROBE_RATIO}x; the aux-merge lift survives "
            "on a kernel that does real arithmetic"
        )
    else:
        verdict = VERDICT_SMALLER
        why = (
            f"merged/incumbent = {ratio:.4f}x, a lift, but well short of the "
            f"stripped probe's {PROBE_RATIO}x (probe was measuring the "
            "streaming ceiling of an XOR/add sink, not this unpack)"
        )

    oc = measurement.get("output_compare") or {}
    bit_identical = bool(oc.get("bit_identical"))
    return {
        "verdict": verdict,
        "why": why,
        "ratio_merged_over_incumbent": round(ratio, 4),
        "order_ab_ratio": round(ab, 4),
        "order_ba_ratio": round(ba, 4),
        "orderings_agree": orderings_agree,
        "vs_probe_ratio": None if vs_probe is None else round(vs_probe, 4),
        "probe_ratio": PROBE_RATIO,
        "same_ratio_bar": SAME_RATIO,
        "probe_survives_slack": PROBE_SURVIVES_SLACK,
        "bit_identical": bit_identical,
        "does_not_promote": True,
        "incumbent_gb_s": float(inc["effective_gb_s"]),
        "merged_gb_s": float(mer["effective_gb_s"]),
        "incumbent_us": float(inc["gpu_us_median"]),
        "merged_us": float(mer["gpu_us_median"]),
        "incumbent_rep_spread": float(inc["rep_spread"]),
        "merged_rep_spread": float(mer["rep_spread"]),
        "bytes_per_thread_iteration": BYTES_PER_ITER,
        "unique_payload_bytes": int(inc["unique_payload_bytes"]),
    }


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("blocked"):
        raise Blocked(str(raw.get("blocked_error") or "blocked with no gpu_ns"))
    if raw.get("schema") not in (
        "hawking.future.mlp_aux_merge_ab.raw.v1",
        SCHEMA,
    ) and "pooled" not in raw:
        raise MissingArm(f"unrecognised schema {raw.get('schema')!r}")
    assert_warmup(raw)
    order_ab = _order_view(raw.get("order_ab") or {}, "order_ab")
    order_ba = _order_view(raw.get("order_ba") or {}, "order_ba")
    pooled = _order_view(raw.get("pooled") or {}, "pooled")
    oc = _output_compare(raw)
    measured = {
        "layer": int(raw.get("layer", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "fast_math": bool(raw.get("fast_math", False)),
        "geometry": raw.get("geometry", "geo_tpr64_tg128"),
        "codec": raw.get("codec", "HGRAVF01 affine2 q2 group64"),
        "organ": raw.get("organ", "mlp"),
        "dispatches": int(raw.get("dispatches") or 3),
        "absolute_gb_s_are_measured_under_load": True,
        "bytes_per_thread_iteration_held": BYTES_PER_ITER,
        "steady_state_max_spread": STEADY_MAX_SPREAD,
        "aux_repack_on_disk": False,
        "does_not_promote": True,
        "production_shader_untouched": True,
        "concurrent_load": raw.get("concurrent_load") or {},
        "concurrent_load_end": raw.get("concurrent_load_end") or {},
        "projections": raw.get("projections") or [],
        "weight_bytes": int(raw.get("weight_bytes") or pooled["incumbent"]["weight_bytes"]),
        "unique_payload_bytes": int(
            raw.get("unique_payload_bytes") or pooled["incumbent"]["unique_payload_bytes"]
        ),
        "merged_aux_bytes": int(raw.get("merged_aux_bytes") or 0),
        "scale_bias_bytes": int(raw.get("scale_bias_bytes") or 0),
        "output_compare": oc,
        "order_ab": order_ab,
        "order_ba": order_ba,
        "pooled": pooled,
    }
    if measured["merged_aux_bytes"] and measured["scale_bias_bytes"]:
        if measured["merged_aux_bytes"] != measured["scale_bias_bytes"]:
            raise ByteMismatch(
                f"merged aux {measured['merged_aux_bytes']} != "
                f"scale+bias {measured['scale_bias_bytes']}"
            )
    measured["judgement"] = judge(measured)
    measured["verdict"] = measured["judgement"]["verdict"]
    return measured


def _finding(measurement: Mapping[str, Any]) -> str:
    j = measurement["judgement"]
    oc = measurement["output_compare"]
    inc = measurement["pooled"]["incumbent"]
    mer = measurement["pooled"]["merged"]
    identity = (
        "bit-identical"
        if oc["bit_identical"]
        else (
            f"NOT bit-identical (n_compared={oc['n_compared']}, "
            f"n_bit_exact={oc['n_bit_exact']}, max_abs_err={oc['max_abs_err']}, "
            f"rel_fro={oc['rel_fro']})"
        )
    )
    return (
        f"Verdict {j['verdict']}. Merged/incumbent = {j['ratio_merged_over_incumbent']}x "
        f"(AB {j['order_ab_ratio']}x, BA {j['order_ba_ratio']}x) on unique payload "
        f"{inc['effective_gb_s']} -> {mer['effective_gb_s']} GB/s, "
        f"{inc['gpu_us_median']} -> {mer['gpu_us_median']} us, "
        f"{BYTES_PER_ITER} B/iter both arms. Output {identity}. {j['why']}. "
        "Does not promote."
    )


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    j = judge(measurement)
    load_open = (measurement.get("concurrent_load") or {}).get("loadavg")
    load_close = (measurement.get("concurrent_load_end") or {}).get("loadavg")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "does_not_promote": True,
        "production_shader_untouched": True,
        "aux_repack_on_disk": False,
        "source": (
            "crates/hawking-core/examples/mlp_aux_merge_ab.rs; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime); "
            "one representative MLP layer of sealed-3.14; production unpack "
            "body vs the same body with scale+bias as one half2"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "cited": {
            "probe_incumbent_gb_s": PROBE_INCUMBENT_GB_S,
            "probe_merged_gb_s": PROBE_MERGED_GB_S,
            "probe_ratio": PROBE_RATIO,
            "mlp_ms": CITED_MLP_MS,
            "token_ms": CITED_TOKEN_MS,
            "gap_to_60_ms": CITED_GAP_TO_60_MS,
            "everything_else_ms": CITED_EVERYTHING_ELSE_MS,
            "from": (
                "receipts/future/MLP_STREAM_COUNT.json, "
                "receipts/future/ORGAN_DECOMPOSITION_SEALED.json"
            ),
        },
        "refuted_elsewhere": [
            "fma_count",
            "total_issue_rate",
            "dependency_chains",
            "register_pressure",
            "stream_count",
        ],
        "bytes_per_thread_iteration_held": BYTES_PER_ITER,
        "incumbent_bytes_per_stream": INCUMBENT_STREAMS,
        "merged_bytes_per_stream": MERGED_STREAMS,
        "same_ratio": SAME_RATIO,
        "steady_state_max_spread": STEADY_MAX_SPREAD,
        "min_warmup": MIN_WARMUP,
        "absolute_gb_s_are_measured_under_load": True,
        "layer": measurement.get("layer"),
        "warmup": measurement.get("warmup"),
        "reps": measurement.get("reps"),
        "weight_bytes": measurement.get("weight_bytes"),
        "unique_payload_bytes": measurement.get("unique_payload_bytes"),
        "dispatches": measurement.get("dispatches"),
        "geometry": measurement.get("geometry"),
        "codec": measurement.get("codec"),
        "timing": measurement.get("timing"),
        "fast_math": measurement.get("fast_math"),
        "projections": measurement.get("projections") or [],
        "output_compare": measurement["output_compare"],
        "order_ab": measurement["order_ab"],
        "order_ba": measurement["order_ba"],
        "pooled": measurement["pooled"],
        "incumbent": measurement["pooled"]["incumbent"],
        "merged": measurement["pooled"]["merged"],
        "judgement": j,
        "verdict": j["verdict"],
        "finding": _finding({**measurement, "judgement": j, "verdict": j["verdict"]}),
        "loadavg_open": load_open,
        "loadavg_close": load_close,
        "concurrent_load": measurement.get("concurrent_load"),
        "concurrent_load_end": measurement.get("concurrent_load_end"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "scar": "WARMUP_5_LEAVES_THE_FIRST_MEASURED_ARM_OUTSIDE_STEADY_STATE",
    }


def blocked_receipt(error: str, raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A receipt with no gpu_ns. Honest. An invented timestamp ends the campaign."""
    raw = raw or {}
    load_open = (raw.get("concurrent_load") or {}).get("loadavg")
    load_close = (raw.get("concurrent_load_end") or {}).get("loadavg")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "does_not_promote": True,
        "production_shader_untouched": True,
        "aux_repack_on_disk": False,
        "verdict": VERDICT_BLOCKED,
        "blocked_error": error,
        "finding": f"BLOCKED: {error}",
        "claim_boundary": CLAIM_BOUNDARY,
        "warmup": raw.get("warmup"),
        "reps": raw.get("reps"),
        "layer": raw.get("layer", 0),
        "loadavg_open": load_open,
        "loadavg_close": load_close,
        "concurrent_load": raw.get("concurrent_load") or {},
        "concurrent_load_end": raw.get("concurrent_load_end") or {},
        "artifact_root": raw.get("artifact_root"),
        "git_head": raw.get("git_head", ""),
        "needs": "gate profile (unsandboxed); MTLCreateSystemDefaultDevice returned nil in this sandbox",
    }


def record_blocked(error: str, raw: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    doc = blocked_receipt(error, raw)
    load_open = (raw or {}).get("concurrent_load", {}).get("loadavg") if raw else None
    return write_measured_receipt(
        path or RECEIPT,
        doc,
        RECORDED_BY,
        provenance=measurement_provenance(lock_held=True, loadavg=load_open, lane="mlpaux"),
    )


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise MissingArm("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    load_open = (measurement.get("concurrent_load") or {}).get("loadavg")
    return write_measured_receipt(
        out,
        doc,
        RECORDED_BY,
        provenance=measurement_provenance(
            lock_held=True,
            loadavg=load_open,
            lane="mlpaux",
        ),
    )


def example_binaries() -> list[Path]:
    names = ("mlp_aux_merge_ab",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "target",
            REPO / "workspace" / "ops" / "build" / "rust",
            Path("/Users/scammermike/Downloads/hawking/workspace/ops/build/rust"),
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
    *,
    layer: int = 0,
    warmup: int = MIN_WARMUP,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
    take_lease: bool = True,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "mlp_aux_merge_ab binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example mlp_aux_merge_ab`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--layer",
        str(layer),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--out",
        str(out),
    ]
    if take_lease:
        cmd = ["bash", str(REPO / "tools" / "gpu_lane_lock.sh"), "mlpaux", *inner]
    else:
        cmd = inner
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise Blocked(
            f"{' '.join(cmd)} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example under gpu_lane_lock")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=MIN_WARMUP)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-lease", action="store_true", help="caller already holds gpu_lane_lock")
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        try:
            raw = run_example(
                args.artifact_root,
                layer=args.layer,
                warmup=args.warmup,
                reps=args.reps,
                out=RAW_DEFAULT,
                take_lease=not args.no_lease,
            )
        except Blocked as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            if args.build:
                path = record_blocked(str(exc), path=args.out)
                print(f"wrote {path} verdict={VERDICT_BLOCKED}")
                return 0
            return 2
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

    try:
        measured = measurement_from_raw(raw)
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        if args.build:
            path = record_blocked(str(exc), raw, path=args.out)
            print(f"wrote {path} verdict={VERDICT_BLOCKED}")
            return 0
        return 2

    if args.build:
        path = record(measured, path=args.out)
        print(f"wrote {path} verdict={measured['verdict']}")
    else:
        j = measured["judgement"]
        print(f"verdict={measured['verdict']}")
        print(
            f"  incumbent {j['incumbent_gb_s']} GB/s  {j['incumbent_us']} us  "
            f"spread {j['incumbent_rep_spread']:.4f}"
        )
        print(
            f"  merged    {j['merged_gb_s']} GB/s  {j['merged_us']} us  "
            f"spread {j['merged_rep_spread']:.4f}"
        )
        print(f"  ratio {j['ratio_merged_over_incumbent']}x  (probe {PROBE_RATIO}x)")
        print(f"  why: {j['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
