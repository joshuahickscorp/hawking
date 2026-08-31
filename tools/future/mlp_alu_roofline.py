#!/usr/bin/env python3
"""IS THE MLP KERNEL ALU-BOUND? Matched pair nobody had run.

Four layout/dispatch hypotheses already sit dead (granularity, catalog
addressing, raw dispatch count, bytes-per-dispatch). All of them are also
consistent with: the kernel is not bandwidth-bound at all. This sidecar
runs the production MLP affine2 geo_tpr64 kernel and DeltaNet's dominant
Q4 geo_tpr64 kernel as a matched pair:

    ARM A  bytes identical, arithmetic STRIPPED (XOR/add sink of the same loads)
    ARM B  arithmetic identical, bytes CUT (first half of K, same per-code work)

    A jumps and B scales sub-linearly  ->  ALU_BOUND
    A stays near production            ->  MEMORY_SYSTEM_BOUND
    anything else                      ->  MIXED  (do not force a binary)

    python3 tools/future/mlp_alu_roofline.py --record
    python3 tools/future/mlp_alu_roofline.py --from receipts/future/_MLP_ALU_ROOFLINE_raw.json --record
    python3 tools/future/mlp_alu_roofline.py --measure --record
    python3 -m pytest tools/future/test_mlp_alu_roofline.py -q

evidence_class SELF_MEASURED_DIRTY. Absolute GB/s is measured-under-load.
The verdict is the back-to-back ratio. Does not change the production decode path.
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
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "MLP_ALU_ROOFLINE.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_MLP_ALU_ROOFLINE_raw.json"
SCHEMA = "hawking.future.mlp_alu_roofline.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_alu_roofline.py"

CLEAN_GEMV_GB_S = 703.5
LM_HEAD_GB_S = 497.4
PRODUCTION_CLUSTER_GB_S = 350.0

# Ratio bars, not absolute GB/s: other lanes may share the GPU.
# ARM A same bytes, so gb_s_A / gb_s_prod == time_prod / time_A.
JUMP_RATIO = 1.25          # stripped is materially faster -> toward the LM-head regime
STAY_RATIO = 1.12          # stripped barely moves
# ARM B: time should track bytes if bandwidth-bound. Sub-linear = time did not drop
# as much as bytes (time_B/time_prod > SUBLINEAR_SLACK * bytes_B/bytes_prod).
SUBLINEAR_SLACK = 1.20
LINEAR_TOLERANCE = 0.20    # |time_ratio/byte_ratio - 1| <= this is LINEAR

VERDICT_ALU = "ALU_BOUND"
VERDICT_MEM = "MEMORY_SYSTEM_BOUND"
VERDICT_MIXED = "MIXED"

ORGANS = ("mlp", "deltanet")

CLAIM_BOUNDARY = (
    "One representative layer on sealed-3.14, SELF_MEASURED_DIRTY. GPU time is "
    "MTLCommandBuffer GPUStartTime/GPUEndTime for an isolated command buffer. "
    "Bytes are GPU-resident codes+scales(+biases) of the launched tensors. "
    "Bandwidth is those bytes divided by GPU ns (perfect-locality). ARM A keeps "
    "the production access pattern and byte count and replaces decode+dequant+FMA "
    "with a XOR/add sink; ARM B keeps production arithmetic and cuts K in half. "
    "Absolute GB/s is measured-under-load and is not a clean roof; the verdict "
    "is the back-to-back ratio of the two arms in the same process. Loads on "
    "ARM A are shown to have survived if stripped time exceeds the zero-load "
    "floor and stripped-half time drops with bytes. This does not change the "
    "production decode path. Registers per thread are not exposed by Metal "
    "pipeline state on this toolchain."
)

# Production inner-loop tax, counted from the geo_tpr64 bodies.
# Affine2 g64: 8 weights / iter from 2 B codes + 2 B scale + 2 B bias; 8 x-floats.
# Q4 g64:      8 weights / iter from 4 B codes + 2 B scale; 8 x-floats.
DECODE_TAX = {
    "mlp": {
        "weights_per_iteration": 8,
        "weight_bytes_per_iteration": 6,   # 2 code + 2 scale + 2 bias
        "x_bytes_per_iteration": 32,
        "dequant_fma": 8,                  # float(q)*scale+bias
        "mac_fma": 8,                      # w*x
        "bitops": 16,                      # shift+mask per q
        "int_to_float": 8,
        "fma_per_weight_byte": round(16 / 6, 4),
        "decode_fma_per_weight_byte": round(8 / 6, 4),
    },
    "deltanet": {
        "weights_per_iteration": 8,
        "weight_bytes_per_iteration": 6,   # 4 code + 2 scale
        "x_bytes_per_iteration": 32,
        "dequant_fma": 8,                  # float(nibble-8)*scale
        "mac_fma": 8,
        "bitops": 16,                      # nibble extract + sub 8
        "int_to_float": 8,
        "fma_per_weight_byte": round(16 / 6, 4),
        "decode_fma_per_weight_byte": round(8 / 6, 4),
    },
}


class MissingArm(Exception):
    """Raised rather than emit a verdict over an incomplete matched pair."""


class ByteMismatch(Exception):
    """Raised when ARM A does not read the same weight bytes as production."""


class EmptyGpuSample(Exception):
    """Raised rather than divide by a missing GPU timestamp."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def _require_arm(organ: Mapping[str, Any], key: str, organ_name: str) -> Mapping[str, Any]:
    arm = organ.get(key)
    if not isinstance(arm, Mapping):
        raise MissingArm(
            f"refusing a verdict: {organ_name} is missing {key}"
        )
    for field in ("weight_bytes", "gpu_ns_median"):
        if field not in arm:
            raise MissingArm(
                f"refusing a verdict: {organ_name}.{key} is missing {field}"
            )
    return arm


def loads_survived(organ: Mapping[str, Any]) -> dict[str, Any]:
    """Did ARM A's compiler keep the loads? Finite GB/s is necessary but not enough."""
    a = organ["arm_a_stripped"]
    zero = organ.get("zero_load") or {}
    a_half = organ.get("arm_a_halfk") or {}
    a_ns = int(a["gpu_ns_median"])
    zero_ns = int(zero.get("gpu_ns_median") or 0)
    a_half_ns = int(a_half.get("gpu_ns_median") or 0)
    gb_s = float(a.get("effective_gb_s") or 0.0)
    finite = gb_s > 0.0 and a_ns > 0
    above_floor = zero_ns > 0 and a_ns > 1.3 * zero_ns
    scales = a_half_ns > 0 and a_half_ns < 0.85 * a_ns
    survived = bool(finite and (above_floor or scales))
    return {
        "survived": survived,
        "finite_gb_s": finite,
        "above_zero_load_floor": above_floor,
        "time_scales_with_bytes": scales,
        "stripped_gpu_ns": a_ns,
        "zero_load_gpu_ns": zero_ns or None,
        "stripped_half_gpu_ns": a_half_ns or None,
        "proof": (
            "stripped time exceeds the no-load floor and/or drops when bytes are halved"
            if survived
            else "cannot prove the stripped loads survived; ARM A jump would be MIXED not ALU_BOUND"
        ),
    }


def _arm_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    weight_bytes = int(raw["weight_bytes"])
    gpu_ns = int(raw["gpu_ns_median"])
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    out = {
        "label": str(raw.get("label", "")),
        "kernel": raw.get("kernel"),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "dispatches": int(raw.get("dispatches", 1)),
        "encoders": int(raw.get("encoders", 1)),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "effective_gb_s": round(gb_s, 1),
        "share_of_clean_roof": round(gb_s / CLEAN_GEMV_GB_S, 4),
        "share_of_lm_head": round(gb_s / LM_HEAD_GB_S, 4),
    }
    if "occupancy" in raw:
        out["occupancy"] = raw["occupancy"]
    return out


def cheaper_decode(organ_name: str, production_gb_s: float) -> dict[str, Any]:
    tax = DECODE_TAX[organ_name]
    ops = tax["decode_fma_per_weight_byte"]
    # Same ALU throughput, fewer ops/byte, to reach the LM-head's demonstrated rate.
    target = ops * (production_gb_s / LM_HEAD_GB_S) if production_gb_s > 0 else None
    return {
        "production_decode_fma_per_weight_byte": ops,
        "production_fma_per_weight_byte": tax["fma_per_weight_byte"],
        "inner_loop": tax,
        "to_match_lm_head_497": {
            "target_decode_fma_per_weight_byte": None if target is None else round(target, 4),
            "required_decode_cheapening": None
            if production_gb_s <= 0
            else round(LM_HEAD_GB_S / production_gb_s, 3),
            "note": (
                "If the kernel is ALU-bound at production_gb_s, a cheaper decode "
                "must cut decode-FMA per weight-byte by this factor (or raise "
                "issue rate by the same factor) to sit at the LM head's 497.4 GB/s. "
                "Not a promise that such a decode exists."
            ),
        },
    }


def judge_organ(organ: Mapping[str, Any], name: str) -> dict[str, Any]:
    prod = _require_arm(organ, "production", name)
    arm_a = _require_arm(organ, "arm_a_stripped", name)
    arm_b = _require_arm(organ, "arm_b_halfk", name)

    prod_bytes = int(prod["weight_bytes"])
    a_bytes = int(arm_a["weight_bytes"])
    if a_bytes != prod_bytes:
        raise ByteMismatch(
            f"refusing a verdict: {name} ARM A weight_bytes {a_bytes} != "
            f"production {prod_bytes}"
        )

    prod_ns = int(prod["gpu_ns_median"])
    a_ns = int(arm_a["gpu_ns_median"])
    b_ns = int(arm_b["gpu_ns_median"])
    b_bytes = int(arm_b["weight_bytes"])
    if prod_ns <= 0 or a_ns <= 0 or b_ns <= 0:
        raise EmptyGpuSample(f"{name}: gpu_ns_median must be positive on every arm")
    if b_bytes <= 0 or prod_bytes <= 0:
        raise ValueError(f"{name}: weight_bytes must be positive")

    prod_gb = effective_gb_s(prod_bytes, prod_ns)
    a_gb = effective_gb_s(a_bytes, a_ns)
    b_gb = effective_gb_s(b_bytes, b_ns)
    a_ratio = a_gb / prod_gb
    byte_ratio = b_bytes / prod_bytes
    time_ratio = b_ns / prod_ns
    # time_ratio / byte_ratio == 1 → linear (bandwidth-like). >1 → sub-linear.
    scale = time_ratio / byte_ratio if byte_ratio > 0 else float("inf")
    b_linear = abs(scale - 1.0) <= LINEAR_TOLERANCE
    b_sublinear = scale > SUBLINEAR_SLACK
    a_jump = a_ratio >= JUMP_RATIO
    a_stay = a_ratio <= STAY_RATIO
    survived = loads_survived(organ)

    if a_stay:
        verdict = VERDICT_MEM
    elif a_jump and b_sublinear and survived["survived"]:
        verdict = VERDICT_ALU
    else:
        verdict = VERDICT_MIXED

    occ = (prod.get("occupancy") or {})
    max_tg = occ.get("max_total_threads_per_threadgroup")
    tgpc = occ.get("threadgroups_per_core")
    occupancy_limited = bool(
        (isinstance(max_tg, (int, float)) and 0 < max_tg <= 128)
        or (isinstance(tgpc, (int, float)) and tgpc < 1.0)
    )

    return {
        "verdict": verdict,
        "production_gb_s": round(prod_gb, 1),
        "arm_a_gb_s": round(a_gb, 1),
        "arm_b_gb_s": round(b_gb, 1),
        "arm_a_over_production": round(a_ratio, 4),
        "arm_b_byte_ratio": round(byte_ratio, 4),
        "arm_b_time_ratio": round(time_ratio, 4),
        "arm_b_time_over_byte": round(scale, 4),
        "arm_a_jump": a_jump,
        "arm_a_stay": a_stay,
        "arm_b_linear": b_linear,
        "arm_b_sublinear": b_sublinear,
        "loads_survived": survived,
        "occupancy_limited": occupancy_limited,
        "jump_ratio_bar": JUMP_RATIO,
        "stay_ratio_bar": STAY_RATIO,
        "sublinear_slack": SUBLINEAR_SLACK,
        "why_not_forced": (
            None
            if verdict != VERDICT_MIXED
            else (
                "ARM A jumped but ARM B tracked bytes (half K also halves FMAs, so "
                "linear scaling is predicted by both ALU and bandwidth); the "
                "pre-registered rule will not promote that pair to ALU_BOUND"
                if a_jump and b_linear and survived["survived"]
                else "ARM A jumped but stripped loads were not shown to survive"
                if a_jump and not survived["survived"]
                else "ARM A is in the dead band between stay and jump"
                if not a_stay and not a_jump
                else "organs or arms did not meet both ALU_BOUND conjuncts"
            )
        ),
    }


def judge(measurement: Mapping[str, Any]) -> str:
    """Combined verdict. MIXED if the two organs disagree or either is MIXED."""
    seen: list[str] = []
    for name in ORGANS:
        if name not in measurement:
            raise MissingArm(f"refusing a verdict: organ {name} is missing")
        seen.append(judge_organ(measurement[name], name)["verdict"])
    if all(v == VERDICT_ALU for v in seen):
        return VERDICT_ALU
    if all(v == VERDICT_MEM for v in seen):
        return VERDICT_MEM
    return VERDICT_MIXED


def _organ_from_raw(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    if name not in raw or not isinstance(raw[name], Mapping):
        raise MissingArm(f"refusing a verdict: organ {name} is missing")
    src = raw[name]
    _require_arm(src, "production", name)
    _require_arm(src, "arm_a_stripped", name)
    _require_arm(src, "arm_b_halfk", name)
    prod = _arm_view(src["production"])
    arm_a = _arm_view(src["arm_a_stripped"])
    arm_b = _arm_view(src["arm_b_halfk"])
    organ = {
        "organ": name,
        "kernel": src.get("kernel"),
        "codec": src.get("codec"),
        "threads_per_threadgroup": src.get("threads_per_threadgroup", 128),
        "bytes_per_thread_iteration": src.get("bytes_per_thread_iteration"),
        "inner_loop_trips": src.get("inner_loop_trips") or src.get("inner_loop_trips_gate"),
        "projections": src.get("projections") or src.get("projection"),
        "production": prod,
        "arm_a_stripped": arm_a,
        "arm_b_halfk": arm_b,
        "decode_tax": cheaper_decode(name, prod["effective_gb_s"]),
    }
    if "zero_load" in src:
        organ["zero_load"] = _arm_view(src["zero_load"])
    if "arm_a_halfk" in src:
        organ["arm_a_halfk"] = _arm_view(src["arm_a_halfk"])
    judged = judge_organ(organ, name)
    organ["judgement"] = judged
    organ["verdict"] = judged["verdict"]
    return organ


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    organs = {name: _organ_from_raw(raw, name) for name in ORGANS}
    return {
        "layer": int(raw.get("layer", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "concurrent_load": raw.get("concurrent_load") or {},
        "absolute_gb_s_are_measured_under_load": True,
        "mlp": organs["mlp"],
        "deltanet": organs["deltanet"],
        "verdict": judge(organs),
    }


def _finding(measurement: Mapping[str, Any]) -> str:
    mlp_v = measurement["mlp"]["verdict"]
    dn_v = measurement["deltanet"]["verdict"]
    combined = measurement["verdict"]
    mj = measurement["mlp"]["judgement"]
    dj = measurement["deltanet"]["judgement"]
    lines = [
        f"Combined verdict {combined} (MLP {mlp_v}, DeltaNet {dn_v}).",
        (
            f"MLP production {mj['production_gb_s']} GB/s, ARM A (stripped) "
            f"{mj['arm_a_gb_s']} GB/s ({mj['arm_a_over_production']}x), "
            f"ARM B (half K) time/byte {mj['arm_b_time_over_byte']} "
            f"(time_ratio {mj['arm_b_time_ratio']}, byte_ratio {mj['arm_b_byte_ratio']})."
        ),
        (
            f"DeltaNet qkvz production {dj['production_gb_s']} GB/s, ARM A "
            f"{dj['arm_a_gb_s']} GB/s ({dj['arm_a_over_production']}x), "
            f"ARM B time/byte {dj['arm_b_time_over_byte']}."
        ),
    ]
    if combined == VERDICT_ALU:
        lines.append(
            "344 GB/s is an arithmetic ceiling, not a memory ceiling. The lever "
            "is cheaper decode per byte, not fewer bytes and not better placement."
        )
    elif combined == VERDICT_MEM:
        lines.append(
            "Stripping decode+dequant+FMA did not lift GB/s. The executor is "
            "closed for this representation: 344 GB/s is a memory-system ceiling "
            "the LM head escapes because it streams a simpler format at much "
            "larger per-dispatch working set, but this pair says the arithmetic "
            "is not what holds these two kernels."
        )
    else:
        why = mj.get("why_not_forced") or dj.get("why_not_forced") or ""
        lines.append(
            "Pre-registered rule: MIXED. "
            + (why + ". " if why else "")
            + "ARM A landing near the LM-head rate with surviving loads is "
            "evidence that production arithmetic is expensive, but ARM B "
            "halves K (and therefore also halves FMAs), so linear time/byte "
            "is predicted by both ALU and bandwidth and cannot close the "
            "conjunction. Do not promote to ALU_BOUND. Do not rescue a "
            "refuted layout lever with this result."
        )
    return " ".join(lines)


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    for name in ORGANS:
        if name not in measurement:
            raise MissingArm(f"refusing a verdict: organ {name} is missing")
        judge_organ(measurement[name], name)
    verdict = judge(measurement)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/alu_roofline_organs.rs; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime); "
            "one representative layer of sealed-3.14, production MLP affine2 "
            "geo_tpr64 and DeltaNet Q4 geo_tpr64"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "production_cluster_gb_s": PRODUCTION_CLUSTER_GB_S,
        "jump_ratio": JUMP_RATIO,
        "stay_ratio": STAY_RATIO,
        "sublinear_slack": SUBLINEAR_SLACK,
        "absolute_gb_s_are_measured_under_load": True,
        "layer": measurement.get("layer"),
        "warmup": measurement.get("warmup"),
        "reps": measurement.get("reps"),
        "mlp": measurement["mlp"],
        "deltanet": measurement["deltanet"],
        "verdict": verdict,
        "finding": _finding({**measurement, "verdict": verdict}),
        "timing": measurement.get("timing"),
        "concurrent_load": measurement.get("concurrent_load"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "refuted_elsewhere": [
            "region_granularity",
            "catalog_addressing",
            "raw_dispatch_count",
            "fuse_representation_decode",
            "bytes_per_dispatch",
        ],
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise MissingArm("refusing to record a receipt without a measurement")
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
            lane="mlp_alu_roofline",
            # A receipt rebuilt from a stored raw capture must not stamp the
            # rebuild time as the measurement time. The raw files carry no
            # timestamp, so a retrofit records the measurement time as UNKNOWN.
            retrofit=not os.environ.get("HAWKING_MEASURED_NOW"),
        ),
    )
    write_measured_receipt(out, doc, "tools/future/mlp_alu_roofline.py")
    return out


def example_binaries() -> list[Path]:
    names = ("alu_roofline_organs",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "target",
            REPO / "workspace" / "ops" / "build" / "rust",
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
    warmup: int = 5,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "alu_roofline_organs binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example alu_roofline_organs`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{exe} exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/Users/scammermike/noetic/NOETIC_PARENT_A"),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
            layer=args.layer,
            warmup=args.warmup,
            reps=args.reps,
            out=RAW_DEFAULT,
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
        print(f"wrote {path} verdict={measured['verdict']}")
    else:
        print(f"verdict={measured['verdict']}")
        for name in ORGANS:
            o = measured[name]
            j = o["judgement"]
            print(
                f"  {name}: {o['verdict']}  prod {j['production_gb_s']}  "
                f"A {j['arm_a_gb_s']} ({j['arm_a_over_production']}x)  "
                f"B time/byte {j['arm_b_time_over_byte']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
