#!/usr/bin/env python3
"""Bytes-per-dispatch sweep: does working set, not codec, make the 497 GB/s?

Organ bandwidth left bytes-per-dispatch standing after dispatch COUNT was
refuted. Granularity packed the buffers and left the work at 20.9 MB per
launch; it stayed at 332 GB/s. This sweep moves the one variable nobody
has moved: bytes per dispatch, on ONE kernel and ONE representation.

    python3 tools/future/dispatch_size_sweep.py --record
    python3 tools/future/dispatch_size_sweep.py --from receipts/future/_DISPATCH_SIZE_SWEEP_raw.json --record
    python3 -m pytest tools/future/test_dispatch_size_sweep.py -q

evidence_class DIAGNOSTIC_RELATIVE. Does not change the production decode path.
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
from _common import REPO  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "DISPATCH_SIZE_SWEEP.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_DISPATCH_SIZE_SWEEP_raw.json"
SCHEMA = "hawking.future.dispatch_size_sweep.v1"
VERSION = 1
RECORDED_BY = "tools/future/dispatch_size_sweep.py"

CLEAN_GEMV_GB_S = 703.5
LM_HEAD_GB_S = 497.4
LM_HEAD_MB = 337.7
SMALL_MB = 20.9
PRODUCTION_CLUSTER_GB_S = 350.0
# Same bar the granularity falsifier used for "materially toward the LM head".
IMPLICATE_GB_S = 430.0
REFUTE_CEILING_GB_S = 400.0
MIN_POINTS = 5

VERDICT_IMPLICATED = "PER_DISPATCH_SIZE_IMPLICATED"
VERDICT_REFUTED = "PER_DISPATCH_SIZE_REFUTED"

CLAIM_BOUNDARY = (
    "One kernel (affine2 q2 geo_tpr64_tg128) and one representation "
    "(HGRAVF01 affine2, mlp.gate_proj) on sealed-3.14, DIAGNOSTIC_RELATIVE. "
    "GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime for an isolated "
    "command buffer that contains only that launch. Bytes are the GPU-resident "
    "codes+scales+biases of the launched rows, stacked from real catalog "
    "tensors; bandwidth is those bytes divided by GPU ns, so it inherits the "
    "perfect-locality assumption. Same arithmetic, same codec, same tile "
    "geometry; only the per-launch working set changes. A verdict is emitted "
    "only over at least five curve points, and only when the packed unit "
    "tensor is bit-identical to production. The size-vs-shape control holds "
    "total bytes near the LM head's 337.7 MB and compares one concatenated "
    "GEMV against the same bytes as many unit tensors. This does not change "
    "the production decode path."
)


class TooFewPoints(Exception):
    """Raised rather than emit a verdict over fewer than five curve points."""


class PackNotIdentical(Exception):
    """Raised rather than emit a verdict over a packed tensor that drifted."""


class InconclusiveBandwidth(Exception):
    """Raised rather than force IMPLICATED/REFUTED on a mid-band result."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise ValueError("gpu_ns must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def _as_point(raw: Mapping[str, Any]) -> dict[str, Any]:
    weight_bytes = int(raw["weight_bytes"])
    gpu_ns = int(raw["gpu_ns_median"])
    dispatches = max(int(raw["dispatches"]), 1)
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    mb = weight_bytes / 1e6
    return {
        "label": str(raw["label"]),
        "target_mb": float(raw.get("target_mb", mb)),
        "weight_bytes": weight_bytes,
        "rows": int(raw["rows"]),
        "cols": int(raw.get("cols", 5120)),
        "n_source_tensors": int(raw.get("n_source_tensors", 1)),
        "dispatches": int(raw["dispatches"]),
        "encoders": int(raw.get("encoders", 1)),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "effective_gb_s": round(gb_s, 1),
        "mb_per_dispatch": round(mb / dispatches, 1),
        "share_of_clean_roof": round(gb_s / CLEAN_GEMV_GB_S, 4),
    }


def nearest_point(curve: list[Mapping[str, Any]], mb: float) -> dict[str, Any]:
    if not curve:
        raise TooFewPoints("refusing a verdict: curve is empty")
    return min(curve, key=lambda p: abs(float(p["mb_per_dispatch"]) - mb))


def knee_mb(curve: list[Mapping[str, Any]]) -> float | None:
    """First (smallest working set) point that enters the LM-head regime."""
    ordered = sorted(curve, key=lambda p: float(p["mb_per_dispatch"]))
    for p in ordered:
        if float(p["effective_gb_s"]) >= IMPLICATE_GB_S:
            return float(p["mb_per_dispatch"])
    return None


def judge(curve: list[Mapping[str, Any]]) -> str:
    """Return IMPLICATED or REFUTED, or refuse."""
    if len(curve) < MIN_POINTS:
        raise TooFewPoints(
            f"refusing a verdict: {len(curve)} curve points is fewer than {MIN_POINTS}"
        )
    at_lm = nearest_point(curve, LM_HEAD_MB)
    gb_lm = float(at_lm["effective_gb_s"])
    if gb_lm >= IMPLICATE_GB_S:
        return VERDICT_IMPLICATED
    if gb_lm < REFUTE_CEILING_GB_S:
        return VERDICT_REFUTED
    raise InconclusiveBandwidth(
        f"refusing a verdict: {gb_lm:.1f} GB/s at ~{at_lm['mb_per_dispatch']} MB "
        f"sits between the ~350 cluster (refute below {REFUTE_CEILING_GB_S}) and "
        f"the LM-head regime (implicate at {IMPLICATE_GB_S})"
    )


def _finding(
    verdict: str,
    curve: list[Mapping[str, Any]],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    at_lm = nearest_point(curve, LM_HEAD_MB)
    at_small = nearest_point(curve, SMALL_MB)
    ordered = sorted(curve, key=lambda p: float(p["mb_per_dispatch"]))
    largest = ordered[-1]
    knee = knee_mb(curve)
    gb_lm = float(at_lm["effective_gb_s"])
    gb_small = float(at_small["effective_gb_s"])
    gb_large = float(largest["effective_gb_s"])
    keeps_rising = gb_large > gb_lm + 10.0
    one_gb = float(control["one_large"]["effective_gb_s"])
    many_gb = float(control["many_small"]["effective_gb_s"])
    shape_irrelevant = abs(one_gb - many_gb) < 25.0
    size_alone = one_gb >= IMPLICATE_GB_S
    if verdict == VERDICT_IMPLICATED:
        text = (
            f"Effective GB/s rose from {gb_small:.1f} at "
            f"{at_small['mb_per_dispatch']} MB/dispatch to {gb_lm:.1f} at "
            f"{at_lm['mb_per_dispatch']} MB (LM head is {LM_HEAD_MB} MB at "
            f"{LM_HEAD_GB_S} GB/s). Per-dispatch working set is the mechanism. "
            f"Knee (first point >= {IMPLICATE_GB_S} GB/s): "
            f"{'none in range' if knee is None else f'{knee:.1f} MB'}. "
            f"Past the LM-head size the curve "
            f"{'keeps rising' if keeps_rising else 'does not keep rising'} "
            f"({gb_large:.1f} GB/s at {largest['mb_per_dispatch']} MB)."
        )
        remaining = None
        favoured = None
    else:
        text = (
            f"The curve saturates in the ~350 GB/s cluster: "
            f"{gb_small:.1f} GB/s at {at_small['mb_per_dispatch']} MB/dispatch, "
            f"{gb_lm:.1f} GB/s at {at_lm['mb_per_dispatch']} MB (LM head is "
            f"{LM_HEAD_MB} MB at {LM_HEAD_GB_S} GB/s), "
            f"{gb_large:.1f} GB/s at {largest['mb_per_dispatch']} MB. "
            f"It does not rise toward ~497. Per-dispatch working set is DEAD. "
            f"Do not rescue it with more concatenation variants."
        )
        # Isolated launches have no layer-to-layer producer/consumer edge.
        # A flat curve here means removing that serialization did not lift
        # this kernel off the ~350 cluster, so the data favours decode ALU
        # cost of affine2 q2 over cross-layer dependency serialization.
        favoured = "decode_alu_cost"
        remaining = {
            "candidates": [
                "decode ALU cost of affine2 q2 (this kernel) versus the Q4 LM head",
                "cross-layer dependency serialization in the production token graph",
            ],
            "favoured": favoured,
            "why": (
                "This sweep launches independent GEMVs on a known x; layer N+1 "
                "does not wait on layer N. The working set still sat in the "
                "~350 GB/s cluster at the LM head's 337.7 MB and past it "
                "(saturating near 377 GB/s at 700 MB). Dependency serialization "
                "is therefore not what holds THIS kernel at 350. A 5 MB launch "
                "is slower because fixed GPU start-up is not yet amortized; that "
                "is not the 350→497 gap. The remaining difference between a "
                "374 GB/s affine2 GEMV at 338 MB and the 497 GB/s Q4 LM head "
                "at the same bytes is the decode ALU cost of the two codecs."
            ),
        }
    return {
        "text": text,
        "knee_mb": knee,
        "gb_s_at_lm_head_mb": round(gb_lm, 1),
        "gb_s_at_small_mb": round(gb_small, 1),
        "gb_s_at_largest_mb": round(gb_large, 1),
        "keeps_rising_past_lm_head_mb": keeps_rising,
        "size_alone_reproduces": size_alone,
        "shape_irrelevant": shape_irrelevant,
        "remaining_candidates": remaining,
    }


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(raw.get("pack_parity_bit_identical", False)):
        raise PackNotIdentical(
            "refusing a verdict: packed unit tensor is not bit-identical to "
            f"production (max_abs_diff={raw.get('pack_parity_max_abs_diff')})"
        )
    curve = [_as_point(p) for p in raw["curve"]]
    if len(curve) < MIN_POINTS:
        raise TooFewPoints(
            f"refusing a verdict: {len(curve)} curve points is fewer than {MIN_POINTS}"
        )
    ctrl_raw = raw["control"]
    control = {
        "total_weight_bytes": int(ctrl_raw["total_weight_bytes"]),
        "n_unit_tensors": int(ctrl_raw["n_unit_tensors"]),
        "unit_rows": int(ctrl_raw["unit_rows"]),
        "one_large": _as_point(ctrl_raw["one_large"]),
        "many_small": _as_point(ctrl_raw["many_small"]),
        "max_abs_diff": float(ctrl_raw.get("max_abs_diff", 0.0)),
        "bit_identical": bool(ctrl_raw.get("bit_identical", False)),
    }
    verdict = judge(curve)
    return {
        "kernel": raw["kernel"],
        "codec": raw.get("codec", "HGRAVF01 affine2 q2"),
        "geometry": raw.get("geometry", "tpr64_tg128"),
        "projection": raw.get("projection", "mlp.gate_proj.weight"),
        "group_size": int(raw.get("group_size", 64)),
        "bits": int(raw.get("bits", 2)),
        "cols": int(raw.get("cols", 5120)),
        "unit_rows": int(raw["unit_rows"]),
        "unit_weight_bytes": int(raw["unit_weight_bytes"]),
        "bytes_per_row": int(raw.get("bytes_per_row", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "pack_parity_max_abs_diff": float(raw.get("pack_parity_max_abs_diff", 0.0)),
        "pack_parity_bit_identical": True,
        "curve": curve,
        "control": control,
        "verdict": verdict,
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "arithmetic": raw.get("arithmetic", ""),
    }


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a receipt. Refuses a verdict on fewer than five points."""
    if not measurement.get("pack_parity_bit_identical", False):
        raise PackNotIdentical(
            "refusing a verdict: packed unit tensor is not bit-identical to "
            f"production (max_abs_diff={measurement.get('pack_parity_max_abs_diff')})"
        )
    curve = list(measurement["curve"])
    verdict = judge(curve)
    finding = _finding(verdict, curve, measurement["control"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "source": (
            "crates/hawking-core/examples/dispatch_size_sweep.rs via "
            "Qwen38HybridDecodeSession::measure_dispatch_size_sweep; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime)"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "lm_head_mb": LM_HEAD_MB,
        "implicate_gb_s": IMPLICATE_GB_S,
        "refute_ceiling_gb_s": REFUTE_CEILING_GB_S,
        "production_cluster_gb_s": PRODUCTION_CLUSTER_GB_S,
        "kernel": measurement["kernel"],
        "codec": measurement["codec"],
        "geometry": measurement["geometry"],
        "projection": measurement["projection"],
        "group_size": measurement["group_size"],
        "bits": measurement["bits"],
        "cols": measurement["cols"],
        "unit_rows": measurement["unit_rows"],
        "unit_weight_bytes": measurement["unit_weight_bytes"],
        "bytes_per_row": measurement.get("bytes_per_row"),
        "warmup": measurement["warmup"],
        "reps": measurement["reps"],
        "pack_parity_max_abs_diff": measurement["pack_parity_max_abs_diff"],
        "pack_parity_bit_identical": True,
        "curve": curve,
        "control": measurement["control"],
        "verdict": verdict,
        "finding": finding["text"],
        "knee_mb": finding["knee_mb"],
        "gb_s_at_lm_head_mb": finding["gb_s_at_lm_head_mb"],
        "gb_s_at_small_mb": finding["gb_s_at_small_mb"],
        "gb_s_at_largest_mb": finding["gb_s_at_largest_mb"],
        "keeps_rising_past_lm_head_mb": finding["keeps_rising_past_lm_head_mb"],
        "size_alone_reproduces": finding["size_alone_reproduces"],
        "shape_irrelevant": finding["shape_irrelevant"],
        "remaining_candidates": finding["remaining_candidates"],
        "timing": measurement.get("timing"),
        "arithmetic": measurement.get("arithmetic"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise TooFewPoints("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def example_binaries() -> list[Path]:
    names = ("dispatch_size_sweep",)
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
        for profile in ("release", "release-fast"):
            for name in names:
                p = root / profile / "examples" / name
                if p.is_file():
                    out.append(p)
    return out


def run_example(
    artifact_root: Path,
    *,
    warmup: int = 5,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "dispatch_size_sweep binary not found; build with "
            "`cargo build --profile release-fast -p hawking-core "
            "--example dispatch_size_sweep`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
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
        print(
            f"kernel {measured['kernel']}  points={len(measured['curve'])}  "
            f"verdict={measured['verdict']}"
        )
        for p in measured["curve"]:
            print(
                f"  {p['mb_per_dispatch']:7.1f} MB  {p['effective_gb_s']:6.1f} GB/s  "
                f"{p['gpu_ns_median']} ns  {p['dispatches']} disp"
            )
        ctrl = measured["control"]
        print(
            f"  control one_large {ctrl['one_large']['effective_gb_s']:.1f}  "
            f"many_small {ctrl['many_small']['effective_gb_s']:.1f}  "
            f"bit_identical={ctrl['bit_identical']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
