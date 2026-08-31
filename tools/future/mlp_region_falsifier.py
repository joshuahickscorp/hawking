#!/usr/bin/env python3
"""THE granularity falsifier: one MLP layer, contiguous, few regions.

Organ bandwidth left bytes-per-dispatch standing after dispatch COUNT was
refuted (b = -1.008 us). Three organs sit at ~350 GB/s; the LM head reaches
497.4 GB/s on one contiguous 337.7 MB GEMV. This is the cheapest falsifier:

    take one MLP layer
    pack gate/up/down into one contiguous staging buffer
    execute the SAME arithmetic in one serial compute region
    measure GB/s against the production scattered tensors

    If it rises past 430 GB/s: granularity is implicated.
    If it stays near ~350 GB/s: the hypothesis is dead.
    If the two arms are not bit-identical: refuse to emit a verdict.

    python3 tools/future/mlp_region_falsifier.py --record
    python3 tools/future/mlp_region_falsifier.py --from receipts/future/_MLP_REGION_FALSIFIER_raw.json --record
    python3 -m pytest tools/future/test_mlp_region_falsifier.py -q

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


RECEIPT = REPO / "receipts" / "future" / "MLP_REGION_FALSIFIER.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_MLP_REGION_FALSIFIER_raw.json"
SCHEMA = "hawking.future.mlp_region_falsifier.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_region_falsifier.py"

CLEAN_GEMV_GB_S = 703.5
LM_HEAD_GB_S = 497.4
# "materially toward the LM-head regime (say >430 GB/s)"
IMPLICATE_GB_S = 430.0
# "stays near ~350 GB/s"
REFUTE_CEILING_GB_S = 400.0
PRODUCTION_CLUSTER_GB_S = 350.0

CATALOG_BYTES_PER_LAYER = 27_853_103 * 3  # MLP_BYTE_CENSUS one layer gate+up+down
DEFAULT_LAYER = 0
DEFAULT_ARTIFACT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")

VERDICT_IMPLICATED = "GRANULARITY_IMPLICATED"
VERDICT_REFUTED = "GRANULARITY_REFUTED"

SUBFACTORS = (
    "contiguous_staging_buffer: gate/up/down codes+scales+biases packed into one MTLBuffer",
    "one_serial_compute_encoder: begin_serial_group covers gate+up+swiglu+down",
)

CLAIM_BOUNDARY = (
    "One MLP layer on sealed-3.14, DIAGNOSTIC_RELATIVE. GPU time is "
    "MTLCommandBuffer GPUStartTime/GPUEndTime for an isolated command buffer "
    "that contains only that layer's gate+up+SwiGLU+down. Bytes are the GPU-"
    "resident codes+scales+biases of those three tensors (catalog stored bytes "
    "are recorded alongside; they differ by the 303-byte HGRAVF01 headers). "
    "Bandwidth is those bytes divided by GPU ns, so it inherits the "
    "perfect-locality assumption. The two arms share kernels and arithmetic; "
    "the contiguous arm packs the nine weight arrays into one staging buffer "
    "and runs them in one serial compute encoder. A verdict is emitted only "
    "when max_abs_diff is exactly 0. This does not change the production "
    "decode path."
)


class ArmsNotIdentical(Exception):
    """Raised rather than emit a verdict over numerically divergent arms."""


class InconclusiveBandwidth(Exception):
    """Raised rather than force IMPLICATED/REFUTED on a mid-band result."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise ValueError("gpu_ns must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def judge(
    production_gb_s: float,
    contiguous_gb_s: float,
    max_abs_diff: float,
) -> str:
    """Return GRANULARITY_IMPLICATED or GRANULARITY_REFUTED, or refuse."""
    if max_abs_diff != 0.0:
        raise ArmsNotIdentical(
            f"refusing a verdict: arms are not bit-identical "
            f"(max_abs_diff={max_abs_diff})"
        )
    if contiguous_gb_s >= IMPLICATE_GB_S:
        return VERDICT_IMPLICATED
    if contiguous_gb_s < REFUTE_CEILING_GB_S:
        return VERDICT_REFUTED
    raise InconclusiveBandwidth(
        f"refusing a verdict: contiguous {contiguous_gb_s:.1f} GB/s sits "
        f"between the ~350 cluster (refute below {REFUTE_CEILING_GB_S}) and "
        f"the LM-head regime (implicate at {IMPLICATE_GB_S})"
    )


def _arm(raw: Mapping[str, Any], weight_bytes: int) -> dict[str, Any]:
    gpu_ns = int(raw["gpu_ns_median"])
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    return {
        "label": raw["label"],
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "dispatches": int(raw["dispatches"]),
        "encoders": int(raw["encoders"]),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "effective_gb_s": round(gb_s, 1),
        "share_of_clean_roof": round(gb_s / CLEAN_GEMV_GB_S, 4),
        "mb_per_dispatch": round(weight_bytes / 1e6 / max(int(raw["dispatches"]), 1), 1),
    }


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    weight_bytes = int(raw["weight_bytes"])
    max_abs_diff = float(raw["max_abs_diff"])
    production = _arm(raw["production"], weight_bytes)
    contiguous = _arm(raw["contiguous"], weight_bytes)
    verdict = judge(
        production["effective_gb_s"],
        contiguous["effective_gb_s"],
        max_abs_diff,
    )
    return {
        "layer": int(raw["layer"]),
        "kernel": raw["kernel"],
        "weight_bytes": weight_bytes,
        "catalog_bytes": int(raw.get("catalog_bytes", CATALOG_BYTES_PER_LAYER)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_gate": float(raw.get("max_abs_diff_gate", max_abs_diff)),
        "max_abs_diff_up": float(raw.get("max_abs_diff_up", max_abs_diff)),
        "max_abs_diff_act": float(raw.get("max_abs_diff_act", max_abs_diff)),
        "bit_identical": bool(raw.get("bit_identical", max_abs_diff == 0.0)),
        "subfactors_changed": list(raw.get("subfactors_changed", SUBFACTORS)),
        "production": production,
        "contiguous": contiguous,
        "verdict": verdict,
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "arithmetic": raw.get(
            "arithmetic",
            "F(x)=down(silu(gate(x))*up(x)) on a known x; identical kernels",
        ),
    }


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a receipt. Refuses a verdict if the arms diverge."""
    if not measurement.get("bit_identical", False):
        raise ArmsNotIdentical(
            "refusing a verdict: arms are not bit-identical "
            f"(max_abs_diff={measurement.get('max_abs_diff')})"
        )
    prod = measurement["production"]["effective_gb_s"]
    cont = measurement["contiguous"]["effective_gb_s"]
    verdict = judge(prod, cont, float(measurement["max_abs_diff"]))
    finding = {
        VERDICT_IMPLICATED: (
            "One layer's MLP, packed into one staging buffer and executed in "
            "one serial compute region, rose into the LM-head regime. Physical "
            "region granularity / fragmentation is implicated. Subfactors "
            "changed: contiguous staging of gate/up/down, and one serial "
            "compute encoder. Stop there."
        ),
        VERDICT_REFUTED: (
            "One layer's MLP, packed into one staging buffer and executed in "
            "one serial compute region, stayed near the ~350 GB/s cluster. "
            "The granularity / fragmentation hypothesis is DEAD. Do not "
            "rescue it with more packing variants."
        ),
    }[verdict]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "source": (
            "crates/hawking-core/examples/mlp_region_falsifier.rs via "
            "Qwen38HybridDecodeSession::measure_mlp_region_falsifier; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime)"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "implicate_gb_s": IMPLICATE_GB_S,
        "refute_ceiling_gb_s": REFUTE_CEILING_GB_S,
        "production_cluster_gb_s": PRODUCTION_CLUSTER_GB_S,
        "layer": measurement["layer"],
        "kernel": measurement["kernel"],
        "weight_bytes": measurement["weight_bytes"],
        "catalog_bytes": measurement["catalog_bytes"],
        "warmup": measurement["warmup"],
        "reps": measurement["reps"],
        "max_abs_diff": measurement["max_abs_diff"],
        "max_abs_diff_gate": measurement.get("max_abs_diff_gate"),
        "max_abs_diff_up": measurement.get("max_abs_diff_up"),
        "max_abs_diff_act": measurement.get("max_abs_diff_act"),
        "bit_identical": True,
        "subfactors_changed": list(measurement.get("subfactors_changed", SUBFACTORS)),
        "production": measurement["production"],
        "contiguous": measurement["contiguous"],
        "verdict": verdict,
        "finding": finding,
        "timing": measurement.get("timing"),
        "arithmetic": measurement.get("arithmetic"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise ArmsNotIdentical("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def example_binaries() -> list[Path]:
    names = ("mlp_region_falsifier",)
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
    layer: int = DEFAULT_LAYER,
    warmup: int = 5,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "mlp_region_falsifier binary not found; build with "
            "`cargo build --profile release -p hawking-core "
            "--example mlp_region_falsifier`"
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
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
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
        prod = measured["production"]
        cont = measured["contiguous"]
        print(
            f"layer {measured['layer']}  max_abs_diff={measured['max_abs_diff']}  "
            f"verdict={measured['verdict']}"
        )
        print(
            f"  production {prod['effective_gb_s']:.1f} GB/s  "
            f"{prod['gpu_ns_median']} ns  {prod['dispatches']} disp  "
            f"{prod['encoders']} enc"
        )
        print(
            f"  contiguous {cont['effective_gb_s']:.1f} GB/s  "
            f"{cont['gpu_ns_median']} ns  {cont['dispatches']} disp  "
            f"{cont['encoders']} enc"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
