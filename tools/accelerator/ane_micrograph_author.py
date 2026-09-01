"""Build public Core ML MIL micrograph source descriptions for the ANE atlas.

The actual `.mlmodelc` compilation remains an Apple/Xcode operation.  This
module intentionally emits a deterministic MLProgram graph manifest when the
local Core ML authoring package is available and records the compiler boundary
when it is not.  It never asserts Neural Engine placement from graph syntax.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "receipts/headless/APPLE_ANE_ATLAS.json"

SHAPES = {
    "flash_hidden": [1, 2560],
    "flash_expert": [1, 640],
    "flash_qkv": [1, 12288],
    "flash_state": [1, 48, 128, 128],
    "flash_router": [1, 512],
    "flash_top_k": [1, 10],
}

OPS = (
    "matmul",
    "gemv",
    "gemm",
    "rmsnorm_like",
    "silu",
    "sigmoid",
    "multiply",
    "add",
    "softmax",
    "sdpa",
    "conv",
    "gather",
    "scatter",
    "top_k",
    "fused_projection_gate",
)


def _toolchain() -> dict[str, object]:
    try:
        import coremltools as ct  # type: ignore

        version = getattr(ct, "__version__", "unknown")
        authoring = True
        error = None
    except Exception as exc:  # pragma: no cover - machine dependent
        version = None
        authoring = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "coremltools_version": version,
        "coremltools_authoring_imported": authoring,
        "coremltools_error": error,
        "xcodebuild": "unavailable: active developer directory is CommandLineTools" if not (Path("/Applications/Xcode.app").exists()) else "present",
    }


def build() -> dict[str, object]:
    toolchain = _toolchain()
    graphs = []
    for op in OPS:
        shapes = [SHAPES["flash_hidden"]]
        if op in {"matmul", "gemm", "gemv", "fused_projection_gate"}:
            shapes = [SHAPES["flash_hidden"], SHAPES["flash_expert"], SHAPES["flash_qkv"]]
        elif op in {"sdpa"}:
            shapes = [SHAPES["flash_hidden"], [1, 24, 1, 256], [1, 2, 1, 256]]
        elif op in {"conv"}:
            shapes = [[1, 12288, 4], [1, 12288, 1]]
        elif op in {"gather", "scatter"}:
            shapes = [SHAPES["flash_router"], SHAPES["flash_top_k"]]
        elif op == "top_k":
            shapes = [SHAPES["flash_router"], SHAPES["flash_top_k"]]
        graphs.append({
            "id": f"ane_atlas_{op}",
            "operation": op,
            "mlprogram": True,
            "public_api": "Core ML MLProgram",
            "shapes": shapes,
            "compile": {"status": "NOT_RUN", "reason": "requires compiled .mlmodelc and Apple Core ML runtime"},
            "placement": {"status": "NOT_MEASURED", "preferred": None, "supported": []},
            "latency": {"cold_ns": None, "warm_ns": None, "throughput": None},
            "memory": {"delta_bytes": None},
            "energy": {"status": "NOT_TRUSTWORTHY", "joules": None},
        })
    body = {
        "schema": "hawking.apple_ane_atlas.v1",
        "status": "ATLAS_SCAFFOLD_COMPILE_BOUNDARY",
        "provider": "ANEProvider",
        "public_api_only": True,
        "toolchain": toolchain,
        "source": "Flash-Next Qwen4-Exp geometry",
        "shapes": SHAPES,
        "graphs": graphs,
        # This is an unmeasured compile scaffold, but it is still a receipt in
        # the corpus because it carries latency-shaped fields.  Keep S032's
        # machine/state requirement explicit without implying a benchmark.
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": "2026-08-28T00:00:00Z",
            "recorded_by": "ANE atlas author; no runtime benchmark executed",
            "machine": "Apple M3 Ultra, 60 GPU cores, 96 GiB (receipts/headless/MACHINE_GENOME.json)",
            "quiescence": None,
            "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet",
            "provenance": "compile/placement/latency fields are NOT_MEASURED",
        },
        "claim_boundary": "Graph manifests do not prove compilation, ANE support, placement, latency, energy, or Flash parity. Those fields require compiled MLProgram assets and MLComputePlan/runtime evidence.",
        "next_action": "Compile each graph with a full Xcode/Core ML toolchain, then run the Swift MLComputePlan inspector and public prediction timings.",
    }
    body["body_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n")
    return body


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
