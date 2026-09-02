"""HCLI-callable entry to the Apple Neural Engine probe lab.

Before this module the lab was human-shell-only: a person ran ``swiftc`` by
hand (``tools/accelerator/run_ane_lane.sh``) and read the JSON it wrote.
This wraps the same two public-API probes as an importable, testable Python
surface so HCLI code can discover devices, run the existing compiled
fixture, read its placement, and time ``prediction()`` without a human at a
shell.

It reuses ``tools/accelerator/ane_probe.swift`` (device discovery + a live
``MLComputePlan``) and ``tools/accelerator/ane_predict_probe.swift`` (timed
``prediction()`` plus a two-instance concurrent pair). It authors no new
graph and no new probe: it compiles and drives the scripts that already
exist, against the compiled fixture that already exists
(``workspace/ops/ane/python/coremltools/modelrunner/ModelRunner/add_model.mlmodelc``).

CRITICAL HONESTY CONSTRAINT: no ANE placement has ever been demonstrated on
this host. Every prior run of the fixture landed on CPU in every
MLComputePlan. This module reports ``MLComputePlan.deviceUsage`` exactly as
observed in the current run -- never assumed, never defaulted to ANE.
Requested ``MLComputeUnits`` select a candidate set; they are not a
placement. If the observed preferred device is CPU, the report says CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from hcli.persist import atomic_write_json


REPO = Path(__file__).resolve().parents[1]
ANE_PROBE_SWIFT = REPO / "tools" / "accelerator" / "ane_probe.swift"
ANE_PREDICT_PROBE_SWIFT = REPO / "tools" / "accelerator" / "ane_predict_probe.swift"
DEFAULT_FIXTURE = (
    REPO
    / "workspace"
    / "ops"
    / "ane"
    / "python"
    / "coremltools"
    / "modelrunner"
    / "ModelRunner"
    / "add_model.mlmodelc"
)
DEFAULT_SDK = "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk"
DEFAULT_EMIT = REPO / "receipts" / "headless" / "HCLI_FORBIDDEN_FRUIT_LAB.json"
SCHEMA = "hcli.forbidden_fruit_lab.v1"
DEFAULT_COMPUTE_UNITS = "all"
DEFAULT_REPEATS = 8
PUBLIC_COMPUTE_UNITS = frozenset({"cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"})

CLAIM_BOUNDARY = (
    "Public Core ML only. Placement is MLComputePlan.deviceUsage observed in "
    "this run, never inferred and never assumed from a requested compute "
    "unit. No ANE placement has been demonstrated on this host; when the "
    "observed preferred device is CPU, this receipt says CPU."
)


class ForbiddenFruitRefused(RuntimeError):
    """The lab refuses rather than guesses: missing toolchain, fixture, or a failed compile/run."""


def _run(argv: list[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def resolve_sdk(sdk: Optional[str] = None) -> str:
    """Same SDK convention as ``tools/accelerator/run_ane_lane.sh``: override or a pinned default."""
    candidate = sdk or os.environ.get("HAWKING_COREML_SDK") or DEFAULT_SDK
    if not Path(candidate).is_dir():
        raise ForbiddenFruitRefused(
            f"Core ML SDK not found at {candidate}; set HAWKING_COREML_SDK or pass sdk="
        )
    return candidate


def _compile(src: Path, sdk: str, module_cache: Path) -> Path:
    if not src.is_file():
        raise ForbiddenFruitRefused(f"probe source is missing: {src}")
    module_cache.mkdir(parents=True, exist_ok=True)
    binary = module_cache / src.stem
    argv = [
        "swiftc",
        "-module-cache-path", str(module_cache),
        "-parse-as-library",
        "-sdk", sdk,
        "-framework", "CoreML",
        str(src),
        "-o", str(binary),
    ]
    compiled = _run(argv, timeout=180)
    if compiled.returncode != 0 or not binary.is_file():
        err = (compiled.stderr or compiled.stdout or "swiftc produced no binary").strip()
        raise ForbiddenFruitRefused(f"swiftc failed compiling {src.name}: {err[:800]}")
    return binary


def discover_and_place(
    compiled_model: Optional[str] = None,
    *,
    sdk: Optional[str] = None,
    module_cache: Optional[Path] = None,
) -> dict[str, Any]:
    """Compile+run ane_probe.swift: device discovery, plus one live MLComputePlan if given a model."""
    resolved_sdk = resolve_sdk(sdk)
    cache = module_cache or Path(tempfile.mkdtemp(prefix="hawking-forbidden-fruit-cache-"))
    binary = _compile(ANE_PROBE_SWIFT, resolved_sdk, cache)
    out = Path(tempfile.mkdtemp(prefix="hawking-forbidden-fruit-run-")) / "device_profile.json"
    argv = [str(binary), str(out)]
    if compiled_model:
        argv.append(str(compiled_model))
    ran = _run(argv, timeout=60)
    if ran.returncode != 0 or not out.is_file():
        err = (ran.stderr or ran.stdout or "ane_probe produced no output").strip()
        raise ForbiddenFruitRefused(f"ane_probe.swift run failed: {err[:800]}")
    try:
        document = json.loads(out.read_text())
    except json.JSONDecodeError as exc:
        raise ForbiddenFruitRefused(f"ane_probe.swift wrote non-JSON output: {exc}") from exc
    if not isinstance(document, dict):
        raise ForbiddenFruitRefused("ane_probe.swift wrote a non-object JSON document")
    return document


def time_predict(
    compiled_model: str,
    *,
    compute_units: str = DEFAULT_COMPUTE_UNITS,
    repeats: int = DEFAULT_REPEATS,
    sdk: Optional[str] = None,
    module_cache: Optional[Path] = None,
) -> dict[str, Any]:
    """Compile+run ane_predict_probe.swift: timed prediction() plus one two-instance concurrent pair."""
    if compute_units not in PUBLIC_COMPUTE_UNITS:
        raise ForbiddenFruitRefused(
            f"compute_units {compute_units!r} is not a public MLComputeUnits name "
            f"({sorted(PUBLIC_COMPUTE_UNITS)})"
        )
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ForbiddenFruitRefused(f"repeats={repeats!r} must be a positive int")
    resolved_sdk = resolve_sdk(sdk)
    cache = module_cache or Path(tempfile.mkdtemp(prefix="hawking-forbidden-fruit-cache-"))
    binary = _compile(ANE_PREDICT_PROBE_SWIFT, resolved_sdk, cache)
    out = Path(tempfile.mkdtemp(prefix="hawking-forbidden-fruit-run-")) / "predict.json"
    argv = [str(binary), str(out), str(compiled_model), compute_units, str(repeats)]
    ran = _run(argv, timeout=120)
    if ran.returncode != 0 or not out.is_file():
        err = (ran.stderr or ran.stdout or "ane_predict_probe produced no output").strip()
        raise ForbiddenFruitRefused(f"ane_predict_probe.swift run failed: {err[:800]}")
    try:
        document = json.loads(out.read_text())
    except json.JSONDecodeError as exc:
        raise ForbiddenFruitRefused(f"ane_predict_probe.swift wrote non-JSON output: {exc}") from exc
    if not isinstance(document, dict):
        raise ForbiddenFruitRefused("ane_predict_probe.swift wrote a non-object JSON document")
    return document


def observed_placement(device_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a live ane_probe.swift document to an OBSERVED placement summary.

    ``preferred_devices_observed`` and ``ane_preferred_this_run`` describe
    only what THIS run's MLComputePlan reported. An empty/absent plan yields
    ``ane_preferred_this_run=False`` -- absence of evidence is not evidence
    of CPU either, so ``status`` carries the plan's own state through.
    """
    plan = device_profile.get("mlcomputeplan") if isinstance(device_profile, Mapping) else None
    plan = plan if isinstance(plan, Mapping) else {}
    ops = plan.get("operations") if isinstance(plan.get("operations"), list) else []
    preferred = [row.get("preferred") for row in ops if isinstance(row, Mapping)]
    return {
        "status": plan.get("status"),
        "api": plan.get("api"),
        "operations": ops,
        "preferred_devices_observed": preferred,
        "ane_preferred_this_run": "NEURAL_ENGINE" in preferred,
        "evidence_class": "PUBLIC_API_OBSERVED" if plan.get("status") == "PLANNED" else "NOT_MEASURED",
    }


def run_forbidden_fruit_lab(
    *,
    compiled_model: Optional[str | os.PathLike[str]] = None,
    compute_units: str = DEFAULT_COMPUTE_UNITS,
    repeats: int = DEFAULT_REPEATS,
    sdk: Optional[str] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """Minimum HCLI-callable ANE lab surface: discover, place, time, pair, receipt.

    Never raises for an expected refusal (missing fixture, missing
    toolchain, a failed compile or run); those are recorded as
    ``status: REFUSED`` in the returned/written receipt instead.
    """
    fixture = Path(compiled_model).expanduser() if compiled_model else DEFAULT_FIXTURE
    destination = Path(emit).expanduser() if emit else DEFAULT_EMIT
    if not destination.is_absolute():
        destination = REPO / destination
    started = time.time()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "public_api_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "reuses": [str(ANE_PROBE_SWIFT.relative_to(REPO)), str(ANE_PREDICT_PROBE_SWIFT.relative_to(REPO))],
        "started_at": started,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
        },
        "fixture": str(fixture),
        "compute_units_requested": compute_units,
        "requested_compute_units_are_not_placement": True,
        "repeats": repeats,
        "status": "RUNNING",
        "errors": [],
    }
    try:
        if not fixture.exists():
            raise ForbiddenFruitRefused(f"compiled fixture is not on disk: {fixture}")
        cache = Path(tempfile.mkdtemp(prefix="hawking-forbidden-fruit-cache-"))

        device_profile = discover_and_place(str(fixture), sdk=sdk, module_cache=cache)
        report["device_profile"] = device_profile
        report["neural_engine_present"] = bool(device_profile.get("neural_engine_present"))
        report["supported_compute_devices"] = device_profile.get("supported_compute_devices")
        report["placement"] = observed_placement(device_profile)

        predict = time_predict(
            str(fixture), compute_units=compute_units, repeats=repeats, sdk=sdk, module_cache=cache,
        )
        report["predict"] = predict
        report["timing_status"] = predict.get("status")
        report["concurrent_pair"] = predict.get("concurrent")

        report["ane_placement_observed_this_run"] = bool(report["placement"]["ane_preferred_this_run"])
        report["status"] = "PASSED"
    except ForbiddenFruitRefused as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        report["status"] = "REFUSED"
    except Exception as exc:  # noqa: BLE001 - persist the exact failure boundary
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:2000]})
        report["status"] = "FAILED"
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compiled-model", default=None, help=f"default: {DEFAULT_FIXTURE}")
    ap.add_argument("--compute-units", default=DEFAULT_COMPUTE_UNITS, choices=sorted(PUBLIC_COMPUTE_UNITS))
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--sdk", default=None)
    ap.add_argument("--emit", default=None)
    args = ap.parse_args(argv)
    report = run_forbidden_fruit_lab(
        compiled_model=args.compiled_model,
        compute_units=args.compute_units,
        repeats=args.repeats,
        sdk=args.sdk,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "ForbiddenFruitRefused",
    "discover_and_place",
    "observed_placement",
    "resolve_sdk",
    "run_forbidden_fruit_lab",
    "time_predict",
]


if __name__ == "__main__":
    raise SystemExit(main())
