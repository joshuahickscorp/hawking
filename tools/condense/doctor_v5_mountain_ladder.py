#!/usr/bin/env python3.12
"""Inert storage-aware campaign compiler for Doctor tiers above 120B.

The compiler carries the reviewed 120B experiment shape forward without
launching a worker, downloading a source, registering an adapter, or editing
the live Doctor queue.  It answers two separate questions for every horizon:

* can the source and one candidate coexist inside the disk lifecycle; and
* can the resulting candidate fit the 78 GB resident process envelope.

Every estimate is visibly provisional.  Exact source manifests, measured
shard peaks, physical output bytes, native parity, and quality receipts remain
mandatory before a future executor may be wired.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import sys
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_gptoss_parallel_scaffold as parallel


SCHEMA = "hawking.doctor_v5_post120_mountain_ladder.v1"
DEFAULT_OUTPUT = (
    ROOT / "reports/condense/doctor_v5_unbound/post120_mountain/mountain_ladder.json"
)
GB = 1_000_000_000
GIB = 1 << 30
DEFAULT_TOTAL_MEMORY_BYTES = 96 * GIB
DEFAULT_PROCESS_BUDGET_BYTES = 78 * GB
DEFAULT_DISK_RESERVE_BYTES = 150 * GB
DEFAULT_CACHE_RESERVE_BYTES = 32 * GB
DEFAULT_STREAM_WORKSPACE_BYTES = 32 * GB
DEFAULT_RUNTIME_WORKING_BYTES = 20 * GB
DEFAULT_ARTIFACT_OVERHEAD_PPM = 80_000  # 8%; planning reserve, not a codec claim.
THREAD_CANDIDATES = (8, 12, 16, 20)
QUEUE_DEPTH_CANDIDATES = (2, 3, 4, 6)
RATES = tuple(rate_id for rate_id, _rate in mxfp4.CANONICAL_RATES)
RATE_FRACTIONS = dict(mxfp4.CANONICAL_RATES)
BRANCHES = tuple(row[0] for row in parallel.BRANCHES)

# Repository sizes and terminal topology come from the already-reviewed Studio
# ladder.  They are planning observations, not immutable source authority.
MOUNTAINS = (
    {
        "label": "DeepSeek-V4-Flash",
        "nominal_parameters": 284_000_000_000,
        "observed_source_bytes": 168_000_000_000,
        "largest_observed_shard_bytes": None,
        "source_mode": "full-local-then-bounded-stream",
        "full_install_permitted": True,
        "priority_rates": ("2", "1", "0.8", "0.5"),
    },
    {
        "label": "Kimi-K2.6",
        "nominal_parameters": 1_100_000_000_000,
        "observed_source_bytes": 595_205_000_000,
        "largest_observed_shard_bytes": None,
        "source_mode": "guarded-full-local-then-expert-stream",
        "full_install_permitted": True,
        "priority_rates": ("0.8", "0.5", "0.33", "0.25"),
    },
    {
        "label": "DeepSeek-V4-Pro",
        "nominal_parameters": 1_600_000_000_000,
        "observed_source_bytes": 892_763_000_000,
        "largest_observed_shard_bytes": 14_100_000_000,
        "source_mode": "remote-immutable-range-stream-only",
        "full_install_permitted": False,
        "priority_rates": ("0.5", "0.33", "0.25"),
    },
)


class MountainLadderError(RuntimeError):
    """The inert mountain plan is inconsistent or accidentally executable."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": size}


def _tool_bindings() -> list[dict[str, Any]]:
    return [
        _artifact(Path(__file__)),
        _artifact(Path(mxfp4.__file__)),
        _artifact(Path(parallel.__file__)),
    ]


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _stamp(doc: dict[str, Any]) -> dict[str, Any]:
    if "plan_sha256" in doc:
        raise MountainLadderError("refusing to replace plan_sha256")
    doc["plan_sha256"] = _hash_value(doc)
    return doc


def _ceil_fraction(parameters: int, rate: Fraction) -> int:
    numerator = parameters * rate.numerator
    denominator = 8 * rate.denominator
    return (numerator + denominator - 1) // denominator


def _estimated_artifact_bytes(
    parameters: int, rate: Fraction, overhead_ppm: int,
) -> tuple[int, int]:
    nominal = _ceil_fraction(parameters, rate)
    installed = math.ceil(nominal * (1_000_000 + overhead_ppm) / 1_000_000)
    return nominal, installed


def hardware_snapshot(path: Path = ROOT) -> dict[str, Any]:
    """Read cheap local capacity only; no process, model, or network probing."""
    usage = shutil.disk_usage(path)
    return {
        "profile": "Studio-M3Ultra-96GB-1TB",
        "total_memory_bytes": DEFAULT_TOTAL_MEMORY_BYTES,
        "process_budget_bytes": DEFAULT_PROCESS_BUDGET_BYTES,
        "logical_cpu_count": os.cpu_count(),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "disk_reserve_bytes": DEFAULT_DISK_RESERVE_BYTES,
        "cache_reserve_bytes": DEFAULT_CACHE_RESERVE_BYTES,
        "stream_workspace_bytes": DEFAULT_STREAM_WORKSPACE_BYTES,
        "runtime_working_bytes": DEFAULT_RUNTIME_WORKING_BYTES,
        "artifact_overhead_ppm": DEFAULT_ARTIFACT_OVERHEAD_PPM,
    }


def _validate_hardware(hardware: dict[str, Any]) -> None:
    required = {
        "total_memory_bytes", "process_budget_bytes", "logical_cpu_count",
        "disk_total_bytes", "disk_free_bytes", "disk_reserve_bytes",
        "cache_reserve_bytes", "stream_workspace_bytes",
        "runtime_working_bytes", "artifact_overhead_ppm",
    }
    if not required.issubset(hardware):
        raise MountainLadderError("hardware snapshot is incomplete")
    for field in required:
        value = hardware[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MountainLadderError(f"hardware {field} is not a non-negative integer")
    if hardware["logical_cpu_count"] <= 0:
        raise MountainLadderError("logical_cpu_count must be positive")
    if hardware["process_budget_bytes"] > hardware["total_memory_bytes"]:
        raise MountainLadderError("process budget exceeds physical memory")
    if hardware["disk_free_bytes"] > hardware["disk_total_bytes"]:
        raise MountainLadderError("free disk exceeds total disk")


def _storage_projection(model: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    rates: list[dict[str, Any]] = []
    source = model["observed_source_bytes"]
    remote_window = model["largest_observed_shard_bytes"] or hardware["stream_workspace_bytes"]
    source_resident = source if model["full_install_permitted"] else remote_window
    fixed_disk = (
        hardware["disk_reserve_bytes"] + hardware["cache_reserve_bytes"]
        + hardware["stream_workspace_bytes"] + source_resident
    )
    for rate_id in RATES:
        nominal, installed = _estimated_artifact_bytes(
            model["nominal_parameters"], RATE_FRACTIONS[rate_id],
            hardware["artifact_overhead_ppm"],
        )
        disk_peak = fixed_disk + installed
        resident_peak = installed + hardware["runtime_working_bytes"]
        rates.append({
            "rate_id": rate_id,
            "nominal_weight_bytes": nominal,
            "planning_installed_bytes": installed,
            "planning_disk_peak_bytes": disk_peak,
            "fits_total_disk_lifecycle": disk_peak <= hardware["disk_total_bytes"],
            "fits_current_free_disk": disk_peak <= hardware["disk_free_bytes"],
            "additional_free_bytes_required_now": max(
                0, disk_peak - hardware["disk_free_bytes"]
            ),
            "planning_resident_peak_bytes": resident_peak,
            "fits_process_budget": resident_peak <= hardware["process_budget_bytes"],
            "measurement_status": "estimate-only-awaiting-physical-artifact-and-runtime-peak",
        })
    full_source_floor = (
        source + hardware["disk_reserve_bytes"] + hardware["cache_reserve_bytes"]
        + hardware["stream_workspace_bytes"]
    )
    return {
        "source_mode": model["source_mode"],
        "full_install_permitted": model["full_install_permitted"],
        "observed_source_bytes": source,
        "source_observation_is_authority": False,
        "full_source_floor_bytes": full_source_floor,
        "full_source_fits_total_disk_before_output": (
            model["full_install_permitted"]
            and full_source_floor <= hardware["disk_total_bytes"]
        ),
        "active_source_window_bytes": source_resident,
        "rates": rates,
    }


def _cell_templates(model: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for rate_id in RATES:
        for branch in BRANCHES:
            identity = {
                "model_label": model["label"], "rate_id": rate_id,
                "branch": branch, "source_manifest_sha256": None,
                "architecture_adapter_sha256": None,
            }
            cells.append({
                "cell_template_id": (
                    f"post120/{model['label']}/rate={rate_id}/branch={branch}"
                ),
                "identity_template_sha256": _hash_value(identity),
                **identity,
                "status": "blocked-unbound-template",
                "execution_permitted": False,
            })
    return cells


def _phase_templates(model: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = model["label"]
    phases = (
        ("architecture", (), "review and seal exact tensor/adapter/tokenizer authority"),
        ("source-manifest", ("architecture",), "seal every source/range/work-unit byte"),
        ("storage-canary", ("source-manifest",), "measure shard, output, RAM and swap peaks"),
        ("matrix-compile", ("storage-canary",), "compile exact 10-rate x 4-branch identities"),
        ("one-unit-fanout", ("matrix-compile",), "one source traversal fans out ten isolated rates"),
        ("physical-profile", ("one-unit-fanout",), "select threads/depth from exact-output A/B receipts"),
        ("progressive-coverage", ("physical-profile",), "1% -> 5% -> 20% -> 100% source coverage"),
        ("full-quality", ("progressive-coverage",), "zero-skip quality, native parity and capability"),
        ("promotion", ("full-quality",), "CAS promotion with rollback and lifecycle receipt"),
    )
    return [
        {
            "phase_id": f"{prefix}/{phase}",
            "depends_on": [f"{prefix}/{item}" for item in dependencies],
            "intent": intent,
            "status": "blocked-template",
            "command": None,
            "execution_permitted": False,
        }
        for phase, dependencies, intent in phases
    ]


def build_plan(
    hardware: dict[str, Any], *, created_at: str | None = None,
) -> dict[str, Any]:
    _validate_hardware(hardware)
    models: list[dict[str, Any]] = []
    prior = "doctor-v5-120b/final-accepted-checkpoint"
    for model in MOUNTAINS:
        cells = _cell_templates(model)
        phases = _phase_templates(model)
        phases[0]["depends_on"] = [prior]
        model_plan = {
            **model,
            "parameter_count_status": "nominal-until-operator-sealed-authority",
            "priority_rates": list(model["priority_rates"]),
            "experiment_matrix": {
                "rates": list(RATES), "branches": list(BRANCHES),
                "cell_templates": len(cells),
                "coverage_policy": (
                    "progressive coverage is an admission accelerator, never a substitute; "
                    "all 40 identities require terminal receipts for a full-matrix claim"
                ),
            },
            "cells": cells,
            "phases": phases,
            "storage": _storage_projection(model, hardware),
            "execution_permitted": False,
        }
        models.append(model_plan)
        prior = f"{model['label']}/promotion"

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": created_at or _now(),
        "status": "inert-fully-wired-templates-no-model-authority",
        "hardware": dict(hardware),
        "tool_bindings": _tool_bindings(),
        "carry_forward": {
            "from": "Doctor V5 under-120B plus isolated GPT-OSS 120B campaign",
            "contracts": [
                "exact 10-rate x 4-branch cell identity",
                "single bounded source traversal with ten isolated rate outputs",
                "per-rate/per-phase 8/12/16/20-thread physical profiles",
                "deterministic block parallelism and canonical merge",
                "ordered read/RHT/encode/write/attest overlap",
                "shared preprocess cache without shared scientific evidence",
                "dynamic RAM lanes, zero-swap admission and thermal stop",
                "immutable receipts, CAS generations, WAL recovery and rollback",
                "zero-skip quality, native parity and source-bound promotion",
            ],
            "deliberate_escalations": [
                "compile the next architecture and source manifests while the prior tier runs",
                "fan out every bounded unit across rates before releasing its staging window",
                "profile by phase and rate instead of applying one global thread count",
                "use progressive coverage to fail weak cells early while preserving all identities",
                "switch from full-local to immutable-range streaming at the disk cliff",
                "retain only source-bound promoted artifacts; source release is operator-gated",
            ],
        },
        "scheduler_template": {
            "thread_candidates": list(THREAD_CANDIDATES),
            "queue_depth_candidates": list(QUEUE_DEPTH_CANDIDATES),
            "axes": [
                "source_unit", "rate_fanout", "branch_isolation",
                "phase_overlap", "next_tier_metadata_preparation",
            ],
            "selection_authority": "physical exact-output receipts on the same host/generation",
            "default_selection": None,
            "execution_permitted": False,
        },
        "models": models,
        "coverage": {
            "model_templates": len(models),
            "cell_templates": sum(len(row["cells"]) for row in models),
            "phase_templates": sum(len(row["phases"]) for row in models),
            "largest_full_local_source": "Kimi-K2.6",
            "largest_parameter_target": "DeepSeek-V4-Pro",
        },
        "hard_invariants": {
            "network_calls": False,
            "heavy_compute": False,
            "worker_launches": False,
            "live_queue_mutations": False,
            "runtime_defaults_changed": False,
            "automatic_source_deletion": False,
            "estimated_bytes_are_release_authority": False,
            "deepseek_v4_pro_full_install_permitted": False,
        },
        "activation": {
            "execution_permitted": False,
            "required_before_future_wiring": [
                "accepted terminal 120B checkpoint",
                "operator-sealed exact architecture and parameter authority",
                "immutable source/range manifest and tokenizer binding",
                "measured disk/RAM/swap canary",
                "reviewed architecture worker and native runtime",
                "physical profile receipts and rollback point",
            ],
        },
    }
    return _stamp(doc)


def validate_plan(doc: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ["mountain ladder schema is invalid"]
    claimed = doc.get("plan_sha256")
    if not isinstance(claimed, str) or claimed != _hash_value(_without(doc, "plan_sha256")):
        errors.append("plan self-hash is invalid")
    try:
        _validate_hardware(doc.get("hardware", {}))
    except MountainLadderError as exc:
        errors.append(str(exc))
        return errors
    if doc.get("tool_bindings") != _tool_bindings():
        errors.append("compiler or inherited 120B tool bindings are stale")
    models = doc.get("models")
    if not isinstance(models, list) or [row.get("label") for row in models] != [
        row["label"] for row in MOUNTAINS
    ]:
        return errors + ["mountain model ordering or coverage differs"]
    expected_pairs = {(rate, branch) for rate in RATES for branch in BRANCHES}
    prior = "doctor-v5-120b/final-accepted-checkpoint"
    for expected, model in zip(MOUNTAINS, models):
        cells = model.get("cells")
        pairs = {
            (row.get("rate_id"), row.get("branch"))
            for row in cells if isinstance(row, dict)
        } if isinstance(cells, list) else set()
        if len(cells or []) != 40 or pairs != expected_pairs:
            errors.append(f"{expected['label']} exact 10x4 matrix differs")
        if cells != _cell_templates(expected):
            errors.append(f"{expected['label']} cell identity templates differ")
        if model.get("execution_permitted") is not False or any(
            row.get("execution_permitted") is not False for row in (cells or [])
        ):
            errors.append(f"{expected['label']} template became executable")
        phases = model.get("phases")
        expected_phases = _phase_templates(expected)
        expected_phases[0]["depends_on"] = [prior]
        if not isinstance(phases, list) or phases != expected_phases or any(
            row.get("execution_permitted") is not False or row.get("command") is not None
            for row in (phases or []) if isinstance(row, dict)
        ):
            errors.append(f"{expected['label']} phase template differs or is executable")
        fixed = {
            "label": expected["label"],
            "nominal_parameters": expected["nominal_parameters"],
            "observed_source_bytes": expected["observed_source_bytes"],
            "largest_observed_shard_bytes": expected["largest_observed_shard_bytes"],
            "source_mode": expected["source_mode"],
            "full_install_permitted": expected["full_install_permitted"],
            "priority_rates": list(expected["priority_rates"]),
        }
        if any(model.get(key) != value for key, value in fixed.items()):
            errors.append(f"{expected['label']} nominal horizon template differs")
        observed_storage = model.get("storage")
        recalculated = _storage_projection(expected, doc["hardware"])
        if observed_storage != recalculated:
            errors.append(f"{expected['label']} storage projection differs")
        prior = f"{expected['label']}/promotion"
    expected_invariants = {
        "network_calls": False,
        "heavy_compute": False,
        "worker_launches": False,
        "live_queue_mutations": False,
        "runtime_defaults_changed": False,
        "automatic_source_deletion": False,
        "estimated_bytes_are_release_authority": False,
        "deepseek_v4_pro_full_install_permitted": False,
    }
    if doc.get("hard_invariants") != expected_invariants:
        errors.append("hard inertness invariants differ")
    if doc.get("activation", {}).get("execution_permitted") is not False:
        errors.append("activation gate became executable")
    expected_coverage = {
        "model_templates": 3, "cell_templates": 120, "phase_templates": 27,
        "largest_full_local_source": "Kimi-K2.6",
        "largest_parameter_target": "DeepSeek-V4-Pro",
    }
    if doc.get("coverage") != expected_coverage:
        errors.append("coverage summary differs")
    return errors


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp, "x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MountainLadderError("plan JSON root is not an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write an inert plan from cheap capacity facts")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    inspect = sub.add_parser("inspect", help="print an inert plan without writing it")
    inspect.add_argument("--path", type=Path, default=ROOT)
    verify = sub.add_parser("verify", help="validate a previously written inert plan")
    verify.add_argument("--plan", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.command in {"build", "inspect"}:
        plan = build_plan(hardware_snapshot(getattr(args, "path", ROOT)))
        errors = validate_plan(plan)
        if errors:
            raise MountainLadderError("; ".join(errors))
        if args.command == "build":
            _atomic_json(args.output, plan)
            print(json.dumps({
                "ok": True, "path": str(args.output.resolve()),
                "plan_sha256": plan["plan_sha256"], "coverage": plan["coverage"],
            }, indent=2, sort_keys=True))
        else:
            print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    plan = _read_json(args.plan)
    errors = validate_plan(plan)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
