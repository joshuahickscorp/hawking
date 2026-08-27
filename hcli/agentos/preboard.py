"""Negative-science census and FPGA/compiler preboard boundary.

The preboard is deliberately a planning interface.  It defines the contracts
needed by a future simulator, HWIR compiler, kernel genome, linker, partitioner,
cache, harness, and verifier without claiming that an FPGA backend or board is
present.  Existing negative receipts are first-class inputs to the plan.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.preboard.v1"
NEGATIVE_RECEIPTS = (
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    "receipts/headless/ACCELERATOR_SPARSE.json",
    "receipts/headless/ACCELERATOR_ORGAN_REPRESENTATION_FLOOR.json",
    "receipts/headless/ACCELERATOR_FP64_IS_A_HARDWARE_REFUSAL.json",
    "receipts/headless/QWEN_TUNE_SKIPPED.json",
)


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _receipt_summary(path: Path) -> Dict[str, Any]:
    value = _read(path)
    if value is None:
        return {"path": str(path), "present": False, "status": "ABSENT"}
    boundary = value.get("claim_boundary")
    if isinstance(boundary, list):
        boundary = boundary[:3]
    elif isinstance(boundary, str):
        boundary = boundary[:800]
    return {
        "path": str(path),
        "present": True,
        "schema": value.get("schema"),
        "pass": value.get("pass"),
        "status": value.get("status"),
        "verdict": value.get("verdict"),
        "headline": value.get("headline"),
        "next_action": value.get("next_action") or value.get("reopen_condition"),
        "claim_boundary": boundary,
    }


def _fpga_surface(repo: Path) -> Dict[str, Any]:
    paths = []
    try:
        result = subprocess.run(
            ["rg", "--files", str(repo / "hcli"), str(repo / "tools"), str(repo / "crates")],
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
        )
        for raw in (result.stdout or "").splitlines():
            lower = raw.lower()
            if any(token in lower for token in ("fpga", "hwir", "simulator", "verilog", "vhdl")):
                paths.append(raw)
                if len(paths) >= 100:
                    break
        status = "AVAILABLE" if result.returncode == 0 else "PARTIAL"
    except (OSError, subprocess.TimeoutExpired) as exc:
        status = "UNKNOWN"
        return {"status": status, "paths": [], "error": type(exc).__name__}
    return {
        "status": status,
        "candidate_source_paths": paths,
        "physical_board": {"status": "ABSENT", "claim": False, "reason": "no physical FPGA board is attached or claimed by this preboard"},
        "backend": {"status": "NOT_BUILT", "claim": False},
    }


def _preboard_plan() -> Dict[str, Any]:
    interfaces = [
        {"name": "kernel_genome", "schema": "hcli.fpga.kernel_genome.v1", "status": "INTERFACE_DEFINED", "fields": ["kernel_id", "input_layout", "output_layout", "dtype", "resources", "proof_hash"]},
        {"name": "hwir", "schema": "hcli.fpga.hwir.v1", "status": "INTERFACE_DEFINED", "fields": ["nodes", "buffers", "dependencies", "synchronization", "placement_constraints"]},
        {"name": "link", "schema": "hcli.fpga.link.v1", "status": "INTERFACE_DEFINED", "fields": ["symbols", "memory_regions", "relocations", "artifact_hash"]},
        {"name": "partitioner", "schema": "hcli.fpga.partitioner.v1", "status": "INTERFACE_DEFINED", "fields": ["source_graph", "partitions", "cut_bytes", "cut_synchronization", "device_assignment"]},
        {"name": "cache", "schema": "hcli.fpga.cache.v1", "status": "INTERFACE_DEFINED", "fields": ["content_address", "kernel_genome", "toolchain_identity", "target_identity", "verification"]},
        {"name": "simulator", "schema": "hcli.fpga.simulator.v1", "status": "CONTRACT_ONLY", "fields": ["hwir", "cycle_trace", "resource_usage", "numerical_output", "negative_controls"]},
        {"name": "harness", "schema": "hcli.fpga.harness.v1", "status": "CONTRACT_ONLY", "fields": ["input_fixture", "reference_output", "candidate_output", "tolerance", "trace"]},
        {"name": "verifier", "schema": "hcli.fpga.verifier.v1", "status": "INTERFACE_DEFINED", "fields": ["identity", "numerical_parity", "resource_bounds", "capability_contract", "receipt"]},
    ]
    return {
        "compiler_interfaces": interfaces,
        "hwir": {"status": "SCHEMA_ONLY", "nodes": [], "buffers": [], "dependencies": [], "synchronization": []},
        "simulator": {"status": "NOT_IMPLEMENTED", "physical_execution": False, "output_claim": "none"},
        "kernel_genome": {"status": "SCHEMA_ONLY", "physical_kernel": False},
        "linker": {"status": "SCHEMA_ONLY", "binary_generation": False},
        "partitioner": {"status": "SCHEMA_ONLY", "device_assignment": None},
        "cache": {"status": "SCHEMA_ONLY", "key": "sha256(canonical HWIR + kernel genome + target identity)"},
        "harness": {"status": "CONTRACT_ONLY", "capability_claim": False},
        "verifier": {"status": "CONTRACT_ONLY", "acceptance_authority": "deterministic verifier"},
        "fpga_backend": {"status": "NOT_BUILT", "physical_board_claim": False},
        "preboard_fingerprint": _digest(interfaces),
    }


def _write(report: Dict[str, Any], emit: Optional[str], repo: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / "HCLI_AGENTOS_PREBOARD.json"
    if not destination.is_absolute():
        destination = repo / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def run_preboard(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Census negative evidence and emit a no-board preboard plan."""
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    started = time.time()
    summaries = [_receipt_summary(repo / relative) for relative in NEGATIVE_RECEIPTS]
    present = [row for row in summaries if row.get("present")]
    plan = _preboard_plan()
    report = {
        "schema": SCHEMA,
        "status": "PASSED" if present else "FAILED",
        "qualification": "NEGATIVE_SCIENCE_CENSUS_AND_FPGA_PREBOARD_NO_PHYSICAL_BOARD_CLAIM",
        "started_at": started,
        "finished_at": time.time(),
        "elapsed_s": 0.0,
        "repo_root": str(repo),
        "negative_science": {
            "receipts": summaries,
            "present_count": len(present),
            "interpretation": "negative receipts constrain the next experiment; they do not certify impossibility outside their stated boundary",
        },
        "fpga_surface": _fpga_surface(repo),
        "preboard": plan,
        "checks": {
            "negative_science_receipts_censused": bool(present),
            "interfaces_defined": len(plan["compiler_interfaces"]) >= 8,
            "no_physical_board_claim": plan["fpga_backend"]["physical_board_claim"] is False,
            "no_fpga_backend_built_claim": plan["fpga_backend"]["status"] == "NOT_BUILT",
            "deterministic_preboard_fingerprint": bool(plan["preboard_fingerprint"]),
        },
        "claim_boundary": "This is a planning/preboard receipt. No FPGA board, FPGA backend, simulator timing, or hardware performance is claimed.",
    }
    report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    _write(report, str(emit) if emit is not None else None, repo)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_preboard(repo_root=args.repo_root, emit=args.emit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run_preboard"]
