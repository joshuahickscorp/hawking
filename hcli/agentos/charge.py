"""Persist the provider-neutral Hawking initial charge.

The charge is a durable Mission/DAG seed, not a second scheduler.  Its
WorkUnits deliberately carry science roles and resource classes rather than a
model identity.  A later AgentOS instance can recover the same queue and route
each unit through whichever provider is currently healthy.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.persist import atomic_write_json
from hcli.workunit import WorkUnit


SCHEMA = "hcli.agentos.initial_charge.v1"
CHARGE_ID = "HAWKING_INITIAL_CHARGE"
DEFAULT_WORKSPACE_NAME = ".hcli/initial-charge"


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _repo_root(value: Optional[str | os.PathLike[str]]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _evidence(repo: Path, relative_paths: Iterable[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for relative in relative_paths:
        path = (repo / relative).resolve()
        rows.append({
            "path": str(path),
            "relative_path": str(relative),
            "present": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        })
    return rows


def _verifier_for(evidence: list[Dict[str, Any]]) -> str:
    checks = []
    for row in evidence:
        path = row.get("path")
        digest = row.get("sha256")
        if path and digest:
            quoted = shlex.quote(str(path))
            checks.append(f"test -f {quoted} && shasum -a 256 {quoted} | grep -q {shlex.quote(str(digest))}")
    return " && ".join(checks) if checks else "test -f <dependency-evidence>"


def _specs(repo: Path) -> list[Dict[str, Any]]:
    """Return the initial queue with explicit policy for every unit."""
    return [
        {
            "id": "P0-A-modellake-supervision",
            "priority": "P0",
            "role": "model_lake_supervisor",
            "dependencies": [],
            "resource_class": "IO_HEAVY",
            "next_experiment": "Re-census the mounted ModelLake, then supervise the pinned Flash-Next acquisition with resumable hash verification and atomic publication.",
            "stop_condition": "Stop before deletion, capacity headroom below the protected threshold, hash mismatch, or an unrecognized acquisition process.",
            "allowed_mutations": ["ModelLake partial download and verification only", "receipts/headless/**"],
            "evidence_paths": ["receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json"],
            "required_verifier": "exact pinned revision, per-file size/hash, atomic final publication, and no protected-path mutation",
            "receipts": ["receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json"],
        },
        {
            "id": "P0-B-flash-architecture-census",
            "priority": "P0",
            "role": "architecture_science",
            "dependencies": [],
            "resource_class": "STATIC_ANALYSIS",
            "next_experiment": "Extend the pinned metadata organ graph as verified files arrive; keep bytes, active bytes/token, state bytes/token, FLOPs, and dispatches explicitly unknown until measured.",
            "stop_condition": "Stop on pinned identity mismatch or any attempt to infer physical execution from metadata alone.",
            "allowed_mutations": ["receipts/headless/**", "bounded metadata indexes"],
            "evidence_paths": ["receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"],
            "required_verifier": "pinned revision, architecture fingerprint, tensor-index identity, and explicit unmeasured physical fields",
            "receipts": ["receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"],
        },
        {
            "id": "P0-C-precedent-and-transfer-index",
            "priority": "P0",
            "role": "evidence_science",
            "dependencies": [],
            "resource_class": "CPU_HEAVY",
            "next_experiment": "Mine prior Hawking receipts into the Flash precedent map and the Qwen27-versus-Flash transfer map; receipts outrank prose summaries.",
            "stop_condition": "Stop on unreadable authority receipts or a classification that would turn an architecture-specific result into a universal law.",
            "allowed_mutations": ["receipts/headless/**"],
            "evidence_paths": ["receipts/headless/HCLI_ACCELERATOR_REGRESSION.json", "receipts/headless/NOETIC_DISPATCH_FUSION.json"],
            "required_verifier": "source receipt presence, source hash, classification boundary, and deterministic map fingerprint",
            "receipts": ["receipts/headless/HCLI_ACCELERATOR_REGRESSION.json", "receipts/headless/NOETIC_DISPATCH_FUSION.json"],
        },
        {
            "id": "P0-D-qwen27-regression",
            "priority": "P0",
            "role": "accelerator_science",
            "dependencies": ["P0-C-precedent-and-transfer-index"],
            "resource_class": "GPU_EXCLUSIVE",
            "next_experiment": "Use a protected, quiesced native A/B to explain the sealed resident's historical ~34 TPS versus current ~24 TPS, then test one measured complete-token hypothesis at a time.",
            "stop_condition": "Stop on fallback, capability deviation, contaminated benchmark state, missing identity, or any attempt to promote raw TPS as accepted TPS.",
            "allowed_mutations": ["temporary benchmark artifacts", "receipts/headless/**", "native kernel source only with an explicit candidate receipt"],
            "evidence_paths": ["receipts/headless/HCLI_ACCELERATOR_REGRESSION.json", "receipts/headless/NOETIC_DISPATCH_FUSION.json"],
            "required_verifier": "binary/artifact/tokenizer identity, complete wall/GPU timing, dispatches, fallbacks, capability, quiescence, and reproducible A/B receipt",
            "receipts": ["receipts/headless/HCLI_ACCELERATOR_REGRESSION.json"],
        },
        {
            "id": "P1-A-flash-gravity-candidates",
            "priority": "P1",
            "role": "representation_science",
            "dependencies": ["P0-A-modellake-supervision", "P0-B-flash-architecture-census", "P0-C-precedent-and-transfer-index"],
            "resource_class": "CPU_HEAVY",
            "next_experiment": "Search expert-axis sharing/bases/residuals, n-gram factorized lookup/generation, DeltaNet state representation, sparse attention routing, and MTP accounting.",
            "stop_condition": "Stop any candidate that omits required capability, hides MTP work, rematerializes a dense parent, or lacks a complete-system byte ledger.",
            "allowed_mutations": ["temporary candidate artifacts", "receipts/headless/**"],
            "evidence_paths": ["receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json", "receipts/headless/FLASH_NEXT_PRECEDENT_MAP.json"],
            "required_verifier": "complete-system EBPW ledger including all required byte categories and capability-preserving reference comparison",
            "receipts": ["receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"],
        },
        {
            "id": "P1-B-representation-native-kernels",
            "priority": "P1",
            "role": "kernel_engineering",
            "dependencies": ["P1-A-flash-gravity-candidates"],
            "resource_class": "COMPILE",
            "next_experiment": "Build native NF expert GEMV, router/gather, DeltaNet state/update, n-gram lookup, sparse attention, MTP, fused epilogue, and telemetry primitives shared where physically useful with Qwen27.",
            "stop_condition": "Stop on parity failure, hidden dense fallback, incomplete kernel genome, or a primitive-only win that loses complete-token work.",
            "allowed_mutations": ["native kernel source", "temporary build output", "receipts/headless/**"],
            "evidence_paths": ["receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json"],
            "required_verifier": "same-source parity, native executable identity, kernel genome, complete wall/GPU timing, dispatch/sync accounting",
            "receipts": [],
        },
        {
            "id": "P1-C-flash-pareto-ledger",
            "priority": "P1",
            "role": "promotion_verifier",
            "dependencies": ["P1-A-flash-gravity-candidates", "P1-B-representation-native-kernels"],
            "resource_class": "STATIC_ANALYSIS",
            "next_experiment": "Maintain the joint frontier over complete EBPW, hot/active/device bytes per token, FLOPs, dispatches, sync, complete ns, protected capability, and accepted TPS.",
            "stop_condition": "Never promote below 1.00 complete EBPW or below 50 accepted capability-preserving TPS; retain misses as Pareto intermediates.",
            "allowed_mutations": ["receipts/headless/**"],
            "evidence_paths": ["receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"],
            "required_verifier": "evaluate_flash_promotion with all byte fields, fallback disclosure, dense-vs-NF A/B, whole-model reference, and protected receipt",
            "receipts": [],
        },
        {
            "id": "P1-D-dense-vs-nf-organ-ab",
            "priority": "P1",
            "role": "benchmark_science",
            "dependencies": ["P1-C-flash-pareto-ledger"],
            "resource_class": "GPU_EXCLUSIVE",
            "next_experiment": "Run same-source/input/device/bench-state A/B for each serious organ, then repeat at whole-organ and whole-model scale when physically available.",
            "stop_condition": "Stop and record INCONCLUSIVE if any identity, input, device, state, or capability control is not matched.",
            "allowed_mutations": ["temporary benchmark artifacts", "receipts/headless/**"],
            "evidence_paths": ["receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json"],
            "required_verifier": "dense-vs-NF A/B protocol with representation bytes, bytes touched, GPU/wall ns, dispatches, deviation, capability, and complete-token verdict",
            "receipts": [],
        },
        {
            "id": "P1-E-fpga-both-models",
            "priority": "P1",
            "role": "fpga_preboard_engineering",
            "dependencies": ["P0-B-flash-architecture-census", "P0-C-precedent-and-transfer-index"],
            "resource_class": "COMPILE",
            "next_experiment": "Build HWIR, HBM, partition/link simulation, cache, harness, and verifier maps separately for Qwen27 and Flash-Next; simulation labels stay [S].",
            "stop_condition": "Stop at preboard boundaries; no physical board or U50 performance claim without hardware receipt.",
            "allowed_mutations": ["FPGA preboard source and schemas", "receipts/headless/**"],
            "evidence_paths": ["receipts/headless/HCLI_AGENTOS_PREBOARD.json", "receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"],
            "required_verifier": "both model maps, deterministic HWIR fingerprint, simulated transport receipt, and explicit no-board claim boundary",
            "receipts": [],
        },
        {
            "id": "P2-negative-science-doctor-vmcp-prep",
            "priority": "P2",
            "role": "evidence_science",
            "dependencies": ["P0-C-precedent-and-transfer-index"],
            "resource_class": "LIGHT_CONTROL",
            "next_experiment": "Use negative receipts to constrain the next experiment and prepare Doctor/VMCP evidence surfaces without touching protected systems or starting Odyssey.",
            "stop_condition": "Stop at research/evidence/preparation boundaries; do not declare impossibility from a bounded negative and do not promote Odyssey.",
            "allowed_mutations": ["receipts/headless/**"],
            "evidence_paths": ["receipts/headless/HCLI_AGENTOS_PREBOARD.json"],
            "required_verifier": "negative receipt census, explicit boundary, no VMCP/protected-system mutation, no Odyssey launch",
            "receipts": [],
        },
    ]


def _unit(spec: Mapping[str, Any], repo: Path, workspace: Path) -> tuple[WorkUnit, Dict[str, Any]]:
    evidence = _evidence(repo, spec.get("evidence_paths") or [])
    description = (
        f"{spec['next_experiment']} Required verifier: {spec['required_verifier']} "
        f"Stop condition: {spec['stop_condition']}"
    )
    wu = WorkUnit(
        id=str(spec["id"]),
        role=str(spec["role"]),
        description=description,
        dependencies=[str(item) for item in spec.get("dependencies") or []],
        resource_class=str(spec["resource_class"]),
        preferred_backend=None,
        provider=None,
        verifier=_verifier_for(evidence),
        effect_class="READ_ONLY" if str(spec["priority"]).startswith("P0") else "REVERSIBLE",
        workspace=str(workspace),
    )
    row = {
        **dict(spec),
        "evidence": evidence,
        "retry_state": {"attempts": 0, "max_attempts": 3, "policy": "rerun non-adoptable unit after verifier-visible failure"},
        "provider_neutral": wu.preferred_backend is None and wu.provider is None,
        "workunit": wu.to_dict(),
    }
    return wu, row


def create_initial_charge(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    workspace: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Create or idempotently recover the durable initial charge."""
    from hcli.agentos.runtime import AgentOS

    repo = _repo_root(repo_root)
    root = Path(workspace).expanduser().resolve() if workspace else (repo / DEFAULT_WORKSPACE_NAME).resolve()
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HAWKING_INITIAL_CHARGE.json"
    if destination.exists() and not force:
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("charge_id") == CHARGE_ID:
            state_path = Path(str(existing.get("mission_state_path") or ""))
            state = {}
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                state = {}
            state_units = state.get("units") if isinstance(state, dict) else {}
            for row in existing.get("units") or []:
                if not isinstance(row, dict):
                    continue
                row["evidence"] = _evidence(repo, row.get("evidence_paths") or [])
                payload = state_units.get(row.get("id")) if isinstance(state_units, dict) else None
                if isinstance(payload, dict):
                    row["workunit"] = payload
                    retry = dict(row.get("retry_state") or {})
                    retry.update({"attempts": payload.get("attempts", 0), "status": payload.get("status", "pending")})
                    row["retry_state"] = retry
            existing["status"] = "IDEMPOTENT_EXISTING_CHARGE"
            existing["refreshed_at"] = time.time()
            atomic_write_json(destination, existing)
            return existing

    root.mkdir(parents=True, exist_ok=True)
    specs = _specs(repo)
    units: Dict[str, WorkUnit] = {}
    rows: list[Dict[str, Any]] = []
    for spec in specs:
        workunit, row = _unit(spec, repo, root)
        units[workunit.id] = workunit
        rows.append(row)

    goal = (
        "HAWKING_INITIAL_CHARGE: execute provider-neutral Hawking science across "
        "Qwen3.8-27B control, Flash-Next Gravity/Accelerator, ModelLake, FPGA "
        "preboard, research/evidence/negative science, and HCLI self-repair only when a "
        "real failure exposes it. Disk state, typed tools, and fixed verifiers "
        "remain authoritative."
    )
    agent = AgentOS(root, engine=object(), repo_root=repo)
    mission = agent.start_mission(goal, units=units)
    agent.checkpoint()
    state_path = root / ".hcli" / "mission" / "state.json"
    report = {
        "schema": SCHEMA,
        "charge_id": CHARGE_ID,
        "status": "CREATED",
        "created_at": time.time(),
        "repo_root": str(repo),
        "workspace": str(root),
        "mission_id": mission.id,
        "mission_state_path": str(state_path),
        "goal": goal,
        "provider_neutral": True,
        "disk_is_authority": True,
        "units": rows,
        "background_missions": {
            "safe_default": [
                "research/receipt mining/architecture census/source verification/tests",
                "ModelLake supervision and verified resumable acquisition",
            ],
            "governed_exclusive": [
                "Qwen27 protected complete-token benchmark",
                "Flash dense-vs-NF organ/full-model A/B",
            ],
        },
        "retry_state": {
            "policy": "Mission/Scheduler retry and repair budgets remain authoritative",
            "reconstruct": f"AgentOS({root!s}).recover_mission()",
            "continue": f"AgentOS({root!s}).continue_mission()",
        },
        "receipts": sorted({path for row in rows for path in row.get("receipts") or []}),
        "next_action": "recover this Mission from mission_state_path, select the next dependency-ready WorkUnit, route by role/provider policy, and require its fixed verifier",
        "stop_conditions": [
            "verifier integrity failure",
            "canonical deletion or corruption request",
            "capacity headroom or protected-path violation",
            "capability regression or hidden fallback",
            "hardware/auth/payment escalation",
        ],
        "claim_boundary": "This is a durable provider-neutral work charge and queue seed. It does not certify model quality, accelerator performance, FPGA hardware, ModelLake completeness, or sovereignty.",
    }
    atomic_write_json(destination, report)
    report["receipt_path"] = str(destination.resolve())
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--workspace")
    parser.add_argument("--emit")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report = create_initial_charge(
        repo_root=args.repo_root,
        workspace=args.workspace,
        emit=args.emit,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


__all__ = ["CHARGE_ID", "SCHEMA", "create_initial_charge", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
