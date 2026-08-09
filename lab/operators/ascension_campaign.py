"""Persistent, fail-closed supervisor for the complete Ascension V3 campaign.

This is the long-running coordinator requested by Bible §17.  On each tick it
re-evaluates every receipt-bound lifecycle state, maintains a direct map of all
48 Bible §18 steps, refreshes only credential-safe *source metadata* when it
is stale, and writes a durable execution manifest.  It intentionally does not
start a model body, download weights, fabricate evidence, select a manager,
evict an alternate, or activate the production sandbox.

That boundary is what makes a detached process safe to leave running for weeks:
it will keep the whole programme wired and ready to transition immediately
when protected evidence arrives, without treating its own liveness as a gate.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from lab.operators.ascension_execution_plan import audit_execution_sequence, execution_rows
from lab.operators.ascension_foundation_contracts import FILENAME as FOUNDATION_CONTRACTS_FILENAME
from lab.operators.ascension_foundation_contracts import write_foundation_contracts
from lab.operators.ascension_family_workflow import FILENAME as FAMILY_WORKFLOW_FILENAME
from lab.operators.ascension_family_workflow import write_family_workflow
from lab.operators.ascension_kernel_registry import FILENAME as KERNEL_CONTRACT_FILENAME
from lab.operators.ascension_kernel_registry import write_kernel_compiler_contract
from lab.operators.ascension_knowledge_contract import FILENAME as KNOWLEDGE_CONTRACT_FILENAME
from lab.operators.ascension_knowledge_contract import write_knowledge_plane_contract
from lab.operators.ascension_manager_workflow import FILENAME as MANAGER_WORKFLOW_FILENAME
from lab.operators.ascension_manager_workflow import write_dual_manager_workflow
from lab.operators.ascension_release_workflow import FILENAME as RELEASE_WORKFLOW_FILENAME
from lab.operators.ascension_release_workflow import write_release_workflow
from lab.operators.ascension_lifecycle import (
    DEFAULT_BIBLE,
    DEFAULT_ROOT,
    LifecyclePaths,
    evaluate_lifecycle,
)
from lab.operators.ascension_source_admission import capture_all_sources
from lab.operators.ascension_tournament_workflow import FILENAME as TOURNAMENT_WORKFLOW_FILENAME
from lab.operators.ascension_tournament_workflow import write_tournament_workflow
from lab.receipts import seal


SCHEMA = "hawking.ascension.v3_campaign_supervisor.v1"
MANIFEST_SCHEMA = "hawking.ascension.v3_execution_manifest.v1"
DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_METADATA_REFRESH_SECONDS = 24 * 60 * 60.0


class AscensionCampaignError(RuntimeError):
    """The detached campaign controller cannot safely preserve its state."""


@dataclass(frozen=True)
class CampaignPaths:
    """Files owned by this supervisor, kept separate from intake evidence."""

    lifecycle_root: Path
    manifest_path: Path
    status_path: Path
    kernel_contract_path: Path
    manager_workflow_path: Path
    family_workflow_path: Path
    tournament_workflow_path: Path
    release_workflow_path: Path
    foundation_contracts_path: Path
    knowledge_contract_path: Path
    lock_path: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "CampaignPaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            lifecycle_root=resolved,
            manifest_path=resolved / "ASCENSION_V3_EXECUTION_MANIFEST.json",
            status_path=resolved / "ASCENSION_V3_CAMPAIGN_SUPERVISOR.json",
            kernel_contract_path=resolved / KERNEL_CONTRACT_FILENAME,
            manager_workflow_path=resolved / MANAGER_WORKFLOW_FILENAME,
            family_workflow_path=resolved / FAMILY_WORKFLOW_FILENAME,
            tournament_workflow_path=resolved / TOURNAMENT_WORKFLOW_FILENAME,
            release_workflow_path=resolved / RELEASE_WORKFLOW_FILENAME,
            foundation_contracts_path=resolved / FOUNDATION_CONTRACTS_FILENAME,
            knowledge_contract_path=resolved / KNOWLEDGE_CONTRACT_FILENAME,
            lock_path=resolved / ".campaign-supervisor.lock",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _ensure_real_directory(path: Path, mode: int = 0o750) -> None:
    path.mkdir(parents=True, exist_ok=True)
    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise AscensionCampaignError(f"campaign path must be a real directory: {path}")
    os.chmod(path, mode)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _source_metadata_status(lifecycle_root: Path) -> dict[str, Any]:
    path = lifecycle_root / "source-admission" / "SOURCE_ADMISSION_STATUS.json"
    document = _read_json(path)
    if document is None:
        return {
            "path": str(path),
            "present": False,
            "status": "ABSENT",
            "candidate_only": True,
        }
    records = document.get("records") if isinstance(document.get("records"), list) else []
    return {
        "path": str(path),
        "present": True,
        "status": document.get("status"),
        "recorded_at": document.get("recorded_at"),
        "records": [
            {
                "artifact_id": item.get("artifact_id"),
                "status": item.get("status"),
                "repository": item.get("repository"),
                "revision": item.get("revision"),
                "no_model_body_downloaded": item.get("no_model_body_downloaded"),
            }
            for item in records
            if isinstance(item, Mapping)
        ],
        "candidate_only": True,
        "not_lifecycle_certification": True,
    }


def _metadata_refresh_due(lifecycle_root: Path, refresh_seconds: float) -> bool:
    status = _source_metadata_status(lifecycle_root)
    recorded = _parse_timestamp(status.get("recorded_at"))
    if status.get("status") != "ALL_METADATA_CAPTURED" or recorded is None:
        return True
    age = datetime.now(timezone.utc) - recorded.astimezone(timezone.utc)
    return age.total_seconds() >= refresh_seconds


def _state_map(lifecycle_root: Path) -> dict[str, dict[str, Any]]:
    document = _read_json(LifecyclePaths.from_root(lifecycle_root).state_path) or {}
    values = document.get("states") if isinstance(document.get("states"), list) else []
    return {
        str(value["id"]): dict(value)
        for value in values
        if isinstance(value, Mapping) and isinstance(value.get("id"), str)
    }


def _step_operating_status(row: Mapping[str, Any]) -> str:
    if row.get("evidence_complete") is True:
        return "EVIDENCE_COMPLETE"
    dispatch_class = row.get("dispatch_class")
    if dispatch_class == "POST_RELEASE_SEPARATE_PROGRAMME":
        return "OUTSIDE_LAUNCH_CRITICAL_PATH"
    if dispatch_class == "SAFE_CONTROLLER_AUDIT_OR_METADATA":
        return "SAFE_CONTROLLER_MAINTENANCE_ONLY"
    if row.get("admissible_for_evidence_intake") is True:
        return "READY_FOR_PROTECTED_EVIDENCE"
    return "WAITING_FOR_UPSTREAM_CERTIFICATION"


def _execution_manifest(
    *,
    lifecycle_root: Path,
    bible_path: Path,
    lifecycle: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    kernel_contract: Mapping[str, Any],
    manager_workflow: Mapping[str, Any],
    family_workflow: Mapping[str, Any],
    tournament_workflow: Mapping[str, Any],
    release_workflow: Mapping[str, Any],
    foundation_contracts: Mapping[str, Any],
    knowledge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    states = _state_map(lifecycle_root)
    sequence_audit = audit_execution_sequence(bible_path)
    rows = execution_rows(states)
    for row in rows:
        row["operating_status"] = _step_operating_status(row)
    exact_step_coverage = sum(1 for row in rows if row.get("states"))
    safe_automatic_lanes = [
        {
            "lane": "lifecycle-reconciliation",
            "status": "RUNNING",
            "action": "re-evaluate sealed controller intake and derive continuation outputs",
        },
        {
            "lane": "source-metadata-admission",
            "status": source_metadata.get("status"),
            "action": "authenticated metadata refresh only; never model-body transfer",
        },
        {
            "lane": "manager-tournament-controller",
            "status": (lifecycle.get("tournament") or {}).get("status"),
            "action": "preserve fixed candidates/dimensions; never auto-score or select a winner",
        },
        {
            "lane": "exact-model-kernel-registry",
            "status": kernel_contract.get("status"),
            "action": "preserve every required family plugin and Gravity representation class; never call declaration a compiled kernel",
        },
        {
            "lane": "dual-manager-handoff",
            "status": (manager_workflow.get("handoff") or {}).get("tournament_phase"),
            "action": "preserve 30B then 80B qualification order and protected tournament handoff",
        },
        {
            "lane": "family-matrix-workflow",
            "status": (family_workflow.get("sandbox_dependency") or {}).get("status"),
            "action": "preserve all eight families and generic reference as distinct launch obligations",
        },
        {
            "lane": "protected-tournament-post-winner-workflow",
            "status": tournament_workflow.get("runtime_phase"),
            "action": "preserve frozen dimensions, winner authority, alternate offload, and sandbox-start fence",
        },
        {
            "lane": "global-review-and-apple-release-workflow",
            "status": (release_workflow.get("global_audit") or {}).get("status"),
            "action": "preserve all audit, external review, release, and post-release requirements without exceptions",
        },
        {
            "lane": "build-fabric-resource-agent-os-contracts",
            "status": foundation_contracts.get("status"),
            "action": "preserve full lane, resource, and Agent OS contracts for measured evidence intake",
        },
        {
            "lane": "knowledge-plane-contract",
            "status": knowledge_contract.get("status"),
            "action": "preserve Kernel, Representation, Scheduler, negative-science, transfer, and index schemas",
        },
    ]
    return seal(
        {
            "schema": MANIFEST_SCHEMA,
            "recorded_at": _utc_now(),
            "bible_execution_sequence": sequence_audit,
            "coverage": {
                "bible_step_rows": len(rows),
                "required_bible_step_rows": 48,
                "all_bible_steps_wired": len(rows) == 48 and exact_step_coverage == 48 and sequence_audit.get("matches") is True,
                "receipt_complete_rows": sum(1 for row in rows if row.get("evidence_complete") is True),
                "receipt_completion_is_not_wiring_coverage": True,
            },
            "lifecycle": {
                "root": str(lifecycle_root),
                "first_unmet_state": lifecycle.get("first_unmet_state"),
                "state_counts": lifecycle.get("state_counts"),
                "tournament": lifecycle.get("tournament"),
                "launch_gate": lifecycle.get("launch_gate"),
            },
            "source_metadata": dict(source_metadata),
            "kernel_compiler_contract": {
                "path": str(lifecycle_root / KERNEL_CONTRACT_FILENAME),
                "status": kernel_contract.get("status"),
                "seal_sha256": kernel_contract.get("seal_sha256"),
                "configuration_only": kernel_contract.get("status") == "CONTROLLER_CONFIGURATION_ONLY",
            },
            "dual_manager_workflow": {
                "path": str(lifecycle_root / MANAGER_WORKFLOW_FILENAME),
                "status": manager_workflow.get("status"),
                "handoff": manager_workflow.get("handoff"),
                "configuration_only": manager_workflow.get("status") == "CONTROLLER_WORKFLOW_ONLY",
            },
            "family_campaign_workflow": {
                "path": str(lifecycle_root / FAMILY_WORKFLOW_FILENAME),
                "status": family_workflow.get("status"),
                "matrix_handoff": family_workflow.get("matrix_handoff"),
                "configuration_only": family_workflow.get("status") == "CONTROLLER_WORKFLOW_ONLY",
            },
            "manager_tournament_workflow": {
                "path": str(lifecycle_root / TOURNAMENT_WORKFLOW_FILENAME),
                "status": tournament_workflow.get("status"),
                "runtime_phase": tournament_workflow.get("runtime_phase"),
                "configuration_only": tournament_workflow.get("status") == "CONTROLLER_WORKFLOW_ONLY",
            },
            "global_release_workflow": {
                "path": str(lifecycle_root / RELEASE_WORKFLOW_FILENAME),
                "status": release_workflow.get("status"),
                "derived_launch_gate": release_workflow.get("derived_launch_gate"),
                "configuration_only": release_workflow.get("status") == "CONTROLLER_WORKFLOW_ONLY",
            },
            "foundation_contracts": {
                "path": str(lifecycle_root / FOUNDATION_CONTRACTS_FILENAME),
                "status": foundation_contracts.get("status"),
                "configuration_only": foundation_contracts.get("status") == "CONTROLLER_CONFIGURATION_ONLY",
            },
            "knowledge_plane_contract": {
                "path": str(lifecycle_root / KNOWLEDGE_CONTRACT_FILENAME),
                "status": knowledge_contract.get("status"),
                "configuration_only": knowledge_contract.get("status") == "CONTROLLER_CONFIGURATION_ONLY",
            },
            "safe_automatic_lanes": safe_automatic_lanes,
            "steps": rows,
            "claim_boundary": {
                "manifest_is_not_receipt_certification": True,
                "no_model_body_downloaded_by_supervisor": True,
                "no_model_runtime_started_by_supervisor": True,
                "no_winner_auto_selected": True,
                "no_alternate_evicted_by_supervisor": True,
                "no_production_sandbox_activated_by_supervisor": True,
                "token_material_is_never_recorded": True,
            },
        }
    )


def bootstrap_layout(
    root: str | Path = DEFAULT_ROOT, *, bible_path: str | Path = DEFAULT_BIBLE
) -> CampaignPaths:
    """Create only the supervisor-owned directory; never fabricate evidence."""

    paths = CampaignPaths.from_root(root)
    _ensure_real_directory(paths.lifecycle_root)
    return paths


def tick(
    root: str | Path = DEFAULT_ROOT,
    *,
    bible_path: str | Path = DEFAULT_BIBLE,
    metadata_refresh_seconds: float = DEFAULT_METADATA_REFRESH_SECONDS,
    force_metadata_refresh: bool = False,
    capture_sources_fn: Callable[[str | Path], Mapping[str, Any]] = capture_all_sources,
) -> dict[str, Any]:
    """Run one restart-safe controller reconciliation tick.

    ``capture_sources_fn`` is injectable solely for deterministic tests.  The
    production default is the credential-safe metadata collector; it receives
    the lifecycle root and cannot turn a candidate record into certification.
    """

    if metadata_refresh_seconds < 60.0:
        raise AscensionCampaignError("metadata_refresh_seconds must be at least 60")
    paths = bootstrap_layout(root, bible_path=bible_path)
    previous = _read_json(paths.status_path) or {}
    lifecycle = evaluate_lifecycle(paths.lifecycle_root, bible_path=bible_path)
    lifecycle_state = _read_json(LifecyclePaths.from_root(paths.lifecycle_root).state_path) or {}
    bible_info = lifecycle_state.get("bible") if isinstance(lifecycle_state.get("bible"), Mapping) else {}
    kernel_contract = write_kernel_compiler_contract(
        paths.lifecycle_root,
        bible_sha256=bible_info.get("sha256") if isinstance(bible_info.get("sha256"), str) else None,
    )
    foundation_contracts = write_foundation_contracts(
        paths.lifecycle_root,
        bible_sha256=bible_info.get("sha256") if isinstance(bible_info.get("sha256"), str) else None,
    )
    knowledge_contract = write_knowledge_plane_contract(
        paths.lifecycle_root,
        bible_sha256=bible_info.get("sha256") if isinstance(bible_info.get("sha256"), str) else None,
    )
    refreshed = False
    refresh_error: dict[str, str] | None = None
    if force_metadata_refresh or _metadata_refresh_due(paths.lifecycle_root, metadata_refresh_seconds):
        try:
            capture_sources_fn(paths.lifecycle_root)
            refreshed = True
        except Exception as exc:  # Keep the long-running supervisor alive and observable.
            # Do not serialize transport exception text: libraries sometimes
            # include a signed URL or credential-adjacent request detail.
            refresh_error = {"type": type(exc).__name__, "message": "metadata refresh failed"}
    source_metadata = _source_metadata_status(paths.lifecycle_root)
    manager_workflow = write_dual_manager_workflow(
        paths.lifecycle_root, states=_state_map(paths.lifecycle_root)
    )
    family_workflow = write_family_workflow(
        paths.lifecycle_root, states=_state_map(paths.lifecycle_root)
    )
    tournament_workflow = write_tournament_workflow(
        paths.lifecycle_root,
        states=_state_map(paths.lifecycle_root),
        tournament=lifecycle.get("tournament") if isinstance(lifecycle.get("tournament"), Mapping) else {},
    )
    release_workflow = write_release_workflow(
        paths.lifecycle_root,
        states=_state_map(paths.lifecycle_root),
        launch_gate=lifecycle.get("launch_gate") if isinstance(lifecycle.get("launch_gate"), Mapping) else {},
    )
    manifest = _execution_manifest(
        lifecycle_root=paths.lifecycle_root,
        bible_path=Path(bible_path).expanduser().resolve(),
        lifecycle=lifecycle,
        source_metadata=source_metadata,
        kernel_contract=kernel_contract,
        manager_workflow=manager_workflow,
        family_workflow=family_workflow,
        tournament_workflow=tournament_workflow,
        release_workflow=release_workflow,
        foundation_contracts=foundation_contracts,
        knowledge_contract=knowledge_contract,
    )
    _atomic_json(paths.manifest_path, manifest)
    prior_heartbeat = previous.get("heartbeat")
    heartbeat = int(prior_heartbeat) + 1 if isinstance(prior_heartbeat, int) else 1
    status = seal(
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "state": "RUNNING_DEGRADED" if refresh_error else "RUNNING",
            "heartbeat": heartbeat,
            "process": {
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "process_group": os.getpgrp(),
                "python_executable": sys.executable,
            },
            "paths": {
                "lifecycle_root": str(paths.lifecycle_root),
                "manifest": str(paths.manifest_path),
                "status": str(paths.status_path),
            },
            "lifecycle": {
                "first_unmet_state": lifecycle.get("first_unmet_state"),
                "state_counts": lifecycle.get("state_counts"),
                "tournament": lifecycle.get("tournament"),
                "launch_gate": lifecycle.get("launch_gate"),
            },
            "source_metadata": source_metadata,
            "kernel_compiler_contract": {
                "path": str(paths.kernel_contract_path),
                "status": kernel_contract.get("status"),
                "seal_sha256": kernel_contract.get("seal_sha256"),
                "configuration_only": True,
            },
            "dual_manager_workflow": {
                "path": str(paths.manager_workflow_path),
                "status": manager_workflow.get("status"),
                "handoff": manager_workflow.get("handoff"),
                "configuration_only": True,
            },
            "family_campaign_workflow": {
                "path": str(paths.family_workflow_path),
                "status": family_workflow.get("status"),
                "matrix_handoff": family_workflow.get("matrix_handoff"),
                "configuration_only": True,
            },
            "manager_tournament_workflow": {
                "path": str(paths.tournament_workflow_path),
                "status": tournament_workflow.get("status"),
                "runtime_phase": tournament_workflow.get("runtime_phase"),
                "configuration_only": True,
            },
            "global_release_workflow": {
                "path": str(paths.release_workflow_path),
                "status": release_workflow.get("status"),
                "derived_launch_gate": release_workflow.get("derived_launch_gate"),
                "configuration_only": True,
            },
            "foundation_contracts": {
                "path": str(paths.foundation_contracts_path),
                "status": foundation_contracts.get("status"),
                "configuration_only": True,
            },
            "knowledge_plane_contract": {
                "path": str(paths.knowledge_contract_path),
                "status": knowledge_contract.get("status"),
                "configuration_only": True,
            },
            "metadata_refresh": {
                "performed": refreshed,
                "refresh_error": refresh_error,
                "interval_seconds": metadata_refresh_seconds,
                "candidate_metadata_only": True,
            },
            "bible_step_coverage": manifest["coverage"],
            "claim_boundary": {
                "supervisor_liveness_is_not_manager_qualification": True,
                "supervisor_liveness_is_not_tournament_completion": True,
                "supervisor_liveness_is_not_sandbox_activation": True,
                "no_model_body_downloads": True,
                "no_model_runtime": True,
                "no_automatic_promotion": True,
                "no_automatic_deletion_or_eviction": True,
            },
        }
    )
    _atomic_json(paths.status_path, status)
    return status


@contextlib.contextmanager
def _exclusive_watch_lock(paths: CampaignPaths) -> Iterator[None]:
    paths.lock_path.touch(exist_ok=True)
    with paths.lock_path.open("r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AscensionCampaignError(
                f"another campaign supervisor owns {paths.lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def watch(
    root: str | Path = DEFAULT_ROOT,
    *,
    bible_path: str | Path = DEFAULT_BIBLE,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    metadata_refresh_seconds: float = DEFAULT_METADATA_REFRESH_SECONDS,
) -> int:
    """Keep the campaign reconciliation loop detached and singleton-safe."""

    if not 15.0 <= float(interval_seconds) <= 3600.0:
        raise AscensionCampaignError("interval_seconds must be between 15 and 3600")
    paths = bootstrap_layout(root, bible_path=bible_path)
    stopping = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    old_term = signal.signal(signal.SIGTERM, request_stop)
    old_int = signal.signal(signal.SIGINT, request_stop)
    try:
        with _exclusive_watch_lock(paths):
            first_tick = True
            while not stopping:
                status = tick(
                    paths.lifecycle_root,
                    bible_path=bible_path,
                    metadata_refresh_seconds=metadata_refresh_seconds,
                    force_metadata_refresh=first_tick,
                )
                print(
                    json.dumps(
                        {
                            "recorded_at": status["recorded_at"],
                            "state": status["state"],
                            "heartbeat": status["heartbeat"],
                            "first_unmet_state": status["lifecycle"].get("first_unmet_state"),
                            "tournament": (status["lifecycle"].get("tournament") or {}).get("status"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                first_tick = False
                deadline = time.monotonic() + float(interval_seconds)
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        item.add_argument("--bible", type=Path, default=DEFAULT_BIBLE)

    init = sub.add_parser("init", help="create campaign-owned continuation paths")
    common(init)
    once = sub.add_parser("tick", help="reconcile every Bible step once")
    common(once)
    once.add_argument("--metadata-refresh-seconds", type=float, default=DEFAULT_METADATA_REFRESH_SECONDS)
    once.add_argument("--force-metadata-refresh", action="store_true")
    daemon = sub.add_parser("watch", help="run persistent campaign reconciliation")
    common(daemon)
    daemon.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    daemon.add_argument("--metadata-refresh-seconds", type=float, default=DEFAULT_METADATA_REFRESH_SECONDS)
    status = sub.add_parser("status", help="print the last durable supervisor status")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        paths = bootstrap_layout(args.root, bible_path=args.bible)
        evaluate_lifecycle(paths.lifecycle_root, bible_path=args.bible)
        print(json.dumps({"root": str(paths.lifecycle_root), "manifest": str(paths.manifest_path)}, sort_keys=True))
        return 0
    if args.command == "tick":
        print(
            json.dumps(
                tick(
                    args.root,
                    bible_path=args.bible,
                    metadata_refresh_seconds=args.metadata_refresh_seconds,
                    force_metadata_refresh=args.force_metadata_refresh,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "watch":
        return watch(
            args.root,
            bible_path=args.bible,
            interval_seconds=args.interval_seconds,
            metadata_refresh_seconds=args.metadata_refresh_seconds,
        )
    if args.command == "status":
        status = _read_json(CampaignPaths.from_root(args.root).status_path)
        if status is None:
            print(json.dumps({"state": "ABSENT", "status_path": str(CampaignPaths.from_root(args.root).status_path)}))
            return 2
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unknown command {args.command!r}")


__all__ = [
    "CampaignPaths",
    "AscensionCampaignError",
    "bootstrap_layout",
    "main",
    "tick",
    "watch",
]


if __name__ == "__main__":  # pragma: no cover - module CLI convenience
    raise SystemExit(main())
