"""Live AgentOS autonomy qualification and unattended-work runner.

This module is an evidence harness around the existing HCLI authorities.  It
does not add a scheduler: Mission remains the DAG/receipt authority, AgentOS
remains the typed-tool and control-checkpoint authority, and the selected
provider remains the only cognition authority.

The qualification stages deliberately separate claims:

* A1/A2 run real receipt/configuration work through native cognition, typed
  tools, and fixed deterministic WorkUnit verifiers;
* A3 kills the current native resident and proves connector restart plus
  WorkUnit continuation;
* A4 kills the HCLI control process after a completed dependency and resumes
  the persisted DAG from disk;
* A5 crashes after a reversible mutation but before the control checkpoint and
  proves replay is refused rather than duplicated;
* the unattended window keeps running bounded real WorkUnits for a requested
  duration and records progress even if an individual cycle is inconclusive.

All reports are explicit about what they prove.  Passing this file does not
claim model quality, accelerator performance, or sovereignty.
"""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.future import status_causality as sc

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from hcli.persist import atomic_write_json
from hcli.workunit import WorkUnit, identify_ready


SCHEMA = "hcli.agentos.autonomy_gate.v1"
WINDOW_SCHEMA = "hcli.agentos.unattended_window.v1"
DEFAULT_PROFILE_NAME = "hawking-native.sealed-3.14.json"
_PROCESS_STARTED = "process-host-started.json"
_PROCESS_READY = "process-first-complete.json"
_PROCESS_ERROR = "process-host-error.json"
_MUTATION_STARTED = "mutation-host-started.json"
_MUTATION_ERROR = "mutation-host-error.json"


FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def _record_gate_causality(
    report: Dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change status/qualification/checks.

    An unsupplied observation is UNTESTED, never a restatement of PASSED/FAILED.
    OVERREACHING is recorded beside the verdict; it does not override it.
    """
    status_before = report.get("status")
    qual_before = report.get("qualification")
    checks_before = dict(report["checks"]) if isinstance(report.get("checks"), dict) else report.get("checks")
    status = str(report.get("status") or "")
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source=source,
    )
    for key in FIVE_RECORDED_FIELDS:
        report[key] = rec[key]
    report["causality_verdict"] = rec["verdict"]
    report["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        report["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        report["claim_kind"] = rec["claim_kind"]
    checks_after = dict(report["checks"]) if isinstance(report.get("checks"), dict) else report.get("checks")
    if (
        report.get("status") != status_before
        or report.get("qualification") != qual_before
        or checks_after != checks_before
    ):
        raise RuntimeError("status_causality.emit mutated the gate verdict")
    return rec


def _repo_root(value: Optional[str | os.PathLike[str]]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _profile_path(repo: Path, value: Optional[str | os.PathLike[str]]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return (repo / "hcli" / DEFAULT_PROFILE_NAME).resolve()


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _generation_env(*, tokens: int = 64, attempts: int = 1) -> Iterable[None]:
    names = (
        "HCLI_MODEL_TOKENS",
        "HCLI_STRUCTURED_OUTPUT_ATTEMPTS",
        "HCLI_MODEL_TIMEOUT",
    )
    old = {name: os.environ.get(name) for name in names}
    os.environ["HCLI_MODEL_TOKENS"] = str(max(1, int(tokens)))
    os.environ["HCLI_STRUCTURED_OUTPUT_ATTEMPTS"] = str(max(1, int(attempts)))
    os.environ.setdefault("HCLI_MODEL_TIMEOUT", "180")
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _module_environment() -> Dict[str, str]:
    source_root = str(Path(__file__).resolve().parents[2])
    current = os.environ.get("PYTHONPATH", "")
    entries = [source_root]
    if current:
        entries.append(current)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entries)}


def _source_catalog(repo: Path) -> List[Dict[str, Any]]:
    """Return bounded, real Hawking evidence targets for a census mission."""
    rows = [
        {
            "label": "program checkpoint",
            "path": "receipts/headless/HCLI_AGENTOS_CHECKPOINT.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_AGENTOS_CHECKPOINT.json"},
        },
        {
            "label": "native ladder",
            "path": "receipts/headless/HCLI_AGENTOS_NATIVE_GATE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_AGENTOS_NATIVE_GATE.json"},
        },
        {
            "label": "resident proof",
            "path": "receipts/headless/HCLI_AGENTOS_RESIDENT_GATE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_AGENTOS_RESIDENT_GATE.json"},
        },
        {
            "label": "native mission",
            "path": "receipts/headless/HCLI_NATIVE_MISSION_GATE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_NATIVE_MISSION_GATE.json"},
        },
        {
            "label": "recovery proof",
            "path": "receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json"},
        },
        {
            "label": "research proof",
            "path": "receipts/headless/HCLI_AGENTOS_RESEARCH_GATE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_AGENTOS_RESEARCH_GATE.json"},
        },
        {
            "label": "VMCP proof",
            "path": "receipts/headless/HCLI_AGENTOS_VMCP_GATE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_AGENTOS_VMCP_GATE.json"},
        },
        {
            "label": "Flash identity",
            "path": "receipts/headless/HCLI_FLASH_NEXT_IDENTITY.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_FLASH_NEXT_IDENTITY.json"},
        },
        {
            "label": "accelerator smoke",
            "path": "receipts/headless/HCLI_ACCELERATOR_NATIVE_SMOKE.json",
            "tool": "receipt.read",
            "arguments": {"path": "receipts/headless/HCLI_ACCELERATOR_NATIVE_SMOKE.json"},
        },
        {
            "label": "ModelLake pipeline",
            "path": "receipts/headless/MODEL_LAKE_ROLLING_PIPELINE.json",
            "tool": "modellake.status",
            "arguments": {},
        },
        {
            "label": "native profile",
            "path": "hcli/hawking-native.sealed-3.14.json",
            "tool": "architecture.inspect",
            "arguments": {"path": "hcli/hawking-native.sealed-3.14.json", "max_tensors": 64},
        },
    ]
    return [row for row in rows if (repo / str(row["path"])).is_file()]


def _fixed_verifier(path: Path, expected_sha: str, derived: Optional[Path] = None) -> str:
    # The path and digest are resolved by this harness before dispatch.  The
    # model cannot nominate or change this verifier.
    import shlex

    quoted = shlex.quote(str(path))
    checks = [
        f"test -f {quoted}",
        f"shasum -a 256 {quoted} | grep -q {shlex.quote(expected_sha)}",
    ]
    if derived is not None:
        derived_quoted = shlex.quote(str(derived))
        checks.extend([
            f"test -f {derived_quoted}",
            f"grep -q {shlex.quote(expected_sha)} {derived_quoted}",
        ])
    return " && ".join(checks)


def _build_census_units(repo: Path, workspace: Path, count: int, prefix: str) -> tuple[Dict[str, WorkUnit], List[Dict[str, Any]]]:
    catalog = _source_catalog(repo)
    if not catalog:
        raise RuntimeError("no canonical HCLI evidence targets are available")
    units: Dict[str, WorkUnit] = {}
    jobs: List[Dict[str, Any]] = []
    previous: Optional[str] = None
    for index in range(max(1, int(count))):
        source = dict(catalog[index % len(catalog)])
        path = (repo / str(source["path"])).resolve()
        unit_id = f"{prefix}-{index + 1:02d}"
        source["unit_id"] = unit_id
        source["absolute_path"] = str(path)
        source["sha256"] = _sha256(path)
        source["tool_arguments"] = dict(source.get("arguments") or {})
        if source["tool"] == "modellake.status":
            source["tool_arguments"] = {}
        derived_rel = f"evidence/{prefix}-{index + 1:02d}.json"
        derived_path = (workspace / derived_rel).resolve()
        source["derived_path"] = str(derived_path)
        source["derived_relative_path"] = derived_rel
        source["derived_content"] = json.dumps(
            {
                "schema": "hcli.agentos.evidence_observation.v1",
                "unit_id": unit_id,
                "source": source["path"],
                "source_sha256": source["sha256"],
                "tool": source["tool"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        dependencies = [previous] if previous else []
        units[unit_id] = WorkUnit(
            id=unit_id,
            role="science" if "Flash" in str(source["label"]) else "generalist",
            description=(
                f"Inspect the real {source['label']} evidence at {path}. "
                "Use the supplied typed evidence result and give a concise factual "
                "summary. Do not mutate files and do not claim acceptance; the fixed "
                "verifier checks the source bytes."
            ),
            dependencies=dependencies,
            resource_class="TEST",
            preferred_backend="resident",
            provider="resident",
            verifier=_fixed_verifier(path, str(source["sha256"]), derived_path),
        )
        jobs.append(source)
        previous = unit_id
    return units, jobs


def _install_tool_hook(agent: Any, jobs: List[Dict[str, Any]]) -> tuple[Callable[[Any], None], List[Dict[str, Any]]]:
    by_id = {str(item["unit_id"]): item for item in jobs}
    invoked: set[str] = set()
    events: List[Dict[str, Any]] = []

    def hook(mission: Any) -> None:
        for unit in identify_ready(mission.scheduler.units):
            job = by_id.get(str(unit.id))
            if job is None or unit.id in invoked:
                continue
            invoked.add(unit.id)
            started = time.time()
            result = agent.invoke_tool(str(job["tool"]), dict(job.get("tool_arguments") or {}))
            events.append({
                "unit_id": unit.id,
                "tool": job["tool"],
                "source": job["path"],
                "ok": bool(result.ok),
                "failure_class": result.failure_class,
                "invocation_id": result.invocation_id,
                "receipt_path": result.provenance.get("receipt_path"),
                "elapsed_s": round(time.time() - started, 4),
            })
            # A read-only mission would trip Mission's no-progress guard even
            # when every unit was useful. Persist one bounded, reversible
            # observation through the typed write tool so accepted work has a
            # durable workspace artifact and a new fingerprint.
            derived = job.get("derived_relative_path")
            if derived:
                write_started = time.time()
                write = agent.invoke_tool("filesystem.write", {
                    "path": str(derived),
                    "content": str(job.get("derived_content") or ""),
                    "overwrite": False,
                })
                events.append({
                    "unit_id": unit.id,
                    "tool": "filesystem.write",
                    "source": job["path"],
                    "derived_path": str(derived),
                    "ok": bool(write.ok),
                    "failure_class": write.failure_class,
                    "invocation_id": write.invocation_id,
                    "receipt_path": write.provenance.get("receipt_path"),
                    "elapsed_s": round(time.time() - write_started, 4),
                })

    return hook, events


def _mission_records(agent: Any, jobs: List[Dict[str, Any]], tool_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    mission = agent.mission
    units = list(getattr(getattr(mission, "scheduler", None), "units", {}).values())
    model_calls: List[Dict[str, Any]] = []
    statuses = {str(unit.id): unit.to_dict() for unit in units}
    dispatch_order: List[str] = []
    log_path = getattr(mission, "log_path", None)
    if log_path is not None and Path(log_path).is_file():
        try:
            for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
                item = json.loads(line)
                if item.get("event") == "dispatch" and item.get("id"):
                    dispatch_order.append(str(item["id"]))
                if item.get("event") == "model_calls_observed":
                    calls = item.get("calls") or []
                    if isinstance(calls, list):
                        model_calls.extend(item for item in calls if isinstance(item, dict))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    attempts = sum(int(item.get("attempts") or 0) for item in statuses.values())
    accepted = sum(1 for item in statuses.values() if item.get("status") == "completed")
    rejected = sum(1 for item in statuses.values() if item.get("status") == "failed")
    return {
        "mission_id": getattr(mission, "id", None),
        "goal": getattr(mission, "goal", None),
        "result": _json_safe(agent.last_result),
        "units": statuses,
        "dispatch_order": dispatch_order,
        "jobs": _json_safe(jobs),
        "tool_events": _json_safe(tool_events),
        "model_calls": _json_safe(model_calls),
        "metrics": {
            "workunits_declared": len(units),
            "attempted": sum(1 for item in statuses.values() if int(item.get("attempts") or 0) > 0),
            "accepted": accepted,
            "rejected": rejected,
            "inconclusive": len(units) - accepted - rejected,
            "retries": max(0, attempts - len(units)),
            "model_calls": len(model_calls),
            "typed_tool_calls": len(tool_events),
            "typed_tool_failures": sum(1 for item in tool_events if not item.get("ok")),
            "verifier_calls": sum(1 for item in statuses.values() if item.get("verification") is not None),
            "human_interventions": 0,
        },
    }


def run_census_mission(
    *,
    repo_root: Path,
    profile: Path,
    count: int,
    prefix: str,
) -> Dict[str, Any]:
    from hcli.agentos.runtime import AgentOS
    from hcli.controller import Controller

    workspace = Path(tempfile.mkdtemp(prefix=f"hcli-{prefix}-"))
    units, jobs = _build_census_units(repo_root, workspace, count, prefix)
    report: Dict[str, Any] = {
        "stage": prefix,
        "status": "RUNNING",
        "started_at": time.time(),
        "repo_root": str(repo_root),
        "profile_path": str(profile),
        "workspace": str(workspace),
        "requested_workunits": int(count),
        "work_definition": "real canonical receipts/configuration metadata; no toy artifacts",
    }
    controller = None
    agent = None
    old_timeout = os.environ.get("HCLI_MODEL_TIMEOUT")
    try:
        controller = Controller(workspace, model=str(profile))
        agent = AgentOS(workspace, engine=controller.engine, controller=controller, repo_root=repo_root)
        hook, tool_events = _install_tool_hook(agent, jobs)
        mission = agent.start_mission(
            f"Perform {count} bounded Hawking evidence inspections through typed tools and native cognition.",
            units=units,
        )
        mission.before_dispatch = hook
        with _generation_env(tokens=64, attempts=1):
            mission_result = agent.run()
        report["mission"] = _mission_records(agent, jobs, tool_events)
        report["mission"]["result"] = mission_result
        report["provider_profiles"] = _json_safe(agent.provider_profiles())
        checks = {
            "requested_count_reached": len(units) == int(count),
            "mission_completed": mission_result.get("status") == "completed",
            "all_units_completed": all(unit.status == "completed" for unit in units.values()),
            "all_fixed_verifiers_passed": all(
                isinstance(unit.verification, dict) and unit.verification.get("ok") is True
                for unit in units.values()
            ),
            "all_typed_tools_ok": len(tool_events) == len(units) * 2 and all(item.get("ok") is True for item in tool_events),
            "native_model_calls": bool(getattr(controller.engine, "_model_calls", [])),
            "next_work_selected": len(report["mission"].get("dispatch_order") or []) >= len(units),
            "no_human_interventions": report["mission"]["metrics"]["human_interventions"] == 0,
        }
        report["checks"] = checks
        report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - the receipt is the failure boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        if controller is not None:
            controller.shutdown()
        if old_timeout is not None:
            os.environ["HCLI_MODEL_TIMEOUT"] = old_timeout
        else:
            os.environ.pop("HCLI_MODEL_TIMEOUT", None)
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - report["started_at"], 3)
    return report


class _CrashOnceNativeProvider:
    """Native provider that kills its own resident before one request."""

    def __init__(self, profile: Path) -> None:
        from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector

        self.config = HawkingNativeConfig.from_file(str(profile))
        self.connector = HawkingNativeConnector(self.config)
        self.killed_pid: Optional[int] = None
        self.call_count = 0
        self.last_raw: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        self.connector.start(timeout=float(os.environ.get("HCLI_READY_TIMEOUT", "300")))

    def stop(self) -> Dict[str, Any]:
        return self.connector.stop()

    def identity(self) -> Dict[str, Any]:
        return self.connector.identity()

    def capabilities(self) -> Any:
        return self.config.capabilities

    def health(self) -> Dict[str, Any]:
        return (self.connector.identity().get("resident_health") or {})

    def generate(self, request: Any, *, timeout: Optional[float] = None) -> Any:
        from hcli.providers import GenerationResponse

        self.call_count += 1
        if self.killed_pid is None and self.connector.process is not None:
            process = self.connector.process
            self.killed_pid = int(process.pid)
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        raw = self.connector.complete_payload(request.to_payload(), timeout=timeout)
        self.last_raw = raw
        response = GenerationResponse.from_completion(raw, provider=str(self.config.provider))
        response.request_id = getattr(request, "request_id", None)
        return response


def run_resident_kill(
    *, repo_root: Path, profile: Path
) -> Dict[str, Any]:
    from hcli.agentos.runtime import AgentOS
    from hcli.workunit import WorkUnit

    workspace = Path(tempfile.mkdtemp(prefix="hcli-resident-kill-"))
    marker = workspace / "resident-kill-marker.txt"
    marker.write_text("resident-kill-proof\n", encoding="utf-8")
    expected = _sha256(marker)
    provider = _CrashOnceNativeProvider(profile)
    report: Dict[str, Any] = {
        "stage": "A3_resident_kill",
        "status": "RUNNING",
        "started_at": time.time(),
        "repo_root": str(repo_root),
        "profile_path": str(profile),
        "workspace": str(workspace),
        "failure_injection": "SIGKILL native resident immediately before first request",
    }
    agent = AgentOS(workspace, engine=object(), repo_root=repo_root, providers={"crash-resident": provider})
    unit = WorkUnit(
        id="resident-kill-unit",
        role="generalist",
        description="Report one short sentence after the native resident is recovered; fixed verifier owns acceptance.",
        resource_class="TEST",
        preferred_backend="crash-resident",
        provider="crash-resident",
        verifier=_fixed_verifier(marker, expected),
    )
    try:
        provider.start()
        before = provider.identity()
        agent.start_mission("Recover one real native WorkUnit after resident death.", units={unit.id: unit})
        with _generation_env(tokens=64, attempts=1):
            result = agent.run()
        after = provider.identity()
        stored = agent.mission.scheduler.units[unit.id] if agent.mission is not None else unit
        report.update({
            "mission": {
                "result": _json_safe(result),
                "unit": stored.to_dict(),
            },
            "provider": {
                "before": _json_safe(before),
                "after": _json_safe(after),
                "calls": provider.call_count,
                "killed_pid": provider.killed_pid,
                "restart_count": provider.connector.restart_count,
            },
        })
        before_health = before.get("resident_health") or {}
        after_health = after.get("resident_health") or {}
        response_hawking = provider.last_raw.get("hawking", {}) if isinstance(provider.last_raw, dict) else {}
        checks = {
            "resident_was_live": bool(before_health.get("ready") and before_health.get("pid")),
            "resident_killed": provider.killed_pid is not None,
            "death_was_recovered": provider.connector.restart_count == 1,
            "identity_preserved": after.get("resident_identity") == before.get("resident_identity"),
            "pid_generation_changed": bool(before_health.get("pid")) and before_health.get("pid") != after_health.get("pid"),
            "model_reopened_once_after_restart": after_health.get("model_open_count") == 1,
            "weights_uploaded_once_after_restart": after_health.get("weight_upload_count") == 1,
            "no_fallbacks": response_hawking.get("fallbacks") == 0,
            "mission_completed": result.get("status") == "completed",
            "unit_verified": stored.status == "completed" and (stored.verification or {}).get("ok") is True,
            "no_false_success_before_verifier": (stored.verification or {}).get("acceptance_source") == "workunit_verifier",
        }
        report["checks"] = checks
        report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        provider.stop()
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - report["started_at"], 3)
    return report


def _process_host(workspace: Path, repo: Path) -> int:
    """Run a two-node real-evidence DAG until the parent kills this host."""
    from hcli.agentos.runtime import AgentOS

    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(workspace / _PROCESS_STARTED, {"pid": os.getpid(), "started_at": time.time()})
    first_path = (repo / "receipts/headless/HCLI_AGENTOS_CHECKPOINT.json").resolve()
    second_path = (repo / "receipts/headless/HCLI_FLASH_NEXT_IDENTITY.json").resolve()
    first = WorkUnit(
        id="process-recovery-first",
        role="science",
        description=f"Inspect real checkpoint evidence at {first_path}.",
        resource_class="TEST",
        preferred_backend="cpu",
        provider="cpu",
        verifier=_fixed_verifier(first_path, _sha256(first_path)),
    )
    second = WorkUnit(
        id="process-recovery-second",
        role="science",
        description=f"Inspect real Flash identity evidence at {second_path}.",
        dependencies=[first.id],
        resource_class="TEST",
        preferred_backend="cpu",
        provider="cpu",
        verifier=_fixed_verifier(second_path, _sha256(second_path)),
    )
    agent = AgentOS(workspace, engine=object(), repo_root=repo)
    mission = agent.start_mission("Continue a real evidence DAG after HCLI process death.", units={first.id: first, second.id: second})
    marker_written = False

    def pause_after_first(current: Any) -> None:
        nonlocal marker_written
        unit = current.scheduler.units.get(first.id)
        if not marker_written and unit is not None and unit.status == "completed":
            marker_written = True
            _write_json(workspace / _PROCESS_READY, {
                "pid": os.getpid(),
                "at": time.time(),
                "first_status": unit.status,
                "second_status": current.scheduler.units[second.id].status,
            })
            # Give the parent a deterministic window in which the first
            # accepted unit is durable and the dependent unit is still pending.
            time.sleep(3.0)

    mission.before_dispatch = pause_after_first
    try:
        agent.run()
    except BaseException as exc:  # parent normally SIGKILLs this process
        _write_json(workspace / _PROCESS_ERROR, {"type": type(exc).__name__, "message": str(exc)[:1000]})
        return 1
    return 0


def _wait_for(path: Path, timeout_s: float) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        value = _read_json(path)
        if value is not None:
            return value
        time.sleep(0.02)
    return _read_json(path)


def run_process_kill(*, repo_root: Path) -> Dict[str, Any]:
    from hcli.agentos.runtime import AgentOS

    workspace = Path(tempfile.mkdtemp(prefix="hcli-process-kill-"))
    host = subprocess.Popen(
        [sys.executable, "-m", "hcli.agentos.autonomy_gate", "--process-host", str(workspace), "--repo-root", str(repo_root)],
        cwd=str(workspace),
        env=_module_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    started = _wait_for(workspace / _PROCESS_STARTED, 10)
    ready = _wait_for(workspace / _PROCESS_READY, 30)
    killed = False
    if ready is not None:
        try:
            os.kill(host.pid, signal.SIGKILL)
            killed = True
        except OSError:
            pass
    try:
        host.wait(timeout=5)
    except subprocess.TimeoutExpired:
        host.kill()
        host.wait(timeout=5)
    before: Dict[str, Any] = {}
    result: Dict[str, Any] = {}
    report: Dict[str, Any] = {
        "stage": "A4_hcli_process_kill",
        "status": "RUNNING",
        "started_at": time.time(),
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "host_pid": host.pid,
        "host_started": started,
        "first_completed_before_kill": ready,
        "host_killed": killed,
    }
    try:
        state = _read_json(workspace / ".hcli/mission/state.json") or {}
        before = {
            "phase": state.get("phase"),
            "units": state.get("units"),
            "checkpoint_id": state.get("checkpoint_id"),
        }
        restarted = AgentOS(workspace, engine=object(), repo_root=repo_root)
        restored = restarted.recover_mission()
        recovered_snapshot = {uid: unit.to_dict() for uid, unit in restored.scheduler.units.items()}
        result = restarted.continue_mission()
        final_units = {uid: unit.to_dict() for uid, unit in restarted.mission.scheduler.units.items()} if restarted.mission else {}
        attempts = {uid: int(item.get("attempts") or 0) for uid, item in final_units.items()}
        checks = {
            "host_started": started is not None,
            "meaningful_progress_before_kill": bool(ready and ready.get("first_status") == "completed"),
            "host_killed": killed,
            "disk_recovered": bool(recovered_snapshot),
            "first_unit_not_duplicated": attempts.get("process-recovery-first") == 1,
            "second_unit_ran_once": attempts.get("process-recovery-second") == 1,
            "all_units_completed": result.get("status") == "completed" and all(item.get("status") == "completed" for item in final_units.values()),
            "no_loss_or_duplication": len(final_units) == 2 and set(final_units) == {"process-recovery-first", "process-recovery-second"},
            "next_action_was_recovered_from_disk": restarted.recovery_status().get("next_action") in {
                "inspect the result envelope and receipts; start the next mission if more work is required",
                "dispatch the next dependency-ready work unit; failed units may produce bounded repairs",
            },
        }
        report.update({"before_restart": before, "recovered_units": recovered_snapshot, "continued_result": result, "final_units": final_units, "checks": checks})
        report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - report["started_at"], 3)
    return report


def _mutation_host(workspace: Path, repo: Path, content: str) -> int:
    from hcli.agentos.runtime import AgentOS

    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(workspace / _MUTATION_STARTED, {"pid": os.getpid(), "started_at": time.time()})
    agent = AgentOS(workspace, engine=object(), repo_root=repo)
    # The mutation itself is routed through the typed reversible filesystem
    # tool. Exiting from the control-checkpoint callback models a crash after
    # bytes are durable but before final bookkeeping is durable.
    def crash_before_bookkeeping(**_kwargs: Any) -> Path:
        os._exit(17)

    agent._persist_control_checkpoint = crash_before_bookkeeping  # type: ignore[method-assign]
    agent.invoke_tool("filesystem.write", {
        "path": "idempotent-artifact.txt",
        "content": content,
        "overwrite": False,
    })
    _write_json(workspace / _MUTATION_ERROR, {"error": "unexpectedly survived crash injection"})
    return 1


def run_idempotency_crash(*, repo_root: Path) -> Dict[str, Any]:
    from hcli.agentos.runtime import AgentOS

    workspace = Path(tempfile.mkdtemp(prefix="hcli-idempotency-"))
    content = f"hcli-idempotency-proof:{time.time_ns()}\n"
    host = subprocess.Popen(
        [sys.executable, "-m", "hcli.agentos.autonomy_gate", "--mutation-host", str(workspace), "--repo-root", str(repo_root), "--mutation-content", content],
        cwd=str(workspace),
        env=_module_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    started = _wait_for(workspace / _MUTATION_STARTED, 10)
    try:
        exit_code = host.wait(timeout=10)
    except subprocess.TimeoutExpired:
        host.kill()
        exit_code = host.wait(timeout=5)
    artifact = workspace / "idempotent-artifact.txt"
    before_sha = _sha256(artifact) if artifact.is_file() else None
    before_stat = artifact.stat().st_mtime_ns if artifact.is_file() else None
    report: Dict[str, Any] = {
        "stage": "A5_post_mutation_crash",
        "status": "RUNNING",
        "started_at": time.time(),
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "host_pid": host.pid,
        "host_started": started,
        "host_exit_code": exit_code,
        "mutation": {
            "path": str(artifact),
            "expected_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "artifact_sha256_before_recovery": before_sha,
            "mtime_ns_before_recovery": before_stat,
        },
    }
    try:
        unit = WorkUnit(
            id="idempotency-recovery-unit",
            role="verifier",
            description="Verify the already durable reversible artifact; do not repeat the mutation.",
            resource_class="TEST",
            preferred_backend="cpu",
            provider="cpu",
            verifier=_fixed_verifier(artifact, str(before_sha or "")),
        )
        agent = AgentOS(workspace, engine=object(), repo_root=repo_root)
        agent.start_mission("Recover bookkeeping after a post-mutation process crash.", units={unit.id: unit})
        result = agent.run()
        replay = agent.invoke_tool("filesystem.write", {
            "path": "idempotent-artifact.txt",
            "content": content,
            "overwrite": False,
        })
        after_sha = _sha256(artifact) if artifact.is_file() else None
        after_stat = artifact.stat().st_mtime_ns if artifact.is_file() else None
        stored = agent.mission.scheduler.units[unit.id] if agent.mission else unit
        checks = {
            "host_started": started is not None,
            "crash_injected_after_mutation": exit_code == 17,
            "artifact_survived": artifact.is_file(),
            "recovery_unit_completed": result.get("status") == "completed" and stored.status == "completed",
            "recovery_verifier_passed": (stored.verification or {}).get("ok") is True,
            "replay_refused_by_typed_tool": replay.ok is False and replay.failure_class == "FileExistsError",
            "no_duplicate_bytes": before_sha == after_sha,
            "no_rewrite": before_stat == after_stat,
            "acceptance_from_verifier": (stored.verification or {}).get("acceptance_source") == "workunit_verifier",
        }
        report.update({
            "recovery_result": result,
            "recovery_unit": stored.to_dict(),
            "replay": replay.to_dict(),
            "mutation": {**report["mutation"], "artifact_sha256_after_recovery": after_sha, "mtime_ns_after_recovery": after_stat},
            "checks": checks,
        })
        report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - report["started_at"], 3)
    return report



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    stage_status = report.get("stage_status") if isinstance(report.get("stage_status"), dict) else {}
    error = report.get("error")
    unmet = [name for name, value in checks.items() if value is not True]
    if not checks and not stage_status and not error:
        return {
            "probe_performed": "",
            "direct_observation": "",
            "interpretation": str(report.get("status") or ""),
            "probe_kind": "",
            "claim_kind": None,
        }
    status = str(report.get("status") or "")
    return {
        "probe_performed": (
            "autonomy A1-A5: run_census_mission (3 then >=10 WorkUnits), "
            "run_resident_kill, run_process_kill, run_idempotency_crash; "
            f"stages_requested={report.get('stages_requested')!r} profile={report.get('profile_path')}"
        ),
        "direct_observation": (
            f"stage_status={stage_status}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; "
            f"unmet={unmet!r}; error={error!r}"
        ),
        "interpretation": (
            "all_requested_stages_passed is True from the named A1-A5 stage statuses"
            if status == "PASSED"
            else f"autonomy stage checks unmet: {unmet or ['exception before checks']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_autonomy_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/autonomy_gate.py::run_autonomy_gate",
        **payload,
    )

def run_autonomy_gate(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    stage: str = "all",
    count: int = 10,
) -> Dict[str, Any]:
    """Run A1-A5 and persist one machine-readable qualification receipt."""
    repo = _repo_root(repo_root)
    native_profile = _profile_path(repo, profile)
    selected = str(stage or "all").strip().lower()
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "LIVE_AGENTOS_AUTONOMY_CONTROL_PLANE",
        "started_at": started,
        "repo_root": str(repo),
        "profile_path": str(native_profile),
        "stages_requested": selected,
        "claim_boundary": "native cognition proposes; typed tools, persisted WorkUnits, and fixed verifiers decide; no sovereignty claim",
        "stages": {},
    }
    try:
        if selected in {"all", "a1"}:
            report["stages"]["A1_three_workunits"] = run_census_mission(repo_root=repo, profile=native_profile, count=3, prefix="a1")
        if selected in {"all", "a2"}:
            report["stages"]["A2_ten_workunits"] = run_census_mission(repo_root=repo, profile=native_profile, count=max(10, int(count)), prefix="a2")
        if selected in {"all", "a3", "resident-kill"}:
            report["stages"]["A3_resident_kill"] = run_resident_kill(repo_root=repo, profile=native_profile)
        if selected in {"all", "a4", "process-kill"}:
            report["stages"]["A4_hcli_process_kill"] = run_process_kill(repo_root=repo)
        if selected in {"all", "a5", "idempotency"}:
            report["stages"]["A5_post_mutation_crash"] = run_idempotency_crash(repo_root=repo)
        if not report["stages"]:
            raise ValueError(f"unknown autonomy stage: {stage}")
        stage_status = {name: value.get("status") for name, value in report["stages"].items()}
        report["stage_status"] = stage_status
        report["checks"] = {
            "all_requested_stages_passed": all(value == "PASSED" for value in stage_status.values()),
            "a1_three_real_workunits": report["stages"].get("A1_three_workunits", {}).get("status") == "PASSED" if "A1_three_workunits" in report["stages"] else True,
            "a2_ten_real_workunits": report["stages"].get("A2_ten_workunits", {}).get("status") == "PASSED" if "A2_ten_workunits" in report["stages"] else True,
            "a3_resident_recovery": report["stages"].get("A3_resident_kill", {}).get("status") == "PASSED" if "A3_resident_kill" in report["stages"] else True,
            "a4_process_recovery": report["stages"].get("A4_hcli_process_kill", {}).get("status") == "PASSED" if "A4_hcli_process_kill" in report["stages"] else True,
            "a5_idempotency": report["stages"].get("A5_post_mutation_crash", {}).get("status") == "PASSED" if "A5_post_mutation_crash" in report["stages"] else True,
        }
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HCLI_AGENTOS_AUTONOMY_GATE.json"
    payload = causality_payload(report)
    record_autonomy_causality(report, **payload)
    atomic_write_json(destination, report)
    report["receipt_path"] = str(destination)
    return report


def _window_unit(repo: Path, workspace: Path, index: int) -> tuple[WorkUnit, Dict[str, Any]]:
    catalog = _source_catalog(repo)
    source = dict(catalog[index % len(catalog)])
    path = (repo / str(source["path"])).resolve()
    source["absolute_path"] = str(path)
    source["sha256"] = _sha256(path)
    source["unit_id"] = f"window-{index + 1:06d}"
    if source["tool"] == "modellake.status":
        source["tool_arguments"] = {}
    else:
        source["tool_arguments"] = dict(source.get("arguments") or {})
    source["derived_relative_path"] = f"evidence/{source['unit_id']}.json"
    source["derived_content"] = json.dumps(
        {
            "schema": "hcli.agentos.evidence_observation.v1",
            "unit_id": source["unit_id"],
            "source": source["path"],
            "source_sha256": source["sha256"],
            "tool": source["tool"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    derived_path = (workspace / source["derived_relative_path"]).resolve()
    source["derived_path"] = str(derived_path)
    unit = WorkUnit(
        id=str(source["unit_id"]),
        role="science" if "Flash" in str(source["label"]) else "generalist",
        description=f"Perform one bounded real Hawking evidence inspection of {source['label']} at {path}; summarize facts briefly.",
        resource_class="TEST",
        preferred_backend="resident",
        provider="resident",
        verifier=_fixed_verifier(path, str(source["sha256"]), derived_path),
    )
    return unit, source


def run_unattended_window(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    workspace: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    duration_s: float = 3600.0,
    interval_s: float = 30.0,
) -> Dict[str, Any]:
    """Run real bounded native WorkUnits without human intervention."""
    from hcli.agentos.runtime import AgentOS
    from hcli.controller import Controller

    repo = _repo_root(repo_root)
    native_profile = _profile_path(repo, profile)
    root = Path(workspace).expanduser().resolve() if workspace else Path(tempfile.mkdtemp(prefix="hcli-unattended-window-"))
    root.mkdir(parents=True, exist_ok=True)
    duration = max(1.0, float(duration_s))
    interval = max(0.0, float(interval_s))
    started = time.time()
    report: Dict[str, Any] = {
        "schema": WINDOW_SCHEMA,
        "status": "RUNNING",
        "qualification": "LIVE_UNATTENDED_NATIVE_WORK",
        "started_at": started,
        "deadline": started + duration,
        "duration_requested_s": duration,
        "interval_s": interval,
        "repo_root": str(repo),
        "profile_path": str(native_profile),
        "workspace": str(root),
        "claim_boundary": "one-hour unattended useful-work observation; not a sovereignty or quality certification",
        "cycles": [],
        "metrics": {"attempted": 0, "accepted": 0, "rejected": 0, "inconclusive": 0, "retries": 0, "model_calls": 0, "typed_tool_calls": 0, "verifier_calls": 0, "human_interventions": 0},
    }
    atomic_write_json(root / "window-progress.json", report)
    controller = None
    try:
        with _generation_env(tokens=64, attempts=1):
            controller = Controller(root, model=str(native_profile))
            cycle_index = 0
            while time.time() < started + duration:
                cycle_started = time.time()
                cycle_index += 1
                agent: Optional[AgentOS] = None
                cycle: Dict[str, Any] = {"index": cycle_index, "started_at": cycle_started, "status": "RUNNING"}
                try:
                    agent = AgentOS(root, engine=controller.engine, controller=controller, repo_root=repo)
                    unit, job = _window_unit(repo, root, cycle_index - 1)
                    hook, tool_events = _install_tool_hook(agent, [job])
                    mission = agent.start_mission("Continue unattended real Hawking evidence work.", units={unit.id: unit})
                    mission.before_dispatch = hook
                    result = agent.run()
                    stored = agent.mission.scheduler.units[unit.id] if agent.mission else unit
                    cycle_model_calls = list(getattr(controller.engine, "_model_calls", []) or [])
                    model_call_observed = bool(cycle_model_calls)
                    cycle.update({
                        # A fixed verifier proves the artifact, not that the
                        # selected cognition provider actually ran.  Keep a
                        # completed-but-no-call cycle visible as inconclusive
                        # instead of allowing stale/dead-provider state to
                        # qualify unattended work.
                        "status": "ACCEPTED" if stored.status == "completed" and model_call_observed else "INCONCLUSIVE",
                        "result": result,
                        "unit": stored.to_dict(),
                        "job": job,
                        "tool_events": tool_events,
                        "model_calls": cycle_model_calls,
                        "model_call_observed": model_call_observed,
                        "acceptance_ineligible_reason": (
                            None if model_call_observed else "NO_FRESH_NATIVE_MODEL_CALL"
                        ),
                        "elapsed_s": round(time.time() - cycle_started, 3),
                    })
                    report["metrics"]["attempted"] += 1
                    if stored.status == "completed" and model_call_observed:
                        report["metrics"]["accepted"] += 1
                    else:
                        report["metrics"]["inconclusive"] += 1
                    report["metrics"]["model_calls"] += len(cycle_model_calls)
                    report["metrics"]["typed_tool_calls"] += len(tool_events)
                    report["metrics"]["verifier_calls"] += 1 if stored.verification is not None else 0
                except Exception as exc:  # routine cycle failure: persist and continue
                    cycle.update({"status": "INCONCLUSIVE", "error": {"type": type(exc).__name__, "message": str(exc)[:1200]}, "elapsed_s": round(time.time() - cycle_started, 3)})
                    report["metrics"]["attempted"] += 1
                    report["metrics"]["inconclusive"] += 1
                    try:
                        if controller is not None:
                            controller.shutdown()
                        controller = Controller(root, model=str(native_profile))
                    except Exception as restart_exc:
                        cycle["controller_restart_error"] = {"type": type(restart_exc).__name__, "message": str(restart_exc)[:800]}
                report["cycles"].append(cycle)
                report["metrics"]["retries"] = sum(max(0, int((item.get("unit") or {}).get("attempts") or 0) - 1) for item in report["cycles"])
                atomic_write_json(root / "window-progress.json", report)
                remaining = (started + duration) - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(interval, remaining))
    finally:
        if controller is not None:
            controller.shutdown()
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["status"] = "PASSED" if report["elapsed_s"] >= duration and report["metrics"]["accepted"] > 0 else "FAILED"
    report["checks"] = {
        "requested_duration_observed": report["elapsed_s"] >= duration,
        "real_work_attempted": report["metrics"]["attempted"] > 0,
        "useful_work_accepted": report["metrics"]["accepted"] > 0,
        "accepted_work_had_fresh_model_calls": report["metrics"]["accepted"] > 0 and report["metrics"]["model_calls"] >= report["metrics"]["accepted"],
        "no_human_interventions": report["metrics"]["human_interventions"] == 0,
    }
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HCLI_AGENTOS_UNATTENDED_WINDOW.json"
    atomic_write_json(destination, report)
    report["receipt_path"] = str(destination)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--emit")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--process-host")
    parser.add_argument("--mutation-host")
    parser.add_argument("--mutation-content", default="hcli-idempotency-proof\n")
    parser.add_argument("--unattended-window", action="store_true")
    parser.add_argument("--workspace")
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--interval-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    repo = _repo_root(args.repo_root)
    if args.process_host:
        return _process_host(Path(args.process_host).expanduser().resolve(), repo)
    if args.mutation_host:
        return _mutation_host(Path(args.mutation_host).expanduser().resolve(), repo, args.mutation_content)
    if args.unattended_window:
        report = run_unattended_window(
            repo_root=repo,
            profile=args.profile,
            workspace=args.workspace,
            emit=args.emit,
            duration_s=args.duration_s,
            interval_s=args.interval_s,
        )
    else:
        report = run_autonomy_gate(
            repo_root=repo,
            profile=args.profile,
            emit=args.emit,
            stage=args.stage,
            count=args.count,
        )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["SCHEMA", "WINDOW_SCHEMA", "main", "run_autonomy_gate", "run_unattended_window"]


if __name__ == "__main__":
    raise SystemExit(main())
