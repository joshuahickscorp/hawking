"""Physical, bounded AgentOS recovery gate.

The gate uses a disposable fixture provider and real Mission/DagStore/
ToolRegistry code.  It is intentionally a fixture proof: passing it proves
that the control plane survives the tested failure classes, not that every
production model runtime is sovereign.
"""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.future import status_causality as sc

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.persist import atomic_write_json
from hcli.providers import GenerationResponse
from hcli.tool_registry import default_tool_registry
from hcli.workunit import WorkUnit


SCHEMA = "hcli.agentos.recovery_gate.v1"
_HOST_STARTED = "recovery-host-started.json"
_RESIDENT = "recovery-resident.json"
_TOOL_STARTED = "recovery-tool-started.json"
# Ceiling on the fixture child's life when nothing reaps it. The gate itself
# finishes in seconds; this only ever fires for a run that died mid-flight.
_FIXTURE_MAX_LIFETIME_S = 300.0
_REASONING = "recovery-reasoning.json"
_SAFE_ARTIFACT = "recovery-safe-artifact.txt"
_GATE_RECEIPT = "recovery-gate.json"


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


def _module_environment() -> Dict[str, str]:
    """Keep fixture child processes importable after changing cwd.

    The gate deliberately runs its host and resident in a disposable
    workspace.  That workspace is not the source checkout, so relying on the
    interpreter's implicit ``sys.path[0]`` would turn a real resident kill
    into an immediate import failure.
    """
    source_root = str(Path(__file__).resolve().parents[2])
    current = os.environ.get("PYTHONPATH", "")
    entries = [source_root]
    if current:
        entries.append(current)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entries)}


def _write(path: Path, value: Any) -> None:
    atomic_write_json(path, value if isinstance(value, dict) else {"value": value})


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wait_for(path: Path, timeout: float) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        value = _read(path)
        if value is not None:
            return value
        time.sleep(0.02)
    return _read(path)


def _fixture_resident_main(workspace: Path) -> int:
    """Child process representing an inference resident plus a running tool."""
    _write(workspace / _TOOL_STARTED, {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_at": time.time(),
        "tool": "fixture.blocking_tool",
    })
    # The parent kills this process -- but only if the parent lives to do it.
    # `while True` here meant every interrupted or timed-out gate run stranded a
    # child at PPID 1 forever; ten were alive on this host, one per suite run,
    # each holding a temp directory. The loop the comment claimed to be
    # "bounded" was not.
    #
    # Two exits, both cheap: the parent going away, and a wall-clock ceiling far
    # beyond any run of this gate.
    deadline = time.monotonic() + _FIXTURE_MAX_LIFETIME_S
    while time.monotonic() < deadline:
        if os.getppid() == 1:
            return 0
        time.sleep(0.1)
    return 0


class _FixtureProvider:
    def __init__(self, workspace: Path, mode: str) -> None:
        self.workspace = workspace
        self.mode = mode
        self.child: Optional[subprocess.Popen[str]] = None

    def identity(self) -> Dict[str, Any]:
        return {
            "provider": "fixture-resident",
            "runtime": "fixture",
            "model_id": "fixture-model",
            "model_identity": "fixture-model@recovery-gate",
            "capabilities": {"features": {"tool_calling": {"state": "supported"}}},
            "qualification": {"status": "TEST_FIXTURE_ONLY"},
        }

    def generate(self, request: Any, *, timeout: Optional[float] = None) -> GenerationResponse:
        del request, timeout
        if self.mode == "host":
            _write(self.workspace / _REASONING, {
                "status": "REASONING",
                "pid": os.getpid(),
                "at": time.time(),
                "provider": "fixture-resident",
            })
            child = subprocess.Popen(
                [sys.executable, "-m", "hcli.agentos.recovery", "--fixture-resident", str(self.workspace)],
                cwd=str(self.workspace),
                env=_module_environment(),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.child = child
            _write(self.workspace / _RESIDENT, {
                "status": "RUNNING",
                "pid": child.pid,
                "started_at": time.time(),
                "tool": "fixture.blocking_tool",
            })
            return_code = child.wait()
            raise RuntimeError(f"fixture resident stopped before generation: returncode={return_code}")

        # Recovery is a new provider instance. It performs one safe,
        # reversible artifact action and returns ordinary generation data.
        artifact = self.workspace / _SAFE_ARTIFACT
        artifact.write_text("recovered\n", encoding="utf-8")
        return GenerationResponse(
            text=json.dumps({"kind": "answer", "content": "recovered", "operations": [], "tests": []}),
            raw={"content": "recovered", "provider": "fixture-resident"},
            finish_reason="stop",
            provider="fixture-resident",
        )


class _NonsenseProvider:
    def generate(self, request: Any, *, timeout: Optional[float] = None) -> GenerationResponse:
        del request, timeout
        return GenerationResponse(text="nonsense; no evidence", raw={"content": "nonsense"}, provider="nonsense")


def _fixture_host(workspace: Path) -> int:
    from hcli.agentos import AgentOS

    _write(workspace / _HOST_STARTED, {"pid": os.getpid(), "started_at": time.time()})
    unit = WorkUnit(
        id="recovery-unit",
        role="generalist",
        description="survive resident and tool interruption",
        resource_class="TEST",
        preferred_backend="fixture-resident",
        provider="fixture-resident",
        verifier=f"test -f {workspace / _SAFE_ARTIFACT}",
    )
    agent = AgentOS(
        workspace,
        engine=object(),
        repo_root=workspace,
        providers={"fixture-resident": _FixtureProvider(workspace, "host")},
    )
    agent.start_mission("recover after resident and tool interruption", units={unit.id: unit})
    try:
        agent.run()
    except BaseException as exc:  # the parent normally kills this host
        _write(workspace / "recovery-host-error.json", {"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


def _malformed_json_case() -> Dict[str, Any]:
    from hcli.hawking_native import HawkingNativeProtocolError, ResidentProcess

    try:
        ResidentProcess._parse_line("{not-json")
    except HawkingNativeProtocolError as exc:
        return {"status": "REJECTED", "error_type": type(exc).__name__}
    return {"status": "ACCEPTED_UNEXPECTEDLY"}


def _nonsense_case(workspace: Path) -> Dict[str, Any]:
    from hcli.agentos import AgentOS

    root = workspace / "nonsense"
    root.mkdir(parents=True, exist_ok=True)
    unit = WorkUnit(
        id="nonsense-unit",
        role="generalist",
        description="provider must not self-certify",
        resource_class="TEST",
        preferred_backend="nonsense",
        provider="nonsense",
        verifier=f"test -f {root / 'must-not-exist'}",
    )
    agent = AgentOS(
        root,
        engine=object(),
        repo_root=workspace,
        providers={"nonsense": _NonsenseProvider()},
    )
    agent.start_mission("reject unsupported model claim", units={unit.id: unit})
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = agent.run()
    failed = [
        item.id
        for item in agent.mission.scheduler.units.values()
        if item.status == "failed"
    ] if agent.mission is not None else []
    return {
        "status": "REJECTED" if result.get("accepted") == 0 and failed else "ACCEPTED_UNEXPECTEDLY",
        "result": result,
        "failed_units": failed,
    }



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    recovery = report.get("recovery") if isinstance(report.get("recovery"), dict) else {}
    controls = report.get("controls") if isinstance(report.get("controls"), dict) else {}
    unmet = [name for name, value in checks.items() if value is not True]
    control_status = {
        name: (value.get("status") if isinstance(value, dict) else value)
        for name, value in controls.items()
    }
    if not checks and not recovery:
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
            "disposable fixture: spawn AgentOS host + fixture resident, SIGKILL resident "
            "and host, recover_mission + continue_mission; negative controls for "
            "malformed JSON, unreachable web.fetch, benchmark.run without confirm, "
            "nonsense provider that must not self-certify"
        ),
        "direct_observation": (
            f"recovery.status={recovery.get('status')!r}; "
            f"host_started={recovery.get('host_started')!r}; "
            f"resident_killed={recovery.get('resident_killed')!r}; "
            f"host_killed={recovery.get('host_killed')!r}; "
            f"safe_work_unit_verified={recovery.get('safe_work_unit_verified')!r}; "
            f"artifact_persisted={recovery.get('artifact_persisted')!r}; "
            f"control_status={control_status}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; unmet={unmet!r}"
        ),
        "interpretation": (
            "fixture recovery completed a verified WorkUnit and every negative control rejected"
            if status == "PASSED"
            else f"recovery/control checks unmet: {unmet or ['no checks recorded']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_recovery_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/recovery.py::run_recovery_gate",
        **payload,
    )

def run_recovery_gate(
    workspace: Optional[str | os.PathLike[str]] = None,
    *,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Run the disposable physical recovery/rejection gate."""
    owned = workspace is None
    root = Path(workspace).expanduser().resolve() if workspace else Path(tempfile.mkdtemp(prefix="hcli-recovery-gate-"))
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    host = subprocess.Popen(
        [sys.executable, "-m", "hcli.agentos.recovery", "--fixture-host", str(root)],
        cwd=str(root),
        env=_module_environment(),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    host_started = _wait_for(root / _HOST_STARTED, timeout_s / 3)
    resident = _wait_for(root / _RESIDENT, timeout_s / 3)
    # The resident writes its tool-start marker just after the host records
    # the resident pid. Wait for that marker before killing anything so the
    # gate actually covers a running tool, not only a spawned process.
    tool_started = _wait_for(root / _TOOL_STARTED, timeout_s / 3)
    killed = {"resident": False, "host": False, "resident_pid": (resident or {}).get("pid")}
    if resident and resident.get("pid"):
        try:
            os.kill(int(resident["pid"]), signal.SIGKILL)
            killed["resident"] = True
        except OSError:
            killed["resident"] = False
    # Ensure the durable DAG has reached a running unit before taking down the
    # control host. The resident/tool markers are written after dispatch, so a
    # short grace period is enough and avoids a race with the first checkpoint.
    time.sleep(0.08)
    try:
        os.kill(host.pid, signal.SIGKILL)
        killed["host"] = True
    except OSError:
        pass
    try:
        host.wait(timeout=5)
    except subprocess.TimeoutExpired:
        host.kill()
        host.wait(timeout=5)

    from hcli.agentos import AgentOS

    restarted = AgentOS(
        root,
        engine=object(),
        repo_root=root,
        providers={"fixture-resident": _FixtureProvider(root, "recover")},
    )
    restored = restarted.recover_mission()
    recovered_before = {
        "mission_id": restored.id,
        "unit_status": restored.scheduler.units["recovery-unit"].status,
        "unit_state": restored.status().get("unit_states", {}).get("recovery-unit"),
    }
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = restarted.continue_mission()
    unit = restarted.mission.scheduler.units["recovery-unit"] if restarted.mission else None
    completed_recovery_units = [
        candidate
        for candidate in (restarted.mission.scheduler.units.values() if restarted.mission else ())
        if candidate.status == "completed"
        and isinstance(candidate.verification, dict)
        and candidate.verification.get("ok") is True
    ]
    recovered_unit = completed_recovery_units[0] if completed_recovery_units else None
    recovery = {
        "status": "PASSED" if result.get("status") == "completed" and recovered_unit is not None else "FAILED",
        "host_started": bool(host_started),
        "reasoning_started": bool(_read(root / _REASONING)),
        "tool_started": bool(tool_started),
        "resident_killed": bool(killed["resident"]),
        "host_killed": bool(killed["host"]),
        "recovered_before_continue": recovered_before,
        "continued_result": result,
        "safe_work_unit_verified": recovered_unit is not None,
        "recovered_work_unit_id": recovered_unit.id if recovered_unit is not None else None,
        "artifact_persisted": (root / _SAFE_ARTIFACT).is_file(),
        "control_checkpoint": str(restarted.checkpoint_path),
        "mission_state": str(restarted.state_path),
    }
    controls = {
        "malformed_json": _malformed_json_case(),
        "network_unavailable": None,
        "benchmark_rejected": None,
        "nonsense_model": None,
    }
    registry = default_tool_registry(root, repo_root=root)
    network = registry.invoke("web.fetch", {"url": "https://www.rfc-editor.org:1/unavailable", "max_bytes": 1024, "timeout_s": 1})
    controls["network_unavailable"] = {"status": "REJECTED" if not network.ok else "ACCEPTED_UNEXPECTEDLY", "failure_class": network.failure_class}
    benchmark = registry.invoke("benchmark.run", {"confirm": False})
    controls["benchmark_rejected"] = {"status": "REJECTED" if not benchmark.ok else "ACCEPTED_UNEXPECTEDLY", "failure_class": benchmark.failure_class}
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        controls["nonsense_model"] = _nonsense_case(root)
    checks = {
        "recovery": recovery["status"] == "PASSED",
        "malformed_json": controls["malformed_json"]["status"] == "REJECTED",
        "network_unavailable": controls["network_unavailable"]["status"] == "REJECTED",
        "benchmark_rejected": controls["benchmark_rejected"]["status"] == "REJECTED",
        "nonsense_model": controls["nonsense_model"]["status"] == "REJECTED",
    }
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "qualification": "FIXTURE_PHYSICAL_CONTROL_PLANE_ONLY",
        "started_at": started,
        "finished_at": time.time(),
        "workspace": str(root),
        "owned_workspace": owned,
        "checks": checks,
        "recovery": recovery,
        "controls": controls,
        "claim_boundary": "This proves the tested AgentOS control-plane failure paths with a disposable fixture; it is not proof of production resident sovereignty.",
        "next_action": "run the same gate against each configured production provider and persist its receipts before claiming unattended sovereignty",
    }
    receipt_path = root / ".hcli" / "receipts" / _GATE_RECEIPT
    payload = causality_payload(report)
    record_recovery_causality(report, **payload)
    atomic_write_json(receipt_path, report)
    report["receipt_path"] = str(receipt_path)
    if emit:
        atomic_write_json(Path(emit).expanduser(), report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-host")
    parser.add_argument("--fixture-resident")
    parser.add_argument("--workspace")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.fixture_resident:
        return _fixture_resident_main(Path(args.fixture_resident).expanduser().resolve())
    if args.fixture_host:
        return _fixture_host(Path(args.fixture_host).expanduser().resolve())
    report = run_recovery_gate(args.workspace, emit=args.emit, timeout_s=args.timeout_s)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["SCHEMA", "causality_payload", "main", "record_recovery_causality", "records_five_fields", "run_recovery_gate"]


if __name__ == "__main__":
    raise SystemExit(main())
