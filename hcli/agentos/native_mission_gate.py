"""One live HCLI mission using native cognition plus deterministic evidence."""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.future import status_causality as sc

import sys

import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.persist import atomic_write_json
from hcli.workunit import WorkUnit


SCHEMA = "hcli.agentos.native_mission_gate.v1"


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


def _write_receipt(report: Dict[str, Any], emit: Optional[str], repo_root: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo_root / "receipts" / "headless" / "HCLI_NATIVE_MISSION_GATE.json"
    if not destination.is_absolute():
        destination = repo_root / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def _temporary_generation_env() -> Dict[str, Optional[str]]:
    names = ("HCLI_MODEL_TOKENS", "HCLI_STRUCTURED_OUTPUT_ATTEMPTS")
    old = {name: os.environ.get(name) for name in names}
    os.environ["HCLI_MODEL_TOKENS"] = "64"
    os.environ["HCLI_STRUCTURED_OUTPUT_ATTEMPTS"] = "1"
    return old


def _restore_generation_env(old: Dict[str, Optional[str]]) -> None:
    for name, value in old.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    error = report.get("error")
    verifier = report.get("verifier") if isinstance(report.get("verifier"), dict) else {}
    model_proof = report.get("model_proof") if isinstance(report.get("model_proof"), dict) else {}
    envelope = report.get("result_envelope") if isinstance(report.get("result_envelope"), dict) else {}
    unmet = [name for name, value in checks.items() if value is not True]
    if not checks and not error:
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
            "one live native mission: AgentOS.invoke_tool filesystem.write + fs.read, "
            "WorkUnit with shasum -a 256 verifier, AgentOS.start_mission/run, "
            "acceptance from persisted WorkUnit.verification not model prose"
        ),
        "direct_observation": (
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; "
            f"unmet={unmet!r}; verifier.outcome_ok={verifier.get('outcome')!r}; "
            f"acceptance_source={verifier.get('acceptance_source')!r}; "
            f"native_call_count={model_proof.get('native_call_count')!r}; "
            f"envelope.verdict={envelope.get('verdict')!r}; error={error!r}"
        ),
        "interpretation": (
            "typed tools, native cognition call, verifier, and result envelope all held"
            if status == "PASSED"
            else f"native mission checks unmet: {unmet or ['exception before checks']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_native_mission_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/native_mission_gate.py::run_native_mission_gate",
        **payload,
    )

def run_native_mission_gate(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str] = None,
    emit: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one reversible tool/fact/verifier mission through native HCLI."""
    from hcli.agentos.runtime import AgentOS
    from hcli.controller import Controller

    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    profile_path = Path(profile).expanduser().resolve() if profile else (repo / "hcli" / "hawking-native.sealed-3.14.json").resolve()
    workspace = Path(tempfile.mkdtemp(prefix="hcli-native-mission-"))
    marker = workspace / "verified-marker.txt"
    marker_rel = marker.relative_to(workspace).as_posix()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "LIVE_NATIVE_TOOL_VERIFIER_RECEIPT",
        "started_at": time.time(),
        "repo_root": str(repo),
        "workspace": str(workspace),
        "profile_path": str(profile_path),
        "model_claim_boundary": "native model proposes; typed tool and deterministic verifier decide",
    }
    controller = None
    old_env = _temporary_generation_env()
    try:
        controller = Controller(workspace, model=str(profile_path))
        agent = AgentOS(
            workspace,
            engine=controller.engine,
            controller=controller,
            repo_root=repo,
        )
        write_result = agent.invoke_tool(
            "filesystem.write",
            {"path": marker_rel, "content": "native-mission-marker\n", "overwrite": True},
        )
        read_result = agent.invoke_tool("fs.read", {"path": marker_rel})
        report["tool_results"] = {
            "write": write_result.to_dict(),
            "read": read_result.to_dict(),
        }
        if not write_result.ok or not read_result.ok or not isinstance(read_result.value, dict):
            raise RuntimeError("typed tool precondition failed")
        fact = {
            "path": read_result.value.get("path"),
            "bytes": read_result.value.get("bytes"),
            "sha256": read_result.value.get("sha256"),
            "content": read_result.value.get("content"),
            "source": "AgentOS.invoke_tool(fs.read)",
        }
        report["deterministic_fact"] = fact
        expected_sha = str(fact.get("sha256") or "")
        if not expected_sha:
            raise RuntimeError("typed read did not return a content hash")
        quoted_marker = shlex.quote(str(marker))
        verifier = f"test -f {quoted_marker} && shasum -a 256 {quoted_marker} | grep -q {shlex.quote(expected_sha)}"
        unit = WorkUnit(
            id="native-tool-verifier",
            role="generalist",
            description=(
                "Return one short answer confirming the typed read found the "
                "marker. Do not propose a mutation; the fixed verifier owns acceptance."
            ),
            resource_class="TEST",
            preferred_backend="resident",
            provider="resident",
            verifier=verifier,
        )
        mission = agent.start_mission(
            "Use native cognition to report the already-read marker; the deterministic verifier must decide acceptance.",
            units={unit.id: unit},
        )
        mission_result = agent.run()
        stored_unit = mission.scheduler.units[unit.id]
        verification = stored_unit.verification or {}
        engine_result = controller.engine.last_result if hasattr(controller.engine, "last_result") else None
        model_calls = list(getattr(controller.engine, "_model_calls", []) or [])
        profiles = agent.provider_profiles()
        native_calls = [
            call for call in model_calls
            if str(call.get("endpoint") or "").startswith("hawking-native://")
        ]
        # Use the canonical AgentOS boundary.  It derives acceptance from the
        # persisted WorkUnit verification rather than manufacturing a claim
        # from model prose in this evidence harness.
        envelope = agent.result_envelope(mission_result)
        report.update({
            "mission": {
                "id": mission.id,
                "goal": mission.goal,
                "result": mission_result,
                "unit": stored_unit.to_dict(),
            },
            "verifier": {
                "command": verifier,
                "outcome": verification,
                "acceptance_source": verification.get("acceptance_source"),
            },
            "model_proof": {
                "native_call_count": len(native_calls),
                "model_calls": model_calls,
                "engine_receipt": engine_result.get("receipt") if isinstance(engine_result, dict) else None,
                "provider_profiles": profiles,
            },
            "result_envelope": envelope,
        })
        report["checks"] = {
            "typed_write_ok": write_result.ok,
            "typed_read_ok": read_result.ok,
            "fact_hash_present": bool(expected_sha),
            "native_cognition_called": bool(native_calls),
            "mission_completed": mission_result.get("status") == "completed",
            "unit_completed": stored_unit.status == "completed",
            "verifier_passed": verification.get("ok") is True,
            "model_not_acceptance_authority": verification.get("acceptance_source") == "workunit_verifier",
            "result_verdict_accept": envelope.get("verdict") == "ACCEPT",
            "receipt_exists": bool(engine_result and engine_result.get("receipt") and Path(engine_result["receipt"]).is_file()),
        }
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - this is an evidence gate
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        if controller is not None:
            controller.shutdown()
        _restore_generation_env(old_env)
    report["finished_at"] = time.time()
    payload = causality_payload(report)
    record_native_mission_causality(report, **payload)
    _write_receipt(report, emit, repo)
    return report


__all__ = ["SCHEMA", "causality_payload", "record_native_mission_causality", "records_five_fields", "run_native_mission_gate"]
