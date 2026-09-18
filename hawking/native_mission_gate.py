"""One live HAWKING mission using native cognition plus deterministic evidence."""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[1]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.verify import status_causality as sc

import sys

import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hawking.persist import atomic_write_json
from hawking.workunit import WorkUnit


SCHEMA = "hawking.native_mission_gate.v1"


FIVE_RECORDED_FIELDS = sc.FIVE_RECORDED_FIELDS
records_five_fields = sc.records_five_fields


def _write_receipt(report: Dict[str, Any], emit: Optional[str], repo_root: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo_root / "receipts" / "headless" / "HAWKING_NATIVE_MISSION_GATE.json"
    if not destination.is_absolute():
        destination = repo_root / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def _temporary_generation_env() -> Dict[str, Optional[str]]:
    names = ("HAWKING_MODEL_TOKENS", "HAWKING_STRUCTURED_OUTPUT_ATTEMPTS")
    old = {name: os.environ.get(name) for name in names}
    os.environ["HAWKING_MODEL_TOKENS"] = "64"
    os.environ["HAWKING_STRUCTURED_OUTPUT_ATTEMPTS"] = "1"
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
            "one live native mission: Hawking.invoke_tool filesystem.write + fs.read, "
            "WorkUnit with shasum -a 256 verifier, Hawking.start_mission/run, "
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
    return sc.stamp_gate(
        report,
        source="hawking/native_mission_gate.py::run_native_mission_gate",
        **payload,
    )

def run_native_mission_gate(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str] = None,
    emit: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one reversible tool/fact/verifier mission through native HAWKING."""
    from hawking import Hawking
    from hawking.controller import Controller

    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[1]
    profile_path = Path(profile).expanduser().resolve() if profile else (repo / "hawking" / "hawking-native.sealed-3.14.json").resolve()
    workspace = Path(tempfile.mkdtemp(prefix="hawking-native-mission-"))
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
        agent = Hawking(
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
            "source": "Hawking.invoke_tool(fs.read)",
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
        # Use the canonical Hawking boundary.  It derives acceptance from the
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
