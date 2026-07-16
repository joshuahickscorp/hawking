from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_cheap_gates.py"
SPEC = importlib.util.spec_from_file_location("appendix_cheap_gates", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fake_report() -> dict:
    source = MODULE.current_source_capsule()
    execution = MODULE.execution_authority()
    command_manifest = MODULE._command_manifest(MODULE.GATES)
    rows = [
        {
            "id": gate_id,
            "command": command,
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_ns": 1,
            "stdout_sha256": MODULE._digest_text(""),
            "stderr_sha256": MODULE._digest_text(""),
            "stdout_tail": "",
            "stderr_tail": "",
        }
        for gate_id, command in MODULE.GATES
    ]
    return MODULE._stamp({
        "schema": MODULE.SCHEMA,
        "source_commit": "0123456789abcdef",
        "source_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": MODULE.appendix_contract.canonical_sha256(command_manifest),
        "source_capsule": source,
        "source_capsule_sha256": source["capsule_sha256"],
        "source_capsule_after_sha256": source["capsule_sha256"],
        "source_capsule_stable_during_run": True,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution["authority_sha256"],
        "execution_authority_stable_during_run": True,
        "uses_gpu": False,
        "reads_model_artifacts": False,
        "mutates_active_corpus": False,
        "cargo_build_jobs": "2",
        "active_heavy_owner_count_before": 2,
        "active_heavy_owner_count_after": 2,
        "gate_count": len(rows),
        "passed_count": len(rows),
        "failed_count": 0,
        "gates": rows,
    })


def test_fake_all_pass_report_validates() -> None:
    assert MODULE.verify_report(_fake_report()) == []


def test_failure_and_tampering_are_not_hidden() -> None:
    report = _fake_report()
    report["gates"][0]["passed"] = False
    report["gates"][0]["exit_code"] = 1
    report["passed_count"] -= 1
    report["failed_count"] = 1
    report = MODULE._stamp(report)
    assert any("gate failed" in error for error in MODULE.verify_report(report))

    tampered = copy.deepcopy(_fake_report())
    tampered["passed_count"] = 0
    assert "report_sha256 mismatch" in MODULE.verify_report(tampered)


def test_main_report_rejects_restamped_command_drift() -> None:
    report = _fake_report()
    report["gates"][0]["command"] = ["python3.12", "unexpected.py"]
    report = MODULE._stamp(report)
    assert any("command differs" in error for error in MODULE.verify_report(report))


def test_main_report_rejects_restamped_dirty_source_drift(
    monkeypatch,
) -> None:
    report = _fake_report()
    changed = copy.deepcopy(report["source_capsule"])
    changed["entries"][0]["sha256"] = "f" * 64
    changed["entries_sha256"] = MODULE.appendix_contract.canonical_sha256(
        changed["entries"]
    )
    changed.pop("capsule_sha256")
    changed["capsule_sha256"] = MODULE.appendix_contract.canonical_sha256(changed)
    monkeypatch.setattr(MODULE, "current_source_capsule", lambda: changed)
    assert any("source bytes drifted" in error for error in MODULE.verify_report(report))


def test_source_capsule_hashes_current_dirty_tree_bytes(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    source = tmp_path / "tools" / "condense" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "SOURCE_CAPSULE_GLOBS", ("tools/condense/*.py",))
    first = MODULE.current_source_capsule()
    assert MODULE.current_source_capsule() == first
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = MODULE.current_source_capsule()
    assert first["capsule_sha256"] != second["capsule_sha256"]
    assert first["entries"][0]["sha256"] != second["entries"][0]["sha256"]


def test_release_packet_extension_is_owner_safe_exact_and_separately_receipted() -> None:
    source = MODULE.current_source_capsule()
    execution = MODULE.execution_authority()
    command_manifest = MODULE._command_manifest(MODULE.RELEASE_PACKET_GATES)
    rows = [
        {
            "id": gate_id, "command": command, "exit_code": 0, "passed": True,
            "timed_out": False, "duration_ns": 1,
            "stdout_sha256": MODULE._digest_text(""),
            "stderr_sha256": MODULE._digest_text(""),
            "stdout_tail": "", "stderr_tail": "",
        }
        for gate_id, command in MODULE.RELEASE_PACKET_GATES
    ]
    report = MODULE._stamp({
        "schema": MODULE.RELEASE_PACKET_SCHEMA,
        "source_base_commit": "a" * 40,
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": MODULE.appendix_contract.canonical_sha256(command_manifest),
        "source_capsule": source,
        "source_capsule_sha256": source["capsule_sha256"],
        "source_capsule_after_sha256": source["capsule_sha256"],
        "source_capsule_stable_during_run": True,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution["authority_sha256"],
        "execution_authority_stable_during_run": True,
        "uses_gpu": False, "reads_model_artifacts": False,
        "opens_or_hashes_active_corpus": False, "runs_cargo": False,
        "mutates_active_corpus": False, "mutates_runtime_defaults": False,
        "active_heavy_owner_count_before": 4,
        "active_heavy_owner_count_after": 4,
        "gate_count": len(rows), "passed_count": len(rows), "failed_count": 0,
        "gates": rows,
    })
    assert MODULE.verify_release_packet_report(report) == []
    report["runs_cargo"] = True
    report = MODULE._stamp(report)
    assert any(
        "runs_cargo" in error for error in MODULE.verify_release_packet_report(report)
    )


def test_report_rejects_restamped_environment_or_tool_drift(monkeypatch) -> None:
    report = _fake_report()
    report["execution_authority"]["environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
    environment = report["execution_authority"]["environment"]
    report["execution_authority"]["environment_sha256"] = (
        MODULE.appendix_contract.canonical_sha256(environment)
    )
    authority = report["execution_authority"]
    authority.pop("authority_sha256")
    authority["authority_sha256"] = MODULE.appendix_contract.canonical_sha256(authority)
    report["execution_authority_sha256"] = authority["authority_sha256"]
    report["execution_authority_after_sha256"] = authority["authority_sha256"]
    report = MODULE._stamp(report)
    assert any(
        "authority drifted" in error for error in MODULE.verify_report(report)
    )

    tool_drift = _fake_report()
    tool_drift["execution_authority"]["tools"][0]["sha256"] = "f" * 64
    tools = tool_drift["execution_authority"]["tools"]
    tool_drift["execution_authority"]["tools_sha256"] = (
        MODULE.appendix_contract.canonical_sha256(tools)
    )
    authority = tool_drift["execution_authority"]
    authority.pop("authority_sha256")
    authority["authority_sha256"] = MODULE.appendix_contract.canonical_sha256(authority)
    tool_drift["execution_authority_sha256"] = authority["authority_sha256"]
    tool_drift["execution_authority_after_sha256"] = authority["authority_sha256"]
    tool_drift = MODULE._stamp(tool_drift)
    assert any("authority drifted" in error for error in MODULE.verify_report(tool_drift))
