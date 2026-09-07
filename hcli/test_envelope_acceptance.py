"""Deterministic envelope acceptance for cognition WorkUnits.

A unit with no shell verifier used to fall through to
``model_supplied_tests`` with empty tests and score ``NO_EVIDENCE`` forever.
``_route_validation`` now judges ``raw["result_envelope"]`` from disk facts
and a recorded run-and-rejected negative control. Verdict is model
self-grade and is never a pass condition.
"""
from __future__ import annotations

from pathlib import Path

from hcli.mission import Mission
from hcli.workunit import WorkUnit


class _TrapEngine:
    """Would ACCEPT if the envelope path fell through to ``_validate``."""

    def _validate(self, paths, tests=None, pre_mutation=None):
        return {
            "ok": True,
            "reason": "TRAP",
            "checks": [{"kind": "trap"}],
            "acceptance_source": "model_supplied_tests",
        }


class _NoEvidenceEngine:
    """Reproduces today's empty-tests outcome."""

    def _validate(self, paths, tests=None, pre_mutation=None):
        return {"ok": False, "reason": "NO_EVIDENCE", "checks": []}


def _mission(tmp_path, engine=None) -> Mission:
    return Mission(
        tmp_path,
        engine=engine if engine is not None else _TrapEngine(),
        quiet=True,
        no_progress_threshold=100,
    )


def _wu(**kwargs) -> WorkUnit:
    return WorkUnit(
        id=kwargs.pop("id", "u1"),
        role=kwargs.pop("role", "research"),
        description=kwargs.pop("description", "cognition unit"),
        **kwargs,
    )


def _write(tmp_path: Path, name: str, body: str = "measured\n") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _rejected_control(**extra):
    row = {"name": "false-world", "ran": True, "rejected": True}
    row.update(extra)
    return row


def _passed_control(**extra):
    row = {"name": "false-world", "ran": True, "rejected": False}
    row.update(extra)
    return row


def _raw(envelope, **extra):
    raw = {
        "result_envelope": envelope,
        # Trap: falling through to raw validation or model-supplied tests
        # would accept. Envelope-present failures must not take that hatch.
        "validation": {"ok": True, "reason": "TRAP_RAW_VALIDATION"},
        "tests": ["test_trap.py"],
    }
    raw.update(extra)
    return raw


def test_envelope_with_real_receipt_and_rejected_control_is_accepted(tmp_path):
    receipt = _write(tmp_path, "receipts/measured.json", '{"n": 1}\n')
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "claim": "the model grading itself",
                "verified_facts": [{"claim": "also a claim"}],
                "receipt_paths": [str(receipt)],
                "negative_controls": [_rejected_control()],
            }
        ),
    )
    assert val.get("ok") is True
    assert val.get("acceptance_source") == "envelope_receipts"


def test_fabricated_receipt_paths_are_rejected_despite_verdict_success(tmp_path):
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "receipt_paths": ["receipts/does-not-exist.json"],
                "negative_controls": [_rejected_control()],
            }
        ),
    )
    assert val.get("ok") is False
    assert val.get("acceptance_source") == "envelope_receipts"
    assert val.get("reason") == "ENVELOPE_PATH_MISSING"


def test_verdict_success_alone_is_rejected(tmp_path):
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "claim": "done",
                "verified_facts": [{"claim": "trust me"}],
            }
        ),
    )
    assert val.get("ok") is False
    assert val.get("acceptance_source") == "envelope_receipts"
    assert val.get("reason") != "TRAP"
    assert val.get("reason") != "TRAP_RAW_VALIDATION"


def test_negative_control_that_passed_is_rejected(tmp_path):
    receipt = _write(tmp_path, "receipts/measured.json", '{"n": 1}\n')
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "receipt_paths": [str(receipt)],
                "negative_controls": [_passed_control()],
            }
        ),
    )
    assert val.get("ok") is False
    assert val.get("acceptance_source") == "envelope_receipts"
    assert val.get("reason") == "ENVELOPE_CONTROL_PASSED"


def test_missing_or_empty_envelope_is_no_evidence(tmp_path):
    mission = _mission(tmp_path, engine=_NoEvidenceEngine())
    wu = _wu()
    for raw in (
        {},
        {"result_envelope": None},
        {"result_envelope": {}},
        {"result_envelope": []},
        {"result_envelope": "success"},
    ):
        val = mission._route_validation(wu, raw)
        assert val.get("ok") is False, raw
        assert val.get("reason") == "NO_EVIDENCE", raw
        assert val.get("acceptance_source") != "envelope_receipts", raw


def test_path_that_resolves_outside_the_workspace_is_rejected(tmp_path):
    outside = tmp_path.parent / f"envelope-outside-{tmp_path.name}.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        wu = _wu()
        relative = f"../{outside.name}"
        val = _mission(tmp_path)._route_validation(
            wu,
            _raw(
                {
                    "verdict": "success",
                    "receipt_paths": [relative],
                    "negative_controls": [_rejected_control()],
                }
            ),
        )
        assert val.get("ok") is False
        assert val.get("acceptance_source") == "envelope_receipts"
        assert val.get("reason") == "ENVELOPE_PATH_ESCAPE"

        val_abs = _mission(tmp_path)._route_validation(
            wu,
            _raw(
                {
                    "verdict": "success",
                    "receipt_paths": [str(outside.resolve())],
                    "negative_controls": [_rejected_control()],
                }
            ),
        )
        assert val_abs.get("ok") is False
        assert val_abs.get("reason") == "ENVELOPE_PATH_ESCAPE"
    finally:
        outside.unlink(missing_ok=True)


def test_unit_with_verifier_still_routes_to_workunit_verifier(tmp_path):
    receipt = _write(tmp_path, "receipts/measured.json", '{"n": 1}\n')
    wu = _wu(verifier="false")
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "receipt_paths": [str(receipt)],
                "negative_controls": [_rejected_control()],
            }
        ),
    )
    assert val.get("acceptance_source") == "workunit_verifier"
    assert val.get("ok") is False
    assert val.get("reason") == "VACUOUS_COMMAND"


def test_empty_receipt_file_is_rejected(tmp_path):
    receipt = _write(tmp_path, "receipts/empty.json", "")
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "receipt_paths": [str(receipt)],
                "negative_controls": [_rejected_control()],
            }
        ),
    )
    assert val.get("ok") is False
    assert val.get("reason") == "ENVELOPE_PATH_EMPTY"


def test_evidence_path_that_does_not_exist_is_rejected(tmp_path):
    receipt = _write(tmp_path, "receipts/measured.json", '{"n": 1}\n')
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "success",
                "receipt_paths": [str(receipt)],
                "evidence": [{"path": "missing-evidence.bin"}],
                "negative_controls": [_rejected_control()],
            }
        ),
    )
    assert val.get("ok") is False
    assert val.get("reason") == "ENVELOPE_PATH_MISSING"


def test_artifact_path_is_checkable_the_same_way_as_receipts(tmp_path):
    artifact = _write(tmp_path, "out/table.csv", "a,b\n1,2\n")
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "failure",
                "artifacts": [{"path": str(artifact.relative_to(tmp_path))}],
                "negative_controls": [_rejected_control()],
            }
        ),
    )
    assert val.get("ok") is True
    assert val.get("acceptance_source") == "envelope_receipts"


def test_engine_receipt_without_a_rejected_control_is_not_acceptance(tmp_path):
    """The engine always writes a receipt. That file is not a negative control."""
    receipt = _write(tmp_path, ".hcli/receipts/goal.json", '{"status": "completed"}\n')
    wu = _wu()
    val = _mission(tmp_path)._route_validation(
        wu,
        _raw(
            {
                "verdict": "ACCEPT",
                "receipt_paths": [str(receipt)],
                "negative_controls": None,
            }
        ),
    )
    assert val.get("ok") is False
    assert val.get("reason") == "ENVELOPE_NO_REJECTED_CONTROL"
    assert val.get("acceptance_source") == "envelope_receipts"
