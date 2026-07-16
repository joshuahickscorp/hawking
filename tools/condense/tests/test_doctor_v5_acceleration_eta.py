from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "condense" / "doctor_v5_acceleration_eta.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("doctor_v5_acceleration_eta_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_simulation_blockers_preserve_refusal() -> None:
    blockers = MODULE._simulation_blockers({
        "one": {"ok": False, "blocker": "blocked-execution"},
        "two": {"ok": True, "sub_120b_seconds": 1.0},
    })
    assert blockers == ["one: blocked-execution"]


def test_simulation_blockers_fail_closed_on_malformed_results() -> None:
    blockers = MODULE._simulation_blockers({
        "missing": {},
        "not_object": None,
    })
    assert blockers == [
        "missing: schedule unavailable",
        "not_object: simulator returned a non-object",
    ]


def test_status_mode_never_writes(monkeypatch, capsys) -> None:
    document = {"schema": MODULE.SCHEMA, "status": "unavailable-live-schedule"}
    monkeypatch.setattr(MODULE, "build", lambda: document)
    monkeypatch.setattr(MODULE, "validate", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        MODULE, "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("write")),
    )
    assert MODULE.main(["--status"]) == 0
    assert '"status": "unavailable-live-schedule"' in capsys.readouterr().out


def test_explicit_write_uses_requested_path(monkeypatch, tmp_path) -> None:
    document = {"schema": MODULE.SCHEMA, "status": "fixture"}
    writes: list[tuple[pathlib.Path, dict]] = []
    monkeypatch.setattr(MODULE, "build", lambda: document)
    monkeypatch.setattr(MODULE, "validate", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "_atomic_json", lambda path, value: writes.append((path, value)))
    output = tmp_path / "eta.json"
    assert MODULE.main(["--write", str(output)]) == 0
    assert writes == [(output, document)]


def _reseal(value: dict) -> None:
    value["eta_sha256"] = MODULE._hash_value(
        {key: row for key, row in value.items() if key != "eta_sha256"}
    )


def test_current_build_withholds_dates_and_campaign_transferability() -> None:
    value = MODULE.build()
    assert value["status"] in MODULE.CLAIM_LIMITS
    assert value["speedup_evidence"]["transferable_to_campaign_eta"] is False
    assert "real_qwen_tensor_end_to_end_speedup" not in value["speedup_evidence"]
    assert value["speedup_evidence"]["single_tensor_encode_speedup"] > 1
    for name in ("sub_120b", "through_120b", "through_120b_plus_appendix"):
        assert value[name]["available"] is False
        assert value[name]["date_range"] is None
    assert value["mechanical_sensitivity"]["not_an_eta"] is True
    assert value["mechanical_sensitivity"]["calendar_dates_emitted"] is False
    assert MODULE.validate(value, verify_freshness=True) == []


def test_sub_tensor_speedup_never_reaches_through_120b_simulation(monkeypatch) -> None:
    calls: list[tuple[float, bool, int]] = []

    def fake_simulate(_plan, _campaign, *, speedup, include_unready_120b,
                      max_lanes, **_kwargs):
        calls.append((float(speedup), bool(include_unready_120b), int(max_lanes)))
        return {
            "ok": True,
            "sub_120b_seconds": 100.0 if max_lanes == 1 else 80.0,
            "through_120b_seconds": 300.0 if max_lanes == 1 else 250.0,
        }

    monkeypatch.setattr(MODULE.stacked, "simulate", fake_simulate)
    value = MODULE.build()
    tensor_speedup = value["speedup_evidence"]["single_tensor_encode_speedup"]
    assert {(speedup, include) for speedup, include, _lane in calls} == {
        (tensor_speedup, False),
        (1.0, True),
    }
    assert value["through_120b"]["available"] is False
    assert value["through_120b_plus_appendix"]["available"] is False
    assert value["mechanical_sensitivity"]["sub_120b_speedup_applied"] == 1.0
    assert value["mechanical_sensitivity"][
        "observed_sub_120b_speedup_excluded"
    ] == tensor_speedup
    assert value["sub_120b"]["not_an_eta"] is True
    assert value["sub_120b"]["date_range"] is None


def test_resealed_date_and_unknown_field_attacks_are_rejected() -> None:
    base = MODULE.build()
    attacks = []
    through_date = json.loads(json.dumps(base))
    through_date["through_120b"]["date_range"] = ["2026-07-20", "2026-07-21"]
    attacks.append(through_date)
    mechanical_date = json.loads(json.dumps(base))
    mechanical_date["mechanical_sensitivity"]["date_range"] = [
        "2026-07-20", "2026-07-21"
    ]
    attacks.append(mechanical_date)
    top_date = json.loads(json.dumps(base))
    top_date["completion_date"] = "2026-07-20"
    attacks.append(top_date)
    claim_drift = json.loads(json.dumps(base))
    claim_drift["claim_limits"] = ["trust me"]
    attacks.append(claim_drift)
    for value in attacks:
        _reseal(value)
        assert MODULE.validate(value, verify_freshness=False)


def test_resealed_fictitious_canary_binding_is_rejected() -> None:
    value = MODULE.build()
    value["speedup_evidence"]["receipt_path"] = "/tmp/fictitious.json"
    value["speedup_evidence"]["receipt_file_sha256"] = "f" * 64
    _reseal(value)
    errors = MODULE.validate(value, verify_freshness=True)
    assert "single-tensor receipt path is non-canonical" in errors
