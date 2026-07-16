from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib
import struct
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
SPEC = importlib.util.spec_from_file_location(
    "doctor_v5_production_eta", CONDENSE / "doctor_v5_production_eta.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reference(label: str = "fixture") -> dict:
    return {"path": f"{label}.json", "sha256": "a" * 64, "bytes": 1}


def _bindings() -> dict:
    return {
        "plan": _reference("plan"),
        "campaign": _reference("campaign"),
        "observer_state": _reference("observer"),
        "plan_sha256": "b" * 64,
        "campaign_sha256": "c" * 64,
        "observer_state_sha256": "d" * 64,
    }


def _reseal(value: dict) -> None:
    value["document_sha256"] = MODULE._hash_value(
        {key: row for key, row in value.items() if key != "document_sha256"}
    )


def _calibration() -> dict:
    return {
        "cell_id": "fixture-cell",
        "cell_identity_sha256": "e" * 64,
        "model_label": "3B",
        "branch_rate": "codec_control@4",
        "log_artifacts": [{
            "path": "/fixture/encode.log", "sha256": "f" * 64,
            "bytes": 1, "source_path": "/fixture/model.safetensors",
            "attempt_count": 1, "completed_tensor_count": 1,
            "completed_weights": 1,
        }],
        "queue_started_at": "2026-07-14T19:00:00+00:00",
        "first_encode_started_at": "2026-07-14T19:01:00+00:00",
        "queue_attempts": 1,
        "completed_weights": 1,
        "total_two_dimensional_weights": 2,
        "progress_fraction": 0.5,
        "elapsed_seconds": 3600.0,
        "observed_weights_per_second": 1.0,
        "projected_full_cell_seconds": 7200.0,
        "legacy_cell_seconds": 14_400.0,
        "sub_120b_observed_speedup": 2.0,
        "transferable_to_gpt_oss_120b": False,
    }


def _valid(*, bindings: dict | None = None) -> dict:
    created_at = "2026-07-14T20:00:00+00:00"
    anchor = dt.datetime.fromisoformat(created_at)
    sub = [86_400.0, 172_800.0]
    through = [259_200.0, 345_600.0]
    value = {
        "schema": MODULE.SCHEMA,
        "created_at": created_at,
        "status": "provisional-live-production-calibration",
        "eta_scope": "sub-120b-only",
        "calibration_available": True,
        "eta_blocked": False,
        "input_bindings": bindings or _bindings(),
        "calibration": _calibration(),
        "sub_120b": {
            "available": True, "seconds_range": sub,
            "days_range": [row / 86_400 for row in sub],
            "date_range": [MODULE._date(anchor, row) for row in sub],
        },
        "through_120b": MODULE._gated_segment(appendix=False),
        "through_120b_plus_appendix": MODULE._gated_segment(appendix=True),
        "mechanical_sensitivity": MODULE._mechanical_sensitivity(
            observed_sub_120b_speedup=2.0, through_range=through,
        ),
        "claim_limits": MODULE.CLAIM_LIMITS[
            "provisional-live-production-calibration"
        ],
    }
    _reseal(value)
    return value


def test_validation_accepts_a_hashed_fail_closed_eta() -> None:
    assert MODULE.validate(_valid(), verify_freshness=False) == []


def test_validation_rejects_an_execution_overclaim_even_when_rehashed() -> None:
    value = _valid()
    value["through_120b"]["execution_ready"] = True
    value["through_120b"]["date_range"] = ["2026-07-20", "2026-07-21"]
    _reseal(value)
    assert "unreceipted segment exposes ETA data" in MODULE.validate(
        value, verify_freshness=False
    )


def test_validation_rejects_tampering() -> None:
    value = json.loads(json.dumps(_valid()))
    value["sub_120b"]["seconds_range"][0] = 1.0
    assert "production ETA document hash differs" in MODULE.validate(
        value, verify_freshness=False
    )


def test_validation_rejects_resealed_mechanical_math_or_sub_date_drift() -> None:
    cases = []
    wrong_appendix = _valid()
    wrong_appendix["mechanical_sensitivity"][
        "through_120b_plus_appendix_seconds_range"
    ][0] += 1.0
    cases.append(wrong_appendix)

    wrong_date = _valid()
    wrong_date["sub_120b"]["date_range"][0] = "2026-07-16T20:00:00+00:00"
    cases.append(wrong_date)

    missing_availability = _valid()
    del missing_availability["through_120b"]["available"]
    cases.append(missing_availability)

    unknown_calibration = _valid()
    unknown_calibration["calibration"]["claimed_confidence"] = "absolute"
    cases.append(unknown_calibration)

    weakened_claims = _valid()
    weakened_claims["claim_limits"] = ["trust caller"]
    cases.append(weakened_claims)

    for value in cases:
        _reseal(value)
        assert MODULE.validate(value, verify_freshness=False)


def test_validation_accepts_blocked_state_without_a_date() -> None:
    value = _valid()
    value["status"] = "blocked-live-production-calibration"
    value["eta_blocked"] = True
    value["blockers"] = ["one or more cells are blocked-execution"]
    value["failed_simulations"] = ["sub_120b_one_lane"]
    value["claim_limits"] = MODULE.CLAIM_LIMITS[
        "blocked-live-production-calibration"
    ]
    value["sub_120b"] = MODULE._empty_sub_120b()
    value["mechanical_sensitivity"] = MODULE._mechanical_sensitivity(
        observed_sub_120b_speedup=2.0, through_range=None,
        blockers=["one or more cells are blocked-execution"],
    )
    _reseal(value)
    assert MODULE.validate(value, verify_freshness=False) == []
    duplicate = json.loads(json.dumps(value))
    duplicate["blockers"] *= 2
    duplicate["failed_simulations"] *= 2
    _reseal(duplicate)
    assert "blocked production ETA contract is invalid" in MODULE.validate(
        duplicate, verify_freshness=False
    )


def test_validation_rejects_rehashed_blocked_state_with_a_date() -> None:
    value = _valid()
    value["status"] = "blocked-live-production-calibration"
    value["eta_blocked"] = True
    value["blockers"] = ["one or more cells are blocked-execution"]
    value["failed_simulations"] = ["sub_120b_one_lane"]
    value["claim_limits"] = MODULE.CLAIM_LIMITS[
        "blocked-live-production-calibration"
    ]
    value["sub_120b"] = MODULE._empty_sub_120b()
    value["sub_120b"]["completion_date"] = "2026-07-20"
    value["mechanical_sensitivity"] = MODULE._mechanical_sensitivity(
        observed_sub_120b_speedup=2.0, through_range=None,
    )
    _reseal(value)
    assert "blocked production ETA contract is invalid" in MODULE.validate(
        value, verify_freshness=False
    )


def test_validation_refuses_malformed_children_and_nonfinite_values_cleanly() -> None:
    for field in ("calibration", "sub_120b", "through_120b"):
        value = _valid()
        value[field] = []
        _reseal(value)
        assert MODULE.validate(value, verify_freshness=False)

    for speedup in (float("inf"), float("nan"), True):
        value = _valid()
        value["calibration"]["sub_120b_observed_speedup"] = speedup
        # Non-finite documents cannot be canonically sealed; validation must
        # still return errors rather than raising.
        value["document_sha256"] = "0" * 64
        errors = MODULE.validate(value, verify_freshness=False)
        assert "sub-120B production speedup boundary is not proven" in errors

    value = _valid()
    value["sub_120b"]["seconds_range"] = [True, 2.0]
    _reseal(value)
    assert "sub-120B ETA range is invalid" in MODULE.validate(
        value, verify_freshness=False
    )


def test_validation_accepts_sealed_unavailable_calibration_without_a_date() -> None:
    value = _valid()
    value["status"] = "unavailable-live-production-calibration"
    value["calibration_available"] = False
    value["eta_blocked"] = True
    value["blockers"] = ["production progress does not yet prove acceleration"]
    value["claim_limits"] = MODULE.CLAIM_LIMITS[
        "unavailable-live-production-calibration"
    ]
    del value["calibration"]
    value["sub_120b"] = MODULE._empty_sub_120b()
    value["mechanical_sensitivity"] = MODULE._mechanical_sensitivity(
        observed_sub_120b_speedup=None, through_range=None,
        blockers=value["blockers"],
    )
    _reseal(value)
    assert MODULE.validate(value, verify_freshness=False) == []

    for container, field in (
        ("sub_120b", "completion_date"),
        ("through_120b", "completion_date"),
        ("mechanical_sensitivity", "date_range"),
    ):
        forged = json.loads(json.dumps(value))
        forged[container][field] = ["2026-07-15", "2026-07-16"]
        _reseal(forged)
        assert MODULE.validate(forged, verify_freshness=False)


def test_fresh_input_bindings_reject_stale_or_tampered_sources(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {
        "plan": tmp_path / "plan.json",
        "campaign": tmp_path / "campaign.json",
        "observer_state": tmp_path / "observer.json",
    }
    plan = {"kind": "plan"}
    plan["plan_sha256"] = MODULE._hash_value(plan)
    campaign = {"kind": "campaign", "plan_sha256": plan["plan_sha256"]}
    campaign["campaign_sha256"] = MODULE._hash_value(campaign)
    observer = {"kind": "observer"}
    observer["state_sha256"] = MODULE._hash_value(observer)
    for path, document in zip(paths.values(), (plan, campaign, observer), strict=True):
        path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(MODULE, "INPUT_BINDING_PATHS", paths)
    references = {
        name: MODULE._read_bound_json(path)[1] for name, path in paths.items()
    }
    bindings = {
        **references,
        "plan_sha256": plan["plan_sha256"],
        "campaign_sha256": campaign["campaign_sha256"],
        "observer_state_sha256": observer["state_sha256"],
    }
    monkeypatch.setattr(
        MODULE, "_read_inputs", lambda: (plan, campaign, observer, bindings)
    )
    monkeypatch.setattr(MODULE, "_calibration", lambda *_args: _calibration())
    value = _valid(bindings=bindings)
    assert MODULE.validate(value, verify_freshness=True) == []
    forged = json.loads(json.dumps(value))
    forged["calibration"]["sub_120b_observed_speedup"] = 999.0
    _reseal(forged)
    assert "production calibration differs from current bound inputs" in MODULE.validate(
        forged, verify_freshness=True
    )
    paths["campaign"].write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert "ETA input binding is stale: campaign" in MODULE.validate(
        value, verify_freshness=True
    )


def test_current_build_remains_blocked_and_emits_no_120b_or_appendix_date() -> None:
    value = MODULE.build()
    assert value["status"] == "unavailable-live-production-calibration"
    assert value["eta_blocked"] is True
    assert value["blockers"]
    assert all(isinstance(blocker, str) and blocker for blocker in value["blockers"])
    for name in ("sub_120b", "through_120b", "through_120b_plus_appendix"):
        assert value[name]["available"] is False
        assert value[name]["date_range"] is None
    assert value["mechanical_sensitivity"]["not_an_eta"] is True
    assert value["mechanical_sensitivity"]["calendar_dates_emitted"] is False
    assert MODULE.validate(value, verify_freshness=True) == []


def _write_safetensor(path: pathlib.Path, rows: int = 1, columns: int = 1) -> None:
    header = json.dumps({
        "w": {"dtype": "F16", "shape": [rows, columns],
              "data_offsets": [0, 2 * rows * columns]},
    }, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header
                     + b"\0" * (2 * rows * columns))


def _write_log(path: pathlib.Path, source: pathlib.Path,
               starts: list[dt.datetime]) -> None:
    lines: list[str] = []
    for started in starts:
        lines.append(json.dumps({
            "event": "child_start", "at": started.isoformat(),
            "argv": ["quantizer", "--in", str(source)],
        }, separators=(",", ":")))
        lines.append("[done 1/1] w")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_log_retry_snapshot_and_calibration_source_time_gates(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = dt.datetime(2026, 7, 14, 20, 0, tzinfo=dt.timezone.utc)
    active = tmp_path / "model" / "model-00001.safetensors"
    foreign = tmp_path / "foreign.safetensors"
    _write_safetensor(active)
    _write_safetensor(foreign, 1000, 1000)
    log_root = tmp_path / "results" / "active" / "strand_ladder" / "logs"
    log = log_root / "encode-00000.log"
    _write_log(log, active, [now - dt.timedelta(minutes=20),
                             now - dt.timedelta(minutes=10)])
    parsed = MODULE._parse_log(log)
    assert parsed["attempt_count"] == 2
    assert parsed["completed_tensor_count"] == 1

    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "RESULTS", tmp_path / "results")
    plan = {"cells": [{
        "cell_id": "active", "cell_identity_sha256": "a" * 64,
        "model_family": "qwen2.5-dense", "model_label": "1B",
        "model_dir": "model", "branch": "codec_control", "rate_id": "3",
        "exact_stored_parameter_count": 1,
    }]}
    campaign = {
        "active_cells": ["active"],
        "cells": [{"cell_id": "active", "status": "running", "attempts": 2,
                   "started_at": (now - dt.timedelta(hours=1)).isoformat()}],
    }

    _write_log(log, foreign, [now - dt.timedelta(minutes=1)])
    with pytest.raises(MODULE.ProductionEtaError, match="unique active-model"):
        MODULE._calibration(plan, campaign, now)

    _write_log(log, active, [now + dt.timedelta(hours=1)])
    with pytest.raises(MODULE.ProductionEtaError, match="future-dated"):
        MODULE._calibration(plan, campaign, now)

    _write_log(log, active, [now - dt.timedelta(minutes=1)])
    _write_log(log_root / "encode-00001.log", active,
               [now - dt.timedelta(seconds=30)])
    with pytest.raises(MODULE.ProductionEtaError, match="unique active-model"):
        MODULE._calibration(plan, campaign, now)
