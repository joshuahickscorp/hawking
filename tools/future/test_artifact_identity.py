"""Artifact identity must RAISE on a binary older than the field it is about to read."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tools.future import artifact_identity as ai
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


FIELD = "dispatches_per_generated_token"


def _binary(tmp_path: Path, contents: bytes, mtime: float) -> Path:
    path = tmp_path / "ascension_qwen38_resident"
    path.write_bytes(contents)
    os.utime(path, (mtime, mtime))
    return path


def test_inspect_raises_when_binary_predates_field_commit(tmp_path: Path):
    """Acceptance: inspect_artifact RAISES, it does not warn."""
    old = 1_700_000_000.0
    intro = 1_800_000_000.0
    binary = _binary(tmp_path, b"no instrumentation here", old)
    with pytest.raises(ai.StaleBinaryError) as ei:
        ai.inspect_artifact(
            binary,
            fields=[FIELD],
            field_introduced_unix={FIELD: intro},
            serving_mode="probe",
            measurement_mode="STATIC_ONLY",
            nx_id="test-nx",
            env={},
        )
    assert FIELD in str(ei.value)
    assert "REFUSED" in str(ei.value)
    ident = ei.value.identity
    assert ident["refused"] is True
    assert ident["stale_fields"]
    assert ident["stale_fields"][0]["field"] == FIELD
    assert ident["binary"]["sha256"]
    assert ident["binary"]["mtime_unix"] == old
    assert ident["nx"] == "test-nx"
    assert ident["environment"]["env_hash"]
    # Default must refuse. Passing refuse_stale is not required of the caller.
    with pytest.raises(ai.StaleBinaryError):
        ai.inspect_before_instrumentation(
            binary,
            fields=[FIELD],
            field_introduced_unix={FIELD: intro},
        )


def test_fresh_binary_records_identity_and_does_not_raise(tmp_path: Path):
    now = time.time()
    intro = now - 3600
    payload = FIELD.encode() + b"\x00active_bytes_per_token"
    binary = _binary(tmp_path, payload, now)
    ident = ai.inspect_artifact(
        binary,
        fields=[FIELD, "active_bytes_per_token"],
        field_introduced_unix={FIELD: intro, "active_bytes_per_token": intro},
        env=ai.SEALED_FUSION_ENV,
        serving_mode="resident",
        measurement_mode="DIAGNOSTIC_RELATIVE",
        feature_flags=ai.SEALED_FUSION_ENV,
        nx_id={"model_id": "test"},
    )
    assert ident["refused"] is False
    assert ident["stale_fields"] == []
    assert ident["binary"]["sha256"]
    assert ident["binary"]["bytes"] == len(payload)
    assert ident["source"]["dirty"] in (True, False)
    assert ident["feature_flags"]
    assert ident["environment"]["hawking_toggles"]
    assert ident["environment"]["serving_mode"] == "resident"
    assert ident["environment"]["measurement_mode"] == "DIAGNOSTIC_RELATIVE"
    assert ident["fields_in_binary"][FIELD] is True
    assert ident["nx"] == {"model_id": "test"}
    assert ident["resident_identity"] == ident["nx"]


def test_missing_binary_is_not_a_successful_inspect(tmp_path: Path):
    with pytest.raises(ai.ArtifactMissingError):
        ai.inspect_artifact(tmp_path / "no-such-binary", fields=[FIELD])


def test_refuse_stale_false_records_staleness_without_raising(tmp_path: Path):
    old = 1_700_000_000.0
    binary = _binary(tmp_path, b"x", old)
    ident = ai.inspect_artifact(
        binary,
        fields=[FIELD],
        field_introduced_unix={FIELD: old + 86_400},
        refuse_stale=False,
    )
    assert ident["refused"] is False
    assert ident["stale_fields"]
    assert ident["stale_fields"][0]["delta_s"] == 86_400


def test_mismatched_env_does_not_inherit_incumbent_label():
    sealed = ai.sealed_environment()
    unfused = ai.environment_identity(
        {}, serving_mode="probe", measurement_mode="DIAGNOSTIC_RELATIVE"
    )
    fused_probe = ai.environment_identity(
        ai.SEALED_FUSION_ENV,
        serving_mode="probe",
        measurement_mode="DIAGNOSTIC_RELATIVE",
    )
    fused_resident = ai.environment_identity(
        ai.SEALED_FUSION_ENV,
        serving_mode=ai.SEALED_SERVING_MODE,
        measurement_mode=ai.SEALED_MEASUREMENT_MODE,
    )
    unfused_label = ai.benchmark_label(unfused, sealed)
    probe_label = ai.benchmark_label(fused_probe, sealed)
    sealed_label = ai.benchmark_label(fused_resident, sealed)

    assert unfused["env_hash"] != sealed["env_hash"]
    assert unfused_label["inherited_incumbent"] is False
    assert unfused_label["label"] != ai.INCUMBENT_LABEL
    assert unfused_label["label"].startswith("env:")
    # Serving mode is in the hash: fusion flags alone do not earn the incumbent.
    assert probe_label["inherited_incumbent"] is False
    assert probe_label["label"] != ai.INCUMBENT_LABEL
    assert sealed_label["inherited_incumbent"] is True
    assert sealed_label["label"] == ai.INCUMBENT_LABEL
    assert fused_resident["env_hash"] == sealed["env_hash"]
    assert not ai.inherits_incumbent(unfused, sealed)
    assert ai.inherits_incumbent(fused_resident, sealed)


def test_env_hash_covers_hawking_toggles_not_just_fusion_flags():
    a = ai.environment_identity(
        {**ai.SEALED_FUSION_ENV, "HAWKING_QWEN38_FAST": "1"},
        serving_mode=ai.SEALED_SERVING_MODE,
        measurement_mode=ai.SEALED_MEASUREMENT_MODE,
    )
    b = ai.sealed_environment()
    assert a["env_hash"] != b["env_hash"]
    assert "HAWKING_QWEN38_FAST" in a["hawking_toggles"]
    assert ai.benchmark_label(a, b)["inherited_incumbent"] is False


def test_build_emits_sealed_receipt():
    out = ai.build()
    assert out.parent == RECEIPTS
    assert out.name == "ARTIFACT_IDENTITY.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == ai.SCHEMA
    assert doc["seal_sha256"]
    assert doc["bench"]["gpu_authority"] is False
    assert doc["unfused_does_not_inherit_incumbent"] is True
    assert doc["sealed_does_inherit_incumbent"] is True
    assert doc["sealed_environment"]["env_hash"]
    assert doc["refuse_rule"].startswith("inspect_artifact RAISES")
    assert doc["historical_stale_finding"]["scar_id"] == (
        "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE"
    )
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")
