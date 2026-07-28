from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest

EVAL = pathlib.Path(__file__).resolve().parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import support_halo_gate as gate  # noqa: E402


def _attestation(digest: str) -> dict:
    return {
        "status": "ok",
        "runtime": "base",
        "base_runtime": True,
        "fallback_present": False,
        "artifact_index_sha256": digest,
        "model_id": "glm52-candidate",
        "architecture": "glm",
    }


def test_artifact_index_accepts_exactly_one_supported_index(tmp_path):
    activation = tmp_path / "model.activation_aware.index.json"
    activation.write_text("{}")
    assert gate.artifact_index(tmp_path) == activation

    (tmp_path / "model.gravity.index.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        gate.artifact_index(tmp_path)


def test_runtime_attestation_binds_hash_model_and_no_fallback():
    digest = hashlib.sha256(b"index").hexdigest()
    accepted = gate.validate_runtime_attestation(
        _attestation(digest),
        expected_index_sha256=digest,
        requested_model="glm52-candidate",
    )
    assert accepted["fallback_present"] is False
    assert accepted["artifact_index_sha256"] == digest

    wrong_hash = _attestation("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        gate.validate_runtime_attestation(
            wrong_hash,
            expected_index_sha256=digest,
        )

    fallback = _attestation(digest)
    fallback["fallback_present"] = True
    with pytest.raises(ValueError, match="fallback"):
        gate.validate_runtime_attestation(
            fallback,
            expected_index_sha256=digest,
        )


def test_runtime_attestation_refuses_model_alias_drift():
    digest = hashlib.sha256(b"index").hexdigest()
    with pytest.raises(ValueError, match="differs"):
        gate.validate_runtime_attestation(
            _attestation(digest),
            expected_index_sha256=digest,
            requested_model="some-other-model",
        )
