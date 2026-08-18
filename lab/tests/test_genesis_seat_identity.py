"""Focused identity tests for seating Genesis without launching it."""
from __future__ import annotations

import hashlib

import pytest

from lab.lineage.identity import (
    GenesisInstance,
    IdentityError,
    file_sha256,
    make_qwen38_genesis,
)
from lab.lineage.state import LineageState
from tools import genesis_seat
from tools.genesis_seat import build_seated_lineage


def test_file_sha256_hashes_manifest_bytes(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    raw = b'{"artifact":"qwen38-test","version":1}\n'
    manifest.write_bytes(raw)

    assert file_sha256(manifest) == hashlib.sha256(raw).hexdigest()


def test_file_sha256_refuses_missing_manifest(tmp_path) -> None:
    with pytest.raises(IdentityError, match="not a file"):
        file_sha256(tmp_path / "missing-manifest.json")


def test_new_qwen38_identity_uses_supplied_manifest_sha_without_liveness() -> None:
    artifact_sha = hashlib.sha256(b"real manifest bytes").hexdigest()
    genesis = make_qwen38_genesis(artifact_sha=artifact_sha)

    assert genesis.artifact_sha == artifact_sha
    assert genesis.identity["artifact_sha_authority"] == "sha256(manifest.json bytes)"
    assert genesis.live is False
    assert genesis.launched is False


def test_physical_bpw_round_trips_and_refuses_non_finite_or_boolean() -> None:
    genesis = make_qwen38_genesis()
    genesis.physical_bpw = 4.252735126866492
    assert GenesisInstance.from_mapping(genesis.to_dict()).physical_bpw == pytest.approx(
        4.252735126866492
    )
    for invalid in (True, float("nan"), float("inf")):
        raw = genesis.to_dict()
        raw["physical_bpw"] = invalid
        with pytest.raises(IdentityError, match="physical_bpw"):
            GenesisInstance.from_mapping(raw)


def test_installing_artifact_authority_does_not_synthesize_process_liveness() -> None:
    lineage = LineageState()
    snapshot = lineage.install(make_qwen38_genesis())
    current = snapshot["slots"]["CURRENT"]

    assert current["live"] is False
    assert current["launched"] is False


def test_build_seated_lineage_hashes_manifest_and_does_not_claim_process(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    raw = b'{"artifact":"qwen38-seat-test","version":1}\n'
    manifest.write_bytes(raw)

    lineage, genesis = build_seated_lineage(artifact_manifest=manifest)
    snapshot = lineage.to_dict()
    current = snapshot["slots"]["CURRENT"]
    lkg = snapshot["slots"]["LAST_KNOWN_GOOD"]

    assert genesis.artifact_sha == hashlib.sha256(raw).hexdigest()
    assert current["artifact_sha"] == genesis.artifact_sha
    assert current["live"] is False
    assert current["launched"] is False
    assert lkg["live"] is False
    assert lkg["launched"] is False


def test_seat_atomically_serializes_with_trailing_newline(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"artifact":"qwen38-seat-test","version":1}\n')
    lineage, instance = build_seated_lineage(artifact_manifest=manifest)
    state_file = tmp_path / "GENESIS_LINEAGE_CURRENT.json"
    state_file.write_text("previous state\n")
    monkeypatch.setattr(genesis_seat, "REPO", tmp_path)
    monkeypatch.setattr(genesis_seat, "STATE_FILE", state_file)
    monkeypatch.setattr(
        genesis_seat,
        "build_seated_lineage",
        lambda: (lineage, instance),
    )

    assert genesis_seat.seat() == 0
    raw = state_file.read_bytes()
    assert raw.endswith(b"\n")
    assert b"previous state" not in raw
