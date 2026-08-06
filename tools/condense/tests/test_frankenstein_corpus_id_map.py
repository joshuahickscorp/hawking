#!/usr/bin/env python3.12
"""Content-hash corpus ID reconciliation tests (CPU only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators.frankenstein_corpus_id_map import (
    content_key,
    emit_reconciliation,
    load_dsv4f_trace_corpus,
    load_glm_frozen_corpus,
    load_proto_corpus,
    pair_by_content_key,
    pair_by_example_id,
    reconcile,
)

REPO = Path(__file__).resolve().parents[3]
PROTO_L0 = (
    REPO
    / "workspace/campaign/evidence/models/frankenstein/corpus/PROTO_FRANKENSTEIN_V0_L0_CORPUS.jsonl"
)
GLM_L0 = (
    REPO
    / "workspace/campaign/evidence/models/frankenstein/teacher_forced"
    / "official_L0_stream_full_20260805T200728Z/FROZEN_CORPUS_L0.json"
)
DSV_L0 = REPO / "receipts/dsv4f_fullseq_capture_L0/traces"


def test_content_key_is_sha256_utf8() -> None:
    text = "Evaluate (17 × 29) + 7."
    assert content_key(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.skipif(not PROTO_L0.is_file(), reason="PROTO L0 corpus missing")
@pytest.mark.skipif(not GLM_L0.is_file(), reason="GLM frozen L0 missing")
def test_glm_frozen_identical_to_proto_by_id_and_content() -> None:
    proto = load_proto_corpus(PROTO_L0)
    glm = load_glm_frozen_corpus(GLM_L0)
    id_pair = pair_by_example_id(proto, glm)
    ck_pair = pair_by_content_key(proto, glm)
    assert len(proto) == 32
    assert len(glm) == 32
    assert id_pair["identical_id_and_content"] is True
    assert id_pair["n_shared_ids"] == 32
    assert id_pair["n_content_match"] == 32
    assert id_pair["n_content_mismatch"] == 0
    assert ck_pair["n_pairs"] == 32
    assert all(p["ids_equal"] for p in ck_pair["pairs"])


@pytest.mark.skipif(not PROTO_L0.is_file(), reason="PROTO L0 corpus missing")
@pytest.mark.skipif(not DSV_L0.is_dir(), reason="DSV4F L0 traces missing")
def test_legacy_dsv4f_synthetic_has_zero_content_overlap() -> None:
    proto = load_proto_corpus(PROTO_L0)
    dsv = load_dsv4f_trace_corpus(DSV_L0)
    assert len(dsv) == 32
    # Legacy IDs look like v0_math_* not pfv0:*
    assert any(r["example_id"].startswith("v0_") for r in dsv)
    assert not any(r["example_id"].startswith("pfv0:") for r in dsv)
    id_pair = pair_by_example_id(proto, dsv)
    ck_pair = pair_by_content_key(proto, dsv)
    assert id_pair["n_shared_ids"] == 0
    assert ck_pair["n_pairs"] == 0
    assert ck_pair["n_shared_content_keys"] == 0


@pytest.mark.skipif(not PROTO_L0.is_file(), reason="PROTO L0 corpus missing")
def test_reconcile_status_glm_aligned_dsv_needs_recapture() -> None:
    if not GLM_L0.is_file() or not DSV_L0.is_dir():
        pytest.skip("captures missing")
    body = reconcile(
        proto_path=PROTO_L0,
        glm_path=GLM_L0,
        dsv4f_traces=DSV_L0,
        ladder="L0",
    )
    assert body["fabricated"] is False
    assert body["glm"]["identical_to_canonical"] is True
    assert body["dsv4f"]["identical_to_canonical"] is False
    assert body["dsv4f"]["by_content_key"]["n_pairs"] == 0
    assert body["status"] == "GLM_ALIGNED_DSV4F_NEEDS_RECAPTURE"
    assert body["correspondence_ready"] is False
    assert body["normalization_map"]["glm_to_canonical"]
    assert len(body["normalization_map"]["glm_to_canonical"]) == 32
    assert body["normalization_map"]["dsv4f_to_canonical"] == {}


def test_content_key_pairing_on_fixture(tmp_path: Path) -> None:
    """When DSV4F uses the same text (even with a different label), content_key maps."""
    proto = [
        {
            "example_id": "pfv0:demo:a",
            "prompt_text": "hello world",
            "content_key": content_key("hello world"),
            "membership": "train",
        },
        {
            "example_id": "pfv0:demo:b",
            "prompt_text": "other",
            "content_key": content_key("other"),
            "membership": "train",
        },
    ]
    dsv = [
        {
            "example_id": "v0_wrong_label",
            "prompt_text": "hello world",
            "content_key": content_key("hello world"),
            "membership": "train",
        },
    ]
    ck = pair_by_content_key(proto, dsv)
    assert ck["n_pairs"] == 1
    assert ck["pairs"][0]["left_example_id"] == "pfv0:demo:a"
    assert ck["pairs"][0]["right_example_id"] == "v0_wrong_label"
    assert ck["pairs"][0]["ids_equal"] is False

    # Label-only match with different content is NOT a pair by content_key.
    dsv_label_only = [
        {
            "example_id": "pfv0:demo:a",
            "prompt_text": "DIFFERENT TEXT",
            "content_key": content_key("DIFFERENT TEXT"),
            "membership": "train",
        }
    ]
    id_pair = pair_by_example_id(proto, dsv_label_only)
    assert id_pair["n_shared_ids"] == 1
    assert id_pair["n_content_mismatch"] == 1
    assert id_pair["identical_id_and_content"] is False


def test_emit_reconciliation_seals(tmp_path: Path) -> None:
    if not (PROTO_L0.is_file() and GLM_L0.is_file() and DSV_L0.is_dir()):
        pytest.skip("captures missing")
    out = tmp_path / "recon.json"
    doc = emit_reconciliation(
        proto_path=PROTO_L0,
        glm_path=GLM_L0,
        dsv4f_traces=DSV_L0,
        ladder="L0",
        out_path=out,
        write=True,
    )
    assert out.is_file()
    assert "seal_sha256" in doc
    loaded = json.loads(out.read_text())
    assert loaded["seal_sha256"] == doc["seal_sha256"]
    assert loaded["fabricated"] is False
