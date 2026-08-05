"""Offline contracts for Kimi K3 metadata-only source admission."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import kimi_k3_source_admission as admission  # noqa: E402
from lab.receipts import seal  # noqa: E402


def _sha(character: str) -> str:
    return character * 64


def _document() -> dict:
    return seal(
        {
            "schema": admission.ADMISSION_SCHEMA,
            "status": admission.ADMISSION_STATUS,
            "source": {
                "repository": admission.REPOSITORY,
                "revision": "a" * 40,
                "private": False,
                "gated": False,
                "weight_shards": [
                    {
                        "path": "model-00001-of-000096.safetensors",
                        "bytes": 1,
                        "lfs_sha256": _sha("b"),
                    },
                    {
                        "path": "model-00096-of-000096.safetensors",
                        "bytes": 2,
                        "lfs_sha256": _sha("c"),
                    },
                ],
            },
            "storage": {
                "source_body_persisted": False,
                "full_weight_shards_downloaded": 0,
                "persistent_hub_cache_files": [],
                "persistent_xet_cache_files": [],
            },
            "claim_boundary": {"teacher_trace_acquisition_authorized": False},
        }
    )


def test_metadata_only_admission_requires_pinned_weight_identities() -> None:
    document = _document()
    validated = admission.validate_admission(document)
    assert validated["source"]["repository"] == "moonshotai/Kimi-K3"
    assert validated["storage"]["full_weight_shards_downloaded"] == 0
    assert admission._WEIGHT_RE.fullmatch("model-00001-of-000096.safetensors")


def test_admission_rejects_tampering_or_source_cache_claims() -> None:
    tampered = deepcopy(_document())
    tampered["source"]["revision"] = "b" * 40
    with pytest.raises(admission.KimiK3SourceAdmissionError, match="seal mismatch"):
        admission.validate_admission(tampered)

    cached = _document()
    unsigned = dict(cached)
    unsigned.pop("seal_sha256")
    unsigned["storage"] = dict(unsigned["storage"])
    unsigned["storage"]["persistent_xet_cache_files"] = ["chunk"]
    with pytest.raises(admission.KimiK3SourceAdmissionError, match="persistent Hub/Xet cache"):
        admission.validate_admission(seal(unsigned))
