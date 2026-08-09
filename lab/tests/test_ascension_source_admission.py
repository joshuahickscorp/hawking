"""Tests for metadata-only, credential-safe Qwen source admission."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.operators.ascension_source_admission import (
    SCHEMA,
    TARGETS,
    SourceAdmissionPaths,
    capture_all_sources,
    capture_source_metadata,
)
from lab.receipts import verify


@dataclass
class _Lfs:
    oid: str


@dataclass
class _Sibling:
    rfilename: str
    size: int
    lfs: _Lfs | None = None


@dataclass
class _Info:
    sha: str
    siblings: list[_Sibling]
    private: bool = False
    gated: bool = False
    cardData: dict[str, Any] | None = None


class _Client:
    def __init__(self, *, authenticated: bool = True) -> None:
        self.authenticated = authenticated
        self.calls: list[dict[str, Any]] = []

    def whoami(self, *, token: bool) -> dict[str, str]:
        assert token is True
        if not self.authenticated:
            raise RuntimeError("not logged in")
        return {"name": "hidden"}

    def model_info(self, repository: str, *, revision: str, files_metadata: bool, token: bool) -> _Info:
        self.calls.append(
            {
                "repository": repository,
                "revision": revision,
                "files_metadata": files_metadata,
                "token": token,
            }
        )
        return _Info(
            sha="a" * 40,
            siblings=[
                _Sibling("config.json", 100),
                _Sibling("LICENSE", 50),
                _Sibling("model-00001-of-00001.safetensors", 1000, _Lfs("b" * 64)),
            ],
            cardData={"license": "apache-2.0"},
        )


def _downloader(tmp_path: Path):
    def download(**kwargs: Any) -> str:
        assert kwargs["token"] is True
        filename = kwargs["filename"]
        path = Path(kwargs["cache_dir"]) / "test-snapshots" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "config.json":
            path.write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_moe",
                        "architectures": ["Qwen3MoeForCausalLM"],
                        "torch_dtype": "bfloat16",
                        "hidden_size": 2048,
                        "num_hidden_layers": 48,
                        "num_experts": 128,
                        "num_experts_per_tok": 8,
                        "vocab_size": 151936,
                    }
                )
            )
        else:
            path.write_text("license text")
        return str(path)

    return download


def test_capture_metadata_is_pinned_sealed_and_token_safe(tmp_path: Path) -> None:
    client = _Client()
    record = capture_source_metadata(
        TARGETS["qwen30"], root=tmp_path / "controller", client=client, downloader=_downloader(tmp_path)
    )

    verify(record, label="source candidate")
    assert record["schema"] == SCHEMA
    assert record["status"] == "CANDIDATE_METADATA_CAPTURED"
    assert record["authority_level"] == "candidate"
    assert record["source"]["revision"] == "a" * 40
    assert record["inventory"]["weight_file_count"] == 1
    assert record["inventory"]["known_weight_bytes"] == 1000
    assert record["architecture"]["config_captured"] is True
    assert record["authentication"]["authenticated"] is True
    assert record["authentication"]["token_material_recorded"] is False
    assert record["claim_boundary"]["no_model_body_downloaded"] is True
    assert all("hf_" not in str(value) for value in record.values())
    assert client.calls == [
        {
            "repository": TARGETS["qwen30"].repository,
            "revision": "main",
            "files_metadata": True,
            "token": True,
        }
    ]
    paths = SourceAdmissionPaths.from_root(tmp_path / "controller")
    assert (paths.records_root / "QWEN30_SOURCE_METADATA_CANDIDATE.json").is_file()


def test_capture_all_preserves_bible_candidate_order(tmp_path: Path, monkeypatch) -> None:
    client = _Client()

    def fake_hub_client():
        return client, _downloader(tmp_path)

    monkeypatch.setattr("lab.operators.ascension_source_admission._hub_client", fake_hub_client)
    summary = capture_all_sources(tmp_path / "controller")
    verify(summary, label="source summary")
    assert summary["candidate_order"] == [
        TARGETS["qwen30"].model_id,
        TARGETS["qwen80"].model_id,
    ]
    assert [row["artifact_id"] for row in summary["records"]] == [
        "QWEN30_SOURCE_METADATA_CANDIDATE",
        "QWEN80_SOURCE_METADATA_CANDIDATE",
    ]
    assert summary["status"] == "ALL_METADATA_CAPTURED"
    assert all(row["no_model_body_downloaded"] is True for row in summary["records"])
    assert summary["claim_boundary"]["no_token_material_recorded"] is True


def test_unauthenticated_public_metadata_is_not_misreported_as_token_access(tmp_path: Path) -> None:
    record = capture_source_metadata(
        TARGETS["qwen30"],
        root=tmp_path / "controller",
        client=_Client(authenticated=False),
        downloader=_downloader(tmp_path),
    )
    assert record["status"] == "CANDIDATE_METADATA_CAPTURED"
    assert record["authentication"]["authenticated"] is False
    assert record["authentication"]["token_material_recorded"] is False
