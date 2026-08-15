from __future__ import annotations

import json
import stat
from pathlib import Path

from lab.operators.ascension_sandbox import (
    MODE,
    SandboxPaths,
    apply_evidence_retractions,
    bootstrap_layout,
    build_preflight_config,
    tick,
)
from lab.receipts import seal, verify


def test_bootstrap_creates_owned_layout_and_readonly_reviewer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    bible = tmp_path / "bible.md"
    bible.write_text("# bible\n")
    paths = bootstrap_layout(tmp_path / "sandbox", repo_root=repo, bible_path=bible)

    assert paths.executor_root.is_dir()
    assert paths.reviewer_root.is_dir()
    assert stat.S_IMODE(paths.reviewer_root.stat().st_mode) & 0o222 == 0
    config = json.loads(paths.config_path.read_text())
    assert config["sandbox"]["approved_download_ids"] == []
    assert config["sandbox"]["reviewer_enforcement"] == "filesystem_readonly"


def test_preflight_config_keeps_conservative_active_storage_check(tmp_path: Path) -> None:
    paths = SandboxPaths.from_root(tmp_path / "sandbox")
    config = build_preflight_config(paths, repo_root=tmp_path / "repo", bible_path=tmp_path / "bible.md")

    candidates = config["proto"]["active_storage_paths_must_be_absent"]
    assert len(candidates) == 2
    assert all("frankenstein" in path for path in candidates)
    assert config["resources"]["swap_growth_allowed"] is False


def test_tick_is_live_but_does_not_claim_option_c(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    bible = tmp_path / "bible.md"
    bible.write_text("# bible\n")

    result = tick(tmp_path / "sandbox", repo_root=repo, bible_path=bible, run_tests=False)

    verify(result, label="sandbox status")
    assert result["mode"] == MODE
    assert result["state"] == "RUNNING"
    assert result["claim_boundary"]["research_control_plane_only"] is True
    gates = result["production_gate_report"]
    assert gates["research_sandbox_active"] is True
    assert gates["option_c_live"] is False
    assert gates["production_sandbox_active"] is False
    assert result["qwen_metadata_preflight"]["download_permitted"] is False


def test_retracted_receipt_cannot_pass_a_foundation_preflight() -> None:
    preflight = seal(
        {
            "schema": "hawking.ascension.sandbox_ready_preflight.v1",
            "status": "SANDBOX_FOUNDATION_PREFLIGHT_READY",
            "sandbox_foundation_preflight_ready": True,
            "qwen30_body_admission_candidate": True,
            "proto_terminal_claimed": True,
            "blockers": [],
            "claim_boundary": {},
        }
    )
    result = apply_evidence_retractions(
        preflight,
        [
            {
                "path": "/evidence/RETRACTED.json",
                "seal_valid": True,
                "status": "RETRACTED",
                "must_not_gate": True,
            }
        ],
    )

    verify(result, label="retraction-adjusted preflight")
    assert result["status"] == "BLOCKED"
    assert result["sandbox_foundation_preflight_ready"] is False
    assert result["proto_terminal_claimed"] is False
    assert "retracted Proto evidence" in result["blockers"][0]
