#!/usr/bin/env python3.12
"""Contract tests for the extreme non-Doctor condense surface."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


CONDENSE = Path(__file__).resolve().parents[1]
ROOT = CONDENSE.parents[1]
sys.path.insert(0, str(CONDENSE))

import appendix_runtime  # noqa: E402
import condense_profiles  # noqa: E402
import frontier_runtime  # noqa: E402


def test_layout_archive_and_profiles_are_exact() -> None:
    assert condense_profiles.validate_layout() == []
    profile = condense_profiles.profile_document()
    assert profile["retired_count"] >= 80
    assert set(profile["retired_by_family"]) >= {"appendix", "frontier", "core"}
    record = condense_profiles.legacy_record("appendix_catalog")
    assert record["family"] == "appendix"
    assert len(record["source_sha256"]) == 64


def test_appendix_replay_and_default_off_gate() -> None:
    replay = appendix_runtime.replay([
        {"cell_id": "a", "status": "running"},
        {"cell_id": "a", "status": "complete"},
        {"cell_id": "b", "status": "negative"},
    ])
    assert replay["terminal"] == 2
    assert replay["status_counts"]["complete"] == 1
    assert appendix_runtime.validate({}, verify_files=False)
    assert appendix_runtime.PROFILE["physical_default_off"] is True


def test_frontier_classification_replay_and_owner_parser() -> None:
    records = {
        "mp-4a3f": {"config": "mp-4a3f", "eff_bpw": 4.0, "degr_pct": 10.0},
        "mp-4a3f+dr-r64": {
            "config": "mp-4a3f+dr-r64", "eff_bpw": 4.0, "degr_pct": 8.0,
        },
    }
    assert frontier_runtime.classify(
        records, records["mp-4a3f+dr-r64"], 0.25,
    )["verdict"] == "excellent"
    assert frontier_runtime.pareto(records)[0]["config"] == "mp-4a3f+dr-r64"
    owners = frontier_runtime.owners_from_ps(
        "11 python quantize-model --in x\n12 python helper.py\n", own_pid=99,
    )
    assert [row["pid"] for row in owners] == [11]


def test_unified_cli_and_git_backed_legacy_aliases() -> None:
    for command in (
        ["core.profile"],
        ["appendix.runtime"],
        ["frontier.runtime"],
        ["legacy", "appendix_catalog", "--selftest"],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "tools.condense", *command],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout
    profile = json.loads(subprocess.check_output(
        [sys.executable, "-m", "tools.condense", "frontier.runtime"],
        cwd=ROOT, text=True,
    ))
    assert profile["schema"] == frontier_runtime.SCHEMA
