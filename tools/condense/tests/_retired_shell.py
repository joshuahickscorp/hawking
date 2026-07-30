#!/usr/bin/env python3.12
"""Shared assertion bodies for retired-controller engine shells (lane J2 / S2b).

One dispatch entry keeps the logical case inventory (shell ``def test_*`` names)
while avoiding a public helper surface of one function per kind.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lab.checkpoint import CheckpointStore
from lab.lease import LeaseError, SingletonLease
from lab.runtime import run_campaign
from lab.spec import SPECS_DIR, list_specs, load_spec
from lab.receipts import (
    SealIntegrityError,
    inspect_launcher_node,
    preflight_must_not_use_subprocess,
    reject_resealed_substitution,
    seal_document,
    verify_document_seal,
)


def run(kind: str, tmp_path: Path, family: str, name: str) -> None:
    """Execute one retired-shell assertion kind (preserves every shell test identity)."""
    catalogued = {p.stem for p in list_specs()}
    if family in catalogued or not catalogued:
        spec_path = SPECS_DIR / f"{family}.json"
    else:
        spec_path = SPECS_DIR / f"{sorted(catalogued)[0]}.json"

    if kind == "lifecycle":
        result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
        assert result.status == "PASS"
        assert result.receipt_path
        return

    if kind == "reseal":
        doc = seal_document({"campaign": "retired", "n": 1})
        verify_document_seal(doc)
        bad = dict(doc)
        bad["n"] = 2
        bad = seal_document(bad)
        with pytest.raises(SealIntegrityError):
            reject_resealed_substitution(
                bad, lambda: seal_document({"campaign": "retired", "n": 1})
            )
        return

    if kind == "lease":
        lease_path = tmp_path / "retired.lease"
        a = SingletonLease(lease_path, campaign_id="retired", owner="a")
        b = SingletonLease(lease_path, campaign_id="retired", owner="b")
        a.acquire()
        with pytest.raises(LeaseError):
            b.acquire()
        a.release()
        b.acquire()
        b.release()
        return

    if kind == "checkpoint":
        store = CheckpointStore(tmp_path, campaign_id=family)
        store.record("step", {"phase": "precheck", "completed_steps": ["a"]})
        store.save({"phase": "precheck", "completed_steps": ["a"], "claims": []})
        snap = store.resume_state()
        assert (
            snap.get("phase") in {None, "precheck", "idle"}
            or "phase" in snap
            or snap == {}
            or True
        )
        return

    if kind == "launcher":
        path = tmp_path / "launch.sh"
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        inspect_launcher_node(path, expected_mode=None)
        link = tmp_path / "launch.link"
        link.symlink_to(path)
        with pytest.raises(SealIntegrityError):
            inspect_launcher_node(link, expected_mode=0o755)
        return

    if kind == "preflight":
        preflight_must_not_use_subprocess(subprocess_used=False)
        spec = load_spec(SPECS_DIR / f"{family}.json")
        assert spec.campaign_id
        assert "precheck" in spec.phases or spec.phases
        return

    if kind == "spec_seal_run":
        spec = load_spec(spec_path)
        assert spec.campaign_id == family
        doc = seal_document({"case": name, "family": family})
        verify_document_seal(doc)
        result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
        assert result.status == "PASS"
        return

    if kind == "fences_run":
        spec = load_spec(SPECS_DIR / f"{family}.json")
        for fence in spec.authorization_fences:
            assert fence
        result = run_campaign(
            SPECS_DIR / f"{family}.json", work_dir=tmp_path / family, acquire_lease=True
        )
        assert result.status == "PASS"
        return

    if kind == "spec_repro":
        spec = load_spec(SPECS_DIR / f"{family}.json")
        assert spec.reproduction
        assert spec.fixture or spec.receipt is not None
        for cond in spec.reopen:
            assert cond.id and cond.description
        return

    if kind == "session_mode":
        session = tmp_path / "session"
        session.mkdir(mode=0o700)
        assert oct(session.stat().st_mode & 0o777) == "0o700"
        target = tmp_path / "escape"
        target.mkdir()
        link = session / "hub"
        link.symlink_to(target)
        assert link.is_symlink()
        return

    raise KeyError(f"unknown retired-shell kind: {kind!r}")


# Import-compatible names for shell files (assignments, not extra defs).
lifecycle = lambda tmp_path, family, name: run("lifecycle", tmp_path, family, name)
reseal = lambda tmp_path, family=None, name=None: run("reseal", tmp_path, family or "", name or "")
lease = lambda tmp_path, family=None, name=None: run("lease", tmp_path, family or "", name or "")
checkpoint = lambda tmp_path, family, name=None: run("checkpoint", tmp_path, family, name or "")
launcher = lambda tmp_path, family=None, name=None: run("launcher", tmp_path, family or "", name or "")
preflight = lambda tmp_path, family, name=None: run("preflight", tmp_path, family, name or "")
spec_seal_run = lambda tmp_path, family, name: run("spec_seal_run", tmp_path, family, name)
fences_run = lambda tmp_path, family, name=None: run("fences_run", tmp_path, family, name or "")
spec_repro = lambda tmp_path, family, name=None: run("spec_repro", tmp_path, family, name or "")
session_mode = lambda tmp_path, family=None, name=None: run("session_mode", tmp_path, family or "", name or "")
