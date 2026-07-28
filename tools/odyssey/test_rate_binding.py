#!/usr/bin/env python3.12
"""Capability does not inherit across rates.

The capability-first Gravity law binds an approval to the artifact hash AND to the BPW it
was earned at. The hash binding was already enforced; this covers the half that is new, and
the half that is easy to lose: a pipeline repacked at a different rate must re-earn its
approval rather than inherit one.

Without this, the ladder proves nothing. Prove usability once at a comfortable 3/2 BPW,
descend to 1/2, and the register still says APPROVED for a substrate nobody ever ran a
forward pass on.

    .venv/glm52/bin/python -m pytest tools/odyssey/test_rate_binding.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.odyssey import substrate_capability as sc


HASH = "deadbeef" * 8


def _register(tmp: Path, entry: dict) -> None:
    """Point the module at a temporary register holding one substrate."""
    reg = tmp / "SUBSTRATE_CAPABILITY.json"
    reg.write_text(json.dumps({
        "substrates": [entry],
        "default_for_unlisted": {
            "capability_verdict": "UNVERIFIED",
            "treated_as": "REFUSED",
            "why": "silence is not a pass",
        },
    }))
    sc.CAPABILITY = reg


def _artifact(tmp: Path, rate: str | None) -> Path:
    art = tmp / "artifact"
    art.mkdir(exist_ok=True)
    if rate is not None:
        (art / "PACK_RECEIPT.json").write_text(json.dumps({"complete_bpw_exact": rate}))
    return art


def test_rate_that_matches_is_admitted(tmp_path: Path) -> None:
    _register(tmp_path, {"name": "t", "artifact_index_sha256": HASH,
                         "capability_verdict": "APPROVED", "proven_at_rate": "3/2"})
    sc.assert_trainable(HASH, _artifact(tmp_path, "3/2"))


def test_repack_at_a_lower_rate_is_refused(tmp_path: Path) -> None:
    """The failure this test exists for: descend the ladder, keep the old approval."""
    _register(tmp_path, {"name": "t", "artifact_index_sha256": HASH,
                         "capability_verdict": "APPROVED", "proven_at_rate": "3/2"})
    with pytest.raises(sc.SubstrateRefused, match="does not inherit across rates"):
        sc.assert_trainable(HASH, _artifact(tmp_path, "1/2"))


def test_approved_without_a_recorded_rate_is_refused(tmp_path: Path) -> None:
    """An approval that does not say what rate it was earned at cannot be checked."""
    _register(tmp_path, {"name": "t", "artifact_index_sha256": HASH,
                         "capability_verdict": "APPROVED"})
    with pytest.raises(sc.SubstrateRefused, match="records no proven_at_rate"):
        sc.assert_trainable(HASH, _artifact(tmp_path, "9/10"))


def test_equal_value_but_different_rational_is_still_refused(tmp_path: Path) -> None:
    """3/2 and 6/4 are the same number; they are not the same measurement.

    Deliberately strict. Two packs that reduce to the same value came from different
    allocations, and the cheap thing to do -- compare floats and call it equal -- is how a
    rate check stops being a check.
    """
    _register(tmp_path, {"name": "t", "artifact_index_sha256": HASH,
                         "capability_verdict": "APPROVED", "proven_at_rate": "3/2"})
    with pytest.raises(sc.SubstrateRefused):
        sc.assert_trainable(HASH, _artifact(tmp_path, "6/4"))


def test_hash_binding_still_holds(tmp_path: Path) -> None:
    """The older property must survive the new one: an unlisted hash is UNVERIFIED."""
    _register(tmp_path, {"name": "t", "artifact_index_sha256": HASH,
                         "capability_verdict": "APPROVED", "proven_at_rate": "3/2"})
    with pytest.raises(sc.SubstrateRefused):
        sc.assert_trainable("f" * 64, _artifact(tmp_path, "3/2"))


def test_refused_artifact_is_refused_before_any_rate_check(tmp_path: Path) -> None:
    _register(tmp_path, {"name": "t", "artifact_index_sha256": HASH,
                         "capability_verdict": "REFUSED", "proven_at_rate": "3/2"})
    with pytest.raises(sc.SubstrateRefused, match="capability_verdict=REFUSED"):
        sc.assert_trainable(HASH, _artifact(tmp_path, "3/2"))
