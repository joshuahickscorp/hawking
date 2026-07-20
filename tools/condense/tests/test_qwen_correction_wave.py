#!/usr/bin/env python3.12
"""Synthetic-only tests for the Qwen3-235B transfer wave controller.

NO real forward is run and the 438 GiB Qwen source is never touched. These validate:
  * forge_pack per family reconstructs to the right shape with real byte-accounted BPW,
  * the T0-T4 candidate mapping enumerates correctly (Qwen 3-projection classes; Vulture champion
    maps gate/up together and down distinctly),
  * _rows() produces the decisive-first parent-then-candidate set,
  * the source-absent path seals WAITING_SOURCE cleanly (monkeypatched source_present=False, no lease),
  * the durability helpers (lease claim/refuse + atomic sealed checkpoint) work on a tmp dir.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import qwen_correction_wave as qcw  # noqa: E402


# --------------------------------------------------------------------------- #
# forge_pack per family (tiny synthetic weights)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["kept_original", "product_quant", "naive_rvq",
                                    "pq_protected_islands", "pq_doctor_lowrank"])
def test_forge_pack_family_shape_and_bpw(family):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((16, 32)) * 0.08).astype(np.float32)   # [moe_inter, hidden]-shaped
    out = qcw.forge_pack(family, w, seed=1, params=qcw.PARAMS)
    assert out["recon"].shape == w.shape
    assert np.isfinite(out["recon"]).all()
    assert out["whole_bpw"] > 0.0
    if family == "kept_original":
        assert np.array_equal(out["recon"], w)          # exact passthrough, 16 bpw
        assert out["whole_bpw"] == 16.0


def test_forge_pack_rejects_unknown_family():
    with pytest.raises(ValueError):
        qcw.forge_pack("no_such_family", np.zeros((4, 8), np.float32))


# --------------------------------------------------------------------------- #
# T0-T4 candidate mapping enumeration
# --------------------------------------------------------------------------- #
def test_candidates_enumerate_over_three_qwen_classes():
    assert set(qcw.CANDIDATES) == {"T1_vulture_champion", "T2_product_quant",
                                   "T3_qwen_organ_alloc", "T4_naive_rvq_control"}
    for mapping in qcw.CANDIDATES.values():
        assert set(mapping) == {"gate", "up", "down"}       # Qwen has THREE projections
        for cls in ("gate", "up", "down"):
            assert "family" in mapping[cls] and "params" in mapping[cls]


def test_vulture_champion_maps_gateup_together_down_distinct():
    t1 = qcw.CANDIDATES["T1_vulture_champion"]
    # gpt-oss mlp1 -> gate+up (same family), mlp2 -> down
    assert t1["gate"]["family"] == t1["up"]["family"]
    # priors file is present in-repo -> down family is the mlp2 champion pq_protected_islands
    assert t1["down"]["family"] == "pq_protected_islands"


def test_vulture_champion_fallback_when_priors_absent(monkeypatch):
    monkeypatch.setattr(qcw, "PRIORS_PATH", Path("/nonexistent/priors.json"))
    m = qcw._vulture_champion()
    assert m["gate"]["family"] == "product_quant" and m["up"]["family"] == "product_quant"
    assert m["down"]["family"] == "pq_protected_islands"


def test_control_and_baseline_families():
    assert all(v["family"] == "product_quant" for v in qcw.CANDIDATES["T2_product_quant"].values())
    assert all(v["family"] == "naive_rvq" for v in qcw.CANDIDATES["T4_naive_rvq_control"].values())
    # T3 organ allocation is a genuinely distinct mapping: down != gate family, gate/up tighter
    t3 = qcw.CANDIDATES["T3_qwen_organ_alloc"]
    assert t3["down"]["family"] != t3["gate"]["family"]
    assert t3["gate"]["params"]["dim"] == 32 and t3["down"]["params"]["budget_frac"] == 0.03


# --------------------------------------------------------------------------- #
# _rows(): decisive-first parent-then-candidate set
# --------------------------------------------------------------------------- #
def test_rows_are_decisive_first_and_unique():
    rows = qcw._rows()
    n = len(qcw.HOLDOUT)
    assert len(rows) == n * (1 + len(qcw.CANDIDATE_ORDER))          # parent + 4 candidates per prompt
    assert all(r["candidate"] == "parent" for r in rows[:n])        # all T0 parent refs first
    assert rows[n]["candidate"] == "T1_vulture_champion"           # champion block first
    assert rows[n]["row_id"].endswith("__T1_vulture_champion")
    assert len({r["row_id"] for r in rows}) == len(rows)           # unique row ids


# --------------------------------------------------------------------------- #
# Source-absent -> WAITING_SOURCE clean no-op (no lease, exit 0)
# --------------------------------------------------------------------------- #
def test_source_absent_seals_waiting_source(tmp_path, monkeypatch):
    monkeypatch.setattr(qcw, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(qcw, "STATE_PATH", tmp_path / "QWEN_TRANSFER_STATE.json")
    monkeypatch.setattr(qcw, "WAITING_RECEIPT", tmp_path / "WAITING.json")
    monkeypatch.setattr(qcw, "LEASE_PATH", tmp_path / "leases" / "qwen_transfer.lease")

    class _Stub:
        def source_present(self):
            return False

    monkeypatch.setattr(qcw, "_parent_forward", lambda: _Stub())
    rc = qcw.run()
    assert rc == 0                                                  # clean exit, never a crash
    st = json.loads((tmp_path / "QWEN_TRANSFER_STATE.json").read_text())
    assert st["status"] == "WAITING_SOURCE"
    assert (tmp_path / "WAITING.json").exists()
    assert not (tmp_path / "leases" / "qwen_transfer.lease").exists()   # no heavy lease taken


# --------------------------------------------------------------------------- #
# Durability: lease claim/refuse + atomic sealed checkpoint
# --------------------------------------------------------------------------- #
def test_lease_claim_refuse_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(qcw, "_other_heavy_lease_live", lambda: None)   # ignore the live repo leases
    monkeypatch.setattr(qcw, "LEASES", tmp_path / "leases")
    monkeypatch.setattr(qcw, "LEASE_PATH", tmp_path / "leases" / "qwen_transfer.lease")

    qcw._acquire_lease()
    assert qcw.LEASE_PATH.exists()
    held = json.loads(qcw.LEASE_PATH.read_text())
    assert held["owner"] == qcw.LABEL and held["pid"] == os.getpid()

    # re-acquire while our own (live) pid holds the lease -> refuse
    with pytest.raises(SystemExit):
        qcw._acquire_lease()

    # sealed checkpoint: atomic write + stable sha + the resume-skip primitive (cp.exists())
    rec = {"row_id": "gen_paris__T0_parent", "variant": "parent", "quality": {"perplexity": 12.3}}
    rec["sha256"] = qcw._sha(rec)
    cp = tmp_path / "gen_paris__T0_parent.json"
    assert not cp.exists()
    qcw._atomic(cp, rec)
    assert cp.exists()                                             # a sealed row => run() skips it
    reloaded = json.loads(cp.read_text())
    assert reloaded["sha256"] == rec["sha256"]


def test_other_heavy_lease_blocks_claim(tmp_path, monkeypatch):
    # a live foreign lease => _acquire_lease refuses (one-heavy-lease law)
    monkeypatch.setattr(qcw, "_other_heavy_lease_live", lambda: "DOCTOR_CAMPAIGN pid 42390")
    monkeypatch.setattr(qcw, "LEASES", tmp_path / "leases")
    monkeypatch.setattr(qcw, "LEASE_PATH", tmp_path / "leases" / "qwen_transfer.lease")
    with pytest.raises(SystemExit):
        qcw._acquire_lease()
    assert not qcw.LEASE_PATH.exists()
