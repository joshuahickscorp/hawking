"""hcli/acquisition.py: HCLI's acquisition decision surface.

acquire-next is list-order and does not reconcile bytes already on disk.
Right now, live, on this box, that means it would propose re-downloading
O010 (GLM-4.5-Air, 205.76 GiB) even though that exact repo+revision already
sits SEALED under /Volumes/corpdrive/hawking-modellake/specimens/, acquired
through the separate ModelLake watcher pipeline -- ``propose()`` exists to
catch precisely that. This file proves, against the real live state (no
fixture, no re-run) plus a few isolated synthetic candidates for the
branches live state may not currently exercise:

1. propose() never starts a download -- Popen is armed to explode.
2. against real state, list-order's own pick and propose()'s reconciliation
   agree on whether it is redundant (grounds the module in what is actually
   on disk today, not an assumption).
3. a sealed ModelLake specimen is reconciled OUT of the ranked list, even
   when the ladder's own on_disk/ledger flags disagree (synthetic, isolated
   SPECIMEN_ROOT).
4. an incomplete same-repo partial outranks a fresh candidate with higher
   Odyssey value (synthetic, isolated PARTIAL_ROOT/MANIFEST_DIR).
5. disk headroom is bound to the real destination filesystem, and a
   candidate that cannot fit is never silently recommended anyway.

Runnable two ways:

    python3 -m pytest hcli/test_acquisition.py -q
    python3 hcli/test_acquisition.py
"""
from __future__ import annotations

import json
import subprocess

import pytest

from hcli import acquisition
from tools.odyssey import modellake_promote


def _patient(oxx, **kw):
    base = {
        "oxx": oxx, "model": f"Test-{oxx}", "class": "test",
        "canonical_source": f"test-org/{oxx.lower()}",
        "source": f"test-org/{oxx.lower()}",
        "canonical_revision": "00000000000000000000000000000000000000",
        "state": "READY", "on_disk": False, "ledger": "queued",
    }
    base.update(kw)
    return base


def _work(oxx, info=5.0, **kw):
    base = {
        "id": f"ACQ-{oxx}", "oxx": oxx, "title": f"acquire {oxx}",
        "status": "READY", "info": info, "wall_cost": 1, "gpu_cost": 0,
        "opus_cost": 0, "kind": "acquisition",
    }
    base.update(kw)
    return base


def test_propose_never_starts_a_download(monkeypatch):
    """The one codepath that actually launches `hf download` in this repo
    (tools/odyssey_ctl.start_hf_download) spawns it via Popen with "download"
    in argv -- arm exactly that shape, leave ordinary reads (df, ps) alone."""
    real_popen = subprocess.Popen

    def guarded(args, *a, **kw):
        if any("download" in str(x) for x in (args if isinstance(args, (list, tuple)) else [args])):
            raise AssertionError(f"propose() must never launch a download: {args!r}")
        return real_popen(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", guarded)
    result = acquisition.propose(process_rows_fn=lambda: [])
    assert result["schema"] == acquisition.SCHEMA


def test_real_state_reconciliation_matches_disk():
    """No fixture: reflects the Odyssey ladder actually mid-flight right now.

    Whatever list-order would pick, if that repo+revision is independently
    verifiable as already on disk (sealed ModelLake specimen or a live HF
    cache snapshot), propose() must say so and must not rank it -- proven by
    recomputing the same live checks the module itself uses, not by trusting
    the module's own output.
    """
    result = acquisition.propose()
    ranked_oxx = {r["oxx"] for r in result["ranked"]}
    acquired_oxx = {a["oxx"] for a in result["already_acquired"]}
    assert ranked_oxx.isdisjoint(acquired_oxx), "no oxx may be both ranked and already-acquired"

    pick = result["list_order_pick"]
    if pick is not None:
        from tools.odyssey_ctl import ensure_state, hf_cache_snapshot
        from tools.odyssey.modellake_watch import slug

        state = ensure_state()
        patient = next(p for p in state["patients"] if p["oxx"] == pick["oxx"])
        revision = acquisition.overlay_manifest(patient, pick["oxx"]).get("canonical_revision")
        really_sealed = bool(
            pick["repo"] and revision
            and (modellake_promote.SPECIMEN_ROOT / slug(pick["repo"], revision)).is_dir()
        )
        really_in_cache = bool(pick["repo"]) and hf_cache_snapshot(pick["repo"]) is not None
        assert result["list_order_would_redownload_sealed"] == (really_sealed or really_in_cache)
        # disk stat is bound to the real download destination, never "/".
        assert result["disk"]["destination"] == str(
            __import__("tools.odyssey_ctl", fromlist=["modellake_destination"]).modellake_destination()
        )


def test_sealed_modellake_specimen_is_reconciled_out(tmp_path, monkeypatch):
    """The core defect: a candidate already fully downloaded elsewhere must
    never appear as something to acquire, even though its ladder state
    still claims on_disk=False / ledger='queued'."""
    monkeypatch.setattr(modellake_promote, "SPECIMEN_ROOT", tmp_path / "specimens")
    monkeypatch.setattr(modellake_promote, "PARTIAL_ROOT", tmp_path / "partial")
    (tmp_path / "specimens").mkdir()
    (tmp_path / "partial").mkdir()

    patient = _patient("O900", canonical_revision="a" * 40)
    tag = "test-org--o900@" + "a" * 12
    (tmp_path / "specimens" / tag).mkdir()

    state = {"patients": [patient], "work": [_work("O900", info=100.0)]}
    result = acquisition.propose(state, process_rows_fn=lambda: [])

    assert result["ranked"] == []
    assert result["recommended"] is None
    assert len(result["already_acquired"]) == 1
    got = result["already_acquired"][0]
    assert got["oxx"] == "O900"
    assert got["where"] == "modellake specimens"
    assert got["tag"] == tag


def test_incomplete_partial_outranks_higher_value_fresh_candidate(tmp_path, monkeypatch):
    """'finishing one usually beats starting another': a low-value candidate
    with an incomplete same-repo partial elsewhere ranks above a high-value
    candidate with nothing downloaded yet."""
    monkeypatch.setattr(modellake_promote, "SPECIMEN_ROOT", tmp_path / "specimens")
    monkeypatch.setattr(modellake_promote, "PARTIAL_ROOT", tmp_path / "partial")
    monkeypatch.setattr(modellake_promote, "MANIFEST_DIR", tmp_path / "manifests")
    for d in ("specimens", "partial", "manifests"):
        (tmp_path / d).mkdir()

    tag = "test-org--o901@" + "b" * 12
    (tmp_path / "partial" / tag).mkdir()
    (tmp_path / "partial" / tag / "weights.bin").write_bytes(b"\0" * 400)
    (tmp_path / "manifests" / f"{tag}.json").write_text(
        json.dumps({"files": ["weights.bin"], "sizes": {"weights.bin": 1000}})
    )

    low_value_partial = _patient("O901", canonical_revision="b" * 40)
    high_value_fresh = _patient("O902", canonical_revision="c" * 40)
    state = {
        "patients": [low_value_partial, high_value_fresh],
        "work": [_work("O901", info=1.0), _work("O902", info=100.0)],
    }
    result = acquisition.propose(state, process_rows_fn=lambda: [])

    assert [r["oxx"] for r in result["ranked"]] == ["O901", "O902"]
    partial = result["ranked"][0]["partial_elsewhere"]
    assert partial["tag"] == tag
    assert partial["bytes_on_disk"] == 400
    assert partial["expected_bytes"] == 1000
    assert partial["pct"] == 40.0
    assert result["recommended"]["oxx"] == "O901"


def test_candidate_that_cannot_fit_is_flagged_not_silently_recommended(monkeypatch, tmp_path):
    """Disk-hold: a candidate estimated far larger than free space must be
    marked fits_disk=False and only recommended when nothing else qualifies
    -- never presented as though it were safe to start."""
    monkeypatch.setattr(modellake_promote, "SPECIMEN_ROOT", tmp_path / "specimens")
    monkeypatch.setattr(modellake_promote, "PARTIAL_ROOT", tmp_path / "partial")
    (tmp_path / "specimens").mkdir()
    (tmp_path / "partial").mkdir()
    monkeypatch.setattr(
        acquisition, "destination_disk_stat",
        lambda: {"destination": "/fake", "mount": "/fake", "free_bytes": 0, "free_gib": 10.0},
    )

    huge = _patient("O903", canonical_revision="d" * 40)
    state = {"patients": [huge], "work": [_work("O903", info=1.0)]}
    result = acquisition.propose(state, process_rows_fn=lambda: [])

    assert result["ranked"][0]["fits_disk"] is False
    assert result["ranked"][0]["est_gib"] > 10.0
    assert "GiB free" in result["recommendation_reason"]


def test_retired_patient_excluded_not_ranked():
    state = {
        "patients": [_patient("O904", state="RETIRED")],
        "work": [_work("O904")],
    }
    result = acquisition.propose(state, process_rows_fn=lambda: [])
    assert result["ranked"] == []
    assert result["already_acquired"] == []
    assert {"oxx": "O904", "reason": "RETIRED"} in result["blocked"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
