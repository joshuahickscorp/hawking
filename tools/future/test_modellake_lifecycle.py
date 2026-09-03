"""The lake census must come from disk, and absent must not read as empty."""
from __future__ import annotations

from pathlib import Path

from tools.future import modellake_lifecycle as ml


def test_counts_agree_with_the_filesystem():
    d = ml.lifecycle()
    if d["state"] == "VOLUME_ABSENT":
        return
    on_disk = len([p for p in (ml.LAKE / "specimens").iterdir() if p.is_dir()])
    assert d["sealed_specimens"] == on_disk, "the census disagrees with the directory"
    assert d["curriculum_eligible"] <= d["sealed_specimens"]
    assert d["payload_complete_unpromoted"] == d["sealed_specimens"] - d["curriculum_eligible"]


def test_an_unmounted_lake_is_not_an_empty_lake(monkeypatch, tmp_path):
    """The branch this host does not take must still be exercised.

    Reporting an unmounted volume as zero specimens is how a scheduler concludes
    there is no science to do.
    """
    monkeypatch.setattr(ml, "LAKE", tmp_path / "not-mounted")
    d = ml.lifecycle()
    assert d["state"] == "VOLUME_ABSENT"
    assert d["sealed_specimens"] is None
    assert d["curriculum_eligible"] is None
    assert "absent is NOT empty" in d["why"]


def test_a_throughput_rate_is_never_invented_from_one_reading():
    """Two timestamped observations are required; one reading cannot yield a rate."""
    d = ml.lifecycle()
    assert d.get("retained_verified_bytes_per_wall_second") is None
    assert "fabricated measurement" in d.get("why_no_rate", "")


def test_acquisition_workers_are_observed_processes_not_configuration():
    d = ml.lifecycle()
    assert isinstance(d["active_acquisition_workers"], list)
    assert d["acquiring_now"] == bool(d["active_acquisition_workers"])
