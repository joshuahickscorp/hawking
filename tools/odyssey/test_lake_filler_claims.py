"""The claim registry, and the two measurement defects beside it.

Every test here has its negative control next to it: a registry that only ever refuses
is as useless as one that never does.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lake_filler as lf                                            # noqa: E402


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(lf, "CLAIMS", tmp_path / "claims")
    return tmp_path


def test_an_unclaimed_slug_is_free(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert lf.claim_holder("nobody--here@0") is None


def test_taking_a_claim_then_seeing_it(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert lf.take_claim("m--x@1") is None                          # free: taken
    held = lf.claim_holder("m--x@1")
    assert held is not None and held["pid"] == os.getpid()


def test_a_LIVE_holder_is_never_stolen_from(tmp_path, monkeypatch):
    """The defect this exists for: two processes acquiring one repo into one directory."""
    _isolate(tmp_path, monkeypatch)
    lf.CLAIMS.mkdir(parents=True, exist_ok=True)
    (lf.CLAIMS / "m--x@1.json").write_text(json.dumps(
        {"pid": os.getppid(), "slug": "m--x@1", "taken_at": 0}))   # a real, live pid
    held = lf.take_claim("m--x@1")
    assert held is not None and held["pid"] == os.getppid()


def test_a_DEAD_holder_is_reclaimable(tmp_path, monkeypatch):
    """The other direction. One crash must not strand a repo forever -- that would be a
    worse failure than the double claim being fixed."""
    _isolate(tmp_path, monkeypatch)
    lf.CLAIMS.mkdir(parents=True, exist_ok=True)
    dead = 2 ** 22                                                  # above any real pid
    (lf.CLAIMS / "m--x@1.json").write_text(json.dumps(
        {"pid": dead, "slug": "m--x@1", "taken_at": 0}))
    assert lf.claim_holder("m--x@1") is None                        # stale
    assert lf.take_claim("m--x@1") is None                          # and reclaimable


def test_release_frees_it(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    lf.take_claim("m--x@1")
    lf.release_claim("m--x@1")
    assert lf.claim_holder("m--x@1") is None


def test_release_does_NOT_free_someone_elses_live_claim(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    lf.CLAIMS.mkdir(parents=True, exist_ok=True)
    (lf.CLAIMS / "m--x@1.json").write_text(json.dumps(
        {"pid": os.getppid(), "slug": "m--x@1", "taken_at": 0}))
    lf.release_claim("m--x@1")
    assert lf.claim_holder("m--x@1") is not None                    # still held


def test_log_is_opened_append_not_truncate():
    """A retry used to TRUNCATE the prior attempt's log. That is precisely how the
    Qwen2.5-72B failure text became unrecoverable."""
    src = Path(lf.__file__).read_text()
    assert 'open(log, "a")' in src
    assert 'open(log, "w")' not in src


def test_partial_bytes_is_recorded_separately_from_published_bytes():
    """bytes_on_disk measured only TIER2/slug, which exists only after the atomic publish,
    so a download that completed into partial/ and failed to publish reported 0 -- the
    same number as a download that never transferred a byte."""
    src = Path(lf.__file__).read_text()
    assert "partial_bytes" in src and "downloaded_but_unpublished" in src
    assert "ml.du(ml.PARTIAL / item[\"slug\"])" in src
