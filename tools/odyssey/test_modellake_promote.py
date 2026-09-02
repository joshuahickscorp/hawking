"""Tests for the promotion step itself: verified completeness, atomic
os.rename, and -- the property the ModelLake lifecycle gap actually needs --
idempotent replay. Everything runs against a scratch tree (tmp_path); nothing
here ever touches /Volumes/corpdrive or starts a download.
"""
from __future__ import annotations

import json

import pytest

from tools.odyssey import modellake_promote as mp


@pytest.fixture
def lake(tmp_path, monkeypatch):
    model_root = tmp_path / "model_root"
    partial = model_root / "partial"
    specimens = model_root / "specimens"
    manifests = tmp_path / "manifests"
    partial.mkdir(parents=True)
    specimens.mkdir(parents=True)
    manifests.mkdir(parents=True)
    monkeypatch.setattr(mp, "MODEL_ROOT", model_root)
    monkeypatch.setattr(mp, "PARTIAL_ROOT", partial)
    monkeypatch.setattr(mp, "SPECIMEN_ROOT", specimens)
    monkeypatch.setattr(mp, "MANIFEST_DIR", manifests)
    return {"partial": partial, "specimens": specimens, "manifests": manifests}


def _write_manifest(lake, tag, files):
    sizes = {name: len(content) for name, content in files.items()}
    (lake["manifests"] / f"{tag}.json").write_text(json.dumps({
        "repo": "acme/x", "revision": "deadbeefcafe", "mode": "safe",
        "expected": sum(sizes.values()), "files": list(files),
        "sizes": sizes, "resolved_sha": "deadbeefcafe",
    }))
    return sizes


def _write_partial(lake, tag, files):
    d = lake["partial"] / tag
    d.mkdir(parents=True)
    for name, content in files.items():
        (d / name).write_bytes(content)
    return d


TAG = "acme--x@deadbeefcafe"
FILES = {"config.json": b"{}", "weights.bin": b"0" * 4096}


def test_verify_no_partial_dir(lake):
    assert mp.verify(TAG) == (False, "NO_PARTIAL_DIR", {})


def test_verify_no_manifest(lake):
    _write_partial(lake, TAG, FILES)
    complete, reason, detail = mp.verify(TAG)
    assert (complete, reason, detail) == (False, "NO_MANIFEST", {})


def test_verify_missing_file(lake):
    d = _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    (d / "weights.bin").unlink()
    complete, reason, detail = mp.verify(TAG)
    assert not complete and reason == "INCOMPLETE"
    assert detail["missing"] == 1


def test_verify_wrong_size(lake):
    d = _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    (d / "weights.bin").write_bytes(b"short")
    complete, reason, detail = mp.verify(TAG)
    assert not complete and reason == "INCOMPLETE"
    assert detail["wrong_size"] == 1


def test_verify_complete(lake):
    _write_partial(lake, TAG, FILES)
    sizes = _write_manifest(lake, TAG, FILES)
    complete, reason, detail = mp.verify(TAG)
    assert complete and reason == ""
    assert detail["bytes"] == sum(sizes.values())


def test_promote_dry_run_default_moves_nothing(lake):
    d = _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    outcome = mp.promote(TAG)  # go defaults to False
    assert outcome["action"] == "WOULD_PROMOTE"
    assert d.is_dir()
    assert not (lake["specimens"] / TAG).is_dir()


def test_promote_go_moves_atomically(lake):
    d = _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    outcome = mp.promote(TAG, go=True)
    assert outcome["action"] == "PROMOTED"
    assert outcome["verified_at_destination"] is True
    assert not d.is_dir()
    dest = lake["specimens"] / TAG
    assert dest.is_dir()
    for name, content in FILES.items():
        assert (dest / name).read_bytes() == content


def test_promote_refuses_incomplete_source(lake):
    d = _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    (d / "weights.bin").unlink()
    outcome = mp.promote(TAG, go=True)
    assert outcome["action"] == "REFUSED"
    assert outcome["reason"] == "INCOMPLETE"
    assert d.is_dir()


def test_promote_refuses_conflicting_destination_and_touches_neither(lake):
    """A CONFLICTING specimen at the destination is a refusal that preserves
    both -- never a silent overwrite or merge."""
    d = _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    dest = lake["specimens"] / TAG
    dest.mkdir()
    (dest / "sentinel.txt").write_bytes(b"do not touch")

    outcome = mp.promote(TAG, go=True)

    assert outcome["action"] == "REFUSED"
    assert outcome["reason"] == "DESTINATION_EXISTS"
    assert d.is_dir()
    for name, content in FILES.items():
        assert (d / name).read_bytes() == content
    assert (dest / "sentinel.txt").read_bytes() == b"do not touch"


def test_promote_replay_after_success_is_a_noop_not_a_remove(lake):
    """The core idempotency contract: a replayed promotion event over an
    already-correct specimen (partial/ already gone) must report
    ALREADY_PROMOTED, never REFUSED (which would misreport a finished
    promotion as an error) and never attempt any filesystem move."""
    _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    first = mp.promote(TAG, go=True)
    assert first["action"] == "PROMOTED"
    dest = lake["specimens"] / TAG
    before = {name: (dest / name).stat().st_mtime_ns for name in FILES}

    replay = mp.promote(TAG, go=True)

    assert replay["action"] == "ALREADY_PROMOTED"
    assert replay["complete"] is True
    for name in FILES:
        assert (dest / name).stat().st_mtime_ns == before[name]


def test_promote_replay_over_a_corrupted_destination_refuses(lake):
    """Source gone, destination present but wrong -- flagged, not silently
    accepted as done."""
    _write_partial(lake, TAG, FILES)
    _write_manifest(lake, TAG, FILES)
    mp.promote(TAG, go=True)
    dest = lake["specimens"] / TAG
    (dest / "weights.bin").unlink()

    replay = mp.promote(TAG, go=True)

    assert replay["action"] == "REFUSED"
    assert replay["reason"].startswith("DESTINATION_INCOMPLETE")


def test_promote_nothing_anywhere_is_refused_not_already_promoted(lake):
    outcome = mp.promote(TAG, go=True)
    assert outcome["action"] == "REFUSED"
    assert outcome["reason"] == "NO_PARTIAL_DIR"


def test_survey_reports_every_partial_with_its_completeness(lake):
    complete_tag = "acme--complete@aaaaaaaaaaaa"
    incomplete_tag = "acme--incomplete@bbbbbbbbbbbb"
    _write_partial(lake, complete_tag, FILES)
    _write_manifest(lake, complete_tag, FILES)
    d2 = _write_partial(lake, incomplete_tag, FILES)
    _write_manifest(lake, incomplete_tag, FILES)
    (d2 / "weights.bin").unlink()

    rows = {row["tag"]: row for row in mp.survey()}

    assert rows[complete_tag]["complete"] is True
    assert rows[incomplete_tag]["complete"] is False


def test_cli_dry_run_then_go_promotes_exactly_the_complete_ones(lake, capsys):
    complete_tag = "acme--complete@aaaaaaaaaaaa"
    incomplete_tag = "acme--incomplete@bbbbbbbbbbbb"
    _write_partial(lake, complete_tag, FILES)
    _write_manifest(lake, complete_tag, FILES)
    d2 = _write_partial(lake, incomplete_tag, FILES)
    _write_manifest(lake, incomplete_tag, FILES)
    (d2 / "weights.bin").unlink()

    assert mp.main([]) == 0  # dry run
    assert (lake["partial"] / complete_tag).is_dir()  # not moved yet

    assert mp.main(["--go"]) == 0
    assert not (lake["partial"] / complete_tag).is_dir()
    assert (lake["specimens"] / complete_tag).is_dir()
    assert (lake["partial"] / incomplete_tag).is_dir()  # left alone
