"""resolve_patient_weights() resolved into an HF hub cache that is not here.

Every census on this box recorded ``model_dir`` as
``~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>``. That whole
cache is ~37 MB -- none of those directories exist -- so the automated cycle
spawned the runner with ``--weights <path that is not there>`` and the runner
exits at its own ``weights.exists()`` check. The same revisions are sealed on
the ModelLake volume at ``specimens/<org>--<repo>@<rev12>``, in the same flat
layout, which is why a hand-typed lake path already worked.

Runnable two ways:

    python3 -m pytest tools/test_odyssey_weights_resolution.py -q
    python3 tools/test_odyssey_weights_resolution.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import odyssey_ctl as C  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
# O003: real manifest row, real sealed specimen.
OXX = "O003"
REPO_ID = "moonshotai/Kimi-VL-A3B-Instruct"
REV = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
ABSENT_HF = (
    "~/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct"
    f"/snapshots/{REV}"
)


def _body(d: pathlib.Path) -> pathlib.Path:
    """The minimum a snapshot dir needs to read as one."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "kimi_vl"}))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"\0")
    return d


@pytest.fixture()
def empty_hf(tmp_path, monkeypatch):
    """This box's reality: a hub cache with nothing in it."""
    monkeypatch.setattr(C, "HF_HUB", tmp_path / "hub")
    return tmp_path


def test_lake_resolves_when_the_hf_cache_is_absent(tmp_path, empty_hf, monkeypatch):
    """The load-bearing guard: a resolved path that actually exists."""
    lake = tmp_path / "lake"
    want = _body(lake / "specimens" / "moonshotai--Kimi-VL-A3B-Instruct@398eede0903c")
    monkeypatch.setattr(C, "MODELLAKE_ROOT", lake, raising=False)

    got = C.resolve_patient_weights(
        OXX, pkt={"identity": {"source_repo": REPO_ID}},
        census={"model_dir": ABSENT_HF},
    )
    assert got == str(want)
    assert pathlib.Path(got).exists()


def test_hf_cache_still_wins_when_it_is_there(tmp_path, monkeypatch):
    """Order guard: nothing changes on a machine that has the cache."""
    hub = tmp_path / "hub"
    snap = _body(
        hub / "models--moonshotai--Kimi-VL-A3B-Instruct" / "snapshots" / REV
    )
    monkeypatch.setattr(C, "HF_HUB", hub)
    lake = tmp_path / "lake"
    _body(lake / "specimens" / "moonshotai--Kimi-VL-A3B-Instruct@398eede0903c")
    monkeypatch.setattr(C, "MODELLAKE_ROOT", lake, raising=False)

    assert C.resolve_patient_weights(OXX, pkt={}, census={}) == str(snap)


def test_a_different_revision_is_not_substituted(tmp_path, empty_hf, monkeypatch):
    """A lake body for another revision is not this patient's weights."""
    lake = tmp_path / "lake"
    _body(lake / "specimens" / "moonshotai--Kimi-VL-A3B-Instruct@deadbeef0000")
    monkeypatch.setattr(C, "MODELLAKE_ROOT", lake, raising=False)

    got = C.resolve_patient_weights(OXX, pkt={}, census={"model_dir": ABSENT_HF})
    assert got == ABSENT_HF  # unresolved, not the wrong body


def test_an_incomplete_specimen_is_not_offered(tmp_path, empty_hf, monkeypatch):
    """config.json with no shards is a download, not a body."""
    lake = tmp_path / "lake"
    d = lake / "specimens" / "moonshotai--Kimi-VL-A3B-Instruct@398eede0903c"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    monkeypatch.setattr(C, "MODELLAKE_ROOT", lake, raising=False)

    assert C.resolve_patient_weights(OXX, pkt={}, census={}) == ""


@pytest.mark.skipif(
    not (getattr(C, "MODELLAKE_ROOT", pathlib.Path("/nope")) / "specimens").is_dir(),
    reason="ModelLake volume not mounted",
)
def test_real_patients_resolve_to_paths_that_exist():
    """No fixture: the live manifest, the live lake, this box."""
    resolved = {
        oxx: C.resolve_patient_weights(oxx)
        for oxx in ("O003", "O005", "O001")
    }
    for oxx, path in resolved.items():
        assert path, f"{oxx} resolved to nothing"
        assert pathlib.Path(path).is_dir(), f"{oxx} -> {path} does not exist"
        assert list(pathlib.Path(path).glob("*.safetensors")), f"{oxx} -> no shards"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
