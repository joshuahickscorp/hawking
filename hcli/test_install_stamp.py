"""`hcli` execs a copy under ~/.local/share/hcli/current, not the repo. On
2026-09-01 that symlink pointed at a 2026-08-27 snapshot while the working
tree was 2026-08-31, so `hcli` and `PYTHONPATH=. python3 -m hcli` ran two
different codebases and neither said so. The same directory had accumulated
60 build-* snapshots because nothing ever reaped them.
"""
import json
import sys
from pathlib import Path

from hcli.cli import (
    INSTALL_STAMP,
    KEEP_BUILDS,
    install_shims,
    package_digest,
    warn_if_stale,
)


def _prep(tmp_path, monkeypatch):
    """Point install_shims at a throwaway package copy, so the stamp names a
    source we can then edit. It deploys ``Path(hcli.cli.__file__).parent``."""
    src = tmp_path / "repo" / "hcli"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "cli.py").write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr("hcli.cli.__file__", str(src / "cli.py"))
    return src, tmp_path / "home" / ".local" / "share" / "hcli"


def _install(tmp_path, monkeypatch):
    src, share = _prep(tmp_path, monkeypatch)
    install_shims(home=str(tmp_path / "home"))
    return src, share


def test_A_MATCHING_INSTALL_SAYS_NOTHING(tmp_path, monkeypatch, capsys):
    src, share = _install(tmp_path, monkeypatch)
    capsys.readouterr()
    monkeypatch.setattr("hcli.cli.__file__", str(share / "current" / "hcli" / "cli.py"))
    warn_if_stale()
    assert capsys.readouterr().err == ""


def test_THE_MEASURED_DRIFT_IS_LOUD_AND_NAMES_THE_FIX(tmp_path, monkeypatch, capsys):
    src, share = _install(tmp_path, monkeypatch)
    (src / "cli.py").write_text("VERSION = 2\n", encoding="utf-8")
    capsys.readouterr()
    monkeypatch.setattr("hcli.cli.__file__", str(share / "current" / "hcli" / "cli.py"))
    warn_if_stale()
    err = capsys.readouterr().err
    assert "STALE" in err and str(src.parent) in err and "install-shims" in err
    assert err.count("\n") == 1, "one line, not a banner"


def test_A_CHECKOUT_THAT_ONLY_TOUCHED_MTIMES_IS_NOT_STALE(tmp_path, monkeypatch, capsys):
    """Rewriting the same bytes must not fire the warning, or nobody reads it."""
    src, share = _install(tmp_path, monkeypatch)
    (src / "cli.py").write_text("VERSION = 1\n", encoding="utf-8")
    Path(src / "cli.py").touch()
    capsys.readouterr()
    monkeypatch.setattr("hcli.cli.__file__", str(share / "current" / "hcli" / "cli.py"))
    warn_if_stale()
    assert capsys.readouterr().err == ""


def test_AN_EDITABLE_CHECKOUT_HAS_NO_STAMP_AND_STAYS_SILENT(capsys):
    """This test file is running from the repo, so the real package is the case."""
    assert not (Path(__file__).resolve().parent.parent / INSTALL_STAMP).exists()
    warn_if_stale()
    assert capsys.readouterr().err == ""


def test_PYCACHE_IS_NOT_WALKED(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n", encoding="utf-8")
    before = package_digest(pkg)
    (pkg / "__pycache__" / "a.cpython-311.py").write_text("junk\n", encoding="utf-8")
    assert package_digest(pkg) == before


def test_INSTALL_REAPS_OLD_SNAPSHOTS_BUT_NEVER_THE_LIVE_ONE(tmp_path, monkeypatch):
    src, share = _prep(tmp_path, monkeypatch)
    share.mkdir(parents=True)
    for i in range(KEEP_BUILDS + 4):
        (share / f"build-20200101-00000{i}" / "hcli").mkdir(parents=True)
    install_shims(home=str(tmp_path / "home"))
    builds = sorted(p for p in share.glob("build-*") if p.is_dir())
    assert len(builds) == KEEP_BUILDS, builds
    live = (share / "current").resolve()
    assert live in builds and (live / "hcli" / "cli.py").is_file()


def test_REAPING_DOES_NOT_FOLLOW_THE_SYMLINK(tmp_path, monkeypatch):
    """A build-* entry that is a symlink is skipped, not resolved and deleted."""
    src, share = _prep(tmp_path, monkeypatch)
    share.mkdir(parents=True)
    keeper = tmp_path / "elsewhere"
    keeper.mkdir()
    (keeper / "irreplaceable.txt").write_text("x", encoding="utf-8")
    (share / "build-20200101-000000").symlink_to(keeper)
    for i in range(KEEP_BUILDS + 2):
        (share / f"build-20200102-00000{i}").mkdir()
    install_shims(home=str(tmp_path / "home"))
    assert (keeper / "irreplaceable.txt").is_file()
    assert (share / "build-20200101-000000").is_symlink()


def test_THE_STAMP_RECORDS_WHAT_WAS_DEPLOYED(tmp_path, monkeypatch):
    src, share = _install(tmp_path, monkeypatch)
    stamp = json.loads(
        ((share / "current").resolve() / INSTALL_STAMP).read_text(encoding="utf-8")
    )
    assert stamp["source"] == str(src)
    assert stamp["digest"] == package_digest(src)
    assert stamp["installed"]


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
