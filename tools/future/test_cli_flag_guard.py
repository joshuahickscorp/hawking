"""An unknown flag must refuse, not be silently ignored.

`if "--record" in sys.argv` treats every other argument as absent. So `--build` -
the verb most of tools/future uses - printed a freshly computed table, exited 0,
and WROTE NOTHING, while the receipt on disk stayed stale. That cost two
silently-stale receipts before anyone noticed (path_to_71, causal_budget_71), and
a sweep found eleven more modules with the same shape.

This test is the durable part. Fixing thirteen modules by hand does not stop the
fourteenth.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import UnknownFlag, require_known_flags  # noqa: E402

HERE = Path(__file__).resolve().parent
ARGV_DISPATCH = re.compile(r'"(--[a-z0-9-]+)" in sys\.argv')


def _argv_dispatching_modules() -> list[Path]:
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.name.startswith("test_"):
            continue
        text = p.read_text()
        if 'if __name__ == "__main__":' not in text:
            continue
        if ARGV_DISPATCH.search(text):
            out.append(p)
    return out


def test_the_guard_accepts_known_and_refuses_unknown():
    require_known_flags({"--record", "--build"}, ["--record"])
    with pytest.raises(UnknownFlag, match="unknown flag"):
        require_known_flags({"--record"}, ["--build"])


def test_the_guard_names_what_it_would_have_accepted():
    with pytest.raises(UnknownFlag, match=r"known flags are \['--build', '--record'\]"):
        require_known_flags({"--record", "--build"}, ["--nope"])


def test_a_bare_positional_is_not_treated_as_a_flag():
    require_known_flags({"--record"}, ["somefile.json"])


def test_an_equals_form_is_matched_on_its_name():
    require_known_flags({"--out"}, ["--out=x.json"])
    with pytest.raises(UnknownFlag):
        require_known_flags({"--out"}, ["--nope=x"])


def test_there_are_argv_dispatching_modules_to_check():
    assert _argv_dispatching_modules(), "the sweep found nothing; it is broken"


@pytest.mark.parametrize(
    "path", _argv_dispatching_modules(), ids=lambda p: p.stem)
def test_every_argv_dispatching_module_guards_its_flags(path):
    """Dispatching on sys.argv without a guard means every unrecognised
    argument is silently absent."""
    text = path.read_text()
    assert "require_known_flags" in text, (
        f"{path.name} dispatches on sys.argv and does not call "
        "require_known_flags, so an unknown flag runs as a no-op with exit 0"
    )


@pytest.mark.parametrize(
    "path", _argv_dispatching_modules(), ids=lambda p: p.stem)
def test_every_argv_dispatching_module_actually_refuses(path):
    """The source check above can be satisfied by a comment. This runs it."""
    r = subprocess.run([sys.executable, str(path), "--definitely-not-a-flag"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode != 0, f"{path.name} exited 0 on an unknown flag"
    assert "unknown flag" in (r.stdout + r.stderr).lower()
