"""The lake mount is NOT read-only, so the guard IS the protection.

Verified on this host:

    /dev/disk5s1 on /Volumes/corpdrive (apfs, local, nodev, nosuid, journaled, noowners)
    /dev/disk3s1s1 on /            (apfs, sealed, local, read-only, journaled)

No `read-only` flag on the lake. "/Volumes is READ ONLY" is a campaign policy, not an
enforced mount, and it has already failed once today -- commit "a detached drive rewrote
the lake catalog as empty and under budget". 4.29 TiB of irreplaceable source weights sit
behind a convention and whatever the tools check for themselves.

Two problems with what the tools checked:

1. TWO IMPLEMENTATIONS. `dense_sweep._refuse_volumes_write` and
   `state_axis._refuse_volumes_write` were byte-identical copies. S008 3 is explicit --
   ONE OWNER, ONE GUARD, ONE WRITE PATH -- because a guard with two implementations
   becomes two future truths, and only one of them gets fixed.
2. NO SYMLINK RESOLUTION. Both used `os.path.abspath`, which normalises `..` but does not
   follow links. A symlink whose name is outside /Volumes and whose target is inside it
   walked straight through. That is not hypothetical on a machine where the lake is
   mounted at a path other tools symlink to.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "future"))
import campaign_memory_guard as g  # noqa: E402


def test_there_is_exactly_one_implementation():
    hits = []
    for f in (REPO / "tools" / "future").glob("*.py"):
        if "def _refuse_volumes_write" in f.read_text() or "def refuse_volumes_write" in f.read_text():
            hits.append(f.name)
    assert hits == ["campaign_memory_guard.py"], (
        f"the lake write guard is defined in {hits} -- two copies is two future truths")


def test_it_refuses_the_obvious_paths():
    for bad in ("/Volumes", "/Volumes/corpdrive",
                "/Volumes/corpdrive/hawking-modellake/specimens/x/config.json",
                "/Volumes/corpdrive/../corpdrive/hawking-modellake/catalog.json"):
        with pytest.raises(RuntimeError, match="/Volumes"):
            g.refuse_volumes_write(bad)


def test_it_allows_a_normal_repo_write():
    g.refuse_volumes_write(str(REPO / "receipts" / "future" / "X.json"))
    g.refuse_volumes_write("receipts/future/X.json")


def test_a_SYMLINK_into_the_lake_does_not_walk_through(tmp_path):
    """abspath normalises `..`; it does not follow links. realpath does."""
    if not Path("/Volumes").is_dir():
        pytest.skip("no /Volumes on this host")
    link = tmp_path / "innocent_name"
    try:
        os.symlink("/Volumes/corpdrive", link)
    except OSError as e:
        pytest.skip(f"cannot symlink here: {e}")
    with pytest.raises(RuntimeError, match="/Volumes"):
        g.refuse_volumes_write(str(link / "hawking-modellake" / "catalog.json"))
