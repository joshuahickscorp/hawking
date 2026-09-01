"""/context can empty the paste cache and can reach nothing else.

The paste cache is disposable by design, so /context gets delete verbs. The
thing worth locking down is the blast radius: receipts, mission state and
evidence live in sibling directories under the same ``.hcli`` root, and no
spelling of ``/context drop`` may reach them.

Runnable two ways:

    python3 -m pytest hcli/test_context_pastes.py -q
    python3 hcli/test_context_pastes.py
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from hcli.commands import CommandHandler
from hcli.paste_cache import PasteCache


class FakeController:
    """Only what /context touches."""

    def __init__(self, root: Path):
        self.workspace_root = str(root)

    def context_summary(self) -> str:
        return "session sess-1 messages=3"


def fresh():
    root = Path(tempfile.mkdtemp(prefix="context_pastes_test_"))
    atexit.register(shutil.rmtree, root, True)
    # The neighbours that must survive every delete verb.
    for name in ("receipts", "mission", "evidence"):
        (root / ".hcli" / name).mkdir(parents=True, exist_ok=True)
        (root / ".hcli" / name / "keep.json").write_text("{}", encoding="utf-8")
    return CommandHandler(FakeController(root)), PasteCache(root), root


def neighbours_intact(root: Path) -> bool:
    return all(
        (root / ".hcli" / name / "keep.json").is_file()
        for name in ("receipts", "mission", "evidence")
    )


def test_bare_context_still_summarises_and_counts_pastes():
    handler, cache, _ = fresh()
    assert handler.handle("/context") == "session sess-1 messages=3 pastes=0"
    cache.store("hello")
    assert handler.handle("/context").endswith("pastes=1")


def test_list_shows_the_context_ref_of_each_paste():
    handler, cache, _ = fresh()
    assert handler.handle("/context list") == "No cached pastes"
    ref = cache.store("first line\nsecond line\n")
    out = handler.handle("/context list")
    assert ref.id in out
    assert ref.context_ref() in out
    assert [row["id"] for row in handler.last_value] == [ref.id]


def test_drop_removes_exactly_one_paste():
    handler, cache, root = fresh()
    keep = cache.store("keep me")
    doomed = cache.store("delete me")
    assert handler.handle(f"/context drop {doomed.id}") == f"Dropped paste {doomed.id}"
    assert [ref.id for ref in cache.list()] == [keep.id]
    assert neighbours_intact(root)


def test_drop_of_an_unknown_but_well_formed_id_says_so():
    handler, _, _ = fresh()
    ghost = "paste_20260901_010412_deadbeef"
    assert handler.handle(f"/context drop {ghost}") == f"No such paste: {ghost}"
    assert handler.last_value == {"dropped": []}


def test_clear_pastes_empties_only_the_cache():
    handler, cache, root = fresh()
    cache.store("one")
    cache.store("two")
    assert handler.handle("/context clear-pastes") == "Dropped 2 paste(s)"
    assert cache.list() == []
    assert neighbours_intact(root)


def test_no_traversal_reaches_a_sibling_directory():
    """The reason delete verbs are safe here: the id is the only door."""
    handler, _, root = fresh()
    for attack in (
        "../receipts/keep",
        "../../.hcli/receipts/keep",
        "/etc/passwd",
        "keep",
        "paste_20260901_010412_deadbeef/../../receipts/keep",
    ):
        out = handler.handle(f"/context drop {attack}")
        assert "not a paste id" in out or "escapes" in out, (attack, out)
    assert neighbours_intact(root)
    assert (root / ".hcli" / "receipts" / "keep.json").is_file()


def test_an_unknown_verb_shows_usage_and_deletes_nothing():
    handler, cache, root = fresh()
    ref = cache.store("safe")
    out = handler.handle("/context nuke")
    assert "Usage" in out or "/context clear-pastes" in out
    assert [r.id for r in cache.list()] == [ref.id]
    assert neighbours_intact(root)


def test_drop_without_an_id_is_a_usage_error_not_a_wildcard():
    handler, cache, root = fresh()
    ref = cache.store("safe")
    assert handler.handle("/context drop") == "Usage: /context drop <paste-id>"
    assert [r.id for r in cache.list()] == [ref.id]
    assert neighbours_intact(root)


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all green")
