"""Checks for the paste cache: exact bytes in, exact bytes out, nothing escapes.

Every test builds its own throwaway root, so the suite is offline and order
independent. Runnable two ways:

    python3 -m pytest hcli/test_paste_cache.py -q
    python3 hcli/test_paste_cache.py
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from hcli.paste_cache import PasteCache, PasteNotFound, PasteRef

# len("paste_20260901_014840") -- the id up to and including its seconds field.
_SECOND_GRANULAR = 21

# Every byte class that a naive text writer mangles: CRLF, a lone CR, a tab,
# trailing spaces, non-ASCII, and no trailing newline.
NASTY = (
    "diff --git a/x.py b/x.py\r\n"
    "@@ -1,2 +1,2 @@\r\n"
    "-print('café')  \n"
    "+print('café — 🚀')\t\n"
    "\r"
    "no trailing newline"
)


def fresh() -> PasteCache:
    """A throwaway root that reaps itself, under pytest and under __main__ alike."""
    root = tempfile.mkdtemp(prefix="paste_cache_test_")
    atexit.register(shutil.rmtree, root, True)
    return PasteCache(root=root)


def test_round_trip_is_byte_identical():
    cache = fresh()
    ref = cache.store(NASTY)
    assert cache.get(ref.id) == NASTY
    on_disk = (cache.dir / f"{ref.id}.txt").read_bytes()
    assert on_disk == NASTY.encode("utf-8"), on_disk[:80]
    assert ref.size == len(NASTY.encode("utf-8"))
    assert ref.kind == "diff", ref.kind


def test_line_count_and_kinds():
    cache = fresh()
    assert cache.store("").lines == 0
    assert cache.store("a").lines == 1
    assert cache.store("a\n").lines == 1
    assert cache.store("a\nb\n").lines == 2
    assert cache.store('{"a": [1, 2]}').kind == "json"
    assert cache.store("def f():\n    import os\n    return os\n").kind == "code"
    assert cache.store(
        "\n".join(f"2026-09-01 01:04:{n:02d} INFO worker {n} ok" for n in range(30))
    ).kind == "log"
    assert cache.store("just some prose about nothing at all\n").kind == "text"


def test_identical_content_is_stored_once():
    cache = fresh()
    first = cache.store(NASTY, session="s1")
    second = cache.store(NASTY, session="s2")
    assert second.id == first.id
    assert second.session == "s1", "dedupe must return the stored ref, not a new one"
    assert len(list(cache.dir.glob("*.txt"))) == 1
    assert len(cache.list()) == 1
    # A different blob still gets its own id.
    assert cache.store(NASTY + "x").id != first.id


def test_slice_and_search_agree_on_line_numbers():
    cache = fresh()
    ref = cache.store("\n".join(f"line {n} needle" if n == 7 else f"line {n}" for n in range(1, 21)))
    assert cache.slice(ref.id, 3, 5) == "line 3\nline 4\nline 5"
    assert cache.slice(ref.id, 20, 99) == "line 20"
    hits = cache.search(ref.id, "needle")
    assert hits == [(7, "line 7 needle")], hits
    number, _ = hits[0]
    assert cache.slice(ref.id, number, number) == "line 7 needle"
    assert cache.search(ref.id, "line ", limit=3) == [
        (1, "line 1"), (2, "line 2"), (3, "line 3")]
    assert cache.search(ref.id, "not in here") == []


def test_drop_and_prune():
    cache = fresh()
    refs = [cache.store(f"blob {n}\n") for n in range(4)]
    assert cache.drop(refs[0].id) is True
    assert cache.drop(refs[0].id) is False
    try:
        cache.get(refs[0].id)
        raise AssertionError("dropped paste is still readable")
    except PasteNotFound:
        pass

    # Compare against the order they were STORED in, not against a re-sort of
    # the ids. `== sorted(ids, reverse=True)` asserts list() equals the very
    # operation list() performs, so it passed with the ordering bug present.
    assert [r.id for r in cache.list()] == [
        r.id for r in reversed(refs[1:])
    ], "list() must be newest first"

    try:
        cache.prune()
        raise AssertionError("prune ran without a policy")
    except ValueError:
        pass
    assert cache.prune(older_than_days=365) == []
    assert len(cache.prune(keep_last=1)) == 2
    remaining = cache.list()
    assert len(remaining) == 1
    assert remaining[0].id == refs[-1].id, "prune(keep_last=1) must keep the NEWEST"
    assert cache.prune(keep_last=0) == [remaining[0].id]
    assert cache.list() == []
    assert list(cache.dir.glob("*")) == []


def test_a_sub_second_burst_still_orders_newest_first(monkeypatch):
    """Ids are second-granular and tie-broken by content hash, so a burst of
    pastes inside one second is the regime where sorting on the id silently
    returns an arbitrary order and prune deletes the wrong paste.

    The clock is FROZEN rather than raced. Six real `store()` calls landing in
    the same wall-clock second was a coincidence the test asserted on, and it
    stopped holding once the suite ran in parallel on a loaded host: the burst
    straddled a second boundary and the precondition failed, roughly one run in
    six. Pinning the timestamp makes the same-second collision the guaranteed
    condition under test instead of the thing being hoped for.
    """
    import hcli.paste_cache as paste_cache

    frozen = datetime(2026, 9, 1, 12, 34, 56)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(paste_cache, "datetime", FrozenDatetime)

    cache = fresh()
    refs = [cache.store(f"burst {n}\n") for n in range(6)]
    assert len({r.id[:_SECOND_GRANULAR] for r in refs}) == 1, (
        "the frozen clock must put all six in one second"
    )
    assert [r.id for r in cache.list()] == [r.id for r in reversed(refs)]
    cache.prune(keep_last=1)
    assert [r.id for r in cache.list()] == [refs[-1].id]


def test_a_crafted_id_cannot_escape_the_pastes_dir():
    cache = fresh()
    victim = Path(cache.root) / "receipts"
    victim.mkdir()
    (victim / "keep.json").write_text("precious", encoding="utf-8")

    for evil in (
        "../../receipts",
        "../../receipts/keep",
        "/etc/passwd",
        "paste_20260901_010412_a81fb3c2/../../../receipts/keep",
        "..",
        ".",
        "",
        "paste_20260901_010412_A81FB3C2",  # uppercase hex is not the pattern
        "paste_2026901_010412_a81fb3c2",   # short date
    ):
        for verb in (cache.get, cache.drop):
            try:
                verb(evil)
                raise AssertionError(f"{verb.__name__} accepted {evil!r}")
            except ValueError:
                pass
    assert (victim / "keep.json").read_text(encoding="utf-8") == "precious"
    assert victim.is_dir()


def test_a_symlink_planted_in_the_cache_cannot_be_read_or_deleted():
    cache = fresh()
    outsider = Path(cache.root) / "outside.txt"
    outsider.write_text("not yours", encoding="utf-8")
    real = cache.store("real paste\n")  # creates cache.dir
    planted = "paste_20260901_010412_a81fb3c2"
    os.symlink(outsider, cache.dir / f"{planted}.txt")

    for verb in (cache.get, cache.drop):
        try:
            verb(planted)
            raise AssertionError(f"{verb.__name__} followed a symlink out")
        except ValueError:
            pass
    assert outsider.read_text(encoding="utf-8") == "not yours"
    assert cache.get(real.id) == "real paste\n"


def test_a_200kb_blob_costs_one_short_line_of_context():
    cache = fresh()
    blob = "".join(
        f"2026-09-01 01:04:12 INFO shard {n % 17} decoded {n} tokens ok\n"
        for n in range(4200)
    )
    assert len(blob.encode("utf-8")) > 200_000, len(blob)
    ref = cache.store(blob)
    line = ref.context_ref()
    assert len(line) < 120, (len(line), line)
    assert line.startswith(f"[PASTE {ref.id} ") and line.endswith("]"), line
    assert "KB" in line and "lines" in line and ref.kind in line, line
    assert "\n" not in line
    # The reference is what the model sees; the bytes stay on disk, intact.
    assert cache.get(ref.id) == blob


def test_meta_survives_a_round_trip_and_junk_is_ignored():
    cache = fresh()
    ref = cache.store("x\n", session="s", mission="m")
    assert PasteRef.from_dict(ref.to_dict()) == ref
    assert cache.list()[0] == ref
    (cache.dir / "paste_20260901_010412_deadbeef.json").write_text("{ not json",
                                                                   encoding="utf-8")
    assert cache.list() == [ref], "a corrupt sidecar must not break list()"


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all green")
