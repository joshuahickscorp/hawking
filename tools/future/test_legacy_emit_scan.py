"""A migration's own compatibility reads are not violations of it.

work_events declared receipt_ingested a legacy alias of RESULT_INGESTED and
asserted no production module emits it. Five modules did - and exactly ONE of
the five was an emit:

    model_bearing_torture.py:1845   emit          -> migrated
    model_bearing_torture.py:2694   fixture event -> migrated
    model_bearing_torture.py:1444   CONSUMER set
    autonomy_degeneracy.py:225      CONSUMER set, beside "RESULT_INGESTED"
    power_torture.py:175            CONSUMER set

A consumer that still ACCEPTS the old name is how timelines recorded before the
rename stay readable. Flagging that as an emit asks the migration to break its
own evidence, and left this guard permanently red.
"""
from __future__ import annotations

import ast

from tools.future import work_events as we


def test_the_partition_is_clean_of_real_emits():
    assert we.scan_partition_for_legacy_emits() == []


def test_a_bare_emit_still_fires(tmp_path, monkeypatch):
    """The scan must not have been made blind in the course of narrowing it."""
    mod = tmp_path / "planted_module.py"
    mod.write_text('tape.emit("receipt_ingested", {})\n')
    monkeypatch.setattr(we, "__file__", str(tmp_path / "work_events.py"))
    (tmp_path / "work_events.py").write_text("")
    hits = we.scan_partition_for_legacy_emits()
    assert any(h["file"].endswith("planted_module.py") for h in hits), hits


def test_a_membership_read_does_not_fire(tmp_path, monkeypatch):
    mod = tmp_path / "planted_consumer.py"
    mod.write_text('LANDS = {"RESULT_INGESTED", "receipt_ingested"}\n')
    monkeypatch.setattr(we, "__file__", str(tmp_path / "work_events.py"))
    (tmp_path / "work_events.py").write_text("")
    hits = we.scan_partition_for_legacy_emits()
    assert not any(h["file"].endswith("planted_consumer.py") for h in hits), hits


def test_the_distinction_is_syntactic_and_stated():
    """A collection element is a read; a bare argument is an emit."""
    tree = ast.parse('x = {"receipt_ingested"}\ntape.emit("receipt_ingested")\n')
    in_collection = {
        id(e)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Set, ast.List, ast.Tuple))
        for e in n.elts
    }
    bare = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == "receipt_ingested"
        and id(n) not in in_collection
    ]
    assert len(bare) == 1, "exactly the emit, not the membership read"
