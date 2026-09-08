"""Consolidation must reduce the SURFACE, never the CAPABILITY. [S004]

The registry went 60 -> 81 -> 91 tools, and a resident that must choose among
83 schemas every turn pays a selection tax for capability it could reach through
a dozen doors. So families were merged behind dispatching tools.

That is only safe if two things hold, and this module is where they are checked
rather than asserted:

  1. Every name that existed before the merge still RESOLVES. `alias_of` hides a
     spec from discover() but leaves it callable; if a merge ever deletes one
     instead, this fails with the exact name.
  2. No merge crossed a MUTATION CLASS. A tool carries one class, so folding a
     destructive op into a read_only tool would silently widen what a read can
     reach. _merge_group refuses that at build time; this proves the refusal is
     load-bearing by attempting an illegal merge and requiring it to raise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.tool_registry import (
    CONSOLIDATION,
    READ_ONLY,
    RESEARCH,
    REVERSIBLE_REPO,
    REVERSIBLE_RUNTIME,
    _merge_group,
    default_tool_registry,
)

HERE = Path(__file__).resolve().parent
BEFORE = json.loads((HERE / "_tool_surface_before_consolidation.json").read_text())


def _registry(tmp_path):
    return default_tool_registry(
        tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
        permissions={READ_ONLY, RESEARCH, REVERSIBLE_REPO, REVERSIBLE_RUNTIME},
    )


def test_every_pre_consolidation_name_still_resolves(tmp_path):
    reg = _registry(tmp_path)
    lost = [name for name in BEFORE if reg.get(name) is None]
    assert lost == [], f"consolidation DELETED capability, not surface: {lost}"


def test_the_surface_actually_shrank(tmp_path):
    reg = _registry(tmp_path)
    visible = [item["name"] for item in reg.discover()]
    assert len(visible) < len(BEFORE), "consolidation did not reduce the surface"
    # Absorbed names must be hidden from discover() but present as alias metadata,
    # so a reader of the catalog can still find where a name went.
    advertised = {a for item in reg.discover() for a in item.get("aliases", [])}
    absorbed = {n for n in BEFORE if n not in visible}
    orphans = sorted(absorbed - advertised)
    assert orphans == [], f"absorbed but not advertised by any canonical tool: {orphans}"


def test_no_merged_group_spans_a_mutation_class(tmp_path):
    reg = _registry(tmp_path)
    for name in CONSOLIDATION:
        spec = reg.get(name)
        assert spec is not None, f"{name} was not registered"
        members = [reg.get(t) for t in CONSOLIDATION[name]["ops"].values()]
        classes = {m.mutation for m in members if m is not None}
        assert classes == {spec.mutation}, (
            f"{name} advertises {spec.mutation} but covers {sorted(classes)}")


def test_merging_across_mutation_classes_is_REFUSED(tmp_path):
    """The negative control. Delete the guard and this test goes green."""
    reg = _registry(tmp_path)
    illegal = {"summary": "read plus a destructive op",
               "ops": {"look": "fs", "wipe": "git.checkout-safe"}}
    with pytest.raises(RuntimeError, match="mutation classes"):
        _merge_group(reg, "illegal.merge", illegal)


def test_a_merged_tool_dispatches_and_keeps_per_op_validation(tmp_path):
    reg = _registry(tmp_path)
    ok = reg.invoke("fs", {"op": "list"})
    assert ok.ok, ok.error
    # An unknown op is rejected by the schema's own enum, BEFORE the handler
    # runs -- so it fails typed rather than reaching dispatch. That is stricter
    # than a refusal dict and is the contract worth pinning.
    bad = reg.invoke("fs", {"op": "definitely_not_an_op"})
    assert not bad.ok and "op" in (bad.error or ""), bad.error
    # The absorbed tool's OWN required-argument contract survives the merge:
    # fs.read requires `path`, so dispatching without one must refuse.
    missing = reg.invoke("fs", {"op": "read"})
    assert missing.ok and "refused" in missing.value, missing.value
