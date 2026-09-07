"""HCLI could measure and had nowhere to put the answer.

Round 6 ran SIX tool rounds with SIX successful observations and ZERO failures across
five tools, then closed on `bounded_observation_round` -- the all-repeat closer -- because
its last round re-called a tool it had already called. That looks like a model looping.
It is not.

`odyssey.anatomy` RETURNS an anatomy and writes nothing. `odyssey.ingest` is read_only.
`odyssey.record_law` and `odyssey.record_scar` record Laws and Scars, not axis cells.
There was no tool that could move a ledger axis from OWED to MEASURED, so a round that
measured something correctly saw the ledger unchanged and asked again.

The write helpers already existed and had ZERO callers anywhere in the repo:
`odyssey_ledger.measured(rec, axis, value, receipt)` and
`odyssey_ledger.refused(rec, axis, reason)`, both already carrying the right guards --
a value without a receipt is not evidence, and a refusal under 20 characters is how an
unmeasured axis disguises itself as a finding. Built, never connected. Third instance of
that disease in this session.

This is the door. It closes an autonomy gap with a name: HCLI could not close a WorkUnit
because closing requires persisting and nothing offered to persist.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO / "receipts" / "future" / "G034_ODYSSEY_LEDGER.json"


@pytest.fixture
def ledger(tmp_path):
    """A scratch ledger INSIDE the repo.

    pytest's tmp_path is outside the AgentOS write roots and the tool correctly
    refuses it -- which is the guard working, not a bug. Writing under the repo is
    what a real caller does, so that is what gets tested.
    """
    d = REPO / "workspace"
    d.mkdir(exist_ok=True)
    p = d / f"_test_ledger_{tmp_path.name}.json"
    shutil.copy(LIVE, p)
    yield p
    p.unlink(missing_ok=True)
    (p.parent / (p.name + ".tmp")).unlink(missing_ok=True)


def test_a_path_outside_the_write_roots_is_refused(tmp_path):
    """The guard that caught this test first. It must stay."""
    outside = tmp_path / "ledger.json"
    shutil.copy(LIVE, outside)
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(outside), "slug": "x", "axis": "anatomy",
        "value": 1, "receipt": "y.json"})
    assert not res.ok and "write roots" in str(res.error)


def _reg():
    return default_tool_registry(REPO, repo_root=REPO)


def _owed_slug(path: Path, axis: str) -> str:
    doc = json.loads(path.read_text())
    for r in doc["specimens"]:
        if r["axes"][axis]["state"] == "OWED":
            return r["slug"]
    raise AssertionError(f"no specimen owes {axis}")


def test_the_tool_exists_and_is_not_read_only():
    names = {t["name"]: t for t in _reg().discover()}
    assert "odyssey.record_measurement" in names, (
        "HCLI can read the ledger and never write it, so no round can close")
    assert names["odyssey.record_measurement"]["mutation"] != "read_only"


def test_a_measurement_without_a_receipt_is_refused(ledger):
    slug = _owed_slug(ledger, "anatomy")
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": slug, "axis": "anatomy", "value": 12.3})
    assert not res.ok, "a value with no receipt was accepted as evidence"
    assert "receipt" in str(res.error).lower()


def test_a_refusal_without_a_mechanism_is_refused(ledger):
    slug = _owed_slug(ledger, "gpu")
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": slug, "axis": "gpu", "reason": "n/a"})
    assert not res.ok, "'n/a' was accepted as a mechanism"
    assert "mechanism" in str(res.error).lower()


def test_an_unknown_axis_and_an_unknown_slug_are_both_refused(ledger):
    slug = _owed_slug(ledger, "anatomy")
    r = _reg()
    bad_axis = r.invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": slug, "axis": "vibes", "value": 1, "receipt": "x.json"})
    assert not bad_axis.ok and "axis" in str(bad_axis.error).lower()
    bad_slug = r.invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": "not-a-body@0000", "axis": "anatomy",
        "value": 1, "receipt": "x.json"})
    assert not bad_slug.ok


def test_a_real_record_CHANGES_THE_FILE_and_reads_back(ledger):
    """The point. Anything less and the round still sees an unchanged ledger."""
    slug = _owed_slug(ledger, "anatomy")
    r = _reg()
    before = json.loads(ledger.read_text())
    res = r.invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": slug, "axis": "anatomy",
        "value": {"deficit_pct": 4.02}, "receipt": "receipts/future/X.json"})
    assert res.ok, res.error
    after = json.loads(ledger.read_text())
    assert after != before, "the tool reported success and changed nothing on disk"
    cell = next(x for x in after["specimens"] if x["slug"] == slug)["axes"]["anatomy"]
    assert cell["state"] == "MEASURED" and cell["receipt"] == "receipts/future/X.json"
    # and the READ tool must now see it -- a write only the writer can see is not a write
    back = r.invoke("odyssey.ledger", {"path": str(ledger), "slug": slug})
    assert back.ok and back.value["specimen"]["axes"]["anatomy"]["state"] == "MEASURED"


def test_a_refusal_with_a_real_mechanism_lands(ledger):
    slug = _owed_slug(ledger, "gpu")
    res = _reg().invoke("odyssey.record_measurement", {
        "path": str(ledger), "slug": slug, "axis": "gpu",
        "reason": "cannot be executed on this host: mlx_lm has no module for this model_type"})
    assert res.ok, res.error
    cell = next(x for x in json.loads(ledger.read_text())["specimens"]
                if x["slug"] == slug)["axes"]["gpu"]
    assert cell["state"] == "REFUSED" and cell["reason"]


def test_the_live_ledger_is_not_the_default_target():
    """A write tool whose default path is the campaign's own ledger is a foot-gun.

    The schema owns the OMITTED case (`path` is required). It does not own the EMPTY
    case -- "" satisfies `{"type": "string"}` -- so both are asserted, and only the
    second one exercises the handler's own check. Mutating a guard the schema already
    enforced stayed green, which is what sent this test looking for the real seam.
    """
    r = _reg()
    omitted = r.invoke("odyssey.record_measurement", {
        "slug": "x", "axis": "anatomy", "value": 1, "receipt": "y.json"})
    assert not omitted.ok and "path" in str(omitted.error).lower()
    empty = r.invoke("odyssey.record_measurement", {
        "path": "", "slug": "x", "axis": "anatomy", "value": 1, "receipt": "y.json"})
    assert not empty.ok, "an empty path was accepted"
    assert "empty" in str(empty.error).lower(), empty.error
