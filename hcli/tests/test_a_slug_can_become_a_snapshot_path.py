"""Rounds 13 and 15 both fed the LEDGER'S OWN PATH to an anatomy tool, because it was the
only path they had.

Every anatomy tool takes `snapshot`, a DIRECTORY. Every discovery tool hands back a SLUG:

    odyssey.ledger owed[0] -> {slug, gib, class, owed}
    lake.census           -> {slug, klass, complete_ebpw, ..., packed_payload}

Neither returns a filesystem path, and nothing else in the registry converts one to the
other. So a round that correctly selects a specimen, correctly identifies that it owes a
DENSE anatomy, and correctly reaches for odyssey.dense_anatomy has no way to name the body.

Round 13 passed the ledger path and RECORDED the resulting complaint as a specimen fact.
Round 15 passed the ledger path twice, recognised the error itself -- "re-run
odyssey.dense_anatomy with a valid snapshot path (a directory, not the ledger JSON)" --
and correctly refused to record anything. Same input, better handling, same dead end.

That is not a cognition failure. It is the harness demanding an input no tool provides.
Sixth instance this session of the same disease, in a new form: not a capability without
a caller, but a required argument without a source.
"""
from __future__ import annotations

from pathlib import Path

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]


def _reg():
    return default_tool_registry(REPO, repo_root=REPO)


def test_the_owed_list_hands_back_a_usable_snapshot_path():
    v = _reg().invoke("odyssey.ledger", {"owed_only": True}).value
    row = v["owed"][0]
    assert "snapshot" in row, (
        f"a round picking from this list gets {sorted(row)} -- no path it could give an "
        f"anatomy tool")
    assert Path(row["snapshot"]).is_dir(), row["snapshot"]


def test_the_census_hands_back_one_too():
    r = _reg()
    slug = r.invoke("odyssey.ledger", {"owed_only": True}).value["owed"][0]["slug"]
    v = r.invoke("lake.census", {"slug": slug}).value
    assert "snapshot" in v, sorted(v)
    assert Path(v["snapshot"]).is_dir(), v["snapshot"]


def test_the_path_is_the_one_the_anatomy_tool_accepts():
    """End to end: the path a discovery tool returns must work as an anatomy argument.

    Anything less and this fix is cosmetic -- the round would still be guessing.
    """
    r = _reg()
    row = next(x for x in r.invoke("odyssey.ledger", {"owed_only": True}).value["owed"]
               if x["class"] == "DENSE")
    res = r.invoke("odyssey.anatomy", {"snapshot": row["snapshot"]})
    assert res.ok, res.error
    # A DENSE body refuses expert anatomy BY MECHANISM, which is the correct answer here.
    # What must not happen is a complaint about the PATH.
    refused = (res.value or {}).get("refused") or ""
    assert "not a directory" not in refused, refused
    assert "expert" in refused.lower() or (res.value or {}).get("anatomy"), refused
