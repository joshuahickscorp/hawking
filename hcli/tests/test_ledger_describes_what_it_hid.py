"""Two rounds spent their whole budget trying to see the ledger the tool had bounded.

The owed_only view shows 12 of 46 and discloses the count, which was the fix for the
opposite defect -- the resident once received an 8 KB summary naming NOT ONE specimen.
Honest disclosure created a new problem: a careful scientist goes looking for what it was
told is hidden, and looking costs everything.

Round 10 (4 tool rounds, 9 observations): "the 45 other owed specimens are not shown in
the 1.3% ledger view", then closed on all-repeat without measuring.

Round 11 (3 tool rounds, 19 observations, closed on observation_budget_16): "Ledger shows
46 owed; 12 shown are 1.1-13.4 GiB DENSE with 5 axes each. No 13.4+ shown, so 33 hidden
are >= 13.4 GiB ... I need the full list to pick the true worst. I will read the 301 KB
ledger in 10 windows (1-30, 31-60, ...)". It spent the budget on the windowed read.

Round 11's inference was plausible and WRONG, which is the sharper reason to fix this.
The sort is most-axes-owed FIRST and only then smallest-GiB, so a body with fewer owed
axes lands in the tail whatever its size. The real hidden range is 1.4 to 1453.8 GiB, not
">= 13.4". It reasoned as if the list were sorted by size alone, and the tool never said
otherwise -- it published a sorted view and kept the sort key to itself.

Two things are missing and both are one field. The tool never says WHICH ordering it used,
so "the worst" is ambiguous -- most axes owed, or largest body -- and a caller that guesses
guesses wrong. And it never characterises the tail it withheld, so the only way to learn
the hidden range is to read 301 KB in ten windows.
"""
from __future__ import annotations

from pathlib import Path

from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]


def _owed():
    return default_tool_registry(REPO, repo_root=REPO).invoke(
        "odyssey.ledger", {"owed_only": True}).value


def test_it_says_which_ordering_it_used():
    v = _owed()
    assert "ordering" in v, (
        "the view is sorted and never says how, so 'the worst' is ambiguous to the caller")
    o = v["ordering"].lower()
    assert "owed" in o and "gib" in o, v["ordering"]


def test_it_characterises_the_tail_it_withheld():
    """Round 11 derived the hidden range by hand and then read 301 KB to confirm it."""
    v = _owed()
    assert v["n_owed"] > v["shown"], "this fixture assumes the view is actually truncated"
    assert "hidden" in v, "the caller cannot reason about the tail without reading the file"
    h = v["hidden"]
    assert h["n"] == v["n_owed"] - v["shown"], h
    assert h["gib_min"] is not None and h["gib_max"] is not None, h
    assert h["gib_max"] >= h["gib_min"]


def test_the_hidden_range_is_TRUE_not_decorative():
    """Check it against the ledger itself, or it is just another number to trust."""
    import json
    v = _owed()
    doc = json.loads((REPO / v["path"]).read_text())
    owed = [r for r in doc["specimens"]
            if any(a["state"] == "OWED" for a in r["axes"].values())]
    owed.sort(key=lambda r: (-sum(a["state"] == "OWED" for a in r["axes"].values()), r["gib"]))
    tail = owed[v["shown"]:]
    assert v["hidden"]["n"] == len(tail)
    assert v["hidden"]["gib_min"] == min(r["gib"] for r in tail)
    assert v["hidden"]["gib_max"] == max(r["gib"] for r in tail)


def test_the_actionable_head_is_still_first_and_bounded():
    """The fix this replaced must not regress: names first, bounded, truncation disclosed."""
    import json
    v = _owed()
    head = json.dumps(v)[:500]
    assert "owed" in head and "slug" in head, head
    assert v["shown"] <= 12
